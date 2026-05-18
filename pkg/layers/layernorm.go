package layers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// LayerNormVector нормализует один вектор признаков (один пример).
// mean и var считаются по всем элементам вектора; gamma и beta — поэлементно.
func LayerNormVector(v tensor.Vector, gamma, beta []float64, eps float64) []float64 {
	if eps <= 0 {
		eps = 1e-5
	}
	d := float64(len(v))
	if d == 0 {
		return nil
	}

	var sum float64
	for _, el := range v {
		sum += el
	}
	mean := sum / d

	var variance float64
	for _, el := range v {
		diff := el - mean
		variance += diff * diff
	}
	variance /= d

	std := math.Sqrt(variance + eps)
	out := make([]float64, len(v))
	for i, el := range v {
		g := 1.0
		b := 0.0
		if gamma != nil && i < len(gamma) {
			g = gamma[i]
		}
		if beta != nil && i < len(beta) {
			b = beta[i]
		}
		out[i] = g*(el-mean)/std + b
	}
	return out
}

// NewLayerNorm создаёт слой LayerNorm с γ=1 и β=0.
func NewLayerNorm(numFeatures int, engine *autograd.Engine) *LayerNorm {
	gamma := engine.RequireGrad(tensor.Ones(numFeatures))
	beta := engine.RequireGrad(tensor.Zeros(numFeatures))

	return &LayerNorm{
		numFeatures: numFeatures,
		eps:         1e-5,
		gamma:       gamma,
		beta:        beta,
		engine:      engine,
	}
}

// Forward выполняет прямой проход. Вход: [batch_size, num_features].
func (ln *LayerNorm) Forward(x *graph.Node) *graph.Node {
	if len(x.Value.Shape) != 2 {
		panic("LayerNorm expects 2D input [batch_size, num_features]")
	}

	batchSize := x.Value.Shape[0]
	numFeatures := x.Value.Shape[1]
	if numFeatures != ln.numFeatures {
		panic("Input features dimension doesn't match LayerNorm numFeatures")
	}

	mean := make([]float64, batchSize)
	variance := make([]float64, batchSize)
	xhat := tensor.Zeros(x.Value.Shape...)

	d := float64(numFeatures)
	for i := 0; i < batchSize; i++ {
		var sum float64
		base := i * numFeatures
		for j := 0; j < numFeatures; j++ {
			sum += x.Value.Data[base+j]
		}
		mean[i] = sum / d

		var v float64
		for j := 0; j < numFeatures; j++ {
			diff := x.Value.Data[base+j] - mean[i]
			v += diff * diff
		}
		variance[i] = v / d

		std := math.Sqrt(variance[i] + ln.eps)
		for j := 0; j < numFeatures; j++ {
			idx := base + j
			xhat.Data[idx] = (x.Value.Data[idx] - mean[i]) / std
		}
	}

	output := tensor.Zeros(x.Value.Shape...)
	gammaData := ln.gamma.Value.Data
	betaData := ln.beta.Value.Data
	for i := 0; i < batchSize; i++ {
		for j := 0; j < numFeatures; j++ {
			idx := i*numFeatures + j
			output.Data[idx] = gammaData[j]*xhat.Data[idx] + betaData[j]
		}
	}

	op := &layerNormOp{
		x:           x,
		gamma:       ln.gamma,
		beta:        ln.beta,
		mean:        mean,
		variance:    variance,
		xhat:        xhat,
		eps:         ln.eps,
		batchSize:   batchSize,
		numFeatures: numFeatures,
	}

	return graph.NewNode(output, []*graph.Node{x, ln.gamma, ln.beta}, op)
}

func (ln *LayerNorm) Params() []*graph.Node {
	return []*graph.Node{ln.gamma, ln.beta}
}

func (ln *LayerNorm) Train() {}
func (ln *LayerNorm) Eval()  {}

func (ln *LayerNorm) SetEpsilon(eps float64) {
	ln.eps = eps
}

type layerNormOp struct {
	x, gamma, beta *graph.Node
	mean           []float64
	variance       []float64
	xhat           *tensor.Tensor
	eps            float64
	batchSize      int
	numFeatures    int
}

func (op *layerNormOp) Backward(grad *tensor.Tensor) {
	d := float64(op.numFeatures)

	if op.gamma.Grad == nil {
		op.gamma.Grad = tensor.Zeros(op.numFeatures)
	}
	if op.beta.Grad == nil {
		op.beta.Grad = tensor.Zeros(op.numFeatures)
	}
	if op.x.Grad == nil {
		op.x.Grad = tensor.Zeros(op.x.Value.Shape...)
	}

	gammaData := op.gamma.Value.Data
	xData := op.x.Value.Data
	xhatData := op.xhat.Data

	for i := 0; i < op.batchSize; i++ {
		mean := op.mean[i]
		variance := op.variance[i]
		std := math.Sqrt(variance + op.eps)
		invStd := 1.0 / std
		invStd3 := invStd * invStd * invStd

		base := i * op.numFeatures

		var dvar, dmean float64
		dxhat := make([]float64, op.numFeatures)
		for j := 0; j < op.numFeatures; j++ {
			idx := base + j
			g := grad.Data[idx]
			op.gamma.Grad.Data[j] += g * xhatData[idx]
			op.beta.Grad.Data[j] += g
			dxhat[j] = g * gammaData[j]
			dvar += dxhat[j] * (xData[idx] - mean) * (-0.5) * invStd3
		}

		for j := 0; j < op.numFeatures; j++ {
			dmean += dxhat[j] * (-invStd)
		}

		for j := 0; j < op.numFeatures; j++ {
			idx := base + j
			op.x.Grad.Data[idx] += dxhat[j]*invStd + dvar*2.0*(xData[idx]-mean)/d + dmean/d
		}
	}
}
