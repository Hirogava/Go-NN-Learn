package layers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/matrix"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// GRU — Gated Recurrent Unit (упрощённая LSTM).
// Вход: [batch, seq, input_size], выход: [batch, seq, hidden_size].
//
// Порядок ворот как в PyTorch: reset, update, new.
// h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
// n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))
type GRU struct {
	inputSize  int
	hiddenSize int
	training   bool

	// W_ih [3*hidden, input], W_hh [3*hidden, hidden]
	weights []*graph.Node
	biases  []*graph.Node

	hiddenState *tensor.Tensor
}

// NewGRU создаёт слой GRU с одним рекуррентным уровнем.
func NewGRU(inputSize, hiddenSize int, initFunc func([]float64)) *GRU {
	threeH := 3 * hiddenSize

	wihData := make([]float64, threeH*inputSize)
	initFunc(wihData)
	whhData := make([]float64, threeH*hiddenSize)
	initFunc(whhData)
	bihData := make([]float64, threeH)
	initFunc(bihData)
	bhhData := make([]float64, threeH)
	initFunc(bhhData)

	return &GRU{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		training:   true,
		weights: []*graph.Node{
			{Value: &tensor.Tensor{
				Data: wihData, Shape: []int{threeH, inputSize}, Strides: []int{inputSize, 1},
			}},
			{Value: &tensor.Tensor{
				Data: whhData, Shape: []int{threeH, hiddenSize}, Strides: []int{hiddenSize, 1},
			}},
		},
		biases: []*graph.Node{
			{Value: &tensor.Tensor{Data: bihData, Shape: []int{threeH}, Strides: []int{1}}},
			{Value: &tensor.Tensor{Data: bhhData, Shape: []int{threeH}, Strides: []int{1}}},
		},
	}
}

func (g *GRU) Params() []*graph.Node {
	params := make([]*graph.Node, 0, len(g.weights)+len(g.biases))
	params = append(params, g.weights...)
	params = append(params, g.biases...)
	return params
}

func (g *GRU) ResetHiddenState() { g.hiddenState = nil }

func (g *GRU) GetHiddenState() *tensor.Tensor { return g.hiddenState }

func (g *GRU) SetHiddenState(h *tensor.Tensor) { g.hiddenState = h }

func (g *GRU) GetInputSize() int  { return g.inputSize }

func (g *GRU) GetHiddenSize() int { return g.hiddenSize }

func (g *GRU) Train() { g.training = true }

func (g *GRU) Eval() { g.training = false }

type gruStepCache struct {
	r, z, n, nHH *tensor.Tensor
}

type gruOp struct {
	x     *graph.Node
	gru   *GRU
	hStates []*tensor.Tensor
	steps   []*gruStepCache
}

func (g *GRU) Forward(x *graph.Node) *graph.Node {
	batchSize := x.Value.Shape[0]
	seqLen := x.Value.Shape[1]
	h := g.hiddenSize

	var hStates []*tensor.Tensor
	var steps []*gruStepCache
	var hPrev *tensor.Tensor

	if g.training {
		hStates = make([]*tensor.Tensor, seqLen+1)
		steps = make([]*gruStepCache, seqLen)
		if g.hiddenState == nil {
			hStates[0] = tensor.Zeros(batchSize, h)
		} else {
			hStates[0] = g.hiddenState
		}
		hPrev = hStates[0]
	} else {
		if g.hiddenState == nil {
			hPrev = tensor.Zeros(batchSize, h)
		} else {
			hPrev = g.hiddenState
		}
	}

	wihT, _ := matrix.Transposition(matrix.TensorToMatrix(g.weights[0].Value))
	whhT, _ := matrix.Transposition(matrix.TensorToMatrix(g.weights[1].Value))
	bih := g.biases[0].Value
	bhh := g.biases[1].Value

	outputVal := tensor.Zeros(batchSize, seqLen, h)

	for t := 0; t < seqLen; t++ {
		xt := extractSlice(x.Value, t)
		xtM := matrix.TensorToMatrix(xt)

		gatesIHM, _ := matrix.MatMul(xtM, wihT)
		gatesIH, _ := tensor.Add(matrix.MatrixToTensor(gatesIHM), broadcastBias(bih, batchSize))

		hPrevM := matrix.TensorToMatrix(hPrev)
		gatesHHM, _ := matrix.MatMul(hPrevM, whhT)
		gatesHH, _ := tensor.Add(matrix.MatrixToTensor(gatesHHM), broadcastBias(bhh, batchSize))

		rIH, zIH, nIH := splitGates(gatesIH, h)
		rHH, zHH, nHH := splitGates(gatesHH, h)

		rPre, _ := tensor.Add(rIH, rHH)
		zPre, _ := tensor.Add(zIH, zHH)
		rGate := tensor.Apply(rPre, sigmoid)
		zGate := tensor.Apply(zPre, sigmoid)

		rNHH, _ := tensor.Mul(rGate, nHH)
		nLin, _ := tensor.Add(nIH, rNHH)
		nGate := tensor.Apply(nLin, math.Tanh)

		oneMinusZ := tensor.Apply(zGate, func(v float64) float64 { return 1 - v })
		term1, _ := tensor.Mul(oneMinusZ, nGate)
		term2, _ := tensor.Mul(zGate, hPrev)
		hT, _ := tensor.Add(term1, term2)

		if g.training {
			hStates[t+1] = hT
			steps[t] = &gruStepCache{r: rGate, z: zGate, n: nGate, nHH: nHH}
		}
		hPrev = hT
		copySlice(outputVal, hT, t)
	}

	g.hiddenState = hPrev
	if !g.training {
		return graph.NewNode(outputVal, nil, nil)
	}

	op := &gruOp{x: x, gru: g, hStates: hStates, steps: steps}
	parents := append([]*graph.Node{x}, g.weights...)
	parents = append(parents, g.biases...)
	return graph.NewNode(outputVal, parents, op)
}

func (op *gruOp) Backward(gradOutput *tensor.Tensor) {
	g := op.gru
	seqLen := op.x.Value.Shape[1]
	batchSize := op.x.Value.Shape[0]
	h := g.hiddenSize

	dWih := tensor.Zeros(g.weights[0].Value.Shape...)
	dWhh := tensor.Zeros(g.weights[1].Value.Shape...)
	dbih := tensor.Zeros(g.biases[0].Value.Shape...)
	dbhh := tensor.Zeros(g.biases[1].Value.Shape...)
	dx := tensor.Zeros(op.x.Value.Shape...)

	dhNext := tensor.Zeros(batchSize, h)

	wihM := matrix.TensorToMatrix(g.weights[0].Value)
	whhM := matrix.TensorToMatrix(g.weights[1].Value)

	for t := seqLen - 1; t >= 0; t-- {
		stepGrad := extractSlice(gradOutput, t)
		dh, _ := tensor.Add(stepGrad, dhNext)

		cache := op.steps[t]
		hPrev := op.hStates[t]

		oneMinusZ := tensor.Apply(cache.z, func(v float64) float64 { return 1 - v })
		dn, _ := tensor.Mul(dh, oneMinusZ)

		nMinusH, _ := tensor.Sub(hPrev, cache.n)
		dz, _ := tensor.Mul(dh, nMinusH)

		dhPrevDirect, _ := tensor.Mul(dh, cache.z)

		dnLin := tensor.Apply(cache.n, func(v float64) float64 { return 1 - v*v })
		dnLin, _ = tensor.Mul(dnLin, dn)

		xt := extractSlice(op.x.Value, t)

		drFromN, _ := tensor.Mul(dnLin, cache.nHH)
		drPre := tensor.Apply(cache.r, func(v float64) float64 { return v * (1 - v) })
		drPre, _ = tensor.Mul(drPre, drFromN)

		dzPre := tensor.Apply(cache.z, func(v float64) float64 { return v * (1 - v) })
		dzPre, _ = tensor.Mul(dzPre, dz)

		dnHH, _ := tensor.Mul(dnLin, cache.r)

		dGatesIH := stackGates(drPre, dzPre, dnLin, batchSize, h)
		dGatesHH := stackGates(drPre, dzPre, dnHH, batchSize, h)

		dGatesIHM := matrix.TensorToMatrix(dGatesIH)
		dGatesIHT, _ := matrix.Transposition(dGatesIHM)
		xtM := matrix.TensorToMatrix(xt)
		localDWih, _ := matrix.MatMul(dGatesIHT, xtM)
		dWih, _ = tensor.Add(dWih, matrix.MatrixToTensor(localDWih))

		dGatesHHM := matrix.TensorToMatrix(dGatesHH)
		dGatesHHT, _ := matrix.Transposition(dGatesHHM)
		hPrevM := matrix.TensorToMatrix(hPrev)
		localDWhh, _ := matrix.MatMul(dGatesHHT, hPrevM)
		dWhh, _ = tensor.Add(dWhh, matrix.MatrixToTensor(localDWhh))

		dbih, _ = tensor.Add(dbih, sumAlongBatch(dGatesIH))
		dbhh, _ = tensor.Add(dbhh, sumAlongBatch(dGatesHH))

		dGatesIHM2 := matrix.TensorToMatrix(dGatesIH)
		dxtM, _ := matrix.MatMul(dGatesIHM2, wihM)
		copySlice(dx, matrix.MatrixToTensor(dxtM), t)

		dhFromGatesM, _ := matrix.MatMul(dGatesHHM, whhM)
		dhFromGates := matrix.MatrixToTensor(dhFromGatesM)
		dhNext, _ = tensor.Add(dhFromGates, dhPrevDirect)
	}

	accumulate(g.weights[0], dWih)
	accumulate(g.weights[1], dWhh)
	accumulate(g.biases[0], dbih)
	accumulate(g.biases[1], dbhh)
	accumulate(op.x, dx)
}

func sigmoid(v float64) float64 { return 1 / (1 + math.Exp(-v)) }

func splitGates(gates *tensor.Tensor, h int) (r, z, n *tensor.Tensor) {
	b := gates.Shape[0]
	r = gateSlice(gates, b, h, 0)
	z = gateSlice(gates, b, h, h)
	n = gateSlice(gates, b, h, 2*h)
	return r, z, n
}

func gateSlice(gates *tensor.Tensor, batch, h, offset int) *tensor.Tensor {
	data := make([]float64, batch*h)
	for i := 0; i < batch; i++ {
		src := i*3*h + offset
		copy(data[i*h:(i+1)*h], gates.Data[src:src+h])
	}
	return &tensor.Tensor{Data: data, Shape: []int{batch, h}, Strides: []int{h, 1}}
}

func stackGates(r, z, n *tensor.Tensor, batch, h int) *tensor.Tensor {
	data := make([]float64, batch*3*h)
	for i := 0; i < batch; i++ {
		base := i * 3 * h
		copy(data[base:base+h], r.Data[i*h:(i+1)*h])
		copy(data[base+h:base+2*h], z.Data[i*h:(i+1)*h])
		copy(data[base+2*h:base+3*h], n.Data[i*h:(i+1)*h])
	}
	return &tensor.Tensor{Data: data, Shape: []int{batch, 3 * h}, Strides: []int{3 * h, 1}}
}
