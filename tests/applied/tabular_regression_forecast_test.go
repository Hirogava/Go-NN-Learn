package applied_test

// FAIL

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestTabularRegressionForecast(t *testing.T) {

	rand.Seed(42)

	N := 600
	F := 8

	trainRatio := 0.8
	noiseStd := 0.1

	X := make([][]float64, N)
	Y := make([]float64, N)

	wtrue := make([]float64, F)
	for i := range wtrue {
		wtrue[i] = rand.Float64()*4 - 2
	}

	for i := 0; i < N; i++ {

		row := make([]float64, F)

		for j := 0; j < F; j++ {
			row[j] = rand.NormFloat64()*0.5 + rand.Float64()*2
		}

		nonlin := math.Sin(row[0]) * 3

		lin := 0.0
		for j := 0; j < F; j++ {
			lin += wtrue[j] * row[j]
		}

		X[i] = row
		Y[i] = nonlin + lin + rand.NormFloat64()*noiseStd
	}

	nTrain := int(float64(N) * trainRatio)

	trainX := X[:nTrain]
	trainY := Y[:nTrain]

	testX := X[nTrain:]
	testY := Y[nTrain:]

	mean := 0.0
	for _, v := range trainY {
		mean += v
	}
	mean /= float64(len(trainY))

	baseline := make([]float64, len(testY))
	for i := range baseline {
		baseline[i] = mean
	}

	mseBaseline := mse(testY, baseline)

	engine := autograd.NewEngine()

	initFn := func(data []float64) {
		for i := range data {
			data[i] = rand.NormFloat64() * 0.1
		}
	}

	d1 := layers.NewDense(F, 24, initFn)
	d2 := layers.NewDense(24, 1, initFn)

	forward := func(x *graph.Node) *graph.Node {
		h := d1.Forward(x)
		h = engine.ReLU(h)
		return d2.Forward(h)
	}

	params := append(d1.Params(), d2.Params()...)

	epochs := 60
	lr := 0.01

	for e := 0; e < epochs; e++ {

		perm := rand.Perm(len(trainX))

		for _, id := range perm {

			x := trainX[id]
			y := trainY[id]

			xTensor := &tensor.Tensor{
				Data:    x,
				Shape:   []int{F},
				Strides: []int{1},
			}

			xNode := engine.RequireGrad(xTensor)

			pred := forward(xNode)

			targetTensor := &tensor.Tensor{
				Data:    []float64{y},
				Shape:   []int{1, 1}, // ВАЖНО: та же размерность что у pred
				Strides: []int{1, 1},
			}

			loss := engine.MSELoss(pred, targetTensor)

			engine.ZeroGrad()
			engine.Backward(loss)

			for _, p := range params {

				if p.Grad == nil {
					continue
				}

				for i := range p.Value.Data {
					p.Value.Data[i] -= lr * p.Grad.Data[i]
				}
			}
		}
	}

	preds := make([]float64, len(testX))

	for i := range testX {

		xTensor := &tensor.Tensor{
			Data:    testX[i],
			Shape:   []int{F},
			Strides: []int{1},
		}

		xNode := graph.NewNode(xTensor, nil, nil)

		out := forward(xNode)

		preds[i] = out.Value.Data[0]
	}

	mseModel := mse(testY, preds)

	r2 := 1 - mseModel/mseBaseline

	if r2 < 0.80 {
		t.Fatalf("R^2 too low: got %.4f (model %.4f baseline %.4f)", r2, mseModel, mseBaseline)
	}

	if mseModel > mseBaseline*0.75 {
		t.Fatalf("model MSE not 25%% better: model %.4f baseline %.4f", mseModel, mseBaseline)
	}
}

func mse(a, b []float64) float64 {

	s := 0.0

	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}

	return s / float64(len(a))
}
