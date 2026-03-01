package main

import (
	"fmt"
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func main() {
	rand.Seed(42)

	// XOR датасет: 4 точки
	// (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1
	x := tensor.Zeros(4, 2)
	y := tensor.Zeros(4, 1)

	x.Data[0] = 0
	x.Data[1] = 0
	y.Data[0] = 0

	x.Data[2] = 1
	x.Data[3] = 1
	y.Data[1] = 0

	x.Data[4] = 0
	x.Data[5] = 1
	y.Data[2] = 1

	x.Data[6] = 1
	x.Data[7] = 0
	y.Data[3] = 1

	ds := dataloader.NewSimpleDataset(x, y)
	loader := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: 4,
		Shuffle:   true,
		Seed:      42,
	})

	// Модель: Dense(2,8) -> ReLU -> Dense(8,1) -> Sigmoid
	// He-инициализация для ReLU: std = sqrt(2/in)
	he1 := func(w []float64) {
		std := 1.0 // sqrt(2/2) для Dense(2,8)
		for i := range w {
			w[i] = rand.NormFloat64() * std
		}
	}
	he2 := func(w []float64) {
		std := 0.5 // sqrt(2/8) для Dense(8,1)
		for i := range w {
			w[i] = rand.NormFloat64() * std
		}
	}
	dense1 := layers.NewDense(2, 8, he1)
	dense2 := layers.NewDense(8, 1, he2)

	optimizer := optimizers.NewAdam(0.1, 0.9, 0.999, 1e-8)

	const maxSteps = 5000
	acc := metrics.NewAccuracy()

	for step := 0; step < maxSteps; step++ {
		loader.Reset()
		for loader.HasNext() {
			batch := loader.Next()

			ctx := autograd.NewGraph()
			ctx.WithGrad()
			autograd.SetGraph(ctx)

			xNode := graph.NewNode(batch.Features, nil, nil)
			h := dense1.Forward(xNode)
			hRelu := ctx.Engine().ReLU(h)
			z := dense2.Forward(hRelu)
			pred := ctx.Engine().Sigmoid(z)
			loss := ctx.Engine().BinaryCrossEntropy(pred, batch.Targets)

			ctx.Backward(loss)

			params := append(dense1.Params(), dense2.Params()...)
			optimizer.Step(params)
			optimizer.ZeroGrad(params)

			predLabels := sigmoidToLabels(pred.Value)
			targetLabels := tensorToLabels(batch.Targets)
			_ = acc.Update(predLabels, targetLabels)
		}

		if step%500 == 0 || acc.Value() >= 1.0 {
			fmt.Printf("Step %d | Accuracy %.2f\n", step, acc.Value())
		}
		if acc.Value() >= 1.0 {
			fmt.Printf("XOR: converged at step %d with 100%% accuracy\n", step)
			return
		}
		acc.Reset()
	}
	fmt.Printf("XOR: did not converge in %d steps (accuracy %.2f)\n", maxSteps, acc.Value())
}

func sigmoidToLabels(t *tensor.Tensor) []float64 {
	out := make([]float64, len(t.Data))
	for i, v := range t.Data {
		if v >= 0.5 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
	return out
}

func tensorToLabels(t *tensor.Tensor) []float64 {
	out := make([]float64, len(t.Data))
	for i, v := range t.Data {
		out[i] = v
	}
	return out
}
