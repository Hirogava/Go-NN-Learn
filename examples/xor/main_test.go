package main

import (
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// runXORTraining выполняет один запуск обучения и возвращает (accuracy, steps, ok).
func runXORTraining(seed int64) (float64, int, bool) {
	rand.Seed(seed)

	x := tensor.Zeros(4, 2)
	y := tensor.Zeros(4, 1)
	x.Data[0], x.Data[1] = 0, 0
	y.Data[0] = 0
	x.Data[2], x.Data[3] = 1, 1
	y.Data[1] = 0
	x.Data[4], x.Data[5] = 0, 1
	y.Data[2] = 1
	x.Data[6], x.Data[7] = 1, 0
	y.Data[3] = 1

	ds := dataloader.NewSimpleDataset(x, y)
	loader := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: 4,
		Shuffle:   true,
		Seed:      seed,
	})

	he1 := func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64()
		}
	}
	he2 := func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * 0.5
		}
	}
	dense1 := layers.NewDense(2, 8, he1)
	dense2 := layers.NewDense(8, 1, he2)
	optimizer := optimizers.NewAdam(0.1, 0.9, 0.999, 1e-8)

	const maxSteps = 5000
	for step := 0; step < maxSteps; step++ {
		acc := metrics.NewAccuracy()
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

			_ = acc.Update(sigmoidToLabels(pred.Value), tensorToLabels(batch.Targets))
		}
		if acc.Value() >= 1.0 {
			return acc.Value(), step, true
		}
	}
	return 0, maxSteps, false
}

func TestXORClassification(t *testing.T) {
	// DoD: accuracy = 100%, сходится < 5000 шагов, стабильный результат при 5 перезапусках
	const numRestarts = 5
	for r := 0; r < numRestarts; r++ {
		seed := int64(42 + r*100)
		acc, steps, ok := runXORTraining(seed)
		if !ok {
			t.Errorf("restart %d (seed %d): did not converge in 5000 steps", r+1, seed)
			continue
		}
		if acc < 1.0 {
			t.Errorf("restart %d (seed %d): accuracy %.2f < 100%%", r+1, seed, acc)
		}
		if steps >= 5000 {
			t.Errorf("restart %d (seed %d): converged at step %d >= 5000", r+1, seed, steps)
		}
		t.Logf("restart %d (seed %d): ok, accuracy=100%%, steps=%d", r+1, seed, steps)
	}
}
