package main

import (
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

func TestMNISTLikeMinimal(t *testing.T) {
	rand.Seed(123)

	const (
		inputDim   = 28 * 28
		numClasses = 10
		samples    = 800
		batchSize  = 64
		epochs     = 8
	)

	x := tensor.Zeros(samples, inputDim)
	y := tensor.Zeros(samples, numClasses)

	for i := 0; i < samples; i++ {
		label := rand.Intn(numClasses)
		for j := 0; j < inputDim; j++ {
			x.Data[i*inputDim+j] = rand.NormFloat64()
		}
		y.Data[i*y.Strides[0]+label] = 1.0
	}

	ds := dataloader.NewSimpleDataset(x, y)
	loader := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: batchSize,
		Shuffle:   true,
		Seed:      42,
	})

	model := layers.NewDense(inputDim, numClasses, func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * 0.01
		}
	})

	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	acc := metrics.NewAccuracy()

	for epoch := 0; epoch < epochs; epoch++ {
		loader.Reset()
		engine := autograd.NewEngine()
		acc.Reset()

		for loader.HasNext() {
			batch := loader.Next()
			logits := model.Forward(graph.NewNode(batch.Features, nil, nil))
			loss := engine.SoftmaxCrossEntropy(logits, batch.Targets)
			engine.Backward(loss)
			optimizer.Step(model.Params())
			optimizer.ZeroGrad(model.Params())

			_ = acc.Update(
				logitsToLabels(logits.Value),
				oneHotToLabels(batch.Targets),
			)
		}
	}

	if acc.Value() < 0.70 {
		t.Fatalf("accuracy too low: %.3f", acc.Value())
	}
}
