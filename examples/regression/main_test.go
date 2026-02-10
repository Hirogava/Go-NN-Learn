package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// Проверяем, что модель действительно обучается
func TestMinimalTraining(t *testing.T) {
	rand.Seed(42)

	// Dataset
	numSamples := 500

	xData := tensor.Zeros(numSamples, 1)
	yData := tensor.Zeros(numSamples, 1)

	for i := 0; i < numSamples; i++ {
		x := rand.Float64()*2 - 1
		y := 2 * x

		xData.Data[i] = x
		yData.Data[i] = y
	}

	dataset := dataloader.NewSimpleDataset(xData, yData)
	loader := dataloader.NewDataLoader(dataset, dataloader.DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   true,
		Seed:      42,
	})

	// Model
	W := graph.NewNode(tensor.Randn([]int{1, 1}, 1), nil, nil)
	params := []*graph.Node{W}

	optimizer := optimizers.NewAdam(0.1, 0.9, 0.999, 1e-8)

	// Train
	epochs := 15

	for epoch := 0; epoch < epochs; epoch++ {
		loader.Reset()

		for loader.HasNext() {
			batch := loader.Next()

			engine := autograd.NewEngine()
			xNode := graph.NewNode(batch.Features, nil, nil)

			yPred := engine.MatMul(xNode, W)
			loss := engine.MSELoss(yPred, batch.Targets)

			engine.Backward(loss)
			optimizer.Step(params)
			optimizer.ZeroGrad(params)
		}
	}

	// Assert
	w := W.Value.Data[0]

	if math.Abs(w-2.0) > 0.1 {
		t.Fatalf("W not converged: got %.4f, want ~2.0", w)
	}
}
