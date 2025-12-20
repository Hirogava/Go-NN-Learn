package main

import (
	"fmt"
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// Минимальный end-to-end пример обучения
// y = x * W
func main() {
	rand.Seed(42)

	// Dataset
	numSamples := 1000

	xData := tensor.Zeros(numSamples, 1)
	yData := tensor.Zeros(numSamples, 1)

	for i := 0; i < numSamples; i++ {
		x := rand.Float64()*2 - 1
		y := 2 * x // БЕЗ bias

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

	// Training loop
	epochs := 20

	for epoch := 0; epoch < epochs; epoch++ {
		loader.Reset()
		epochLoss := 0.0
		batches := 0

		for loader.HasNext() {
			batch := loader.Next()

			engine := autograd.NewEngine()

			xNode := graph.NewNode(batch.Features, nil, nil)

			// forward
			yPred := engine.MatMul(xNode, W)

			// loss
			lossNode := engine.MSELoss(yPred, batch.Targets)
			loss := lossNode.Value.Data[0]

			// backward
			engine.Backward(lossNode)

			// update
			optimizer.Step(params)
			optimizer.ZeroGrad(params)

			epochLoss += loss
			batches++
		}

		fmt.Printf(
			"Epoch %02d | Loss: %.6f\n",
			epoch,
			epochLoss/float64(batches),
		)
	}

	fmt.Println("\nTrained parameter:")
	fmt.Printf("W ≈ %.4f (true = 2.0)\n", W.Value.Data[0])
}
