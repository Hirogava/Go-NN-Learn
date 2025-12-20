package main

import (
	"fmt"
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

func main() {
	rand.Seed(42)

	const (
		inputDim   = 28 * 28
		numClasses = 10
		samples    = 2000
		batchSize  = 64
		epochs     = 15
	)

	// Синтетический MNIST-подобный датасет
	centers := make([][]float64, numClasses)
	for c := 0; c < numClasses; c++ {
		center := make([]float64, inputDim)
		for i := range center {
			center[i] = rand.NormFloat64() * 3
		}
		centers[c] = center
	}

	x := tensor.Zeros(samples, inputDim)
	y := tensor.Zeros(samples, numClasses)

	for i := 0; i < samples; i++ {
		label := rand.Intn(numClasses)
		for j := 0; j < inputDim; j++ {
			x.Data[i*inputDim+j] = centers[label][j] + rand.NormFloat64()*0.8
		}
		y.Data[i*y.Strides[0]+label] = 1.0
	}

	ds := dataloader.NewSimpleDataset(x, y)
	loader := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: batchSize,
		Shuffle:   true,
		Seed:      42,
	})

	// Модель - один Dense
	model := layers.NewDense(inputDim, numClasses, func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * 0.01
		}
	})

	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)

	// Training loop (epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		loader.Reset()
		engine := autograd.NewEngine()
		acc := metrics.NewAccuracy()

		var epochLoss float64
		var seen int

		for loader.HasNext() {
			batch := loader.Next()

			xNode := graph.NewNode(batch.Features, nil, nil)

			// Forward
			logits := model.Forward(xNode)

			// Loss
			loss := engine.SoftmaxCrossEntropy(logits, batch.Targets)

			// Backward + update
			engine.Backward(loss)
			optimizer.Step(model.Params())
			optimizer.ZeroGrad(model.Params())

			// Metrics
			_ = acc.Update(
				logitsToLabels(logits.Value),
				oneHotToLabels(batch.Targets),
			)

			for _, v := range loss.Value.Data {
				epochLoss += v
			}
			seen += batch.Features.Shape[0]
		}

		fmt.Printf(
			"Epoch %02d | Loss %.4f | Accuracy %.4f\n",
			epoch,
			epochLoss/float64(seen),
			acc.Value(),
		)
	}
}

// Helpers (только для example)

func logitsToLabels(t *tensor.Tensor) []float64 {
	rows, cols := t.Shape[0], t.Shape[1]
	out := make([]float64, rows)

	for i := 0; i < rows; i++ {
		base := i * t.Strides[0]
		maxIdx := 0
		maxVal := t.Data[base]
		for j := 1; j < cols; j++ {
			if t.Data[base+j] > maxVal {
				maxVal = t.Data[base+j]
				maxIdx = j
			}
		}
		out[i] = float64(maxIdx)
	}
	return out
}

func oneHotToLabels(t *tensor.Tensor) []float64 {
	rows, cols := t.Shape[0], t.Shape[1]
	out := make([]float64, rows)

	for i := 0; i < rows; i++ {
		base := i * t.Strides[0]
		for j := 0; j < cols; j++ {
			if t.Data[base+j] == 1.0 {
				out[i] = float64(j)
				break
			}
		}
	}
	return out
}
