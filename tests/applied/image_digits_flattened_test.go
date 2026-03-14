package applied

// FAIL

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// Synthetic MNIST-like dataset generator: creates k centroids in 784-dim space,
// each centroid has a dedicated (non-overlapping) block of active features.
func makeSyntheticDigitsDataset(rng *rand.Rand, nSamples int, nClasses int, noiseStd float64) (data []float64, labels []int) {
	features := 784
	data = make([]float64, nSamples*features)
	labels = make([]int, nSamples)
	block := features / nClasses
	for i := 0; i < nSamples; i++ {
		c := rng.Intn(nClasses)
		labels[i] = c
		for f := 0; f < features; f++ {
			base := 0.0
			if f >= c*block && f < (c+1)*block {
				base = 1.0
			}
			noise := rng.NormFloat64() * noiseStd
			data[i*features+f] = base + noise
		}
	}
	return
}

func makeOneHot(labels []int, nClasses int) *tensor.Tensor {
	n := len(labels)
	out := make([]float64, n*nClasses)
	for i := 0; i < n; i++ {
		out[i*nClasses+labels[i]] = 1.0
	}
	return &tensor.Tensor{
		Data:    out,
		Shape:   []int{n, nClasses},
		Strides: []int{nClasses, 1},
	}
}

func makeInputTensor(batchData []float64, batchSize int, features int) *tensor.Tensor {
	return &tensor.Tensor{
		Data:    batchData,
		Shape:   []int{batchSize, features},
		Strides: []int{features, 1},
	}
}

func argmaxRow(logits []float64, cols int) int {
	bestIdx := 0
	bestVal := logits[0]
	for j := 1; j < cols; j++ {
		if logits[j] > bestVal {
			bestVal = logits[j]
			bestIdx = j
		}
	}
	return bestIdx
}

func anyNaNInf(xs []float64) bool {
	for _, v := range xs {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

func TestImageDigitsFlattened_Synthetic(t *testing.T) {
	// reproducible
	seed := int64(123456)
	rng := rand.New(rand.NewSource(seed))

	// dataset sizes
	trainN := 1200
	testN := 300
	nClasses := 10
	features := 784

	// generate dataset — уменьшенный шум, чтобы задача была обучаемой
	trainX, trainY := makeSyntheticDigitsDataset(rng, trainN, nClasses, 0.03)
	rngTest := rand.New(rand.NewSource(seed + 1))
	testX, testY := makeSyntheticDigitsDataset(rngTest, testN, nClasses, 0.03)

	// модель: Dense(784->128) -> ReLU -> Dense(128->10)
	hidden := 128
	initRng1 := rand.New(rand.NewSource(seed + 42))
	initRng2 := rand.New(rand.NewSource(seed + 43))
	// увеличенная инициализация (больше разброса)
	init1 := func(arr []float64) {
		for i := range arr {
			arr[i] = initRng1.NormFloat64() * 0.2
		}
	}
	init2 := func(arr []float64) {
		for i := range arr {
			arr[i] = initRng2.NormFloat64() * 0.2
		}
	}
	d1 := layers.NewDense(features, hidden, init1)
	d2 := layers.NewDense(hidden, nClasses, init2)

	// hyperparams — увеличенные для ускоренного обучения
	epochs := 60
	batchSize := 64
	lr := 1.0

	// training loop
	for ep := 0; ep < epochs; ep++ {
		// shuffle train indices deterministically
		indices := make([]int, trainN)
		for i := 0; i < trainN; i++ {
			indices[i] = i
		}
		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		epochLossSum := 0.0
		epochBatches := 0

		for start := 0; start < trainN; start += batchSize {
			end := start + batchSize
			if end > trainN {
				end = trainN
			}
			curBatch := end - start
			batchData := make([]float64, curBatch*features)
			batchLabels := make([]int, curBatch)
			for i := 0; i < curBatch; i++ {
				src := indices[start+i]
				// normalize input: (x - 0.5) * 2.0  -> zero-centered in [-1,1]
				for f := 0; f < features; f++ {
					v := trainX[src*features+f]
					batchData[i*features+f] = (v - 0.5) * 2.0
				}
				batchLabels[i] = trainY[src]
			}

			// autograd engine
			e := autograd.NewEngine()
			xTensor := makeInputTensor(batchData, curBatch, features)
			xNode := e.RequireGrad(xTensor)

			// forward: d1 -> ReLU -> d2
			h := d1.Forward(xNode)
			hAct := e.ReLU(h)
			logits := d2.Forward(hAct)

			target := makeOneHot(batchLabels, nClasses)
			loss := e.SoftmaxCrossEntropy(logits, target)

			// accumulate loss value (scalar)
			if loss.Value != nil && len(loss.Value.Data) > 0 {
				epochLossSum += loss.Value.Data[0]
			}
			epochBatches++

			// backward
			e.Backward(loss)

			// collect params from both layers
			params := append(d1.Params(), d2.Params()...)
			for _, p := range params {
				if p.Grad == nil {
					continue
				}
				for i := range p.Value.Data {
					p.Value.Data[i] -= lr * p.Grad.Data[i]
				}
				p.Grad = nil
			}
		}

		// log epoch avg loss (use t.Logf so it appears on -v)
		avg := epochLossSum / math.Max(1.0, float64(epochBatches))
		t.Logf("epoch %d avg loss: %.6f", ep+1, avg)
		// optional early exit if loss is already very low
		if avg < 0.01 {
			t.Logf("early stop at epoch %d (loss %.6f)", ep+1, avg)
			break
		}
	}

	// evaluation on test set
	correct := 0
	for start := 0; start < testN; start += batchSize {
		end := start + batchSize
		if end > testN {
			end = testN
		}
		curBatch := end - start
		batchData := make([]float64, curBatch*features)
		batchLabels := make([]int, curBatch)
		for i := 0; i < curBatch; i++ {
			src := start + i
			for f := 0; f < features; f++ {
				v := testX[src*features+f]
				batchData[i*features+f] = (v - 0.5) * 2.0
			}
			batchLabels[i] = testY[src]
		}

		e := autograd.NewEngine()
		xTensor := makeInputTensor(batchData, curBatch, features)
		xNode := e.RequireGrad(xTensor)

		h := d1.Forward(xNode)
		hAct := e.ReLU(h)
		logits := d2.Forward(hAct)

		// basic sanity: no NaN/Inf in logits
		if anyNaNInf(logits.Value.Data) {
			t.Fatalf("NaN/Inf found in logits")
		}

		for i := 0; i < curBatch; i++ {
			row := logits.Value.Data[i*nClasses : i*nClasses+nClasses]
			pred := argmaxRow(row, nClasses)
			if pred == batchLabels[i] {
				correct++
			}
		}
	}

	accuracy := float64(correct) / float64(testN)
	t.Logf("Test accuracy: %.4f", accuracy)

	if accuracy < 0.88 {
		t.Fatalf("accuracy too low: got %.4f, want >= 0.88", accuracy)
	}
}