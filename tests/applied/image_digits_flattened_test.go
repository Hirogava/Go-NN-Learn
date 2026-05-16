package applied_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

const (
	digitsClasses  = 10
	digitsFeatures = 784
)

func digitsMakeDataset(rng *rand.Rand, n int) ([]float64, []int) {
	data := make([]float64, n*digitsFeatures)
	labels := make([]int, n)

	for i := 0; i < n; i++ {
		cls := rng.Intn(digitsClasses)
		labels[i] = cls

		// Фоновый шум
		for f := 0; f < digitsFeatures; f++ {
			data[i*digitsFeatures+f] = rng.Float64() * 0.02
		}

		// Класс-специфический "яркий" участок
		start := cls * 70
		for k := 0; k < 60; k++ {
			idx := start + k
			if idx >= digitsFeatures {
				break
			}
			data[i*digitsFeatures+idx] = 1.0 + rng.Float64()*0.05
		}
	}

	return data, labels
}

func digitsBatchTensor(data []float64, rows, cols int) *tensor.Tensor {
	return &tensor.Tensor{
		Data:    data,
		Shape:   []int{rows, cols},
		Strides: []int{cols, 1},
	}
}

func digitsOneHot(labels []int) *tensor.Tensor {
	data := make([]float64, len(labels)*digitsClasses)
	for i, cls := range labels {
		data[i*digitsClasses+cls] = 1
	}
	return digitsBatchTensor(data, len(labels), digitsClasses)
}

func digitsArgmaxRow(row []float64) int {
	best := 0
	bestVal := row[0]
	for i := 1; i < len(row); i++ {
		if row[i] > bestVal {
			bestVal = row[i]
			best = i
		}
	}
	return best
}

func digitsAllFinite(xs []float64) bool {
	for _, v := range xs {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

func digitsSoftmaxDeviation(logits []float64, rows, cols int) float64 {
	maxDev := 0.0
	for r := 0; r < rows; r++ {
		row := logits[r*cols : (r+1)*cols]

		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		sum := 0.0
		for _, v := range row {
			sum += math.Exp(v - maxVal)
		}

		rowSum := 0.0
		for _, v := range row {
			rowSum += math.Exp(v-maxVal) / sum
		}

		dev := math.Abs(rowSum - 1)
		if dev > maxDev {
			maxDev = dev
		}
	}
	return maxDev
}

func digitsConfusion(preds, labels []int) [][]int {
	conf := make([][]int, digitsClasses)
	for i := range conf {
		conf[i] = make([]int, digitsClasses)
	}
	for i := range preds {
		conf[labels[i]][preds[i]]++
	}
	return conf
}

func TestImageDigitsFlattened_Product(t *testing.T) {
	seed := int64(20260508)

	trainRng := rand.New(rand.NewSource(seed))
	testRng := rand.New(rand.NewSource(seed + 1))
	initRng := rand.New(rand.NewSource(seed + 2))

	trainX, trainY := digitsMakeDataset(trainRng, 2000)
	testX, testY := digitsMakeDataset(testRng, 500)

	initFn := func(dst []float64) {
		for i := range dst {
			dst[i] = initRng.NormFloat64() * 0.01
		}
	}

	d1 := layers.NewDense(digitsFeatures, 128, initFn)
	d2 := layers.NewDense(128, digitsClasses, initFn)
	params := append(d1.Params(), d2.Params()...)

	opt := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)

	epochs := 25
	batchSize := 64

	for epoch := 0; epoch < epochs; epoch++ {
		perm := trainRng.Perm(len(trainY))
		epochLossSum := 0.0
		batches := 0

		for start := 0; start < len(perm); start += batchSize {
			end := start + batchSize
			if end > len(perm) {
				end = len(perm)
			}
			bs := end - start

			batchX := make([]float64, bs*digitsFeatures)
			batchY := make([]int, bs)

			for i := 0; i < bs; i++ {
				src := perm[start+i]
				batchY[i] = trainY[src]
				copy(
					batchX[i*digitsFeatures:(i+1)*digitsFeatures],
					trainX[src*digitsFeatures:(src+1)*digitsFeatures],
				)
			}

			engine := autograd.NewEngine()
			input := engine.RequireGrad(digitsBatchTensor(batchX, bs, digitsFeatures))

			h := engine.ReLU(d1.Forward(input))
			logits := d2.Forward(h)

			if logits == nil || logits.Value == nil || !digitsAllFinite(logits.Value.Data) {
				t.Fatalf("найдены NaN/Inf в логитах / NaN/Inf found in logits")
			}

			loss := engine.SoftmaxCrossEntropy(logits, digitsOneHot(batchY))
			if loss == nil || loss.Value == nil || len(loss.Value.Data) == 0 || !digitsAllFinite(loss.Value.Data) {
				t.Fatalf("найдены NaN/Inf в loss / NaN/Inf found in loss")
			}

			epochLossSum += loss.Value.Data[0]
			batches++

			engine.Backward(loss)
			opt.Step(params)
			opt.ZeroGrad(params)
		}

		avgLoss := epochLossSum / float64(batches)
		t.Logf("эпоха %d / epoch %d завершена, avg loss = %.6f", epoch+1, epoch+1, avgLoss)
	}

	correct := 0
	total := 0
	maxSoftmaxDev := 0.0
	preds := make([]int, 0, len(testY))
	labels := make([]int, 0, len(testY))

	for start := 0; start < len(testY); start += batchSize {
		end := start + batchSize
		if end > len(testY) {
			end = len(testY)
		}
		bs := end - start

		batchX := make([]float64, bs*digitsFeatures)
		batchY := make([]int, bs)

		for i := 0; i < bs; i++ {
			src := start + i
			batchY[i] = testY[src]
			copy(
				batchX[i*digitsFeatures:(i+1)*digitsFeatures],
				testX[src*digitsFeatures:(src+1)*digitsFeatures],
			)
		}

		engine := autograd.NewEngine()
		input := engine.RequireGrad(digitsBatchTensor(batchX, bs, digitsFeatures))

		h := engine.ReLU(d1.Forward(input))
		logits := d2.Forward(h)

		if logits == nil || logits.Value == nil {
			t.Fatalf("пустые логиты / empty logits")
		}
		if !digitsAllFinite(logits.Value.Data) {
			t.Fatalf("найдены NaN/Inf в логитах на тесте / NaN/Inf found in test logits")
		}

		dev := digitsSoftmaxDeviation(logits.Value.Data, bs, digitsClasses)
		if dev > maxSoftmaxDev {
			maxSoftmaxDev = dev
		}

		for i := 0; i < bs; i++ {
			row := logits.Value.Data[i*digitsClasses : (i+1)*digitsClasses]
			pred := digitsArgmaxRow(row)
			preds = append(preds, pred)
			labels = append(labels, batchY[i])
			if pred == batchY[i] {
				correct++
			}
			total++
		}
	}

	conf := digitsConfusion(preds, labels)
	for cls := 0; cls < digitsClasses; cls++ {
		rowTotal := 0
		for _, v := range conf[cls] {
			rowTotal += v
		}
		t.Logf("class %d: correct=%d total=%d row=%v", cls, conf[cls][cls], rowTotal, conf[cls])
	}

	accuracy := float64(correct) / float64(total)
	t.Logf("test accuracy = %.4f / test accuracy = %.4f", accuracy, accuracy)
	t.Logf("max softmax deviation = %.6e / max softmax deviation = %.6e", maxSoftmaxDev, maxSoftmaxDev)

	if accuracy < 0.88 {
		t.Fatalf("accuracy below threshold: %.4f < 0.88 / accuracy below threshold: %.4f < 0.88", accuracy, accuracy)
	}
	if maxSoftmaxDev > 1e-6 {
		t.Fatalf("softmax sum deviation too large: %.6e / softmax sum deviation too large: %.6e", maxSoftmaxDev, maxSoftmaxDev)
	}
}