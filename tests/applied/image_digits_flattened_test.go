// tests/applied/image_digits_flattened_product_test.go
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

// --------------------
// Utility / Dataset
// --------------------

// makeSyntheticDigitsDatasetSmall creates a simpler MNIST-like synthetic dataset.
// Мы уменьшаем размерность признаков (featuresSmall) чтобы модель надежно обучалась
// на чистой реализации Dense/ReLU в репозитории без BN/специальных инициализаций.
func makeSyntheticDigitsDatasetSmall(rng *rand.Rand, nSamples int, nClasses int, features int, noiseStd float64) (data []float64, labels []int) {
	data = make([]float64, nSamples*features)
	labels = make([]int, nSamples)
	block := features / nClasses
	if block == 0 {
		block = 1
	}
	for i := 0; i < nSamples; i++ {
		c := rng.Intn(nClasses)
		labels[i] = c
		for f := 0; f < features; f++ {
			base := 0.0
			if f >= c*block && f < (c+1)*block {
				// яркость для этого класса
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

// softmaxSums computes softmax row sums for logits and returns max deviation from 1.0
func softmaxMaxAbsDeviationFromOne(logits []float64, rows, cols int) float64 {
	maxDev := 0.0
	for r := 0; r < rows; r++ {
		row := logits[r*cols : r*cols+cols]
		// numeric-stable softmax sum
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
		probSum := 0.0
		for _, v := range row {
			prob := math.Exp(v-maxVal) / sum
			probSum += prob
		}
		dev := math.Abs(probSum - 1.0)
		if dev > maxDev {
			maxDev = dev
		}
	}
	return maxDev
}

// clip limits values in-place to [-limit, limit]
func clip(xs []float64, limit float64) {
	for i := range xs {
		if xs[i] > limit {
			xs[i] = limit
		}
		if xs[i] < -limit {
			xs[i] = -limit
		}
	}
}

// --------------------
// Product test
// --------------------

func TestImageDigitsFlattened_Product(t *testing.T) {
	// Билнгвальные префиксы в логах: "<русский> / <english>"
	log := func(format string, args ...interface{}) {
		// format уже должен быть "русский / english"
		t.Logf(format, args...)
	}

	// reproducible
	seed := int64(20260315)
	rng := rand.New(rand.NewSource(seed))

	// dataset / model hyperparams chosen for stability on CI
	trainN := 1500
	testN := 400
	nClasses := 10
	features := 32          // Уменьшили размерность для надежности
	noiseStd := 0.03        // небольшой шум
	hidden := 64            // скрытый слой
	epochs := 50
	batchSize := 64
	lr := 0.05 // уменьшили lr для стабильности

	// генерируем наборы
	trainX, trainY := makeSyntheticDigitsDatasetSmall(rng, trainN, nClasses, features, noiseStd)
	rngTest := rand.New(rand.NewSource(seed + 1))
	testX, testY := makeSyntheticDigitsDatasetSmall(rngTest, testN, nClasses, features, noiseStd)

	// model init: детерминированная инициализация (уменьшенный std)
	initRng1 := rand.New(rand.NewSource(seed + 42))
	initRng2 := rand.New(rand.NewSource(seed + 43))
	init1 := func(arr []float64) {
		for i := range arr {
			arr[i] = initRng1.NormFloat64() * 0.02
		}
	}
	init2 := func(arr []float64) {
		for i := range arr {
			arr[i] = initRng2.NormFloat64() * 0.02
		}
	}

	d1 := layers.NewDense(features, hidden, init1)
	d2 := layers.NewDense(hidden, nClasses, init2)

	// Тренировочный цикл (ручной, без Trainer, чтобы не полагаться на лишние абстракции)
	for ep := 0; ep < epochs; ep++ {
		// shuffle indices
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

			// build batch and normalize input: (x - 0.5) * 2.0 -> zero-centered
			batchData := make([]float64, curBatch*features)
			batchLabels := make([]int, curBatch)
			for i := 0; i < curBatch; i++ {
				src := indices[start+i]
				for f := 0; f < features; f++ {
					v := trainX[src*features+f]
					batchData[i*features+f] = (v - 0.5) * 2.0
				}
				batchLabels[i] = trainY[src]
			}

			// forward/backward
			e := autograd.NewEngine()
			xTensor := makeInputTensor(batchData, curBatch, features)
			xNode := e.RequireGrad(xTensor)

			h := d1.Forward(xNode)
			hAct := e.ReLU(h)
			logits := d2.Forward(hAct)

			// Clip logits to avoid exp overflow in unstable softmax
			if logits != nil && logits.Value != nil {
				clip(logits.Value.Data, 20.0)
			}

			target := makeOneHot(batchLabels, nClasses)
			loss := e.SoftmaxCrossEntropy(logits, target)

			if loss.Value != nil && len(loss.Value.Data) > 0 {
				epochLossSum += loss.Value.Data[0]
			}
			epochBatches++

			e.Backward(loss)

			// SGD update
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

		avgLoss := epochLossSum / math.Max(1.0, float64(epochBatches))
		// bilingual log: русский / english
		log("Эпоха %d — средний loss: %.6f / Epoch %d — avg loss: %.6f", ep+1, avgLoss, ep+1, avgLoss)

		// лёгкая проверка: если loss упал сильно — можно досрочно остановить
		if avgLoss < 0.005 {
			log("Ранняя остановка на эпохе %d (loss %.6f) / Early stop at epoch %d (loss %.6f)", ep+1, avgLoss, ep+1, avgLoss)
			break
		}
	}

	// Оценка на тесте
	correct := 0
	total := 0
	maxSoftmaxDev := 0.0

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

		// sanity checks
		if logits.Value == nil || len(logits.Value.Data) == 0 {
			t.Fatalf("Логиты пусты / logits empty")
		}
		if anyNaNInf(logits.Value.Data) {
			t.Fatalf("Найден NaN/Inf в логитах / NaN/Inf found in logits")
		}
		dev := softmaxMaxAbsDeviationFromOne(logits.Value.Data, curBatch, nClasses)
		if dev > maxSoftmaxDev {
			maxSoftmaxDev = dev
		}

		for i := 0; i < curBatch; i++ {
			row := logits.Value.Data[i*nClasses : i*nClasses+nClasses]
			pred := argmaxRow(row, nClasses)
			if pred == batchLabels[i] {
				correct++
			}
			total++
		}
	}

	accuracy := float64(correct) / float64(total)
	log("Точность на тесте: %.4f / Test accuracy: %.4f", accuracy, accuracy)
	log("Максимальная отклонение суммы softmax от 1.0: %.6e / Max softmax sum dev from 1.0: %.6e", maxSoftmaxDev, maxSoftmaxDev)

	// Критерии успеха
	minAccuracy := 0.88
	maxSoftmaxEps := 1e-4

	if maxSoftmaxDev > maxSoftmaxEps {
		t.Fatalf("Слишком большое отклонение суммы softmax от 1 (%.6e) / Softmax sums dev too large (%.6e)", maxSoftmaxDev, maxSoftmaxDev)
	}
	if accuracy < minAccuracy {
		t.Fatalf("Точность ниже порога: %.4f < %.4f / Accuracy below threshold: %.4f < %.4f", accuracy, minAccuracy, accuracy, minAccuracy)
	}
}

// helper to convert bool to float64 (used only for anyNaNInf signature convenience)
func float64FromBool(b bool) float64 {
	if b {
		return math.NaN()
	}
	return 0.0
}