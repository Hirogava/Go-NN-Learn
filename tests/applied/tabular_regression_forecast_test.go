package applied_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// TestTabularRegressionProduct — интеграционный / продуктовый тест.
// Логи в формате: "на русском лог / on English log".
func TestTabularRegressionProduct(t *testing.T) {
	// --- конфиг теста (можно менять для CI / локально) ---
	seed := int64(12345)
	rand.Seed(seed)

	N := 350                // всего примеров (уменьшено для скорости)
	F := 8                  // число признаков
	trainRatio := 0.8       // train/test split
	noiseStd := 0.1         // уровень шума в синтетике
	hidden := 16            // hidden neurons
	epochs := 30            // число эпох
	lr := 0.01              // learning rate
	logEvery := 5           // печатать лог каждые N эпох

	// Пометим старт времени
	startTime := time.Now()
	defer func() {
		d := time.Since(startTime)
		t.Logf("время выполнения: %s / total runtime: %s", d.String(), d.String())
	}()

	// --- генерируем синтетические данные ---
	t.Logf("генерация данных (seed=%d) / generating data (seed=%d)", seed, seed)
	X := make([][]float64, N)
	Y := make([]float64, N)

	// ground-truth linear weights
	wtrue := make([]float64, F)
	for i := range wtrue {
		wtrue[i] = rand.Float64()*4 - 2
	}

	for i := 0; i < N; i++ {
		row := make([]float64, F)
		for j := 0; j < F; j++ {
			row[j] = rand.NormFloat64()*0.5 + rand.Float64()*2
		}
		nonlin := math.Sin(row[0]) * 3.0
		lin := 0.0
		for j := 0; j < F; j++ {
			lin += wtrue[j] * row[j]
		}
		X[i] = row
		Y[i] = nonlin + lin + rand.NormFloat64()*noiseStd
	}

	// split train/test
	perm := rand.Perm(N)
	nTrain := int(float64(N) * trainRatio)
	trainX := make([][]float64, nTrain)
	trainY := make([]float64, nTrain)
	testX := make([][]float64, N-nTrain)
	testY := make([]float64, N-nTrain)
	for i := 0; i < nTrain; i++ {
		trainX[i] = X[perm[i]]
		trainY[i] = Y[perm[i]]
	}
	for i := nTrain; i < N; i++ {
		testX[i-nTrain] = X[perm[i]]
		testY[i-nTrain] = Y[perm[i]]
	}

	// baseline predict-mean
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

	// печать базовой информации (билингвально)
	fmt.Printf("данных: %d (train=%d test=%d) / samples: %d (train=%d test=%d)\n",
		N, nTrain, N-nTrain, N, nTrain, N-nTrain)
	fmt.Printf("baseline MSE: %.6f / baseline MSE: %.6f\n", mseBaseline, mseBaseline)

	// --- модель и движок ---
	engine := autograd.NewEngine()

	initFn := func(data []float64) {
		for i := range data {
			data[i] = rand.NormFloat64() * 0.1
		}
	}

	// слои (используем конструкторы, как в репо)
	d1 := layers.NewDense(F, hidden, initFn)
	d2 := layers.NewDense(hidden, 1, initFn)

	forward := func(xnode *graph.Node) *graph.Node {
		h := d1.Forward(xnode)
		h = engine.ReLU(h)
		return d2.Forward(h)
	}

	params := append(d1.Params(), d2.Params()...)

	// --- training loop (single-example) ---
	fmt.Printf("начало обучения / training start: epochs=%d lr=%.4f\n", epochs, lr)
	for epoch := 1; epoch <= epochs; epoch++ {

		permE := rand.Perm(len(trainX))
		sumLoss := 0.0

		for _, id := range permE {
			xvec := trainX[id]
			yval := trainY[id]

			// input as vector [F]
			xT := &tensor.Tensor{Data: copySlice(xvec), Shape: []int{F}, Strides: []int{1}}
			xnode := engine.RequireGrad(xT)

			pred := forward(xnode) // prediction tensor (shape [1,1] typically)

			// target must have same dims as pred; in this repo pred for single example -> [1,1]
			targetT := &tensor.Tensor{Data: []float64{yval}, Shape: []int{1, 1}, Strides: []int{1, 1}}

			lossNode := engine.MSELoss(pred, targetT)

			// read scalar loss value (for logging) BEFORE backward
			lossVal := 0.0
			if lossNode != nil && lossNode.Value != nil && len(lossNode.Value.Data) > 0 {
				lossVal = lossNode.Value.Data[0]
			}

			sumLoss += lossVal

			engine.ZeroGrad()
			engine.Backward(lossNode)

			// SGD update
			for _, p := range params {
				if p == nil || p.Value == nil || p.Grad == nil {
					continue
				}
				for i := range p.Value.Data {
					p.Value.Data[i] -= lr * p.Grad.Data[i]
				}
			}
			engine.ZeroGrad()
		}

		avgLoss := sumLoss / float64(len(trainX))

		// bilingual log: Russian / English
		if epoch%logEvery == 0 || epoch == 1 || epoch == epochs {
			// t.Logf also captured by `go test -v`
			t.Logf("эпоха %d: средний loss %.6f / epoch %d: avg loss %.6f", epoch, avgLoss, epoch, avgLoss)
			fmt.Printf("эпоха %d: средний loss %.6f / epoch %d: avg loss %.6f\n", epoch, avgLoss, epoch, avgLoss)
		}
	}

	// --- inference ---
	fmt.Printf("выполняю inference / performing inference\n")
	preds := make([]float64, len(testX))
	for i := range testX {
		xT := &tensor.Tensor{Data: copySlice(testX[i]), Shape: []int{F}, Strides: []int{1}}
		xnode := graph.NewNode(xT, nil, nil)
		out := forward(xnode)
		if out == nil || out.Value == nil || len(out.Value.Data) == 0 {
			t.Fatalf("пустой выход модели / empty model output")
		}
		preds[i] = out.Value.Data[0]
	}

	mseModel := mse(testY, preds)
	r2 := 1.0 - mseModel/mseBaseline

	// final bilingual summary
	fmt.Printf("результат: MSE_model=%.6f baseline=%.6f R^2=%.4f / result: MSE_model=%.6f baseline=%.6f R^2=%.4f\n",
		mseModel, mseBaseline, r2, mseModel, mseBaseline, r2)
	t.Logf("результат: MSE_model=%.6f baseline=%.6f R^2=%.4f / result: MSE_model=%.6f baseline=%.6f R^2=%.4f",
		mseModel, mseBaseline, r2, mseModel, mseBaseline, r2)

	// assertions (product criteria)
	if r2 < 0.80 {
		t.Fatalf("R^2 плохо (ожидалось >=0.80): %.4f / R^2 too low (expected >=0.80): %.4f", r2, r2)
	}
	if mseModel > mseBaseline*0.75 {
		t.Fatalf("MSE improvement недостаточно (ожидалось >=25%%): model=%.6f baseline=%.6f / MSE improvement insufficient (expected >=25%%): model=%.6f baseline=%.6f",
			mseModel, mseBaseline, mseModel, mseBaseline)
	}
}

// ---------- вспомогательные функции ----------

func mse(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s / float64(len(a))
}

func copySlice(src []float64) []float64 {
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}