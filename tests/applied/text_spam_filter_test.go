package applied_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Product test: spam filter (binary classification)
// Logs are bilingual: "на русском / in English"

// ------------------ Helpers ------------------

func setSeed(seed int64) {
	if seed == 0 {
		seed = 42
	}
	rand.Seed(seed)
}

func oneHotTargets(labels []int, classes int) *tensor.Tensor {
	n := len(labels)
	data := make([]float64, n*classes)
	for i, lab := range labels {
		if lab < 0 || lab >= classes {
			panic("label out of range")
		}
		data[i*classes+lab] = 1.0
	}
	// shape [n, classes]
	return &tensor.Tensor{
		Data:    data,
		Shape:   []int{n, classes},
		Strides: []int{classes, 1},
	}
}

// makeTensorBatch builds a tensor for batch X ([][]float64) with shape [N, D]
func makeTensorBatch(X [][]float64) *tensor.Tensor {
	if len(X) == 0 {
		return &tensor.Tensor{Data: []float64{}, Shape: []int{0, 0}, Strides: []int{0, 1}}
	}
	N := len(X)
	D := len(X[0])
	data := make([]float64, 0, N*D)
	for i := 0; i < N; i++ {
		row := X[i]
		if len(row) != D {
			panic("inconsistent row length")
		}
		data = append(data, row...)
	}
	return &tensor.Tensor{Data: data, Shape: []int{N, D}, Strides: []int{D, 1}}
}

// argmax per-row for logits shape [N, C]
func argmaxRows(t *tensor.Tensor) []int {
	rows := t.Shape[0]
	cols := t.Shape[1]
	out := make([]int, rows)
	for i := 0; i < rows; i++ {
		best := 0
		base := i*cols
		bestVal := t.Data[base]
		for j := 1; j < cols; j++ {
			v := t.Data[base+j]
			if v > bestVal {
				bestVal = v
				best = j
			}
		}
		out[i] = best
	}
	return out
}

func accuracy(ytrue []int, ypred []int) float64 {
	ok := 0
	for i := range ytrue {
		if ytrue[i] == ypred[i] {
			ok++
		}
	}
	return float64(ok) / float64(len(ytrue))
}

func precisionRecall(ytrue []int, ypred []int) (precision, recall float64) {
	tp, fp, fn := 0, 0, 0
	for i := range ytrue {
		if ypred[i] == 1 && ytrue[i] == 1 {
			tp++
		}
		if ypred[i] == 1 && ytrue[i] == 0 {
			fp++
		}
		if ypred[i] == 0 && ytrue[i] == 1 {
			fn++
		}
	}
	if tp+fp == 0 {
		precision = 0
	} else {
		precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn == 0 {
		recall = 0
	} else {
		recall = float64(tp) / float64(tp+fn)
	}
	return
}

// AlmostEqual check
func almostEqual(a, b float64, eps float64) bool {
	return math.Abs(a-b) <= eps
}

// compare params (all param nodes) — elementwise with eps
func compareParams(t *testing.T, aParams, bParams []*graph.Node, eps float64) {
	if len(aParams) != len(bParams) {
		t.Fatalf("параметры: разное количество / params: count mismatch %d vs %d", len(aParams), len(bParams))
	}
	for pi := range aParams {
		a := aParams[pi].Value.Data
		b := bParams[pi].Value.Data
		if len(a) != len(b) {
			t.Fatalf("параметры: форма отличается / params shape mismatch for param %d", pi)
		}
		for i := range a {
			if !almostEqual(a[i], b[i], eps) {
				t.Fatalf("параметры: элемент %d параметра %d отличается более чем eps (a=%g b=%g eps=%g) / params: element %d of param %d differs (a=%g b=%g eps=%g)", i, pi, a[i], b[i], eps, i, pi, a[i], b[i], eps)
			}
		}
	}
}

// ------------------ Small local layers to compose Module ------------------

// simpleSequential implements layers.Module minimally so we can call api.SaveCheckpoint
type simpleSequential struct {
	mods []layers.Layer
}

func (s *simpleSequential) Layers() []layers.Layer { return s.mods }

func (s *simpleSequential) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, m := range s.mods {
		out = m.Forward(out)
	}
	return out
}

func (s *simpleSequential) Params() []*graph.Node {
	var res []*graph.Node
	for _, m := range s.mods {
		res = append(res, m.Params()...)
	}
	return res
}

func (s *simpleSequential) Train() {
	for _, m := range s.mods {
		m.Train()
	}
}
func (s *simpleSequential) Eval() {
	for _, m := range s.mods {
		m.Eval()
	}
}

// ------------------ Data generator (synthetic) ------------------

// GenerateBagOfWordsSpam generates N samples with vocabSize features.
// spamFeatureCount features are more likely in spam examples.
func GenerateBagOfWordsSpam(N, vocabSize int, spamRatio float64, spamFeatureCount int) (X [][]float64, y []int) {
	X = make([][]float64, N)
	y = make([]int, N)
	for i := 0; i < N; i++ {
		row := make([]float64, vocabSize)
		isSpam := rand.Float64() < spamRatio
		if isSpam {
			// activate several spam-indicative features
			for k := 0; k < 3; k++ {
				idx := rand.Intn(spamFeatureCount)
				row[idx] = 1.0
			}
			// random noise
			for j := 0; j < vocabSize; j++ {
				if rand.Float64() < 0.02 {
					row[j] = 1.0
				}
			}
			y[i] = 1
		} else {
			// ham: rare spam features
			for j := 0; j < spamFeatureCount; j++ {
				if rand.Float64() < 0.01 {
					row[j] = 1.0
				}
			}
			y[i] = 0
		}
		X[i] = row
	}
	return
}

// ------------------ The product test ------------------

func TestTextSpamFilter(t *testing.T) {
	setSeed(42)
	// Hyperparams (tweak if needed)
	N := 800
	vocabSize := 300
	spamRatio := 0.4
	spamFeatureCount := 12
	hidden := 64
	epochs := 40
	batchSize := 32
	lr := 0.05

	// Generate data
	X, y := GenerateBagOfWordsSpam(N, vocabSize, spamRatio, spamFeatureCount)

	// Train/test split deterministic
	perm := rand.Perm(N)
	trainN := int(float64(N) * 0.8)
	trainIdx := perm[:trainN]
	testIdx := perm[trainN:]

	Xtrain := make([][]float64, len(trainIdx))
	ytrain := make([]int, len(trainIdx))
	for i, idx := range trainIdx {
		Xtrain[i] = X[idx]
		ytrain[i] = y[idx]
	}
	Xtest := make([][]float64, len(testIdx))
	ytest := make([]int, len(testIdx))
	for i, idx := range testIdx {
		Xtest[i] = X[idx]
		ytest[i] = y[idx]
	}

	// Build model using library layers
	engine := autograd.NewEngine()
	initFn := func(arr []float64) {
		for i := range arr {
			arr[i] = rand.NormFloat64() * 0.05
		}
	}
	d1 := layers.NewDense(vocabSize, hidden, initFn)
	r1 := &reluLayer{}
	d2 := layers.NewDense(hidden, 2, initFn)

	model := &simpleSequential{mods: []layers.Layer{d1, r1, d2}}

	// Training loop: mini-batch SGD using autograd.SoftmaxCrossEntropy + Sum
	bestLoss := math.Inf(1)
	tmpDir := os.TempDir()
	checkpointPath := filepath.Join(tmpDir, fmt.Sprintf("spam_product_best_%d.chk", time.Now().UnixNano()))

	// helper to get params and zero grads
	params := model.Params()
	zeroParams := func() {
		for _, p := range params {
			if p != nil {
				p.ZeroGrad()
			}
		}
	}

	// initial train loss
	{
		// evaluate full train loss before training
		Xt := makeTensorBatch(Xtrain)
		inNode := graph.NewNode(Xt, nil, nil)
		out := model.Forward(inNode) // logits [N,2]
		// build targets tensor
		tensorY := oneHotTargets(ytrain, 2)
		lossNode := engine.SoftmaxCrossEntropy(out, tensorY) // [N,1]
		sumLoss := engine.Sum(lossNode)                      // scalar
		// no backward, just read sum
		startLoss := sumLoss.Value.Data[0] / float64(len(ytrain))
		t.Logf("начальный train loss: %.6f / start train loss: %.6f", startLoss, startLoss)
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		// shuffle train indices per epoch (deterministic given seed)
		permT := rand.Perm(len(Xtrain))
		epochLossSum := 0.0
		batches := 0
		for start := 0; start < len(permT); start += batchSize {
			end := start + batchSize
			if end > len(permT) {
				end = len(permT)
			}
			batchIdx := permT[start:end]
			// prepare batch arrays
			B := len(batchIdx)
			Xb := make([][]float64, B)
			yb := make([]int, B)
			for i, bi := range batchIdx {
				Xb[i] = Xtrain[bi]
				yb[i] = ytrain[bi]
			}
			// forward
			inT := makeTensorBatch(Xb)
			inNode := graph.NewNode(inT, nil, nil)
			logits := model.Forward(inNode) // shape [B,2]
			// build target tensor
			targ := oneHotTargets(yb, 2)
			lossPerSample := engine.SoftmaxCrossEntropy(logits, targ) // [B,1]
			loss := engine.Sum(lossPerSample)                         // scalar [1]
			// zero grads
			zeroParams()
			engine.ZeroGrad()
			// backward
			engine.Backward(loss)
			// update params (SGD)
			for _, p := range params {
				if p.Grad == nil {
					continue
				}
				// p.Value.Data and p.Grad.Data same length
				for i := 0; i < len(p.Value.Data); i++ {
					grad := p.Grad.Data[i] / float64(B) // average over batch
					p.Value.Data[i] -= lr * grad
				}
			}
			// record loss numeric
			epochLossSum += loss.Value.Data[0] / float64(B)
			batches++
		}
		avgEpochLoss := epochLossSum / float64(batches)
		t.Logf("эпоха %d / epoch %d: avg loss = %.6f", epoch, epoch, avgEpochLoss)

		// compute full train loss for checkpoint criteria
		Xt := makeTensorBatch(Xtrain)
		inNode := graph.NewNode(Xt, nil, nil)
		out := model.Forward(inNode)
		tensorY := oneHotTargets(ytrain, 2)
		fullLossPerSample := engine.SoftmaxCrossEntropy(out, tensorY)
		sumLoss := engine.Sum(fullLossPerSample)
		fullTrainLoss := sumLoss.Value.Data[0] / float64(len(ytrain))
		t.Logf("эпоха %d / epoch %d: full train loss = %.6f", epoch, epoch, fullTrainLoss)

		// checkpoint save if improved
		if fullTrainLoss < bestLoss {
			if err := api.SaveCheckpoint(model, checkpointPath); err != nil {
				t.Fatalf("не удалось сохранить чекпоинт: %v / failed to save checkpoint: %v", err, err)
			}
			t.Logf("сохранён лучший чекпоинт (epoch %d) / saved best checkpoint (epoch %d): %s", epoch, epoch, checkpointPath)
			bestLoss = fullTrainLoss
		}
	}

	// Load checkpoint into a fresh model and compare params+predictions
	// build fresh model (same architecture)
	d1b := layers.NewDense(vocabSize, hidden, initFn)
	r1b := &reluLayer{}
	d2b := layers.NewDense(hidden, 2, initFn)
	model2 := &simpleSequential{mods: []layers.Layer{d1b, r1b, d2b}}

	// load
	if err := api.LoadCheckpoint(model2, checkpointPath); err != nil {
		t.Fatalf("не удалось загрузить чекпоинт: %v / failed to load checkpoint: %v", err, err)
	}

	// compare params (within small eps)
	compareParams(t, model.Params(), model2.Params(), 1e-9)
	t.Logf("параметры после загрузки совпадают / parameters equal after load")

	// Compare predictions on fixed test batch (take 100 samples or all)
	nEval := len(Xtest)
	if nEval > 200 {
		nEval = 200
	}
	Xeval := Xtest[:nEval]

	// preds from model
	inT := makeTensorBatch(Xeval)
	inNode := graph.NewNode(inT, nil, nil)
	out1 := model.Forward(inNode)
	preds1 := argmaxRows(out1.Value)

	// preds from model2
	inNode2 := graph.NewNode(inT, nil, nil)
	out2 := model2.Forward(inNode2)
	preds2 := argmaxRows(out2.Value)

	// assert predictions very close (exact equality for argmax expected)
	for i := 0; i < len(preds1); i++ {
		if preds1[i] != preds2[i] {
			t.Fatalf("предсказания модели и загруженной модели различаются на образце %d: %d vs %d / loaded vs original predict mismatch at sample %d: %d vs %d", i, preds1[i], preds2[i], i, preds1[i], preds2[i])
		}
	}
	t.Logf("предсказания совпадают для тестовой подборки / predictions equal on eval slice")

	// final evaluation metrics on test set using loaded model
	inAll := makeTensorBatch(Xtest)
	inNodeAll := graph.NewNode(inAll, nil, nil)
	outLogits := model2.Forward(inNodeAll)
	finalPreds := argmaxRows(outLogits.Value)
	acc := accuracy(ytest, finalPreds)
	prec, rec := precisionRecall(ytest, finalPreds)
	t.Logf("тест: accuracy=%.4f precision(spam)=%.4f recall(spam)=%.4f / test: accuracy=%.4f precision(spam)=%.4f recall(spam)=%.4f", acc, acc, prec, prec, rec, rec)

	// checks: thresholds from spec
	if acc < 0.90 {
		t.Fatalf("тестовая accuracy %.4f < 0.90 / test accuracy %.4f < 0.90", acc, acc)
	}
	if prec < 0.85 {
		t.Fatalf("precision(spam) %.4f < 0.85 / precision(spam) %.4f < 0.85", prec, prec)
	}

	// cleanup
	_ = os.Remove(checkpointPath)
}