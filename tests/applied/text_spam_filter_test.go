package applied_test

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ----------------- Utilities -----------------

func seedAll(seed int64) {
	rand.Seed(seed)
	// Если надо — сюда добавить другие PRNG.
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func dsigmoidFromOutput(s float64) float64 {
	// derivative d/dz sigmoid(z) = s*(1-s) if s = sigmoid(z)
	return s * (1.0 - s)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func drelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// SaveWeights saves model params to file (gob)
func SaveWeights(path string, w1 [][]float64, b1 []float64, w2 [][]float64, b2 []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if err := enc.Encode(w1); err != nil {
		return err
	}
	if err := enc.Encode(b1); err != nil {
		return err
	}
	if err := enc.Encode(w2); err != nil {
		return err
	}
	if err := enc.Encode(b2); err != nil {
		return err
	}
	return nil
}

func LoadWeights(path string) ([][]float64, []float64, [][]float64, []float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var w1 [][]float64
	var b1 []float64
	var w2 [][]float64
	var b2 []float64
	if err := dec.Decode(&w1); err != nil {
		return nil, nil, nil, nil, err
	}
	if err := dec.Decode(&b1); err != nil {
		return nil, nil, nil, nil, err
	}
	if err := dec.Decode(&w2); err != nil {
		return nil, nil, nil, nil, err
	}
	if err := dec.Decode(&b2); err != nil {
		return nil, nil, nil, nil, err
	}
	return w1, b1, w2, b2, nil
}

// ----------------- Data generation -----------------

// GenerateSpamLikeData: создаем бинарные/счетные признаки для текста.
// Идея: есть vocabSize признаков; для spam примеров вероятность выставлять
// "spam-feature" гораздо выше, поэтому задача легко линейно/не очень линейно разделима.
func GenerateSpamLikeData(n int, vocabSize int, spamRatio float64, spamFeatureCount int) (X [][]float64, y []int) {
	X = make([][]float64, n)
	y = make([]int, n)

	// Indices that signal spam (first spamFeatureCount)
	spamIdx := make([]int, spamFeatureCount)
	for i := 0; i < spamFeatureCount; i++ {
		spamIdx[i] = i
	}

	for i := 0; i < n; i++ {
		isSpam := rand.Float64() < spamRatio
		row := make([]float64, vocabSize)
		// baseline random noise
		for j := 0; j < vocabSize; j++ {
			// low chance of normal words
			if rand.Float64() < 0.03 {
				row[j] = 1
			} else {
				row[j] = 0
			}
		}
		if isSpam {
			// activate several spam-indicative features
			for k := 0; k < 3; k++ {
				idx := spamIdx[rand.Intn(spamFeatureCount)]
				row[idx] = 1
			}
			// also some other words with moderate chance
			for j := 0; j < vocabSize; j++ {
				if rand.Float64() < 0.02 {
					row[j] = 1
				}
			}
			y[i] = 1
		} else {
			// ham: ensure spam-indices rarely active
			for j := 0; j < spamFeatureCount; j++ {
				if rand.Float64() < 0.005 {
					row[j] = 1
				}
			}
			y[i] = 0
		}
		X[i] = row
	}
	return
}

// ----------------- Model helpers -----------------

// Create network params with small random init
func NewModel(vocabSize, hidden int) ([][]float64, []float64, [][]float64, []float64) {
	// w1: vocabSize x hidden
	w1 := make([][]float64, vocabSize)
	for i := 0; i < vocabSize; i++ {
		w1[i] = make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			// Xavier-ish small init
			w1[i][j] = rand.NormFloat64() * 0.1
		}
	}
	b1 := make([]float64, hidden)
	for j := 0; j < hidden; j++ {
		b1[j] = 0
	}
	// w2: hidden x 1 (store as hidden x 1)
	w2 := make([][]float64, hidden)
	for j := 0; j < hidden; j++ {
		w2[j] = make([]float64, 1)
		w2[j][0] = rand.NormFloat64() * 0.1
	}
	b2 := make([]float64, 1)
	return w1, b1, w2, b2
}

func ForwardSingle(x []float64, w1 [][]float64, b1 []float64, w2 [][]float64, b2 []float64) (float64, []float64) {
	// x: [vocab], w1: vocab x hidden -> z1 = x * w1
	hidden := len(b1)
	z1 := make([]float64, hidden)
	for j := 0; j < hidden; j++ {
		var sum float64
		for i := 0; i < len(x); i++ {
			if x[i] == 0 {
				continue
			}
			sum += x[i] * w1[i][j]
		}
		sum += b1[j]
		z1[j] = relu(sum)
	}
	// output
	var z2 float64
	for j := 0; j < hidden; j++ {
		z2 += z1[j] * w2[j][0]
	}
	z2 += b2[0]
	y := sigmoid(z2)
	return y, z1
}

func ComputeLossBatch(X [][]float64, y []int, w1 [][]float64, b1 []float64, w2 [][]float64, b2 []float64) float64 {
	var loss float64
	N := len(X)
	for i := 0; i < N; i++ {
		pred, _ := ForwardSingle(X[i], w1, b1, w2, b2)
		// BCE: -[y log p + (1-y) log(1-p)]
		p := math.Min(math.Max(pred, 1e-12), 1-1e-12)
		if y[i] == 1 {
			loss += -math.Log(p)
		} else {
			loss += -math.Log(1 - p)
		}
	}
	return loss / float64(N)
}

// Simple SGD training for one epoch, returns avg loss
func TrainEpoch(X [][]float64, y []int, batchSize int, lr float64, w1 [][]float64, b1 []float64, w2 [][]float64, b2 []float64) float64 {
	N := len(X)
	perm := rand.Perm(N)
	totalLoss := 0.0
	for start := 0; start < N; start += batchSize {
		end := start + batchSize
		if end > N {
			end = N
		}
		// zero grads
		// grads have same shapes
		gw1 := make([][]float64, len(w1))
		for i := 0; i < len(w1); i++ {
			gw1[i] = make([]float64, len(w1[0]))
		}
		gb1 := make([]float64, len(b1))
		gw2 := make([][]float64, len(w2))
		for j := 0; j < len(w2); j++ {
			gw2[j] = make([]float64, 1)
		}
		gb2 := make([]float64, 1)

		// accumulate gradients over batch
		batchCount := 0
		for _, idx := range perm[start:end] {
			x := X[idx]
			label := y[idx]
			pred, hiddenActiv := ForwardSingle(x, w1, b1, w2, b2)
			// loss derivative wrt pre-sigmoid z2: dL/dz2 = p - y
			dz2 := pred - float64(label)
			// gradients for w2 and b2
			for j := 0; j < len(w2); j++ {
				gw2[j][0] += dz2 * hiddenActiv[j]
			}
			gb2[0] += dz2
			// backprop to hidden
			for j := 0; j < len(hiddenActiv); j++ {
				// derivative through ReLU
				var drelu_j float64
				// we need pre-activation value to know drelu; but we only stored post-ReLU (hiddenActiv)
				// assume drelu = 1 if hiddenActiv>0 else 0 (works)
				if hiddenActiv[j] > 0 {
					drelu_j = 1
				} else {
					drelu_j = 0
				}
				dhidden := dz2 * w2[j][0] * drelu_j
				// gradients for w1[:, j]
				for i := 0; i < len(x); i++ {
					if x[i] == 0 {
						continue
					}
					gw1[i][j] += dhidden * x[i]
				}
				gb1[j] += dhidden
			}
			batchCount++
		}
		// average grads and update params
		bs := float64(batchCount)
		if bs == 0 {
			continue
		}
		for i := 0; i < len(w1); i++ {
			for j := 0; j < len(w1[0]); j++ {
				w1[i][j] -= lr * (gw1[i][j] / bs)
			}
		}
		for j := 0; j < len(b1); j++ {
			b1[j] -= lr * (gb1[j] / bs)
		}
		for j := 0; j < len(w2); j++ {
			w2[j][0] -= lr * (gw2[j][0] / bs)
		}
		b2[0] -= lr * (gb2[0] / bs)

		// compute loss for this mini-batch (for logging)
		for _, idx := range perm[start:end] {
			l := ComputeLossBatch([][]float64{X[idx]}, []int{y[idx]}, w1, b1, w2, b2)
			totalLoss += l
		}
	}
	// average batch loss (approx)
	numBatches := float64((N + batchSize - 1) / batchSize)
	if numBatches == 0 {
		return 0
	}
	return totalLoss / numBatches
}

// ----------------- Metrics -----------------

func PredictBatch(X [][]float64, w1 [][]float64, b1 []float64, w2 [][]float64, b2 []float64, threshold float64) ([]int, []float64) {
	N := len(X)
	preds := make([]int, N)
	scores := make([]float64, N)
	for i := 0; i < N; i++ {
		p, _ := ForwardSingle(X[i], w1, b1, w2, b2)
		scores[i] = p
		if p >= threshold {
			preds[i] = 1
		} else {
			preds[i] = 0
		}
	}
	return preds, scores
}

func ComputeAccuracy(yTrue []int, yPred []int) float64 {
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}

func PrecisionRecall(yTrue []int, yPred []int) (precision, recall float64) {
	tp := 0
	fp := 0
	fn := 0
	for i := range yTrue {
		if yPred[i] == 1 && yTrue[i] == 1 {
			tp++
		}
		if yPred[i] == 1 && yTrue[i] == 0 {
			fp++
		}
		if yPred[i] == 0 && yTrue[i] == 1 {
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

// ----------------- The Test -----------------

func TestTextSpamFilter(t *testing.T) {
	seedAll(42)

	// Hyperparams / dataset size
	N := 600                 // total samples (200-1000 as in spec)
	vocabSize := 200         // vocabulary size
	spamRatio := 0.4         // 40% spam (adjustable)
	spamFeatureCount := 10   // how many features are "spam indicators"
	hidden := 64             // hidden units
	trainRatio := 0.8
	batchSize := 32
	lr := 0.05
	epochs := 40             // between 20-60 as spec recommends

	// Generate synthetic dataset
	X, y := GenerateSpamLikeData(N, vocabSize, spamRatio, spamFeatureCount)

	// deterministic split
	indices := rand.Perm(N)
	trainN := int(float64(N) * trainRatio)
	trainIdx := indices[:trainN]
	testIdx := indices[trainN:]

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

	// Initialize model
	w1, b1, w2, b2 := NewModel(vocabSize, hidden)

	// compute starting loss
	startLoss := ComputeLossBatch(Xtrain, ytrain, w1, b1, w2, b2)
	t.Logf("start train loss: %.6f", startLoss)

	// training loop with checkpoint-by-best-loss
	var bestLoss = math.MaxFloat64
	tmpdir := os.TempDir()
	checkpointPath := filepath.Join(tmpdir, fmt.Sprintf("spam_best_checkpoint_%d.chk", time.Now().UnixNano()))

	for epoch := 1; epoch <= epochs; epoch++ {
		epochLoss := TrainEpoch(Xtrain, ytrain, batchSize, lr, w1, b1, w2, b2)
		// also compute full train loss at epoch end
		fullTrainLoss := ComputeLossBatch(Xtrain, ytrain, w1, b1, w2, b2)
		t.Logf("epoch %d: epochLoss(approx)=%.6f fullTrainLoss=%.6f", epoch, epochLoss, fullTrainLoss)

		if fullTrainLoss < bestLoss {
			// save checkpoint
			if err := SaveWeights(checkpointPath, w1, b1, w2, b2); err != nil {
				t.Fatalf("failed to save checkpoint: %v", err)
			}
			bestLoss = fullTrainLoss
		}
	}

	// After training: load best checkpoint and evaluate on test
	loadingW1, loadingB1, loadingW2, loadingB2, err := LoadWeights(checkpointPath)
	if err != nil {
		t.Fatalf("failed to load checkpoint: %v", err)
	}
	// compute metrics on test
	preds, _ := PredictBatch(Xtest, loadingW1, loadingB1, loadingW2, loadingB2, 0.5)
	acc := ComputeAccuracy(ytest, preds)
	precision, recall := PrecisionRecall(ytest, preds)

	t.Logf("test accuracy=%.4f precision(spam)=%.4f recall(spam)=%.4f", acc, precision, recall)

	// compute final train loss and compare with startLoss for >=30% drop
	finalTrainLoss := ComputeLossBatch(Xtrain, ytrain, loadingW1, loadingB1, loadingW2, loadingB2)
	t.Logf("startLoss=%.6f finalTrainLoss=%.6f (reduction %.2f%%)", startLoss, finalTrainLoss, (startLoss-finalTrainLoss)/startLoss*100.0)

	// Criteria (as specified):
	minAcc := 0.90
	minPrecision := 0.85
	minReduction := 0.30 // 30%

	if acc < minAcc {
		t.Fatalf("test accuracy %.4f < required %.2f", acc, minAcc)
	}
	if precision < minPrecision {
		t.Fatalf("precision(spam) %.4f < required %.2f", precision, minPrecision)
	}
	reduction := (startLoss - finalTrainLoss) / startLoss
	if reduction < minReduction {
		t.Fatalf("train loss reduction %.2f%% < required %.2f%%", reduction*100.0, minReduction*100.0)
	}

	// cleanup checkpoint file
	_ = os.Remove(checkpointPath)
}