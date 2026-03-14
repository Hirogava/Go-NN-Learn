package applied_test

import (
	"math"
	"math/rand"
	"sort"
	"strings"
	"testing"
	// Adjust import path to your module path if needed
	// "github.com/Hirogava/Go-NN-Learn/pkg/layers"
	// "github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	// "github.com/Hirogava/Go-NN-Learn/pkg/train"
)

const (
	seed = 42
)


var classKeywords = [][]string{
	{"invoice", "charge", "billing", "payment", "refund"},
	{"wifi", "crash", "error", "login", "bug", "latency"},
	{"shipment", "delivery", "tracking", "delay", "package"},
	{"hello", "question", "info", "other", "support"},
}

func makePhrase(r *rand.Rand, cls int, vocab []string) string {
	// choose 1-3 keywords of the class and fillrest words
	n := 1 + r.Intn(3)
	parts := make([]string, 0, 8)
	for i := 0; i < n; i++ {
		parts = append(parts, classKeywords[cls][r.Intn(len(classKeywords[cls]))])
	}
	// add 0-3 filler words
	m := r.Intn(4)
	for i := 0; i < m; i++ {
		parts = append(parts, vocab[r.Intn(len(vocab))])
	}
	// shuffle
	for i := range parts {
		j := r.Intn(i + 1)
		parts[i], parts[j] = parts[j], parts[i]
	}
	return strings.Join(parts, " ")
}

// Build a small global filler vocab (common non-keyword words)
func buildFillerVocab() []string {
	return []string{"please", "help", "now", "asap", "account", "order", "status", "need", "thanks", "urgent", "information"}
}

// Build vocabulary from a corpus of phrases, take top K tokens by frequency
func buildVocabFromCorpus(corpus []string, maxVocab int) []string {
	freq := map[string]int{}
	for _, s := range corpus {
		for _, t := range strings.Fields(s) {
			freq[strings.ToLower(t)]++
		}
	}
	types := make([]string, 0, len(freq))
	for k := range freq {
		types = append(types, k)
	}
	sort.Slice(types, func(i, j int) bool { return freq[types[i]] > freq[types[j]] })
	if len(types) > maxVocab {
		types = types[:maxVocab]
	}
	return types
}

// Bag-of-words vectorize a phrase into a float64 slice of length vocabSize
func vectorizeBoW(phrase string, vocab []string) []float64 {
	idx := map[string]int{}
	for i, w := range vocab {
		idx[w] = i
	}
	vec := make([]float64, len(vocab))
	for _, t := range strings.Fields(strings.ToLower(phrase)) {
		if i, ok := idx[t]; ok {
			vec[i]++
		}
	}
	return vec
}

// One-hot encode label
func oneHotLabel(cls int, nClasses int) []float64 {
	o := make([]float64, nClasses)
	o[cls] = 1.0
	return o
}

// Simple softmax + cross-entropy helpers used in the plain trainer below
func softmax(logits []float64) []float64 {
	mx := logits[0]
	for _, v := range logits[1:] {
		if v > mx {
			mx = v
		}
	}
	expSum := 0.0
	exps := make([]float64, len(logits))
	for i, v := range logits {
		exps[i] = math.Exp(v - mx)
		expSum += exps[i]
	}
	for i := range exps {
		exps[i] /= expSum
	}
	return exps
}

func crossEntropyLoss(probs []float64, target []float64) float64 {
	// target is one-hot
	n := len(probs)
	loss := 0.0
	for i := 0; i < n; i++ {
		if target[i] > 0 {
			loss -= math.Log(math.Max(probs[i], 1e-12))
		}
	}
	return loss
}

// Argmax
func argmax(xs []float64) int {
	mi := 0
	mx := xs[0]
	for i, v := range xs[1:] {
		if v > mx {
			mx = v
			mi = i + 1
		}
	}
	return mi
}

// ------------------ Test: Text topic classification (template) ------------------
func TestTextTopicClassification_Template(t *testing.T) {
	// determinism
	r := rand.New(rand.NewSource(seed))
	filler := buildFillerVocab()

	// generate dataset
	nPerClass := 300 // total N = nPerClass * nClasses
	nClasses := len(classKeywords)
	allPhrases := make([]string, 0, nPerClass*nClasses)
	allLabels := make([]int, 0, nPerClass*nClasses)
	for c := 0; c < nClasses; c++ {
		for i := 0; i < nPerClass; i++ {
			p := makePhrase(r, c, filler)
			allPhrases = append(allPhrases, p)
			allLabels = append(allLabels, c)
		}
	}

	// build vocab (keep it modest)
	vocab := buildVocabFromCorpus(allPhrases, 200)
	vocabSize := len(vocab)
	if vocabSize == 0 {
		t.Fatal("vocab is empty")
	}

	// vectorize dataset
	N := len(allPhrases)
	X := make([][]float64, N)
	y := make([][]float64, N)
	for i := 0; i < N; i++ {
		X[i] = vectorizeBoW(allPhrases[i], vocab)
		y[i] = oneHotLabel(allLabels[i], nClasses)
	}

	// train/test split 80/20 deterministic
	perm := make([]int, N)
	for i := range perm {
		perm[i] = i
	}
	r.Shuffle(N, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
	trainN := (N*80)/100
	trainIdx := perm[:trainN]
	testIdx := perm[trainN:]

	// --- Model hyperparams ---
	hidden := 128
	lr := 0.01
	batchSize := 32
	epochs := 30

	// --- Initialize model parameters (simple 2-layer fully connected net) ---
	// W1: vocabSize x hidden ; b1: hidden
	// W2: hidden x nClasses ; b2: nClasses
	rng := rand.New(rand.NewSource(seed + 1))
	W1 := randMatrix(vocabSize, hidden, rng)
	b1 := make([]float64, hidden)
	W2 := randMatrix(hidden, nClasses, rng)
	b2 := make([]float64, nClasses)

	// helper closures for forward/backward
	// forward: returns logits and caches
	forward := func(x []float64) (logits []float64, a1 []float64) {
		// z1 = x * W1 + b1
		a1 = make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			sum := b1[j]
			for i := 0; i < vocabSize; i++ {
				sum += x[i] * W1[i][j]
			}
			// ReLU
			if sum > 0 {
				a1[j] = sum
			} else {
				a1[j] = 0
			}
		}
		// logits = a1 * W2 + b2
		logits = make([]float64, nClasses)
		for k := 0; k < nClasses; k++ {
			s := b2[k]
			for j := 0; j < hidden; j++ {
				s += a1[j] * W2[j][k]
			}
			logits[k] = s
		}
		return logits, a1
	}

	// training loop: very basic SGD with manual grads
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		// shuffle training idx
		r.Shuffle(len(trainIdx), func(i, j int) { trainIdx[i], trainIdx[j] = trainIdx[j], trainIdx[i] })
		for bsStart := 0; bsStart < len(trainIdx); bsStart += batchSize {
			bsEnd := bsStart + batchSize
			if bsEnd > len(trainIdx) {
				bsEnd = len(trainIdx)
			}
			// zero grads
			// accumulate grads over batch
			W2grad := zeroMatrix(hidden, nClasses)
			b2grad := make([]float64, nClasses)
			W1grad := zeroMatrix(vocabSize, hidden)
			b1grad := make([]float64, hidden)
			bsize := bsEnd - bsStart
			for _, ii := range trainIdx[bsStart:bsEnd] {
				x := X[ii]
				target := y[ii]
				logits, a1 := forward(x)
				probs := softmax(logits)
				loss := crossEntropyLoss(probs, target)
				totalLoss += loss
				// grads for logits: dL/dlogits = probs - target
				dlog := make([]float64, nClasses)
				for k := 0; k < nClasses; k++ {
					dlog[k] = probs[k] - target[k]
					b2grad[k] += dlog[k]
					for j := 0; j < hidden; j++ {
						W2grad[j][k] += a1[j] * dlog[k]
					}
				}
				// backprop into a1: da1 = W2 * dlog
				da1 := make([]float64, hidden)
				for j := 0; j < hidden; j++ {
					s := 0.0
					for k := 0; k < nClasses; k++ {
						s += W2[j][k] * dlog[k]
					}
					// ReLU grad
					if a1[j] > 0 {
						da1[j] = s
					} else {
						da1[j] = 0
					}
					b1grad[j] += da1[j]
					for i := 0; i < vocabSize; i++ {
						W1grad[i][j] += x[i] * da1[j]
					}
				}
			}
			// sgd step (average grads)
			invB := 1.0 / float64(bsize)
			for j := 0; j < hidden; j++ {
				b1grad[j] *= invB
			}
			for k := 0; k < nClasses; k++ {
				b2grad[k] *= invB
			}
			for i := 0; i < vocabSize; i++ {
				for j := 0; j < hidden; j++ {
					W1grad[i][j] *= invB
				}
			}
			for j := 0; j < hidden; j++ {
				for k := 0; k < nClasses; k++ {
					W2grad[j][k] *= invB
				}
			}
			// update
			for i := 0; i < vocabSize; i++ {
				for j := 0; j < hidden; j++ {
					W1[i][j] -= lr * W1grad[i][j]
				}
			}
			for j := 0; j < hidden; j++ {
				b1[j] -= lr * b1grad[j]
				for k := 0; k < nClasses; k++ {
					W2[j][k] -= lr * W2grad[j][k]
				}
			}
			for k := 0; k < nClasses; k++ {
				b2[k] -= lr * b2grad[k]
			}
		}
		avgLoss := totalLoss / float64(trainN)
		if epoch%5 == 0 {
			t.Logf("epoch %d avg loss %.4f", epoch, avgLoss)
		}
	}

	// evaluate on test set
	correct := 0
	for _, ii := range testIdx {
		logits, _ := forward(X[ii])
		pred := argmax(softmax(logits))
		if pred == allLabels[ii] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(testIdx))
	t.Logf("test accuracy: %.4f", acc)
	if acc < 0.85 {
		t.Fatalf("test_accuracy %.4f < 0.85", acc)
	}

	// manual inference checks (8-12 phrases) — make sure some expected phrase maps correctly
	sampleChecks := []struct{
		text string
		want int
	}{
		{"I have a question about my invoice and refund", 0},
		{"My wifi has a constant error and high latency", 1},
		{"Where is my package tracking number", 2},
		{"Hello I need general support information", 3},
	}
	ok := 0
	for _, sc := range sampleChecks {
		v := vectorizeBoW(sc.text, vocab)
		logits, _ := forward(v)
		pred := argmax(softmax(logits))
		if pred == sc.want {
			ok++
		}
	}
	t.Logf("manual check passed %d/%d", ok, len(sampleChecks))
	if ok < int(math.Ceil(0.75*float64(len(sampleChecks)))) {
		t.Fatalf("manual check success rate %d/%d < 75%%", ok, len(sampleChecks))
	}
}

// ----------------- small helpers for matrices -----------------
func randMatrix(r, c int, rng *rand.Rand) [][]float64 {
	m := make([][]float64, r)
	// Xavier init-ish
	scale := math.Sqrt(2.0 / float64(r+c))
	for i := range m {
		m[i] = make([]float64, c)
		for j := range m[i] {
			m[i][j] = rng.NormFloat64() * scale
		}
	}
	return m
}

func zeroMatrix(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}
