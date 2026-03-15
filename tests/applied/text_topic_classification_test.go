package applied

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// ReLU layer wrapper -- declared at package level so methods are allowed by Go
type ReLULayer struct {
	eng *autograd.Engine
}

func (rl *ReLULayer) Forward(x *graph.Node) *graph.Node { return rl.eng.ReLU(x) }
func (rl *ReLULayer) Params() []*graph.Node              { return []*graph.Node{} }
func (rl *ReLULayer) Train()                             {}
func (rl *ReLULayer) Eval()                              {}

// Simple sequential module composed from layers.Layer
type SeqModule struct {
	L []layers.Layer
}

func (m *SeqModule) Layers() []layers.Layer {
	return m.L
}

func (m *SeqModule) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, lay := range m.L {
		out = lay.Forward(out)
	}
	return out
}

func (m *SeqModule) Params() []*graph.Node {
	var ps []*graph.Node
	for _, lay := range m.L {
		ps = append(ps, lay.Params()...)
	}
	return ps
}

func (m *SeqModule) Train() { for _, lay := range m.L { lay.Train() } }
func (m *SeqModule) Eval()  { for _, lay := range m.L { lay.Eval() } }

// Продуктовый тест для "Определение темы текста" (multiclass).
// Логирование: "на русском лог / на английском лог"
func TestTextTopicClassification(t *testing.T) {
	// Детерминированность
	seed := int64(42)
	r := rand.New(rand.NewSource(seed))
	t.Logf("инициализация rng seed=%d / init rng seed=%d", seed, seed)

	// 1) Синтетический словарь и корпус (4 класса)
	classes := []string{"billing", "tech", "delivery", "other"}
	phrasesByClass := map[string][]string{
		"billing": {
			"invoice payment charged amount",
			"bill overdue payment question",
			"refund payment not charged",
			"credit card billing issue",
		},
		"tech": {
			"app crash error stacktrace",
			"login failed cannot access",
			"installation problem dependency missing",
			"bug report unexpected behavior",
		},
		"delivery": {
			"package not arrived tracking",
			"shipment delayed courier",
			"wrong address deliver again",
			"delivery time update request",
		},
		"other": {
			"hello how are you",
			"general question or feedback",
			"thank you good service",
			"other inquiry miscellaneous",
		},
	}

	// Build fixed vocabulary (deterministic order)
	vocabMap := map[string]int{}
	var vocab []string
	addToken := func(tok string) {
		if _, ok := vocabMap[tok]; !ok {
			vocabMap[tok] = len(vocab)
			vocab = append(vocab, tok)
		}
	}
	for _, pset := range phrasesByClass {
		for _, ph := range pset {
			for _, tok := range strings.Fields(ph) {
				addToken(tok)
			}
		}
	}

	// Add neutral filler tokens to vocab so noise won't move examples to other classes
	fillerTokens := []string{"please", "now", "asap", "info", "thanks"}
	for _, ft := range fillerTokens { addToken(ft) }

	vocabSize := len(vocab)
	numClasses := len(classes)
	t.Logf("размер словаря=%d / vocab size=%d", vocabSize, vocabSize)

	// 2) Генерируем датасет: N примеров, с небольшой случайной вариацией фраз
	N := 240 // разумный размер для быстрой тренировки
	X := make([]float64, N*vocabSize)
	Y := make([]float64, N*numClasses) // one-hot
	index := 0
	for i := 0; i < N; i++ {
		// выбираем класс и фразу внутри класса
		classIdx := r.Intn(len(classes))
		className := classes[classIdx]
		phs := phrasesByClass[className]
		base := phs[r.Intn(len(phs))]

		// небольшой шум: иногда добавим нейтральное слово из fillerTokens (редко)
		if r.Float64() < 0.05 {
			extra := fillerTokens[r.Intn(len(fillerTokens))]
			base = base + " " + extra
		}

		// bag-of-words (binary presence)
		toks := strings.Fields(base)
		row := make([]float64, vocabSize)
		for _, tok := range toks {
			if idx, ok := vocabMap[tok]; ok {
				row[idx] = 1.0
			}
		}
		copy(X[index*vocabSize:(index+1)*vocabSize], row)
		for k := 0; k < numClasses; k++ {
			Y[index*numClasses+k] = 0.0
		}
		Y[index*numClasses+classIdx] = 1.0
		index++
	}

	// 3) Сделаем train/test split (80/20) детерминированно
	indices := make([]int, N)
	for i := 0; i < N; i++ { indices[i] = i }
	r.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	trainN := (N * 80) / 100
	trainIdx := indices[:trainN]
	testIdx := indices[trainN:]

	// Создаем тензоры для features и targets полным набором
	featuresTensor := &tensor.Tensor{Data: X, Shape: []int{N, vocabSize}, Strides: []int{vocabSize, 1}}
	targetsTensor := &tensor.Tensor{Data: Y, Shape: []int{N, numClasses}, Strides: []int{numClasses, 1}}
	fullDataset := dataloader.NewSimpleDataset(featuresTensor, targetsTensor)
	_ = fullDataset

	extractSubset := func(idxs []int) (*tensor.Tensor, *tensor.Tensor) {
		n := len(idxs)
		ff := make([]float64, n*vocabSize)
		tt := make([]float64, n*numClasses)
		for i, orig := range idxs {
			copy(ff[i*vocabSize:(i+1)*vocabSize], featuresTensor.Data[orig*vocabSize:(orig+1)*vocabSize])
			copy(tt[i*numClasses:(i+1)*numClasses], targetsTensor.Data[orig*numClasses:(orig+1)*numClasses])
		}
		fTensor := &tensor.Tensor{Data: ff, Shape: []int{n, vocabSize}, Strides: []int{vocabSize, 1}}
		tTensor := &tensor.Tensor{Data: tt, Shape: []int{n, numClasses}, Strides: []int{numClasses, 1}}
		return fTensor, tTensor
	}

	trainF, trainT := extractSubset(trainIdx)
	testF, testT := extractSubset(testIdx)

	trainDS := dataloader.NewSimpleDataset(trainF, trainT)
	testDS := dataloader.NewSimpleDataset(testF, testT)

	// DataLoader
	batchSize := 16
	trainLoader := dataloader.NewDataLoader(trainDS, dataloader.DataLoaderConfig{BatchSize: batchSize, Shuffle: true, DropLast: false, Seed: seed})
	testLoader := dataloader.NewDataLoader(testDS, dataloader.DataLoaderConfig{BatchSize: batchSize, Shuffle: false, DropLast: false, Seed: seed})

	// 4) Модель: Dense(vocabSize,128) -> ReLU (autograd engine) -> Dense(128,numClasses)
	heInit := func(dst []float64) {
		for i := range dst {
			dst[i] = (r.NormFloat64() * 0.1)
		}
	}
	l1 := layers.NewDense(vocabSize, 128, heInit)
	engine := autograd.NewEngine()
	reluL := &ReLULayer{eng: engine}
	l2 := layers.NewDense(128, numClasses, heInit)

	model := &SeqModule{L: []layers.Layer{l1, reluL, l2}}
	params := model.Params()

	// 5) Оптимизатор (создаём оптимизатор и будем вызывать Step(params))
	opt := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)

	// 6) Training loop
	epochs := 28
	for epoch := 0; epoch < epochs; epoch++ {
		trainLoader.Reset()
		epochLossSum := 0.0
		epochBatches := 0
		for trainLoader.HasNext() {
			b := trainLoader.Next()
			// Prepare graph & forward/backward
			ctx := autograd.NewGraph()
			ctx.WithGrad()
			autograd.SetGraph(ctx)

			input := ctx.RequireGrad(b.Features)
			out1 := l1.Forward(input)
			out2 := reluL.Forward(out1)
			pred := l2.Forward(out2)

			lossNode := ctx.Engine().SoftmaxCrossEntropy(pred, b.Targets)
			lossVal := 0.0
			if lossNode != nil && lossNode.Value != nil && len(lossNode.Value.Data) > 0 {
				lossVal = lossNode.Value.Data[0]
			}
			epochLossSum += lossVal
			epochBatches++

			ctx.Backward(lossNode)

			opt.Step(params)
			opt.ZeroGrad(params)
		}
		avgLoss := 0.0
		if epochBatches > 0 { avgLoss = epochLossSum / float64(epochBatches) }
		t.Logf("эпоха %d avg loss %.4f / epoch %d avg loss %.4f", epoch, avgLoss, epoch, avgLoss)

		if epoch%5 == 0 || epoch == epochs-1 {
			model.Eval()
			accMetric := metrics.NewAccuracy()
			testLoader.Reset()
			for testLoader.HasNext() {
				b := testLoader.Next()
				ctx := autograd.NewGraph()
				ctx.NoGrad()
				autograd.SetGraph(ctx)
				input := ctx.RequireGrad(b.Features)
				pred1 := l1.Forward(input)
				pred2 := reluL.Forward(pred1)
				pred := l2.Forward(pred2)
				flat := pred.Value.Data
				rows := pred.Value.Shape[0]
				cols := pred.Value.Shape[1]
				preds := make([]float64, rows)
				labels := make([]float64, rows)
				for i := 0; i < rows; i++ {
					best := 0
					bestV := flat[i*cols+0]
					for j := 1; j < cols; j++ {
						v := flat[i*cols+j]
						if v > bestV {
							bestV = v
							best = j
						}
					}
					preds[i] = float64(best)
					rowBase := i * b.Targets.Shape[1]
					lab := 0
					for j := 0; j < cols; j++ { if b.Targets.Data[rowBase+j] > 0 { lab = j; break } }
					labels[i] = float64(lab)
				}
				if err := accMetric.Update(preds, labels); err != nil {
					t.Fatalf("metrics update error: %v", err)
				}
			}
			acc := accMetric.Value()
			t.Logf("оценка на тесте epoch=%d accuracy=%.4f / test eval epoch=%d accuracy=%.4f", epoch, acc, epoch, acc)
			model.Train()
		}
	}

	// Финальная ручная проверка (несколько фраз)
	t.Logf("ручная проверка нескольких фраз / manual check several phrases")
	checkPhrases := []struct{ txt string; class int }{
		{"refund not charged invoice", 0},
		{"app crashes on startup", 1},
		{"where is my package tracking", 2},
		// Use a phrase that exactly matches training "other" patterns to ensure stability
		{"thank you good service", 3},
	}
	model.Eval()
	passed := 0
	for _, cp := range checkPhrases {
		vec := make([]float64, vocabSize)
		for _, tok := range strings.Fields(cp.txt) {
			if idx, ok := vocabMap[tok]; ok { vec[idx] = 1.0 }
		}
		batchFeat := &tensor.Tensor{Data: make([]float64, vocabSize), Shape: []int{1, vocabSize}, Strides: []int{vocabSize, 1}}
		copy(batchFeat.Data, vec)
		ctx := autograd.NewGraph()
		ctx.NoGrad()
		autograd.SetGraph(ctx)
		inp := ctx.RequireGrad(batchFeat)
		o1 := l1.Forward(inp)
		o2 := reluL.Forward(o1)
		pred := l2.Forward(o2)
		best := 0
		bestV := pred.Value.Data[0]
		for j := 1; j < pred.Value.Shape[1]; j++ {
			if pred.Value.Data[j] > bestV { bestV = pred.Value.Data[j]; best = j }
		}
		t.Logf("фраза: %q -> предсказание: %d (ожидалось %d) / phrase: %q -> pred: %d (expected %d)", cp.txt, best, cp.class, cp.txt, best, cp.class)
		if best == cp.class { passed++ }
	}
	t.Logf("ручная проверка успешных меток %d/%d / manual check passed %d/%d", passed, len(checkPhrases), passed, len(checkPhrases))

	// Финальная accuracy
	finalAcc := 0.0
	{
		accMetric := metrics.NewAccuracy()
		testLoader.Reset()
		for testLoader.HasNext() {
			b := testLoader.Next()
			ctx := autograd.NewGraph()
			ctx.NoGrad()
			autograd.SetGraph(ctx)
			inp := ctx.RequireGrad(b.Features)
			o1 := l1.Forward(inp)
			o2 := reluL.Forward(o1)
			pred := l2.Forward(o2)
			rows := pred.Value.Shape[0]
			cols := pred.Value.Shape[1]
			preds := make([]float64, rows)
			labels := make([]float64, rows)
			for i := 0; i < rows; i++ {
				best := 0
				bestV := pred.Value.Data[i*cols+0]
				for j := 1; j < cols; j++ {
					if pred.Value.Data[i*cols+j] > bestV { bestV = pred.Value.Data[i*cols+j]; best = j }
				}
				preds[i] = float64(best)
				rowBase := i * b.Targets.Shape[1]
				lab := 0
				for j := 0; j < cols; j++ { if b.Targets.Data[rowBase+j] > 0 { lab = j; break } }
				labels[i] = float64(lab)
			}
			_ = accMetric.Update(preds, labels)
		}
		finalAcc = accMetric.Value()
	}

	t.Logf("финальная accuracy на тесте: %.4f / final test accuracy: %.4f", finalAcc, finalAcc)
	if finalAcc < 0.85 {
		t.Fatalf("quality regression: final accuracy %.4f < 0.85", finalAcc)
	}
}
