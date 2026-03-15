package applied

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/train"
)

const (
	productSamples  = 1200
	productFeatures = 5
)

// Dataset now stores both continuous score (for training with MSE) and class label (for evaluation)
type RiskDatasetProduct struct {
	X     [][]float64
	Score []float64 // continuous target (used in training)
	Label []float64 // binary label 0/1 (used for accuracy/AUC)
}

func (d *RiskDatasetProduct) Len() int { return len(d.X) }

func (d *RiskDatasetProduct) Get(i int) (*tensor.Tensor, *tensor.Tensor) {
	x := &tensor.Tensor{
		Data:    append([]float64{}, d.X[i]...),
		Shape:   []int{productFeatures},
		Strides: []int{1},
	}
	// training target is continuous score (shape [1])
	y := &tensor.Tensor{
		Data:    []float64{d.Score[i]},
		Shape:   []int{1},
		Strides: []int{1},
	}
	return x, y
}

// generate dataset that returns both continuous score and binary label (score>0)
func generateProductData(seed int64) *RiskDatasetProduct {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, productSamples)
	scoreArr := make([]float64, productSamples)
	labelArr := make([]float64, productSamples)

	for i := 0; i < productSamples; i++ {
		income := rng.Float64()*80000 + 20000
		age := rng.Float64()*40 + 20
		debt := rng.Float64() * 40000
		history := rng.Float64()
		ratio := debt / (income + 1.0)

		// **continuous** score used to derive label
		score := -0.00005*income + 0.00008*debt - 3.0*history + 4.0*ratio + rng.NormFloat64()*0.1

		label := 0.0
		if score > 0 {
			label = 1.0
		}

		X[i] = []float64{income, age, debt, history, ratio}
		scoreArr[i] = score
		labelArr[i] = label
	}
	return &RiskDatasetProduct{X: X, Score: scoreArr, Label: labelArr}
}

func standardizeProduct(data [][]float64) {
	means := make([]float64, productFeatures)
	stds := make([]float64, productFeatures)
	n := float64(len(data))
	for _, row := range data {
		for j := 0; j < productFeatures; j++ {
			means[j] += row[j]
		}
	}
	for j := 0; j < productFeatures; j++ {
		means[j] /= n
	}
	for _, row := range data {
		for j := 0; j < productFeatures; j++ {
			diff := row[j] - means[j]
			stds[j] += diff * diff
		}
	}
	for j := 0; j < productFeatures; j++ {
		stds[j] = math.Sqrt(stds[j] / n)
		if stds[j] == 0 {
			stds[j] = 1
		}
	}
	for i := range data {
		for j := 0; j < productFeatures; j++ {
			data[i][j] = (data[i][j] - means[j]) / stds[j]
		}
	}
}

// simple sequential module (local) implementing layers.Module
type SeqModuleProduct struct {
	layers []layers.Layer
}

func NewSeqModuleProduct(l ...layers.Layer) *SeqModuleProduct { return &SeqModuleProduct{layers: l} }
func (s *SeqModuleProduct) Layers() []layers.Layer            { return s.layers }
func (s *SeqModuleProduct) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}
func (s *SeqModuleProduct) Params() []*graph.Node {
	var ps []*graph.Node
	for _, l := range s.layers {
		ps = append(ps, l.Params()...)
	}
	return ps
}
func (s *SeqModuleProduct) Train() { for _, l := range s.layers { l.Train() } }
func (s *SeqModuleProduct) Eval()  { for _, l := range s.layers { l.Eval() } }

// compute AUC (rank-sum / Mann-Whitney)
func computeAUC(scores []float64, labels []float64) float64 {
	n := len(scores)
	if n == 0 {
		return 0.5
	}
	pos := 0
	neg := 0
	type pair struct {
		s float64
		l float64
	}
	p := make([]pair, n)
	for i := 0; i < n; i++ {
		p[i] = pair{scores[i], labels[i]}
		if labels[i] > 0.5 {
			pos++
		} else {
			neg++
		}
	}
	if pos == 0 || neg == 0 {
		return 0.5
	}
	sort.Slice(p, func(i, j int) bool { return p[i].s > p[j].s })
	sumPosRanks := 0.0
	for i := 0; i < n; i++ {
		if p[i].l > 0.5 {
			sumPosRanks += float64(i + 1)
		}
	}
	auc := (sumPosRanks - float64(pos*(pos+1)/2)) / float64(pos*neg)
	return auc
}

func computeMSE(preds, labels []float64) float64 {
	n := float64(len(preds))
	if n == 0 {
		return 0.0
	}
	sum := 0.0
	for i := range preds {
		diff := preds[i] - labels[i]
		sum += diff * diff
	}
	return sum / n
}

// package-level simple const scheduler
type constSchedulerProduct struct{ lr float64 }
func (c *constSchedulerProduct) Step() float64     { return c.lr }
func (c *constSchedulerProduct) GetLastLR() float64 { return c.lr }

// single run: train MSE to predict continuous score; eval by thresholding at 0
func productRun(seed int64, t *testing.T) (float64, float64, float64, float64) {
	t.Logf("Генерация данных (seed=%d) / Generating data (seed=%d)", seed, seed)
	ds := generateProductData(seed)
	standardizeProduct(ds.X)

	split := int(float64(productSamples) * 0.8)
	trainSet := &RiskDatasetProduct{X: ds.X[:split], Score: ds.Score[:split], Label: ds.Label[:split]}
	testSet := &RiskDatasetProduct{X: ds.X[split:], Score: ds.Score[split:], Label: ds.Label[split:]}

	t.Logf("Создаём DataLoader / Creating DataLoader")
	trainLoader := dataloader.NewDataLoader(trainSet, dataloader.DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   true,
		Seed:      seed,
	})

	// init
	initFn := func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * 0.01
		}
	}

	t.Logf("Строим модель Dense(5->1) (регрессия score) / Building model Dense(5->1) (score regression)")
	d1 := layers.NewDense(productFeatures, 1, initFn)
	model := NewSeqModuleProduct(d1)

	t.Logf("Создаём оптимизатор Adam lr=0.01 / Creating Adam optimizer lr=0.01")
	opt := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)

	t.Logf("Loss: MSE (train continuous score) / Loss: MSE")
	loss := &autograd.MSELossOp{}

	t.Logf("Scheduler + metric + callbacks / Scheduler + metric + callbacks")
	sched := &constSchedulerProduct{lr: 0.01}
	accMetric := metrics.NewAccuracy()
	callbacks := train.NewCallbackList()

	t.Logf("Инициализируем тренера (epochs=60) / Initializing trainer (epochs=60)")
	trainer := train.NewTrainer(
		model,
		trainLoader, // deref as required by this repo's NewTrainer signature
		opt,
		loss,
		sched,
		accMetric,
		callbacks,
		60,
	)

	// initial train loss (predict continuous score, compare to continuous target)
	t.Logf("Вычисляем начальный loss (train) / Computing initial train loss")
	rawPreds := make([]float64, 0, trainSet.Len())
	rawLabels := make([]float64, 0, trainSet.Len())
	for i := 0; i < trainSet.Len(); i++ {
		x, y := trainSet.Get(i)
		n := graph.NewNode(x, nil, nil)
		out := model.Forward(n)
		val := 0.0
		if out != nil && out.Value != nil && len(out.Value.Data) > 0 {
			val = out.Value.Data[0]
		}
		rawPreds = append(rawPreds, val)
		rawLabels = append(rawLabels, y.Data[0]) // y is continuous score
	}
	initialLoss := computeMSE(rawPreds, rawLabels)
	t.Logf("Начальный MSE = %.6f / Initial MSE = %.6f", initialLoss, initialLoss)

	t.Logf("Запускаем обучение / Training started")
	trainer.Train()
	t.Logf("Обучение завершено / Training finished")

	// inference: predicted score; threshold at 0 => class; AUC on raw predicted scores
	preds := make([]float64, 0, testSet.Len())
	labels := make([]float64, 0, testSet.Len())
	for i := 0; i < testSet.Len(); i++ {
		x := &tensor.Tensor{Data: append([]float64{}, testSet.X[i]...), Shape: []int{productFeatures}, Strides: []int{1}}
		// note: use graph.NewNode(x, nil, nil)
		n := graph.NewNode(x, nil, nil)
		out := model.Forward(n)
		val := 0.0
		if out != nil && out.Value != nil && len(out.Value.Data) > 0 {
			val = out.Value.Data[0] // predicted continuous score
		}
		preds = append(preds, val)
		labels = append(labels, testSet.Label[i]) // binary 0/1
	}

	// accuracy by threshold at 0 (same rule as generation)
	correct := 0
	for i := range preds {
		predClass := 0.0
		if preds[i] > 0.0 {
			predClass = 1.0
		}
		if predClass == labels[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(preds))
	auc := computeAUC(preds, labels)

	// final train loss
	finalRawPreds := make([]float64, 0, trainSet.Len())
	finalRawLabels := make([]float64, 0, trainSet.Len())
	for i := 0; i < trainSet.Len(); i++ {
		x, y := trainSet.Get(i)
		n := graph.NewNode(x, nil, nil)
		out := model.Forward(n)
		val := 0.0
		if out != nil && out.Value != nil && len(out.Value.Data) > 0 {
			val = out.Value.Data[0]
		}
		finalRawPreds = append(finalRawPreds, val)
		finalRawLabels = append(finalRawLabels, y.Data[0])
	}
	finalLoss := computeMSE(finalRawPreds, finalRawLabels)

	t.Logf("Результат: accuracy=%.4f, AUC=%.4f / Result: accuracy=%.4f, AUC=%.4f", acc, auc, acc, auc)
	t.Logf("Итоговый MSE = %.6f / Final MSE = %.6f", finalLoss, finalLoss)
	return acc, auc, initialLoss, finalLoss
}

func TestProductTabularRiskScoring(t *testing.T) {
	t.Logf("=== ПРОДУКТОВЫЙ ТЕСТ: Скоринг заявки / PRODUCT TEST: Tabular Risk Scoring ===")
	seeds := []int64{1, 2, 3}
	resultsAcc := make([]float64, 0, len(seeds))
	resultsAuc := make([]float64, 0, len(seeds))
	initLosses := make([]float64, 0, len(seeds))
	finalLosses := make([]float64, 0, len(seeds))

	for _, s := range seeds {
		t.Logf("=== Запуск с seed=%d / Run with seed=%d ===", s, s)
		acc, auc, initLoss, finalLoss := productRun(s, t)
		resultsAcc = append(resultsAcc, acc)
		resultsAuc = append(resultsAuc, auc)
		initLosses = append(initLosses, initLoss)
		finalLosses = append(finalLosses, finalLoss)
	}

	// means & variance
	meanAcc := 0.0
	for _, v := range resultsAcc {
		meanAcc += v
	}
	meanAcc /= float64(len(resultsAcc))

	meanAuc := 0.0
	for _, v := range resultsAuc {
		meanAuc += v
	}
	meanAuc /= float64(len(resultsAuc))

	varAcc := 0.0
	for _, v := range resultsAcc {
		diff := v - meanAcc
		varAcc += diff * diff
	}
	varAcc /= float64(len(resultsAcc))

	// loss decrease average
	avgInit := 0.0
	avgFinal := 0.0
	for i := range initLosses {
		avgInit += initLosses[i]
		avgFinal += finalLosses[i]
	}
	avgInit /= float64(len(initLosses))
	avgFinal /= float64(len(finalLosses))
	lossDecrease := 1.0 - (avgFinal / avgInit)

	t.Logf("Средняя точность / Mean accuracy: %.4f; Средний AUC / Mean AUC: %.4f", meanAcc, meanAuc)
	t.Logf("Вариация accuracy / Accuracy variance: %.6f", varAcc)
	t.Logf("Средний MSE до/после: %.6f -> %.6f (падение %.2f%%) / Avg MSE before/after: %.6f -> %.6f (decrease %.2f%%)",
		avgInit, avgFinal, lossDecrease*100.0, avgInit, avgFinal, lossDecrease*100.0)

	// Business checks
	if !(meanAcc >= 0.85 || meanAuc >= 0.85) {
		t.Fatalf("Бизнес-критерий не выполнен: meanAcc=%.4f meanAuc=%.4f / Business criteria failed", meanAcc, meanAuc)
	}
	if varAcc > 0.05 {
		t.Fatalf("Нестабильность между запусками слишком велика: variance=%.6f / Stability between runs too high", varAcc)
	}
	if lossDecrease < 0.30 {
		t.Fatalf("Loss не упал на 30%%: decrease=%.2f%% / Loss did not decrease by 30%%", lossDecrease*100.0)
	}

	t.Logf("=== ПРОДУКТОВЫЙ ТЕСТ ПРОЙДЕН УСПЕШНО / PRODUCT TEST PASSED ===")
}