package applied

import (
	"math"
	"math/rand"
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
	samples  = 1200
	features = 5
)

// --- Dataset ---------------------------------------------------------------------------------
type RiskDataset struct {
	X [][]float64
	Y [][]float64
}

func (d *RiskDataset) Len() int {
	return len(d.X)
}

// NOTE: DataLoader expects pointers to tensor.Tensor here
func (d *RiskDataset) Get(i int) (*tensor.Tensor, *tensor.Tensor) {
	x := &tensor.Tensor{
		Data:    append([]float64{}, d.X[i]...),
		Shape:   []int{features},
		Strides: []int{1},
	}
	y := &tensor.Tensor{
		Data:    []float64{d.Y[i][0]},
		Shape:   []int{1}, // scalar label
		Strides: []int{1},
	}
	return x, y
}

func generateData(seed int64) *RiskDataset {
	rng := rand.New(rand.NewSource(seed))

	X := make([][]float64, samples)
	Y := make([][]float64, samples)

	for i := 0; i < samples; i++ {
		income := rng.Float64()*100000 + 20000
		age := rng.Float64()*40 + 20
		debt := rng.Float64() * 50000
		history := rng.Float64()
		ratio := debt / income

		x := []float64{
			income,
			age,
			debt,
			history,
			ratio,
		}

		score :=
			-0.00002*income +
				0.00004*debt -
				2*history +
				3*ratio +
				rng.NormFloat64()*0.2

		label := 0.0
		if score > 0 {
			label = 1
		}

		X[i] = x
		Y[i] = []float64{label}
	}

	return &RiskDataset{X: X, Y: Y}
}

func standardize(data [][]float64) {
	means := make([]float64, features)
	stds := make([]float64, features)
	n := float64(len(data))

	for _, row := range data {
		for j := 0; j < features; j++ {
			means[j] += row[j]
		}
	}
	for j := 0; j < features; j++ {
		means[j] /= n
	}
	for _, row := range data {
		for j := 0; j < features; j++ {
			diff := row[j] - means[j]
			stds[j] += diff * diff
		}
	}
	for j := 0; j < features; j++ {
		stds[j] = math.Sqrt(stds[j] / n)
		if stds[j] == 0 {
			stds[j] = 1
		}
	}
	for i := range data {
		for j := 0; j < features; j++ {
			data[i][j] = (data[i][j] - means[j]) / stds[j]
		}
	}
}

func accuracy(preds []float64, labels []float64) float64 {
	correct := 0
	for i := range preds {
		p := 0.0
		if preds[i] > 0.5 {
			p = 1
		}
		if p == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(preds))
}

// --- Simple sequential Module (локальная обёртка, чтобы не менять библиотеку) -------------
type SeqModule struct {
	layers []layers.Layer
}

func NewSeqModule(l ...layers.Layer) *SeqModule {
	return &SeqModule{layers: l}
}

func (s *SeqModule) Layers() []layers.Layer {
	return s.layers
}

func (s *SeqModule) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}

func (s *SeqModule) Params() []*graph.Node {
	var ps []*graph.Node
	for _, l := range s.layers {
		ps = append(ps, l.Params()...)
	}
	return ps
}

func (s *SeqModule) Train() {
	for _, l := range s.layers {
		l.Train()
	}
}

func (s *SeqModule) Eval() {
	for _, l := range s.layers {
		l.Eval()
	}
}

// --- уникальный const scheduler (имя не конфликтует с другими тестами) ------------------
type constSchedulerRisk struct {
	lr float64
}

func (c *constSchedulerRisk) Step() float64 {
	return c.lr
}

func (c *constSchedulerRisk) GetLastLR() float64 {
	return c.lr
}

// --- main run seed ---------------------------------------------------------------
func runSeed(seed int64) float64 {
	ds := generateData(seed)
	standardize(ds.X)

	split := int(float64(samples) * 0.8)

	trainSet := &RiskDataset{
		X: ds.X[:split],
		Y: ds.Y[:split],
	}

	testSet := &RiskDataset{
		X: ds.X[split:],
		Y: ds.Y[split:],
	}

	// --- создание DataLoader (возвращает *dataloader.DataLoader) ---
	trainLoader := dataloader.NewDataLoader(trainSet, dataloader.DataLoaderConfig{
    	BatchSize: 32,
	    Shuffle:   true,
    	Seed:      seed,
	})

	// --- создание callbacks (NewCallbackList возвращает *CallbackList) ---
	callbacks := train.NewCallbackList()

	// init func (локальная инициализация)
	initFn := func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * 0.1
		}
	}

	
	// слои
	d1 := layers.NewDense(features, 32, initFn)
	d2 := layers.NewDense(32, 1, initFn) // single output scalar

	model := NewSeqModule(d1, d2)

	// optimizer (Adam signature: lr, beta1, beta2, eps)
	opt := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	// если в оптимайзере требуется регистрация параметров, можно в тесте вызвать:
	// opt.SetParams(model.Params())  // <- если такая функция присутствует в repo

	// loss: используем MSE (Trainer поддерживает MSE)
	loss := &autograd.MSELossOp{}

	// lr scheduler — простой константный
	scheduler := &constSchedulerRisk{lr: 0.01}

	// metric
	accMetric := metrics.NewAccuracy()

	// --- вызов NewTrainer ---
	trainer := train.NewTrainer(
    	model,
    	trainLoader,      
    	opt,
    	loss,
    	scheduler,
    	accMetric,
    	callbacks,        
    	30,               
	)

	// запуск обучения
	trainer.Train()

	// --- inference on test set (по одному примеру) ---
	preds := []float64{}
	labels := []float64{}

	for i := 0; i < testSet.Len(); i++ {
		x, y := testSet.Get(i)
		n := graph.NewNode(x, nil, nil) // leaf node: no parents, no op
		out := model.Forward(n)
		if out != nil && out.Value != nil && len(out.Value.Data) > 0 {
			preds = append(preds, out.Value.Data[0])
		} else {
			preds = append(preds, 0.0)
		}
		labels = append(labels, y.Data[0])
	}

	return accuracy(preds, labels)
}

func TestTabularRiskScoring(t *testing.T) {
	seeds := []int64{1, 2, 3}
	results := []float64{}

	for _, s := range seeds {
		acc := runSeed(s)
		results = append(results, acc)
		if acc < 0.85 {
			t.Fatalf("accuracy too low: %f", acc)
		}
	}

	mean := 0.0
	for _, v := range results {
		mean += v
	}
	mean /= float64(len(results))

	variance := 0.0
	for _, v := range results {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(results))

	if variance > 0.05 {
		t.Fatalf("variance too high: %f", variance)
	}
}