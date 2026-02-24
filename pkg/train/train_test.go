package train

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// простой фейковый слой, возвращает заданные параметры
type fakeLayer struct {
	params []*graph.Node
}

func (f *fakeLayer) Params() []*graph.Node { return f.params }

func (f *fakeLayer) Forward(x *graph.Node) *graph.Node { return x }

// простой фейковый модель, хранит последний предикт как узел
type fakeModel struct {
	lastPred *graph.Node
}

func (m *fakeModel) Forward(n *graph.Node) *graph.Node {
	// всегда возвращаем предсказание 2.0 (чтобы совпадало с таргетом в тесте)
	m.lastPred = graph.NewNode(&tensor.Tensor{Data: []float64{2.0}, Shape: []int{1}}, nil, nil)
	return m.lastPred
}

func (m *fakeModel) Layers() []layers.Layer {
	if m.lastPred == nil {
		return []layers.Layer{&fakeLayer{params: []*graph.Node{}}}
	}
	return []layers.Layer{&fakeLayer{params: []*graph.Node{m.lastPred}}}
}

func (m *fakeModel) Params() []*graph.Node {
	if m.lastPred == nil {
		return []*graph.Node{}
	}
	return []*graph.Node{m.lastPred}
}

// фейковый оптимизатор для проверки вызова Step
type fakeOpt struct {
	stepped bool
	lr      float64
}

func (o *fakeOpt) Step(params []*graph.Node)     { o.stepped = true }
func (o *fakeOpt) SetLearningRate(lr float64)    { o.lr = lr }
func (o *fakeOpt) ZeroGrad(params []*graph.Node) { /* no-op for test */ }

func TestProcessBatch_ComputesLossAndStepsOptimizer(t *testing.T) {
	model := &fakeModel{}
	opt := &fakeOpt{}

	// создаём тренер вручную, используем MSE в качестве типа lossFn
	tr := &Trainer{
		model:   model,
		opt:     opt,
		lossFn:  &autograd.MSELossOp{}, // type used in switch внутри calculateLoss
		metric:  metrics.NewMAE(),
		context: *NewTrainingContext(model, 1),
	}

	// батч: фича 1.0, таргет 2.0 -> модель вернёт 2.0, метрика accuracy должна пройти
	batch := &dataloader.Batch{
		Features: &tensor.Tensor{Data: []float64{1.0}, Shape: []int{1}},
		Targets:  &tensor.Tensor{Data: []float64{2.0}, Shape: []int{1}},
	}

	if err := tr.processBatch(batch); err != nil {
		t.Fatalf("processBatch returned error: %v", err)
	}

	// проверяем, что loss записался в метрики
	if tr.context.Metrics == nil {
		t.Fatalf("expected Metrics to be initialized")
	}
	if _, ok := tr.context.Metrics["loss"]; !ok {
		t.Fatalf("expected loss in Metrics")
	}

	// проверяем, что оптимизатор сделал step
	if !opt.stepped {
		t.Fatalf("expected optimizer Step to be called")
	}
}

// TestNewTrainerFromConfig_CreatesTrainer проверяет, что Trainer создаётся через конфиг
// и контекст содержит правильное число эпох из конфига.
func TestNewTrainerFromConfig_CreatesTrainer(t *testing.T) {
	cfg := &TrainerConfig{
		Epochs:    5,
		BatchSize: 2,
		Device:    "cpu",
		Seed:      42,
	}
	features := &tensor.Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{4, 1}}
	targets := &tensor.Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{4, 1}}
	ds := dataloader.NewSimpleDataset(features, targets)
	dl := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: 2,
		Shuffle:   false,
		Seed:      cfg.Seed,
	})
	model := &fakeModel{}
	opt := &fakeOpt{lr: 0.01}
	scheduler := optimizers.NewStepLR(0.01, 0.5, 1)

	tr := NewTrainerFromConfig(
		cfg,
		model,
		dl,
		opt,
		&autograd.MSELossOp{},
		scheduler,
		metrics.NewMAE(),
		NewCallbackList(),
	)

	if tr == nil {
		t.Fatal("NewTrainerFromConfig returned nil")
	}
	if tr.context.NumEpochs != cfg.Epochs {
		t.Fatalf("expected NumEpochs %d, got %d", cfg.Epochs, tr.context.NumEpochs)
	}
}

// TestTrainerFromConfig_Deterministic проверяет, что при одном и том же seed
// два тренера, созданных через конфиг и обработавших один батч, дают одинаковый loss.
func TestTrainerFromConfig_Deterministic(t *testing.T) {
	seed := int64(12345)
	cfg := &TrainerConfig{
		Epochs:    1,
		BatchSize: 1,
		Device:    "cpu",
		Seed:      seed,
	}
	features := &tensor.Tensor{Data: []float64{1.0, 2.0}, Shape: []int{2, 1}}
	targets := &tensor.Tensor{Data: []float64{2.0, 3.0}, Shape: []int{2, 1}}
	ds1 := dataloader.NewSimpleDataset(features, targets)
	ds2 := dataloader.NewSimpleDataset(features, targets)
	dlConfig := dataloader.DataLoaderConfig{
		BatchSize: 1,
		Shuffle:   true,
		Seed:      seed,
	}
	dl1 := dataloader.NewDataLoader(ds1, dlConfig)
	dl2 := dataloader.NewDataLoader(ds2, dlConfig)

	tr1 := NewTrainerFromConfig(
		cfg,
		&fakeModel{},
		dl1,
		&fakeOpt{},
		&autograd.MSELossOp{},
		optimizers.NewStepLR(0.01, 0.5, 1),
		metrics.NewMAE(),
		NewCallbackList(),
	)
	tr2 := NewTrainerFromConfig(
		cfg,
		&fakeModel{},
		dl2,
		&fakeOpt{},
		&autograd.MSELossOp{},
		optimizers.NewStepLR(0.01, 0.5, 1),
		metrics.NewMAE(),
		NewCallbackList(),
	)

	batch1 := dl1.Next()
	batch2 := dl2.Next()

	if err := tr1.processBatch(batch1); err != nil {
		t.Fatalf("tr1 processBatch: %v", err)
	}
	if err := tr2.processBatch(batch2); err != nil {
		t.Fatalf("tr2 processBatch: %v", err)
	}

	loss1 := tr1.context.Metrics["loss"]
	loss2 := tr2.context.Metrics["loss"]
	if loss1 != loss2 {
		t.Fatalf("deterministic test failed: loss1=%v loss2=%v (expected equal for same seed)", loss1, loss2)
	}
}
