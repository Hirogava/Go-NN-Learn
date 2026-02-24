package train

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
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
