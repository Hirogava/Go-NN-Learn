package autograd

import (
	"testing"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
)

// Вспомогательная функция для создания тензора из массива
func newTensor(data []float64, shape ...int) *tensor.Tensor {
	strides := make([]int, len(shape))
	stride := 1
	for i := range len(shape) {
		idx := len(shape) - 1 - i
		strides[idx] = stride
		stride *= shape[idx]
	}
	return &tensor.Tensor{
		Data:    data,
		Shape:   shape,
		Strides: strides,
	}
}

// =============================================================================
// Grad-Check Tests для MSELoss
// =============================================================================

func TestMSELossGradCheck(t *testing.T) {
	// Простой случай: вектор
	pred := newTensor([]float64{1.0, 2.0, 3.0}, 3)
	target := newTensor([]float64{1.5, 2.5, 2.5}, 3)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.MSELoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("MSELoss gradient check failed for vector")
	}
}

func TestMSELossGradCheckMatrix(t *testing.T) {
	// Более сложный случай: матрица 3x4
	pred := tensor.Randn([]int{3, 4}, 42)
	target := tensor.Randn([]int{3, 4}, 123)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.MSELoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("MSELoss gradient check failed for matrix")
	}
}

func TestMSELossGradCheckLargeBatch(t *testing.T) {
	// Батч 32x10
	pred := tensor.Randn([]int{32, 10}, 555)
	target := tensor.Randn([]int{32, 10}, 777)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.MSELoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("MSELoss gradient check failed for large batch")
	}
}

// =============================================================================
// Grad-Check Tests для CrossEntropyLoss
// =============================================================================

func TestCrossEntropyLossGradCheck(t *testing.T) {
	// Простой случай: 2 примера, 3 класса
	logits := newTensor([]float64{
		1.0, 2.0, 3.0,
		3.0, 1.0, 2.0,
	}, 2, 3)

	// One-hot метки
	target := newTensor([]float64{
		0.0, 0.0, 1.0,
		1.0, 0.0, 0.0,
	}, 2, 3)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.CrossEntropyLoss(inputs[0], target)
	}

	logitsNode := graph.NewNode(logits, nil, nil)
	inputs := []*graph.Node{logitsNode}

	if !CheckGradientEngine(build, inputs, 1e-5, 1e-3) {
		t.Error("CrossEntropyLoss gradient check failed")
	}
}

func TestCrossEntropyLossGradCheckSingleExample(t *testing.T) {
	// Один пример, 5 классов
	logits := newTensor([]float64{0.5, 1.0, 0.2, 1.5, 0.8}, 1, 5)

	// Класс 3
	target := newTensor([]float64{0.0, 0.0, 0.0, 1.0, 0.0}, 1, 5)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.CrossEntropyLoss(inputs[0], target)
	}

	logitsNode := graph.NewNode(logits, nil, nil)
	inputs := []*graph.Node{logitsNode}

	if !CheckGradientEngine(build, inputs, 1e-5, 1e-3) {
		t.Error("CrossEntropyLoss gradient check failed for single example")
	}
}

func TestCrossEntropyLossGradCheckLargeBatch(t *testing.T) {
	// Батч 16x10 (реалистичный размер)
	logits := tensor.Randn([]int{16, 10}, 42)

	// Создаём one-hot метки
	target := tensor.Zeros(16, 10)
	for i := range 16 {
		classIdx := i % 10 // Циклически распределяем классы
		target.Data[i*10+classIdx] = 1.0
	}

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.CrossEntropyLoss(inputs[0], target)
	}

	logitsNode := graph.NewNode(logits, nil, nil)
	inputs := []*graph.Node{logitsNode}

	if !CheckGradientEngine(build, inputs, 1e-5, 1e-3) {
		t.Error("CrossEntropyLoss gradient check failed for large batch")
	}
}

func TestCrossEntropyLossGradCheckExtremeValues(t *testing.T) {
	// Проверка с экстремальными значениями
	logits := newTensor([]float64{
		-10.0, 100.0, -50.0,
		50.0, -100.0, 10.0,
	}, 2, 3)

	target := newTensor([]float64{
		0.0, 1.0, 0.0,
		1.0, 0.0, 0.0,
	}, 2, 3)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.CrossEntropyLoss(inputs[0], target)
	}

	logitsNode := graph.NewNode(logits, nil, nil)
	inputs := []*graph.Node{logitsNode}

	// Используем более слабый tolerance для экстремальных значений
	if !CheckGradientEngine(build, inputs, 1e-5, 1e-2) {
		t.Error("CrossEntropyLoss gradient check failed for extreme values")
	}
}

// =============================================================================
// Grad-Check Tests для HingeLoss
// =============================================================================

func TestHingeLossGradCheck(t *testing.T) {
	// Простой случай
	pred := newTensor([]float64{0.5, -0.5, 1.5}, 3)
	target := newTensor([]float64{1.0, 1.0, 1.0}, 3)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.HingeLoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("HingeLoss gradient check failed")
	}
}

func TestHingeLossGradCheckBinaryLabels(t *testing.T) {
	// Бинарные метки (-1, +1)
	pred := newTensor([]float64{0.3, -0.8, 1.2, -0.5, 0.9}, 5)
	target := newTensor([]float64{1.0, -1.0, 1.0, -1.0, 1.0}, 5)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.HingeLoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("HingeLoss gradient check failed for binary labels")
	}
}

func TestHingeLossGradCheckMatrix(t *testing.T) {
	// Матрица 4x3
	pred := tensor.Randn([]int{4, 3}, 42)
	target := tensor.Ones(4, 3)
	// Меняем некоторые метки на -1
	for i := range 6 {
		target.Data[i*2] = -1.0
	}

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.HingeLoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("HingeLoss gradient check failed for matrix")
	}
}

func TestHingeLossGradCheckLargeBatch(t *testing.T) {
	// Батч 64x1 (типичный для бинарной классификации)
	pred := tensor.Randn([]int{64, 1}, 999)
	target := tensor.Ones(64, 1)
	// Половина меток -1
	for i := range 32 {
		target.Data[i] = -1.0
	}

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.HingeLoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	if !CheckGradientEngine(build, inputs, 1e-6, 1e-4) {
		t.Error("HingeLoss gradient check failed for large batch")
	}
}

func TestHingeLossGradCheckNearBoundary(t *testing.T) {
	// Значения около границы margin=1
	// Этот тест может быть нестабильным из-за разрывной производной max()
	// около нуля, поэтому используем значения чуть дальше от границы
	pred := newTensor([]float64{
		0.9, 1.1, // Около границы для y=+1
		-0.9, -1.1, // Около границы для y=-1
	}, 4)
	target := newTensor([]float64{
		1.0, 1.0,
		-1.0, -1.0,
	}, 4)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.HingeLoss(inputs[0], target)
	}

	predNode := graph.NewNode(pred, nil, nil)
	inputs := []*graph.Node{predNode}

	// Используем более слабый tolerance около границы из-за разрывной производной
	if !CheckGradientEngine(build, inputs, 1e-4, 1e-3) {
		t.Error("HingeLoss gradient check failed near boundary")
	}
}

// =============================================================================
// Интеграционные Grad-Check тесты
// =============================================================================

func TestLossesGradCheckComparison(t *testing.T) {
	// Проверяем что все три функции потерь проходят grad-check на одинаковых данных

	// MSE
	predMSE := tensor.Randn([]int{8, 5}, 111)
	targetMSE := tensor.Randn([]int{8, 5}, 222)

	buildMSE := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.MSELoss(inputs[0], targetMSE)
	}

	if !CheckGradientEngine(buildMSE, []*graph.Node{graph.NewNode(predMSE, nil, nil)}, 1e-6, 1e-4) {
		t.Error("MSE comparison test failed")
	}

	// CrossEntropy
	logitsCE := tensor.Randn([]int{8, 5}, 333)
	targetCE := tensor.Zeros(8, 5)
	for i := range 8 {
		targetCE.Data[i*5+i%5] = 1.0
	}

	buildCE := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.CrossEntropyLoss(inputs[0], targetCE)
	}

	if !CheckGradientEngine(buildCE, []*graph.Node{graph.NewNode(logitsCE, nil, nil)}, 1e-5, 1e-3) {
		t.Error("CrossEntropy comparison test failed")
	}

	// Hinge
	predHinge := tensor.Randn([]int{8, 5}, 444)
	targetHinge := tensor.Ones(8, 5)
	for i := range 20 {
		targetHinge.Data[i] = -1.0
	}

	buildHinge := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.HingeLoss(inputs[0], targetHinge)
	}

	if !CheckGradientEngine(buildHinge, []*graph.Node{graph.NewNode(predHinge, nil, nil)}, 1e-6, 1e-4) {
		t.Error("Hinge comparison test failed")
	}
}

func TestLossesGradCheckChainedOperations(t *testing.T) {
	// Проверяем градиенты когда loss используется после других операций
	input := tensor.Randn([]int{4, 3}, 555)
	target := tensor.Randn([]int{4, 3}, 666)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		// input -> ReLU -> MSE
		activated := e.ReLU(inputs[0])
		return e.MSELoss(activated, target)
	}

	inputNode := graph.NewNode(input, nil, nil)
	if !CheckGradientEngine(build, []*graph.Node{inputNode}, 1e-6, 1e-4) {
		t.Error("Chained operations grad-check failed")
	}
}
