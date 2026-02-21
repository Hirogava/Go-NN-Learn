package autograd

import (
	"math"
	"testing"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
)

// Вспомогательная функция для создания тензора из массива
func newTensorTest(data []float64, shape ...int) *tensor.Tensor {
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

// Вспомогательная функция для сравнения float64 с погрешностью
func feq(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

// Вспомогательная функция для сравнения с заданным tolerance
func feqTol(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// =============================================================================
// MSELoss Tests
// =============================================================================

func TestMSELossForward(t *testing.T) {
	e := NewEngine()

	// Тестовые данные
	pred := e.RequireGrad(newTensorTest([]float64{1.0, 2.0, 3.0}, 3))
	target := newTensorTest([]float64{1.5, 2.5, 2.5}, 3)

	// Forward pass
	loss := e.MSELoss(pred, target)

	// Ожидаемый результат: mean([(1.0-1.5)^2, (2.0-2.5)^2, (3.0-2.5)^2])
	// = mean([0.25, 0.25, 0.25]) = 0.25
	expected := 0.25

	if !feq(loss.Value.Data[0], expected) {
		t.Errorf("MSELoss forward failed: expected %v, got %v", expected, loss.Value.Data[0])
	}
}

func TestMSELossBackward(t *testing.T) {
	e := NewEngine()

	// Тестовые данные
	pred := e.RequireGrad(newTensorTest([]float64{1.0, 2.0, 3.0}, 3))
	target := newTensorTest([]float64{1.5, 2.5, 2.5}, 3)

	// Forward + Backward
	loss := e.MSELoss(pred, target)
	e.Backward(loss)

	// Градиент: 2 * (pred - target) / n
	// = 2 * [-0.5, -0.5, 0.5] / 3
	// = [-1/3, -1/3, 1/3]
	expectedGrad := []float64{-1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0}

	for i, expected := range expectedGrad {
		if !feq(pred.Grad.Data[i], expected) {
			t.Errorf("MSELoss backward[%d] failed: expected %v, got %v",
				i, expected, pred.Grad.Data[i])
		}
	}
}

func TestMSELossMatrixBatch(t *testing.T) {
	e := NewEngine()

	// Батч предсказаний 2x3
	pred := e.RequireGrad(newTensorTest([]float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	}, 2, 3))

	target := newTensorTest([]float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 5.0,
	}, 2, 3)

	loss := e.MSELoss(pred, target)

	// Только последний элемент отличается: (6-5)^2 = 1
	// mean([0, 0, 0, 0, 0, 1]) = 1/6 ≈ 0.1667
	expected := 1.0 / 6.0

	if !feq(loss.Value.Data[0], expected) {
		t.Errorf("MSELoss batch forward failed: expected %v, got %v",
			expected, loss.Value.Data[0])
	}
}

func TestMSELossZero(t *testing.T) {
	e := NewEngine()

	// Идеальные предсказания
	pred := e.RequireGrad(newTensorTest([]float64{1.0, 2.0, 3.0}, 3))
	target := newTensorTest([]float64{1.0, 2.0, 3.0}, 3)

	loss := e.MSELoss(pred, target)

	if !feq(loss.Value.Data[0], 0.0) {
		t.Errorf("MSELoss with perfect predictions should be 0, got %v",
			loss.Value.Data[0])
	}

	// Градиент должен быть нулевым
	e.Backward(loss)
	for i := range pred.Grad.Data {
		if !feq(pred.Grad.Data[i], 0.0) {
			t.Errorf("MSELoss gradient[%d] should be 0, got %v",
				i, pred.Grad.Data[i])
		}
	}
}

// =============================================================================
// CrossEntropyLoss Tests
// =============================================================================

func TestCrossEntropyLossForward(t *testing.T) {
	e := NewEngine()

	// Логиты для 2 примеров, 3 класса
	logits := e.RequireGrad(newTensorTest([]float64{
		1.0, 2.0, 3.0, // Пример 1
		3.0, 1.0, 2.0, // Пример 2
	}, 2, 3))

	// One-hot метки: класс 2 для первого примера, класс 0 для второго
	target := newTensorTest([]float64{
		0.0, 0.0, 1.0,
		1.0, 0.0, 0.0,
	}, 2, 3)

	loss := e.CrossEntropyLoss(logits, target)

	// Loss должен быть положительным числом
	if loss.Value.Data[0] <= 0 {
		t.Errorf("CrossEntropyLoss should be positive, got %v", loss.Value.Data[0])
	}

	// Проверим что loss разумный (не NaN, не Inf)
	if math.IsNaN(loss.Value.Data[0]) || math.IsInf(loss.Value.Data[0], 0) {
		t.Errorf("CrossEntropyLoss is NaN or Inf: %v", loss.Value.Data[0])
	}
}

func TestCrossEntropyLossBackward(t *testing.T) {
	e := NewEngine()

	logits := e.RequireGrad(newTensorTest([]float64{
		1.0, 2.0, 3.0,
	}, 1, 3))

	// Класс 1
	target := newTensorTest([]float64{
		0.0, 1.0, 0.0,
	}, 1, 3)

	loss := e.CrossEntropyLoss(logits, target)
	e.Backward(loss)

	// Градиенты должны быть вычислены
	if logits.Grad == nil {
		t.Fatal("Gradients not computed")
	}

	// Сумма градиентов должна быть близка к 0 (так как softmax нормализован)
	sumGrad := 0.0
	for _, g := range logits.Grad.Data {
		sumGrad += g
	}

	if !feqTol(sumGrad, 0.0, 1e-6) {
		t.Errorf("Sum of CE gradients should be ~0, got %v", sumGrad)
	}

	// Градиент для правильного класса должен быть отрицательным
	// (softmax_prob - 1) < 0
	if logits.Grad.Data[1] >= 0 {
		t.Errorf("Gradient for correct class should be negative, got %v",
			logits.Grad.Data[1])
	}
}

func TestCrossEntropyLossNumericalStability(t *testing.T) {
	e := NewEngine()

	// Экстремальные значения для проверки численной стабильности
	logits := e.RequireGrad(newTensorTest([]float64{
		-1000.0, 1000.0, -500.0,
	}, 1, 3))

	target := newTensorTest([]float64{
		0.0, 1.0, 0.0,
	}, 1, 3)

	loss := e.CrossEntropyLoss(logits, target)

	// Проверяем что нет NaN или Inf
	if math.IsNaN(loss.Value.Data[0]) {
		t.Error("CrossEntropyLoss produced NaN with extreme values")
	}
	if math.IsInf(loss.Value.Data[0], 0) {
		t.Error("CrossEntropyLoss produced Inf with extreme values")
	}

	// Loss должен быть очень маленьким (почти 0), так как логит для
	// правильного класса очень большой
	if loss.Value.Data[0] > 1.0 {
		t.Errorf("CrossEntropyLoss with correct class having large logit "+
			"should be small, got %v", loss.Value.Data[0])
	}
}

func TestCrossEntropyLossPerfectPrediction(t *testing.T) {
	e := NewEngine()

	// Очень высокий логит для правильного класса
	logits := e.RequireGrad(newTensorTest([]float64{
		-10.0, 100.0, -10.0,
	}, 1, 3))

	target := newTensorTest([]float64{
		0.0, 1.0, 0.0,
	}, 1, 3)

	loss := e.CrossEntropyLoss(logits, target)

	// Loss должен быть близок к 0
	if loss.Value.Data[0] > 0.001 {
		t.Errorf("CrossEntropyLoss with perfect prediction should be ~0, got %v",
			loss.Value.Data[0])
	}
}

func TestCrossEntropyLossBatch(t *testing.T) {
	e := NewEngine()

	// Батч из 3 примеров
	logits := e.RequireGrad(newTensorTest([]float64{
		1.0, 2.0, 3.0,
		2.0, 3.0, 1.0,
		3.0, 1.0, 2.0,
	}, 3, 3))

	target := newTensorTest([]float64{
		0.0, 0.0, 1.0,
		0.0, 1.0, 0.0,
		1.0, 0.0, 0.0,
	}, 3, 3)

	loss := e.CrossEntropyLoss(logits, target)
	e.Backward(loss)

	// Проверяем что gradients вычислены для всего батча
	if logits.Grad == nil {
		t.Fatal("Gradients not computed for batch")
	}

	// Все градиенты должны быть конечными
	for i, g := range logits.Grad.Data {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Errorf("Gradient[%d] is NaN or Inf: %v", i, g)
		}
	}
}

// =============================================================================
// HingeLoss Tests
// =============================================================================

func TestHingeLossForward(t *testing.T) {
	e := NewEngine()

	// Предсказания
	pred := e.RequireGrad(newTensorTest([]float64{0.5, -0.5, 1.5}, 3))

	// Метки (-1 или +1)
	target := newTensorTest([]float64{1.0, 1.0, 1.0}, 3)

	loss := e.HingeLoss(pred, target)

	// Вычисляем вручную:
	// margin = [1 - 1*0.5, 1 - 1*(-0.5), 1 - 1*1.5]
	//        = [0.5, 1.5, -0.5]
	// max(0, margin) = [0.5, 1.5, 0]
	// mean = 2.0 / 3
	expected := 2.0 / 3.0

	if !feqTol(loss.Value.Data[0], expected, 1e-9) {
		t.Errorf("HingeLoss forward failed: expected %v, got %v",
			expected, loss.Value.Data[0])
	}
}

func TestHingeLossBackward(t *testing.T) {
	e := NewEngine()

	pred := e.RequireGrad(newTensorTest([]float64{0.5, -0.5, 1.5}, 3))
	target := newTensorTest([]float64{1.0, 1.0, 1.0}, 3)

	loss := e.HingeLoss(pred, target)
	e.Backward(loss)

	// Градиенты:
	// Для pred[0]: margin=0.5 > 0 => grad = -target/n = -1/3
	// Для pred[1]: margin=1.5 > 0 => grad = -1/3
	// Для pred[2]: margin=-0.5 <= 0 => grad = 0
	expectedGrad := []float64{-1.0 / 3.0, -1.0 / 3.0, 0.0}

	for i, expected := range expectedGrad {
		if !feq(pred.Grad.Data[i], expected) {
			t.Errorf("HingeLoss backward[%d] failed: expected %v, got %v",
				i, expected, pred.Grad.Data[i])
		}
	}
}

func TestHingeLossPerfectClassification(t *testing.T) {
	e := NewEngine()

	// Все предсказания правильные с большим margin
	pred := e.RequireGrad(newTensorTest([]float64{2.0, -2.0, 3.0}, 3))
	target := newTensorTest([]float64{1.0, -1.0, 1.0}, 3)

	loss := e.HingeLoss(pred, target)

	// Все margins отрицательны => loss = 0
	if !feq(loss.Value.Data[0], 0.0) {
		t.Errorf("HingeLoss with perfect classification should be 0, got %v",
			loss.Value.Data[0])
	}

	// Градиенты должны быть нулевыми
	e.Backward(loss)
	for i, g := range pred.Grad.Data {
		if !feq(g, 0.0) {
			t.Errorf("HingeLoss gradient[%d] should be 0, got %v", i, g)
		}
	}
}

func TestHingeLossAllWrong(t *testing.T) {
	e := NewEngine()

	// Все предсказания полностью неправильные
	pred := e.RequireGrad(newTensorTest([]float64{-1.0, 1.0, -1.0}, 3))
	target := newTensorTest([]float64{1.0, -1.0, 1.0}, 3)

	loss := e.HingeLoss(pred, target)

	// margins = [1-(-1), 1-(-1), 1-(-1)] = [2, 2, 2]
	// loss = mean([2, 2, 2]) = 2.0
	expected := 2.0

	if !feq(loss.Value.Data[0], expected) {
		t.Errorf("HingeLoss with all wrong predictions failed: expected %v, got %v",
			expected, loss.Value.Data[0])
	}
}

func TestHingeLossMatrixBatch(t *testing.T) {
	e := NewEngine()

	// Батч 2x2
	pred := e.RequireGrad(newTensorTest([]float64{
		0.5, -0.5,
		1.5, -1.5,
	}, 2, 2))

	target := newTensorTest([]float64{
		1.0, -1.0,
		1.0, -1.0,
	}, 2, 2)

	loss := e.HingeLoss(pred, target)

	// margins = [1-0.5, 1-0.5, 1-1.5, 1-1.5] = [0.5, 0.5, -0.5, -0.5]
	// max(0, margins) = [0.5, 0.5, 0, 0]
	// mean = 1.0 / 4 = 0.25
	expected := 0.25

	if !feq(loss.Value.Data[0], expected) {
		t.Errorf("HingeLoss batch failed: expected %v, got %v",
			expected, loss.Value.Data[0])
	}
}

func TestHingeLossBinaryClassification(t *testing.T) {
	e := NewEngine()

	// Типичный случай бинарной классификации
	pred := e.RequireGrad(newTensorTest([]float64{0.8, -0.3, 1.2, -0.9}, 4))
	target := newTensorTest([]float64{1.0, -1.0, 1.0, -1.0}, 4)

	loss := e.HingeLoss(pred, target)
	e.Backward(loss)

	// Проверяем что loss и градиенты конечны
	if math.IsNaN(loss.Value.Data[0]) || math.IsInf(loss.Value.Data[0], 0) {
		t.Error("HingeLoss produced NaN or Inf")
	}

	for i, g := range pred.Grad.Data {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Errorf("Gradient[%d] is NaN or Inf: %v", i, g)
		}
	}
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

func TestMSELossShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MSELoss should panic on shape mismatch")
		}
	}()

	e := NewEngine()
	pred := e.RequireGrad(newTensorTest([]float64{1.0, 2.0}, 2))
	target := newTensorTest([]float64{1.0, 2.0, 3.0}, 3)
	e.MSELoss(pred, target)
}

func TestCrossEntropyLossShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("CrossEntropyLoss should panic on shape mismatch")
		}
	}()

	e := NewEngine()
	logits := e.RequireGrad(newTensorTest([]float64{1.0, 2.0, 3.0}, 1, 3))
	target := newTensorTest([]float64{1.0, 2.0, 3.0, 4.0}, 1, 4)
	e.CrossEntropyLoss(logits, target)
}

func TestCrossEntropyLossNot2D(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("CrossEntropyLoss should panic on 1D tensors")
		}
	}()

	e := NewEngine()
	logits := e.RequireGrad(newTensorTest([]float64{1.0, 2.0, 3.0}, 3))
	target := newTensorTest([]float64{1.0, 0.0, 0.0}, 3)
	e.CrossEntropyLoss(logits, target)
}

func TestHingeLossShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("HingeLoss should panic on shape mismatch")
		}
	}()

	e := NewEngine()
	pred := e.RequireGrad(newTensorTest([]float64{1.0, 2.0}, 2))
	target := newTensorTest([]float64{1.0}, 1)
	e.HingeLoss(pred, target)
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkMSELoss(b *testing.B) {
	e := NewEngine()
	pred := e.RequireGrad(tensor.Randn([]int{100, 10}, 42))
	target := tensor.Randn([]int{100, 10}, 123)

	b.ResetTimer()
	for range b.N {
		e.MSELoss(pred, target)
	}
}

func BenchmarkCrossEntropyLoss(b *testing.B) {
	e := NewEngine()
	logits := e.RequireGrad(tensor.Randn([]int{100, 10}, 42))
	target := tensor.Zeros(100, 10)
	// Установим one-hot метки
	for i := range 100 {
		target.Data[i*10+i%10] = 1.0
	}

	b.ResetTimer()
	for range b.N {
		e.CrossEntropyLoss(logits, target)
	}
}

func BenchmarkHingeLoss(b *testing.B) {
	e := NewEngine()
	pred := e.RequireGrad(tensor.Randn([]int{100, 10}, 42))
	target := tensor.Ones(100, 10)

	b.ResetTimer()
	for range b.N {
		e.HingeLoss(pred, target)
	}
}
