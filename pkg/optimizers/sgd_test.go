package optimizers_test

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// TestSGDBasicStep проверяет базовое обновление параметров
func TestSGDBasicStep(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0},
			Shape:   []int{3},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.1, 0.2, 0.3},
			Shape:   []int{3},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewSGD(0.01)
	optimizer.Step([]*graph.Node{param})

	// param -= lr * grad
	// param = [1.0, 2.0, 3.0] - 0.01 * [0.1, 0.2, 0.3] = [0.999, 1.998, 2.997]
	expected := []float64{0.999, 1.998, 2.997}
	for i, exp := range expected {
		if math.Abs(param.Value.Data[i]-exp) > 1e-6 {
			t.Fatalf("SGD step mismatch at index %d: got %v want %v", i, param.Value.Data[i], exp)
		}
	}
}

// TestSGDZeroGrad проверяет обнуление градиентов
func TestSGDZeroGrad(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{5.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewSGD(0.01)
	optimizer.ZeroGrad([]*graph.Node{param})

	if param.Grad.Data[0] != 0.0 {
		t.Fatalf("SGD ZeroGrad failed: got %v want 0.0", param.Grad.Data[0])
	}
}

// TestSGDNilGradient проверяет обработку nil градиентов
func TestSGDNilGradient(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: nil,
	}

	optimizer := optimizers.NewSGD(0.01)

	// Не должно быть паники
	optimizer.Step([]*graph.Node{param})

	// Параметр должен остаться неизменным
	if param.Value.Data[0] != 1.0 {
		t.Fatalf("Parameter with nil gradient changed: got %v", param.Value.Data[0])
	}
}

// TestSGDConvergence проверяет сходимость к минимуму
func TestSGDConvergence(t *testing.T) {
	// Симуляция простой квадратичной функции: f(x) = x^2
	// Градиент: df/dx = 2x
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{10.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewSGD(0.01)

	// Несколько итераций оптимизации
	for i := 0; i < 500; i++ {
		// Вычисляем градиент: df/dx = 2x
		param.Grad = &tensor.Tensor{
			Data:    []float64{2 * param.Value.Data[0]},
			Shape:   []int{1},
			Strides: []int{1},
		}

		optimizer.Step([]*graph.Node{param})
	}

	// Параметр должен быть близок к нулю (минимуму)
	if math.Abs(param.Value.Data[0]) > 0.1 {
		t.Fatalf("SGD convergence failed: parameter too far from zero: %v", param.Value.Data[0])
	}
}

// TestSGDWeightDecay проверяет применение WeightDecay в SGD
func TestSGDWeightDecay(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0},
			Shape:   []int{2},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.0, 0.0}, // Нулевой градиент
			Shape:   []int{2},
			Strides: []int{1},
		},
	}

	// Создаем оптимизатор с WeightDecay
	optimizer := optimizers.NewSGD(0.01, optimizers.WithSGDWeightDecay(0.1))

	initialValue := make([]float64, len(param.Value.Data))
	copy(initialValue, param.Value.Data)

	optimizer.Step([]*graph.Node{param})

	// С WeightDecay параметры должны уменьшаться даже при нулевом градиенте
	// grad_with_decay = 0 + 0.1 * weight = 0.1 * weight
	// param -= lr * grad_with_decay = param - 0.01 * 0.1 * weight
	for i := range param.Value.Data {
		if param.Value.Data[i] >= initialValue[i] {
			t.Fatalf("WeightDecay not applied: param[%d] = %v, expected < %v", i, param.Value.Data[i], initialValue[i])
		}
	}
}

// TestSGDWeightDecayComparison сравнивает оптимизаторы с и без WeightDecay
func TestSGDWeightDecayComparison(t *testing.T) {
	param1 := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.1},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	param2 := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.1},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizerWithoutDecay := optimizers.NewSGD(0.01)
	optimizerWithDecay := optimizers.NewSGD(0.01, optimizers.WithSGDWeightDecay(0.01))

	optimizerWithoutDecay.Step([]*graph.Node{param1})
	optimizerWithDecay.Step([]*graph.Node{param2})

	// С WeightDecay параметр должен уменьшиться больше
	if param2.Value.Data[0] >= param1.Value.Data[0] {
		t.Fatalf("WeightDecay should decrease parameter more: with decay=%v, without=%v", param2.Value.Data[0], param1.Value.Data[0])
	}
}

// TestSGDWeightDecayZero проверяет, что без WeightDecay поведение не меняется
func TestSGDWeightDecayZero(t *testing.T) {
	param1 := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0},
			Shape:   []int{3},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.1, 0.2, 0.3},
			Shape:   []int{3},
			Strides: []int{1},
		},
	}

	param2 := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0},
			Shape:   []int{3},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.1, 0.2, 0.3},
			Shape:   []int{3},
			Strides: []int{1},
		},
	}

	optimizer1 := optimizers.NewSGD(0.01)
	optimizer2 := optimizers.NewSGD(0.01, optimizers.WithSGDWeightDecay(0.0))

	optimizer1.Step([]*graph.Node{param1})
	optimizer2.Step([]*graph.Node{param2})

	// Результаты должны быть одинаковыми
	for i := range param1.Value.Data {
		if math.Abs(param1.Value.Data[i]-param2.Value.Data[i]) > 1e-10 {
			t.Fatalf("WeightDecay=0 should behave same as no WeightDecay: got %v vs %v", param1.Value.Data[i], param2.Value.Data[i])
		}
	}
}

// BenchmarkSGDStep бенчмарк для операции Step
func BenchmarkSGDStep(b *testing.B) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    make([]float64, 1000),
			Shape:   []int{1000},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    make([]float64, 1000),
			Shape:   []int{1000},
			Strides: []int{1},
		},
	}

	// Заполняем начальные значения
	for i := range param.Value.Data {
		param.Value.Data[i] = float64(i) / 100.0
		param.Grad.Data[i] = 0.01 * float64(i)
	}

	optimizer := optimizers.NewSGD(0.001)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.Step([]*graph.Node{param})
	}
}

// BenchmarkSGDZeroGrad бенчмарк для операции ZeroGrad
func BenchmarkSGDZeroGrad(b *testing.B) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    make([]float64, 1000),
			Shape:   []int{1000},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    make([]float64, 1000),
			Shape:   []int{1000},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewSGD(0.001)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.ZeroGrad([]*graph.Node{param})
	}
}

