package optimizers_test

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// TestMomentumBasicStep проверяет базовое обновление параметров
func TestMomentumBasicStep(t *testing.T) {
	// Создаем параметр с начальным значением и градиентом
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

	// Создаем оптимизатор
	optimizer := optimizers.NewMomentum(0.01, 0.9)

	// Первый шаг
	optimizer.Step([]*graph.Node{param})

	// На первом шаге импульс v = 0 * 0.9 + 0.01 * grad
	// v = 0.001, 0.002, 0.003
	// param -= v
	expected := []float64{0.999, 1.998, 2.997}
	for i, exp := range expected {
		if math.Abs(param.Value.Data[i]-exp) > 1e-6 {
			t.Fatalf("First step mismatch at index %d: got %v want %v", i, param.Value.Data[i], exp)
		}
	}

	// Второй шаг - проверяем, что импульс накапливается
	optimizer.Step([]*graph.Node{param})

	// На втором шаге v = 0.001 * 0.9 + 0.01 * grad
	// v = 0.0009 + 0.001 = 0.0019 и т.д.
	// param -= v
	// Проверяем, что параметры изменились
	if param.Value.Data[0] >= 0.999 {
		t.Fatalf("Second step: parameter should continue decreasing, got %v", param.Value.Data[0])
	}
}

// TestMomentumZeroGrad проверяет обнуление градиентов
func TestMomentumZeroGrad(t *testing.T) {
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

	optimizer := optimizers.NewMomentum(0.01, 0.9)
	optimizer.ZeroGrad([]*graph.Node{param})

	if param.Grad.Data[0] != 0.0 {
		t.Fatalf("ZeroGrad failed: got %v want 0.0", param.Grad.Data[0])
	}
}

// TestMomentumMultipleParameters проверяет работу с несколькими параметрами
func TestMomentumMultipleParameters(t *testing.T) {
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
			Data:    []float64{2.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.2},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewMomentum(0.01, 0.9)
	optimizer.Step([]*graph.Node{param1, param2})

	// Проверяем, что оба параметра обновлены
	if param1.Value.Data[0] >= 1.0 || param2.Value.Data[0] >= 2.0 {
		t.Fatalf("Multiple parameters update failed")
	}
}

// TestMomentumNilGradient проверяет обработку nil градиентов
func TestMomentumNilGradient(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: nil,
	}

	optimizer := optimizers.NewMomentum(0.01, 0.9)

	// Не должно быть паники
	optimizer.Step([]*graph.Node{param})

	// Параметр должен остаться неизменным
	if param.Value.Data[0] != 1.0 {
		t.Fatalf("Parameter with nil gradient changed: got %v", param.Value.Data[0])
	}
}

// TestMomentumConvergence проверяет сходимость к минимуму
func TestMomentumConvergence(t *testing.T) {
	// Симуляция простой квадратичной функции: f(x) = x^2
	// Градиент: df/dx = 2x
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{10.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewMomentum(0.01, 0.9)

	// Несколько итераций оптимизации
	for i := 0; i < 100; i++ {
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
		t.Fatalf("Momentum convergence failed: parameter too far from zero: %v", param.Value.Data[0])
	}
}

// BenchmarkMomentumStep бенчмарк для операции Step
func BenchmarkMomentumStep(b *testing.B) {
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

	optimizer := optimizers.NewMomentum(0.001, 0.9)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.Step([]*graph.Node{param})
	}
}

// BenchmarkMomentumZeroGrad бенчмарк для операции ZeroGrad w
func BenchmarkMomentumZeroGrad(b *testing.B) {
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

	optimizer := optimizers.NewMomentum(0.001, 0.9)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.ZeroGrad([]*graph.Node{param})
	}
}
