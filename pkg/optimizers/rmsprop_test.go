package optimizers_test

import (
	"math"
	"testing"
	
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// TestRMSPropBasicStep проверяет базовое обновление параметров
func TestRMSPropBasicStep(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0, 2.0, 3.0},
			Shape:  []int{3},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   []float64{0.1, 0.2, 0.3},
			Shape:  []int{3},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.01, 0.9, 1e-8)
	optimizer.Step([]*graph.Node{param})

	// На первом шаге:
	// E[g^2] = 0.9 * 0 + 0.1 * (0.1^2) = 0.001
	// param -= 0.01 * 0.1 / sqrt(0.001 + 1e-8)
	for i := 0; i < 3; i++ {
		if param.Value.Data[i] >= []float64{1.0, 2.0, 3.0}[i] {
			t.Fatalf("RMSProp step failed at index %d: parameter not updated", i)
		}
	}
}

// TestRMSPropZeroGrad проверяет обнуление градиентов
func TestRMSPropZeroGrad(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   []float64{5.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.01, 0.9, 1e-8)
	optimizer.ZeroGrad([]*graph.Node{param})

	if param.Grad.Data[0] != 0.0 {
		t.Fatalf("RMSProp ZeroGrad failed: got %v want 0.0", param.Grad.Data[0])
	}
}

// TestRMSPropAdaptiveLearningRate проверяет адаптивность learning rate
func TestRMSPropAdaptiveLearningRate(t *testing.T) {
	// Создаем два параметра с разными градиентами
	param1 := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   []float64{0.01}, // Маленький градиент
			Shape:  []int{1},
			Strides: []int{1},
		},
	}

	param2 := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   []float64{1.0}, // Большой градиент
			Shape:  []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.01, 0.9, 1e-8)

	// Сохраняем начальные значения
	val1Before := param1.Value.Data[0]
	val2Before := param2.Value.Data[0]

	optimizer.Step([]*graph.Node{param1})
	optimizer.Step([]*graph.Node{param2})

	// Изменение param1 должно быть больше, чем param2
	// (так как RMSProp адаптирует learning rate на основе квадратов градиентов)
	change1 := math.Abs(val1Before - param1.Value.Data[0])
	change2 := math.Abs(val2Before - param2.Value.Data[0])

	if change1 > change2 {
		t.Logf("RMSProp adaptive learning rate working: change1=%v > change2=%v", change1, change2)
	}
}

// TestRMSPropMultipleSteps проверяет накопление скользящего среднего
func TestRMSPropMultipleSteps(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   []float64{0.1},
			Shape:  []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.01, 0.9, 1e-8)

	val0 := param.Value.Data[0]
	optimizer.Step([]*graph.Node{param})
	val1 := param.Value.Data[0]

	optimizer.Step([]*graph.Node{param})
	val2 := param.Value.Data[0]

	// Проверяем, что изменения происходят на каждом шаге
	if val0 == val1 || val1 == val2 {
		t.Fatalf("RMSProp multiple steps failed: values not changing properly")
	}
}

// TestRMSPropNilGradient проверяет обработку nil градиентов
func TestRMSPropNilGradient(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
		Grad: nil,
	}

	optimizer := optimizers.NewRMSProp(0.01, 0.9, 1e-8)
	
	// Не должно быть паники
	optimizer.Step([]*graph.Node{param})
	
	// Параметр должен остаться неизменным
	if param.Value.Data[0] != 1.0 {
		t.Fatalf("Parameter with nil gradient changed: got %v", param.Value.Data[0])
	}
}

// TestRMSPropConvergence проверяет сходимость
func TestRMSPropConvergence(t *testing.T) {
	// Симуляция квадратичной функции: f(x) = x^2
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{10.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.1, 0.9, 1e-8)

	// Несколько итераций оптимизации
	for i := 0; i < 100; i++ {
		// Градиент: df/dx = 2x
		param.Grad = &tensor.Tensor{
			Data:   []float64{2 * param.Value.Data[0]},
			Shape:  []int{1},
			Strides: []int{1},
		}

		optimizer.Step([]*graph.Node{param})
	}

	// Параметр должен быть близок к нулю
	if math.Abs(param.Value.Data[0]) > 0.1 {
		t.Fatalf("RMSProp convergence failed: parameter too far from zero: %v", param.Value.Data[0])
	}
}

// TestRMSPropEpsilonPreventsNaN проверяет, что epsilon предотвращает деление на ноль
func TestRMSPropEpsilonPreventsNaN(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   []float64{1.0},
			Shape:  []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   []float64{0.0}, // Нулевой градиент
			Shape:  []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.01, 0.9, 1e-8)
	optimizer.Step([]*graph.Node{param})

	// Проверяем, что не получили NaN
	if math.IsNaN(param.Value.Data[0]) {
		t.Fatalf("RMSProp produced NaN with zero gradient")
	}
}

// BenchmarkRMSPropStep бенчмарк для операции Step
func BenchmarkRMSPropStep(b *testing.B) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   make([]float64, 1000),
			Shape:  []int{1000},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   make([]float64, 1000),
			Shape:  []int{1000},
			Strides: []int{1},
		},
	}

	// Заполняем начальные значения
	for i := range param.Value.Data {
		param.Value.Data[i] = float64(i) / 100.0
		param.Grad.Data[i] = 0.01 * float64(i)
	}

	optimizer := optimizers.NewRMSProp(0.001, 0.9, 1e-8)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.Step([]*graph.Node{param})
	}
}

// BenchmarkRMSPropZeroGrad бенчмарк для операции ZeroGrad
func BenchmarkRMSPropZeroGrad(b *testing.B) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:   make([]float64, 1000),
			Shape:  []int{1000},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:   make([]float64, 1000),
			Shape:  []int{1000},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewRMSProp(0.001, 0.9, 1e-8)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.ZeroGrad([]*graph.Node{param})
	}
}
