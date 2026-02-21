package optimizers_test

import (
	"math"
	"testing"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
	"github.com/Hirogava/Go-NN-Learn/internal/optimizers"
)

// TestAdamBasicStep проверяет базовое обновление параметров и bias correction.
func TestAdamBasicStep(t *testing.T) {
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

	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	optimizer.Step([]*graph.Node{param})

	expected := []float64{0.99, 1.99, 2.99}
	for i, exp := range expected {
		if math.Abs(param.Value.Data[i]-exp) > 1e-6 {
			t.Fatalf("Adam first step mismatch at %d: got %v want %v", i, param.Value.Data[i], exp)
		}
	}
}

// TestAdamMultipleSteps проверяет накопление моментов.
func TestAdamMultipleSteps(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: &tensor.Tensor{
			Data:    []float64{0.5},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)

	val0 := param.Value.Data[0]
	optimizer.Step([]*graph.Node{param})
	val1 := param.Value.Data[0]

	// Изменяем градиент, чтобы проверить адаптивность
	param.Grad.Data[0] = 0.1
	optimizer.Step([]*graph.Node{param})
	val2 := param.Value.Data[0]

	if !(val0 > val1 && val1 > val2) {
		t.Fatalf("Adam multiple steps failed: %v -> %v -> %v", val0, val1, val2)
	}
}

// TestAdamZeroGrad проверяет обнуление градиентов.
func TestAdamZeroGrad(t *testing.T) {
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

	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	optimizer.ZeroGrad([]*graph.Node{param})

	if param.Grad.Data[0] != 0.0 {
		t.Fatalf("Adam ZeroGrad failed: got %v", param.Grad.Data[0])
	}
}

// TestAdamNilGradient проверяет обработку отсутствующих градиентов.
func TestAdamNilGradient(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
		Grad: nil,
	}

	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	optimizer.Step([]*graph.Node{param})

	if param.Value.Data[0] != 1.0 {
		t.Fatalf("Parameter with nil grad changed: got %v", param.Value.Data[0])
	}
}

// TestAdamConvergence проверяет, что оптимизатор сходится к минимуму простой функции.
func TestAdamConvergence(t *testing.T) {
	param := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{5.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	optimizer := optimizers.NewAdam(0.1, 0.9, 0.999, 1e-8)

	for i := 0; i < 200; i++ {
		param.Grad = &tensor.Tensor{
			Data:    []float64{2 * param.Value.Data[0]},
			Shape:   []int{1},
			Strides: []int{1},
		}
		optimizer.Step([]*graph.Node{param})
	}

	if math.Abs(param.Value.Data[0]) > 0.1 {
		t.Fatalf("Adam convergence failed: parameter too far from zero %v", param.Value.Data[0])
	}
}

// TestAdamWeightDecay проверяет применение WeightDecay в Adam
func TestAdamWeightDecay(t *testing.T) {
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
	optimizer := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8, optimizers.WithAdamWeightDecay(0.1))

	initialValue := make([]float64, len(param.Value.Data))
	copy(initialValue, param.Value.Data)

	optimizer.Step([]*graph.Node{param})

	// С WeightDecay параметры должны уменьшаться даже при нулевом градиенте
	for i := range param.Value.Data {
		if param.Value.Data[i] >= initialValue[i] {
			t.Fatalf("WeightDecay not applied: param[%d] = %v, expected < %v", i, param.Value.Data[i], initialValue[i])
		}
	}
}

// TestAdamWeightDecayComparison сравнивает оптимизаторы с и без WeightDecay
func TestAdamWeightDecayComparison(t *testing.T) {
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

	optimizerWithoutDecay := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	optimizerWithDecay := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8, optimizers.WithAdamWeightDecay(0.01))

	optimizerWithoutDecay.Step([]*graph.Node{param1})
	optimizerWithDecay.Step([]*graph.Node{param2})

	// С WeightDecay параметр должен уменьшиться больше
	if param2.Value.Data[0] >= param1.Value.Data[0] {
		t.Fatalf("WeightDecay should decrease parameter more: with decay=%v, without=%v", param2.Value.Data[0], param1.Value.Data[0])
	}
}

// TestAdamWeightDecayZero проверяет, что без WeightDecay поведение не меняется
func TestAdamWeightDecayZero(t *testing.T) {
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

	optimizer1 := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8)
	optimizer2 := optimizers.NewAdam(0.01, 0.9, 0.999, 1e-8, optimizers.WithAdamWeightDecay(0.0))

	optimizer1.Step([]*graph.Node{param1})
	optimizer2.Step([]*graph.Node{param2})

	// Результаты должны быть одинаковыми
	for i := range param1.Value.Data {
		if math.Abs(param1.Value.Data[i]-param2.Value.Data[i]) > 1e-10 {
			t.Fatalf("WeightDecay=0 should behave same as no WeightDecay: got %v vs %v", param1.Value.Data[i], param2.Value.Data[i])
		}
	}
}
