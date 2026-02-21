package dataloader

import (
	"testing"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
)

// TestSimpleDatasetCreation проверяет создание SimpleDataset.
func TestSimpleDatasetCreation(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 2}, 123)

	dataset := NewSimpleDataset(features, targets)

	if dataset.Len() != 10 {
		t.Errorf("Expected dataset length 10, got %d", dataset.Len())
	}

	if dataset.numSamples != 10 {
		t.Errorf("Expected numSamples 10, got %d", dataset.numSamples)
	}
}

// TestSimpleDatasetGet проверяет получение примеров по индексу.
func TestSimpleDatasetGet(t *testing.T) {
	// Создаем простой датасет с известными значениями
	featureData := make([]float64, 20) // 10 примеров x 2 признака
	for i := range featureData {
		featureData[i] = float64(i)
	}

	targetData := make([]float64, 10) // 10 примеров x 1 значение
	for i := range targetData {
		targetData[i] = float64(i * 10)
	}

	features := &tensor.Tensor{
		Data:    featureData,
		Shape:   []int{10, 2},
		Strides: []int{2, 1},
	}

	targets := &tensor.Tensor{
		Data:    targetData,
		Shape:   []int{10, 1},
		Strides: []int{1, 1},
	}

	dataset := NewSimpleDataset(features, targets)

	// Проверяем первый пример
	f0, t0 := dataset.Get(0)
	if len(f0.Shape) != 1 || f0.Shape[0] != 2 {
		t.Errorf("Expected feature shape [2], got %v", f0.Shape)
	}
	if f0.Data[0] != 0.0 || f0.Data[1] != 1.0 {
		t.Errorf("Expected feature [0, 1], got %v", f0.Data)
	}
	if len(t0.Shape) != 1 || t0.Shape[0] != 1 {
		t.Errorf("Expected target shape [1], got %v", t0.Shape)
	}
	if t0.Data[0] != 0.0 {
		t.Errorf("Expected target [0], got %v", t0.Data)
	}

	// Проверяем пятый пример (индекс 5)
	f5, t5 := dataset.Get(5)
	if f5.Data[0] != 10.0 || f5.Data[1] != 11.0 {
		t.Errorf("Expected feature [10, 11], got %v", f5.Data)
	}
	if t5.Data[0] != 50.0 {
		t.Errorf("Expected target [50], got %v", t5.Data)
	}
}

// TestSimpleDataset1D проверяет работу с 1D тензорами.
func TestSimpleDataset1D(t *testing.T) {
	features := &tensor.Tensor{
		Data:    []float64{1, 2, 3, 4, 5},
		Shape:   []int{5},
		Strides: []int{1},
	}

	targets := &tensor.Tensor{
		Data:    []float64{10, 20, 30, 40, 50},
		Shape:   []int{5},
		Strides: []int{1},
	}

	dataset := NewSimpleDataset(features, targets)

	if dataset.Len() != 5 {
		t.Errorf("Expected length 5, got %d", dataset.Len())
	}

	f2, t2 := dataset.Get(2)
	if f2.Data[0] != 3.0 {
		t.Errorf("Expected feature 3, got %v", f2.Data)
	}
	if t2.Data[0] != 30.0 {
		t.Errorf("Expected target 30, got %v", t2.Data)
	}
}

// TestSimpleDatasetMultiDimensional проверяет работу с многомерными тензорами.
func TestSimpleDatasetMultiDimensional(t *testing.T) {
	// 3D тензор: [5, 2, 3] - 5 примеров, каждый 2x3
	featureData := make([]float64, 30)
	for i := range featureData {
		featureData[i] = float64(i)
	}

	features := &tensor.Tensor{
		Data:    featureData,
		Shape:   []int{5, 2, 3},
		Strides: []int{6, 3, 1},
	}

	targets := tensor.Randn([]int{5, 1}, 123)

	dataset := NewSimpleDataset(features, targets)

	// Получаем второй пример (индекс 1)
	f1, _ := dataset.Get(1)

	// Проверяем размерность
	if len(f1.Shape) != 2 || f1.Shape[0] != 2 || f1.Shape[1] != 3 {
		t.Errorf("Expected feature shape [2, 3], got %v", f1.Shape)
	}

	// Проверяем данные (должны быть элементы с индексов 6-11)
	expected := []float64{6, 7, 8, 9, 10, 11}
	for i, v := range expected {
		if f1.Data[i] != v {
			t.Errorf("Expected f1.Data[%d] = %v, got %v", i, v, f1.Data[i])
		}
	}
}

// TestSimpleDatasetOutOfBounds проверяет обработку выхода за границы.
func TestSimpleDatasetOutOfBounds(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 2}, 123)
	dataset := NewSimpleDataset(features, targets)

	// Негативный индекс
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for negative index")
		}
	}()
	dataset.Get(-1)
}

// TestSimpleDatasetOutOfBoundsUpper проверяет индекс больше размера.
func TestSimpleDatasetOutOfBoundsUpper(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 2}, 123)
	dataset := NewSimpleDataset(features, targets)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for index >= length")
		}
	}()
	dataset.Get(10)
}

// TestSimpleDatasetMismatchedSamples проверяет несовпадение количества примеров.
func TestSimpleDatasetMismatchedSamples(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{8, 2}, 123) // Разное количество примеров

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for mismatched sample counts")
		}
	}()

	NewSimpleDataset(features, targets)
}

// TestSimpleDatasetEmptyTensor проверяет пустые тензоры.
func TestSimpleDatasetEmptyTensor(t *testing.T) {
	features := &tensor.Tensor{
		Data:    []float64{},
		Shape:   []int{},
		Strides: []int{},
	}
	targets := &tensor.Tensor{
		Data:    []float64{},
		Shape:   []int{},
		Strides: []int{},
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for empty tensors")
		}
	}()

	NewSimpleDataset(features, targets)
}

// BenchmarkSimpleDatasetGet бенчмарк для операции Get.
func BenchmarkSimpleDatasetGet(b *testing.B) {
	features := tensor.Randn([]int{1000, 784}, 42)
	targets := tensor.Randn([]int{1000, 10}, 123)
	dataset := NewSimpleDataset(features, targets)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dataset.Get(i % 1000)
	}
}
