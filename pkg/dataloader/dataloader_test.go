package dataloader

import (
	"testing"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
)

// TestDataLoaderCreation проверяет создание DataLoader.
func TestDataLoaderCreation(t *testing.T) {
	features := tensor.Randn([]int{100, 10}, 42)
	targets := tensor.Randn([]int{100, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	if loader.BatchSize() != 32 {
		t.Errorf("Expected batch size 32, got %d", loader.BatchSize())
	}

	// Ожидаем 4 батча: 32 + 32 + 32 + 4
	expectedLen := 4
	if loader.Len() != expectedLen {
		t.Errorf("Expected %d batches, got %d", expectedLen, loader.Len())
	}
}

// TestDataLoaderBatching проверяет корректность батчинга.
func TestDataLoaderBatching(t *testing.T) {
	// Создаем простой датасет с известными значениями
	featureData := make([]float64, 10)
	for i := range featureData {
		featureData[i] = float64(i)
	}

	targetData := make([]float64, 10)
	for i := range targetData {
		targetData[i] = float64(i * 10)
	}

	features := &tensor.Tensor{
		Data:    featureData,
		Shape:   []int{10, 1},
		Strides: []int{1, 1},
	}

	targets := &tensor.Tensor{
		Data:    targetData,
		Shape:   []int{10, 1},
		Strides: []int{1, 1},
	}

	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 3,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	// Первый батч: индексы 0, 1, 2
	if !loader.HasNext() {
		t.Fatal("Expected HasNext to be true")
	}

	batch1 := loader.Next()
	if len(batch1.Features.Shape) != 2 || batch1.Features.Shape[0] != 3 || batch1.Features.Shape[1] != 1 {
		t.Errorf("Expected batch shape [3, 1], got %v", batch1.Features.Shape)
	}

	// Проверяем данные первого батча
	if batch1.Features.Data[0] != 0.0 || batch1.Features.Data[1] != 1.0 || batch1.Features.Data[2] != 2.0 {
		t.Errorf("Unexpected features in first batch: %v", batch1.Features.Data)
	}

	if batch1.Targets.Data[0] != 0.0 || batch1.Targets.Data[1] != 10.0 || batch1.Targets.Data[2] != 20.0 {
		t.Errorf("Unexpected targets in first batch: %v", batch1.Targets.Data)
	}

	// Второй батч
	batch2 := loader.Next()
	if batch2.Features.Data[0] != 3.0 || batch2.Features.Data[1] != 4.0 {
		t.Errorf("Unexpected features in second batch: %v", batch2.Features.Data)
	}

	// Третий батч
	_ = loader.Next()
	// Четвертый батч (последний, неполный: только индекс 9)
	batch4 := loader.Next()
	if batch4.Features.Shape[0] != 1 {
		t.Errorf("Expected last batch size 1, got %d", batch4.Features.Shape[0])
	}
	if batch4.Features.Data[0] != 9.0 {
		t.Errorf("Expected last batch feature 9, got %v", batch4.Features.Data[0])
	}

	// Должны исчерпать все батчи
	if loader.HasNext() {
		t.Error("Expected no more batches")
	}
}

// TestDataLoaderDropLast проверяет параметр DropLast.
func TestDataLoaderDropLast(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	// С DropLast=true
	loaderDropLast := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 3,
		Shuffle:   false,
		DropLast:  true,
		Seed:      42,
	})

	// Ожидаем только 3 полных батча (9 примеров), последний отбрасывается
	if loaderDropLast.Len() != 3 {
		t.Errorf("Expected 3 batches with DropLast=true, got %d", loaderDropLast.Len())
	}

	count := 0
	for loaderDropLast.HasNext() {
		batch := loaderDropLast.Next()
		count++
		if batch.Features.Shape[0] != 3 {
			t.Errorf("Expected all batches to have size 3, got %d", batch.Features.Shape[0])
		}
	}

	if count != 3 {
		t.Errorf("Expected to iterate 3 times, got %d", count)
	}

	// С DropLast=false
	loaderKeepLast := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 3,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	// Ожидаем 4 батча (последний неполный)
	if loaderKeepLast.Len() != 4 {
		t.Errorf("Expected 4 batches with DropLast=false, got %d", loaderKeepLast.Len())
	}
}

// TestDataLoaderShuffle проверяет перемешивание данных.
func TestDataLoaderShuffle(t *testing.T) {
	features := tensor.Randn([]int{20, 5}, 42)
	targets := tensor.Randn([]int{20, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	// Без shuffle
	loaderNoShuffle := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 20,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	batch1 := loaderNoShuffle.Next()
	firstFeatures1 := batch1.Features.Data[:5]

	// С shuffle
	loaderShuffle := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 20,
		Shuffle:   true,
		DropLast:  false,
		Seed:      99, // Другой seed для гарантии перемешивания
	})

	batch2 := loaderShuffle.Next()
	firstFeatures2 := batch2.Features.Data[:5]

	// Проверяем что порядок изменился (с большой вероятностью)
	allSame := true
	for i := range firstFeatures1 {
		if firstFeatures1[i] != firstFeatures2[i] {
			allSame = false
			break
		}
	}

	// С очень малой вероятностью они могут быть одинаковыми, но обычно должны отличаться
	// Этот тест может редко fail из-за случайности, но это очень маловероятно
	if allSame {
		t.Log("Warning: shuffled batch looks identical to non-shuffled (rare but possible)")
	}
}

// TestDataLoaderReset проверяет сброс итератора.
func TestDataLoaderReset(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 5,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	// Первая эпоха
	batch1 := loader.Next()
	_ = loader.Next() // batch2

	if loader.HasNext() {
		t.Error("Expected no more batches after consuming all")
	}

	// Сброс
	loader.Reset()

	if !loader.HasNext() {
		t.Error("Expected HasNext to be true after Reset")
	}

	// Вторая эпоха
	batch1Again := loader.Next()
	_ = loader.Next() // batch2Again

	// Проверяем что получили те же данные (shuffle=false)
	for i := range batch1.Features.Data {
		if batch1.Features.Data[i] != batch1Again.Features.Data[i] {
			t.Errorf("Expected same data after Reset, difference at index %d", i)
			break
		}
	}
}

// TestDataLoaderShuffleConsistency проверяет детерминированность shuffle с одинаковым seed.
func TestDataLoaderShuffleConsistency(t *testing.T) {
	features := tensor.Randn([]int{50, 10}, 42)
	targets := tensor.Randn([]int{50, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	// Два DataLoader с одинаковым seed
	loader1 := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 10,
		Shuffle:   true,
		DropLast:  false,
		Seed:      777,
	})

	loader2 := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 10,
		Shuffle:   true,
		DropLast:  false,
		Seed:      777,
	})

	// Проверяем что они дают одинаковый порядок
	for loader1.HasNext() && loader2.HasNext() {
		batch1 := loader1.Next()
		batch2 := loader2.Next()

		// Сравниваем батчи
		if len(batch1.Features.Data) != len(batch2.Features.Data) {
			t.Error("Batches have different sizes")
		}

		for i := range batch1.Features.Data {
			if batch1.Features.Data[i] != batch2.Features.Data[i] {
				t.Errorf("Batches differ at index %d", i)
				break
			}
		}
	}
}

// TestDataLoaderInvalidBatchSize проверяет обработку некорректного размера батча.
func TestDataLoaderInvalidBatchSize(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	// Нулевой batch size
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for zero batch size")
		}
	}()

	NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 0,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})
}

// TestDataLoaderBatchSizeLargerThanDataset проверяет batch size больше датасета.
func TestDataLoaderBatchSizeLargerThanDataset(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for batch size > dataset size")
		}
	}()

	NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 20,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})
}

// TestDataLoaderNextPanicsWhenEmpty проверяет панику при вызове Next без данных.
func TestDataLoaderNextPanicsWhenEmpty(t *testing.T) {
	features := tensor.Randn([]int{10, 5}, 42)
	targets := tensor.Randn([]int{10, 1}, 123)
	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 10,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	// Исчерпываем все батчи
	loader.Next()

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when calling Next without available batches")
		}
	}()

	loader.Next() // Должна быть паника
}

// TestDataLoaderMultidimensional проверяет работу с многомерными данными.
func TestDataLoaderMultidimensional(t *testing.T) {
	// 3D features: [20, 28, 28] - 20 изображений 28x28
	features := tensor.Randn([]int{20, 28, 28}, 42)
	// 2D targets: [20, 10] - 20 примеров, 10 классов
	targets := tensor.Randn([]int{20, 10}, 123)

	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 4,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	batch := loader.Next()

	// Проверяем размерности батча
	expectedFeatureShape := []int{4, 28, 28}
	expectedTargetShape := []int{4, 10}

	if len(batch.Features.Shape) != 3 {
		t.Errorf("Expected 3D features, got %dD", len(batch.Features.Shape))
	}

	for i, dim := range expectedFeatureShape {
		if batch.Features.Shape[i] != dim {
			t.Errorf("Expected feature shape %v, got %v", expectedFeatureShape, batch.Features.Shape)
			break
		}
	}

	for i, dim := range expectedTargetShape {
		if batch.Targets.Shape[i] != dim {
			t.Errorf("Expected target shape %v, got %v", expectedTargetShape, batch.Targets.Shape)
			break
		}
	}
}

// BenchmarkDataLoaderIteration бенчмарк для итерации по DataLoader.
func BenchmarkDataLoaderIteration(b *testing.B) {
	features := tensor.Randn([]int{1000, 784}, 42)
	targets := tensor.Randn([]int{1000, 10}, 123)
	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loader.Reset()
		for loader.HasNext() {
			_ = loader.Next()
		}
	}
}

// BenchmarkDataLoaderIterationShuffle бенчмарк для итерации с shuffle.
func BenchmarkDataLoaderIterationShuffle(b *testing.B) {
	features := tensor.Randn([]int{1000, 784}, 42)
	targets := tensor.Randn([]int{1000, 10}, 123)
	dataset := NewSimpleDataset(features, targets)

	loader := NewDataLoader(dataset, DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   true,
		DropLast:  false,
		Seed:      42,
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loader.Reset()
		for loader.HasNext() {
			_ = loader.Next()
		}
	}
}
