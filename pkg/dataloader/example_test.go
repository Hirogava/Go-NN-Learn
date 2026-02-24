package dataloader_test

import (
	"fmt"

	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// ExampleSimpleDataset демонстрирует базовое использование SimpleDataset.
func ExampleSimpleDataset() {
	// Создаем синтетические данные
	features := tensor.Randn([]int{100, 10}, 42)  // 100 примеров, 10 признаков
	targets := tensor.Randn([]int{100, 1}, 123)   // 100 примеров, 1 выходное значение

	// Создаем датасет
	dataset := dataloader.NewSimpleDataset(features, targets)

	fmt.Printf("Dataset size: %d\n", dataset.Len())

	// Получаем один пример
	feature, target := dataset.Get(0)
	fmt.Printf("Feature shape: %v\n", feature.Shape)
	fmt.Printf("Target shape: %v\n", target.Shape)

	// Output:
	// Dataset size: 100
	// Feature shape: [10]
	// Target shape: [1]
}

// ExampleDataLoader демонстрирует базовое использование DataLoader.
func ExampleDataLoader() {
	// Создаем датасет
	features := tensor.Randn([]int{100, 784}, 42)  // 100 MNIST-подобных примеров
	targets := tensor.Randn([]int{100, 10}, 123)   // 100 примеров, 10 классов

	dataset := dataloader.NewSimpleDataset(features, targets)

	// Создаем DataLoader
	loader := dataloader.NewDataLoader(dataset, dataloader.DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   false,
		DropLast:  false,
		Seed:      42,
	})

	fmt.Printf("Number of batches: %d\n", loader.Len())
	fmt.Printf("Batch size: %d\n", loader.BatchSize())

	// Итерация по одной эпохе
	batchCount := 0
	for loader.HasNext() {
		batch := loader.Next()
		batchCount++
		fmt.Printf("Batch %d: features shape %v, targets shape %v\n",
			batchCount, batch.Features.Shape, batch.Targets.Shape)
	}

	// Output:
	// Number of batches: 4
	// Batch size: 32
	// Batch 1: features shape [32 784], targets shape [32 10]
	// Batch 2: features shape [32 784], targets shape [32 10]
	// Batch 3: features shape [32 784], targets shape [32 10]
	// Batch 4: features shape [4 784], targets shape [4 10]
}

// ExampleDataLoader_shuffle демонстрирует использование shuffle.
func ExampleDataLoader_shuffle() {
	features := tensor.Randn([]int{50, 5}, 42)
	targets := tensor.Randn([]int{50, 1}, 123)
	dataset := dataloader.NewSimpleDataset(features, targets)

	// DataLoader с перемешиванием
	loader := dataloader.NewDataLoader(dataset, dataloader.DataLoaderConfig{
		BatchSize: 10,
		Shuffle:   true,  // Включаем перемешивание
		DropLast:  false,
		Seed:      777,
	})

	// Первая эпоха
	fmt.Println("Epoch 1:")
	for loader.HasNext() {
		batch := loader.Next()
		fmt.Printf("  Batch shape: %v\n", batch.Features.Shape)
	}

	// Начинаем новую эпоху с новым перемешиванием
	loader.Reset()
	fmt.Println("Epoch 2 (after reset with new shuffle):")
	for loader.HasNext() {
		batch := loader.Next()
		fmt.Printf("  Batch shape: %v\n", batch.Features.Shape)
	}

	// Output:
	// Epoch 1:
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	// Epoch 2 (after reset with new shuffle):
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
	//   Batch shape: [10 5]
}

// ExampleDataLoader_dropLast демонстрирует использование DropLast.
func ExampleDataLoader_dropLast() {
	features := tensor.Randn([]int{25, 3}, 42)
	targets := tensor.Randn([]int{25, 1}, 123)
	dataset := dataloader.NewSimpleDataset(features, targets)

	// С DropLast=true отбрасываем последний неполный батч
	loader := dataloader.NewDataLoader(dataset, dataloader.DataLoaderConfig{
		BatchSize: 10,
		Shuffle:   false,
		DropLast:  true,  // Отбрасываем неполный батч
		Seed:      42,
	})

	fmt.Printf("Total batches with DropLast=true: %d\n", loader.Len())

	for loader.HasNext() {
		batch := loader.Next()
		fmt.Printf("Batch size: %d\n", batch.Features.Shape[0])
	}

	// Output:
	// Total batches with DropLast=true: 2
	// Batch size: 10
	// Batch size: 10
}

// ExampleDataLoader_trainingLoop демонстрирует типичный цикл обучения.
func ExampleDataLoader_trainingLoop() {
	// Подготовка данных
	features := tensor.Randn([]int{100, 20}, 42)
	targets := tensor.Randn([]int{100, 5}, 123)
	dataset := dataloader.NewSimpleDataset(features, targets)

	loader := dataloader.NewDataLoader(dataset, dataloader.DataLoaderConfig{
		BatchSize: 32,
		Shuffle:   true,
		DropLast:  false,
		Seed:      42,
	})

	// Симуляция нескольких эпох обучения
	numEpochs := 3
	for epoch := 1; epoch <= numEpochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch, numEpochs)

		// Сбрасываем loader для новой эпохи (с новым shuffle)
		loader.Reset()

		batchNum := 0
		for loader.HasNext() {
			batch := loader.Next()
			batchNum++

			// Здесь был бы forward pass, backward pass, и обновление параметров
			// Для примера просто выводим информацию
			_ = batch // Используем batch

			if batchNum == 1 {
				fmt.Printf("  First batch shape: %v\n", batch.Features.Shape)
			}
		}

		fmt.Printf("  Completed %d batches\n", batchNum)
	}

	// Output:
	// Epoch 1/3
	//   First batch shape: [32 20]
	//   Completed 4 batches
	// Epoch 2/3
	//   First batch shape: [32 20]
	//   Completed 4 batches
	// Epoch 3/3
	//   First batch shape: [32 20]
	//   Completed 4 batches
}
