package dataloader

import "github.com/Hirogava/Go-NN-Learn/pkg/tensor"

// Dataset представляет абстрактную коллекцию данных.
// Предоставляет интерфейс для доступа к отдельным примерам по индексу.
type Dataset interface {
	// Get возвращает пример данных по индексу.
	// Возвращает входные данные (features) и целевые значения (target).
	Get(index int) (features *tensor.Tensor, target *tensor.Tensor)

	// Len возвращает общее количество примеров в датасете.
	Len() int
}

// SimpleDataset — простая in-memory реализация Dataset.
// Хранит все данные в памяти в виде тензоров.
type SimpleDataset struct {
	features *tensor.Tensor // Входные данные: [num_samples, ...]
	targets  *tensor.Tensor // Целевые значения: [num_samples, ...]
	numSamples int          // Количество примеров
}

// NewSimpleDataset создает новый SimpleDataset из тензоров features и targets.
// features и targets должны иметь одинаковое количество примеров (первая размерность).
//
// Пример:
//   features := tensor.Randn([]int{100, 784}, 42)  // 100 примеров, 784 признака
//   targets := tensor.Randn([]int{100, 10}, 123)   // 100 примеров, 10 классов
//   dataset := NewSimpleDataset(features, targets)
func NewSimpleDataset(features, targets *tensor.Tensor) *SimpleDataset {
	if len(features.Shape) == 0 || len(targets.Shape) == 0 {
		panic("features and targets must be at least 1D tensors")
	}

	if features.Shape[0] != targets.Shape[0] {
		panic("features and targets must have the same number of samples (first dimension)")
	}

	return &SimpleDataset{
		features:   features,
		targets:    targets,
		numSamples: features.Shape[0],
	}
}

// Get возвращает пример данных по индексу.
// Для SimpleDataset возвращает slice из оригинальных тензоров.
func (ds *SimpleDataset) Get(index int) (*tensor.Tensor, *tensor.Tensor) {
	if index < 0 || index >= ds.numSamples {
		panic("index out of bounds")
	}

	// Извлечь один пример из features
	featureSample := ds.extractSample(ds.features, index)
	targetSample := ds.extractSample(ds.targets, index)

	return featureSample, targetSample
}

// Len возвращает общее количество примеров в датасете.
func (ds *SimpleDataset) Len() int {
	return ds.numSamples
}

// extractSample извлекает один пример из тензора по индексу.
// Возвращает новый тензор с уменьшенной на 1 размерностью.
func (ds *SimpleDataset) extractSample(t *tensor.Tensor, index int) *tensor.Tensor {
	if len(t.Shape) == 1 {
		// Для 1D тензора возвращаем скаляр как тензор с shape []
		return &tensor.Tensor{
			Data:    []float64{t.Data[index]},
			Shape:   []int{1},
			Strides: []int{1},
		}
	}

	// Для многомерного тензора извлекаем slice
	// Например, для [100, 784] -> [784]
	sampleShape := t.Shape[1:]
	sampleSize := 1
	for _, dim := range sampleShape {
		sampleSize *= dim
	}

	// Вычисляем начальный индекс в плоском массиве
	startIdx := index * sampleSize
	endIdx := startIdx + sampleSize

	// Копируем данные для безопасности
	sampleData := make([]float64, sampleSize)
	copy(sampleData, t.Data[startIdx:endIdx])

	// Вычисляем новые strides
	sampleStrides := make([]int, len(sampleShape))
	if len(sampleShape) > 0 {
		sampleStrides[len(sampleStrides)-1] = 1
		for i := len(sampleStrides) - 2; i >= 0; i-- {
			sampleStrides[i] = sampleStrides[i+1] * sampleShape[i+1]
		}
	}

	return &tensor.Tensor{
		Data:    sampleData,
		Shape:   sampleShape,
		Strides: sampleStrides,
	}
}
