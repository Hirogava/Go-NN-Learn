package dataloader

import (
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// Batch представляет один мини-батч данных.
// Features — входные данные батча, Targets — целевые значения.
type Batch struct {
	Features *tensor.Tensor // [batch_size, ...]
	Targets  *tensor.Tensor // [batch_size, ...]
}

// DataLoader — итератор для загрузки данных мини-батчами.
// Отвечает за батчинг, перемешивание и итерацию по Dataset.
type DataLoader struct {
	dataset   Dataset    // Источник данных
	batchSize int        // Размер мини-батча
	shuffle   bool       // Перемешивать ли данные перед каждой эпохой
	dropLast  bool       // Отбрасывать ли последний неполный батч
	rng       *rand.Rand // Генератор случайных чисел для shuffle

	// Внутреннее состояние итератора
	indices    []int // Порядок индексов для текущей эпохи
	currentIdx int   // Текущая позиция в indices
}

// DataLoaderConfig — конфигурация для создания DataLoader.
type DataLoaderConfig struct {
	BatchSize int   // Размер батча (обязательный)
	Shuffle   bool  // Перемешивать данные (по умолчанию false)
	DropLast  bool  // Отбрасывать последний неполный батч (по умолчанию false)
	Seed      int64 // Seed для генератора случайных чисел (по умолчанию 0)
}

// NewDataLoader создает новый DataLoader с заданной конфигурацией.
//
// Параметры:
//   - dataset: источник данных (реализация интерфейса Dataset)
//   - config: конфигурация DataLoader
//
// Пример:
//
//	loader := NewDataLoader(dataset, DataLoaderConfig{
//	    BatchSize: 32,
//	    Shuffle:   true,
//	    DropLast:  false,
//	    Seed:      42,
//	})
func NewDataLoader(dataset Dataset, config DataLoaderConfig) *DataLoader {
	if config.BatchSize <= 0 {
		panic("batch size must be positive")
	}

	if config.BatchSize > dataset.Len() {
		panic("batch size cannot be larger than dataset size")
	}

	// Создаем генератор случайных чисел
	rng := rand.New(rand.NewSource(config.Seed))

	// Инициализируем порядок индексов
	indices := make([]int, dataset.Len())
	for i := range indices {
		indices[i] = i
	}

	dl := &DataLoader{
		dataset:    dataset,
		batchSize:  config.BatchSize,
		shuffle:    config.Shuffle,
		dropLast:   config.DropLast,
		rng:        rng,
		indices:    indices,
		currentIdx: 0,
	}

	// Если нужно перемешивание, делаем его сразу
	if dl.shuffle {
		dl.shuffleIndices()
	}

	return dl
}

// shuffleIndices перемешивает массив индексов случайным образом.
func (dl *DataLoader) shuffleIndices() {
	// Fisher-Yates shuffle
	n := len(dl.indices)
	for i := n - 1; i > 0; i-- {
		j := dl.rng.Intn(i + 1)
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
	}
}

// Reset сбрасывает итератор в начало и перемешивает данные (если shuffle=true).
// Используется для начала новой эпохи обучения.
func (dl *DataLoader) Reset() {
	dl.currentIdx = 0
	if dl.shuffle {
		dl.shuffleIndices()
	}
}

// HasNext проверяет, есть ли еще батчи для итерации.
func (dl *DataLoader) HasNext() bool {
	remaining := len(dl.indices) - dl.currentIdx

	if dl.dropLast {
		// Если dropLast=true, нужен полный батч
		return remaining >= dl.batchSize
	}

	// Иначе достаточно хотя бы одного элемента
	return remaining > 0
}

// Next возвращает следующий батч данных.
// Вызывать только если HasNext() возвращает true.
//
// Возвращает:
//   - Batch с входными данными и целевыми значениями
//
// Паникует если нет доступных батчей.
func (dl *DataLoader) Next() *Batch {
	if !dl.HasNext() {
		panic("no more batches available, call Reset() to start a new epoch")
	}

	// Определяем размер текущего батча
	remaining := len(dl.indices) - dl.currentIdx
	currentBatchSize := dl.batchSize
	if remaining < dl.batchSize {
		currentBatchSize = remaining
	}

	// Собираем индексы для текущего батча
	batchIndices := dl.indices[dl.currentIdx : dl.currentIdx+currentBatchSize]
	dl.currentIdx += currentBatchSize

	// Собираем данные батча
	return dl.collectBatch(batchIndices)
}

// collectBatch собирает данные для батча из dataset по индексам.
func (dl *DataLoader) collectBatch(indices []int) *Batch {
	if len(indices) == 0 {
		panic("cannot create batch from empty indices")
	}

	// Получаем первый пример чтобы узнать размерности
	firstFeature, firstTarget := dl.dataset.Get(indices[0])

	batchSize := len(indices)
	featureShape := append([]int{batchSize}, firstFeature.Shape...)
	targetShape := append([]int{batchSize}, firstTarget.Shape...)

	// Вычисляем общий размер данных
	featureSize := 1
	for _, dim := range featureShape {
		featureSize *= dim
	}
	targetSize := 1
	for _, dim := range targetShape {
		targetSize *= dim
	}

	// Создаем буферы для батча
	featureData := make([]float64, featureSize)
	targetData := make([]float64, targetSize)

	// Размер одного примера
	sampleFeatureSize := 1
	for _, dim := range firstFeature.Shape {
		sampleFeatureSize *= dim
	}
	sampleTargetSize := 1
	for _, dim := range firstTarget.Shape {
		sampleTargetSize *= dim
	}

	// Заполняем батч данными
	for i, idx := range indices {
		feature, target := dl.dataset.Get(idx)

		// Копируем features
		featureOffset := i * sampleFeatureSize
		copy(featureData[featureOffset:featureOffset+sampleFeatureSize], feature.Data)

		// Копируем targets
		targetOffset := i * sampleTargetSize
		copy(targetData[targetOffset:targetOffset+sampleTargetSize], target.Data)
	}

	// Вычисляем strides для батча
	featureStrides := make([]int, len(featureShape))
	if len(featureShape) > 0 {
		featureStrides[len(featureStrides)-1] = 1
		for i := len(featureStrides) - 2; i >= 0; i-- {
			featureStrides[i] = featureStrides[i+1] * featureShape[i+1]
		}
	}

	targetStrides := make([]int, len(targetShape))
	if len(targetShape) > 0 {
		targetStrides[len(targetStrides)-1] = 1
		for i := len(targetStrides) - 2; i >= 0; i-- {
			targetStrides[i] = targetStrides[i+1] * targetShape[i+1]
		}
	}

	return &Batch{
		Features: &tensor.Tensor{
			Data:    featureData,
			Shape:   featureShape,
			Strides: featureStrides,
		},
		Targets: &tensor.Tensor{
			Data:    targetData,
			Shape:   targetShape,
			Strides: targetStrides,
		},
	}
}

// Len возвращает количество батчей в одной эпохе.
func (dl *DataLoader) Len() int {
	totalSamples := dl.dataset.Len()

	if dl.dropLast {
		return totalSamples / dl.batchSize
	}

	// Округление вверх
	return (totalSamples + dl.batchSize - 1) / dl.batchSize
}

// BatchSize возвращает размер батча.
func (dl *DataLoader) BatchSize() int {
	return dl.batchSize
}
