package backend

import (
	"math/rand"
)

// Zeros создаёт тензор заполненный нулями с указанной формой.
// Используется для инициализации градиентов и промежуточных результатов.
func Zeros(shape ...int) *Tensor {
	size := calculateSize(shape)
	data := make([]float64, size)
	strides := calculateStrides(shape)

	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: strides,
	}
}

// Ones создаёт тензор заполненный единицами с указанной формой.
// Используется для инициализации bias и масок.
func Ones(shape ...int) *Tensor {
	size := calculateSize(shape)
	data := make([]float64, size)

	for i := range data {
		data[i] = 1.0
	}

	strides := calculateStrides(shape)

	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: strides,
	}
}

// Randn создаёт тензор с случайными значениями из нормального распределения N(0, 1).
// seed определяет начальное значение генератора случайных чисел для воспроизводимости.
// Используется для инициализации весов нейронных сетей.
func Randn(shape []int, seed int64) *Tensor {
	size := calculateSize(shape)
	data := make([]float64, size)

	rng := rand.New(rand.NewSource(seed))

	for i := range data {
		data[i] = rng.NormFloat64()
	}

	strides := calculateStrides(shape)

	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: strides,
	}
}

// calculateSize вычисляет общее количество элементов в тензоре по его форме.
func calculateSize(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

// calculateStrides вычисляет шаги (strides) для row-major порядка хранения.
func calculateStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}
