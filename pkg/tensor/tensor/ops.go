package tensor

import (
	"fmt"
	"math"
)

// add выполняет поэлементное сложение двух тензоров.
// возвращает новый тензор с, где с[i] = а[i] + в[i]
// тензоры должны иметь одинаковую форму (shape).
func Add(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match: %v vs %v", a.Shape, b.Shape)
	}

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}

	return result, nil
}

// mul выполняет поэлементное умножение двух тензоров (операция Адамара).
// возвращает новый тензор с, где с[i] = а[i] * в[i]
// тензоры должны иметь одинаковую форму (shape).
func Mul(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match: %v vs %v", a.Shape, b.Shape)
	}

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] * b.Data[i]
	}

	return result, nil
}

// apply применяет функцию f к каждому элементу тензора.
// возвращает новый тензор с результатами применения функции.
// используется для функций активации (reLU, sigmoid, tanh).
func Apply(a *Tensor, f func(float64) float64) *Tensor {
	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = f(a.Data[i])
	}

	return result
}

// shapesEqual проверяет равенство форм двух тензоров.
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Reshape изменяет форму тензора без изменения данных.
// Возвращает новый тензор с новой формой, используя те же данные.
// Общее количество элементов должно совпадать.
func Reshape(a *Tensor, newShape []int) (*Tensor, error) {
	// Вычисляем общее количество элементов в новой форме
	newSize := 1
	for _, dim := range newShape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid dimension: %d", dim)
		}
		newSize *= dim
	}

	// Проверяем, что размер совпадает
	oldSize := 1
	for _, dim := range a.Shape {
		oldSize *= dim
	}

	if newSize != oldSize {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v (size %d)", oldSize, newShape, newSize)
	}

	// Вычисляем новые strides
	newStrides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	return &Tensor{
		Data:    a.Data, // Используем те же данные
		Shape:   append([]int{}, newShape...),
		Strides: newStrides,
	}, nil
}

// Transpose транспонирует двумерный тензор (матрицу).
// Меняет местами оси: строки становятся столбцами и наоборот.
// Для матрицы [m, n] возвращает матрицу [n, m].
func Transpose(a *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 {
		return nil, fmt.Errorf("transpose requires 2D tensor, got %dD", len(a.Shape))
	}

	rows := a.Shape[0]
	cols := a.Shape[1]

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   []int{cols, rows},
		Strides: []int{rows, 1},
	}

	// Транспонируем данные
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Data[j*rows+i] = a.Data[i*cols+j]
		}
	}

	return result, nil
}

// Sum вычисляет сумму всех элементов тензора.
// Возвращает скаляр (одноэлементный тензор).
func Sum(a *Tensor) *Tensor {
	sum := 0.0
	for _, val := range a.Data {
		sum += val
	}

	return &Tensor{
		Data:    []float64{sum},
		Shape:   []int{1},
		Strides: []int{1},
	}
}

// Exp применяет экспоненциальную функцию e^x к каждому элементу тензора.
// Возвращает новый тензор с результатами применения exp.
func Exp(a *Tensor) *Tensor {
	return Apply(a, math.Exp)
}

// Log применяет натуральный логарифм ln(x) к каждому элементу тензора.
// Возвращает новый тензор с результатами применения log.
func Log(a *Tensor) *Tensor {
	return Apply(a, math.Log)
}
