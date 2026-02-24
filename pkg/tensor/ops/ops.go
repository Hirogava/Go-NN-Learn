package ops

import (
	"fmt"
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// Add выполняет поэлементное сложение двух тензоров.
// Возвращает новый тензор C, где C[i] = A[i] + B[i]
// Тензоры должны иметь одинаковую форму (Shape).
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match: %v vs %v", a.Shape, b.Shape)
	}

	result := &tensor.Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}

	return result, nil
}

// Mul выполняет поэлементное умножение двух тензоров (операция Адамара).
// Возвращает новый тензор C, где C[i] = A[i] * B[i]
// Тензоры должны иметь одинаковую форму (Shape).
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match: %v vs %v", a.Shape, b.Shape)
	}

	result := &tensor.Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] * b.Data[i]
	}

	return result, nil
}

// Apply применяет функцию f к каждому элементу тензора.
// Возвращает новый тензор с результатами применения функции.
// Используется для функций активации (ReLU, Sigmoid, Tanh).
func Apply(a *tensor.Tensor, f func(float64) float64) *tensor.Tensor {
	result := &tensor.Tensor{
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
func Reshape(a *tensor.Tensor, newShape []int) (*tensor.Tensor, error) {
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

	return &tensor.Tensor{
		Data:    a.Data, // Используем те же данные
		Shape:   append([]int{}, newShape...),
		Strides: newStrides,
	}, nil
}

// Transpose транспонирует двумерный тензор (матрицу).
// Меняет местами оси: строки становятся столбцами и наоборот.
// Для матрицы [m, n] возвращает матрицу [n, m].
func Transpose(a *tensor.Tensor) (*tensor.Tensor, error) {
	if len(a.Shape) != 2 {
		return nil, fmt.Errorf("transpose requires 2D tensor, got %dD", len(a.Shape))
	}

	rows := a.Shape[0]
	cols := a.Shape[1]

	result := &tensor.Tensor{
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
func Sum(a *tensor.Tensor) *tensor.Tensor {
	sum := 0.0
	for _, val := range a.Data {
		sum += val
	}

	return &tensor.Tensor{
		Data:    []float64{sum},
		Shape:   []int{1},
		Strides: []int{1},
	}
}

// Exp применяет экспоненциальную функцию e^x к каждому элементу тензора.
// Возвращает новый тензор с результатами применения exp.
func Exp(a *tensor.Tensor) *tensor.Tensor {
	return Apply(a, math.Exp)
}

// Log применяет натуральный логарифм ln(x) к каждому элементу тензора.
// Возвращает новый тензор с результатами применения log.
func Log(a *tensor.Tensor) *tensor.Tensor {
	return Apply(a, math.Log)
}

func Max(a *tensor.Tensor) *tensor.Tensor {
	if len(a.Data) == 0 {
		panic("Cannot compute max of empty tensor")
	}
	maxVal := a.Data[0]
	for _, val := range a.Data {
		if val > maxVal {
			maxVal = val
		}
	}
	return &tensor.Tensor{Data: []float64{maxVal}, Shape: []int{1}}
}

func Sub(a *tensor.Tensor, scalar *tensor.Tensor) *tensor.Tensor {
	if len(scalar.Data) != 1 {
		panic("Sub expects a scalar tensor as the second argument")
	}
	result := tensor.Zeros(a.Shape...)
	for i := range a.Data {
		result.Data[i] = a.Data[i] - scalar.Data[0]
	}
	return result
}

func Div(a *tensor.Tensor, other *tensor.Tensor) *tensor.Tensor {
	if len(a.Shape) != 2 || len(other.Shape) != 1 || a.Shape[0] != other.Shape[0] {
		panic("Invalid shapes for division")
	}
	rows, cols := a.Shape[0], a.Shape[1]
	result := tensor.Zeros(a.Shape...)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Data[i*cols+j] = a.Data[i*cols+j] / other.Data[i]
		}
	}
	return result
}
