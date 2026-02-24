package tensor

import (
	"fmt"
	"math"
)

// Константы для оптимизации
const (
	UnrollFactor = 8 // Развертывание циклов для SIMD
)

func Add(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают: %v != %v", a.DType, b.DType)
	}
	n := a.DataLen()
	result := &Tensor{
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
		DType:   a.DType,
	}
	if a.DType == Float32 {
		result.Data32 = make([]float32, n)
		for i := 0; i < n; i++ {
			result.Data32[i] = a.Data32[i] + b.Data32[i]
		}
	} else {
		result.Data = make([]float64, n)
		addVectorized(a.Data, b.Data, result.Data)
	}
	return result, nil
}

func Mul(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают: %v != %v", a.DType, b.DType)
	}
	n := a.DataLen()
	result := &Tensor{
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
		DType:   a.DType,
	}
	if a.DType == Float32 {
		result.Data32 = make([]float32, n)
		for i := 0; i < n; i++ {
			result.Data32[i] = a.Data32[i] * b.Data32[i]
		}
	} else {
		result.Data = make([]float64, n)
		mulVectorized(a.Data, b.Data, result.Data)
	}
	return result, nil
}

func Sub(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают: %v != %v", a.DType, b.DType)
	}
	n := a.DataLen()
	result := &Tensor{
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
		DType:   a.DType,
	}
	if a.DType == Float32 {
		result.Data32 = make([]float32, n)
		for i := 0; i < n; i++ {
			result.Data32[i] = a.Data32[i] - b.Data32[i]
		}
	} else {
		result.Data = make([]float64, n)
		subVectorized(a.Data, b.Data, result.Data)
	}
	return result, nil
}

func Div(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают: %v != %v", a.DType, b.DType)
	}
	n := a.DataLen()
	result := &Tensor{
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
		DType:   a.DType,
	}
	if a.DType == Float32 {
		result.Data32 = make([]float32, n)
		for i := 0; i < n; i++ {
			result.Data32[i] = a.Data32[i] / b.Data32[i]
		}
	} else {
		result.Data = make([]float64, n)
		divVectorized(a.Data, b.Data, result.Data)
	}
	return result, nil
}

func Apply(a *Tensor, f func(float64) float64) *Tensor {
	result := &Tensor{
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
		DType:   a.DType,
	}
	if a.DType == Float32 {
		result.Data32 = make([]float32, len(a.Data32))
		for i := range a.Data32 {
			result.Data32[i] = float32(f(float64(a.Data32[i])))
		}
	} else {
		result.Data = make([]float64, len(a.Data))
		for i := range a.Data {
			result.Data[i] = f(a.Data[i])
		}
	}
	return result
}

// Векторизованные внутренние функции

func addVectorized(a, b, result []float64) {
	n := len(a)
	i := 0

	// Основной цикл с развертыванием x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		result[idx0] = a[idx0] + b[idx0]
		result[idx1] = a[idx1] + b[idx1]
		result[idx2] = a[idx2] + b[idx2]
		result[idx3] = a[idx3] + b[idx3]
		result[idx4] = a[idx4] + b[idx4]
		result[idx5] = a[idx5] + b[idx5]
		result[idx6] = a[idx6] + b[idx6]
		result[idx7] = a[idx7] + b[idx7]
	}

	// Остаток
	for ; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

func mulVectorized(a, b, result []float64) {
	n := len(a)
	i := 0

	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		result[idx0] = a[idx0] * b[idx0]
		result[idx1] = a[idx1] * b[idx1]
		result[idx2] = a[idx2] * b[idx2]
		result[idx3] = a[idx3] * b[idx3]
		result[idx4] = a[idx4] * b[idx4]
		result[idx5] = a[idx5] * b[idx5]
		result[idx6] = a[idx6] * b[idx6]
		result[idx7] = a[idx7] * b[idx7]
	}

	for ; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

func subVectorized(a, b, result []float64) {
	n := len(a)
	i := 0

	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		result[idx0] = a[idx0] - b[idx0]
		result[idx1] = a[idx1] - b[idx1]
		result[idx2] = a[idx2] - b[idx2]
		result[idx3] = a[idx3] - b[idx3]
		result[idx4] = a[idx4] - b[idx4]
		result[idx5] = a[idx5] - b[idx5]
		result[idx6] = a[idx6] - b[idx6]
		result[idx7] = a[idx7] - b[idx7]
	}

	for ; i < n; i++ {
		result[i] = a[i] - b[i]
	}
}

func divVectorized(a, b, result []float64) {
	n := len(a)
	i := 0

	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		result[idx0] = a[idx0] / b[idx0]
		result[idx1] = a[idx1] / b[idx1]
		result[idx2] = a[idx2] / b[idx2]
		result[idx3] = a[idx3] / b[idx3]
		result[idx4] = a[idx4] / b[idx4]
		result[idx5] = a[idx5] / b[idx5]
		result[idx6] = a[idx6] / b[idx6]
		result[idx7] = a[idx7] / b[idx7]
	}

	for ; i < n; i++ {
		result[i] = a[i] / b[i]
	}
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
			return nil, fmt.Errorf("некорректная размерность: %d", dim)
		}
		newSize *= dim
	}

	// Проверяем, что размер совпадает
	oldSize := 1
	for _, dim := range a.Shape {
		oldSize *= dim
	}

	if newSize != oldSize {
		return nil, fmt.Errorf("невозможно изменить форму тензора размера %d на форму %v (размер %d)", oldSize, newShape, newSize)
	}
	newStrides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}
	out := &Tensor{
		Shape:   append([]int{}, newShape...),
		Strides: newStrides,
		DType:   a.DType,
	}
	if a.DType == Float32 {
		out.Data32 = a.Data32
	} else {
		out.Data = a.Data
	}
	return out, nil
}

func Transpose(a *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 {
		return nil, fmt.Errorf("транспонирование требует 2D тензор, получен %dD", len(a.Shape))
	}
	rows := a.Shape[0]
	cols := a.Shape[1]
	result := &Tensor{
		Shape:   []int{cols, rows},
		Strides: []int{rows, 1},
		DType:   a.DType,
	}
	if a.DType == Float32 {
		result.Data32 = make([]float32, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result.Data32[j*rows+i] = a.Data32[i*cols+j]
			}
		}
	} else {
		result.Data = make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result.Data[j*rows+i] = a.Data[i*cols+j]
			}
		}
	}
	return result, nil
}

func Sum(a *Tensor) *Tensor {
	if a.DType == Float32 {
		var s float32
		for _, v := range a.Data32 {
			s += v
		}
		return &Tensor{
			Data32:  []float32{s},
			Shape:   []int{1},
			Strides: []int{1},
			DType:   Float32,
		}
	}
	sum := 0.0
	for _, val := range a.Data {
		sum += val
	}
	return &Tensor{
		Data:    []float64{sum},
		Shape:   []int{1},
		Strides: []int{1},
		DType:   Float64,
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

// Max возвращает одноэлементный тензор с максимальным значением.
func Max(a *Tensor) *Tensor {
	if len(a.Data) == 0 {
		panic("cannot compute Max of empty tensor")
	}
	maxVal := a.Data[0]
	for _, v := range a.Data {
		if v > maxVal {
			maxVal = v
		}
	}
	return &Tensor{
		Data:    []float64{maxVal},
		Shape:   []int{1},
		Strides: []int{1},
		DType:   Float64,
	}
}
