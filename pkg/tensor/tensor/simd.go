package tensor

import "fmt"

// SIMD-оптимизированные операции для векторов
// Использует развертывание циклов (loop unrolling) для векторизации компилятором

const (
	// Размер для развертывания циклов
	UnrollFactor = 8
)

// AddOptimized - оптимизированная версия поэлементного сложения
// Использует развертывание циклов для лучшей векторизации
func AddOptimized(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match: %v vs %v", a.Shape, b.Shape)
	}

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	addVectorized(a.Data, b.Data, result.Data)
	return result, nil
}

// MulOptimized - оптимизированная версия поэлементного умножения
func MulOptimized(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match: %v vs %v", a.Shape, b.Shape)
	}

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	mulVectorized(a.Data, b.Data, result.Data)
	return result, nil
}

// addVectorized - векторизованное сложение
func addVectorized(a, b, result []float64) {
	n := len(a)
	i := 0

	// Основной цикл с развертыванием
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		result[i] = a[i] + b[i]
		result[i+1] = a[i+1] + b[i+1]
		result[i+2] = a[i+2] + b[i+2]
		result[i+3] = a[i+3] + b[i+3]
		result[i+4] = a[i+4] + b[i+4]
		result[i+5] = a[i+5] + b[i+5]
		result[i+6] = a[i+6] + b[i+6]
		result[i+7] = a[i+7] + b[i+7]
	}

	// Остаток
	for ; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

// mulVectorized - векторизованное умножение
func mulVectorized(a, b, result []float64) {
	n := len(a)
	i := 0

	// Основной цикл с развертыванием
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		result[i] = a[i] * b[i]
		result[i+1] = a[i+1] * b[i+1]
		result[i+2] = a[i+2] * b[i+2]
		result[i+3] = a[i+3] * b[i+3]
		result[i+4] = a[i+4] * b[i+4]
		result[i+5] = a[i+5] * b[i+5]
		result[i+6] = a[i+6] * b[i+6]
		result[i+7] = a[i+7] * b[i+7]
	}

	// Остаток
	for ; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

// dotProduct - оптимизированное скалярное произведение векторов
func dotProduct(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}

	var sum float64
	i := 0

	// Развертывание для векторизации
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		sum += a[i]*b[i] +
			a[i+1]*b[i+1] +
			a[i+2]*b[i+2] +
			a[i+3]*b[i+3] +
			a[i+4]*b[i+4] +
			a[i+5]*b[i+5] +
			a[i+6]*b[i+6] +
			a[i+7]*b[i+7]
	}

	// Остаток
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// FusedMultiplyAdd - оптимизированная FMA операция: result = a * b + c
// Важна для нейронных сетей (линейный слой: y = Wx + b)
func FusedMultiplyAdd(a, b, c *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) || !shapesEqual(a.Shape, c.Shape) {
		return nil, fmt.Errorf("shapes must match")
	}

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	n := len(a.Data)
	i := 0

	// Развертывание цикла для FMA
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		result.Data[i] = a.Data[i]*b.Data[i] + c.Data[i]
		result.Data[i+1] = a.Data[i+1]*b.Data[i+1] + c.Data[i+1]
		result.Data[i+2] = a.Data[i+2]*b.Data[i+2] + c.Data[i+2]
		result.Data[i+3] = a.Data[i+3]*b.Data[i+3] + c.Data[i+3]
		result.Data[i+4] = a.Data[i+4]*b.Data[i+4] + c.Data[i+4]
		result.Data[i+5] = a.Data[i+5]*b.Data[i+5] + c.Data[i+5]
		result.Data[i+6] = a.Data[i+6]*b.Data[i+6] + c.Data[i+6]
		result.Data[i+7] = a.Data[i+7]*b.Data[i+7] + c.Data[i+7]
	}

	for ; i < n; i++ {
		result.Data[i] = a.Data[i]*b.Data[i] + c.Data[i]
	}

	return result, nil
}

// ScaleAdd - оптимизированная операция: result = alpha * a + b
// Используется в оптимизаторах (momentum, adam)
func ScaleAdd(alpha float64, a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("shapes must match")
	}

	result := &Tensor{
		Data:    make([]float64, len(a.Data)),
		Shape:   append([]int{}, a.Shape...),
		Strides: append([]int{}, a.Strides...),
	}

	n := len(a.Data)
	i := 0

	// Развертывание цикла
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		result.Data[i] = alpha*a.Data[i] + b.Data[i]
		result.Data[i+1] = alpha*a.Data[i+1] + b.Data[i+1]
		result.Data[i+2] = alpha*a.Data[i+2] + b.Data[i+2]
		result.Data[i+3] = alpha*a.Data[i+3] + b.Data[i+3]
		result.Data[i+4] = alpha*a.Data[i+4] + b.Data[i+4]
		result.Data[i+5] = alpha*a.Data[i+5] + b.Data[i+5]
		result.Data[i+6] = alpha*a.Data[i+6] + b.Data[i+6]
		result.Data[i+7] = alpha*a.Data[i+7] + b.Data[i+7]
	}

	for ; i < n; i++ {
		result.Data[i] = alpha*a.Data[i] + b.Data[i]
	}

	return result, nil
}

// ScaleInPlace - оптимизированное масштабирование in-place: a = alpha * a
func ScaleInPlace(alpha float64, a *Tensor) {
	n := len(a.Data)
	i := 0

	// Развертывание цикла
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		a.Data[i] *= alpha
		a.Data[i+1] *= alpha
		a.Data[i+2] *= alpha
		a.Data[i+3] *= alpha
		a.Data[i+4] *= alpha
		a.Data[i+5] *= alpha
		a.Data[i+6] *= alpha
		a.Data[i+7] *= alpha
	}

	for ; i < n; i++ {
		a.Data[i] *= alpha
	}
}
