package tensor

import "fmt"

// Add выполняет поэлементное сложение двух тензоров.
// Возвращает новый тензор C, где C[i] = A[i] + B[i]
// Тензоры должны иметь одинаковую форму (Shape).
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

// Mul выполняет поэлементное умножение двух тензоров (операция Адамара).
// Возвращает новый тензор C, где C[i] = A[i] * B[i]
// Тензоры должны иметь одинаковую форму (Shape).
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

// Apply применяет функцию f к каждому элементу тензора.
// Возвращает новый тензор с результатами применения функции.
// Используется для функций активации (ReLU, Sigmoid, Tanh).
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
