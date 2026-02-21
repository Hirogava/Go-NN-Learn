package backend

import "fmt"

// In-place операции для минимизации аллокаций памяти
// Эти функции изменяют тензор на месте вместо создания нового

// AddInPlace выполняет a = a + b (изменяет a)
func AddInPlace(a, b *Tensor) error {
	if !shapesEqual(a.Shape, b.Shape) {
		return fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}

	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		a.Data[idx0] += b.Data[idx0]
		a.Data[idx1] += b.Data[idx1]
		a.Data[idx2] += b.Data[idx2]
		a.Data[idx3] += b.Data[idx3]
		a.Data[idx4] += b.Data[idx4]
		a.Data[idx5] += b.Data[idx5]
		a.Data[idx6] += b.Data[idx6]
		a.Data[idx7] += b.Data[idx7]
	}

	for ; i < n; i++ {
		a.Data[i] += b.Data[i]
	}

	return nil
}

// SubInPlace выполняет a = a - b (изменяет a)
func SubInPlace(a, b *Tensor) error {
	if !shapesEqual(a.Shape, b.Shape) {
		return fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}

	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		a.Data[idx0] -= b.Data[idx0]
		a.Data[idx1] -= b.Data[idx1]
		a.Data[idx2] -= b.Data[idx2]
		a.Data[idx3] -= b.Data[idx3]
		a.Data[idx4] -= b.Data[idx4]
		a.Data[idx5] -= b.Data[idx5]
		a.Data[idx6] -= b.Data[idx6]
		a.Data[idx7] -= b.Data[idx7]
	}

	for ; i < n; i++ {
		a.Data[i] -= b.Data[i]
	}

	return nil
}

// MulInPlace выполняет a = a * b (изменяет a)
func MulInPlace(a, b *Tensor) error {
	if !shapesEqual(a.Shape, b.Shape) {
		return fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}

	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		a.Data[idx0] *= b.Data[idx0]
		a.Data[idx1] *= b.Data[idx1]
		a.Data[idx2] *= b.Data[idx2]
		a.Data[idx3] *= b.Data[idx3]
		a.Data[idx4] *= b.Data[idx4]
		a.Data[idx5] *= b.Data[idx5]
		a.Data[idx6] *= b.Data[idx6]
		a.Data[idx7] *= b.Data[idx7]
	}

	for ; i < n; i++ {
		a.Data[i] *= b.Data[i]
	}

	return nil
}

// DivInPlace выполняет a = a / b (изменяет a)
func DivInPlace(a, b *Tensor) error {
	if !shapesEqual(a.Shape, b.Shape) {
		return fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}

	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		a.Data[idx0] /= b.Data[idx0]
		a.Data[idx1] /= b.Data[idx1]
		a.Data[idx2] /= b.Data[idx2]
		a.Data[idx3] /= b.Data[idx3]
		a.Data[idx4] /= b.Data[idx4]
		a.Data[idx5] /= b.Data[idx5]
		a.Data[idx6] /= b.Data[idx6]
		a.Data[idx7] /= b.Data[idx7]
	}

	for ; i < n; i++ {
		a.Data[i] /= b.Data[i]
	}

	return nil
}

// ApplyInPlace применяет функцию к каждому элементу тензора in-place
func ApplyInPlace(a *Tensor, f func(float64) float64) {
	for i := range a.Data {
		a.Data[i] = f(a.Data[i])
	}
}

// ClipInPlace ограничивает значения тензора в диапазоне [min, max]
func ClipInPlace(a *Tensor, minVal, maxVal float64) {
	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		for j := 0; j < UnrollFactor; j++ {
			idx := i + j
			if a.Data[idx] < minVal {
				a.Data[idx] = minVal
			} else if a.Data[idx] > maxVal {
				a.Data[idx] = maxVal
			}
		}
	}

	for ; i < n; i++ {
		if a.Data[i] < minVal {
			a.Data[i] = minVal
		} else if a.Data[i] > maxVal {
			a.Data[i] = maxVal
		}
	}
}

// FillInPlace заполняет тензор константой
func FillInPlace(a *Tensor, value float64) {
	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		a.Data[idx0] = value
		a.Data[idx1] = value
		a.Data[idx2] = value
		a.Data[idx3] = value
		a.Data[idx4] = value
		a.Data[idx5] = value
		a.Data[idx6] = value
		a.Data[idx7] = value
	}

	for ; i < n; i++ {
		a.Data[i] = value
	}
}

// ZeroInPlace обнуляет тензор
func ZeroInPlace(a *Tensor) {
	FillInPlace(a, 0.0)
}

// CopyInto копирует данные из src в dst (dst изменяется)
func CopyInto(dst, src *Tensor) error {
	if !shapesEqual(dst.Shape, src.Shape) {
		return fmt.Errorf("формы тензоров должны совпадать: %v != %v", dst.Shape, src.Shape)
	}

	n := len(src.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		dst.Data[idx0] = src.Data[idx0]
		dst.Data[idx1] = src.Data[idx1]
		dst.Data[idx2] = src.Data[idx2]
		dst.Data[idx3] = src.Data[idx3]
		dst.Data[idx4] = src.Data[idx4]
		dst.Data[idx5] = src.Data[idx5]
		dst.Data[idx6] = src.Data[idx6]
		dst.Data[idx7] = src.Data[idx7]
	}

	for ; i < n; i++ {
		dst.Data[i] = src.Data[i]
	}

	return nil
}

// ScaleInPlace умножает все элементы тензора на скаляр (изменяет тензор)
// a = scale * a
func ScaleInPlace(scale float64, a *Tensor) {
	n := len(a.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		a.Data[idx0] *= scale
		a.Data[idx1] *= scale
		a.Data[idx2] *= scale
		a.Data[idx3] *= scale
		a.Data[idx4] *= scale
		a.Data[idx5] *= scale
		a.Data[idx6] *= scale
		a.Data[idx7] *= scale
	}

	for ; i < n; i++ {
		a.Data[i] *= scale
	}
}

// AccumulateInto выполняет dst = dst + alpha * src
// Полезно для градиентного накопления
func AccumulateInto(dst, src *Tensor, alpha float64) error {
	if !shapesEqual(dst.Shape, src.Shape) {
		return fmt.Errorf("формы тензоров должны совпадать: %v != %v", dst.Shape, src.Shape)
	}

	n := len(src.Data)
	i := 0

	// Векторизация x8
	for ; i <= n-UnrollFactor; i += UnrollFactor {
		idx0, idx1, idx2, idx3 := i, i+1, i+2, i+3
		idx4, idx5, idx6, idx7 := i+4, i+5, i+6, i+7

		dst.Data[idx0] += alpha * src.Data[idx0]
		dst.Data[idx1] += alpha * src.Data[idx1]
		dst.Data[idx2] += alpha * src.Data[idx2]
		dst.Data[idx3] += alpha * src.Data[idx3]
		dst.Data[idx4] += alpha * src.Data[idx4]
		dst.Data[idx5] += alpha * src.Data[idx5]
		dst.Data[idx6] += alpha * src.Data[idx6]
		dst.Data[idx7] += alpha * src.Data[idx7]
	}

	for ; i < n; i++ {
		dst.Data[i] += alpha * src.Data[i]
	}

	return nil
}
