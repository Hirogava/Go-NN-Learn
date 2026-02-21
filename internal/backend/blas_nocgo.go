//go:build !cgo
// +build !cgo

package backend

// BLASAvailable указывает, доступна ли BLAS библиотека
// В этой сборке без CGO BLAS недоступна
const BLASAvailable = false

// MatMulBLAS - заглушка для сборки без CGO
// Возвращает ошибку, так как BLAS недоступна
func MatMulBLAS(a, b *Tensor) (*Tensor, error) {
	// Fallback на нативную оптимизированную версию
	return MatMul(a, b)
}

// MatMulTransposeBBLAS - заглушка для сборки без CGO
func MatMulTransposeBBLAS(a, b *Tensor) (*Tensor, error) {
	return MatMulTransposeB(a, b)
}

// MatMulTransposeABLAS - заглушка для сборки без CGO
func MatMulTransposeABLAS(a, b *Tensor) (*Tensor, error) {
	return MatMulTransposeA(a, b)
}

// VectorAddBLAS - заглушка для сборки без CGO
func VectorAddBLAS(alpha float64, x, y []float64) {
	for i := range x {
		y[i] += alpha * x[i]
	}
}

// VectorScaleBLAS - заглушка для сборки без CGO
func VectorScaleBLAS(alpha float64, x []float64) {
	for i := range x {
		x[i] *= alpha
	}
}

// DotProductBLAS - заглушка для сборки без CGO
func DotProductBLAS(x, y []float64) float64 {
	sum := 0.0
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum
}

// MatrixVectorMultiplyBLAS - заглушка для сборки без CGO
func MatrixVectorMultiplyBLAS(alpha float64, a *Tensor, x []float64, beta float64, y []float64) error {
	m := a.Shape[0]
	n := a.Shape[1]

	// y = beta*y
	for i := 0; i < m; i++ {
		y[i] *= beta
	}

	// y += alpha * A * x
	for i := 0; i < m; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += a.Data[i*n+j] * x[j]
		}
		y[i] += alpha * sum
	}

	return nil
}
