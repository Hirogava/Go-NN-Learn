//go:build !blas
// +build !blas

package tensor

import "fmt"

const BLASAvailable = false

func MatMulBLAS(a, b *Tensor) (*Tensor, error) {
	return MatMul(a, b)
}

func MatMulTransposeBBLAS(a, b *Tensor) (*Tensor, error) {
	return MatMulTransposeB(a, b)
}

func MatMulTransposeABLAS(a, b *Tensor) (*Tensor, error) {
	return MatMulTransposeA(a, b)
}

func VectorAddBLAS(alpha float64, x, y []float64) {
	for i := range x {
		y[i] += alpha * x[i]
	}
}

func VectorScaleBLAS(alpha float64, x []float64) {
	for i := range x {
		x[i] *= alpha
	}
}

func DotProductBLAS(x, y []float64) float64 {
	sum := 0.0
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum
}

func MatrixVectorMultiplyBLAS(alpha float64, a *Tensor, x []float64, beta float64, y []float64) error {
	if len(a.Shape) != 2 {
		return fmt.Errorf("матрица должна быть 2D")
	}
	m := a.Shape[0]
	n := a.Shape[1]
	if len(x) < n || len(y) < m {
		return fmt.Errorf("неверная длина векторов: x=%d y=%d, нужно x>=%d y>=%d", len(x), len(y), n, m)
	}
	for i := 0; i < m; i++ {
		y[i] *= beta
	}
	if a.DType == Float32 {
		for i := 0; i < m; i++ {
			var sum float32
			for j := 0; j < n; j++ {
				sum += a.Data32[i*n+j] * float32(x[j])
			}
			y[i] += alpha * float64(sum)
		}
	} else {
		for i := 0; i < m; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				sum += a.Data[i*n+j] * x[j]
			}
			y[i] += alpha * sum
		}
	}
	return nil
}
