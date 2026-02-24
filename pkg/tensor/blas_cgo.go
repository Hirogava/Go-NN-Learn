//go:build cgo && blas
// +build cgo,blas

package tensor

/*
#cgo CFLAGS: -I/usr/include/openblas
#cgo LDFLAGS: -lopenblas -lm

#include <cblas.h>

// Обертки для вызова BLAS функций из Go
// Эти функции используют оптимизированные BLAS реализации (OpenBLAS, MKL, и т.д.)
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// BLASAvailable указывает, доступна ли BLAS библиотека
const BLASAvailable = true

// MatMulBLAS выполняет умножение матриц используя CBLAS (OpenBLAS/MKL)
// Это самая быстрая реализация, использующая оптимизированные BLAS библиотеки
// Использует cblas_dgemm (Double precision GEneral Matrix-Matrix multiply)
//
// C = alpha * A * B + beta * C
// Для обычного умножения: alpha=1.0, beta=0.0
func MatMulBLAS(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("умножение матриц требует 2D тензоры")
	}

	m := a.Shape[0] // строки A
	n := a.Shape[1] // столбцы A = строки B
	p := b.Shape[1] // столбцы B

	if n != b.Shape[0] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц: [%d,%d] и [%d,%d]", m, n, b.Shape[0], p)
	}

	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// Вызываем cblas_dgemm
	// void cblas_dgemm(
	//     const enum CBLAS_ORDER Order,           // CblasRowMajor или CblasColMajor
	//     const enum CBLAS_TRANSPOSE TransA,      // CblasNoTrans, CblasTrans, CblasConjTrans
	//     const enum CBLAS_TRANSPOSE TransB,
	//     const int M,                            // строки A
	//     const int N,                            // столбцы B
	//     const int K,                            // столбцы A = строки B
	//     const double alpha,                     // скаляр alpha
	//     const double *A,                        // матрица A
	//     const int lda,                          // leading dimension A (обычно K для row-major)
	//     const double *B,                        // матрица B
	//     const int ldb,                          // leading dimension B (обычно N для row-major)
	//     const double beta,                      // скаляр beta
	//     double *C,                              // результат C
	//     const int ldc                           // leading dimension C (обычно N для row-major)
	// );

	C.cblas_dgemm(
		C.CblasRowMajor,                         // Row-major порядок (как в Go)
		C.CblasNoTrans,                          // A не транспонирована
		C.CblasNoTrans,                          // B не транспонирована
		C.int(m),                                // M
		C.int(p),                                // N
		C.int(n),                                // K
		C.double(1.0),                           // alpha = 1.0
		(*C.double)(unsafe.Pointer(&a.Data[0])), // A
		C.int(n),                                // lda
		(*C.double)(unsafe.Pointer(&b.Data[0])), // B
		C.int(p),                                // ldb
		C.double(0.0),                           // beta = 0.0
		(*C.double)(unsafe.Pointer(&result.Data[0])), // C
		C.int(p), // ldc
	)

	return result, nil
}

// MatMulTransposeBBLAS выполняет A * B^T используя BLAS
func MatMulTransposeBBLAS(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("умножение матриц требует 2D тензоры")
	}

	m := a.Shape[0]
	n := a.Shape[1]
	p := b.Shape[0] // B транспонирована

	if n != b.Shape[1] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц с транспонированием")
	}

	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// B транспонирована
	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans, // A не транспонирована
		C.CblasTrans,   // B транспонирована
		C.int(m),
		C.int(p),
		C.int(n),
		C.double(1.0),
		(*C.double)(unsafe.Pointer(&a.Data[0])),
		C.int(n),
		(*C.double)(unsafe.Pointer(&b.Data[0])),
		C.int(n), // ldb для транспонированной матрицы
		C.double(0.0),
		(*C.double)(unsafe.Pointer(&result.Data[0])),
		C.int(p),
	)

	return result, nil
}

// MatMulTransposeABLAS выполняет A^T * B используя BLAS
func MatMulTransposeABLAS(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("умножение матриц требует 2D тензоры")
	}

	m := a.Shape[1] // A транспонирована
	n := a.Shape[0]
	p := b.Shape[1]

	if n != b.Shape[0] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц с транспонированием")
	}

	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// A транспонирована
	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasTrans,   // A транспонирована
		C.CblasNoTrans, // B не транспонирована
		C.int(m),
		C.int(p),
		C.int(n),
		C.double(1.0),
		(*C.double)(unsafe.Pointer(&a.Data[0])),
		C.int(m), // lda для транспонированной матрицы
		(*C.double)(unsafe.Pointer(&b.Data[0])),
		C.int(p),
		C.double(0.0),
		(*C.double)(unsafe.Pointer(&result.Data[0])),
		C.int(p),
	)

	return result, nil
}

// VectorAddBLAS выполняет векторное сложение: y = alpha*x + y используя BLAS
// cblas_daxpy (Double precision Alpha X Plus Y)
func VectorAddBLAS(alpha float64, x, y []float64) {
	if len(x) != len(y) {
		panic("векторы должны быть одинаковой длины")
	}

	n := len(x)
	C.cblas_daxpy(
		C.int(n),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&x[0])),
		C.int(1), // stride x
		(*C.double)(unsafe.Pointer(&y[0])),
		C.int(1), // stride y
	)
}

// VectorScaleBLAS масштабирует вектор: x = alpha*x используя BLAS
// cblas_dscal (Double precision SCALe)
func VectorScaleBLAS(alpha float64, x []float64) {
	n := len(x)
	C.cblas_dscal(
		C.int(n),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&x[0])),
		C.int(1), // stride
	)
}

// DotProductBLAS вычисляет скалярное произведение используя BLAS
// cblas_ddot (Double precision DOT product)
func DotProductBLAS(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("векторы должны быть одинаковой длины")
	}

	n := len(x)
	result := C.cblas_ddot(
		C.int(n),
		(*C.double)(unsafe.Pointer(&x[0])),
		C.int(1), // stride x
		(*C.double)(unsafe.Pointer(&y[0])),
		C.int(1), // stride y
	)

	return float64(result)
}

// MatrixVectorMultiplyBLAS выполняет умножение матрицы на вектор: y = alpha*A*x + beta*y
// cblas_dgemv (Double precision GEneral Matrix-Vector multiply)
func MatrixVectorMultiplyBLAS(alpha float64, a *Tensor, x []float64, beta float64, y []float64) error {
	if len(a.Shape) != 2 {
		return fmt.Errorf("требуется 2D матрица")
	}

	m := a.Shape[0]
	n := a.Shape[1]

	if len(x) != n || len(y) != m {
		return fmt.Errorf("несовместимые размеры векторов")
	}

	C.cblas_dgemv(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.int(m),
		C.int(n),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&a.Data[0])),
		C.int(n), // lda
		(*C.double)(unsafe.Pointer(&x[0])),
		C.int(1), // stride x
		C.double(beta),
		(*C.double)(unsafe.Pointer(&y[0])),
		C.int(1), // stride y
	)

	return nil
}
