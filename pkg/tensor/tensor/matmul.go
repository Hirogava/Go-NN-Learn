package tensor

import (
	"fmt"
	"sync"
)

// Размер блока для блочного умножения матриц (оптимален для кеша L1)
const (
	BlockSize = 64
	// Порог для использования параллельного умножения
	ParallelThreshold = 128
)

// MatMul выполняет умножение матриц (тензоров 2D).
// Для матриц A[m,n] и B[n,p] возвращает C[m,p], где C = A * B
// Использует оптимизированный алгоритм с блочным умножением для лучшей локальности кеша.
func MatMul(a, b *Tensor) (*Tensor, error) {
	// Проверка размерностей
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors, got %dD and %dD", len(a.Shape), len(b.Shape))
	}

	m := a.Shape[0] // строки A
	n := a.Shape[1] // столбцы A = строки B
	p := b.Shape[1] // столбцы B

	if n != b.Shape[0] {
		return nil, fmt.Errorf("incompatible shapes for matmul: [%d,%d] and [%d,%d]", m, n, b.Shape[0], p)
	}

	// Создаем результирующий тензор
	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// Выбираем оптимальный алгоритм в зависимости от размера
	if m >= ParallelThreshold || p >= ParallelThreshold {
		// Параллельное блочное умножение для больших матриц
		matmulParallelBlocked(a.Data, b.Data, result.Data, m, n, p)
	} else if m >= BlockSize || p >= BlockSize {
		// Блочное умножение для средних матриц
		matmulBlocked(a.Data, b.Data, result.Data, m, n, p)
	} else {
		// Простое умножение для малых матриц
		matmulNaive(a.Data, b.Data, result.Data, m, n, p)
	}

	return result, nil
}

// matmulNaive - простое умножение матриц (для малых матриц)
func matmulNaive(a, b, c []float64, m, n, p int) {
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += a[i*n+k] * b[k*p+j]
			}
			c[i*p+j] = sum
		}
	}
}

// matmulBlocked - блочное умножение матриц (кеш-оптимизированное)
func matmulBlocked(a, b, c []float64, m, n, p int) {
	// Инициализируем результат нулями
	for i := range c {
		c[i] = 0.0
	}

	// Блочное умножение для оптимизации использования кеша
	for ii := 0; ii < m; ii += BlockSize {
		iEnd := min(ii+BlockSize, m)
		for kk := 0; kk < n; kk += BlockSize {
			kEnd := min(kk+BlockSize, n)
			for jj := 0; jj < p; jj += BlockSize {
				jEnd := min(jj+BlockSize, p)

				// Умножение блоков
				for i := ii; i < iEnd; i++ {
					for k := kk; k < kEnd; k++ {
						aik := a[i*n+k]
						for j := jj; j < jEnd; j++ {
							c[i*p+j] += aik * b[k*p+j]
						}
					}
				}
			}
		}
	}
}

// matmulParallelBlocked - параллельное блочное умножение матриц
func matmulParallelBlocked(a, b, c []float64, m, n, p int) {
	// Инициализируем результат нулями
	for i := range c {
		c[i] = 0.0
	}

	// Используем worker pool для параллельной обработки строк
	numWorkers := 4 // Можно настроить под количество ядер
	rowsPerWorker := (m + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := min((w+1)*rowsPerWorker, m)

		go func(start, end int) {
			defer wg.Done()

			// Блочное умножение для диапазона строк
			for ii := start; ii < end; ii += BlockSize {
				iEnd := min(ii+BlockSize, end)
				for kk := 0; kk < n; kk += BlockSize {
					kEnd := min(kk+BlockSize, n)
					for jj := 0; jj < p; jj += BlockSize {
						jEnd := min(jj+BlockSize, p)

						// Умножение блоков
						for i := ii; i < iEnd; i++ {
							for k := kk; k < kEnd; k++ {
								aik := a[i*n+k]
								for j := jj; j < jEnd; j++ {
									c[i*p+j] += aik * b[k*p+j]
								}
							}
						}
					}
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
}

// MatMulTransposeB - умножение A * B^T (оптимизированная версия)
// Полезно для градиентов в обратном распространении
func MatMulTransposeB(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors")
	}

	m := a.Shape[0]
	n := a.Shape[1]
	p := b.Shape[0] // B транспонирована, поэтому берем строки B

	if n != b.Shape[1] {
		return nil, fmt.Errorf("incompatible shapes for matmul with transpose")
	}

	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// Оптимизированное умножение A * B^T
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				// B транспонирована, поэтому обращаемся как B[j,k]
				sum += a.Data[i*n+k] * b.Data[j*n+k]
			}
			result.Data[i*p+j] = sum
		}
	}

	return result, nil
}

// MatMulTransposeA - умножение A^T * B (оптимизированная версия)
func MatMulTransposeA(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors")
	}

	m := a.Shape[1] // A транспонирована
	n := a.Shape[0]
	p := b.Shape[1]

	if n != b.Shape[0] {
		return nil, fmt.Errorf("incompatible shapes for matmul with transpose")
	}

	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// Оптимизированное умножение A^T * B
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				// A транспонирована, поэтому обращаемся как A[k,i]
				sum += a.Data[k*m+i] * b.Data[k*p+j]
			}
			result.Data[i*p+j] = sum
		}
	}

	return result, nil
}

// min возвращает минимум из двух чисел
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
