package tensor

import (
	"fmt"
	"runtime"
	"sync"
)

// Размер блока для блочного умножения матриц (оптимален для кеша L1)
const (
	BlockSize = 64
	// Порог для использования параллельного умножения
	ParallelThreshold = 128
	// Размер микро-ядра для внутреннего цикла (оптимизация регистров)
	MicroKernelSize = 4
)

// MatMul выполняет умножение матриц (тензоров 2D).
// Для матриц A[m,n] и B[n,p] возвращает C[m,p], где C = A * B
// Использует оптимизированный алгоритм с блочным умножением для лучшей локальности кеша.
func MatMul(a, b *Tensor) (*Tensor, error) {
	// Проверка размерностей
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("умножение матриц требует 2D тензоры, получены %dD и %dD", len(a.Shape), len(b.Shape))
	}

	m := a.Shape[0] // строки A
	n := a.Shape[1] // столбцы A = строки B
	p := b.Shape[1] // столбцы B

	if n != b.Shape[0] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц: [%d,%d] и [%d,%d]", m, n, b.Shape[0], p)
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
		// Оптимизированное умножение для малых матриц
		matmulOptimized(a.Data, b.Data, result.Data, m, n, p)
	}

	return result, nil
}

// matmulOptimized - оптимизированное умножение для малых матриц
// Использует переупорядочивание циклов ikj для лучшей локальности кеша
func matmulOptimized(a, b, c []float64, m, n, p int) {
	// Обнуляем результат
	for i := range c {
		c[i] = 0.0
	}

	// Цикл ikj - лучшая локальность кеша для row-major матриц
	for i := 0; i < m; i++ {
		iOffset := i * n
		cOffset := i * p
		for k := 0; k < n; k++ {
			aik := a[iOffset+k]
			bOffset := k * p
			// Векторизация внутреннего цикла
			j := 0
			// Развертка x4 для векторизации
			for ; j <= p-MicroKernelSize; j += MicroKernelSize {
				c[cOffset+j] += aik * b[bOffset+j]
				c[cOffset+j+1] += aik * b[bOffset+j+1]
				c[cOffset+j+2] += aik * b[bOffset+j+2]
				c[cOffset+j+3] += aik * b[bOffset+j+3]
			}
			// Остаток
			for ; j < p; j++ {
				c[cOffset+j] += aik * b[bOffset+j]
			}
		}
	}
}

// matmulBlocked - блочное умножение матриц (кеш-оптимизированное)
// Использует 3-уровневую блочную структуру для оптимизации кеша
func matmulBlocked(a, b, c []float64, m, n, p int) {
	// Инициализируем результат нулями
	for i := range c {
		c[i] = 0.0
	}

	// Блочное умножение: разбиваем на блоки для L1 кеша
	// Порядок циклов: kk-ii-jj для минимизации промахов кеша
	for kk := 0; kk < n; kk += BlockSize {
		kEnd := min(kk+BlockSize, n)
		for ii := 0; ii < m; ii += BlockSize {
			iEnd := min(ii+BlockSize, m)
			for jj := 0; jj < p; jj += BlockSize {
				jEnd := min(jj+BlockSize, p)

				// Микро-ядро: умножение блоков
				matmulMicroKernel(a, b, c, ii, iEnd, kk, kEnd, jj, jEnd, n, p)
			}
		}
	}
}

// matmulMicroKernel - микро-ядро умножения блоков
// Оптимизировано для регистров процессора
func matmulMicroKernel(a, b, c []float64, iStart, iEnd, kStart, kEnd, jStart, jEnd, n, p int) {
	for i := iStart; i < iEnd; i++ {
		iOffsetA := i * n
		iOffsetC := i * p
		for k := kStart; k < kEnd; k++ {
			aik := a[iOffsetA+k]
			kOffset := k * p
			j := jStart
			// Развертка внутреннего цикла для векторизации
			for ; j <= jEnd-MicroKernelSize; j += MicroKernelSize {
				c[iOffsetC+j] += aik * b[kOffset+j]
				c[iOffsetC+j+1] += aik * b[kOffset+j+1]
				c[iOffsetC+j+2] += aik * b[kOffset+j+2]
				c[iOffsetC+j+3] += aik * b[kOffset+j+3]
			}
			for ; j < jEnd; j++ {
				c[iOffsetC+j] += aik * b[kOffset+j]
			}
		}
	}
}

// matmulParallelBlocked - параллельное блочное умножение матриц
// Использует worker pool для параллелизма на уровне строк
func matmulParallelBlocked(a, b, c []float64, m, n, p int) {
	// Инициализируем результат нулями
	for i := range c {
		c[i] = 0.0
	}

	// Определяем количество воркеров (по числу ядер)
	numWorkers := runtime.NumCPU()
	if numWorkers > m {
		numWorkers = m
	}

	// Разбиваем работу на блоки строк
	blockRows := (m + numWorkers - 1) / numWorkers
	if blockRows < BlockSize {
		blockRows = BlockSize
	}

	var wg sync.WaitGroup

	// Запускаем воркеры
	for w := 0; w < numWorkers; w++ {
		startRow := w * blockRows
		if startRow >= m {
			break
		}
		endRow := min((w+1)*blockRows, m)

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			// Блочное умножение для диапазона строк
			matmulBlockedRange(a, b, c, start, end, n, p)
		}(startRow, endRow)
	}

	wg.Wait()
}

// matmulBlockedRange - блочное умножение для диапазона строк
func matmulBlockedRange(a, b, c []float64, rowStart, rowEnd, n, p int) {
	// Блочное умножение для заданного диапазона строк
	for kk := 0; kk < n; kk += BlockSize {
		kEnd := min(kk+BlockSize, n)
		for ii := rowStart; ii < rowEnd; ii += BlockSize {
			iEnd := min(ii+BlockSize, rowEnd)
			for jj := 0; jj < p; jj += BlockSize {
				jEnd := min(jj+BlockSize, p)
				// Микро-ядро
				matmulMicroKernel(a, b, c, ii, iEnd, kk, kEnd, jj, jEnd, n, p)
			}
		}
	}
}

// MatMulTransposeB - умножение A * B^T (оптимизированная версия)
// Полезно для градиентов в обратном распространении
// Использует dot product для лучшей производительности
func MatMulTransposeB(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("умножение матриц требует 2D тензоры")
	}

	m := a.Shape[0]
	n := a.Shape[1]
	p := b.Shape[0] // B транспонирована, поэтому берем строки B

	if n != b.Shape[1] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц с транспонированием")
	}

	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// Оптимизированное умножение A * B^T через dot product
	for i := 0; i < m; i++ {
		iOffsetA := i * n
		iOffsetC := i * p
		for j := 0; j < p; j++ {
			jOffsetB := j * n
			sum := 0.0
			k := 0
			// Развертка для векторизации
			for ; k <= n-MicroKernelSize; k += MicroKernelSize {
				sum += a.Data[iOffsetA+k] * b.Data[jOffsetB+k]
				sum += a.Data[iOffsetA+k+1] * b.Data[jOffsetB+k+1]
				sum += a.Data[iOffsetA+k+2] * b.Data[jOffsetB+k+2]
				sum += a.Data[iOffsetA+k+3] * b.Data[jOffsetB+k+3]
			}
			for ; k < n; k++ {
				sum += a.Data[iOffsetA+k] * b.Data[jOffsetB+k]
			}
			result.Data[iOffsetC+j] = sum
		}
	}

	return result, nil
}

// MatMulTransposeA - умножение A^T * B (оптимизированная версия)
func MatMulTransposeA(a, b *Tensor) (*Tensor, error) {
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

	// Обнуляем результат
	for i := range result.Data {
		result.Data[i] = 0.0
	}

	// Оптимизированное умножение A^T * B
	// Используем порядок циклов для лучшей локальности
	for k := 0; k < n; k++ {
		kOffsetA := k * m
		kOffsetB := k * p
		for i := 0; i < m; i++ {
			aki := a.Data[kOffsetA+i]
			iOffsetC := i * p
			j := 0
			// Векторизация
			for ; j <= p-MicroKernelSize; j += MicroKernelSize {
				result.Data[iOffsetC+j] += aki * b.Data[kOffsetB+j]
				result.Data[iOffsetC+j+1] += aki * b.Data[kOffsetB+j+1]
				result.Data[iOffsetC+j+2] += aki * b.Data[kOffsetB+j+2]
				result.Data[iOffsetC+j+3] += aki * b.Data[kOffsetB+j+3]
			}
			for ; j < p; j++ {
				result.Data[iOffsetC+j] += aki * b.Data[kOffsetB+j]
			}
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
