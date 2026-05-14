package tensor

import (
	"fmt"
	"sync"
)

// Размер блока для блочного умножения матриц (оптимален для кеша L1)
const (
	BlockSize = 64
	// Альтернативные размеры блоков для адаптивного выбора
	BlockSizeSmall  = 32 // Для L1 кеша (малые матрицы)
	BlockSizeMedium = 64 // Для L1/L2 кеша (средние матрицы)
	// Порог для использования параллельного умножения
	ParallelThreshold = 128
	// Порог для использования BLAS (если доступен)
	BLASThreshold = 512
	// Размер микро-ядра для внутреннего цикла (оптимизация регистров)
	MicroKernelSize = 4
)

// MatMul выполняет умножение матриц (тензоров 2D).
// Для матриц A[m,n] и B[n,p] возвращает C[m,p], где C = A * B
// Использует оптимизированный алгоритм с блочным умножением для лучшей локальности кеша.
// Автоматически выбирает наилучшую реализацию (BLAS, SIMD, блочное умножение).
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

	// Для очень больших матриц используем BLAS (если доступен)
	if BLASAvailable && (m >= BLASThreshold || p >= BLASThreshold || n >= BLASThreshold) {
		return MatMulBLAS(a, b)
	}

	// Создаем результирующий тензор
	result := &Tensor{
		Data:    make([]float64, m*p),
		Shape:   []int{m, p},
		Strides: []int{p, 1},
	}

	// Адаптивный выбор алгоритма на основе размера матриц
	matrixSize := m * n * p

	if m >= ParallelThreshold || p >= ParallelThreshold {
		// Tiled MatMul v2 для больших матриц: worker по строкам + упаковка плиток B
		blockSize := chooseBlockSize(m, n, p)
		matmulParallelBlockedV2(a.Data, b.Data, result.Data, m, n, p, blockSize)
	} else if m >= BlockSizeSmall || p >= BlockSizeSmall {
		// Cache-blocked MatMul v2 для средних матриц
		blockSize := chooseBlockSize(m, n, p)
		matmulBlockedV2(a.Data, b.Data, result.Data, m, n, p, blockSize)
	} else if matrixSize < 1000 {
		// Простое оптимизированное умножение для очень малых матриц
		matmulOptimized(a.Data, b.Data, result.Data, m, n, p)
	} else {
		// Cache-blocked MatMul v2
		blockSize := chooseBlockSize(m, n, p)
		matmulBlockedV2(a.Data, b.Data, result.Data, m, n, p, blockSize)
	}

	return result, nil
}

// chooseBlockSize выбирает оптимальный размер блока на основе размеров матриц
// Использует эвристику на основе размеров кеша процессора
func chooseBlockSize(m, n, p int) int {
	// Оцениваем размер данных, которые будут в кеше
	// Для блока размером B нам нужно ~B*B*3*8 байт (3 матрицы, float64)

	maxDim := m
	if n > maxDim {
		maxDim = n
	}
	if p > maxDim {
		maxDim = p
	}

	// Эвристика выбора размера блока
	if maxDim <= 256 {
		return BlockSizeSmall
	}
	return BlockSizeMedium
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

// matmulBlocked - baseline версия блочного умножения.
// Используется как эталон для benchmark'ов MatMul v2.
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

// matmulBlockedV2 - cache-blocked MatMul с упаковкой плитки B в транспонированный буфер.
// Упаковка устраняет strided access по B и делает внутренний цикл плотным по памяти.
func matmulBlockedV2(a, b, c []float64, m, n, p int, blockSize int) {
	for i := range c {
		c[i] = 0.0
	}

	packedB := make([]float64, blockSize*blockSize)
	for jj := 0; jj < p; jj += blockSize {
		jSize := min(blockSize, p-jj)
		for kk := 0; kk < n; kk += blockSize {
			kSize := min(blockSize, n-kk)
			packBTileTransposed(b, packedB[:jSize*kSize], kk, jj, kSize, jSize, p)

			for ii := 0; ii < m; ii += blockSize {
				iEnd := min(ii+blockSize, m)
				matmulKernelPackedB(a, c, packedB[:jSize*kSize], ii, iEnd, kk, kSize, jj, jSize, n, p)
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

	numWorkers := matmulWorkerCount(m, BlockSize)
	blockRows := matmulChunkRows(m, numWorkers, BlockSize)
	scheduler := newMatmulRowScheduler(m, blockRows)

	var wg sync.WaitGroup

	// Запускаем воркеры
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				start, end, ok := scheduler.next()
				if !ok {
					return
				}
				// Блочное умножение для диапазона строк
				matmulBlockedRange(a, b, c, start, end, n, p)
			}
		}()
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

func packBTileTransposed(b, packed []float64, kk, jj, kSize, jSize, p int) {
	for j := 0; j < jSize; j++ {
		dst := j * kSize
		srcCol := jj + j
		for k := 0; k < kSize; k++ {
			packed[dst+k] = b[(kk+k)*p+srcCol]
		}
	}
}

// matmulBlockedSIMD - историческая SIMD-реализация tiled matmul.
// Сохраняется как отдельный путь для прямого вызова и обратной совместимости.
// Основной MatMul при этом использует v2 и не зависит от этой функции.
func matmulBlockedSIMD(a, b, c []float64, m, n, p int, blockSize int) {
	for i := range c {
		c[i] = 0.0
	}

	for kk := 0; kk < n; kk += blockSize {
		kEnd := min(kk+blockSize, n)
		for ii := 0; ii < m; ii += blockSize {
			iEnd := min(ii+blockSize, m)
			for jj := 0; jj < p; jj += blockSize {
				jEnd := min(jj+blockSize, p)
				MatMulSIMDKernel(a, b, c, m, n, p, ii, iEnd, kk, kEnd, jj, jEnd)
			}
		}
	}
}

func matmulKernelPackedB(a, c, packedB []float64, iStart, iEnd, kk, kSize, jj, jSize, n, p int) {
	for i := iStart; i < iEnd; i++ {
		aRow := a[i*n+kk : i*n+kk+kSize]
		cRow := c[i*p+jj : i*p+jj+jSize]

		j := 0
		for ; j <= jSize-4; j += 4 {
			b0 := packedB[(j+0)*kSize : (j+1)*kSize]
			b1 := packedB[(j+1)*kSize : (j+2)*kSize]
			b2 := packedB[(j+2)*kSize : (j+3)*kSize]
			b3 := packedB[(j+3)*kSize : (j+4)*kSize]

			sum0 := cRow[j+0]
			sum1 := cRow[j+1]
			sum2 := cRow[j+2]
			sum3 := cRow[j+3]

			k := 0
			for ; k <= kSize-4; k += 4 {
				a0 := aRow[k+0]
				a1 := aRow[k+1]
				a2 := aRow[k+2]
				a3 := aRow[k+3]

				sum0 += a0*b0[k+0] + a1*b0[k+1] + a2*b0[k+2] + a3*b0[k+3]
				sum1 += a0*b1[k+0] + a1*b1[k+1] + a2*b1[k+2] + a3*b1[k+3]
				sum2 += a0*b2[k+0] + a1*b2[k+1] + a2*b2[k+2] + a3*b2[k+3]
				sum3 += a0*b3[k+0] + a1*b3[k+1] + a2*b3[k+2] + a3*b3[k+3]
			}
			for ; k < kSize; k++ {
				av := aRow[k]
				sum0 += av * b0[k]
				sum1 += av * b1[k]
				sum2 += av * b2[k]
				sum3 += av * b3[k]
			}

			cRow[j+0] = sum0
			cRow[j+1] = sum1
			cRow[j+2] = sum2
			cRow[j+3] = sum3
		}

		for ; j < jSize; j++ {
			sum := cRow[j]
			bCol := packedB[j*kSize : (j+1)*kSize]
			for k := 0; k < kSize; k++ {
				sum += aRow[k] * bCol[k]
			}
			cRow[j] = sum
		}
	}
}

// matmulParallelBlockedV2 - параллельный tiled MatMul v2.
// Каждый worker владеет диапазоном строк C и собственной упакованной плиткой B.
func matmulParallelBlockedV2(a, b, c []float64, m, n, p int, blockSize int) {
	for i := range c {
		c[i] = 0.0
	}

	numWorkers := matmulWorkerCount(m, blockSize)
	blockRows := matmulChunkRows(m, numWorkers, blockSize)
	scheduler := newMatmulRowScheduler(m, blockRows)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			packedB := make([]float64, blockSize*blockSize)
			for {
				start, end, ok := scheduler.next()
				if !ok {
					return
				}

				for jj := 0; jj < p; jj += blockSize {
					jSize := min(blockSize, p-jj)
					for kk := 0; kk < n; kk += blockSize {
						kSize := min(blockSize, n-kk)
						packBTileTransposed(b, packedB[:jSize*kSize], kk, jj, kSize, jSize, p)
						for ii := start; ii < end; ii += blockSize {
							iEnd := min(ii+blockSize, end)
							matmulKernelPackedB(a, c, packedB[:jSize*kSize], ii, iEnd, kk, kSize, jj, jSize, n, p)
						}
					}
				}
			}
		}()
	}

	wg.Wait()
}
