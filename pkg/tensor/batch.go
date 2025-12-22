package tensor

import (
	"fmt"
	"sync"
)

// BatchOps содержит пакетные операции для обработки множества тензоров

// BatchMatMul выполняет умножение матриц для батча
// Для батча размера [batch, m, n] * [batch, n, p] возвращает [batch, m, p]
// Каждая матрица в батче умножается независимо
func BatchMatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 3 || len(b.Shape) != 3 {
		return nil, fmt.Errorf("батчевое умножение матриц требует 3D тензоры (batch, m, n), получены %dD и %dD", len(a.Shape), len(b.Shape))
	}

	batchSize := a.Shape[0]
	m := a.Shape[1]
	n := a.Shape[2]
	p := b.Shape[2]

	if a.Shape[0] != b.Shape[0] {
		return nil, fmt.Errorf("размеры батчей должны совпадать: %d != %d", a.Shape[0], b.Shape[0])
	}
	if n != b.Shape[1] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц: [%d,%d] и [%d,%d]", m, n, b.Shape[1], p)
	}

	// Создаем результирующий тензор
	result := &Tensor{
		Data:    make([]float64, batchSize*m*p),
		Shape:   []int{batchSize, m, p},
		Strides: []int{m * p, p, 1},
	}

	// Параллельная обработка батча
	var wg sync.WaitGroup
	numWorkers := min(batchSize, 8) // Ограничиваем количество воркеров

	batchPerWorker := (batchSize + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startBatch := w * batchPerWorker
		if startBatch >= batchSize {
			break
		}
		endBatch := min((w+1)*batchPerWorker, batchSize)

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			for batchIdx := start; batchIdx < end; batchIdx++ {
				// Извлекаем срезы для текущего батча
				aOffset := batchIdx * m * n
				bOffset := batchIdx * n * p
				cOffset := batchIdx * m * p

				aSlice := a.Data[aOffset : aOffset+m*n]
				bSlice := b.Data[bOffset : bOffset+n*p]
				cSlice := result.Data[cOffset : cOffset+m*p]

				// Выполняем умножение матриц для этого элемента батча
				if m >= ParallelThreshold || p >= ParallelThreshold {
					matmulBlocked(aSlice, bSlice, cSlice, m, n, p)
				} else if m >= BlockSize || p >= BlockSize {
					matmulBlocked(aSlice, bSlice, cSlice, m, n, p)
				} else {
					matmulOptimized(aSlice, bSlice, cSlice, m, n, p)
				}
			}
		}(startBatch, endBatch)
	}

	wg.Wait()
	return result, nil
}

// BatchAdd выполняет поэлементное сложение батча тензоров
func BatchAdd(tensors []*Tensor) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("пустой батч тензоров")
	}

	baseShape := tensors[0].Shape
	size := len(tensors[0].Data)

	// Проверяем что все тензоры одного размера
	for i, t := range tensors {
		if !shapesEqual(t.Shape, baseShape) {
			return nil, fmt.Errorf("тензор %d имеет другую форму: %v != %v", i, t.Shape, baseShape)
		}
	}

	result := &Tensor{
		Data:    make([]float64, size),
		Shape:   append([]int{}, baseShape...),
		Strides: append([]int{}, tensors[0].Strides...),
	}

	// Параллельное сложение
	numWorkers := 4
	chunkSize := (size + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		if start >= size {
			break
		}
		end := min((w+1)*chunkSize, size)

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				sum := 0.0
				for _, t := range tensors {
					sum += t.Data[i]
				}
				result.Data[i] = sum
			}
		}(start, end)
	}

	wg.Wait()
	return result, nil
}

// BatchScale масштабирует каждый тензор в батче своим коэффициентом
func BatchScale(tensors []*Tensor, scales []float64) error {
	if len(tensors) != len(scales) {
		return fmt.Errorf("количество тензоров и коэффициентов должно совпадать: %d != %d", len(tensors), len(scales))
	}

	var wg sync.WaitGroup
	for i, t := range tensors {
		wg.Add(1)
		go func(tensor *Tensor, scale float64) {
			defer wg.Done()
			ScaleInPlace(scale, tensor)
		}(t, scales[i])
	}

	wg.Wait()
	return nil
}

// BatchNorm выполняет батч-нормализацию
// Вычисляет среднее и дисперсию по батчу и нормализует
func BatchNorm(batch *Tensor, eps float64) (*Tensor, error) {
	if len(batch.Shape) < 2 {
		return nil, fmt.Errorf("батч-нормализация требует тензор размерности не менее 2D")
	}

	batchSize := batch.Shape[0]
	featureSize := 1
	for i := 1; i < len(batch.Shape); i++ {
		featureSize *= batch.Shape[i]
	}

	// Вычисляем среднее по батчу для каждого признака
	mean := make([]float64, featureSize)
	for b := 0; b < batchSize; b++ {
		offset := b * featureSize
		for f := 0; f < featureSize; f++ {
			mean[f] += batch.Data[offset+f]
		}
	}
	for f := range mean {
		mean[f] /= float64(batchSize)
	}

	// Вычисляем дисперсию
	variance := make([]float64, featureSize)
	for b := 0; b < batchSize; b++ {
		offset := b * featureSize
		for f := 0; f < featureSize; f++ {
			diff := batch.Data[offset+f] - mean[f]
			variance[f] += diff * diff
		}
	}
	for f := range variance {
		variance[f] /= float64(batchSize)
	}

	// Нормализация
	result := &Tensor{
		Data:    make([]float64, len(batch.Data)),
		Shape:   append([]int{}, batch.Shape...),
		Strides: append([]int{}, batch.Strides...),
	}

	for b := 0; b < batchSize; b++ {
		offset := b * featureSize
		for f := 0; f < featureSize; f++ {
			idx := offset + f
			// (x - mean) / sqrt(var + eps)
			result.Data[idx] = (batch.Data[idx] - mean[f]) / (sqrt(variance[f] + eps))
		}
	}

	return result, nil
}

// sqrt вычисляет квадратный корень
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	// Метод Ньютона для вычисления корня
	if x == 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// BatchMean вычисляет среднее значение по батчу
func BatchMean(batch *Tensor, axis int) (*Tensor, error) {
	if axis < 0 || axis >= len(batch.Shape) {
		return nil, fmt.Errorf("некорректная ось %d для тензора с %d измерениями", axis, len(batch.Shape))
	}

	// Создаем новую форму без указанной оси
	newShape := make([]int, 0, len(batch.Shape)-1)
	for i, dim := range batch.Shape {
		if i != axis {
			newShape = append(newShape, dim)
		}
	}

	if len(newShape) == 0 {
		// Скалярный результат
		newShape = []int{1}
	}

	resultSize := 1
	for _, dim := range newShape {
		resultSize *= dim
	}

	result := &Tensor{
		Data:  make([]float64, resultSize),
		Shape: newShape,
	}

	// Вычисляем strides для результата
	result.Strides = make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		result.Strides[i] = stride
		stride *= newShape[i]
	}

	// Вычисляем среднее
	divisor := float64(batch.Shape[axis])

	// Простая реализация для axis=0 (батч)
	if axis == 0 {
		batchSize := batch.Shape[0]
		featureSize := len(batch.Data) / batchSize

		for f := 0; f < featureSize; f++ {
			sum := 0.0
			for b := 0; b < batchSize; b++ {
				sum += batch.Data[b*featureSize+f]
			}
			result.Data[f] = sum / divisor
		}
	} else {
		// Для других осей - более общая реализация
		// TODO: реализовать для произвольных осей
		return nil, fmt.Errorf("среднее вдоль оси %d еще не реализовано", axis)
	}

	return result, nil
}

// BatchMatMulSIMD выполняет батчевое умножение матриц с SIMD оптимизациями
func BatchMatMulSIMD(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 3 || len(b.Shape) != 3 {
		return nil, fmt.Errorf("батчевое умножение матриц требует 3D тензоры")
	}

	batchSize := a.Shape[0]
	m := a.Shape[1]
	n := a.Shape[2]
	p := b.Shape[2]

	if a.Shape[0] != b.Shape[0] || n != b.Shape[1] {
		return nil, fmt.Errorf("несовместимые формы для батчевого умножения")
	}

	result := &Tensor{
		Data:    make([]float64, batchSize*m*p),
		Shape:   []int{batchSize, m, p},
		Strides: []int{m * p, p, 1},
	}

	// Параллельная обработка с SIMD
	var wg sync.WaitGroup
	numWorkers := min(batchSize, 8)
	batchPerWorker := (batchSize + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startBatch := w * batchPerWorker
		if startBatch >= batchSize {
			break
		}
		endBatch := min((w+1)*batchPerWorker, batchSize)

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			for batchIdx := start; batchIdx < end; batchIdx++ {
				aOffset := batchIdx * m * n
				bOffset := batchIdx * n * p
				cOffset := batchIdx * m * p

				aSlice := a.Data[aOffset : aOffset+m*n]
				bSlice := b.Data[bOffset : bOffset+n*p]
				cSlice := result.Data[cOffset : cOffset+m*p]

				// Используем SIMD-оптимизированное умножение
				blockSize := chooseBlockSize(m, n, p)
				matmulBlockedSIMD(aSlice, bSlice, cSlice, m, n, p, blockSize)
			}
		}(startBatch, endBatch)
	}

	wg.Wait()
	return result, nil
}

// BatchConcat объединяет список тензоров вдоль указанной оси
func BatchConcat(tensors []*Tensor, axis int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("пустой список тензоров")
	}

	// Проверяем что все тензоры имеют одинаковое количество измерений
	ndim := len(tensors[0].Shape)
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("некорректная ось %d для тензоров с %d измерениями", axis, ndim)
	}

	// Проверяем совместимость форм
	for i, t := range tensors {
		if len(t.Shape) != ndim {
			return nil, fmt.Errorf("тензор %d имеет %d измерений, ожидалось %d", i, len(t.Shape), ndim)
		}
		for j := 0; j < ndim; j++ {
			if j != axis && t.Shape[j] != tensors[0].Shape[j] {
				return nil, fmt.Errorf("несовместимые формы тензоров вдоль оси %d", j)
			}
		}
	}

	// Вычисляем форму результата
	newShape := make([]int, ndim)
	copy(newShape, tensors[0].Shape)
	for _, t := range tensors[1:] {
		newShape[axis] += t.Shape[axis]
	}

	resultSize := 1
	for _, dim := range newShape {
		resultSize *= dim
	}

	result := &Tensor{
		Data:  make([]float64, resultSize),
		Shape: newShape,
	}

	// Вычисляем strides
	result.Strides = make([]int, ndim)
	stride := 1
	for i := ndim - 1; i >= 0; i-- {
		result.Strides[i] = stride
		stride *= newShape[i]
	}

	// Копируем данные
	// Для простоты делаем для axis=0
	if axis == 0 {
		offset := 0
		for _, t := range tensors {
			copy(result.Data[offset:], t.Data)
			offset += len(t.Data)
		}
	} else {
		// Для других осей требуется более сложная логика
		return nil, fmt.Errorf("конкатенация вдоль оси %d еще не реализована", axis)
	}

	return result, nil
}

// BatchSplit разделяет тензор на список тензоров вдоль указанной оси
func BatchSplit(tensor *Tensor, numSplits int, axis int) ([]*Tensor, error) {
	if axis < 0 || axis >= len(tensor.Shape) {
		return nil, fmt.Errorf("некорректная ось %d", axis)
	}

	if tensor.Shape[axis]%numSplits != 0 {
		return nil, fmt.Errorf("размер %d вдоль оси %d не делится на %d", tensor.Shape[axis], axis, numSplits)
	}

	splitSize := tensor.Shape[axis] / numSplits
	result := make([]*Tensor, numSplits)

	// Для простоты делаем для axis=0
	if axis == 0 {
		chunkSize := len(tensor.Data) / numSplits
		for i := 0; i < numSplits; i++ {
			newShape := make([]int, len(tensor.Shape))
			copy(newShape, tensor.Shape)
			newShape[axis] = splitSize

			result[i] = &Tensor{
				Data:  make([]float64, chunkSize),
				Shape: newShape,
			}

			// Вычисляем strides
			result[i].Strides = make([]int, len(newShape))
			stride := 1
			for j := len(newShape) - 1; j >= 0; j-- {
				result[i].Strides[j] = stride
				stride *= newShape[j]
			}

			// Копируем данные
			copy(result[i].Data, tensor.Data[i*chunkSize:(i+1)*chunkSize])
		}
	} else {
		return nil, fmt.Errorf("разделение вдоль оси %d еще не реализовано", axis)
	}

	return result, nil
}

// BatchGather собирает элементы из тензора по индексам
func BatchGather(tensor *Tensor, indices []int, axis int) (*Tensor, error) {
	if axis < 0 || axis >= len(tensor.Shape) {
		return nil, fmt.Errorf("некорректная ось %d", axis)
	}

	// Проверяем индексы
	for _, idx := range indices {
		if idx < 0 || idx >= tensor.Shape[axis] {
			return nil, fmt.Errorf("индекс %d вне диапазона [0, %d)", idx, tensor.Shape[axis])
		}
	}

	// Создаем форму результата
	newShape := make([]int, len(tensor.Shape))
	copy(newShape, tensor.Shape)
	newShape[axis] = len(indices)

	resultSize := 1
	for _, dim := range newShape {
		resultSize *= dim
	}

	result := &Tensor{
		Data:  make([]float64, resultSize),
		Shape: newShape,
	}

	// Вычисляем strides
	result.Strides = make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		result.Strides[i] = stride
		stride *= newShape[i]
	}

	// Для axis=0
	if axis == 0 {
		chunkSize := len(tensor.Data) / tensor.Shape[0]
		for i, idx := range indices {
			srcOffset := idx * chunkSize
			dstOffset := i * chunkSize
			copy(result.Data[dstOffset:dstOffset+chunkSize], tensor.Data[srcOffset:srcOffset+chunkSize])
		}
	} else {
		return nil, fmt.Errorf("gather вдоль оси %d еще не реализован", axis)
	}

	return result, nil
}
