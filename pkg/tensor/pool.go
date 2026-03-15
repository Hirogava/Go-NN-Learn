package tensor

import (
	"fmt"
	"sync"
)

// TensorPool - пул для переиспользования тензоров и снижения аллокаций
// TensorPool - пул для переиспользования тензоров и снижения аллокаций
type TensorPool struct {
	pools64 map[int]*sync.Pool
	pools32 map[int]*sync.Pool
	mu      sync.RWMutex
}

// NewTensorPool создает новый пул тензоров
func NewTensorPool() *TensorPool {
	return &TensorPool{
		pools64: make(map[int]*sync.Pool),
		pools32: make(map[int]*sync.Pool),
	}
}

// Get получает тензор из пула или создает новый
func (tp *TensorPool) Get(size int) *Tensor {
	dtype := GetDefaultDType()

	tp.mu.RLock()
	var pool *sync.Pool
	var exists bool
	if dtype == Float32 {
		pool, exists = tp.pools32[size]
	} else {
		pool, exists = tp.pools64[size]
	}
	tp.mu.RUnlock()

	if !exists {
		tp.mu.Lock()
		// Проверяем еще раз после получения write lock
		if dtype == Float32 {
			pool, exists = tp.pools32[size]
			if !exists {
				pool = &sync.Pool{
					New: func() interface{} {
						return &Tensor{
							Data32: make([]float32, size),
							DType:  Float32,
						}
					},
				}
				tp.pools32[size] = pool
			}
		} else {
			pool, exists = tp.pools64[size]
			if !exists {
				pool = &sync.Pool{
					New: func() interface{} {
						return &Tensor{
							Data:  make([]float64, size),
							DType: Float64,
						}
					},
				}
				tp.pools64[size] = pool
			}
		}
		tp.mu.Unlock()
	}

	tensor := pool.Get().(*Tensor)

	// Обнуляем данные для безопасности
	if tensor.DType == Float32 {
		// Optimization: check if we need to zero out?
		// Ideally checking if dirtied. For now always zero.
		for i := range tensor.Data32 {
			tensor.Data32[i] = 0.0
		}
	} else {
		for i := range tensor.Data {
			tensor.Data[i] = 0.0
		}
	}

	return tensor
}

// Put возвращает тензор в пул для переиспользования
func (tp *TensorPool) Put(t *Tensor) {
	if t == nil {
		return
	}

	var size int
	if t.DType == Float32 {
		if t.Data32 == nil {
			return
		}
		size = len(t.Data32)
	} else {
		if t.Data == nil {
			return
		}
		size = len(t.Data)
	}

	tp.mu.RLock()
	var pool *sync.Pool
	var exists bool
	if t.DType == Float32 {
		pool, exists = tp.pools32[size]
	} else {
		pool, exists = tp.pools64[size]
	}
	tp.mu.RUnlock()

	if exists {
		// Очищаем метаданные перед возвратом в пул
		t.Shape = nil
		t.Strides = nil
		pool.Put(t)
	}
}

// globalPool - глобальный пул тензоров
var globalPool = NewTensorPool()

// GetTensor - глобальная функция для получения тензора из пула
func GetTensor(size int) *Tensor {
	return globalPool.Get(size)
}

// PutTensor - глобальная функция для возврата тензора в пул
func PutTensor(t *Tensor) {
	globalPool.Put(t)
}

// WithPooledTensor выполняет функцию с тензором из пула
// и автоматически возвращает его в пул после выполнения
func WithPooledTensor(size int, fn func(*Tensor) error) error {
	t := GetTensor(size)
	defer PutTensor(t)
	return fn(t)
}

// MatMulPooled - версия MatMul с использованием пула памяти
func MatMulPooled(a, b *Tensor) (*Tensor, error) {
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают: %v != %v", a.DType, b.DType)
	}

	// Проверка размерностей
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("умножение матриц требует 2D тензоры")
	}

	m := a.Shape[0]
	n := a.Shape[1]
	p := b.Shape[1]

	if n != b.Shape[0] {
		return nil, fmt.Errorf("несовместимые формы для умножения матриц")
	}

	// Получаем тензор из пула
	result := GetTensor(m * p)
	result.Shape = []int{m, p}
	result.Strides = []int{p, 1}

	// Выполняем умножение
	if a.DType == Float32 {
		matmulV2Float32(a.Data32, b.Data32, result.Data32, m, n, p)
	} else {
		matmulV2(a.Data, b.Data, result.Data, m, n, p)
	}

	return result, nil
}

// AddPooled - версия Add с использованием пула памяти
func AddPooled(a, b *Tensor) (*Tensor, error) {
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают")
	}
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать")
	}

	size := a.DataLen()
	result := GetTensor(size)
	result.Shape = append([]int{}, a.Shape...)
	result.Strides = append([]int{}, a.Strides...)

	if a.DType == Float32 {
		for i := 0; i < size; i++ {
			result.Data32[i] = a.Data32[i] + b.Data32[i]
		}
	} else {
		for i := 0; i < size; i++ {
			result.Data[i] = a.Data[i] + b.Data[i]
		}
	}

	return result, nil
}

// MulPooled - версия Mul с использованием пула памяти
func MulPooled(a, b *Tensor) (*Tensor, error) {
	if a.DType != b.DType {
		return nil, fmt.Errorf("типы данных тензоров не совпадают")
	}
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать")
	}

	size := a.DataLen()
	result := GetTensor(size)
	result.Shape = append([]int{}, a.Shape...)
	result.Strides = append([]int{}, a.Strides...)

	if a.DType == Float32 {
		for i := 0; i < size; i++ {
			result.Data32[i] = a.Data32[i] * b.Data32[i]
		}
	} else {
		for i := 0; i < size; i++ {
			result.Data[i] = a.Data[i] * b.Data[i]
		}
	}

	return result, nil
}

// ApplyPooled - версия Apply с использованием пула памяти
func ApplyPooled(a *Tensor, f func(float64) float64) *Tensor {
	size := a.DataLen()
	result := GetTensor(size)
	result.Shape = append([]int{}, a.Shape...)
	result.Strides = append([]int{}, a.Strides...)

	if a.DType == Float32 {
		for i := 0; i < size; i++ {
			result.Data32[i] = float32(f(float64(a.Data32[i])))
		}
	} else {
		for i := 0; i < size; i++ {
			result.Data[i] = f(a.Data[i])
		}
	}

	return result
}
