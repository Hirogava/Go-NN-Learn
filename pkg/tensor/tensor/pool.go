package tensor

import (
	"fmt"
	"sync"
)

// TensorPool - пул для переиспользования тензоров и снижения аллокаций
type TensorPool struct {
	pools map[int]*sync.Pool
	mu    sync.RWMutex
}

// globalPool - глобальный пул тензоров
var globalPool = NewTensorPool()

// NewTensorPool создает новый пул тензоров
func NewTensorPool() *TensorPool {
	return &TensorPool{
		pools: make(map[int]*sync.Pool),
	}
}

// Get получает тензор из пула или создает новый
func (tp *TensorPool) Get(size int) *Tensor {
	tp.mu.RLock()
	pool, exists := tp.pools[size]
	tp.mu.RUnlock()

	if !exists {
		tp.mu.Lock()
		// Проверяем еще раз после получения write lock
		pool, exists = tp.pools[size]
		if !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return &Tensor{
						Data: make([]float64, size),
					}
				},
			}
			tp.pools[size] = pool
		}
		tp.mu.Unlock()
	}

	tensor := pool.Get().(*Tensor)
	// Обнуляем данные для безопасности
	for i := range tensor.Data {
		tensor.Data[i] = 0.0
	}
	return tensor
}

// Put возвращает тензор в пул для переиспользования
func (tp *TensorPool) Put(t *Tensor) {
	if t == nil || t.Data == nil {
		return
	}

	size := len(t.Data)
	tp.mu.RLock()
	pool, exists := tp.pools[size]
	tp.mu.RUnlock()

	if exists {
		// Очищаем метаданные перед возвратом в пул
		t.Shape = nil
		t.Strides = nil
		pool.Put(t)
	}
}

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
	if m >= ParallelThreshold || p >= ParallelThreshold {
		matmulParallelBlocked(a.Data, b.Data, result.Data, m, n, p)
	} else if m >= BlockSize || p >= BlockSize {
		matmulBlocked(a.Data, b.Data, result.Data, m, n, p)
	} else {
		matmulOptimized(a.Data, b.Data, result.Data, m, n, p)
	}

	return result, nil
}

// AddPooled - версия Add с использованием пула памяти
func AddPooled(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}

	result := GetTensor(len(a.Data))
	result.Shape = append([]int{}, a.Shape...)
	result.Strides = append([]int{}, a.Strides...)

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}

	return result, nil
}

// MulPooled - версия Mul с использованием пула памяти
func MulPooled(a, b *Tensor) (*Tensor, error) {
	if !shapesEqual(a.Shape, b.Shape) {
		return nil, fmt.Errorf("формы тензоров должны совпадать: %v != %v", a.Shape, b.Shape)
	}

	result := GetTensor(len(a.Data))
	result.Shape = append([]int{}, a.Shape...)
	result.Strides = append([]int{}, a.Strides...)

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] * b.Data[i]
	}

	return result, nil
}

// ApplyPooled - версия Apply с использованием пула памяти
func ApplyPooled(a *Tensor, f func(float64) float64) *Tensor {
	result := GetTensor(len(a.Data))
	result.Shape = append([]int{}, a.Shape...)
	result.Strides = append([]int{}, a.Strides...)

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = f(a.Data[i])
	}

	return result
}
