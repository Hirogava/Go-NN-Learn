package tensor

import (
	"context"
	"fmt"
)

// ProfiledOps содержит профилируемые версии тензорных операций
// Эти функции принимают context для доступа к профилировщику

// AddWithContext выполняет поэлементное сложение с профилированием
func AddWithContext(ctx context.Context, a, b *Tensor) (*Tensor, error) {
	// Импортируем профилировщик динамически, чтобы избежать циклических зависимостей
	// Если профилировщик есть в контексте, используем его
	if ctx != nil {
		// Создаем обертку для записи метрик
		type opTimer interface {
			SetInputSize(int64)
			SetOutputSize(int64)
			Stop()
		}

		// Пытаемся получить функцию записи операции из контекста
		type profilerInterface interface {
			RecordOperation(name string, inputSize, outputSize int64) opTimer
		}

		// Проверяем наличие профилировщика
		if profilerKey := ctx.Value("profiler"); profilerKey != nil {
			if profiler, ok := profilerKey.(profilerInterface); ok {
				timer := profiler.RecordOperation("tensor.Add", a.Size()+b.Size(), 0)
				defer func() {
					if timer != nil {
						timer.Stop()
					}
				}()
			}
		}
	}

	return Add(a, b)
}

// MulWithContext выполняет поэлементное умножение с профилированием
func MulWithContext(ctx context.Context, a, b *Tensor) (*Tensor, error) {
	if ctx != nil {
		type opTimer interface {
			SetInputSize(int64)
			SetOutputSize(int64)
			Stop()
		}

		type profilerInterface interface {
			RecordOperation(name string, inputSize, outputSize int64) opTimer
		}

		if profilerKey := ctx.Value("profiler"); profilerKey != nil {
			if profiler, ok := profilerKey.(profilerInterface); ok {
				timer := profiler.RecordOperation("tensor.Mul", a.Size()+b.Size(), 0)
				defer func() {
					if timer != nil {
						timer.Stop()
					}
				}()
			}
		}
	}

	return Mul(a, b)
}

// ApplyWithContext применяет функцию с профилированием
func ApplyWithContext(ctx context.Context, a *Tensor, f func(float64) float64, opName string) *Tensor {
	if ctx != nil {
		type opTimer interface {
			SetInputSize(int64)
			SetOutputSize(int64)
			Stop()
		}

		type profilerInterface interface {
			RecordOperation(name string, inputSize, outputSize int64) opTimer
		}

		if profilerKey := ctx.Value("profiler"); profilerKey != nil {
			if profiler, ok := profilerKey.(profilerInterface); ok {
				timer := profiler.RecordOperation(fmt.Sprintf("tensor.Apply.%s", opName), a.Size(), a.Size())
				defer func() {
					if timer != nil {
						timer.Stop()
					}
				}()
			}
		}
	}

	return Apply(a, f)
}
