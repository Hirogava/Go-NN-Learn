package profiling

import (
	"context"
)

type profilerKeyType struct{}

var profilerKey = profilerKeyType{}

// WithProfiler добавляет профилировщик в контекст
func WithProfiler(ctx context.Context, profiler *Profiler) context.Context {
	return context.WithValue(ctx, profilerKey, profiler)
}

// FromContext извлекает профилировщик из контекста
func FromContext(ctx context.Context) *Profiler {
	if profiler, ok := ctx.Value(profilerKey).(*Profiler); ok {
		return profiler
	}
	return nil
}

// TraceOperation выполняет операцию с трассировкой
// Пример использования:
//
//	err := profiling.TraceOperation(ctx, "MyOperation", func() error {
//	    // ваш код
//	    return nil
//	})
func TraceOperation(ctx context.Context, operationName string, fn func() error) error {
	profiler := FromContext(ctx)
	if profiler == nil || !profiler.Config.EnableOperationMetrics {
		// Если профилировщика нет или метрики отключены, просто выполняем функцию
		return fn()
	}

	timer := profiler.OperationMetrics.StartOperation(operationName)
	defer timer.Stop()

	return fn()
}

// TraceOperationWithSize выполняет операцию с трассировкой и указанием размеров данных
func TraceOperationWithSize(ctx context.Context, operationName string, inputSize, outputSize int64, fn func() error) error {
	profiler := FromContext(ctx)
	if profiler == nil || !profiler.Config.EnableOperationMetrics {
		return fn()
	}

	timer := profiler.OperationMetrics.StartOperation(operationName)
	timer.SetInputSize(inputSize)
	timer.SetOutputSize(outputSize)
	defer timer.Stop()

	return fn()
}

// RecordOperation записывает операцию без выполнения функции
// Полезно когда нужно просто зафиксировать уже выполненную работу
func RecordOperation(ctx context.Context, operationName string, inputSize, outputSize int64) *OpTimer {
	profiler := FromContext(ctx)
	if profiler == nil || !profiler.Config.EnableOperationMetrics {
		return nil
	}

	timer := profiler.OperationMetrics.StartOperation(operationName)
	if inputSize > 0 {
		timer.SetInputSize(inputSize)
	}
	if outputSize > 0 {
		timer.SetOutputSize(outputSize)
	}

	return timer
}
