package autograd

import (
	"context"
	"time"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type Engine struct {
	Nodes []*graph.Node

	// Контекст для профилирования
	ctx context.Context

	// Метрики производительности
	backwardDuration time.Duration
	forwardDuration  time.Duration
}

func NewEngine() *Engine {
	return &Engine{
		Nodes: make([]*graph.Node, 0),
		ctx:   context.Background(),
	}
}

// NewEngineWithContext создает engine с контекстом для профилирования
func NewEngineWithContext(ctx context.Context) *Engine {
	return &Engine{
		Nodes: make([]*graph.Node, 0),
		ctx:   ctx,
	}
}

// SetContext устанавливает контекст для профилирования
func (e *Engine) SetContext(ctx context.Context) {
	e.ctx = ctx
}

// GetBackwardDuration возвращает время выполнения backward pass
func (e *Engine) GetBackwardDuration() time.Duration {
	return e.backwardDuration
}

// GetForwardDuration возвращает время выполнения forward pass
func (e *Engine) GetForwardDuration() time.Duration {
	return e.forwardDuration
}

// Обратное распространение по всему графу
func (e *Engine) Backward(finalNode *graph.Node) {
	startTime := time.Now()
	defer func() {
		e.backwardDuration = time.Since(startTime)
	}()

	// Профилирование backward pass
	if e.ctx != nil {
		// Используем интерфейс для избежания прямой зависимости
		type profilerInterface interface {
			RecordOperation(name string, inputSize, outputSize int64) interface{ Stop() }
		}

		if profilerKey := e.ctx.Value("profiler"); profilerKey != nil {
			if profiler, ok := profilerKey.(profilerInterface); ok {
				timer := profiler.RecordOperation("autograd.Backward", int64(len(e.Nodes)), 0)
				if timer != nil {
					defer timer.Stop()
				}
			}
		}
	}

	// TODO: Инициализировать градиент конечного узла единицей

	// TODO: Выполнить топологическую сортировку узлов
	sortedNodes := e.topologicalSort()

	// TODO: Выполнить обратное распространение в обратном порядке
	for i := len(sortedNodes) - 1; i >= 0; i-- {
		node := sortedNodes[i]
		if node.Operation != nil {
			node.Operation.Backward(node.Grad)
		}
	}
}

// Топологическая сортировка узлов
func (e *Engine) topologicalSort() []*graph.Node {
	// TODO: Реализовать топологическую сортировку
	return e.Nodes // временная заглушка
}

// Обнуление градиентов всех узлов
func (e *Engine) ZeroGrad() {
	for _, node := range e.Nodes {
		node.ZeroGrad()
	}
}

// Сложение двух узлов
func (e *Engine) Add(a, b *graph.Node) *graph.Node {
	// TODO: Реализовать операцию сложения
	result := &graph.Node{
		Value:   &tensor.Tensor{}, // TODO: вычислить реальное значение
		Parents: []*graph.Node{a, b},
	}
	e.Nodes = append(e.Nodes, result)
	return result
}
