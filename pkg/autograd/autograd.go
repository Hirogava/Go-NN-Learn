package autograd

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

type Engine struct {
	Nodes []*graph.Node
}

func NewEngine() *Engine {
	return &Engine{
		Nodes: make([]*graph.Node, 0),
	}
}

//	Обратное распространение по всему графу
func (e *Engine) Backward(finalNode *graph.Node) {
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

//	Обнуление градиентов всех узлов
func (e *Engine) ZeroGrad() {
	for _, node := range e.Nodes {
		node.ZeroGrad()
	}
}

//	Сложение двух узлов
func (e *Engine) Add(a, b *graph.Node) *graph.Node {
	// TODO: Реализовать операцию сложения
	result := &graph.Node{
		Value:   &tensor.Tensor{}, // TODO: вычислить реальное значение
		Parents: []*graph.Node{a, b},
	}
	e.Nodes = append(e.Nodes, result)
	return result
}
