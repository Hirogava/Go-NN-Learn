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

// Backward выполняет обратное распространение по всему графу
func (e *Engine) Backward(finalNode *graph.Node) {
	// Инициализировать градиент конечного узла единицами той же формы
	finalNode.Grad = tensor.Ones(finalNode.Value.Shape...)

	// Выполнить топологическую сортировку узлов
	sortedNodes := e.topologicalSort(finalNode)

	// Выполнить обратное распространение в обратном порядке
	for i := len(sortedNodes) - 1; i >= 0; i-- {
		node := sortedNodes[i]
		if node.Operation != nil {
			node.Operation.Backward(node.Grad)
		}
	}
}

// Топологическая сортировка узлов от finalNode
func (e *Engine) topologicalSort(root *graph.Node) []*graph.Node {
	visited := make(map[*graph.Node]bool)
	stack := make([]*graph.Node, 0)

	var dfs func(*graph.Node)
	dfs = func(node *graph.Node) {
		if visited[node] {
			return
		}
		visited[node] = true

		// Сначала посещаем всех родителей
		for _, parent := range node.Parents {
			dfs(parent)
		}

		// Затем добавляем текущий узел в стек
		stack = append(stack, node)
	}

	dfs(root)
	return stack
}

// RequireGrad оборачивает тензор в узел, включаемый в граф
func (e *Engine) RequireGrad(t *tensor.Tensor) *graph.Node {
	node := graph.NewNode(t, nil, nil) // листовой узел без родителей и операций
	e.Nodes = append(e.Nodes, node)
	return node
}

// Обнуление градиентов всех узлов
func (e *Engine) ZeroGrad() {
	for _, node := range e.Nodes {
		node.ZeroGrad()
	}
}
