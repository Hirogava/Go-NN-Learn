package autograd

import (
	"math"

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

type ReLUOp struct {
	input *graph.Node
}

func NewReLUOp(input *graph.Node) *ReLUOp {
	return &ReLUOp{input: input}
}

func (e *Engine) ReLU(input *graph.Node) *graph.Node {
	op := NewReLUOp(input)
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		if op.input.Value.Data[i] > 0 {
			result.Data[i] = op.input.Value.Data[i]
		}
	}
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *ReLUOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		if op.input.Value.Data[i] > 0 {
			gradInput.Data[i] = grad.Data[i]
		}
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type SigmoidOp struct {
	input  *graph.Node
	output *tensor.Tensor
}

func NewSigmoidOp(input *graph.Node) *SigmoidOp {
	return &SigmoidOp{input: input}
}

func (e *Engine) Sigmoid(input *graph.Node) *graph.Node {
	op := NewSigmoidOp(input)
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		result.Data[i] = 1.0 / (1.0 + math.Exp(-op.input.Value.Data[i]))
	}
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *SigmoidOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		s := op.output.Data[i]
		gradInput.Data[i] = grad.Data[i] * s * (1.0 - s)
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type TanhOp struct {
	input  *graph.Node
	output *tensor.Tensor
}

func NewTanhOp(input *graph.Node) *TanhOp {
	return &TanhOp{input: input}
}

func (e *Engine) Tahn(input *graph.Node) *graph.Node {
	op := NewTanhOp(input)
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		result.Data[i] = math.Tanh(op.input.Value.Data[i])
	}
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *TanhOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		t := op.output.Data[i]
		gradInput.Data[i] = grad.Data[i] * (1.0 - t*t)
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type SoftmaxCrossEntropyOp struct {
	input   *graph.Node
	target  *tensor.Tensor
	output  *tensor.Tensor
	softmax *tensor.Tensor
}

func NewSoftmaxCrossEntropyOp(input *graph.Node, target *tensor.Tensor) *SoftmaxCrossEntropyOp {
	return &SoftmaxCrossEntropyOp{input: input, target: target}
}

func (e *Engine) SoftmaxCrossEntropy(input *graph.Node, target *tensor.Tensor) *graph.Node {
	op := NewSoftmaxCrossEntropyOp(input, target)
	maxVal := tensor.Max(op.input.Value)
	shifted := tensor.Sub(op.input.Value, maxVal)

	exp := tensor.Exp(shifted)
	sumExp := tensor.Sum(exp)
	softmax := tensor.Div(exp, sumExp)
	op.softmax = softmax

	loss := tensor.Zeros(1)
	for i := range op.input.Value.Data {
		if op.target.Data[i] > 0 {
			loss.Data[i] -= math.Log(math.Max(softmax.Data[i], 1e-15)) * op.target.Data[i]
		}
	}
	node := graph.NewNode(loss, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *SoftmaxCrossEntropyOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		gradInput.Data[i] = (op.softmax.Data[i] - op.target.Data[i]) * grad.Data[0]
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}
