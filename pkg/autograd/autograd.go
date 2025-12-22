<<<<<<< HEAD
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
=======
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

func (op *ReLUOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		if op.input.Value.Data[i] > 0 {
			result.Data[i] = op.input.Value.Data[i]
		}
	}
	return result
}

func (e *Engine) ReLU(input *graph.Node) *graph.Node {
	op := NewReLUOp(input)
	result := op.Forward()
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

func (op *SigmoidOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		result.Data[i] = 1.0 / (1.0 + math.Exp(-op.input.Value.Data[i]))
	}
	op.output = result
	return result
}

func (e *Engine) Sigmoid(input *graph.Node) *graph.Node {
	op := NewSigmoidOp(input)
	result := op.Forward()
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

func (op *TanhOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		result.Data[i] = math.Tanh(op.input.Value.Data[i])
	}
	op.output = result
	return result
}

func (e *Engine) Tanh(input *graph.Node) *graph.Node {
	op := NewTanhOp(input)
	result := op.Forward()
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

func (op *SoftmaxCrossEntropyOp) Forward() *tensor.Tensor {
	if len(op.input.Value.Shape) != 2 || len(op.target.Shape) != 2 || op.input.Value.Shape[0] != op.target.Shape[0] {
		panic("Input and target must be 2D tensors with matching batch sizes")
	}

	maxVal := tensor.Max(op.input.Value)
	shifted := tensor.Sub(op.input.Value, maxVal)
	exp := tensor.Exp(shifted)

	rows, cols := op.input.Value.Shape[0], op.input.Value.Shape[1]
	sumExp := tensor.Zeros(rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			idx := i*op.input.Value.Strides[0] + j*op.input.Value.Strides[1]
			sum += exp.Data[idx]
		}
		sumExp.Data[i] = sum
	}

	softmax := tensor.Div(exp, sumExp)
	op.softmax = softmax

	loss := tensor.Zeros(rows, 1)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*op.input.Value.Strides[0] + j*op.input.Value.Strides[1]
			targetIdx := i*op.target.Strides[0] + j*op.target.Strides[1]
			if op.target.Data[targetIdx] > 0 {
				loss.Data[i] -= math.Log(math.Max(softmax.Data[idx], 1e-15)) * op.target.Data[targetIdx]
			}
		}
	}
	op.output = loss
	return loss
}

func (e *Engine) SoftmaxCrossEntropy(input *graph.Node, target *tensor.Tensor) *graph.Node {
	op := NewSoftmaxCrossEntropyOp(input, target)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *SoftmaxCrossEntropyOp) Backward(grad *tensor.Tensor) {
	rows, cols := op.input.Value.Shape[0], op.input.Value.Shape[1]
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*op.input.Value.Strides[0] + j*op.input.Value.Strides[1]
			targetIdx := i*op.target.Strides[0] + j*op.target.Strides[1]
			gradInput.Data[idx] = (op.softmax.Data[idx] - op.target.Data[targetIdx]) * grad.Data[i]
		}
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}
>>>>>>> main
