package autograd

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type Engine struct {
	Nodes []*graph.Node
}

func NewEngine() *Engine {
	return &Engine{
		Nodes: make([]*graph.Node, 0),
	}
}

func (e *Engine) Backward(finalNode *graph.Node) {
	finalNode.Grad = tensor.Ones(finalNode.Value.Shape...)

	sortedNodes := e.topologicalSort(finalNode)

	for i := len(sortedNodes) - 1; i >= 0; i-- {
		node := sortedNodes[i]
		if node.Operation != nil {
			node.Operation.Backward(node.Grad)
		}
	}
}

func (e *Engine) topologicalSort(root *graph.Node) []*graph.Node {
	visited := make(map[*graph.Node]bool)
	stack := make([]*graph.Node, 0)

	var dfs func(*graph.Node)
	dfs = func(node *graph.Node) {
		if visited[node] {
			return
		}
		visited[node] = true

		for _, parent := range node.Parents {
			dfs(parent)
		}

		stack = append(stack, node)
	}

	dfs(root)
	return stack
}

// backwardOp adapts any Backward func to the graph.Operation interface.
type backwardOp struct {
	backwardFn func(*tensor.Tensor)
}

func (b *backwardOp) Forward(inputs ...*graph.Node) *graph.Node { return nil }
func (b *backwardOp) Backward(grad *tensor.Tensor)              { b.backwardFn(grad) }

func (e *Engine) RequireGrad(t *tensor.Tensor) *graph.Node {
	node := graph.NewNode(t, nil, nil)
	e.Nodes = append(e.Nodes, node)
	return node
}

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
	node := graph.NewNode(result, []*graph.Node{input}, &backwardOp{op.Backward})
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
	node := graph.NewNode(result, []*graph.Node{input}, &backwardOp{op.Backward})
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
	node := graph.NewNode(result, []*graph.Node{input}, &backwardOp{op.Backward})
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

	maxVal := tensor.Max(op.input.Value).Data[0]

	rows, cols := op.input.Value.Shape[0], op.input.Value.Shape[1]

	exp := tensor.Zeros(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*op.input.Value.Strides[0] + j*op.input.Value.Strides[1]
			exp.Data[idx] = math.Exp(op.input.Value.Data[idx] - maxVal)
		}
	}

	sumExp := tensor.Zeros(rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			idx := i*exp.Strides[0] + j*exp.Strides[1]
			sum += exp.Data[idx]
		}
		sumExp.Data[i] = sum
	}

	softmax := tensor.Zeros(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*exp.Strides[0] + j*exp.Strides[1]
			softmax.Data[idx] = exp.Data[idx] / sumExp.Data[i]
		}
	}
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
	node := graph.NewNode(result, []*graph.Node{input}, &backwardOp{op.Backward})
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
