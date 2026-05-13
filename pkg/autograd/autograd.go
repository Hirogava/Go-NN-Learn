package autograd

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/ops"
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

	maxVal := ops.Max(op.input.Value)
	shifted := ops.Sub(op.input.Value, maxVal)
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

	softmax := ops.Div(exp, sumExp)
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

type LeakyReLUOp struct {
	input *graph.Node
	slope float64
}

func NewLeakyReLUOp(input *graph.Node, slope float64) *LeakyReLUOp {
	return &LeakyReLUOp{input: input, slope: slope}
}

func (op *LeakyReLUOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		val := op.input.Value.Data[i]
		if val > 0 {
			result.Data[i] = val
		} else {
			result.Data[i] = op.slope * val
		}
	}
	return result
}

func (e *Engine) LeakyReLU(input *graph.Node, slope float64) *graph.Node {
	op := NewLeakyReLUOp(input, slope)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *LeakyReLUOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		if op.input.Value.Data[i] > 0 {
			gradInput.Data[i] = grad.Data[i]
		} else {
			gradInput.Data[i] = grad.Data[i] * op.slope
		}
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type ELUOp struct {
	input *graph.Node
	alpha float64
}

func NewELUOp(input *graph.Node, alpha float64) *ELUOp {
	return &ELUOp{input: input, alpha: alpha}
}

func (op *ELUOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		val := op.input.Value.Data[i]
		if val > 0 {
			result.Data[i] = val
		} else {
			result.Data[i] = op.alpha * (math.Exp(val) - 1.0)
		}
	}
	return result
}

func (e *Engine) ELU(input *graph.Node, alpha float64) *graph.Node {
	op := NewELUOp(input, alpha)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *ELUOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		val := op.input.Value.Data[i]
		if val > 0 {
			gradInput.Data[i] = grad.Data[i]
		} else {
			gradInput.Data[i] = grad.Data[i] * op.alpha * math.Exp(val)
		}
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type SoftPlusOp struct {
	input *graph.Node
}

func NewSoftPlusOp(input *graph.Node) *SoftPlusOp {
	return &SoftPlusOp{input: input}
}

func (op *SoftPlusOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		val := op.input.Value.Data[i]
		if val > 20 {
			result.Data[i] = val
		} else {
			result.Data[i] = math.Log(1.0 + math.Exp(val))
		}
	}
	return result
}

func (e *Engine) SoftPlus(input *graph.Node) *graph.Node {
	op := NewSoftPlusOp(input)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *SoftPlusOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		val := op.input.Value.Data[i]
		s := 1.0 / (1.0 + math.Exp(-val))
		gradInput.Data[i] = grad.Data[i] * s
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type GELUOp struct {
	input *graph.Node
}

func NewGELUOp(input *graph.Node) *GELUOp {
	return &GELUOp{input: input}
}

func (op *GELUOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		x := op.input.Value.Data[i]
		inner := math.Sqrt(2.0/math.Pi) * (x + 0.044715*math.Pow(x, 3))
		result.Data[i] = 0.5 * x * (1.0 + math.Tanh(inner))
	}
	return result
}

func (e *Engine) GELU(input *graph.Node) *graph.Node {
	op := NewGELUOp(input)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *GELUOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		x := op.input.Value.Data[i]
		cdf_inner := math.Sqrt(2.0/math.Pi) * (x + 0.044715*math.Pow(x, 3))
		t := math.Tanh(cdf_inner)
		term1 := 0.5 * (1.0 + t)
		term2 := 0.5 * x * (1.0 - t*t) * math.Sqrt(2.0/math.Pi) * (1.0 + 3.0*0.044715*x*x)
		gradInput.Data[i] = grad.Data[i] * (term1 + term2)
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

type SoftmaxOp struct {
	input  *graph.Node
	output *tensor.Tensor
}

func NewSoftmaxOp(input *graph.Node) *SoftmaxOp {
	return &SoftmaxOp{input: input}
}

func (op *SoftmaxOp) Forward() *tensor.Tensor {
	in := op.input.Value
	if len(in.Shape) != 2 {
		panic("Softmax expects 2D tensor (batch, classes)")
	}
	rows, cols := in.Shape[0], in.Shape[1]
	result := tensor.Zeros(in.Shape...)

	for i := 0; i < rows; i++ {
		maxVal := in.Data[i*cols]
		for j := 1; j < cols; j++ {
			if in.Data[i*cols+j] > maxVal {
				maxVal = in.Data[i*cols+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < cols; j++ {
			val := math.Exp(in.Data[i*cols+j] - maxVal)
			result.Data[i*cols+j] = val
			sumExp += val
		}

		for j := 0; j < cols; j++ {
			result.Data[i*cols+j] /= sumExp
		}
	}
	op.output = result
	return result
}

func (e *Engine) Softmax(input *graph.Node) *graph.Node {
	op := NewSoftmaxOp(input)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{input}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

func (op *SoftmaxOp) Backward(grad *tensor.Tensor) {
	rows, cols := op.output.Shape[0], op.output.Shape[1]
	gradInput := tensor.Zeros(op.output.Shape...)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sj := op.output.Data[i*cols+j]
			sum := 0.0
			for k := 0; k < cols; k++ {
				sk := op.output.Data[i*cols+k]
				gk := grad.Data[i*cols+k]
				if k == j {
					sum += gk * sj * (1.0 - sj)
				} else {
					sum -= gk * sk * sj
				}
			}
			gradInput.Data[i*cols+j] = sum
		}
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}
