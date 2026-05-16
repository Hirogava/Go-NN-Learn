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

type SoftPlusOp struct {
	input   *graph.Node
	output  *tensor.Tensor
	sigmoid *tensor.Tensor // d/dx softplus(x) = sigmoid(x)
}

func NewSoftPlusOp(input *graph.Node) *SoftPlusOp {
	return &SoftPlusOp{input: input}
}

func (op *SoftPlusOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	sigmoid := tensor.Zeros(op.input.Value.Shape...)

	// Стабильные вычисления без переполнений:
	// softplus(x) = log(1 + exp(x))
	// if x > 0:  softplus(x) = x + log(1 + exp(-x))
	// else:      softplus(x) = log(1 + exp(x))
	for i, x := range op.input.Value.Data {
		if x >= 0 {
			expNeg := math.Exp(-x)
			// sigmoid(x) = 1 / (1 + exp(-x))
			s := 1.0 / (1.0 + expNeg)
			sigmoid.Data[i] = s
			result.Data[i] = x + math.Log(1.0+expNeg)
		} else {
			expPos := math.Exp(x)
			// sigmoid(x) = exp(x) / (1 + exp(x))
			s := expPos / (1.0 + expPos)
			sigmoid.Data[i] = s
			result.Data[i] = math.Log(1.0 + expPos)
		}
	}

	op.output = result
	op.sigmoid = sigmoid
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
		gradInput.Data[i] = grad.Data[i] * op.sigmoid.Data[i]
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

// geluNormalCDF — Φ(x), CDF стандартного нормального распределения.
// Используется связка с math.Erf (устойчива на краях и для больших |x|).
func geluNormalCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
}

// geluNormalPDF — плотность N(0,1) в точке x: exp(-x²/2) / √(2π).
func geluNormalPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) * (1.0 / math.Sqrt(2.0*math.Pi))
}

type GELUOp struct {
	input *graph.Node
}

func NewGELUOp(input *graph.Node) *GELUOp {
	return &GELUOp{input: input}
}

// Forward: y_i = x_i * Φ(x_i) (активация GELU из BERT/GPT).
func (op *GELUOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		x := op.input.Value.Data[i]
		result.Data[i] = x * geluNormalCDF(x)
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

// Backward: d/dx [x·Φ(x)] = Φ(x) + x·φ(x).
func (op *GELUOp) Backward(grad *tensor.Tensor) {
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		x := op.input.Value.Data[i]
		cdf := geluNormalCDF(x)
		pdf := geluNormalPDF(x)
		gradInput.Data[i] = grad.Data[i] * (cdf + x*pdf)
	}
	if op.input.Grad == nil {
		op.input.Grad = tensor.Zeros(op.input.Value.Shape...)
	}
	op.input.Grad = gradInput
}

// LeakyReLUOp: y_i = max(slope * x_i, x_i). Обычно 0 < slope < 1 (например 0.01).
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
		x := op.input.Value.Data[i]
		if x > 0 {
			result.Data[i] = x
		} else {
			result.Data[i] = op.slope * x
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
	input  *graph.Node
	alpha  float64
	output *tensor.Tensor
}

func NewELUOp(input *graph.Node, alpha float64) *ELUOp {
	return &ELUOp{input: input, alpha: alpha}
}

func (op *ELUOp) Forward() *tensor.Tensor {
	result := tensor.Zeros(op.input.Value.Shape...)
	for i := range op.input.Value.Data {
		x := op.input.Value.Data[i]
		if x > 0 {
			result.Data[i] = x
		} else {
			result.Data[i] = op.alpha * (math.Exp(x) - 1.0)
		}
	}
	op.output = result
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
		x := op.input.Value.Data[i]
		if x > 0 {
			gradInput.Data[i] = grad.Data[i]
		} else {
			gradInput.Data[i] = grad.Data[i] * (op.output.Data[i] + op.alpha)
		}
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
	shape := op.input.Value.Shape
	result := tensor.Zeros(shape...)

	switch len(shape) {
	case 1:
		maxVal := op.input.Value.Data[0]
		for i := range op.input.Value.Data {
			if op.input.Value.Data[i] > maxVal {
				maxVal = op.input.Value.Data[i]
			}
		}
		sumExp := 0.0
		for i := range op.input.Value.Data {
			e := math.Exp(op.input.Value.Data[i] - maxVal)
			result.Data[i] = e
			sumExp += e
		}
		for i := range result.Data {
			result.Data[i] /= sumExp
		}
	case 2:
		rows, cols := shape[0], shape[1]
		for r := range rows {
			base := r * cols
			maxVal := op.input.Value.Data[base]
			for c := 1; c < cols; c++ {
				v := op.input.Value.Data[base+c]
				if v > maxVal {
					maxVal = v
				}
			}
			sumExp := 0.0
			for c := range cols {
				e := math.Exp(op.input.Value.Data[base+c] - maxVal)
				result.Data[base+c] = e
				sumExp += e
			}
			for c := range cols {
				result.Data[base+c] /= sumExp
			}
		}
	default:
		panic("Softmax expects 1D or 2D input")
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
	shape := op.input.Value.Shape
	gradInput := tensor.Zeros(shape...)

	switch len(shape) {
	case 1:
		dot := 0.0
		for i := range grad.Data {
			dot += grad.Data[i] * op.output.Data[i]
		}
		for i := range gradInput.Data {
			gradInput.Data[i] = op.output.Data[i] * (grad.Data[i] - dot)
		}
	case 2:
		rows, cols := shape[0], shape[1]
		for r := range rows {
			base := r * cols
			dot := 0.0
			for c := range cols {
				dot += grad.Data[base+c] * op.output.Data[base+c]
			}
			for c := range cols {
				gradInput.Data[base+c] = op.output.Data[base+c] * (grad.Data[base+c] - dot)
			}
		}
	default:
		panic("Softmax expects 1D or 2D input")
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
