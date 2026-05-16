package autograd

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// CheckGradientEngine compares analytical and numerical gradients for a graph.
// build receives a fresh Engine and fresh leaf nodes created from inputs.
func CheckGradientEngine(build func(e *Engine, inputs []*graph.Node) *graph.Node, inputs []*graph.Node, eps float64, tol float64) bool {
	sizes := make([]int, len(inputs))
	total := 0
	for i, inp := range inputs {
		size := 1
		for _, d := range inp.Value.Shape {
			size *= d
		}
		sizes[i] = size
		total += size
	}

	pack := func(nodes []*graph.Node) []float64 {
		x := make([]float64, total)
		pos := 0
		for i, n := range nodes {
			for j := 0; j < sizes[i]; j++ {
				x[pos] = n.Value.Data[j]
				pos++
			}
		}
		return x
	}

	makeInputNodes := func(x []float64) []*graph.Node {
		nodes := make([]*graph.Node, len(inputs))
		pos := 0
		for i, orig := range inputs {
			data := make([]float64, sizes[i])
			copy(data, x[pos:pos+sizes[i]])
			pos += sizes[i]
			t := &tensor.Tensor{
				Data:    data,
				Shape:   append([]int{}, orig.Value.Shape...),
				Strides: append([]int{}, orig.Value.Strides...),
			}
			nodes[i] = graph.NewNode(t, nil, nil)
		}
		return nodes
	}

	eval := func(x []float64) float64 {
		e := NewEngine()
		inNodes := makeInputNodes(x)
		out := build(e, inNodes)
		if out == nil || out.Value == nil {
			return math.NaN()
		}
		if len(out.Value.Data) == 1 {
			return out.Value.Data[0]
		}
		s := 0.0
		for _, v := range out.Value.Data {
			s += v
		}
		return s
	}

	x0 := pack(inputs)

	eAnal := NewEngine()
	inAnal := makeInputNodes(x0)
	outAnal := build(eAnal, inAnal)
	if outAnal == nil || outAnal.Value == nil {
		return false
	}

	eAnal.Backward(outAnal)

	analytic := make([]float64, total)
	pos := 0
	for i, n := range inAnal {
		if n.Grad == nil {
			for j := 0; j < sizes[i]; j++ {
				analytic[pos] = 0
				pos++
			}
			continue
		}
		for j := 0; j < sizes[i]; j++ {
			analytic[pos] = n.Grad.Data[j]
			pos++
		}
	}

	numeric := make([]float64, total)
	for i := 0; i < total; i++ {
		xInc := make([]float64, total)
		xDec := make([]float64, total)
		copy(xInc, x0)
		copy(xDec, x0)
		xInc[i] += eps
		xDec[i] -= eps
		numeric[i] = (eval(xInc) - eval(xDec)) / (2 * eps)
	}

	for i := 0; i < total; i++ {
		absErr := math.Abs(analytic[i] - numeric[i])
		m := math.Max(1.0, math.Max(math.Abs(analytic[i]), math.Abs(numeric[i])))
		relErr := absErr / m
		if relErr > tol {
			return false
		}
	}
	return true
}

func tensorFromData(data []float64, shape []int) (*tensor.Tensor, bool) {
	if len(shape) == 0 || len(data) == 0 {
		return nil, false
	}

	size := 1
	for _, d := range shape {
		if d <= 0 {
			return nil, false
		}
		size *= d
	}
	if len(data) != size {
		return nil, false
	}

	t := tensor.Zeros(shape...)
	copy(t.Data, data)
	return t, true
}

func gradientCheckUnary(build func(e *Engine, input *graph.Node) *graph.Node, data []float64, shape []int, eps, tol float64) bool {
	t, ok := tensorFromData(data, shape)
	if !ok {
		return false
	}
	proto := graph.NewNode(t, nil, nil)
	return CheckGradientEngine(func(e *Engine, inputs []*graph.Node) *graph.Node {
		return build(e, inputs[0])
	}, []*graph.Node{proto}, eps, tol)
}

// ReLUGradientCheckOK проверяет backward для ReLU.
func ReLUGradientCheckOK(data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.ReLU(input)
	}, data, shape, eps, tol)
}

// SigmoidGradientCheckOK проверяет backward для Sigmoid.
func SigmoidGradientCheckOK(data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.Sigmoid(input)
	}, data, shape, eps, tol)
}

// TanhGradientCheckOK проверяет backward для Tanh.
func TanhGradientCheckOK(data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.Tanh(input)
	}, data, shape, eps, tol)
}

// SoftPlusGradientCheckOK проверяет backward для SoftPlus.
func SoftPlusGradientCheckOK(data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.SoftPlus(input)
	}, data, shape, eps, tol)
}

// GELUGradientCheckOK проверяет backward для GELU.
func GELUGradientCheckOK(data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.GELU(input)
	}, data, shape, eps, tol)
}

// LeakyReLUGradientCheckOK проверяет backward для LeakyReLU.
func LeakyReLUGradientCheckOK(slope float64, data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.LeakyReLU(input, slope)
	}, data, shape, eps, tol)
}

// ELUGradientCheckOK проверяет backward для ELU.
func ELUGradientCheckOK(alpha float64, data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.ELU(input, alpha)
	}, data, shape, eps, tol)
}

// SoftmaxGradientCheckOK проверяет backward для Softmax.
func SoftmaxGradientCheckOK(data []float64, shape []int, eps, tol float64) bool {
	return gradientCheckUnary(func(e *Engine, input *graph.Node) *graph.Node {
		return e.Softmax(input)
	}, data, shape, eps, tol)
}

// SoftmaxCrossEntropyGradientCheckOK проверяет backward для SoftmaxCrossEntropy.
func SoftmaxCrossEntropyGradientCheckOK(logits []float64, shape []int, target []float64, eps, tol float64) bool {
	inputTensor, ok := tensorFromData(logits, shape)
	if !ok {
		return false
	}
	targetTensor, ok := tensorFromData(target, shape)
	if !ok {
		return false
	}
	proto := graph.NewNode(inputTensor, nil, nil)
	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.SoftmaxCrossEntropy(inputs[0], targetTensor)
	}
	return CheckGradientEngine(build, []*graph.Node{proto}, eps, tol)
}