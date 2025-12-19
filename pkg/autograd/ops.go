package autograd

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/matrix"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// Add
type Add struct{ Parents []*graph.Node }

func (op *Add) Backward(grad *tensor.Tensor) {
	for _, p := range op.Parents {
		if p.Grad == nil {
			p.Grad = tensor.Zeros(p.Value.Shape...)
		}
		g, _ := tensor.Add(p.Grad, grad)
		p.Grad = g
	}
}

func (e *Engine) Add(a, b *graph.Node) *graph.Node {
	val, err := tensor.Add(a.Value, b.Value)
	if err != nil {
		return nil
	}
	op := &Add{Parents: []*graph.Node{a, b}}
	n := graph.NewNode(val, []*graph.Node{a, b}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

// Mul
type MulOperation struct {
	Parents []*graph.Node
	A, B    *tensor.Tensor
}

func (op *MulOperation) Backward(grad *tensor.Tensor) {
	// d/da (a * b) = b, d/db (a * b) = a
	// для A
	if op.Parents[0].Grad == nil {
		op.Parents[0].Grad = tensor.Zeros(op.Parents[0].Value.Shape...)
	}
	gA_local, _ := tensor.Mul(op.B, grad)
	gA, _ := tensor.Add(op.Parents[0].Grad, gA_local)
	op.Parents[0].Grad = gA

	// для B
	if op.Parents[1].Grad == nil {
		op.Parents[1].Grad = tensor.Zeros(op.Parents[1].Value.Shape...)
	}
	gB_local, _ := tensor.Mul(op.A, grad)
	gB, _ := tensor.Add(op.Parents[1].Grad, gB_local)
	op.Parents[1].Grad = gB
}

func (e *Engine) Mul(a, b *graph.Node) *graph.Node {
	val, err := tensor.Mul(a.Value, b.Value)
	if err != nil {
		return nil
	}
	op := &MulOperation{Parents: []*graph.Node{a, b}, A: a.Value, B: b.Value}
	n := graph.NewNode(val, []*graph.Node{a, b}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

type MatMul struct {
	Parents []*graph.Node
	A       *tensor.Tensor
	B       *tensor.Tensor
}

type Transpose struct {
	Parents []*graph.Node
}

func (op *MatMul) Backward(grad *tensor.Tensor) {
	// Градиент для A
	if op.Parents[0].Grad == nil {
		op.Parents[0].Grad = tensor.Zeros(op.Parents[0].Value.Shape...)
	}

	gradM := matrix.TensorToMatrix(grad)
	bM := matrix.TensorToMatrix(op.B)

	bTransposed, _ := matrix.Transposition(bM)
	gA_local, _ := matrix.MatMul(gradM, bTransposed)
	gA_localT := matrix.MatrixToTensor(gA_local)
	gA, _ := tensor.Add(op.Parents[0].Grad, gA_localT)
	op.Parents[0].Grad = gA

	// Градиент для B
	if op.Parents[1].Grad == nil {
		op.Parents[1].Grad = tensor.Zeros(op.Parents[1].Value.Shape...)
	}

	aM := matrix.TensorToMatrix(op.A)
	aTransposed, _ := matrix.Transposition(aM)
	gB_local, _ := matrix.MatMul(aTransposed, gradM)
	gB_localT := matrix.MatrixToTensor(gB_local)
	gB, _ := tensor.Add(op.Parents[1].Grad, gB_localT)
	op.Parents[1].Grad = gB
}

func (op *Transpose) Backward(grad *tensor.Tensor) {
	if op.Parents[0].Grad == nil {
		op.Parents[0].Grad = tensor.Zeros(op.Parents[0].Value.Shape...)
	}

	gradM := matrix.TensorToMatrix(grad)
	gradTransposed, _ := matrix.Transposition(gradM)
	gradTransposedT := matrix.MatrixToTensor(gradTransposed)
	g, _ := tensor.Add(op.Parents[0].Grad, gradTransposedT)
	op.Parents[0].Grad = g
}

func (e *Engine) MatMul(a, b *graph.Node) *graph.Node {
	aM := matrix.TensorToMatrix(a.Value)
	bM := matrix.TensorToMatrix(b.Value)
	valM, err := matrix.MatMul(aM, bM)
	if err != nil {
		return nil
	}
	val := matrix.MatrixToTensor(valM)
	op := &MatMul{Parents: []*graph.Node{a, b}, A: a.Value, B: b.Value}
	n := graph.NewNode(val, []*graph.Node{a, b}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

func (e *Engine) Transpose(a *graph.Node) *graph.Node {
	aM := matrix.TensorToMatrix(a.Value)
	valM, err := matrix.Transposition(aM)
	if err != nil {
		return nil
	}
	val := matrix.MatrixToTensor(valM)
	op := &Transpose{Parents: []*graph.Node{a}}
	n := graph.NewNode(val, []*graph.Node{a}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

type Sum struct {
	Parents    []*graph.Node
	InputShape []int
}

func (op *Sum) Backward(grad *tensor.Tensor) {
	p := op.Parents[0]
	if p.Grad == nil {
		p.Grad = tensor.Zeros(op.InputShape...)
	}

	size := 1
	for _, d := range op.InputShape {
		size *= d
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = grad.Data[0]
	}
	strides := make([]int, len(op.InputShape))
	stride := 1
	for i := len(op.InputShape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= op.InputShape[i]
	}
	scaled := &tensor.Tensor{Data: data, Shape: append([]int{}, op.InputShape...), Strides: strides}

	g, _ := tensor.Add(p.Grad, scaled)
	p.Grad = g
}

func (e *Engine) Sum(a *graph.Node) *graph.Node {
	val := tensor.Sum(a.Value) // скаляр [1]
	inShape := append([]int{}, a.Value.Shape...)
	op := &Sum{Parents: []*graph.Node{a}, InputShape: inShape}
	n := graph.NewNode(val, []*graph.Node{a}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

// EXP
type Exp struct {
	Parents []*graph.Node
	Out     *tensor.Tensor
}

func (op *Exp) Backward(grad *tensor.Tensor) {
	p := op.Parents[0]
	if p.Grad == nil {
		p.Grad = tensor.Zeros(p.Value.Shape...)
	}
	gLocal, _ := tensor.Mul(op.Out, grad)
	g, _ := tensor.Add(p.Grad, gLocal)
	p.Grad = g
}

func (e *Engine) Exp(a *graph.Node) *graph.Node {
	val := tensor.Exp(a.Value)
	op := &Exp{Parents: []*graph.Node{a}, Out: val}
	n := graph.NewNode(val, []*graph.Node{a}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

// LOG
type Log struct {
	Parents []*graph.Node
	In      *tensor.Tensor
	Eps     float64
}

func (op *Log) Backward(grad *tensor.Tensor) {
	p := op.Parents[0]
	if p.Grad == nil {
		p.Grad = tensor.Zeros(p.Value.Shape...)
	}
	eps := op.Eps
	inv := tensor.Apply(op.In, func(x float64) float64 {
		if eps > 0 && x < eps {
			x = eps
		}
		return 1.0 / x
	})
	gLocal, _ := tensor.Mul(inv, grad)
	g, _ := tensor.Add(p.Grad, gLocal)
	p.Grad = g
}

func (e *Engine) Log(a *graph.Node) *graph.Node {
	val := tensor.Log(a.Value)
	op := &Log{Parents: []*graph.Node{a}, In: a.Value, Eps: 1e-12}
	n := graph.NewNode(val, []*graph.Node{a}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

// Reshape
type ReshapeOp struct {
	Parents  []*graph.Node
	InShape  []int
	OutShape []int
}

func (op *ReshapeOp) Backward(grad *tensor.Tensor) {
	p := op.Parents[0]
	if p.Grad == nil {
		p.Grad = tensor.Zeros(op.InShape...)
	}
	gradIn, err := tensor.Reshape(grad, op.InShape)
	if err != nil {
		return
	}
	g, _ := tensor.Add(p.Grad, gradIn)
	p.Grad = g
}

func (e *Engine) Reshape(a *graph.Node, newShape []int) *graph.Node {
	val, err := tensor.Reshape(a.Value, newShape)
	if err != nil {
		return nil
	}
	op := &ReshapeOp{
		Parents:  []*graph.Node{a},
		InShape:  append([]int{}, a.Value.Shape...),
		OutShape: append([]int{}, newShape...),
	}
	n := graph.NewNode(val, []*graph.Node{a}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}
