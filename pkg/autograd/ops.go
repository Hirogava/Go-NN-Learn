package autograd

import (
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

// MatMul
func (op *MatMul) Backward(grad *tensor.Tensor) {
	// Правило дифференцирования для матричного умножения:
	// d/dA (A @ B) = grad @ B^T
	// d/dB (A @ B) = A^T @ grad

	// Градиент для A
	if op.Parents[0].Grad == nil {
		op.Parents[0].Grad = tensor.Zeros(op.Parents[0].Value.Shape...)
	}

	// grad @ B^T
	bTransposed, _ := Transposition(op.B)
	gA_local, _ := MatMul(grad, bTransposed)
	gA, _ := tensor.Add(op.Parents[0].Grad, gA_local)
	op.Parents[0].Grad = gA

	// Градиент для B
	if op.Parents[1].Grad == nil {
		op.Parents[1].Grad = tensor.Zeros(op.Parents[1].Value.Shape...)
	}

	// A^T @ grad
	aTransposed, _ := Transposition(op.A)
	gB_local, _ := MatMul(aTransposed, grad)
	gB, _ := tensor.Add(op.Parents[1].Grad, gB_local)
	op.Parents[1].Grad = gB
}

func (e *Engine) MatMul(a, b *graph.Node) *graph.Node {
	val, err := MatMul(a.Value, b.Value) // используем функцию из main
	if err != nil {
		return nil
	}
	op := &MatMul{Parents: []*graph.Node{a, b}, A: a.Value, B: b.Value}
	n := graph.NewNode(val, []*graph.Node{a, b}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}

// Transpose - транспонирование матрицы
type Transpose struct {
	Parents []*graph.Node
}

func (op *Transpose) Backward(grad *tensor.Tensor) {
	// Правило дифференцирования для транспонирования:
	// d/dA (A^T) = (grad)^T
	// Транспонирование - линейная операция, поэтому градиент просто транспонируется

	if op.Parents[0].Grad == nil {
		op.Parents[0].Grad = tensor.Zeros(op.Parents[0].Value.Shape...)
	}

	// Транспонируем входящий градиент
	gradTransposed, _ := Transposition(grad)
	g, _ := tensor.Add(op.Parents[0].Grad, gradTransposed)
	op.Parents[0].Grad = g
}

func (e *Engine) Transpose(a *graph.Node) *graph.Node {
	val, err := Transposition(a.Value)
	if err != nil {
		return nil
	}
	op := &Transpose{Parents: []*graph.Node{a}}
	n := graph.NewNode(val, []*graph.Node{a}, op)
	e.Nodes = append(e.Nodes, n)
	return n
}
