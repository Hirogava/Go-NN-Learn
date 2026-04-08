package autograd

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// TestExampleDotSumGradient строит примитивный граф y = sum(a * b) и проверяет его градиента.
func TestExampleDotSumGradient(t *testing.T) {
	// Создание нодов a и b
	aProto := graph.NewNode(&tensor.Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}, Strides: []int{1}}, nil, nil)
	bProto := graph.NewNode(&tensor.Tensor{Data: []float64{4, 5, 6}, Shape: []int{3}, Strides: []int{1}}, nil, nil)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		// inputs[0] = a, inputs[1] = b
		return e.Sum(e.Mul(inputs[0], inputs[1]))
	}

	ok := CheckGradientEngine(build, []*graph.Node{aProto, bProto}, 1e-6, 1e-4)
	if !ok {
		t.Fatal("CheckGradientEngine failed for y = sum(a * b)")
	}

	// Дополнительно вычислим аналитические градиенты и проверим значения.
	e := NewEngine()
	a := graph.NewNode(&tensor.Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}, Strides: []int{1}}, nil, nil)
	b := graph.NewNode(&tensor.Tensor{Data: []float64{4, 5, 6}, Shape: []int{3}, Strides: []int{1}}, nil, nil)
	out := build(e, []*graph.Node{a, b})
	e.Backward(out)

	// Для y = sum(a * b): d/dA = b, d/dB = a
	wantA := []float64{4, 5, 6}
	wantB := []float64{1, 2, 3}
	for i := range wantA {
		if a.Grad == nil || a.Grad.Data[i] != wantA[i] {
			t.Fatalf("a.Grad[%d] = %v, want %v", i, a.Grad.Data[i], wantA[i])
		}
	}
	for i := range wantB {
		if b.Grad == nil || b.Grad.Data[i] != wantB[i] {
			t.Fatalf("b.Grad[%d] = %v, want %v", i, b.Grad.Data[i], wantB[i])
		}
	}
}

// численный градиент
func numericalGrad(f func(x *graph.Node) *graph.Node, x *tensor.Tensor, eps float64) *tensor.Tensor {
	grad := tensor.Zeros(x.Shape...)
	for i := range x.Data {
		orig := x.Data[i]

		x.Data[i] = orig + eps
		y1 := SumTensor(f(&graph.Node{Value: x}).Value)

		x.Data[i] = orig - eps
		y2 := SumTensor(f(&graph.Node{Value: x}).Value)

		grad.Data[i] = (y1 - y2) / (2 * eps)
		x.Data[i] = orig
	}
	return grad
}

func SumTensor(t *tensor.Tensor) float64 {
	s := 0.0
	for _, v := range t.Data {
		s += v
	}
	return s
}

// grad_check
func gradCheck(t *testing.T, f func(x *graph.Node) *graph.Node, x *tensor.Tensor, eps, tol float64) {
	node := &graph.Node{Value: x}
	node.Grad = nil

	// forward + backward
	eng := NewEngine()
	y := f(node)
	eng.Backward(y)

	// численный градиент
	numGrad := numericalGrad(f, x, eps)

	// сравнение
	for i := range x.Data {
		a := node.Grad.Data[i]
		n := numGrad.Data[i]
		if math.Abs(a-n) > tol {
			t.Fatalf("grad check failed at index %d: analytic=%v, numeric=%v", i, a, n)
		}
	}
}

// тест Reshape
func TestReshapeGradCheck(t *testing.T) {
	x := tensor.Randn([]int{2, 3}, 42) // фиксированный seed
	f := func(xn *graph.Node) *graph.Node {
		eng := NewEngine()
		return eng.Reshape(xn, []int{3, 2})
	}

	gradCheck(t, f, x, 1e-6, 1e-4)
}

// тест Transpose
func TestTransposeGradCheck(t *testing.T) {
	x := tensor.Randn([]int{2, 3}, 42)
	f := func(xn *graph.Node) *graph.Node {

		eng := NewEngine()
		y := eng.Transpose(xn)
		return eng.Sum(y) // скалярный выход
	}

	gradCheck(t, f, x, 1e-6, 1e-4)
}
