package autograd

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
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
