package autograd_test

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestGraphContext_GradEnabledHelper(t *testing.T) {
	if autograd.GradEnabled() {
		t.Fatalf("expected GradEnabled() to be false when no context is set")
	}

	ctx := autograd.NewGraph()
	autograd.SetGraph(ctx)
	if !autograd.GradEnabled() {
		t.Fatalf("expected GradEnabled() to be true after SetGraph(NewGraph())")
	}

	ctx.NoGrad()
	if autograd.GradEnabled() {
		t.Fatalf("expected GradEnabled() to be false after ctx.NoGrad()")
	}

	ctx.WithGrad()
	if !autograd.GradEnabled() {
		t.Fatalf("expected GradEnabled() to be true after ctx.WithGrad()")
	}
}

func TestGraphContext_BackwardAndRelease(t *testing.T) {
	ctx := autograd.NewGraph()
	ctx.WithGrad()
	autograd.SetGraph(ctx)

	e := ctx.Engine()

	a := graph.NewNode(
		&tensor.Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}, Strides: []int{1}},
		nil,
		nil,
	)
	b := graph.NewNode(
		&tensor.Tensor{Data: []float64{4, 5, 6}, Shape: []int{3}, Strides: []int{1}},
		nil,
		nil,
	)

	out := e.Sum(e.Mul(a, b))

	ctx.Backward(out)

	wantA := []float64{4, 5, 6}
	wantB := []float64{1, 2, 3}

	if a.Grad == nil || len(a.Grad.Data) != len(wantA) {
		t.Fatalf("unexpected a.Grad size: %+v", a.Grad)
	}
	if b.Grad == nil || len(b.Grad.Data) != len(wantB) {
		t.Fatalf("unexpected b.Grad size: %+v", b.Grad)
	}

	for i, v := range wantA {
		if a.Grad.Data[i] != v {
			t.Fatalf("a.Grad[%d] = %v, want %v", i, a.Grad.Data[i], v)
		}
	}
	for i, v := range wantB {
		if b.Grad.Data[i] != v {
			t.Fatalf("b.Grad[%d] = %v, want %v", i, b.Grad.Data[i], v)
		}
	}

	if autograd.GetGraph() != nil {
		t.Fatalf("expected GetGraph() to return nil after Backward/release")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic on second Backward (context already released)")
		}
	}()
	ctx.Backward(out)
}

