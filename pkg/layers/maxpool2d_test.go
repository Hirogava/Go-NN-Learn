package layers

import (
	"reflect"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestMaxPooling2DForward(t *testing.T) {
	pool := NewMaxPooling2D(2, 2)
	input := &graph.Node{
		Value: &tensor.Tensor{
			Data: []float64{
				1, 3, 2, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7,
			},
			Shape:   []int{1, 1, 4, 4},
			Strides: []int{16, 16, 4, 1},
		},
	}

	out := pool.Forward(input)

	expectedShape := []int{1, 1, 2, 2}
	if !reflect.DeepEqual(out.Value.Shape, expectedShape) {
		t.Fatalf("expected shape %v, got %v", expectedShape, out.Value.Shape)
	}

	expectedData := []float64{6, 8, 9, 7}
	if !reflect.DeepEqual(out.Value.Data, expectedData) {
		t.Fatalf("expected data %v, got %v", expectedData, out.Value.Data)
	}
}

func TestMaxPooling2DForwardMultiChannel(t *testing.T) {
	pool := NewMaxPooling2D(2, 2)
	input := &graph.Node{
		Value: &tensor.Tensor{
			Data: []float64{
				1, 2,
				3, 4,
				8, 7,
				6, 5,
			},
			Shape:   []int{1, 2, 2, 2},
			Strides: []int{8, 4, 2, 1},
		},
	}

	out := pool.Forward(input)

	expectedShape := []int{1, 2, 1, 1}
	if !reflect.DeepEqual(out.Value.Shape, expectedShape) {
		t.Fatalf("expected shape %v, got %v", expectedShape, out.Value.Shape)
	}

	expectedData := []float64{4, 8}
	if !reflect.DeepEqual(out.Value.Data, expectedData) {
		t.Fatalf("expected data %v, got %v", expectedData, out.Value.Data)
	}
}

func TestMaxPooling2DBackward(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	pool := NewMaxPooling2D(2, 2)
	ctx := autograd.GetGraph()
	input := ctx.RequireGrad(&tensor.Tensor{
		Data: []float64{
			1, 3, 2, 4,
			5, 6, 7, 8,
			9, 1, 2, 3,
			4, 5, 6, 7,
		},
		Shape:   []int{1, 1, 4, 4},
		Strides: []int{16, 16, 4, 1},
	})

	out := pool.Forward(input)
	if out.Operation == nil {
		t.Fatal("Operation is nil - GradEnabled may be false")
	}

	out.Operation.Backward(tensor.Ones(out.Value.Shape...))

	expectedGrad := []float64{
		0, 0, 0, 0,
		0, 1, 0, 1,
		1, 0, 0, 0,
		0, 0, 0, 1,
	}
	if !reflect.DeepEqual(input.Grad.Data, expectedGrad) {
		t.Fatalf("expected grad %v, got %v", expectedGrad, input.Grad.Data)
	}
}

func TestMaxPooling2DBackwardOverlappingWindows(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	pool := NewMaxPooling2D(2, 1)
	ctx := autograd.GetGraph()
	input := ctx.RequireGrad(&tensor.Tensor{
		Data: []float64{
			1, 2, 3,
			4, 9, 6,
			7, 8, 5,
		},
		Shape:   []int{1, 1, 3, 3},
		Strides: []int{9, 9, 3, 1},
	})

	out := pool.Forward(input)
	out.Operation.Backward(tensor.Ones(out.Value.Shape...))

	expectedGrad := []float64{
		0, 0, 0,
		0, 4, 0,
		0, 0, 0,
	}
	if !reflect.DeepEqual(input.Grad.Data, expectedGrad) {
		t.Fatalf("expected grad %v, got %v", expectedGrad, input.Grad.Data)
	}
}
