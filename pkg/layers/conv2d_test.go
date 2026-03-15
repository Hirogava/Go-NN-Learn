package layers

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func initFuncOnes(data []float64) {
	for i := range data {
		data[i] = 1.0
	}
}

func TestConv2DForward_Simple(t *testing.T) {
	conv := NewConv2D(1, 2, 2, 1, 0, initFuncOnes)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
			Shape:   []int{1, 1, 3, 3},
			Strides: []int{9, 9, 3, 1},
		},
	}

	out := conv.Forward(input)

	if len(out.Value.Shape) != 4 {
		t.Fatalf("expected 4D output, got %dD", len(out.Value.Shape))
	}
	if out.Value.Shape[0] != 1 || out.Value.Shape[1] != 2 || out.Value.Shape[2] != 2 || out.Value.Shape[3] != 2 {
		t.Errorf("expected shape [1,2,2,2], got %v", out.Value.Shape)
	}

	sum := 0.0
	for _, v := range out.Value.Data {
		sum += v
	}
	if sum == 0 {
		t.Error("output is all zeros - convolution not implemented correctly")
	}
}

func TestConv2DForward_WithPadding(t *testing.T) {
	conv := NewConv2D(1, 1, 3, 1, 1, initFuncOnes)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
			Shape:   []int{1, 1, 3, 3},
			Strides: []int{9, 9, 3, 1},
		},
	}

	out := conv.Forward(input)

	expectedH, expectedW := 3, 3
	if out.Value.Shape[2] != expectedH || out.Value.Shape[3] != expectedW {
		t.Errorf("expected output spatial [%d,%d], got [%d,%d]", expectedH, expectedW, out.Value.Shape[2], out.Value.Shape[3])
	}
}

func TestConv2DForward_Stride2(t *testing.T) {
	conv := NewConv2D(1, 1, 2, 2, 0, initFuncOnes)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    make([]float64, 1*1*5*5),
			Shape:   []int{1, 1, 5, 5},
			Strides: []int{25, 25, 5, 1},
		},
	}
	for i := range input.Value.Data {
		input.Value.Data[i] = 1.0
	}

	out := conv.Forward(input)

	if out.Value.Shape[2] != 2 || out.Value.Shape[3] != 2 {
		t.Errorf("expected 2x2 output with stride=2, got %dx%d", out.Value.Shape[2], out.Value.Shape[3])
	}
}

func TestConv2DBackward(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	conv := NewConv2D(2, 3, 2, 1, 0, initFuncFixed)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8},
			Shape:   []int{1, 2, 2, 4},
			Strides: []int{16, 8, 4, 1},
		},
	}

	out := conv.Forward(input)
	if out.Operation == nil {
		t.Fatal("Operation is nil - GradEnabled may be false")
	}

	grad := tensor.Ones(out.Value.Shape...)
	op := out.Operation.(*conv2dOp)
	op.Backward(grad)

	if input.Grad == nil {
		t.Fatal("input.Grad is nil after backward")
	}
	if conv.weights.Grad == nil {
		t.Fatal("weights.Grad is nil after backward")
	}
	if conv.bias.Grad == nil {
		t.Fatal("bias.Grad is nil after backward")
	}
	if len(input.Grad.Shape) != 4 || input.Grad.Shape[0] != 1 || input.Grad.Shape[1] != 2 {
		t.Errorf("expected input grad shape [1,2,2,4], got %v", input.Grad.Shape)
	}
}

func TestConv2DNumericalGradient(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	conv := NewConv2D(1, 2, 2, 1, 0, initFuncFixed)
	inputData := tensor.Randn([]int{1, 1, 4, 4}, 42)
	ctx := autograd.GetGraph()
	inputNode := ctx.RequireGrad(inputData)

	out := conv.Forward(inputNode)
	if out.Operation == nil {
		t.Fatal("need GradEnabled for backward")
	}
	grad := tensor.Ones(out.Value.Shape...)
	op := out.Operation.(*conv2dOp)
	op.Backward(grad)

	eps := 1e-5
	tol := 1e-2
	for i := 0; i < len(inputData.Data) && i < 8; i++ {
		orig := inputData.Data[i]
		inputData.Data[i] = orig + eps
		outPlus := conv.Forward(&graph.Node{Value: inputData})
		sumPlus := 0.0
		for _, v := range outPlus.Value.Data {
			sumPlus += v
		}
		inputData.Data[i] = orig - eps
		outMinus := conv.Forward(&graph.Node{Value: inputData})
		sumMinus := 0.0
		for _, v := range outMinus.Value.Data {
			sumMinus += v
		}
		inputData.Data[i] = orig

		numerical := (sumPlus - sumMinus) / (2 * eps)
		analytical := inputNode.Grad.Data[i]
		relErr := math.Abs(numerical - analytical)
		if math.Abs(numerical) > 1e-10 {
			relErr /= math.Abs(numerical)
		}
		if relErr > tol {
			t.Errorf("input[%d] numerical=%v analytical=%v relErr=%v", i, numerical, analytical, relErr)
		}
	}
}
