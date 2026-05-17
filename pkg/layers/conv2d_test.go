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
	conv := NewConv2D(1, 2, 2, 1, 0, initFuncOnes, ZeroInit())

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

	want := []float64{12, 16, 24, 28, 12, 16, 24, 28}
	for i := range want {
		if out.Value.Data[i] != want[i] {
			t.Fatalf("output[%d] = %v, want %v; full output %v", i, out.Value.Data[i], want[i], out.Value.Data)
		}
	}
}

func TestConv2DForward_WithPadding(t *testing.T) {
	conv := NewConv2D(1, 1, 3, 1, 1, initFuncOnes, ZeroInit())

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

	want := []float64{12, 21, 16, 27, 45, 33, 24, 39, 28}
	for i := range want {
		if out.Value.Data[i] != want[i] {
			t.Fatalf("output[%d] = %v, want %v; full output %v", i, out.Value.Data[i], want[i], out.Value.Data)
		}
	}
}

func TestConv2DForward_Stride2(t *testing.T) {
	conv := NewConv2D(1, 1, 2, 2, 0, initFuncOnes, ZeroInit())

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

func TestConv2DForward_PaddingModesAndDilation(t *testing.T) {
	input := &graph.Node{
		Value: &tensor.Tensor{
			Data: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
				16, 17, 18, 19, 20,
				21, 22, 23, 24, 25,
			},
			Shape:   []int{1, 1, 5, 5},
			Strides: []int{25, 25, 5, 1},
		},
	}

	valid := NewConv2DWithConfig(Conv2DConfig{
		InChannels:  1,
		OutChannels: 1,
		KernelSize:  3,
		Stride:      2,
		Padding:     "valid",
		Dilation:    2,
		WInit:       initFuncOnes,
		BInit:       ZeroInit(),
	})
	validOut := valid.Forward(input)
	if !sameInts(validOut.Value.Shape, []int{1, 1, 1, 1}) {
		t.Fatalf("valid+dilation shape = %v, want [1 1 1 1]", validOut.Value.Shape)
	}
	if validOut.Value.Data[0] != 117 {
		t.Fatalf("valid+dilation output = %v, want 117", validOut.Value.Data[0])
	}

	same := NewConv2DWithConfig(Conv2DConfig{
		InChannels:  1,
		OutChannels: 1,
		KernelSize:  3,
		Stride:      2,
		Padding:     "same",
		Dilation:    1,
		WInit:       initFuncOnes,
		BInit:       ZeroInit(),
	})
	sameOut := same.Forward(input)
	if !sameInts(sameOut.Value.Shape, []int{1, 1, 3, 3}) {
		t.Fatalf("same shape = %v, want [1 1 3 3]", sameOut.Value.Shape)
	}
	want := []float64{16, 33, 28, 69, 117, 87, 76, 123, 88}
	for i := range want {
		if sameOut.Value.Data[i] != want[i] {
			t.Fatalf("same output[%d] = %v, want %v; full output %v", i, sameOut.Value.Data[i], want[i], sameOut.Value.Data)
		}
	}
}

func TestConv2DForward_BatchLayout(t *testing.T) {
	conv := NewConv2D(1, 1, 2, 1, 0, initFuncOnes, ZeroInit())
	input := &graph.Node{
		Value: &tensor.Tensor{
			Data: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				10, 20, 30,
				40, 50, 60,
				70, 80, 90,
			},
			Shape:   []int{2, 1, 3, 3},
			Strides: []int{9, 9, 3, 1},
		},
	}

	out := conv.Forward(input)
	if !sameInts(out.Value.Shape, []int{2, 1, 2, 2}) {
		t.Fatalf("shape = %v, want [2 1 2 2]", out.Value.Shape)
	}
	want := []float64{12, 16, 24, 28, 120, 160, 240, 280}
	for i := range want {
		if out.Value.Data[i] != want[i] {
			t.Fatalf("output[%d] = %v, want %v; full output %v", i, out.Value.Data[i], want[i], out.Value.Data)
		}
	}
}

func TestConv2DBackward(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	conv := NewConv2D(2, 3, 2, 1, 0, initFuncFixed, ZeroInit())

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

func TestConv2DBackward_WithGraphContext(t *testing.T) {
	ctx := autograd.NewGraph()
	autograd.SetGraph(ctx)
	defer autograd.ClearGraph()

	conv := NewConv2DWithConfig(Conv2DConfig{
		InChannels:  1,
		OutChannels: 1,
		KernelSize:  2,
		Stride:      1,
		Padding:     "same",
		Dilation:    1,
		WInit:       initFuncOnes,
		BInit:       ZeroInit(),
	})
	input := ctx.RequireGrad(&tensor.Tensor{
		Data: []float64{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		},
		Shape:   []int{1, 1, 3, 3},
		Strides: []int{9, 9, 3, 1},
	})

	out := conv.Forward(input)
	ctx.Backward(out)

	if input.Grad == nil {
		t.Fatal("input.Grad is nil after ctx.Backward")
	}
	if conv.weights.Grad == nil {
		t.Fatal("weights.Grad is nil after ctx.Backward")
	}
	if conv.bias.Grad == nil {
		t.Fatal("bias.Grad is nil after ctx.Backward")
	}
}

func TestConv2DNumericalGradient(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	conv := NewConv2D(1, 2, 2, 1, 0, initFuncFixed, ZeroInit())
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

func TestConv2DNumericalGradient_SameStrideDilation(t *testing.T) {
	autograd.SetGraph(autograd.NewGraph())
	defer autograd.ClearGraph()

	conv := NewConv2DWithConfig(Conv2DConfig{
		InChannels:  1,
		OutChannels: 1,
		KernelSize:  2,
		Stride:      2,
		Padding:     "same",
		Dilation:    2,
		WInit:       initFuncFixed,
		BInit:       ZeroInit(),
	})
	inputData := tensor.Randn([]int{1, 1, 4, 4}, 7)
	ctx := autograd.GetGraph()
	inputNode := ctx.RequireGrad(inputData)

	out := conv.Forward(inputNode)
	op := out.Operation.(*conv2dOp)
	op.Backward(tensor.Ones(out.Value.Shape...))

	eps := 1e-5
	tol := 1e-2
	for i := 0; i < len(inputData.Data); i++ {
		orig := inputData.Data[i]
		inputData.Data[i] = orig + eps
		outPlus := conv.Forward(&graph.Node{Value: inputData})
		sumPlus := sumSlice(outPlus.Value.Data)
		inputData.Data[i] = orig - eps
		outMinus := conv.Forward(&graph.Node{Value: inputData})
		sumMinus := sumSlice(outMinus.Value.Data)
		inputData.Data[i] = orig

		numerical := (sumPlus - sumMinus) / (2 * eps)
		analytical := inputNode.Grad.Data[i]
		if math.Abs(numerical-analytical) > tol {
			t.Fatalf("input[%d] numerical=%v analytical=%v", i, numerical, analytical)
		}
	}
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sumSlice(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum
}
