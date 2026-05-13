package layers

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

func TestActivations(t *testing.T) {
	input := &tensor.Tensor{
		Data:    []float64{1.0, -1.0, 2.0, -2.0},
		Shape:   []int{1, 4},
		Strides: []int{4, 1},
	}

	testCases := []struct {
		name  string
		layer Layer
	}{
		{"ReLU", NewReLU()},
		{"Sigmoid", NewSigmoid()},
		{"Tanh", NewTanh()},
		{"LeakyReLU", NewLeakyReLU(0.01)},
		{"ELU", NewELU(1.0)},
		{"SoftPlus", NewSoftPlus()},
		{"GELU", NewGELU()},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := autograd.NewGraph()
			autograd.SetGraph(ctx)
			inputNode := ctx.RequireGrad(input)

			out := tc.layer.Forward(inputNode)
			if out == nil {
				t.Fatalf("%s Forward returned nil", tc.name)
			}
			if len(out.Value.Data) != 4 {
				t.Errorf("%s output length mismatch: expected 4, got %d", tc.name, len(out.Value.Data))
			}
			
			// Test Params
			params := tc.layer.Params()
			if params != nil {
				t.Errorf("%s should have nil params, got %v", tc.name, params)
			}
			
			// Test Backward
			ctx.Backward(out)
			if inputNode.Grad == nil {
				t.Errorf("%s input gradient is nil after backward", tc.name)
			}
			ctx.ZeroGrad()
		})
	}
}

func TestSoftmaxLayer(t *testing.T) {
	ctx := autograd.NewGraph()
	autograd.SetGraph(ctx)

	input := &tensor.Tensor{
		Data:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
	}
	inputNode := ctx.RequireGrad(input)

	layer := NewSoftmax()
	out := layer.Forward(inputNode)

	if out == nil {
		t.Fatal("Softmax Forward returned nil")
	}

	// Verify sums are 1.0 for each row
	for i := 0; i < 2; i++ {
		sum := 0.0
		for j := 0; j < 3; j++ {
			sum += out.Value.Data[i*3+j]
		}
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("Softmax row %d sum mismatch: expected 1.0, got %f", i, sum)
		}
	}

	ctx.Backward(out)
	if inputNode.Grad == nil {
		t.Error("Softmax input gradient is nil after backward")
	}
}
