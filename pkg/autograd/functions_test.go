package autograd

import (
	"fmt"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

func TestFunctions(t *testing.T) {
	e := NewEngine()
	input := &tensor.Tensor{
		Data:    []float64{1.0, -2.0, 3.0, -4.0},
		Shape:   []int{1, 4},
		Strides: []int{4, 1},
	}
	inputNode := e.RequireGrad(input)

	reluNode := e.ReLU(inputNode)
	e.Backward(reluNode)
	fmt.Println(reluNode.Value.Data)
	fmt.Println(inputNode.Grad.Data)
	e.ZeroGrad()

	sigmoidNode := e.Sigmoid(inputNode)
	e.Backward(sigmoidNode)
	fmt.Println(sigmoidNode.Value.Data)
	fmt.Println(inputNode.Grad.Data)
	e.ZeroGrad()

	tanhNode := e.Tanh(inputNode)
	e.Backward(tanhNode)
	fmt.Println(tanhNode.Value.Data)
	fmt.Println(inputNode.Grad.Data)
	e.ZeroGrad()

	input2 := &tensor.Tensor{
		Data:    []float64{2.0, 1.0, 0.0, 0.0, 1.0, 2.0},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
	}
	target := &tensor.Tensor{
		Data:    []float64{0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
	}
	inputNode2 := e.RequireGrad(input2)
	lossNode := e.SoftmaxCrossEntropy(inputNode2, target)
	e.Backward(lossNode)
	fmt.Println(lossNode.Value.Data)
	fmt.Println(inputNode2.Grad.Data)
}
