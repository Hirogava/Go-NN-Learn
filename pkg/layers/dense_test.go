package layers

import (
	"fmt"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func initFuncFixed(data []float64) {
	for i := range data {
		data[i] = float64(i + 1)
	}
}

func TestDenseForwardVector(t *testing.T) {
	dense := NewDense(3, 2, initFuncFixed)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0},
			Shape:   []int{3},
			Strides: []int{1},
		},
	}

	output := dense.Forward(input)

	fmt.Println(output.Value.Data)
}

func TestDenseForwardMatrix(t *testing.T) {
	dense := NewDense(3, 2, initFuncFixed)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			Shape:   []int{2, 3},
			Strides: []int{3, 1},
		},
	}

	output := dense.Forward(input)

	fmt.Println(output.Value.Data)
}

func TestDenseBackward(t *testing.T) {
	dense := NewDense(3, 2, initFuncFixed)

	// ПРАВИЛЬНО: используем NewNode
	xVal := &tensor.Tensor{
		Data:    []float64{1.0, 2.0, 3.0},
		Shape:   []int{1, 3},
		Strides: []int{3, 1},
	}
	input := graph.NewNode(xVal, nil, nil)

	output := dense.Forward(input)

	if output.Operation == nil {
		t.Fatal("Operation is nil! Check if Forward logic uses graph.NewNode")
	}

	grad := &tensor.Tensor{
		Data:    []float64{1.0, 1.0},
		Shape:   []int{1, 2},
		Strides: []int{2, 1},
	}

	output.Operation.Backward(grad)
	// ... дальше проверки fmt.Println ...
}
