package layers

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

func TestDropoutForward(t *testing.T) {
	rand.Seed(42)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			Shape:   []int{2, 3},
			Strides: []int{3, 1},
		},
	}

	dropout := NewDropout(0.5)
	dropout.SetTraining(true)

	output := dropout.Forward(input)

	fmt.Println("Input:", input.Value.Data)
	fmt.Println("Output (training):", output.Value.Data)
	fmt.Println("Mask:", dropout.mask.Data)

	zeroCount := 0
	for _, val := range output.Value.Data {
		if val == 0.0 {
			zeroCount++
		}
	}
	fmt.Printf("Zeros: %d out of %d\n", zeroCount, len(output.Value.Data))
}

func TestDropoutInference(t *testing.T) {
	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			Shape:   []int{2, 3},
			Strides: []int{3, 1},
		},
	}

	dropout := NewDropout(0.5)
	dropout.SetTraining(false)

	output := dropout.Forward(input)

	fmt.Println("\nInput:", input.Value.Data)
	fmt.Println("Output (inference):", output.Value.Data)

	for i := range input.Value.Data {
		if input.Value.Data[i] != output.Value.Data[i] {
			t.Errorf("In inference mode, output should equal input")
		}
	}
}

// TestDropoutBackward проверяет обратное распространение
func TestDropoutBackward(t *testing.T) {
	rand.Seed(42)

	input := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0, 4.0},
			Shape:   []int{2, 2},
			Strides: []int{2, 1},
		},
	}

	dropout := NewDropout(0.5)
	dropout.SetTraining(true)

	output := dropout.Forward(input)

	grad := &tensor.Tensor{
		Data:    []float64{1.0, 1.0, 1.0, 1.0},
		Shape:   []int{2, 2},
		Strides: []int{2, 1},
	}

	op := output.Operation.(*dropoutOp)
	op.Backward(grad)

	fmt.Println("\nGradient input:", grad.Data)
	fmt.Println("Mask:", dropout.mask.Data)
	fmt.Println("Gradient output:", input.Grad.Data)
}

func TestDropoutRates(t *testing.T) {
	rates := []float64{0.2, 0.5, 0.8}

	for _, rate := range rates {
		rand.Seed(42)

		input := &graph.Node{
			Value: &tensor.Tensor{
				Data:    make([]float64, 1000),
				Shape:   []int{1000},
				Strides: []int{1},
			},
		}

		for i := range input.Value.Data {
			input.Value.Data[i] = 1.0
		}

		dropout := NewDropout(rate)
		dropout.SetTraining(true)
		output := dropout.Forward(input)

		zeroCount := 0
		for _, val := range output.Value.Data {
			if val == 0.0 {
				zeroCount++
			}
		}

		actualRate := float64(zeroCount) / float64(len(output.Value.Data))
		fmt.Printf("\nRate: %.1f, Actual dropout: %.3f\n", rate, actualRate)
	}
}
