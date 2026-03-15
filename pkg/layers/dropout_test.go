package layers

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestDropoutForward(t *testing.T) {
	rand.Seed(42)

	// Правильная инициализация ноды через конструктор
	input := graph.NewNode(&tensor.Tensor{
		Data:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
	}, nil, nil)

	dropout := NewDropout(0.5)
	dropout.Train() // Используем метод Train()

	output := dropout.Forward(input)

	// Достаем маску из операции для вывода
	op := output.Operation.(*dropoutOp)

	fmt.Println("Input:", input.Value.Data)
	fmt.Println("Output (training):", output.Value.Data)
	fmt.Println("Mask used:", op.mask.Data)

	zeroCount := 0
	for _, val := range output.Value.Data {
		if val == 0.0 {
			zeroCount++
		}
	}
	fmt.Printf("Zeros: %d out of %d\n", zeroCount, len(output.Value.Data))
}

func TestDropoutInference(t *testing.T) {
	input := graph.NewNode(&tensor.Tensor{
		Data:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
	}, nil, nil)

	dropout := NewDropout(0.5)
	dropout.Eval() // Режим инференса

	output := dropout.Forward(input)

	fmt.Println("\nInput:", input.Value.Data)
	fmt.Println("Output (inference):", output.Value.Data)

	for i := range input.Value.Data {
		if input.Value.Data[i] != output.Value.Data[i] {
			t.Errorf("In inference mode, output should equal input")
		}
	}
}

func TestDropoutBackward(t *testing.T) {
	rand.Seed(42)

	input := graph.NewNode(&tensor.Tensor{
		Data:    []float64{1.0, 2.0, 3.0, 4.0},
		Shape:   []int{2, 2},
		Strides: []int{2, 1},
	}, nil, nil)

	dropout := NewDropout(0.5)
	dropout.Train()

	output := dropout.Forward(input)

	grad := &tensor.Tensor{
		Data:    []float64{1.0, 1.0, 1.0, 1.0},
		Shape:   []int{2, 2},
		Strides: []int{2, 1},
	}

	// Проверяем наличие операции перед приведением
	if output.Operation == nil {
		t.Fatal("Operation is nil. Graph did not record the forward pass.")
	}

	op := output.Operation.(*dropoutOp)
	op.Backward(grad)

	fmt.Println("\nGradient input:", grad.Data)
	fmt.Println("Mask used:", op.mask.Data)
	fmt.Println("Gradient output (accumulated in input):", input.Grad.Data)
}

func TestDropoutRates(t *testing.T) {
	rates := []float64{0.2, 0.5, 0.8}

	for _, rate := range rates {
		rand.Seed(42)

		data := make([]float64, 1000)
		for i := range data {
			data[i] = 1.0
		}

		input := graph.NewNode(&tensor.Tensor{
			Data:    data,
			Shape:   []int{1000},
			Strides: []int{1},
		}, nil, nil)

		dropout := NewDropout(rate)
		dropout.Train()
		output := dropout.Forward(input)

		zeroCount := 0
		for _, val := range output.Value.Data {
			if val == 0.0 {
				zeroCount++
			}
		}

		actualRate := float64(zeroCount) / float64(len(output.Value.Data))
		fmt.Printf("Rate: %.1f, Actual dropout: %.3f\n", rate, actualRate)
	}
}
