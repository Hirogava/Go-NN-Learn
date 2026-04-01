package autograd

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestSoftPlusForward(t *testing.T) {
	e := NewEngine()

	input := &tensor.Tensor{
		Data:    []float64{-2.0, -1.0, 0.0, 1.0, 2.0},
		Shape:   []int{5},
		Strides: []int{1},
	}
	inputNode := e.RequireGrad(input)

	out := e.SoftPlus(inputNode)

	for i, x := range input.Data {
		want := math.Log(1.0 + math.Exp(x))
		if math.Abs(out.Value.Data[i]-want) > 1e-10 {
			t.Fatalf("SoftPlus forward mismatch at %d: got=%v want=%v", i, out.Value.Data[i], want)
		}
	}
}

func TestSoftPlusGradCheck(t *testing.T) {
	// Сгладим точки так, чтобы exp не переполнялся в численном градиенте.
	x := newTensor([]float64{-3.0, -1.2, -0.3, 0.0, 0.2, 1.1, 2.5}, 7)
	inNode := graph.NewNode(x, nil, nil)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.SoftPlus(inputs[0])
	}

	if !CheckGradientEngine(build, []*graph.Node{inNode}, 1e-6, 1e-4) {
		t.Error("SoftPlus gradient check failed")
	}
}

