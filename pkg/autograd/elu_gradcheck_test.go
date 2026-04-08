package autograd

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestELUGradCheck(t *testing.T) {
	x := newTensor([]float64{1.0, -0.5, 0.0, -2.0, 0.3}, 5)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.ELU(inputs[0])
	}

	inputNode := graph.NewNode(x, nil, nil)
	if !CheckGradientEngine(build, []*graph.Node{inputNode}, 1e-6, 1e-4) {
		t.Error("ELU gradient check failed for vector")
	}
}

func TestELUGradCheckMatrix(t *testing.T) {
	x := tensor.Randn([]int{5, 4}, 202)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		return e.ELU(inputs[0])
	}

	inputNode := graph.NewNode(x, nil, nil)
	if !CheckGradientEngine(build, []*graph.Node{inputNode}, 1e-6, 1e-4) {
		t.Error("ELU gradient check failed for matrix")
	}
}
