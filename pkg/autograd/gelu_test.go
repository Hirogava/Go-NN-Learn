package autograd

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestGELUForward_Reference(t *testing.T) {
	e := NewEngine()
	// Φ(1) ≈ 0.8413447460685429
	x1 := 1.0
	want1 := 0.8413447460685429
	// x=0 -> 0
	x0 := 0.0

	in := graph.NewNode(&tensor.Tensor{
		Data:    []float64{x0, x1, -2.0},
		Shape:   []int{3},
		Strides: []int{1},
	}, nil, nil)
	e.Nodes = append(e.Nodes, in)
	out := e.GELU(in)

	if math.Abs(out.Value.Data[0]) > 1e-15 {
		t.Errorf("GELU(0) want 0, got %v", out.Value.Data[0])
	}
	if math.Abs(out.Value.Data[1]-want1) > 1e-12 {
		t.Errorf("GELU(1) want %v, got %v", want1, out.Value.Data[1])
	}
	// y = -2 * Φ(-2); Φ(-2) ≈ 0.022750131948179195
	wantNeg2 := -2.0 * 0.022750131948179195
	if math.Abs(out.Value.Data[2]-wantNeg2) > 1e-12 {
		t.Errorf("GELU(-2) want %v, got %v", wantNeg2, out.Value.Data[2])
	}
}

func TestGELUGradCheck(t *testing.T) {
	inp := newTensor([]float64{-2.0, -0.5, 0.0, 0.7, 1.5, 3.0}, 2, 3)
	inNode := graph.NewNode(inp, nil, nil)

	build := func(e *Engine, inputs []*graph.Node) *graph.Node {
		g := e.GELU(inputs[0])
		return e.Sum(g)
	}

	if !CheckGradientEngine(build, []*graph.Node{inNode}, 1e-6, 1e-4) {
		t.Error("GELU gradient check failed")
	}
}

func TestGELUForward_LargeMagnitude(t *testing.T) {
	e := NewEngine()
	// Для больших x: Φ(x)→1, GELU(x)→x; для сильно отрицательных: Φ(x)→0, GELU(x)→0
	pos := graph.NewNode(&tensor.Tensor{Data: []float64{12.0}, Shape: []int{1}, Strides: []int{1}}, nil, nil)
	neg := graph.NewNode(&tensor.Tensor{Data: []float64{-12.0}, Shape: []int{1}, Strides: []int{1}}, nil, nil)
	e.Nodes = append(e.Nodes, pos, neg)

	outP := e.GELU(pos)
	outN := e.GELU(neg)

	if math.Abs(outP.Value.Data[0]-12.0) > 1e-6 {
		t.Errorf("GELU(large +) want ≈x, got %v", outP.Value.Data[0])
	}
	if math.Abs(outN.Value.Data[0]) > 1e-6 {
		t.Errorf("GELU(large -) want ≈0, got %v", outN.Value.Data[0])
	}
}
