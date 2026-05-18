package layers

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestLayerNormVector(t *testing.T) {
	v := tensor.Vector{1, 2, 3, 4}
	gamma := []float64{1, 1, 1, 1}
	beta := []float64{0, 0, 0, 0}
	out := LayerNormVector(v, gamma, beta, 1e-5)

	var sum float64
	for _, el := range out {
		sum += el
	}
	mean := sum / float64(len(out))
	if math.Abs(mean) > 1e-5 {
		t.Errorf("LayerNormVector mean = %v, want ~0", mean)
	}

	var variance float64
	for _, el := range out {
		variance += el * el
	}
	variance /= float64(len(out))
	if math.Abs(variance-1.0) > 1e-4 {
		t.Errorf("LayerNormVector variance = %v, want ~1", variance)
	}
}

func TestLayerNormForward(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(3, engine)

	inputData := tensor.Zeros(2, 3)
	inputData.Data = []float64{1, 2, 3, 10, 20, 30}
	input := graph.NewNode(inputData, nil, nil)

	out := ln.Forward(input)
	if len(out.Value.Shape) != 2 || out.Value.Shape[0] != 2 || out.Value.Shape[1] != 3 {
		t.Fatalf("output shape = %v, want [2, 3]", out.Value.Shape)
	}

	for i := 0; i < 2; i++ {
		var sum float64
		for j := 0; j < 3; j++ {
			sum += out.Value.Data[i*3+j]
		}
		mean := sum / 3
		if math.Abs(mean) > 1e-5 {
			t.Errorf("sample %d: mean = %v, want ~0", i, mean)
		}

		var variance float64
		for j := 0; j < 3; j++ {
			v := out.Value.Data[i*3+j]
			variance += v * v
		}
		variance /= 3
		if math.Abs(variance-1.0) > 1e-4 {
			t.Errorf("sample %d: variance = %v, want ~1", i, variance)
		}
	}
}

func TestLayerNormTrainEvalSame(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(2, engine)

	inputData := tensor.Zeros(1, 2)
	inputData.Data = []float64{3, 5}
	input := graph.NewNode(inputData, nil, nil)

	ln.Train()
	outTrain := ln.Forward(input).Value.Data

	ln.Eval()
	outEval := ln.Forward(input).Value.Data

	for i := range outTrain {
		if math.Abs(outTrain[i]-outEval[i]) > 1e-9 {
			t.Fatalf("train vs eval differ at %d: %v vs %v", i, outTrain[i], outEval[i])
		}
	}
}

func TestLayerNormParams(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(4, engine)
	params := ln.Params()
	if len(params) != 2 {
		t.Fatalf("params len = %d, want 2", len(params))
	}
	if params[0] != ln.gamma || params[1] != ln.beta {
		t.Fatal("Params should return gamma and beta")
	}
}

func TestLayerNormGammaBeta(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(2, engine)
	ln.gamma.Value.Data = []float64{2.0, 3.0}
	ln.beta.Value.Data = []float64{1.0, -1.0}

	inputData := tensor.Zeros(1, 2)
	inputData.Data = []float64{0, 0}
	input := graph.NewNode(inputData, nil, nil)

	out := ln.Forward(input).Value.Data
	if math.Abs(out[0]-1.0) > 1e-6 || math.Abs(out[1]-(-1.0)) > 1e-6 {
		t.Errorf("output = %v, want [1, -1] for zero-centered input with custom gamma/beta", out)
	}
}

func TestLayerNormInvalidInput(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(3, engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for 1D input")
		}
	}()
	input := graph.NewNode(tensor.Zeros(3), nil, nil)
	ln.Forward(input)
}

func TestLayerNormInvalidFeatures(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(3, engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for feature mismatch")
		}
	}()
	input := graph.NewNode(tensor.Zeros(2, 2), nil, nil)
	ln.Forward(input)
}

func TestLayerNormBackward(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(3, engine)

	xData := tensor.Zeros(2, 3)
	xData.Data = []float64{1, 2, 3, 4, 5, 6}
	x := engine.RequireGrad(xData)

	out := ln.Forward(x)
	engine.Backward(out)

	if x.Grad == nil {
		t.Fatal("expected gradient on input x")
	}
	if ln.gamma.Grad == nil || ln.beta.Grad == nil {
		t.Fatal("expected gradients on gamma and beta")
	}

	ok := autograd.CheckGradientEngine(
		func(e *autograd.Engine, inputs []*graph.Node) *graph.Node {
			layer := NewLayerNorm(3, e)
			return layer.Forward(inputs[0])
		},
		[]*graph.Node{x},
		1e-5,
		1e-4,
	)
	if !ok {
		t.Error("gradient check failed for LayerNorm input")
	}
}

func TestLayerNormBackwardParams(t *testing.T) {
	engine := autograd.NewEngine()
	ln := NewLayerNorm(2, engine)

	xData := tensor.Zeros(1, 2)
	xData.Data = []float64{1, 3}
	x := graph.NewNode(xData, nil, nil)

	out := ln.Forward(x)
	gradOut := tensor.Ones(out.Value.Shape...)
	out.Operation.Backward(gradOut)

	if ln.beta.Grad == nil {
		t.Fatal("expected beta gradient")
	}
	if math.Abs(ln.beta.Grad.Data[0]-1.0) > 1e-9 || math.Abs(ln.beta.Grad.Data[1]-1.0) > 1e-9 {
		t.Errorf("beta grad = %v, want [1, 1]", ln.beta.Grad.Data)
	}
}
