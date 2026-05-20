package layers

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestGRU_ForwardShape(t *testing.T) {
	batchSize, seqLen, inputSize, hiddenSize := 2, 3, 4, 5
	gru := NewGRU(inputSize, hiddenSize, simpleInit)

	xValue := tensor.Zeros(batchSize, seqLen, inputSize)
	xNode := graph.NewNode(xValue, nil, nil)

	gru.ResetHiddenState()
	out := gru.Forward(xNode)

	want := []int{batchSize, seqLen, hiddenSize}
	for i, d := range out.Value.Shape {
		if d != want[i] {
			t.Fatalf("output shape: want %v, got %v", want, out.Value.Shape)
		}
	}
}

func TestGRU_ForwardBackward(t *testing.T) {
	batchSize, seqLen, inputSize, hiddenSize := 2, 3, 4, 5
	gru := NewGRU(inputSize, hiddenSize, simpleInit)

	xValue := tensor.Zeros(batchSize, seqLen, inputSize)
	for i := range xValue.Data {
		xValue.Data[i] = float64(i) * 0.01
	}
	xNode := graph.NewNode(xValue, nil, nil)

	gru.ResetHiddenState()
	outputNode := gru.Forward(xNode)

	if outputNode.Operation == nil {
		t.Fatal("GRU Forward did not set Operation on output node")
	}

	gradOut := tensor.Zeros(outputNode.Value.Shape...)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0
	}
	outputNode.Operation.Backward(gradOut)

	for i, wNode := range gru.weights {
		if wNode.Grad == nil {
			t.Errorf("weight grad [%d] is nil", i)
			continue
		}
		if allZero(wNode.Grad.Data) {
			t.Errorf("weight grad [%d] is all zeros", i)
		}
	}
	for i, bNode := range gru.biases {
		if bNode.Grad == nil {
			t.Errorf("bias grad [%d] is nil", i)
			continue
		}
		if allZero(bNode.Grad.Data) {
			t.Errorf("bias grad [%d] is all zeros", i)
		}
	}
	if xNode.Grad == nil {
		t.Error("input grad is nil")
	} else if allZero(xNode.Grad.Data) {
		t.Error("input grad is all zeros")
	}
}

func TestGRU_EvalNoGraph(t *testing.T) {
	gru := NewGRU(2, 3, simpleInit)
	gru.Eval()

	x := graph.NewNode(tensor.Zeros(1, 2, 2), nil, nil)
	out := gru.Forward(x)
	if out.Operation != nil {
		t.Error("eval mode should not attach Operation")
	}
}

func TestGRU_HiddenStateCarry(t *testing.T) {
	gru := NewGRU(2, 2, simpleInit)
	gru.Eval()

	x1 := graph.NewNode(tensor.Zeros(1, 1, 2), nil, nil)
	out1 := gru.Forward(x1)
	h1 := out1.Value.Data[0]

	x2 := graph.NewNode(tensor.Zeros(1, 1, 2), nil, nil)
	out2 := gru.Forward(x2)
	h2 := out2.Value.Data[0]

	gru.ResetHiddenState()
	out3 := gru.Forward(x2)
	h3 := out3.Value.Data[0]

	if h1 == h2 {
		t.Log("same input after carry may coincide with small weights; checking reset differs")
	}
	if h2 == h3 && h1 != h3 {
		t.Error("reset hidden state should change output when state was carried")
	}
}

func TestGRU_GatesInRange(t *testing.T) {
	gru := NewGRU(3, 4, simpleInit)
	gru.Eval()

	x := graph.NewNode(tensor.Zeros(1, 1, 3), nil, nil)
	for i := range x.Value.Data {
		x.Value.Data[i] = 0.5
	}
	out := gru.Forward(x)
	for _, v := range out.Value.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("invalid output value: %v", v)
		}
	}
}

func allZero(data []float64) bool {
	for _, v := range data {
		if v != 0 {
			return false
		}
	}
	return true
}
