package layers

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestLSTMCell_Gates(t *testing.T) {
	inputSize, hiddenSize, batch := 2, 3, 1
	dir := newLSTMDirection(inputSize, hiddenSize, zeroInit)
	cell := NewLSTMCell(dir, hiddenSize)

	x := tensor.Zeros(batch, inputSize)
	hPrev := tensor.Zeros(batch, hiddenSize)
	cPrev := tensor.Zeros(batch, hiddenSize)

	step := cell.Forward(x, hPrev, cPrev, batch)
	if step.i == nil || step.f == nil || step.g == nil || step.o == nil {
		t.Fatal("gate tensors must be non-nil")
	}
	if step.c == nil || step.h == nil {
		t.Fatal("state tensors must be non-nil")
	}

	// При нулевых весах и входе: i,f,o = 0.5, g = 0, c = 0.5*c_prev, h = 0.5*tanh(c)
	for i := range step.i.Data {
		if math.Abs(step.i.Data[i]-0.5) > 1e-9 {
			t.Errorf("input gate expected 0.5, got %v", step.i.Data[i])
		}
	}
}

func TestLSTM_ForwardBackward(t *testing.T) {
	batchSize := 2
	seqLen := 3
	inputSize := 4
	hiddenSize := 5

	lstm := NewLSTM(inputSize, hiddenSize, false, simpleInit)
	xValue := tensor.Zeros(batchSize, seqLen, inputSize)
	for i := range xValue.Data {
		xValue.Data[i] = float64(i) * 0.01
	}
	xNode := graph.NewNode(xValue, nil, nil)

	lstm.ResetState()
	out := lstm.Forward(xNode)

	expectedShape := []int{batchSize, seqLen, hiddenSize}
	for i, dim := range out.Value.Shape {
		if dim != expectedShape[i] {
			t.Fatalf("output shape: want %v, got %v", expectedShape, out.Value.Shape)
		}
	}

	gradOut := tensor.Zeros(out.Value.Shape...)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0
	}

	if out.Operation == nil {
		t.Fatal("Forward did not attach Operation")
	}
	out.Operation.Backward(gradOut)

	for i, w := range lstm.fwd.params()[:8] {
		if w.Grad == nil {
			t.Errorf("weight grad [%d] is nil", i)
			continue
		}
		allZero := true
		for _, v := range w.Grad.Data {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("weight grad [%d] is all zeros", i)
		}
	}

	if xNode.Grad == nil {
		t.Error("input gradient is nil")
	}
}

func TestLSTM_BidirectionalShape(t *testing.T) {
	batchSize, seqLen, inputSize, hiddenSize := 2, 4, 3, 5
	lstm := NewLSTM(inputSize, hiddenSize, true, simpleInit)
	x := graph.NewNode(tensor.Zeros(batchSize, seqLen, inputSize), nil, nil)

	out := lstm.Forward(x)
	want := []int{batchSize, seqLen, hiddenSize * 2}
	for i, d := range want {
		if out.Value.Shape[i] != d {
			t.Fatalf("shape[%d]: want %d, got %d", i, d, out.Value.Shape[i])
		}
	}
}

func TestLSTM_ResetState(t *testing.T) {
	lstm := NewLSTM(2, 3, false, simpleInit)
	x := graph.NewNode(tensor.Zeros(1, 2, 2), nil, nil)
	_ = lstm.Forward(x)
	if lstm.GetHiddenState() == nil {
		t.Fatal("hidden state should be set after forward")
	}
	lstm.ResetState()
	if lstm.GetHiddenState() != nil || lstm.GetCellState() != nil {
		t.Error("ResetState should clear hidden and cell state")
	}
}

func zeroInit(data []float64) {
	for i := range data {
		data[i] = 0
	}
}
