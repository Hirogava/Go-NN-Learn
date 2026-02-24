package optimizers_test

import (
	"bytes"
	"os"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type addLayerForSummary struct {
	v      float64
	params []*graph.Node
}

func (l *addLayerForSummary) Train() {}
func (l *addLayerForSummary) Eval()  {}

func (a *addLayerForSummary) Forward(x *graph.Node) *graph.Node {
	in := x.Value
	var inVal float64
	if len(in.Data) > 0 {
		inVal = in.Data[0]
	}
	out := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{inVal + a.v},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}
	return out
}

func (a *addLayerForSummary) Params() []*graph.Node { return a.params }

func TestSummaryDoesNotPanicAndPrints(t *testing.T) {
	l1 := &addLayerForSummary{v: 1.0, params: nil}
	l2 := &addLayerForSummary{v: 2.0, params: nil}
	seq := optimizers.NewSequential(l1, l2)

	sample := &tensor.Tensor{
		Data:    []float64{0},
		Shape:   []int{1},
		Strides: []int{1},
	}

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe error: %v", err)
	}
	os.Stdout = w

	optimizers.Summary(seq, sample)

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	_, err = buf.ReadFrom(r)
	if err != nil {
		t.Fatalf("read from pipe failed: %v", err)
	}
	outStr := buf.String()
	if outStr == "" {
		t.Fatalf("Summary printed empty output")
	}
	if !(contains(outStr, "Total") || contains(outStr, "Params") || contains(outStr, "Layer")) {
		t.Fatalf("Summary output looks unexpected: %s", outStr)
	}
}

func contains(s, sub string) bool { return bytes.Contains([]byte(s), []byte(sub)) }

func BenchmarkSummary(b *testing.B) {
	layers := []*addLayer{
		{v: 1.0},
		{v: 2.0},
		{v: 3.0},
		{v: 4.0},
		{v: 5.0},
	}
	seq := optimizers.NewSequential(
		layers[0], layers[1], layers[2], layers[3], layers[4],
	)

	sample := &tensor.Tensor{
		Data:    []float64{0},
		Shape:   []int{1},
		Strides: []int{1},
	}

	old := os.Stdout
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		b.Fatalf("open devnull failed: %v", err)
	}
	defer devNull.Close()
	os.Stdout = devNull
	defer func() { os.Stdout = old }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizers.Summary(seq, sample)
	}
}
