package optimizers_test

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

type addLayer struct {
	v      float64
	params []*graph.Node
}

func (l *addLayer) Train() {}
func (l *addLayer) Eval()  {}

func (a *addLayer) Forward(x *graph.Node) *graph.Node {
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

func (a *addLayer) Params() []*graph.Node {
	return a.params
}

func TestSequentialForwardAndParams(t *testing.T) {
	l1 := &addLayer{v: 1.0}
	l2 := &addLayer{v: 2.0}

	seq := optimizers.NewSequential(l1, l2)

	in := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{10.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}
	out := seq.Forward(in)
	if out == nil || len(out.Value.Data) == 0 {
		t.Fatal("Forward returned nil or empty output")
	}
	got := out.Value.Data[0]
	want := 13.0
	if got != want {
		t.Fatalf("Forward result mismatch: got %v want %v", got, want)
	}

	p1 := &graph.Node{Value: &tensor.Tensor{Data: []float64{1, 2}, Shape: []int{2}, Strides: []int{1}}}
	p2 := &graph.Node{Value: &tensor.Tensor{Data: []float64{3}, Shape: []int{1}, Strides: []int{1}}}
	l1.params = []*graph.Node{p1}
	l2.params = []*graph.Node{p2}

	all := seq.Params()
	if len(all) != 2 {
		t.Fatalf("Params length mismatch: got %d want 2", len(all))
	}
	if all[0].Value.Data[0] != 1 || all[1].Value.Data[0] != 3 {
		t.Fatalf("Params content mismatch: %+v", all)
	}
}

func BenchmarkSequentialForward(b *testing.B) {
	l1 := &addLayer{v: 1.0}
	l2 := &addLayer{v: 2.0}
	l3 := &addLayer{v: 3.0}
	seq := optimizers.NewSequential(l1, l2, l3)

	in := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{10.0},
			Shape:   []int{1},
			Strides: []int{1},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := seq.Forward(in)
		if out == nil {
			b.Fatalf("Forward returned nil")
		}
	}
}
