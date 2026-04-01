package optimizers_test

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type addLayer struct {
	v      float64
	params []*graph.Node
	trains int
	evals  int
}

func (l *addLayer) Train() { l.trains++ }
func (l *addLayer) Eval()  { l.evals++ }

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

func TestSequentialAddLayer(t *testing.T) {
	l1 := &addLayer{v: 1.0}
	l2 := &addLayer{v: 2.0}

	seq := optimizers.NewSequential(l1)
	seq.AddLayer(l2)

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
	if got, want := out.Value.Data[0], 13.0; got != want {
		t.Fatalf("Forward result mismatch after AddLayer: got %v want %v", got, want)
	}
}

func TestSequentialTrainEval(t *testing.T) {
	l1 := &addLayer{v: 1.0}
	l2 := &addLayer{v: 2.0}
	seq := optimizers.NewSequential(l1, l2)

	seq.Train()
	seq.Eval()

	if l1.trains != 1 || l2.trains != 1 {
		t.Fatalf("Train was not propagated to all layers: l1=%d l2=%d", l1.trains, l2.trains)
	}
	if l1.evals != 1 || l2.evals != 1 {
		t.Fatalf("Eval was not propagated to all layers: l1=%d l2=%d", l1.evals, l2.evals)
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
