package api_test

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

type identityLayer struct{}

func (l *identityLayer) Forward(x *graph.Node) *graph.Node { return x }
func (l *identityLayer) Params() []*graph.Node             { return nil }

type simpleModule struct {
	layers []layers.Layer
}

func (s *simpleModule) Layers() []layers.Layer {
	return s.layers
}

func (s *simpleModule) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}

func (s *simpleModule) Params() []*graph.Node {
	var ps []*graph.Node
	for _, l := range s.layers {
		ps = append(ps, l.Params()...)
	}
	return ps
}

func buildIdentityModule() layers.Module {
	return &simpleModule{layers: []layers.Layer{&identityLayer{}}}
}

func TestPredictAndEval(t *testing.T) {
	mod := buildIdentityModule()

	in := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0},
			Shape:   []int{1, 2},
			Strides: nil,
		},
	}

	out := api.Predict(mod, in)
	if out == nil {
		t.Fatal("Predict returned nil")
	}
	if len(out.Value.Data) != 2 {
		t.Fatalf("Predict output shape mismatch: got %d elements", len(out.Value.Data))
	}

	metric := func(pred *graph.Node, target *graph.Node) float64 {
		var s float64
		for i := range pred.Value.Data {
			d := pred.Value.Data[i] - target.Value.Data[i]
			s += d * d
		}
		return s / float64(len(pred.Value.Data))
	}
	inputs := []*graph.Node{in, in}
	targets := []*graph.Node{in, in}
	avg, err := api.Eval(mod, inputs, targets, metric)
	if err != nil {
		t.Fatalf("Eval returned error: %v", err)
	}
	if avg != 0 {
		t.Fatalf("Eval expected 0 metric, got %v", avg)
	}
}

func BenchmarkPredict(b *testing.B) {
	mod := buildIdentityModule()

	in := &graph.Node{
		Value: &tensor.Tensor{
			Data:    []float64{1.0, 2.0, 3.0, 4.0},
			Shape:   []int{1, 4},
			Strides: nil,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := api.Predict(mod, in)
		if out == nil {
			b.Fatalf("Predict returned nil")
		}
	}
}
