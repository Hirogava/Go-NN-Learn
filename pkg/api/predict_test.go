package api_test

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type identityLayer struct{}

func (l *identityLayer) Forward(x *graph.Node) *graph.Node { return x }
func (l *identityLayer) Params() []*graph.Node             { return nil }
func (l *identityLayer) Train()                            {}
func (l *identityLayer) Eval()                             {}

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

func (s *simpleModule) Train() {
	for _, l := range s.layers {
		l.Train()
	}
}

func (s *simpleModule) Eval() {
	for _, l := range s.layers {
		l.Eval()
	}
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

func TestTopKPredictions(t *testing.T) {
	logits := &tensor.Tensor{
		Data:    []float64{1, 3, 2, 10, 9, 1},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
		DType:   tensor.Float64,
	}

	items, err := api.TopKPredictions(logits, api.PredictionOptions{
		K:      2,
		Labels: []string{"cat", "dog", "fox"},
	})
	if err != nil {
		t.Fatalf("TopKPredictions returned error: %v", err)
	}

	if len(items) != 2 {
		t.Fatalf("len(items) = %d, want 2", len(items))
	}
	if len(items[0]) != 2 || len(items[1]) != 2 {
		t.Fatalf("unexpected top-k lengths: %+v", items)
	}

	if items[0][0].ClassID != 1 || items[0][0].ClassLabel != "dog" {
		t.Fatalf("unexpected top prediction for row 0: %+v", items[0][0])
	}
	if items[1][0].ClassID != 0 || items[1][0].ClassLabel != "cat" {
		t.Fatalf("unexpected top prediction for row 1: %+v", items[1][0])
	}
	if items[0][0].Probability < items[0][1].Probability {
		t.Fatalf("row 0 is not sorted by probability: %+v", items[0])
	}
}

func TestTopKPredictionsThreshold(t *testing.T) {
	logits := &tensor.Tensor{
		Data:    []float64{0, 0, 0},
		Shape:   []int{1, 3},
		Strides: []int{3, 1},
		DType:   tensor.Float64,
	}

	items, err := api.TopKPredictions(logits, api.PredictionOptions{
		K:         3,
		Threshold: 0.34,
	})
	if err != nil {
		t.Fatalf("TopKPredictions returned error: %v", err)
	}

	if len(items) != 1 {
		t.Fatalf("len(items) = %d, want 1", len(items))
	}
	if len(items[0]) != 0 {
		t.Fatalf("expected threshold to filter all predictions, got %+v", items[0])
	}
}

func TestTopKPredictionsValidation(t *testing.T) {
	badShape := &tensor.Tensor{
		Data:    []float64{1, 2, 3},
		Shape:   []int{3},
		Strides: []int{1},
		DType:   tensor.Float64,
	}
	if _, err := api.TopKPredictions(badShape, api.PredictionOptions{K: 1}); err == nil {
		t.Fatal("expected shape validation error")
	}

	logits := &tensor.Tensor{
		Data:    []float64{1, 2, 3, 4},
		Shape:   []int{2, 2},
		Strides: []int{2, 1},
		DType:   tensor.Float64,
	}
	if _, err := api.TopKPredictions(logits, api.PredictionOptions{K: 0}); err == nil {
		t.Fatal("expected k validation error")
	}
	if _, err := api.TopKPredictions(logits, api.PredictionOptions{K: 1, Threshold: 2}); err == nil {
		t.Fatal("expected threshold validation error")
	}
	if _, err := api.TopKPredictions(logits, api.PredictionOptions{K: 1, Labels: []string{"only-one-label"}}); err == nil {
		t.Fatal("expected labels validation error")
	}
}

func TestTopKPredictionsDefaultLabels(t *testing.T) {
	logits := &tensor.Tensor{
		Data:    []float64{1, 5, 2},
		Shape:   []int{1, 3},
		Strides: []int{3, 1},
		DType:   tensor.Float64,
	}

	items, err := api.TopKPredictions(logits, api.PredictionOptions{K: 1})
	if err != nil {
		t.Fatalf("TopKPredictions returned error: %v", err)
	}
	if items[0][0].ClassLabel != "1" {
		t.Fatalf("unexpected default class label: %+v", items[0][0])
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
