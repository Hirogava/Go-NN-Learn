package gnn_test

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/gnn"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type identityLayer struct{}

func (l *identityLayer) Forward(x *graph.Node) *graph.Node { return x }
func (l *identityLayer) Params() []*graph.Node             { return nil }

type simpleModule struct {
	layers []layers.Layer
}

func (s *simpleModule) Layers() []layers.Layer   { return s.layers }
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

func TestNoGrad_Predict_NoGraphNoAllocs(t *testing.T) {
	mod := &simpleModule{layers: []layers.Layer{&identityLayer{}}}
	x := &graph.Node{
		Value: &tensor.Tensor{Data: []float64{1.0, 2.0}, Shape: []int{1, 2}},
	}

	var out *graph.Node
	gnn.NoGrad(func() {
		out = api.Predict(mod, x)
	})

	if out == nil {
		t.Fatal("Predict вернул nil")
	}
	// DoD: граф не создаётся
	if out.Parents != nil {
		t.Fatalf("После NoGrad Predict Parents должен быть nil, получено %v", out.Parents)
	}
	if out.Operation != nil {
		t.Fatalf("После NoGrad Predict Operation должен быть nil, получено %v", out.Operation)
	}
	// DoD: нет аллокаций autograd
	if out.Grad != nil {
		t.Fatalf("Grad должен быть nil (без аллокаций autograd), получено %v", out.Grad)
	}
}

func TestNoGrad_AllocsPerRun(t *testing.T) {
	mod := &simpleModule{layers: []layers.Layer{&identityLayer{}}}
	x := &graph.Node{
		Value: &tensor.Tensor{Data: []float64{1.0, 2.0}, Shape: []int{1, 2}},
	}

	allocs := testing.AllocsPerRun(100, func() {
		gnn.NoGrad(func() {
			_ = api.Predict(mod, x)
		})
	})
	// Допускаем минимум аллокаций (например, замыкание), но аллокаций autograd
	// быть не должно (нет tensor.Zeros для Grad). Identity-модель + no_grad не
	// должна аллоцировать Grad; при аллокациях в этом пути их было бы много за проход.
	if allocs > 5 {
		t.Errorf("NoGrad(Predict) не должен сильно аллоцировать (autograd); получено %.0f аллокаций/запуск", allocs)
	}
}
