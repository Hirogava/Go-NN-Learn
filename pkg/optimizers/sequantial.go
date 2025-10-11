package optimizers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Sequential — простая последовательная модель
type Sequential struct {
	layers []layers.Layer
}

// NewSequential — конструктор последовательной модели
func NewSequential(ls ...layers.Layer) *Sequential {
	copied := make([]layers.Layer, len(ls))
	copy(copied, ls)
	return &Sequential{layers: copied}
}

func (s *Sequential) Layers() []layers.Layer {
	return s.layers
}

// Forward — прогон входа по всем слоям последовательно
func (s *Sequential) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}

// Params — собирает параметры всех слоёв
func (s *Sequential) Params() []*graph.Node {
	var params []*graph.Node
	for _, l := range s.layers {
		params = append(params, l.Params()...)
	}
	return params
}
