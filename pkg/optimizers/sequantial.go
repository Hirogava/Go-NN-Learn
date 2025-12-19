package optimizers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Sequential — простой контейнер, который применяет набор слоёв последовательно:
// output = L_n( ... L_2(L_1(input)) ... )
type Sequential struct {
	layers []layers.Layer
}

// NewSequential создаёт новый Sequential, копируя переданный срез слоёв.
func NewSequential(ls ...layers.Layer) *Sequential {
	copied := make([]layers.Layer, len(ls))
	copy(copied, ls)
	return &Sequential{layers: copied}
}

// Layers возвращает список слоёв в Sequential (без копирования).
func (s *Sequential) Layers() []layers.Layer {
	return s.layers
}

// Forward выполняет прямой проход входного узла x по всем слоям последовательно.
func (s *Sequential) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}

// Params собирает и возвращает все параметры от всех вложенных слоёв,
// в порядке обхода слоёв (удобно для оптимизаторов и сериализации).
func (s *Sequential) Params() []*graph.Node {
	var params []*graph.Node
	for _, l := range s.layers {
		params = append(params, l.Params()...)
	}
	return params
}
