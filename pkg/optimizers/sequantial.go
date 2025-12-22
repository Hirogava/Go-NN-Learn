package optimizers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Sequential — простая последовательная модель: out = L_n(...L_1(x)...).
// Используется как удобная обёртка для быстрого прототипирования.
type Sequential struct {
	layers []layers.Layer
}

// NewSequential создаёт новый Sequential экземпляр, копируя срез слоёв.
func NewSequential(ls ...layers.Layer) *Sequential {
	copied := make([]layers.Layer, len(ls))
	copy(copied, ls)
	return &Sequential{layers: copied}
}

// Layers возвращает список слоёв в модели (не копирует).
func (s *Sequential) Layers() []layers.Layer {
	return s.layers
}

// Forward прогоняет вход по слоям последовательно и возвращает результат.
func (s *Sequential) Forward(x *graph.Node) *graph.Node {
	out := x
	for _, l := range s.layers {
		out = l.Forward(out)
	}
	return out
}

// Params собирает параметры всех вложенных слоёв (в порядке слоёв)
// и возвращает их в виде среза []*graph.Node.
func (s *Sequential) Params() []*graph.Node {
	var params []*graph.Node
	for _, l := range s.layers {
		params = append(params, l.Params()...)
	}
	return params
}
