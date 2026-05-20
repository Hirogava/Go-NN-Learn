package layers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Layer — базовый интерфейс слоя
// Forward: принимает узел графа и возвращает узел
// Params: возвращает срез узлов-параметров
type Layer interface {
	Forward(x *graph.Node) *graph.Node
	Params() []*graph.Node
	Train()
	Eval()
}

// Module — модуль, состоящий из слоёв
type Module interface {
	Layers() []Layer
	Forward(x *graph.Node) *graph.Node
	Params() []*graph.Node
	Train()
	Eval()
}

type TrainEval interface {
	Train()
	Eval()
}

// LayerNorm — нормализация по признакам каждого примера (Transformer-style).
// Реализация: layernorm.go.
type LayerNorm struct {
	numFeatures int
	eps         float64

	gamma *graph.Node
	beta  *graph.Node

	engine *autograd.Engine
}
