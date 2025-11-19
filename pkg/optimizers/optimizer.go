package optimizers

import "github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"

// Optimizer - интерфейс для различных оптимизаторов модели
type Optimizer interface {
	Step(params []*graph.Node)
	ZeroGrad(params []*graph.Node)
}
