package optimizers

import "github.com/Hirogava/Go-NN-Learn/internal/backend/graph"

// Optimizer - интерфейс для различных оптимизаторов модели
type Optimizer interface {
	Step(params []*graph.Node)
	SetLearningRate(lr float64)
	ZeroGrad(params []*graph.Node)
}
