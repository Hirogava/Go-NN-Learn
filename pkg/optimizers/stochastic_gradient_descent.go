package optimizers

import "github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"

// StochasticGradientDescent - простой оптимизатор Stochastic Gradient Descent.
// Обновляет параметры модели, вычитая градиент, умноженный на learning rate.
type StochasticGradientDescent struct {
	LearningRate float64 // Скорость обучения
}

// NewSGD создает новый экземпляр SGD с заданным learning rate.
func NewSGD(lr float64) *StochasticGradientDescent {
	return &StochasticGradientDescent{LearningRate: lr}
}

// Step обновляет параметры методом SGD, умножая градиент на LearningRate: param.Value -= lr * param.Grad
func (s *StochasticGradientDescent) Step(params []*graph.Node) {
	for _, p := range params {
		if p.Grad == nil {
			continue // или паника, в зависимости от требований
		}
		// Проходимся по тензорам и умножаем на скорость обучения
		for i := range p.Value.Data { // если Tensor
			p.Value.Data[i] -= s.LearningRate * p.Grad.Data[i]
		}
	}
}

// ZeroGrad обнуляет градиенты всех параметров
func (s *StochasticGradientDescent) ZeroGrad(params []*graph.Node) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0.0
			}
		}
	}
}
