package optimizers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// RMSProp - оптимизатор RMSProp (Root Mean Square Propagation).
// Адаптирует темп обучения для каждого параметра, деля его на скользящее среднее квадратов градиентов.
type RMSProp struct {
	LearningRate float64                   // Скорость обучения (learning rate)
	Alpha        float64                   // Коэффициент затухания для скользящего среднего (decay rate)
	Epsilon      float64                   // Малое число для предотвращения деления на ноль
	squaredGrad  map[*graph.Node][]float64 // Скользящее среднее квадратов градиентов
}

// NewRMSProp создает новый экземпляр оптимизатора RMSProp.
func NewRMSProp(lr, alpha, eps float64) *RMSProp {
	return &RMSProp{
		LearningRate: lr,
		Alpha:        alpha,
		Epsilon:      eps,
		squaredGrad:  make(map[*graph.Node][]float64),
	}
}

// Step обновляет параметры методом RMSProp.
// Формула: E[g^2]_t = alpha * E[g^2]_{t-1} + (1 - alpha) * grad^2
//
//	param -= lr * grad / (sqrt(E[g^2]_t) + epsilon)
func (r *RMSProp) Step(params []*graph.Node) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}

		if _, exists := r.squaredGrad[p]; !exists {
			r.squaredGrad[p] = make([]float64, len(p.Value.Data))
		}

		sg := r.squaredGrad[p]

		// Обновляем скользящее среднее квадратов градиентов и параметры
		for i := range p.Value.Data {
			// E[g^2]_t = alpha * E[g^2]_{t-1} + (1 - alpha) * grad^2
			sg[i] = r.Alpha*sg[i] + (1-r.Alpha)*p.Grad.Data[i]*p.Grad.Data[i]

			// param -= lr * grad / (sqrt(E[g^2]_t) + epsilon)
			p.Value.Data[i] -= r.LearningRate * p.Grad.Data[i] / (math.Sqrt(sg[i]) + r.Epsilon)
		}
	}
}

func (r *RMSProp) ZeroGrad(params []*graph.Node) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0.0
			}
		}
	}
}
