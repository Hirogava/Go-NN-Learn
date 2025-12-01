package optimizers

import "github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"

// Momentum - оптимизатор с импульсом (Momentum).
// Ускоряет SGD в релевантном направлении и подавляет осцилляции.
type Momentum struct {
	LearningRate float64                   // Скорость обучения (learning rate)
	Mu           float64                   // Коэффициент инерции (momentum coefficient)
	velocity     map[*graph.Node][]float64 // Скорость (импульс) для каждого параметра
}

// NewMomentum создает новый экземпляр оптимизатора Momentum.
func NewMomentum(lr, mu float64) *Momentum {
	return &Momentum{
		LearningRate: lr,
		Mu:           mu,
		velocity:     make(map[*graph.Node][]float64),
	}
}

// Step обновляет параметры методом Momentum.
// Формула: v_t = mu * v_{t-1} + lr * grad
//
//	param -= v_t
func (m *Momentum) Step(params []*graph.Node) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}

		// Инициализируем velocity для этого параметра, если его еще нет
		if _, exists := m.velocity[p]; !exists {
			m.velocity[p] = make([]float64, len(p.Value.Data))
		}

		v := m.velocity[p]

		// Обновляем velocity и параметры
		for i := range p.Value.Data {
			// v_t = mu * v_{t-1} + lr * grad
			v[i] = m.Mu*v[i] + m.LearningRate*p.Grad.Data[i]
			// param -= v_t
			p.Value.Data[i] -= v[i]
		}
	}
}

// ZeroGrad обнуляет градиенты всех параметров
func (m *Momentum) ZeroGrad(params []*graph.Node) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0.0
			}
		}
	}
}
