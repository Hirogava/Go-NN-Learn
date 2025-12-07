package optimizers

import (
	"runtime"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// MomentumOption - функциональная опция для настройки оптимизатора Momentum.
type MomentumOption func(*Momentum)

// WithMomentumWeightDecay устанавливает коэффициент L2 регуляризации (weight decay) для Momentum.
func WithMomentumWeightDecay(decay float64) MomentumOption {
	return func(m *Momentum) {
		m.weightDecay = decay
	}
}

// Momentum - оптимизатор с импульсом (Momentum).
// Ускоряет SGD в релевантном направлении и подавляет осцилляции.
type Momentum struct {
	LearningRate float64                   // Скорость обучения (learning rate)
	Mu           float64                   // Коэффициент инерции (momentum coefficient)
	weightDecay  float64                   // Коэффициент L2 регуляризации (weight decay)
	velocity     map[*graph.Node][]float64 // Скорость (импульс) для каждого параметра
}

// NewMomentum создает новый экземпляр оптимизатора Momentum.
// Принимает опциональные параметры для настройки оптимизатора.
func NewMomentum(lr, mu float64, opts ...MomentumOption) *Momentum {
	m := &Momentum{
		LearningRate: lr,
		Mu:           mu,
		weightDecay:  0.0,
		velocity:     make(map[*graph.Node][]float64),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Step обновляет параметры методом Momentum.
// Формула: v_t = mu * v_{t-1} + lr * grad
//
//	param -= v_t
func (m *Momentum) Step(params []*graph.Node) {
	mu := m.Mu
	lr := m.LearningRate
	weightDecay := m.weightDecay

	for _, param := range params {
		p := param
		if p.Grad == nil {
			continue
		}

		// Инициализируем velocity для этого параметра, если его еще нет
		if _, exists := m.velocity[p]; !exists {
			m.velocity[p] = make([]float64, len(p.Value.Data))
		}

		v := m.velocity[p]
		value := p.Value.Data
		grad := p.Grad.Data
		length := len(value)

		updateRange := func(start, end int) {
			for i := start; i < end; i++ {
				// Применяем WeightDecay: grad_with_decay = grad + lambda * weight
				g := grad[i]
				if weightDecay > 0 {
					g += weightDecay * value[i]
				}
				// v_t = mu * v_{t-1} + lr * grad_with_decay
				v[i] = mu*v[i] + lr*g
				// param -= v_t
				value[i] -= v[i]
			}
		}

		const parallelThreshold = 1024
		if length < parallelThreshold {
			updateRange(0, length)
			continue
		}

		workers := runtime.GOMAXPROCS(0)
		if workers > length {
			workers = length
		}
		chunk := (length + workers - 1) / workers
		var wg sync.WaitGroup
		for start := 0; start < length; start += chunk {
			end := start + chunk
			if end > length {
				end = length
			}
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				updateRange(s, e)
			}(start, end)
		}
		wg.Wait()
	}
}

// ZeroGrad обнуляет градиенты всех параметров w
func (m *Momentum) ZeroGrad(params []*graph.Node) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0.0
			}
		}
	}
}
