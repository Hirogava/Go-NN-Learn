package optimizers

import (
	"math"
	"runtime"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// RMSPropOption - функциональная опция для настройки оптимизатора RMSProp.
type RMSPropOption func(*RMSProp)

// WithRMSPropWeightDecay устанавливает коэффициент L2 регуляризации (weight decay) для RMSProp.
func WithRMSPropWeightDecay(decay float64) RMSPropOption {
	return func(r *RMSProp) {
		r.weightDecay = decay
	}
}

// RMSProp - оптимизатор RMSProp (Root Mean Square Propagation).
// Адаптирует темп обучения для каждого параметра, деля его на скользящее среднее квадратов градиентов.
type RMSProp struct {
	LearningRate float64                   // Скорость обучения (learning rate)
	Alpha        float64                   // Коэффициент затухания для скользящего среднего (decay rate)
	Epsilon      float64                   // Малое число для предотвращения деления на ноль
	weightDecay  float64                   // Коэффициент L2 регуляризации (weight decay)
	squaredGrad  map[*graph.Node][]float64 // Скользящее среднее квадратов градиентов
}

// NewRMSProp создает новый экземпляр оптимизатора RMSProp.
// Принимает опциональные параметры для настройки оптимизатора.
func NewRMSProp(lr, alpha, eps float64, opts ...RMSPropOption) *RMSProp {
	r := &RMSProp{
		LearningRate: lr,
		Alpha:        alpha,
		Epsilon:      eps,
		weightDecay:  0.0,
		squaredGrad:  make(map[*graph.Node][]float64),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Step обновляет параметры методом RMSProp.
// Формула: E[g^2]_t = alpha * E[g^2]_{t-1} + (1 - alpha) * grad^2
//
//	param -= lr * grad / (sqrt(E[g^2]_t) + epsilon)
func (r *RMSProp) Step(params []*graph.Node) {
	lr := r.LearningRate
	alpha := r.Alpha
	eps := r.Epsilon
	weightDecay := r.weightDecay

	for _, param := range params {
		p := param
		if p.Grad == nil {
			continue
		}

		if _, exists := r.squaredGrad[p]; !exists {
			r.squaredGrad[p] = make([]float64, len(p.Value.Data))
		}

		sg := r.squaredGrad[p]
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

				// E[g^2]_t = alpha * E[g^2]_{t-1} + (1 - alpha) * grad_with_decay^2
				sg[i] = alpha*sg[i] + (1-alpha)*g*g

				// param -= lr * grad_with_decay / (sqrt(E[g^2]_t) + epsilon)
				value[i] -= lr * g / (math.Sqrt(sg[i]) + eps)
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

func (r *RMSProp) ZeroGrad(params []*graph.Node) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0.0
			}
		}
	}
}

// SetLearningRate устанавливает новый Learning Rate для оптимизатора RMSProp.
func (r *RMSProp) SetLearningRate(lr float64) {
	r.LearningRate = lr
}
