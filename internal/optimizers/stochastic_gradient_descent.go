package optimizers

import (
	"runtime"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
)

// SGDOption - функциональная опция для настройки оптимизатора SGD.
type SGDOption func(*StochasticGradientDescent)

// WithSGDWeightDecay устанавливает коэффициент L2 регуляризации (weight decay) для SGD.
func WithSGDWeightDecay(decay float64) SGDOption {
	return func(s *StochasticGradientDescent) {
		s.weightDecay = decay
	}
}

// StochasticGradientDescent - простой оптимизатор Stochastic Gradient Descent.
// Обновляет параметры модели, вычитая градиент, умноженный на learning rate.
type StochasticGradientDescent struct {
	LearningRate float64 // Скорость обучения
	weightDecay  float64 // Коэффициент L2 регуляризации (weight decay)
}

// NewSGD создает новый экземпляр SGD с заданным learning rate.
// Принимает опциональные параметры для настройки оптимизатора.
func NewSGD(lr float64, opts ...SGDOption) *StochasticGradientDescent {
	s := &StochasticGradientDescent{
		LearningRate: lr,
		weightDecay:  0.0,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step обновляет параметры методом SGD, умножая градиент на LearningRate: param.Value -= lr * param.Grad
func (s *StochasticGradientDescent) Step(params []*graph.Node) {
	lr := s.LearningRate
	weightDecay := s.weightDecay

	for _, param := range params {
		p := param
		if p.Grad == nil {
			continue
		}

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
				// param -= lr * grad_with_decay
				value[i] -= lr * g
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

// SetLearningRate устанавливает новый Learning Rate для оптимизатора SGD.
func (s *StochasticGradientDescent) SetLearningRate(lr float64) {
	s.LearningRate = lr
}
