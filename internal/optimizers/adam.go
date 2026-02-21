package optimizers

import (
	"math"
	"runtime"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
)

// AdamOption - функциональная опция для настройки оптимизатора Adam.
type AdamOption func(*Adam)

// WithAdamWeightDecay устанавливает коэффициент L2 регуляризации (weight decay) для Adam.
func WithAdamWeightDecay(decay float64) AdamOption {
	return func(a *Adam) {
		a.weightDecay = decay
	}
}

// Adam - оптимизатор Adaptive Moment Estimation.
// Сочетает Momentum и RMSProp, используя первые и вторые моменты градиента.
type Adam struct {
	LearningRate float64                   // Скорость обучения
	Beta1        float64                   // Коэффициент для первого момента
	Beta2        float64                   // Коэффициент для второго момента
	Epsilon      float64                   // Малое число для стабильности
	weightDecay  float64                   // Коэффициент L2 регуляризации (weight decay)
	m            map[*graph.Node][]float64 // Первый момент
	v            map[*graph.Node][]float64 // Второй момент
	t            int                       // Номер шага для bias correction
}

// NewAdam создает новый экземпляр оптимизатора Adam.
// Принимает опциональные параметры для настройки оптимизатора.
func NewAdam(lr, beta1, beta2, eps float64, opts ...AdamOption) *Adam {
	a := &Adam{
		LearningRate: lr,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      eps,
		weightDecay:  0.0,
		m:            make(map[*graph.Node][]float64),
		v:            make(map[*graph.Node][]float64),
		t:            0,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step обновляет параметры методом Adam.
//
// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
//
//	param -= lr * m̂_t / (sqrt(v̂_t) + eps)
func (a *Adam) Step(params []*graph.Node) {
	a.t++
	beta1Corr := 1 - math.Pow(a.Beta1, float64(a.t))
	beta2Corr := 1 - math.Pow(a.Beta2, float64(a.t))
	beta1Tail := 1 - a.Beta1
	beta2Tail := 1 - a.Beta2
	lr := a.LearningRate
	eps := a.Epsilon
	beta1 := a.Beta1
	beta2 := a.Beta2
	weightDecay := a.weightDecay

	for _, param := range params {
		p := param
		if p.Grad == nil {
			continue
		}

		if _, exists := a.m[p]; !exists {
			a.m[p] = make([]float64, len(p.Value.Data))
		}
		if _, exists := a.v[p]; !exists {
			a.v[p] = make([]float64, len(p.Value.Data))
		}

		mVec := a.m[p]
		vVec := a.v[p]
		value := p.Value.Data
		grad := p.Grad.Data
		length := len(value)

		updateRange := func(start, end int) {
			for i := start; i < end; i++ {
				g := grad[i]

				// Применяем WeightDecay: grad_with_decay = grad + lambda * weight
				if weightDecay > 0 {
					g += weightDecay * value[i]
				}

				// Обновление первых и вторых моментов
				mVec[i] = beta1*mVec[i] + beta1Tail*g
				vVec[i] = beta2*vVec[i] + beta2Tail*g*g

				// Коррекция смещения
				mHat := mVec[i] / beta1Corr
				vHat := vVec[i] / beta2Corr

				// Обновление параметров
				value[i] -= lr * mHat / (math.Sqrt(vHat) + eps)
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

// ZeroGrad обнуляет градиенты всех параметров.
func (a *Adam) ZeroGrad(params []*graph.Node) {
	for _, p := range params {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] = 0.0
			}
		}
	}
}

// SetLearningRate устанавливает новый Learning Rate для оптимизатора Adam.
func (a *Adam) SetLearningRate(lr float64) {
	a.LearningRate = lr
}
