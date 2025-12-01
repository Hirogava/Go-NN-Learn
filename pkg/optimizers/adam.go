package optimizers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Adam - оптимизатор Adaptive Moment Estimation.
// Сочетает Momentum и RMSProp, используя первые и вторые моменты градиента.
type Adam struct {
	LearningRate float64                   // Скорость обучения
	Beta1        float64                   // Коэффициент для первого момента
	Beta2        float64                   // Коэффициент для второго момента
	Epsilon      float64                   // Малое число для стабильности
	m            map[*graph.Node][]float64 // Первый момент
	v            map[*graph.Node][]float64 // Второй момент
	t            int                       // Номер шага для bias correction
}

// NewAdam создает новый экземпляр оптимизатора Adam.
func NewAdam(lr, beta1, beta2, eps float64) *Adam {
	return &Adam{
		LearningRate: lr,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      eps,
		m:            make(map[*graph.Node][]float64),
		v:            make(map[*graph.Node][]float64),
		t:            0,
	}
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

	for _, p := range params {
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

		for i := range p.Value.Data {
			grad := p.Grad.Data[i]

			// Обновление первых и вторых моментов
			mVec[i] = a.Beta1*mVec[i] + (1-a.Beta1)*grad
			vVec[i] = a.Beta2*vVec[i] + (1-a.Beta2)*grad*grad

			// Коррекция смещения
			mHat := mVec[i] / beta1Corr
			vHat := vVec[i] / beta2Corr

			// Обновление параметров
			p.Value.Data[i] -= a.LearningRate * mHat / (math.Sqrt(vHat) + a.Epsilon)
		}
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
