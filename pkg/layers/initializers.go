package layers

import (
	"math"
	"math/rand"
)

// Initializer - функция для инициализации слайса данных.
type Initializer func([]float64)

// ZeroInit инициализирует все значения нулями.
func ZeroInit() Initializer {
	return func(w []float64) {
		for i := range w {
			w[i] = 0
		}
	}
}

// HeInit (He Normal) - специально для слоев с активацией ReLU.
// std = sqrt(2 / fanIn)
func HeInit(fanIn int) Initializer {
	std := math.Sqrt(2.0 / float64(fanIn))
	return func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * std
		}
	}
}

// XavierInit (Xavier Normal) - для слоев с Sigmoid/Tanh.
// std = sqrt(2 / (fanIn + fanOut))
func XavierInit(fanIn, fanOut int) Initializer {
	std := math.Sqrt(2.0 / float64(fanIn+fanOut))
	return func(w []float64) {
		for i := range w {
			w[i] = rand.NormFloat64() * std
		}
	}
}

// HeNormal - алиас для HeInit
func HeNormal(fanIn int) Initializer {
	return HeInit(fanIn)
}

// HeUniform инициализация.
func HeUniform(fanIn int) Initializer {
	limit := math.Sqrt(6.0 / float64(fanIn))
	return func(w []float64) {
		for i := range w {
			w[i] = rand.Float64()*2*limit - limit
		}
	}
}

// XavierNormal - алиас для XavierInit
func XavierNormal(fanIn, fanOut int) Initializer {
	return XavierInit(fanIn, fanOut)
}

// XavierUniform инициализация.
func XavierUniform(fanIn, fanOut int) Initializer {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	return func(w []float64) {
		for i := range w {
			w[i] = rand.Float64()*2*limit - limit
		}
	}
}
