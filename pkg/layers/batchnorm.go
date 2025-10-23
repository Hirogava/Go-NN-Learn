package layers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// BatchNorm нормализует входящий батч (вектор), путем изменения общего среднего и дисперсии.
// После нормализации выходные данные будут иметь среднее 0 и дисперсию 1.
// Параметры gamma и betta позволяют восстановить любую нужную дисперсию и среднее, если это полезно для модели.
// gamma масштабирует батч, то есть умножает все нормализованные значения на свое значени
// betta смещает значения нормализованные значения вверх или вниз.
// По умолчанию gamma = betta = 1
func BatchNormVector(batch tensor.Vector, gamma float64, betta float64) []float64 {
	m := float64(len(batch))
	var sum float64
	for _, el := range batch {
		sum += el
	}

	avg := sum / m // Высчитываем среднее значение батча

	var dispersion float64 // Дисперсия значений
	for _, el := range batch {
		dispersion += math.Pow(el-avg, 2)
	}
	dispersion /= m

	normBatch := make([]float64, int64(m))
	for i := 0; i < int(m); i++ {
		normBatch[i] = (batch[i] - avg) / math.Sqrt(dispersion+1e-20)
	}
	return normBatch
}
