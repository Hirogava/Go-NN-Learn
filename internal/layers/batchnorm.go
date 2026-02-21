package layers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/internal/autograd"
	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
)

// BatchNormVector нормализует входящий батч (вектор), путем изменения общего среднего и дисперсии.
// После нормализации выходные данные будут иметь среднее 0 и дисперсию 1.
// Параметры gamma и beta позволяют восстановить любую нужную дисперсию и среднее, если это полезно для модели.
// gamma масштабирует батч, то есть умножает все нормализованные значения на свое значение
// beta смещает нормализованные значения вверх или вниз.
// По умолчанию gamma = beta = 1
func BatchNormVector(batch tensor.Vector, gamma float64, beta float64) []float64 {
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

// BatchNorm реализует слой батч-нормализации для нейронных сетей.
// Нормализует активации по мини-батчу для стабилизации обучения.
//
// Формулы:
//
//	Forward (training):
//	  μ = mean(x)                     // среднее по батчу
//	  σ² = var(x)                     // дисперсия по батчу
//	  x̂ = (x - μ) / sqrt(σ² + ε)     // нормализация
//	  y = γ * x̂ + β                  // масштабирование и сдвиг
//
//	Forward (inference):
//	  x̂ = (x - running_mean) / sqrt(running_var + ε)
//	  y = γ * x̂ + β
//
// Параметры:
//   - numFeatures: количество признаков (размерность входа)
//   - eps: малое число для численной стабильности (по умолчанию 1e-5)
//   - momentum: коэффициент для обновления running mean/var (по умолчанию 0.1)
type BatchNorm struct {
	numFeatures int
	eps         float64
	momentum    float64
	training    bool

	// Обучаемые параметры
	gamma *graph.Node // масштаб (scale)
	beta  *graph.Node // сдвиг (shift)

	// Скользящие статистики для inference
	runningMean *tensor.Tensor
	runningVar  *tensor.Tensor

	// Engine для autograd
	engine *autograd.Engine
}

// NewBatchNorm создаёт новый слой BatchNorm.
// numFeatures - количество признаков (размерность входа).
// engine - движок автодифференцирования.
func NewBatchNorm(numFeatures int, engine *autograd.Engine) *BatchNorm {
	// Инициализация gamma единицами, beta нулями
	gammaData := tensor.Ones(numFeatures)
	betaData := tensor.Zeros(numFeatures)

	gamma := engine.RequireGrad(gammaData)
	beta := engine.RequireGrad(betaData)

	// Инициализация running mean/var
	runningMean := tensor.Zeros(numFeatures)
	runningVar := tensor.Ones(numFeatures)

	return &BatchNorm{
		numFeatures: numFeatures,
		eps:         1e-5,
		momentum:    0.1,
		training:    true,
		gamma:       gamma,
		beta:        beta,
		runningMean: runningMean,
		runningVar:  runningVar,
		engine:      engine,
	}
}

// Forward выполняет прямой проход через слой BatchNorm.
// Входной тензор x должен иметь форму [batch_size, num_features].
func (bn *BatchNorm) Forward(x *graph.Node) *graph.Node {
	if len(x.Value.Shape) != 2 {
		panic("BatchNorm expects 2D input [batch_size, num_features]")
	}

	batchSize := x.Value.Shape[0]
	numFeatures := x.Value.Shape[1]

	if numFeatures != bn.numFeatures {
		panic("Input features dimension doesn't match BatchNorm numFeatures")
	}

	if bn.training {
		// Training mode: используем статистики батча
		return bn.forwardTraining(x, batchSize, numFeatures)
	} else {
		// Inference mode: используем running mean/var
		return bn.forwardInference(x, batchSize, numFeatures)
	}
}

// forwardTraining выполняет forward pass в режиме обучения.
func (bn *BatchNorm) forwardTraining(x *graph.Node, batchSize, numFeatures int) *graph.Node {
	// Вычисляем mean и variance по батчу для каждого признака
	batchMean := tensor.Zeros(numFeatures)
	batchVar := tensor.Zeros(numFeatures)

	// 1. Вычисляем среднее: μ = (1/N) Σ x
	for j := range numFeatures {
		sum := 0.0
		for i := range batchSize {
			idx := i*numFeatures + j
			sum += x.Value.Data[idx]
		}
		batchMean.Data[j] = sum / float64(batchSize)
	}

	// 2. Вычисляем дисперсию: σ² = (1/N) Σ (x - μ)²
	for j := range numFeatures {
		sum := 0.0
		for i := range batchSize {
			idx := i*numFeatures + j
			diff := x.Value.Data[idx] - batchMean.Data[j]
			sum += diff * diff
		}
		batchVar.Data[j] = sum / float64(batchSize)
	}

	// 3. Обновляем running mean/var экспоненциальным скользящим средним
	for j := range numFeatures {
		bn.runningMean.Data[j] = (1-bn.momentum)*bn.runningMean.Data[j] + bn.momentum*batchMean.Data[j]
		bn.runningVar.Data[j] = (1-bn.momentum)*bn.runningVar.Data[j] + bn.momentum*batchVar.Data[j]
	}

	// 4. Нормализуем: x̂ = (x - μ) / sqrt(σ² + ε)
	normalized := tensor.Zeros(x.Value.Shape...)
	for i := range batchSize {
		for j := range numFeatures {
			idx := i*numFeatures + j
			normalized.Data[idx] = (x.Value.Data[idx] - batchMean.Data[j]) / math.Sqrt(batchVar.Data[j]+bn.eps)
		}
	}

	// 5. Применяем масштабирование и сдвиг: y = γ * x̂ + β
	// Используем autograd операции для автоматического вычисления градиентов
	normalizedNode := graph.NewNode(normalized, []*graph.Node{x}, nil)

	// γ * x̂
	scaled := bn.multiplyWithBroadcast(normalizedNode, bn.gamma, batchSize, numFeatures)

	// γ * x̂ + β
	output := bn.addWithBroadcast(scaled, bn.beta, batchSize, numFeatures)

	return output
}

// forwardInference выполняет forward pass в режиме inference.
func (bn *BatchNorm) forwardInference(x *graph.Node, batchSize, numFeatures int) *graph.Node {
	// Нормализуем используя running mean/var: x̂ = (x - μ_running) / sqrt(σ²_running + ε)
	normalized := tensor.Zeros(x.Value.Shape...)
	for i := range batchSize {
		for j := range numFeatures {
			idx := i*numFeatures + j
			normalized.Data[idx] = (x.Value.Data[idx] - bn.runningMean.Data[j]) / math.Sqrt(bn.runningVar.Data[j]+bn.eps)
		}
	}

	normalizedNode := graph.NewNode(normalized, []*graph.Node{x}, nil)

	// Применяем γ и β
	scaled := bn.multiplyWithBroadcast(normalizedNode, bn.gamma, batchSize, numFeatures)
	output := bn.addWithBroadcast(scaled, bn.beta, batchSize, numFeatures)

	return output
}

// multiplyWithBroadcast умножает входной тензор [batch_size, num_features]
// на вектор параметров [num_features] с broadcasting.
func (bn *BatchNorm) multiplyWithBroadcast(x *graph.Node, param *graph.Node, batchSize, numFeatures int) *graph.Node {
	result := tensor.Zeros(x.Value.Shape...)
	for i := range batchSize {
		for j := range numFeatures {
			idx := i*numFeatures + j
			result.Data[idx] = x.Value.Data[idx] * param.Value.Data[j]
		}
	}
	return graph.NewNode(result, []*graph.Node{x, param}, nil)
}

// addWithBroadcast добавляет вектор параметров [num_features] к тензору [batch_size, num_features].
func (bn *BatchNorm) addWithBroadcast(x *graph.Node, param *graph.Node, batchSize, numFeatures int) *graph.Node {
	result := tensor.Zeros(x.Value.Shape...)
	for i := range batchSize {
		for j := range numFeatures {
			idx := i*numFeatures + j
			result.Data[idx] = x.Value.Data[idx] + param.Value.Data[j]
		}
	}
	return graph.NewNode(result, []*graph.Node{x, param}, nil)
}

// Params возвращает обучаемые параметры слоя (gamma и beta).
func (bn *BatchNorm) Params() []*graph.Node {
	return []*graph.Node{bn.gamma, bn.beta}
}

// Train переводит слой в режим обучения.
// func (bn *BatchNorm) Train() {
// 	bn.training = true
// }

// Eval переводит слой в режим inference (оценки).
// func (bn *BatchNorm) Eval() {
// 	bn.training = false
// }

func (bn *BatchNorm) Train() { bn.training = true }
func (bn *BatchNorm) Eval()  { bn.training = false }

// SetMomentum устанавливает коэффициент momentum для обновления running mean/var.
func (bn *BatchNorm) SetMomentum(momentum float64) {
	bn.momentum = momentum
}

// SetEpsilon устанавливает epsilon для численной стабильности.
func (bn *BatchNorm) SetEpsilon(eps float64) {
	bn.eps = eps
}
