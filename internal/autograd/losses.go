package autograd

import (
	"math"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
)

type LossOp interface {
	Forward() *tensor.Tensor
	Backward(grad *tensor.Tensor)
}

// MSELoss (Mean Squared Error Loss)
// Используется для задач регрессии
// Forward: Loss = mean((y_pred - y_true)^2)
// Backward: dL/dy_pred = 2 * (y_pred - y_true) / n
type MSELossOp struct {
	pred   *graph.Node    // Предсказанные значения
	target *tensor.Tensor // Истинные значения
	diff   *tensor.Tensor // Сохраненная разность для backward (pred - target)
	n      float64        // Количество элементов для нормализации
}

// NewMSELossOp создает новую операцию MSE Loss
func NewMSELossOp(pred *graph.Node, target *tensor.Tensor) *MSELossOp {
	return &MSELossOp{pred: pred, target: target}
}

// Forward вычисляет MSE loss
func (op *MSELossOp) Forward() *tensor.Tensor {
	// Проверка совпадения форм
	if len(op.pred.Value.Shape) != len(op.target.Shape) {
		panic("MSELoss: pred and target must have the same number of dimensions")
	}
	for i := range op.pred.Value.Shape {
		if op.pred.Value.Shape[i] != op.target.Shape[i] {
			panic("MSELoss: pred and target must have the same shape")
		}
	}

	// Вычисляем разность: diff = pred - target
	diff := tensor.Zeros(op.pred.Value.Shape...)
	for i := range op.pred.Value.Data {
		diff.Data[i] = op.pred.Value.Data[i] - op.target.Data[i]
	}
	op.diff = diff

	// Вычисляем квадраты разностей и их сумму
	sumSquares := 0.0
	for i := range diff.Data {
		sumSquares += diff.Data[i] * diff.Data[i]
	}

	// Среднее значение
	op.n = float64(len(diff.Data))
	meanLoss := sumSquares / op.n

	// Возвращаем скаляр (тензор размера 1)
	result := tensor.Zeros(1)
	result.Data[0] = meanLoss
	return result
}

// Backward вычисляет градиенты для MSE loss
func (op *MSELossOp) Backward(grad *tensor.Tensor) {
	// Градиент: dL/dy_pred = 2 * (y_pred - y_true) / n * grad
	// grad обычно равен 1.0 для функции потерь
	if op.pred.Grad == nil {
		op.pred.Grad = tensor.Zeros(op.pred.Value.Shape...)
	}

	gradScale := 2.0 / op.n * grad.Data[0]
	for i := range op.diff.Data {
		op.pred.Grad.Data[i] += gradScale * op.diff.Data[i]
	}
}

// MSELoss создает узел графа для MSE loss
func (e *Engine) MSELoss(pred *graph.Node, target *tensor.Tensor) *graph.Node {
	op := NewMSELossOp(pred, target)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{pred}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

// CrossEntropyLogitsOp (Cross-Entropy Loss with Logits)
// Численно стабильная реализация для задач классификации
// Объединяет Softmax и Cross-Entropy для численной стабильности
// Forward: Loss = -sum(target * log_softmax(logits))
// Backward: dL/dlogits = (softmax(logits) - target) / batch_size
type CrossEntropyLogitsOp struct {
	logits  *graph.Node    // Входные логиты (сырые выходы сети)
	target  *tensor.Tensor // One-hot закодированные метки
	softmax *tensor.Tensor // Сохраненные вероятности softmax для backward
}

// NewCrossEntropyLogitsOp создает новую операцию Cross-Entropy Loss
func NewCrossEntropyLogitsOp(logits *graph.Node, target *tensor.Tensor) *CrossEntropyLogitsOp {
	return &CrossEntropyLogitsOp{logits: logits, target: target}
}

// Forward вычисляет cross-entropy loss численно стабильным способом
func (op *CrossEntropyLogitsOp) Forward() *tensor.Tensor {
	// Проверка: должны быть 2D тензоры [batch_size, num_classes]
	if len(op.logits.Value.Shape) != 2 || len(op.target.Shape) != 2 {
		panic("CrossEntropyLogits: logits and target must be 2D tensors")
	}
	if op.logits.Value.Shape[0] != op.target.Shape[0] {
		panic("CrossEntropyLogits: batch sizes must match")
	}
	if op.logits.Value.Shape[1] != op.target.Shape[1] {
		panic("CrossEntropyLogits: number of classes must match")
	}

	batchSize := op.logits.Value.Shape[0]
	numClasses := op.logits.Value.Shape[1]

	// Шаг 1: Численно стабильный softmax с log-sum-exp трюком
	// Находим максимум для каждого примера в батче
	softmax := tensor.Zeros(batchSize, numClasses)

	for i := range batchSize {
		// Находим максимум по строке
		maxVal := op.logits.Value.Data[i*op.logits.Value.Strides[0]]
		for j := 1; j < numClasses; j++ {
			idx := i*op.logits.Value.Strides[0] + j*op.logits.Value.Strides[1]
			if op.logits.Value.Data[idx] > maxVal {
				maxVal = op.logits.Value.Data[idx]
			}
		}

		// Вычисляем exp(x - max) и сумму
		sumExp := 0.0
		for j := range numClasses {
			idx := i*op.logits.Value.Strides[0] + j*op.logits.Value.Strides[1]
			shifted := op.logits.Value.Data[idx] - maxVal
			expVal := math.Exp(shifted)
			softmax.Data[idx] = expVal
			sumExp += expVal
		}

		// Нормализуем для получения softmax
		for j := range numClasses {
			idx := i*op.logits.Value.Strides[0] + j*op.logits.Value.Strides[1]
			softmax.Data[idx] /= sumExp
		}
	}
	op.softmax = softmax

	// Шаг 2: Вычисляем cross-entropy: -sum(target * log(softmax))
	totalLoss := 0.0
	const epsilon = 1e-15 // Для численной стабильности log

	for i := range batchSize {
		for j := range numClasses {
			idx := i*op.logits.Value.Strides[0] + j*op.logits.Value.Strides[1]
			targetIdx := i*op.target.Strides[0] + j*op.target.Strides[1]

			if op.target.Data[targetIdx] > 0 {
				// Добавляем epsilon для избежания log(0)
				prob := math.Max(softmax.Data[idx], epsilon)
				totalLoss -= op.target.Data[targetIdx] * math.Log(prob)
			}
		}
	}

	// Возвращаем средний loss по батчу
	avgLoss := totalLoss / float64(batchSize)
	result := tensor.Zeros(1)
	result.Data[0] = avgLoss
	return result
}

// Backward вычисляет градиенты для cross-entropy loss
func (op *CrossEntropyLogitsOp) Backward(grad *tensor.Tensor) {
	batchSize := op.logits.Value.Shape[0]
	numClasses := op.logits.Value.Shape[1]

	if op.logits.Grad == nil {
		op.logits.Grad = tensor.Zeros(op.logits.Value.Shape...)
	}

	// Градиент: dL/dlogits = (softmax - target) / batch_size * grad
	gradScale := grad.Data[0] / float64(batchSize)

	for i := range batchSize {
		for j := range numClasses {
			idx := i*op.logits.Value.Strides[0] + j*op.logits.Value.Strides[1]
			targetIdx := i*op.target.Strides[0] + j*op.target.Strides[1]

			op.logits.Grad.Data[idx] += (op.softmax.Data[idx] - op.target.Data[targetIdx]) * gradScale
		}
	}
}

// CrossEntropyLoss создает узел графа для cross-entropy loss
func (e *Engine) CrossEntropyLoss(logits *graph.Node, target *tensor.Tensor) *graph.Node {
	op := NewCrossEntropyLogitsOp(logits, target)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{logits}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}

// HingeLossOp (Hinge Loss)
// Используется в SVM и задачах бинарной классификации
// Forward: Loss = mean(max(0, 1 - y_true * y_pred))
// Backward: dL/dy_pred = -y_true / n  если (1 - y_true * y_pred) > 0, иначе 0
type HingeLossOp struct {
	pred   *graph.Node    // Предсказанные значения
	target *tensor.Tensor // Истинные метки (-1 или +1)
	margin *tensor.Tensor // Сохраненные значения margin для backward
	n      float64        // Количество элементов для нормализации
}

// NewHingeLossOp создает новую операцию Hinge Loss
func NewHingeLossOp(pred *graph.Node, target *tensor.Tensor) *HingeLossOp {
	return &HingeLossOp{pred: pred, target: target}
}

// Forward вычисляет hinge loss
func (op *HingeLossOp) Forward() *tensor.Tensor {
	// Проверка совпадения форм
	if len(op.pred.Value.Shape) != len(op.target.Shape) {
		panic("HingeLoss: pred and target must have the same number of dimensions")
	}
	for i := range op.pred.Value.Shape {
		if op.pred.Value.Shape[i] != op.target.Shape[i] {
			panic("HingeLoss: pred and target must have the same shape")
		}
	}

	// Вычисляем margin: 1 - y_true * y_pred
	margin := tensor.Zeros(op.pred.Value.Shape...)
	sumLoss := 0.0

	for i := range op.pred.Value.Data {
		marginVal := 1.0 - op.target.Data[i]*op.pred.Value.Data[i]
		// Применяем max(0, margin)
		if marginVal > 0 {
			margin.Data[i] = marginVal
			sumLoss += marginVal
		} else {
			margin.Data[i] = 0
		}
	}
	op.margin = margin

	// Среднее значение loss
	op.n = float64(len(margin.Data))
	avgLoss := sumLoss / op.n

	result := tensor.Zeros(1)
	result.Data[0] = avgLoss
	return result
}

// Backward вычисляет градиенты для hinge loss
func (op *HingeLossOp) Backward(grad *tensor.Tensor) {
	if op.pred.Grad == nil {
		op.pred.Grad = tensor.Zeros(op.pred.Value.Shape...)
	}

	// Градиент: dL/dy_pred = -y_true / n * grad  если margin > 0, иначе 0
	gradScale := grad.Data[0] / op.n

	for i := range op.margin.Data {
		if op.margin.Data[i] > 0 {
			// Производная max(0, 1 - y_true * y_pred) по y_pred = -y_true
			op.pred.Grad.Data[i] += -op.target.Data[i] * gradScale
		}
		// Если margin <= 0, градиент = 0 (ничего не добавляем)
	}
}

// HingeLoss создает узел графа для hinge loss
func (e *Engine) HingeLoss(pred *graph.Node, target *tensor.Tensor) *graph.Node {
	op := NewHingeLossOp(pred, target)
	result := op.Forward()
	node := graph.NewNode(result, []*graph.Node{pred}, op)
	e.Nodes = append(e.Nodes, node)
	return node
}
