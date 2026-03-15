package layers

import (
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// cлой для регуляризации нейронной сети
type Dropout struct {
	rate     float64 // вероятность отключения нейрона (dropout rate)
	training bool    // флаг режима обучения
	mask     *tensor.Tensor
}

// NewDropout создает новый слой Dropout
// rate - вероятность отключения нейрона (например, 0.5 означает 50% нейронов будут отключены)
func NewDropout(rate float64) *Dropout {
	return &Dropout{
		rate:     rate,
		training: true,
		mask:     nil,
	}
}

// SetTraining устанавливает режим работы слоя
func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

// Forward выполняет прямое распространение через слой Dropout
func (d *Dropout) Forward(x *graph.Node) *graph.Node {
	if !d.training {
		return x
	}

	xTensor := x.Value
	maskData := make([]float64, len(xTensor.Data))
	keepProb := 1.0 - d.rate
	scale := 1.0 / keepProb

	for i := range maskData {
		if rand.Float64() < keepProb {
			maskData[i] = scale
		} else {
			maskData[i] = 0.0
		}
	}

	outputData := make([]float64, len(xTensor.Data))
	for i := range outputData {
		outputData[i] = xTensor.Data[i] * maskData[i]
	}

	// Создаем тензор результата
	resTensor := &tensor.Tensor{
		Data:    outputData,
		Shape:   append([]int{}, xTensor.Shape...),
		Strides: append([]int{}, xTensor.Strides...),
	}

	// Маску создаем локально для операции
	mask := &tensor.Tensor{
		Data:    maskData,
		Shape:   append([]int{}, xTensor.Shape...),
		Strides: append([]int{}, xTensor.Strides...),
	}

	op := &dropoutOp{
		x:    x,
		mask: mask,
	}

	// ИСПОЛЬЗУЕМ NewNode вместо ручного создания
	return graph.NewNode(resTensor, []*graph.Node{x}, op)
}

// Параметры слоя
func (d *Dropout) Params() []*graph.Node {
	return []*graph.Node{}
}

// Обратное распрстрнение для дропа
type dropoutOp struct {
	x    *graph.Node
	mask *tensor.Tensor
}

func (op *dropoutOp) Backward(grad *tensor.Tensor) {
	// Используем твою логику суммирования (она верная),
	// но убедимся, что градиент инициализирован
	if op.x.Grad == nil {
		op.x.Grad = tensor.Zeros(op.x.Value.Shape...)
	}

	for i := range grad.Data {
		// Применяем маску (масштабирование уже включено в значения маски)
		op.x.Grad.Data[i] += grad.Data[i] * op.mask.Data[i]
	}
}

func (d *Dropout) Train() {
	d.training = true
}

func (d *Dropout) Eval() {
	d.training = false
}
