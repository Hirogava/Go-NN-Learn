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
		//В режиме инференса просто возращаем
		return x
	}

	// В режиме обучения применяем dropout
	xTensor := x.Value

	// Создаем маску dropout: 1 с вероятностью (1-rate), 0 с вероятностью rate
	maskData := make([]float64, len(xTensor.Data))
	keepProb := 1.0 - d.rate

	for i := range maskData {
		if rand.Float64() < keepProb {
			// on и масштабируется
			maskData[i] = 1.0 / keepProb
		} else {
			// off
			maskData[i] = 0.0
		}
	}

	// Сохраняем маску для backward pass
	d.mask = &tensor.Tensor{
		Data:    maskData,
		Shape:   append([]int{}, xTensor.Shape...),
		Strides: append([]int{}, xTensor.Strides...),
	}

	// Применяем маску к входным данным
	outputData := make([]float64, len(xTensor.Data))
	for i := range outputData {
		outputData[i] = xTensor.Data[i] * maskData[i]
	}

	output := &graph.Node{
		Value: &tensor.Tensor{
			Data:    outputData,
			Shape:   append([]int{}, xTensor.Shape...),
			Strides: append([]int{}, xTensor.Strides...),
		},
	}

	// Устанавливаем операцию для backward pass
	output.Operation = &dropoutOp{
		x:    x,
		mask: d.mask,
	}

	return output
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

// Обратное распространение градиента
func (op *dropoutOp) Backward(grad *tensor.Tensor) {
	if op.x.Grad == nil {
		op.x.Grad = tensor.Zeros(op.x.Value.Shape...)
	}

	gradData := make([]float64, len(grad.Data))
	for i := range gradData {
		gradData[i] = grad.Data[i] * op.mask.Data[i]
	}

	gradInput := &tensor.Tensor{
		Data:    gradData,
		Shape:   append([]int{}, grad.Shape...),
		Strides: append([]int{}, grad.Strides...),
	}

	result, _ := tensor.Add(op.x.Grad, gradInput)
	op.x.Grad = result
}

func (d *Dropout) Train() {
	d.training = true
}

func (d *Dropout) Eval() {
	d.training = false
}
