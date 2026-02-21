package layers

import (
	"fmt"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
)

// Conv2D представляет сверточный слой для обработки 2D пространственных данных.
// Применяет свертку (convolution) с обучаемыми фильтрами (весами).
//
// Входной формат: [batch_size, in_channels, height, width]
// Выходной формат: [batch_size, out_channels, out_height, out_width]
//
// Параметры:
//   - inChannels: количество входных каналов (например, 3 для RGB изображений)
//   - outChannels: количество выходных каналов (количество фильтров)
//   - kernelSize: размер ядра свертки (например, 3 для 3x3 фильтра)
//   - stride: шаг свертки (по умолчанию 1)
//   - padding: дополнение нулями вокруг входа (по умолчанию 0)
//
// Это каркас (skeleton) - полная реализация свертки будет добавлена позже.
type Conv2D struct {
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	padding     int

	// Обучаемые параметры
	weights *graph.Node // Фильтры: [out_channels, in_channels, kernel_h, kernel_w]
	bias    *graph.Node // Смещение: [out_channels]

	// Внутреннее состояние для backward pass
	inputShape []int // Сохраняется для вычисления градиентов
}

// NewConv2D создает новый сверточный слой Conv2D.
// initFunc используется для инициализации весов (например, He/Xavier).
//
// Пример:
//
//	conv := NewConv2D(3, 64, 3, 1, 1, heInit) // 3->64 каналов, ядро 3x3, stride=1, padding=1
func NewConv2D(
	inChannels, outChannels, kernelSize, stride, padding int,
	initFunc func([]float64),
) *Conv2D {
	// Вычисляем размер тензора весов
	// Веса имеют форму [out_channels, in_channels, kernel_h, kernel_w]
	weightsSize := outChannels * inChannels * kernelSize * kernelSize
	weightsData := make([]float64, weightsSize)
	initFunc(weightsData)

	weights := &graph.Node{
		Value: &tensor.Tensor{
			Data:    weightsData,
			Shape:   []int{outChannels, inChannels, kernelSize, kernelSize},
			Strides: []int{inChannels * kernelSize * kernelSize, kernelSize * kernelSize, kernelSize, 1},
		},
	}

	// Инициализация bias
	biasData := make([]float64, outChannels)
	initFunc(biasData)

	bias := &graph.Node{
		Value: &tensor.Tensor{
			Data:    biasData,
			Shape:   []int{outChannels},
			Strides: []int{1},
		},
	}

	return &Conv2D{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		weights:     weights,
		bias:        bias,
	}
}

// Forward выполняет прямое распространение через сверточный слой.
// Вход x должен иметь форму [batch_size, in_channels, height, width].
//
// TODO: Реализовать полную свертку с использованием im2col или прямого алгоритма.
// Сейчас это каркас, который проверяет вход и возвращает заглушку.
func (c *Conv2D) Forward(x *graph.Node) *graph.Node {
	if x == nil || x.Value == nil {
		panic("Conv2D.Forward: input is nil")
	}

	// Проверка формы входа: должна быть 4D [batch, channels, height, width]
	if len(x.Value.Shape) != 4 {
		panic(fmt.Sprintf("Conv2D expects 4D input [batch, channels, height, width], got %dD", len(x.Value.Shape)))
	}

	batchSize := x.Value.Shape[0]
	inputChannels := x.Value.Shape[1]
	inputHeight := x.Value.Shape[2]
	inputWidth := x.Value.Shape[3]

	// Проверка соответствия количества каналов
	if inputChannels != c.inChannels {
		panic(fmt.Sprintf("Conv2D: input channels mismatch: expected %d, got %d", c.inChannels, inputChannels))
	}

	// Сохраняем форму входа для backward pass
	c.inputShape = append([]int{}, x.Value.Shape...)

	// Вычисляем размеры выхода
	// out_height = (in_height + 2*padding - kernel_size) / stride + 1
	// out_width = (in_width + 2*padding - kernel_size) / stride + 1
	outHeight := (inputHeight+2*c.padding-c.kernelSize)/c.stride + 1
	outWidth := (inputWidth+2*c.padding-c.kernelSize)/c.stride + 1

	// TODO: Реализовать реальную свертку
	// Временная заглушка: возвращаем тензор правильной формы, заполненный нулями
	outputSize := batchSize * c.outChannels * outHeight * outWidth
	outputData := make([]float64, outputSize)
	// Инициализация нулями (заглушка)

	output := &graph.Node{
		Value: &tensor.Tensor{
			Data:    outputData,
			Shape:   []int{batchSize, c.outChannels, outHeight, outWidth},
			Strides: []int{c.outChannels * outHeight * outWidth, outHeight * outWidth, outWidth, 1},
		},
	}

	// TODO: Реализовать операцию для backward pass
	// output.Operation = &conv2dOp{...}

	// Временное предупреждение
	fmt.Printf("WARNING: Conv2D.Forward is a skeleton - full convolution not implemented yet\n")

	return output
}

// Params возвращает обучаемые параметры слоя (веса и смещение).
func (c *Conv2D) Params() []*graph.Node {
	return []*graph.Node{c.weights, c.bias}
}

// GetInChannels возвращает количество входных каналов.
func (c *Conv2D) GetInChannels() int {
	return c.inChannels
}

// GetOutChannels возвращает количество выходных каналов.
func (c *Conv2D) GetOutChannels() int {
	return c.outChannels
}

// GetKernelSize возвращает размер ядра свертки.
func (c *Conv2D) GetKernelSize() int {
	return c.kernelSize
}

// GetStride возвращает шаг свертки.
func (c *Conv2D) GetStride() int {
	return c.stride
}

// GetPadding возвращает размер дополнения.
func (c *Conv2D) GetPadding() int {
	return c.padding
}

// conv2dOp представляет операцию свертки для backward pass.
// TODO: Реализовать Backward для вычисления градиентов.
type conv2dOp struct {
	x      *graph.Node
	conv2d *Conv2D
}

func (op *conv2dOp) Backward(grad *tensor.Tensor) {
	// TODO: Реализовать обратное распространение для свертки
	// Это включает:
	// 1. Вычисление градиента по входу (обратная свертка)
	// 2. Вычисление градиента по весам
	// 3. Вычисление градиента по bias
	panic("Conv2D.Backward not implemented yet - this is a skeleton")
}
