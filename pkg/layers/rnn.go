package layers

import (
	"fmt"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// RNN представляет базовый рекуррентный слой для обработки последовательных данных.
// Обрабатывает входные последовательности, поддерживая скрытое состояние между временными шагами.
//
// Входной формат: [batch_size, sequence_length, input_size] или [sequence_length, batch_size, input_size]
// Выходной формат: [batch_size, sequence_length, hidden_size] или [sequence_length, batch_size, hidden_size]
//
// Параметры:
//   - inputSize: размер входного вектора на каждом временном шаге
//   - hiddenSize: размер скрытого состояния
//   - numLayers: количество слоёв RNN (по умолчанию 1)
//   - bidirectional: использовать ли двунаправленный RNN (по умолчанию false)
//
// Это каркас (skeleton) - полная реализация RNN будет добавлена позже.
type RNN struct {
	inputSize    int
	hiddenSize   int
	numLayers    int
	bidirectional bool

	// Обучаемые параметры для каждого слоя
	// Для каждого слоя нужны веса:
	//   - W_ih: веса input-to-hidden [hidden_size, input_size]
	//   - W_hh: веса hidden-to-hidden [hidden_size, hidden_size]
	//   - b_ih: смещение input-to-hidden [hidden_size]
	//   - b_hh: смещение hidden-to-hidden [hidden_size]
	weights []*graph.Node // Список всех весовых матриц
	biases  []*graph.Node // Список всех векторов смещения

	// Внутреннее состояние
	hiddenState *tensor.Tensor // Текущее скрытое состояние [batch_size, hidden_size]
}

// NewRNN создает новый рекуррентный слой RNN.
// initFunc используется для инициализации весов.
//
// Пример:
//   rnn := NewRNN(128, 256, 1, false, heInit) // input=128, hidden=256, 1 слой, однонаправленный
func NewRNN(
	inputSize, hiddenSize, numLayers int,
	bidirectional bool,
	initFunc func([]float64),
) *RNN {
	weights := make([]*graph.Node, 0)
	biases := make([]*graph.Node, 0)

	// Создаем параметры для каждого слоя
	for layer := 0; layer < numLayers; layer++ {
		layerInputSize := inputSize
		if layer > 0 {
			// Для последующих слоёв вход - это выход предыдущего
			layerInputSize = hiddenSize
			if bidirectional {
				layerInputSize = hiddenSize * 2 // Двунаправленный удваивает размер
			}
		}

		// W_ih: веса input-to-hidden
		wihSize := hiddenSize * layerInputSize
		wihData := make([]float64, wihSize)
		initFunc(wihData)
		weights = append(weights, &graph.Node{
			Value: &tensor.Tensor{
				Data:    wihData,
				Shape:   []int{hiddenSize, layerInputSize},
				Strides: []int{layerInputSize, 1},
			},
		})

		// W_hh: веса hidden-to-hidden
		whhSize := hiddenSize * hiddenSize
		whhData := make([]float64, whhSize)
		initFunc(whhData)
		weights = append(weights, &graph.Node{
			Value: &tensor.Tensor{
				Data:    whhData,
				Shape:   []int{hiddenSize, hiddenSize},
				Strides: []int{hiddenSize, 1},
			},
		})

		// b_ih: смещение input-to-hidden
		bihData := make([]float64, hiddenSize)
		initFunc(bihData)
		biases = append(biases, &graph.Node{
			Value: &tensor.Tensor{
				Data:    bihData,
				Shape:   []int{hiddenSize},
				Strides: []int{1},
			},
		})

		// b_hh: смещение hidden-to-hidden
		bhhData := make([]float64, hiddenSize)
		initFunc(bhhData)
		biases = append(biases, &graph.Node{
			Value: &tensor.Tensor{
				Data:    bhhData,
				Shape:   []int{hiddenSize},
				Strides: []int{1},
			},
		})
	}

	return &RNN{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		numLayers:    numLayers,
		bidirectional: bidirectional,
		weights:      weights,
		biases:       biases,
	}
}

// Forward выполняет прямое распространение через RNN слой.
// Вход x должен иметь форму [batch_size, sequence_length, input_size] или
// [sequence_length, batch_size, input_size] в зависимости от реализации.
//
// TODO: Реализовать полную обработку последовательности с сохранением скрытого состояния.
// Сейчас это каркас, который проверяет вход и возвращает заглушку.
func (r *RNN) Forward(x *graph.Node) *graph.Node {
	if x == nil || x.Value == nil {
		panic("RNN.Forward: input is nil")
	}

	// Проверка формы входа: должна быть 3D [batch, seq_len, input_size] или [seq_len, batch, input_size]
	if len(x.Value.Shape) != 3 {
		panic(fmt.Sprintf("RNN expects 3D input [batch, seq_len, input_size] or [seq_len, batch, input_size], got %dD", len(x.Value.Shape)))
	}

	// Определяем формат входа (batch-first или sequence-first)
	// Предполагаем batch-first: [batch_size, sequence_length, input_size]
	batchSize := x.Value.Shape[0]
	sequenceLength := x.Value.Shape[1]
	inputSize := x.Value.Shape[2]

	// Проверка соответствия размера входа
	if inputSize != r.inputSize {
		panic(fmt.Sprintf("RNN: input size mismatch: expected %d, got %d", r.inputSize, inputSize))
	}

	// Инициализация скрытого состояния, если ещё не инициализировано
	if r.hiddenState == nil {
		r.hiddenState = tensor.Zeros(batchSize, r.hiddenSize)
	}

	// TODO: Реализовать реальную обработку последовательности
	// Алгоритм для каждого временного шага:
	//   h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
	// где:
	//   - x_t: вход на шаге t [batch_size, input_size]
	//   - h_{t-1}: скрытое состояние на предыдущем шаге [batch_size, hidden_size]
	//   - h_t: новое скрытое состояние [batch_size, hidden_size]

	// Временная заглушка: возвращаем тензор правильной формы
	outputSize := batchSize * sequenceLength * r.hiddenSize
	outputData := make([]float64, outputSize)
	// Инициализация нулями (заглушка)

	output := &graph.Node{
		Value: &tensor.Tensor{
			Data:    outputData,
			Shape:   []int{batchSize, sequenceLength, r.hiddenSize},
			Strides: []int{sequenceLength * r.hiddenSize, r.hiddenSize, 1},
		},
	}

	// TODO: Реализовать операцию для backward pass
	// output.Operation = &rnnOp{...}

	// Временное предупреждение
	fmt.Printf("WARNING: RNN.Forward is a skeleton - full RNN processing not implemented yet\n")

	return output
}

// Params возвращает все обучаемые параметры слоя (веса и смещения всех слоёв).
func (r *RNN) Params() []*graph.Node {
	params := make([]*graph.Node, 0, len(r.weights)+len(r.biases))
	params = append(params, r.weights...)
	params = append(params, r.biases...)
	return params
}

// ResetHiddenState сбрасывает скрытое состояние RNN.
// Полезно при начале новой последовательности.
func (r *RNN) ResetHiddenState() {
	r.hiddenState = nil
}

// GetHiddenState возвращает текущее скрытое состояние.
func (r *RNN) GetHiddenState() *tensor.Tensor {
	return r.hiddenState
}

// SetHiddenState устанавливает скрытое состояние (полезно для инициализации).
func (r *RNN) SetHiddenState(h *tensor.Tensor) {
	r.hiddenState = h
}

// GetInputSize возвращает размер входного вектора.
func (r *RNN) GetInputSize() int {
	return r.inputSize
}

// GetHiddenSize возвращает размер скрытого состояния.
func (r *RNN) GetHiddenSize() int {
	return r.hiddenSize
}

// GetNumLayers возвращает количество слоёв RNN.
func (r *RNN) GetNumLayers() int {
	return r.numLayers
}

// IsBidirectional возвращает true если RNN двунаправленный.
func (r *RNN) IsBidirectional() bool {
	return r.bidirectional
}

// rnnOp представляет операцию RNN для backward pass.
// TODO: Реализовать Backward для вычисления градиентов через время (BPTT).
type rnnOp struct {
	x   *graph.Node
	rnn *RNN
}

func (op *rnnOp) Backward(grad *tensor.Tensor) {
	// TODO: Реализовать обратное распространение через время (BPTT - Backpropagation Through Time)
	// Это включает:
	// 1. Развёртку последовательности
	// 2. Вычисление градиентов для каждого временного шага в обратном порядке
	// 3. Накопление градиентов по весам и смещениям
	// 4. Вычисление градиента по входу
	panic("RNN.Backward not implemented yet - this is a skeleton")
}

