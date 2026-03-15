package layers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/matrix"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
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
	inputSize     int
	hiddenSize    int
	numLayers     int
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
//
//	rnn := NewRNN(128, 256, 1, false, heInit) // input=128, hidden=256, 1 слой, однонаправленный
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
		inputSize:     inputSize,
		hiddenSize:    hiddenSize,
		numLayers:     numLayers,
		bidirectional: bidirectional,
		weights:       weights,
		biases:        biases,
	}
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

// rnnOp реализует интерфейс graph.Operation для твоей RNN
type rnnOp struct {
	x       *graph.Node
	rnn     *RNN
	hStates []*tensor.Tensor // Храним h_t для BPTT
}

// Forward выполняет проход по последовательности [batch, seq, input]
func (r *RNN) Forward(x *graph.Node) *graph.Node {
	batchSize := x.Value.Shape[0]
	seqLen := x.Value.Shape[1]

	hStates := make([]*tensor.Tensor, seqLen+1)
	if r.hiddenState == nil {
		hStates[0] = tensor.Zeros(batchSize, r.hiddenSize)
	} else {
		hStates[0] = r.hiddenState
	}

	// Оптимизация: транспонируем веса один раз до цикла
	wihT, _ := matrix.Transposition(matrix.TensorToMatrix(r.weights[0].Value))
	whhT, _ := matrix.Transposition(matrix.TensorToMatrix(r.weights[1].Value))

	bih := r.biases[0].Value
	bhh := r.biases[1].Value

	outputVal := tensor.Zeros(batchSize, seqLen, r.hiddenSize)

	for t := 0; t < seqLen; t++ {
		xt := extractSlice(x.Value, t)
		xtM := matrix.TensorToMatrix(xt)

		// ih = xt * Wih^T
		ihM, _ := matrix.MatMul(xtM, wihT)
		ih := matrix.MatrixToTensor(ihM)

		// hh = h_{t-1} * Whh^T
		htM := matrix.TensorToMatrix(hStates[t])
		hhM, _ := matrix.MatMul(htM, whhT)
		hh := matrix.MatrixToTensor(hhM)

		// Сложение всех компонентов
		sum, _ := tensor.Add(ih, hh)
		sum, _ = tensor.Add(sum, broadcastBias(bih, batchSize))
		sum, _ = tensor.Add(sum, broadcastBias(bhh, batchSize))

		// Активация
		h_t := tensor.Apply(sum, math.Tanh)

		hStates[t+1] = h_t
		copySlice(outputVal, h_t, t)
	}

	r.hiddenState = hStates[seqLen]
	op := &rnnOp{x: x, rnn: r, hStates: hStates}

	parents := append([]*graph.Node{x}, r.weights...)
	parents = append(parents, r.biases...)

	return graph.NewNode(outputVal, parents, op)
}

// Backward реализует ручной BPTT
func (op *rnnOp) Backward(gradOutput *tensor.Tensor) {
	r := op.rnn
	seqLen := op.x.Value.Shape[1]
	batchSize := op.x.Value.Shape[0]

	// Инициализируем накопители градиентов
	dWih := tensor.Zeros(r.weights[0].Value.Shape...)
	dWhh := tensor.Zeros(r.weights[1].Value.Shape...)
	dbih := tensor.Zeros(r.biases[0].Value.Shape...)
	dbhh := tensor.Zeros(r.biases[1].Value.Shape...)
	dx := tensor.Zeros(op.x.Value.Shape...)

	dhNext := tensor.Zeros(batchSize, r.hiddenSize)

	// Идем назад во времени
	for t := seqLen - 1; t >= 0; t-- {
		stepGrad := extractSlice(gradOutput, t)
		dh, _ := tensor.Add(stepGrad, dhNext)

		// d_tanh = (1 - h^2) * dh
		h_t := op.hStates[t+1]
		dtanh := tensor.Apply(h_t, func(v float64) float64 { return 1.0 - v*v })
		dtanh, _ = tensor.Mul(dtanh, dh)

		// Градиенты по весам
		dtanhM := matrix.TensorToMatrix(dtanh)
		dtanhT, _ := matrix.Transposition(dtanhM)

		xt := extractSlice(op.x.Value, t)
		xtM := matrix.TensorToMatrix(xt)
		localDWih, _ := matrix.MatMul(dtanhT, xtM)
		dWih, _ = tensor.Add(dWih, matrix.MatrixToTensor(localDWih))

		hPrevM := matrix.TensorToMatrix(op.hStates[t])
		localDWhh, _ := matrix.MatMul(dtanhT, hPrevM)
		dWhh, _ = tensor.Add(dWhh, matrix.MatrixToTensor(localDWhh))

		// Смещения
		dbih, _ = tensor.Add(dbih, sumAlongBatch(dtanh))
		dbhh, _ = tensor.Add(dbhh, sumAlongBatch(dtanh))

		// Прокидываем градиент назад: dhNext = dtanh * Whh
		whhM := matrix.TensorToMatrix(r.weights[1].Value)
		dhNextM, _ := matrix.MatMul(dtanhM, whhM)
		dhNext = matrix.MatrixToTensor(dhNextM)

		// Градиент по входу x_t
		wihM := matrix.TensorToMatrix(r.weights[0].Value)
		dxtM, _ := matrix.MatMul(dtanhM, wihM)
		copySlice(dx, matrix.MatrixToTensor(dxtM), t)
	}

	// Пишем накопленные градиенты в ноды графа
	accumulate(r.weights[0], dWih)
	accumulate(r.weights[1], dWhh)
	accumulate(r.biases[0], dbih)
	accumulate(r.biases[1], dbhh)
	accumulate(op.x, dx)
}

// --- Вспомогательные функции (вставь их ниже) ---

func accumulate(n *graph.Node, g *tensor.Tensor) {
	if n.Grad == nil {
		n.Grad = tensor.Zeros(n.Value.Shape...)
	}
	n.Grad, _ = tensor.Add(n.Grad, g)
}

func extractSlice(t *tensor.Tensor, step int) *tensor.Tensor {
	b, s, f := t.Shape[0], t.Shape[1], t.Shape[2]
	res := make([]float64, b*f)
	for i := 0; i < b; i++ {
		srcStart := i*s*f + step*f
		// Указываем [цель : до куда] и [источник : до куда]
		copy(res[i*f:(i+1)*f], t.Data[srcStart:srcStart+f])
	}
	return &tensor.Tensor{Data: res, Shape: []int{b, f}, Strides: []int{f, 1}}
}

func copySlice(dest *tensor.Tensor, src *tensor.Tensor, step int) {
	b, s, f := dest.Shape[0], dest.Shape[1], dest.Shape[2]
	for i := 0; i < b; i++ {
		destStart := i*s*f + step*f
		copy(dest.Data[destStart:destStart+f], src.Data[i*f:(i+1)*f])
	}
}

func sumAlongBatch(t *tensor.Tensor) *tensor.Tensor {
	b, f := t.Shape[0], t.Shape[1]
	res := make([]float64, f)
	for i := 0; i < b; i++ {
		for j := 0; j < f; j++ {
			res[j] += t.Data[i*f+j]
		}
	}
	return &tensor.Tensor{Data: res, Shape: []int{f}, Strides: []int{1}}
}

func broadcastBias(b *tensor.Tensor, batch int) *tensor.Tensor {
	f := b.Shape[0]
	data := make([]float64, batch*f)
	for i := 0; i < batch; i++ {
		copy(data[i*f:], b.Data)
	}
	return &tensor.Tensor{Data: data, Shape: []int{batch, f}, Strides: []int{f, 1}}
}
