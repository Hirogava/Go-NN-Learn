package layers

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// simpleInit заполняет веса небольшим константным значением для предсказуемости теста
func simpleInit(data []float64) {
	for i := range data {
		data[i] = 0.1
	}
}

func TestRNN_ForwardBackward(t *testing.T) {
	// Параметры для теста
	batchSize := 2
	seqLen := 3
	inputSize := 4
	hiddenSize := 5

	// 1. Создаем слой RNN
	rnn := NewRNN(inputSize, hiddenSize, 1, false, simpleInit)

	// 2. Создаем входной узел [batch, seq, input]
	xValue := tensor.Zeros(batchSize, seqLen, inputSize)
	for i := range xValue.Data {
		xValue.Data[i] = float64(i) * 0.01
	}
	// Важно: создаем Node через NewNode, чтобы граф построился
	xNode := graph.NewNode(xValue, nil, nil)

	// 3. Выполняем Forward
	// Сбрасываем состояние перед тестом
	rnn.ResetHiddenState()
	outputNode := rnn.Forward(xNode)

	// Проверка формы выхода: [batch, seq, hidden]
	expectedShape := []int{batchSize, seqLen, hiddenSize}
	for i, dim := range outputNode.Value.Shape {
		if dim != expectedShape[i] {
			t.Errorf("Wrong output shape: expected %v, got %v", expectedShape, outputNode.Value.Shape)
		}
	}

	// 4. Выполняем Backward
	// Создаем "входящий" градиент (например, от функции потерь)
	gradOut := tensor.Zeros(outputNode.Value.Shape...)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0 // Допустим, ошибка везде одинаковая
	}

	// В твоем графе поле называется Operation (с большой буквы)
	if outputNode.Operation == nil {
		t.Fatal("RNN Forward did not set Operation in output node")
	}

	// Запускаем обратный проход для слоя
	outputNode.Operation.Backward(gradOut)

	// 5. Проверяем градиенты

	// Проверка весов W_ih (индекс 0) и W_hh (индекс 1)
	for i := 0; i < 2; i++ {
		wNode := rnn.weights[i]
		if wNode.Grad == nil {
			t.Errorf("Weight grad [%d] is nil", i)
			continue
		}

		// Проверяем, что градиент не нулевой
		isZero := true
		for _, v := range wNode.Grad.Data {
			if v != 0 {
				isZero = false
				break
			}
		}
		if isZero {
			t.Errorf("Weight grad [%d] is all zeros, BPTT might be broken", i)
		}
	}

	// Проверка градиента по входу X
	if xNode.Grad == nil {
		t.Error("Input grad (dx) is nil")
	} else {
		t.Log("Successfully computed input gradient")
	}
}
