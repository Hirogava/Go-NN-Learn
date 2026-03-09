package layers

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestInferenceModeSwitch(t *testing.T) {
	engine := autograd.NewEngine()

	// 1. Инициализация слоев
	drop := NewDropout(1.0) // В режиме Train зануляет всё
	bn := NewBatchNorm(1, engine)
	bn.momentum = 1.0 // Для теста мгновенно обновляем статистики

	// --- ТЕСТ РЕЖИМА TRAIN ---
	drop.Train()
	bn.Train()

	input := &tensor.Tensor{Data: []float64{10.0}, Shape: []int{1, 1}}
	inputNode := graph.NewNode(input, nil, nil)

	// В режиме Train (rate 1.0) Dropout выдаст 0. BN от 0 выдаст 0.
	outTrain := bn.Forward(drop.Forward(inputNode))
	if outTrain.Value.Data[0] != 0 {
		t.Errorf("Train mode failed: Dropout should zero out everything, got %v", outTrain.Value.Data[0])
	}

	// --- ПОДГОТОВКА СТАТИСТИКИ (BN Learning) ---
	bn.Train()
	// Подаем два числа: 40 и 60.
	// Среднее (μ) = 50.
	// Дисперсия (σ²) = ((40-50)² + (60-50)²)/2 = (100+100)/2 = 100.
	trainData := &tensor.Tensor{
		Data:  []float64{40.0, 60.0},
		Shape: []int{2, 1},
	}
	bn.Forward(graph.NewNode(trainData, nil, nil))

	// --- ТЕСТ РЕЖИМА EVAL ---
	drop.Eval()
	bn.Eval()

	// Снова подаем вход 10.0
	// 1. Dropout в Eval пропустит 10.0 как есть.
	// 2. BatchNorm использует runningMean=50 и runningVar=100.
	// Результат: (10 - 50) / sqrt(100 + eps) ≈ -40 / 10 = -4.
	outEval := bn.Forward(drop.Forward(inputNode))
	val := outEval.Value.Data[0]

	expected := (10.0 - 50.0) / math.Sqrt(100.0+bn.eps)

	if math.Abs(val-expected) > 1e-2 {
		t.Errorf("Eval mode failed. Got %v, expected %v. Check switches!", val, expected)
	}
}
