package layers

import (
	"fmt"
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// Тест для старой функции BatchNormVector
func TestBatchNormVector(t *testing.T) {
	batch := []float64{-3, 0, 100, 100000}
	norm := BatchNormVector(batch, 1.0, 0.0)

	// Проверяем что сумма нормализованных значений близка к 0
	sum := 0.0
	for _, el := range norm {
		sum += el
	}

	// Среднее должно быть близко к 0
	mean := sum / 4
	if math.Abs(mean) > 1e-10 {
		t.Errorf("BatchNormVector mean = %v, want ~0", mean)
	}

	// Проверяем дисперсию (должна быть близка к 1)
	variance := 0.0
	for _, el := range norm {
		variance += el * el
	}
	variance /= 4

	if math.Abs(variance-1.0) > 1e-5 {
		t.Errorf("BatchNormVector variance = %v, want ~1.0", variance)
	}
}

// TestBatchNormForwardTraining тестирует forward pass в режиме training
func TestBatchNormForwardTraining(t *testing.T) {
	engine := autograd.NewEngine()

	// Создаем простой батч: 2 примера, 3 признака
	// [[1, 2, 3],
	//  [4, 5, 6]]
	inputData := tensor.Zeros(2, 3)
	inputData.Data = []float64{1, 2, 3, 4, 5, 6}

	inputNode := graph.NewNode(inputData, nil, nil)

	// Создаем BatchNorm слой
	bn := NewBatchNorm(3, engine)
	bn.Train() // Режим обучения

	// Forward pass
	output := bn.Forward(inputNode)

	// Проверяем форму выхода
	if len(output.Value.Shape) != 2 || output.Value.Shape[0] != 2 || output.Value.Shape[1] != 3 {
		t.Errorf("Output shape = %v, want [2, 3]", output.Value.Shape)
	}

	// Проверяем что для каждого признака среднее ~0 и дисперсия ~1
	for j := 0; j < 3; j++ {
		sum := 0.0
		for i := 0; i < 2; i++ {
			idx := i*3 + j
			sum += output.Value.Data[idx]
		}
		mean := sum / 2

		if math.Abs(mean) > 1e-5 {
			t.Errorf("Feature %d: mean = %v, want ~0", j, mean)
		}
	}

	// Проверяем что running mean/var обновились
	if bn.runningMean.Data[0] == 0.0 {
		t.Error("Running mean should be updated after forward pass")
	}
}

// TestBatchNormForwardInference тестирует forward pass в режиме inference
func TestBatchNormForwardInference(t *testing.T) {
	engine := autograd.NewEngine()

	// Создаем BatchNorm слой
	bn := NewBatchNorm(3, engine)

	// Устанавливаем известные running mean/var
	bn.runningMean.Data = []float64{1.0, 2.0, 3.0}
	bn.runningVar.Data = []float64{1.0, 4.0, 9.0}

	bn.Eval() // Режим inference

	// Создаем вход
	inputData := tensor.Zeros(2, 3)
	inputData.Data = []float64{1, 2, 3, 4, 5, 6}
	inputNode := graph.NewNode(inputData, nil, nil)

	// Forward pass
	output := bn.Forward(inputNode)

	// Проверяем что используются running mean/var
	// Для первого примера, первого признака:
	// normalized = (1 - 1) / sqrt(1 + 1e-5) ≈ 0
	// output = gamma * normalized + beta = 1 * 0 + 0 = 0
	epsilon := 1e-3
	if math.Abs(output.Value.Data[0]) > epsilon {
		t.Errorf("Inference output[0] = %v, expected ~0", output.Value.Data[0])
	}
}

// TestBatchNormParams тестирует возврат параметров
func TestBatchNormParams(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(5, engine)

	params := bn.Params()

	if len(params) != 2 {
		t.Errorf("Expected 2 parameters (gamma, beta), got %d", len(params))
	}

	// Проверяем что gamma инициализирован единицами
	for i := 0; i < 5; i++ {
		if params[0].Value.Data[i] != 1.0 {
			t.Errorf("Gamma[%d] = %v, want 1.0", i, params[0].Value.Data[i])
		}
	}

	// Проверяем что beta инициализирован нулями
	for i := 0; i < 5; i++ {
		if params[1].Value.Data[i] != 0.0 {
			t.Errorf("Beta[%d] = %v, want 0.0", i, params[1].Value.Data[i])
		}
	}
}

// TestBatchNormTrainEvalSwitch тестирует переключение режимов
func TestBatchNormTrainEvalSwitch(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(2, engine)

	// По умолчанию должен быть training mode
	if !bn.training {
		t.Error("BatchNorm should start in training mode")
	}

	// Переключаем в eval
	bn.Eval()
	if bn.training {
		t.Error("BatchNorm should be in eval mode after Eval()")
	}

	// Переключаем обратно в train
	bn.Train()
	if !bn.training {
		t.Error("BatchNorm should be in training mode after Train()")
	}
}

// TestBatchNormSetters тестирует установку параметров
func TestBatchNormSetters(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(3, engine)

	// Проверяем значения по умолчанию
	if bn.eps != 1e-5 {
		t.Errorf("Default epsilon = %v, want 1e-5", bn.eps)
	}
	if bn.momentum != 0.1 {
		t.Errorf("Default momentum = %v, want 0.1", bn.momentum)
	}

	// Устанавливаем новые значения
	bn.SetEpsilon(1e-6)
	bn.SetMomentum(0.2)

	if bn.eps != 1e-6 {
		t.Errorf("After SetEpsilon: eps = %v, want 1e-6", bn.eps)
	}
	if bn.momentum != 0.2 {
		t.Errorf("After SetMomentum: momentum = %v, want 0.2", bn.momentum)
	}
}

// TestBatchNormRunningStats тестирует обновление скользящих статистик
func TestBatchNormRunningStats(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(2, engine)
	bn.SetMomentum(0.1)
	bn.Train()

	// Начальные running mean/var
	initialMean := make([]float64, 2)
	initialVar := make([]float64, 2)
	copy(initialMean, bn.runningMean.Data)
	copy(initialVar, bn.runningVar.Data)

	// Создаем вход со смещенным распределением
	inputData := tensor.Zeros(4, 2)
	inputData.Data = []float64{
		10, 20,
		10, 20,
		10, 20,
		10, 20,
	}
	inputNode := graph.NewNode(inputData, nil, nil)

	// Forward pass
	bn.Forward(inputNode)

	// Проверяем что running mean обновилось в сторону batch mean
	// Batch mean для первого признака = 10
	// running_mean = 0.9 * 0 + 0.1 * 10 = 1.0
	expectedMean := 0.9*initialMean[0] + 0.1*10
	if math.Abs(bn.runningMean.Data[0]-expectedMean) > 1e-6 {
		t.Errorf("Running mean[0] = %v, want %v", bn.runningMean.Data[0], expectedMean)
	}
}

// TestBatchNormNormalization тестирует что нормализация работает корректно
func TestBatchNormNormalization(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(1, engine)
	bn.Train()

	// Создаем вход с известным средним и дисперсией
	// mean = 5, var = 4, std = 2
	inputData := tensor.Zeros(4, 1)
	inputData.Data = []float64{3, 5, 5, 7} // mean=5, var=2
	inputNode := graph.NewNode(inputData, nil, nil)

	// Forward pass
	output := bn.Forward(inputNode)

	// Вычисляем среднее и дисперсию выхода
	mean := 0.0
	for i := 0; i < 4; i++ {
		mean += output.Value.Data[i]
	}
	mean /= 4

	variance := 0.0
	for i := 0; i < 4; i++ {
		diff := output.Value.Data[i] - mean
		variance += diff * diff
	}
	variance /= 4

	// После нормализации среднее должно быть ~0, дисперсия ~1
	if math.Abs(mean) > 1e-5 {
		t.Errorf("Normalized mean = %v, want ~0", mean)
	}
	if math.Abs(variance-1.0) > 1e-4 {
		t.Errorf("Normalized variance = %v, want ~1", variance)
	}
}

// TestBatchNormGammaBeта тестирует влияние gamma и beta
func TestBatchNormGammaBeta(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(1, engine)
	bn.Train()

	// Устанавливаем gamma=2, beta=3
	bn.gamma.Value.Data[0] = 2.0
	bn.beta.Value.Data[0] = 3.0

	// Создаем простой вход
	inputData := tensor.Zeros(2, 1)
	inputData.Data = []float64{0, 2} // mean=1, std=1
	inputNode := graph.NewNode(inputData, nil, nil)

	// Forward pass
	output := bn.Forward(inputNode)

	// После нормализации получим [-1, 1]
	// После gamma*x + beta: 2*[-1, 1] + 3 = [1, 5]
	// Но из-за численных погрешностей проверим приблизительно
	expected0 := 1.0
	expected1 := 5.0

	if math.Abs(output.Value.Data[0]-expected0) > 0.1 {
		t.Errorf("Output[0] = %v, want ~%v", output.Value.Data[0], expected0)
	}
	if math.Abs(output.Value.Data[1]-expected1) > 0.1 {
		t.Errorf("Output[1] = %v, want ~%v", output.Value.Data[1], expected1)
	}
}

// TestBatchNormMultipleFeatures тестирует работу с несколькими признаками
func TestBatchNormMultipleFeatures(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(3, engine)
	bn.Train()

	// Создаем батч с 4 примерами и 3 признаками
	inputData := tensor.Zeros(4, 3)
	inputData.Data = []float64{
		1, 10, 100,
		2, 20, 200,
		3, 30, 300,
		4, 40, 400,
	}
	inputNode := graph.NewNode(inputData, nil, nil)

	// Forward pass
	output := bn.Forward(inputNode)

	// Для каждого признака проверяем нормализацию
	for j := 0; j < 3; j++ {
		mean := 0.0
		for i := 0; i < 4; i++ {
			idx := i*3 + j
			mean += output.Value.Data[idx]
		}
		mean /= 4

		variance := 0.0
		for i := 0; i < 4; i++ {
			idx := i*3 + j
			diff := output.Value.Data[idx] - mean
			variance += diff * diff
		}
		variance /= 4

		if math.Abs(mean) > 1e-5 {
			t.Errorf("Feature %d: mean = %v, want ~0", j, mean)
		}
		if math.Abs(variance-1.0) > 1e-4 {
			t.Errorf("Feature %d: variance = %v, want ~1", j, variance)
		}
	}
}

// TestBatchNormInvalidInput тестирует обработку некорректных входов
func TestBatchNormInvalidInput(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(3, engine)

	// Тест 1: неправильная размерность (1D вместо 2D)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for 1D input")
		}
	}()

	inputData := tensor.Zeros(10)
	inputNode := graph.NewNode(inputData, nil, nil)
	bn.Forward(inputNode)
}

// TestBatchNormInvalidFeatures тестирует несоответствие количества признаков
func TestBatchNormInvalidFeatures(t *testing.T) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(3, engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for mismatched features")
		}
	}()

	// Создаем вход с 5 признаками вместо 3
	inputData := tensor.Zeros(2, 5)
	inputNode := graph.NewNode(inputData, nil, nil)
	bn.Forward(inputNode)
}

// BenchmarkBatchNormForward бенчмарк для forward pass
func BenchmarkBatchNormForward(b *testing.B) {
	engine := autograd.NewEngine()
	bn := NewBatchNorm(128, engine)
	bn.Train()

	inputData := tensor.Randn([]int{32, 128}, 42)
	inputNode := graph.NewNode(inputData, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bn.Forward(inputNode)
	}
}

// ExampleBatchNorm демонстрирует использование BatchNorm
func ExampleBatchNorm() {
	engine := autograd.NewEngine()

	// Создаем BatchNorm слой для 3 признаков
	bn := NewBatchNorm(3, engine)

	// Создаем входной батч [2 примера x 3 признака]
	inputData := tensor.Zeros(2, 3)
	inputData.Data = []float64{1, 2, 3, 4, 5, 6}
	inputNode := graph.NewNode(inputData, nil, nil)

	// Forward pass в режиме обучения
	bn.Train()
	output := bn.Forward(inputNode)

	fmt.Printf("Output shape: %v\n", output.Value.Shape)
	fmt.Printf("Number of parameters: %d\n", len(bn.Params()))

	// Переключаемся в режим inference
	bn.Eval()
	outputInference := bn.Forward(inputNode)
	fmt.Printf("Inference output shape: %v\n", outputInference.Value.Shape)

	// Output:
	// Output shape: [2 3]
	// Number of parameters: 2
	// Inference output shape: [2 3]
}
