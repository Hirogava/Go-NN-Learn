package optimizers_test

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
)

// TestStepLR проверяет работу StepLR scheduler
func TestStepLR(t *testing.T) {
	initialLR := 0.1
	gamma := 0.5
	stepSize := 3

	scheduler := optimizers.NewStepLR(initialLR, gamma, stepSize)

	// Эпоха 1: floor(1/3) = 0, LR = 0.1 * 0.5^0 = 0.1
	lr := scheduler.Step()
	if math.Abs(lr-initialLR) > 1e-10 {
		t.Fatalf("StepLR epoch 1: expected %v, got %v", initialLR, lr)
	}

	// Эпоха 2: floor(2/3) = 0, LR = 0.1 * 0.5^0 = 0.1
	lr = scheduler.Step()
	if math.Abs(lr-initialLR) > 1e-10 {
		t.Fatalf("StepLR epoch 2: expected %v, got %v", initialLR, lr)
	}

	// Эпоха 3: floor(3/3) = 1, LR = 0.1 * 0.5^1 = 0.05
	lr = scheduler.Step()
	expected := initialLR * gamma
	if math.Abs(lr-expected) > 1e-10 {
		t.Fatalf("StepLR epoch 3: expected %v, got %v", expected, lr)
	}

	// Эпоха 4: floor(4/3) = 1, LR = 0.1 * 0.5^1 = 0.05
	lr = scheduler.Step()
	if math.Abs(lr-expected) > 1e-10 {
		t.Fatalf("StepLR epoch 4: expected %v, got %v", expected, lr)
	}

	// Эпоха 5: floor(5/3) = 1, LR = 0.1 * 0.5^1 = 0.05
	lr = scheduler.Step()
	if math.Abs(lr-expected) > 1e-10 {
		t.Fatalf("StepLR epoch 5: expected %v, got %v", expected, lr)
	}

	// Эпоха 6: floor(6/3) = 2, LR = 0.1 * 0.5^2 = 0.025
	lr = scheduler.Step()
	expected = initialLR * gamma * gamma
	if math.Abs(lr-expected) > 1e-10 {
		t.Fatalf("StepLR epoch 6: expected %v, got %v", expected, lr)
	}
}

// TestStepLRGetLastLR проверяет метод GetLastLR
func TestStepLRGetLastLR(t *testing.T) {
	scheduler := optimizers.NewStepLR(0.1, 0.5, 2)

	if scheduler.GetLastLR() != 0.1 {
		t.Fatalf("GetLastLR: expected 0.1, got %v", scheduler.GetLastLR())
	}

	scheduler.Step()
	if scheduler.GetLastLR() != 0.1 {
		t.Fatalf("GetLastLR after step: expected 0.1, got %v", scheduler.GetLastLR())
	}

	scheduler.Step()
	if scheduler.GetLastLR() != 0.05 {
		t.Fatalf("GetLastLR after step_size: expected 0.05, got %v", scheduler.GetLastLR())
	}
}

// TestExponentialLR проверяет работу ExponentialLR scheduler
func TestExponentialLR(t *testing.T) {
	initialLR := 0.1
	gamma := 0.9

	scheduler := optimizers.NewExponentialLR(initialLR, gamma)

	// Проверяем экспоненциальное уменьшение
	for i := 0; i < 5; i++ {
		lr := scheduler.Step()
		expected := initialLR * math.Pow(gamma, float64(i+1))
		if math.Abs(lr-expected) > 1e-10 {
			t.Fatalf("ExponentialLR epoch %d: expected %v, got %v", i+1, expected, lr)
		}
	}
}

// TestExponentialLRGetLastLR проверяет метод GetLastLR для ExponentialLR
func TestExponentialLRGetLastLR(t *testing.T) {
	scheduler := optimizers.NewExponentialLR(0.1, 0.9)

	if scheduler.GetLastLR() != 0.1 {
		t.Fatalf("GetLastLR: expected 0.1, got %v", scheduler.GetLastLR())
	}

	scheduler.Step()
	expected := 0.1 * 0.9
	if math.Abs(scheduler.GetLastLR()-expected) > 1e-10 {
		t.Fatalf("GetLastLR after step: expected %v, got %v", expected, scheduler.GetLastLR())
	}
}

// TestCosineAnnealingLR проверяет работу CosineAnnealingLR scheduler
func TestCosineAnnealingLR(t *testing.T) {
	initialLR := 0.1
	TMax := 10
	etaMin := 0.01

	scheduler := optimizers.NewCosineAnnealingLR(initialLR, TMax, etaMin)

	// Проверяем начальное значение
	if math.Abs(scheduler.GetLastLR()-initialLR) > 1e-10 {
		t.Fatalf("CosineAnnealingLR initial: expected %v, got %v", initialLR, scheduler.GetLastLR())
	}

	// Проверяем значение в середине (должно быть близко к etaMin)
	for i := 0; i < TMax/2; i++ {
		scheduler.Step()
	}
	midLR := scheduler.GetLastLR()
	if midLR < etaMin || midLR > initialLR {
		t.Fatalf("CosineAnnealingLR mid: expected between %v and %v, got %v", etaMin, initialLR, midLR)
	}

	// Проверяем значение в конце (должно быть близко к etaMin)
	for i := 0; i < TMax/2; i++ {
		scheduler.Step()
	}
	finalLR := scheduler.GetLastLR()
	if math.Abs(finalLR-etaMin) > 0.01 {
		t.Fatalf("CosineAnnealingLR final: expected close to %v, got %v", etaMin, finalLR)
	}
}

// TestCosineAnnealingLRGetLastLR проверяет метод GetLastLR для CosineAnnealingLR
func TestCosineAnnealingLRGetLastLR(t *testing.T) {
	scheduler := optimizers.NewCosineAnnealingLR(0.1, 10, 0.01)

	if scheduler.GetLastLR() != 0.1 {
		t.Fatalf("GetLastLR: expected 0.1, got %v", scheduler.GetLastLR())
	}

	scheduler.Step()
	lr := scheduler.GetLastLR()
	if lr <= 0 || lr > 0.1 {
		t.Fatalf("GetLastLR after step: expected between 0 and 0.1, got %v", lr)
	}
}

// TestOneCycleLR проверяет работу OneCycleLR scheduler
func TestOneCycleLR(t *testing.T) {
	minLR := 0.01
	maxLR := 0.1
	maxEpochs := 10
	finalLR := 0.001

	scheduler := optimizers.NewOneCycleLR(minLR, maxLR, maxEpochs, finalLR)

	// Проверяем начальное значение
	if math.Abs(scheduler.GetLastLR()-minLR) > 1e-10 {
		t.Fatalf("OneCycleLR initial: expected %v, got %v", minLR, scheduler.GetLastLR())
	}

	// Проверяем фазу увеличения
	lrs := make([]float64, 0, maxEpochs/2+1)
	lrs = append(lrs, scheduler.GetLastLR())
	for i := 0; i < maxEpochs/2; i++ {
		lr := scheduler.Step()
		lrs = append(lrs, lr)
		// LR должен увеличиваться
		if i > 0 && lr <= lrs[i] {
			t.Fatalf("OneCycleLR increasing phase: LR should increase, got %v -> %v", lrs[i], lr)
		}
	}

	// Проверяем максимальное значение
	maxLRReached := scheduler.GetLastLR()
	if math.Abs(maxLRReached-maxLR) > 0.01 {
		t.Fatalf("OneCycleLR max: expected close to %v, got %v", maxLR, maxLRReached)
	}

	// Проверяем фазу уменьшения
	prevLR := scheduler.GetLastLR()
	for i := 0; i < maxEpochs/2; i++ {
		lr := scheduler.Step()
		// LR должен уменьшаться
		if lr >= prevLR {
			t.Fatalf("OneCycleLR decreasing phase epoch %d: LR should decrease, got %v -> %v", i+1, prevLR, lr)
		}
		prevLR = lr
	}

	// Проверяем финальное значение
	finalLRReached := scheduler.GetLastLR()
	if math.Abs(finalLRReached-finalLR) > 0.01 {
		t.Fatalf("OneCycleLR final: expected close to %v, got %v", finalLR, finalLRReached)
	}

	// Проверяем, что после maxEpochs LR остается финальным
	scheduler.Step()
	if math.Abs(scheduler.GetLastLR()-finalLR) > 0.01 {
		t.Fatalf("OneCycleLR after maxEpochs: expected %v, got %v", finalLR, scheduler.GetLastLR())
	}
}

// TestOneCycleLRDefaultFinal проверяет, что finalLR по умолчанию равен minLR
func TestOneCycleLRDefaultFinal(t *testing.T) {
	minLR := 0.01
	maxLR := 0.1
	maxEpochs := 10

	scheduler := optimizers.NewOneCycleLR(minLR, maxLR, maxEpochs, 0)

	// Доводим до конца
	for i := 0; i < maxEpochs+1; i++ {
		scheduler.Step()
	}

	// Финальный LR должен быть равен minLR
	if math.Abs(scheduler.GetLastLR()-minLR) > 0.01 {
		t.Fatalf("OneCycleLR default final: expected %v, got %v", minLR, scheduler.GetLastLR())
	}
}

// TestOneCycleLRGetLastLR проверяет метод GetLastLR для OneCycleLR
func TestOneCycleLRGetLastLR(t *testing.T) {
	scheduler := optimizers.NewOneCycleLR(0.01, 0.1, 10, 0.001)

	if scheduler.GetLastLR() != 0.01 {
		t.Fatalf("GetLastLR: expected 0.01, got %v", scheduler.GetLastLR())
	}

	scheduler.Step()
	lr := scheduler.GetLastLR()
	if lr <= 0.01 || lr > 0.1 {
		t.Fatalf("GetLastLR after step: expected between 0.01 and 0.1, got %v", lr)
	}
}

// TestSchedulerWithOptimizer проверяет интеграцию scheduler с оптимизатором
func TestSchedulerWithOptimizer(t *testing.T) {
	// Создаем оптимизатор (возвращает интерфейс Optimizer)
	sgd := optimizers.NewSGD(0.1)

	// Создаем scheduler
	scheduler := optimizers.NewStepLR(0.1, 0.5, 2)

	// Проверяем, что оптимизатор реализует интерфейс LearningRateAdjustable
	adjustable := optimizers.LearningRateAdjustable(sgd)

	initialLR := sgd.LearningRate
	if math.Abs(initialLR-0.1) > 1e-10 {
		t.Fatalf("Initial LR: expected 0.1, got %v", initialLR)
	}

	// Выполняем несколько шагов scheduler
	for i := 0; i < 3; i++ {
		newLR := scheduler.Step()
		adjustable.SetLearningRate(newLR)
		currentLR := sgd.LearningRate
		if math.Abs(currentLR-newLR) > 1e-10 {
			t.Fatalf("LR after step %d: expected %v, got %v", i+1, newLR, currentLR)
		}
	}
}

// TestAllOptimizersImplementSetLearningRate проверяет, что все оптимизаторы реализуют SetLearningRate
func TestAllOptimizersImplementSetLearningRate(t *testing.T) {
	// Проверяем SGD
	sgd := optimizers.NewSGD(0.1)
	adjustable := optimizers.LearningRateAdjustable(sgd)
	adjustable.SetLearningRate(0.05)

	if sgd.LearningRate != 0.05 {
		t.Fatalf("SGD: SetLearningRate failed, got %v", sgd.LearningRate)
	}

	// Проверяем Momentum
	momentum := optimizers.NewMomentum(0.1, 0.9)
	adjustable = optimizers.LearningRateAdjustable(momentum)
	adjustable.SetLearningRate(0.05)

	if momentum.LearningRate != 0.05 {
		t.Fatalf("Momentum: SetLearningRate failed, got %v", momentum.LearningRate)
	}

	// Проверяем Adam
	adam := optimizers.NewAdam(0.1, 0.9, 0.999, 1e-8)
	adjustable = optimizers.LearningRateAdjustable(adam)
	adjustable.SetLearningRate(0.05)

	if adam.LearningRate != 0.05 {
		t.Fatalf("Adam: SetLearningRate failed, got %v", adam.LearningRate)
	}

	// Проверяем RMSProp
	rmsprop := optimizers.NewRMSProp(0.1, 0.9, 1e-8)
	adjustable = optimizers.LearningRateAdjustable(rmsprop)
	adjustable.SetLearningRate(0.05)

	if rmsprop.LearningRate != 0.05 {
		t.Fatalf("RMSProp: SetLearningRate failed, got %v", rmsprop.LearningRate)
	}
}
