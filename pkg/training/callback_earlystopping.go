package training

import (
	"fmt"
	"math"
)

// EarlyStopping останавливает обучение досрочно если метрика не улучшается.
// Полезен для предотвращения переобучения и экономии времени.
//
// Логика работы:
//  1. Отслеживает выбранную метрику каждую эпоху
//  2. Если метрика не улучшается patience эпох подряд -> устанавливает ctx.StopTraining = true
//  3. Улучшение определяется с учетом minDelta (минимальное изменение)
//
// Пример:
//   Если patience=3 и метрика не улучшается 3 эпохи подряд, обучение останавливается.
type EarlyStopping struct {
	BaseCallback

	monitor  string  // Метрика для мониторинга ("loss", "val_loss", etc.)
	patience int     // Количество эпох без улучшения до остановки
	mode     string  // "min" для минимизации, "max" для максимизации
	minDelta float64 // Минимальное изменение для считывания улучшением
	verbose  bool    // Выводить сообщения

	// Внутреннее состояние
	bestValue float64 // Лучшее значение метрики
	wait      int     // Счетчик эпох без улучшения
	stopped   bool    // Флаг: обучение было остановлено
	bestEpoch int     // Эпоха с лучшим значением
}

// NewEarlyStopping создает новый колбэк для досрочной остановки обучения.
//
// Параметры:
//   monitor - метрика для мониторинга ("loss", "val_loss", "accuracy" и т.д.)
//   patience - количество эпох без улучшения до остановки (например, 5)
//   mode - "min" для минимизации метрики (loss), "max" для максимизации (accuracy)
//   minDelta - минимальное изменение для считывания улучшением (например, 0.001)
//              Для mode="min": улучшение если new_value < best_value - minDelta
//              Для mode="max": улучшение если new_value > best_value + minDelta
//   verbose - true: выводить сообщения о состоянии
//
// Примеры:
//   // Остановить если loss не улучшается 5 эпох
//   NewEarlyStopping("loss", 5, "min", 0.001, true)
//
//   // Остановить если accuracy не улучшается 10 эпох
//   NewEarlyStopping("accuracy", 10, "max", 0.0001, true)
func NewEarlyStopping(monitor string, patience int, mode string, minDelta float64, verbose bool) *EarlyStopping {
	initialValue := 0.0
	if mode == "min" {
		initialValue = math.Inf(1) // +Inf для минимизации
	} else {
		initialValue = math.Inf(-1) // -Inf для максимизации
	}

	return &EarlyStopping{
		monitor:   monitor,
		patience:  patience,
		mode:      mode,
		minDelta:  minDelta,
		verbose:   verbose,
		bestValue: initialValue,
		wait:      0,
		stopped:   false,
		bestEpoch: -1,
	}
}

// OnEpochEnd проверяет улучшение метрики и устанавливает флаг остановки при необходимости.
func (es *EarlyStopping) OnEpochEnd(ctx *TrainingContext) error {
	// Получаем текущее значение монitorируемой метрики
	currentValue, exists := ctx.Metrics[es.monitor]
	if !exists {
		if es.verbose {
			fmt.Printf("EarlyStopping: metric '%s' not found in context.Metrics\n", es.monitor)
		}
		return nil
	}

	// Проверяем улучшение с учетом minDelta
	improved := false

	if es.mode == "min" {
		// Для минимизации: улучшение если new < best - minDelta
		if currentValue < (es.bestValue - es.minDelta) {
			improved = true
		}
	} else if es.mode == "max" {
		// Для максимизации: улучшение если new > best + minDelta
		if currentValue > (es.bestValue + es.minDelta) {
			improved = true
		}
	}

	if improved {
		// Метрика улучшилась
		es.bestValue = currentValue
		es.bestEpoch = ctx.Epoch
		es.wait = 0 // Сбрасываем счетчик ожидания

		if es.verbose {
			fmt.Printf("EarlyStopping: %s improved to %.6f\n", es.monitor, currentValue)
		}
	} else {
		// Метрика не улучшилась
		es.wait++

		if es.verbose {
			fmt.Printf("EarlyStopping: %s did not improve from %.6f (patience: %d/%d)\n",
				es.monitor, es.bestValue, es.wait, es.patience)
		}

		// Проверяем достигнут ли лимит терпения
		if es.wait >= es.patience {
			es.stopped = true
			ctx.StopTraining = true // Устанавливаем флаг остановки

			if es.verbose {
				fmt.Printf("EarlyStopping: stopping training at epoch %d (best %s: %.6f at epoch %d)\n",
					ctx.Epoch+1, es.monitor, es.bestValue, es.bestEpoch+1)
			}
		}
	}

	return nil
}

// IsStopped возвращает true если обучение было остановлено этим колбэком.
func (es *EarlyStopping) IsStopped() bool {
	return es.stopped
}

// GetBestEpoch возвращает номер эпохи с лучшим значением метрики.
func (es *EarlyStopping) GetBestEpoch() int {
	return es.bestEpoch
}

// GetBestValue возвращает лучшее значение монitorируемой метрики.
func (es *EarlyStopping) GetBestValue() float64 {
	return es.bestValue
}

// GetWaitCount возвращает текущее количество эпох без улучшения.
func (es *EarlyStopping) GetWaitCount() int {
	return es.wait
}
