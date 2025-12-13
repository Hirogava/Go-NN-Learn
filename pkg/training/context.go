package training

import (
	"fmt"
	"math"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// TrainingContext содержит информацию о текущем состоянии обучения.
// Передается во все колбэки для получения доступа к метрикам, модели и управлению циклом.
type TrainingContext struct {
	// Информация о текущем состоянии
	Epoch      int // Текущая эпоха (0-based)
	NumEpochs  int // Общее количество эпох
	Batch      int // Текущий батч в эпохе (0-based)
	NumBatches int // Количество батчей в эпохе

	// Метрики текущего состояния
	Metrics map[string]float64 // Текущие метрики: {"loss": 0.5, "accuracy": 0.9}
	History *MetricsHistory    // История метрик по эпохам

	// Модель и параметры
	Model  layers.Module   // Обучаемая модель
	Params []*graph.Node   // Параметры модели (кэш от Model.Params())

	// Флаги управления
	StopTraining bool // Установить в true для досрочной остановки обучения
}

// NewTrainingContext создает новый контекст обучения.
func NewTrainingContext(model layers.Module, numEpochs int) *TrainingContext {
	return &TrainingContext{
		NumEpochs: numEpochs,
		Model:     model,
		Params:    model.Params(),
		Metrics:   make(map[string]float64),
		History:   NewMetricsHistory(),
	}
}

// MetricsHistory хранит историю метрик по эпохам.
// Thread-safe для использования в многопоточных сценариях.
type MetricsHistory struct {
	mu      sync.RWMutex
	Epochs  []int                    // Список эпох: [0, 1, 2, ...]
	Metrics map[string][]float64     // Метрики по эпохам: {"loss": [0.5, 0.4, ...], "acc": [0.8, 0.85, ...]}
}

// NewMetricsHistory создает новую историю метрик.
func NewMetricsHistory() *MetricsHistory {
	return &MetricsHistory{
		Epochs:  make([]int, 0),
		Metrics: make(map[string][]float64),
	}
}

// Append добавляет метрики для указанной эпохи.
// Если метрика встречается впервые, создается новый список для нее.
func (h *MetricsHistory) Append(epoch int, metrics map[string]float64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.Epochs = append(h.Epochs, epoch)

	for name, value := range metrics {
		if _, exists := h.Metrics[name]; !exists {
			h.Metrics[name] = make([]float64, 0)
		}
		h.Metrics[name] = append(h.Metrics[name], value)
	}
}

// Get возвращает все значения указанной метрики.
// Возвращает nil если метрика не найдена.
func (h *MetricsHistory) Get(metricName string) []float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	values, exists := h.Metrics[metricName]
	if !exists {
		return nil
	}

	// Возвращаем копию для безопасности
	result := make([]float64, len(values))
	copy(result, values)
	return result
}

// GetLast возвращает последнее значение указанной метрики.
// Возвращает 0 если метрика не найдена или пуста.
func (h *MetricsHistory) GetLast(metricName string) float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	values, exists := h.Metrics[metricName]
	if !exists || len(values) == 0 {
		return 0
	}

	return values[len(values)-1]
}

// Best возвращает лучшее значение метрики и эпоху, когда оно было достигнуто.
// mode: "min" для минимизации (например, loss), "max" для максимизации (например, accuracy).
// Возвращает (epoch, value) или (-1, 0) если метрика не найдена.
func (h *MetricsHistory) Best(metricName string, mode string) (int, float64) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	values, exists := h.Metrics[metricName]
	if !exists || len(values) == 0 {
		return -1, 0
	}

	bestEpoch := 0
	bestValue := values[0]

	for i := 1; i < len(values); i++ {
		if mode == "min" {
			if values[i] < bestValue {
				bestValue = values[i]
				bestEpoch = i
			}
		} else if mode == "max" {
			if values[i] > bestValue {
				bestValue = values[i]
				bestEpoch = i
			}
		}
	}

	// Возвращаем реальную эпоху из списка
	if bestEpoch < len(h.Epochs) {
		return h.Epochs[bestEpoch], bestValue
	}

	return bestEpoch, bestValue
}

// Len возвращает количество записанных эпох.
func (h *MetricsHistory) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.Epochs)
}

// HasMetric проверяет существование метрики в истории.
func (h *MetricsHistory) HasMetric(metricName string) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	_, exists := h.Metrics[metricName]
	return exists
}

// MetricNames возвращает список всех метрик в истории.
func (h *MetricsHistory) MetricNames() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	names := make([]string, 0, len(h.Metrics))
	for name := range h.Metrics {
		names = append(names, name)
	}
	return names
}

// Clear очищает всю историю метрик.
func (h *MetricsHistory) Clear() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.Epochs = make([]int, 0)
	h.Metrics = make(map[string][]float64)
}

// String возвращает строковое представление последних метрик.
func (h *MetricsHistory) String() string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.Epochs) == 0 {
		return "MetricsHistory{empty}"
	}

	lastEpoch := h.Epochs[len(h.Epochs)-1]
	result := fmt.Sprintf("MetricsHistory{epoch: %d", lastEpoch)

	for name, values := range h.Metrics {
		if len(values) > 0 {
			result += fmt.Sprintf(", %s: %.4f", name, values[len(values)-1])
		}
	}

	result += "}"
	return result
}

// IsImproved проверяет, улучшилось ли значение метрики по сравнению с лучшим.
// mode: "min" или "max"
// minDelta: минимальное изменение для считывания улучшением
// Возвращает true если текущее значение лучше предыдущего лучшего на minDelta или более.
func (h *MetricsHistory) IsImproved(metricName string, mode string, minDelta float64) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	values, exists := h.Metrics[metricName]
	if !exists || len(values) < 2 {
		return true // Первое значение всегда считается улучшением
	}

	currentValue := values[len(values)-1]

	// Находим лучшее значение среди предыдущих
	var bestPrevious float64
	if mode == "min" {
		bestPrevious = math.Inf(1)
		for i := 0; i < len(values)-1; i++ {
			if values[i] < bestPrevious {
				bestPrevious = values[i]
			}
		}
		// Для минимизации: улучшение если current < best - minDelta
		return currentValue < (bestPrevious - minDelta)
	} else {
		bestPrevious = math.Inf(-1)
		for i := 0; i < len(values)-1; i++ {
			if values[i] > bestPrevious {
				bestPrevious = values[i]
			}
		}
		// Для максимизации: улучшение если current > best + minDelta
		return currentValue > (bestPrevious + minDelta)
	}
}
