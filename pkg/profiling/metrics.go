package profiling

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// OperationMetrics собирает метрики по операциям
type OperationMetrics struct {
	mu sync.RWMutex

	// Карта операций: имя -> метрика
	operations map[string]*OperationMetric
}

// OperationMetric содержит метрики для одной операции
type OperationMetric struct {
	Name string

	// Количество вызовов
	Count int64

	// Общее время выполнения
	TotalDuration time.Duration

	// Минимальное время
	MinDuration time.Duration

	// Максимальное время
	MaxDuration time.Duration

	// Размеры входных данных (для анализа)
	InputSizes []int64

	// Размеры выходных данных
	OutputSizes []int64
}

// NewOperationMetrics создает новый экземпляр метрик операций
func NewOperationMetrics() *OperationMetrics {
	return &OperationMetrics{
		operations: make(map[string]*OperationMetric),
	}
}

// RecordOperation записывает выполнение операции
func (om *OperationMetrics) RecordOperation(name string, duration time.Duration, inputSize, outputSize int64) {
	om.mu.Lock()
	defer om.mu.Unlock()

	metric, exists := om.operations[name]
	if !exists {
		metric = &OperationMetric{
			Name:        name,
			MinDuration: duration,
			InputSizes:  make([]int64, 0),
			OutputSizes: make([]int64, 0),
		}
		om.operations[name] = metric
	}

	metric.Count++
	metric.TotalDuration += duration

	if duration < metric.MinDuration {
		metric.MinDuration = duration
	}
	if duration > metric.MaxDuration {
		metric.MaxDuration = duration
	}

	if inputSize > 0 {
		metric.InputSizes = append(metric.InputSizes, inputSize)
	}
	if outputSize > 0 {
		metric.OutputSizes = append(metric.OutputSizes, outputSize)
	}
}

// GetMetric возвращает метрику для конкретной операции
func (om *OperationMetrics) GetMetric(name string) *OperationMetric {
	om.mu.RLock()
	defer om.mu.RUnlock()

	return om.operations[name]
}

// GetAllMetrics возвращает все метрики
func (om *OperationMetrics) GetAllMetrics() []*OperationMetric {
	om.mu.RLock()
	defer om.mu.RUnlock()

	metrics := make([]*OperationMetric, 0, len(om.operations))
	for _, m := range om.operations {
		metrics = append(metrics, m)
	}

	// Сортировка по общему времени (от большего к меньшему)
	sort.Slice(metrics, func(i, j int) bool {
		return metrics[i].TotalDuration > metrics[j].TotalDuration
	})

	return metrics
}

// PrintReport выводит отчет по всем операциям
func (om *OperationMetrics) PrintReport() {
	metrics := om.GetAllMetrics()

	if len(metrics) == 0 {
		fmt.Println("No operations recorded")
		return
	}

	fmt.Printf("%-30s %10s %15s %15s %15s %15s\n",
		"Operation", "Count", "Total Time", "Avg Time", "Min Time", "Max Time")
	fmt.Println("---------------------------------------------------------------------------------------------------")

	for _, m := range metrics {
		avgDuration := time.Duration(0)
		if m.Count > 0 {
			avgDuration = m.TotalDuration / time.Duration(m.Count)
		}

		fmt.Printf("%-30s %10d %15v %15v %15v %15v\n",
			m.Name, m.Count, m.TotalDuration, avgDuration, m.MinDuration, m.MaxDuration)
	}
}

// AvgDuration возвращает среднее время выполнения операции
func (m *OperationMetric) AvgDuration() time.Duration {
	if m.Count == 0 {
		return 0
	}
	return m.TotalDuration / time.Duration(m.Count)
}

// AvgInputSize возвращает средний размер входных данных
func (m *OperationMetric) AvgInputSize() int64 {
	if len(m.InputSizes) == 0 {
		return 0
	}
	var sum int64
	for _, size := range m.InputSizes {
		sum += size
	}
	return sum / int64(len(m.InputSizes))
}

// AvgOutputSize возвращает средний размер выходных данных
func (m *OperationMetric) AvgOutputSize() int64 {
	if len(m.OutputSizes) == 0 {
		return 0
	}
	var sum int64
	for _, size := range m.OutputSizes {
		sum += size
	}
	return sum / int64(len(m.OutputSizes))
}

// Statistics содержит общую статистику профилирования
type Statistics struct {
	mu sync.RWMutex

	// Общая продолжительность
	TotalDuration time.Duration

	// Количество аллокаций памяти
	TotalAllocations uint64

	// Общий размер выделенной памяти
	TotalMemoryAllocated uint64

	// Количество GC циклов
	NumGC uint32

	// Пользовательские счетчики
	Counters map[string]int64
}

// NewStatistics создает новый экземпляр статистики
func NewStatistics() *Statistics {
	return &Statistics{
		Counters: make(map[string]int64),
	}
}

// IncrementCounter увеличивает счетчик
func (s *Statistics) IncrementCounter(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.Counters[name]++
}

// AddToCounter добавляет значение к счетчику
func (s *Statistics) AddToCounter(name string, value int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.Counters[name] += value
}

// GetCounter возвращает значение счетчика
func (s *Statistics) GetCounter(name string) int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.Counters[name]
}

// OpTimer используется для измерения времени выполнения операции
type OpTimer struct {
	name       string
	startTime  time.Time
	metrics    *OperationMetrics
	inputSize  int64
	outputSize int64
}

// StartOperation начинает измерение операции
func (om *OperationMetrics) StartOperation(name string) *OpTimer {
	return &OpTimer{
		name:      name,
		startTime: time.Now(),
		metrics:   om,
	}
}

// SetInputSize устанавливает размер входных данных
func (ot *OpTimer) SetInputSize(size int64) {
	ot.inputSize = size
}

// SetOutputSize устанавливает размер выходных данных
func (ot *OpTimer) SetOutputSize(size int64) {
	ot.outputSize = size
}

// Stop останавливает измерение и записывает метрику
func (ot *OpTimer) Stop() {
	duration := time.Since(ot.startTime)
	ot.metrics.RecordOperation(ot.name, duration, ot.inputSize, ot.outputSize)
}
