package profiling

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

func TestProfilerStartStop(t *testing.T) {
	config := DefaultConfig()
	config.CPUProfilePath = "test_cpu.prof"
	config.MemProfilePath = "test_mem.prof"
	config.EnableTrace = false
	config.Verbose = false

	profiler := NewProfiler(config)

	// Запуск
	if err := profiler.Start(); err != nil {
		t.Fatalf("Failed to start profiler: %v", err)
	}

	// Небольшая работа
	time.Sleep(10 * time.Millisecond)

	// Остановка
	if err := profiler.Stop(); err != nil {
		t.Fatalf("Failed to stop profiler: %v", err)
	}

	// Проверяем создание файлов
	if _, err := os.Stat(config.CPUProfilePath); os.IsNotExist(err) {
		t.Errorf("CPU profile file was not created")
	}
	if _, err := os.Stat(config.MemProfilePath); os.IsNotExist(err) {
		t.Errorf("Memory profile file was not created")
	}

	// Очистка
	os.Remove(config.CPUProfilePath)
	os.Remove(config.MemProfilePath)
}

func TestOperationMetrics(t *testing.T) {
	metrics := NewOperationMetrics()

	// Записываем несколько операций
	metrics.RecordOperation("test.op1", 100*time.Millisecond, 1000, 1000)
	metrics.RecordOperation("test.op1", 150*time.Millisecond, 1000, 1000)
	metrics.RecordOperation("test.op2", 50*time.Millisecond, 500, 500)

	// Проверяем метрику op1
	metric1 := metrics.GetMetric("test.op1")
	if metric1 == nil {
		t.Fatal("Metric test.op1 not found")
	}
	if metric1.Count != 2 {
		t.Errorf("Expected count 2, got %d", metric1.Count)
	}
	if metric1.MinDuration != 100*time.Millisecond {
		t.Errorf("Expected min duration 100ms, got %v", metric1.MinDuration)
	}
	if metric1.MaxDuration != 150*time.Millisecond {
		t.Errorf("Expected max duration 150ms, got %v", metric1.MaxDuration)
	}

	// Проверяем метрику op2
	metric2 := metrics.GetMetric("test.op2")
	if metric2 == nil {
		t.Fatal("Metric test.op2 not found")
	}
	if metric2.Count != 1 {
		t.Errorf("Expected count 1, got %d", metric2.Count)
	}
}

func TestOpTimer(t *testing.T) {
	metrics := NewOperationMetrics()

	// Используем таймер
	timer := metrics.StartOperation("test.timed")
	timer.SetInputSize(100)
	timer.SetOutputSize(200)

	// Симулируем работу
	time.Sleep(10 * time.Millisecond)

	timer.Stop()

	// Проверяем метрику
	metric := metrics.GetMetric("test.timed")
	if metric == nil {
		t.Fatal("Timed metric not found")
	}
	if metric.Count != 1 {
		t.Errorf("Expected count 1, got %d", metric.Count)
	}
	if len(metric.InputSizes) != 1 || metric.InputSizes[0] != 100 {
		t.Errorf("Expected input size 100, got %v", metric.InputSizes)
	}
	if len(metric.OutputSizes) != 1 || metric.OutputSizes[0] != 200 {
		t.Errorf("Expected output size 200, got %v", metric.OutputSizes)
	}
}

func TestContextProfiling(t *testing.T) {
	config := DefaultConfig()
	config.EnableCPUProfile = false
	config.EnableMemProfile = false
	config.EnableOperationMetrics = true

	profiler := NewProfiler(config)

	if err := profiler.Start(); err != nil {
		t.Fatalf("Failed to start profiler: %v", err)
	}

	ctx := WithProfiler(context.Background(), profiler)

	// Проверяем извлечение профилировщика
	retrieved := FromContext(ctx)
	if retrieved != profiler {
		t.Error("Failed to retrieve profiler from context")
	}

	// Используем TraceOperation
	err := TraceOperation(ctx, "test.operation", func() error {
		time.Sleep(5 * time.Millisecond)
		return nil
	})
	if err != nil {
		t.Errorf("TraceOperation failed: %v", err)
	}

	if err := profiler.Stop(); err != nil {
		t.Fatalf("Failed to stop profiler: %v", err)
	}

	// Проверяем что операция была записана
	metric := profiler.OperationMetrics.GetMetric("test.operation")
	if metric == nil {
		t.Error("Operation was not recorded")
	}
}

func TestStatistics(t *testing.T) {
	stats := NewStatistics()

	// Тестируем счетчики
	stats.IncrementCounter("test.counter")
	stats.IncrementCounter("test.counter")
	stats.AddToCounter("test.counter", 5)

	count := stats.GetCounter("test.counter")
	if count != 7 { // 1 + 1 + 5
		t.Errorf("Expected counter value 7, got %d", count)
	}
}

func TestOperationMetricAverages(t *testing.T) {
	metric := &OperationMetric{
		Name:         "test",
		Count:        3,
		TotalDuration: 300 * time.Millisecond,
		InputSizes:   []int64{100, 200, 300},
		OutputSizes:  []int64{50, 100, 150},
	}

	// Проверяем среднее время
	avgDuration := metric.AvgDuration()
	if avgDuration != 100*time.Millisecond {
		t.Errorf("Expected avg duration 100ms, got %v", avgDuration)
	}

	// Проверяем средний размер входа
	avgInput := metric.AvgInputSize()
	if avgInput != 200 {
		t.Errorf("Expected avg input size 200, got %d", avgInput)
	}

	// Проверяем средний размер выхода
	avgOutput := metric.AvgOutputSize()
	if avgOutput != 100 {
		t.Errorf("Expected avg output size 100, got %d", avgOutput)
	}
}

func BenchmarkProfilerOverhead(b *testing.B) {
	config := DefaultConfig()
	config.EnableCPUProfile = false
	config.EnableMemProfile = false
	config.EnableTrace = false
	config.EnableOperationMetrics = true

	profiler := NewProfiler(config)
	profiler.Start()
	defer profiler.Stop()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		timer := profiler.OperationMetrics.StartOperation("bench.operation")
		// Симулируем легкую операцию
		_ = math.Sin(float64(i))
		timer.Stop()
	}
}

func BenchmarkWithoutProfiling(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = math.Sin(float64(i))
	}
}

func BenchmarkContextProfiling(b *testing.B) {
	config := DefaultConfig()
	config.EnableCPUProfile = false
	config.EnableMemProfile = false
	config.EnableOperationMetrics = true

	profiler := NewProfiler(config)
	profiler.Start()
	defer profiler.Stop()

	ctx := WithProfiler(context.Background(), profiler)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		TraceOperation(ctx, "bench.context", func() error {
			_ = math.Sin(float64(i))
			return nil
		})
	}
}
