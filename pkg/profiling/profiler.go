package profiling

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"time"
)

// Profiler представляет центральный профилировщик для end-to-end мониторинга
type Profiler struct {
	mu sync.RWMutex

	// Конфигурация
	Config *Config

	// CPU профилирование
	cpuFile *os.File
	cpuActive bool

	// Memory профилирование
	memFile *os.File

	// Trace профилирование
	traceFile *os.File
	traceActive bool

	// Метрики операций
	OperationMetrics *OperationMetrics

	// Временные метки
	startTime time.Time
	endTime   time.Time

	// Статистика
	Stats *Statistics
}

// Config содержит конфигурацию профилировщика
type Config struct {
	// Включить CPU профилирование
	EnableCPUProfile bool
	// Путь к файлу CPU профиля
	CPUProfilePath string

	// Включить Memory профилирование
	EnableMemProfile bool
	// Путь к файлу Memory профиля
	MemProfilePath string

	// Включить трассировку
	EnableTrace bool
	// Путь к файлу трассировки
	TracePath string

	// Включить метрики операций
	EnableOperationMetrics bool

	// Включить подробное логирование
	Verbose bool

	// Интервал сбора статистики (для периодического профилирования)
	SamplingInterval time.Duration
}

// DefaultConfig возвращает конфигурацию по умолчанию
func DefaultConfig() *Config {
	return &Config{
		EnableCPUProfile:       true,
		CPUProfilePath:         "cpu.prof",
		EnableMemProfile:       true,
		MemProfilePath:         "mem.prof",
		EnableTrace:            false,
		TracePath:              "trace.out",
		EnableOperationMetrics: true,
		Verbose:                false,
		SamplingInterval:       100 * time.Millisecond,
	}
}

// NewProfiler создает новый профилировщик с заданной конфигурацией
func NewProfiler(config *Config) *Profiler {
	if config == nil {
		config = DefaultConfig()
	}

	return &Profiler{
		Config:           config,
		OperationMetrics: NewOperationMetrics(),
		Stats:            NewStatistics(),
	}
}

// Start запускает профилирование
func (p *Profiler) Start() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.startTime = time.Now()

	if p.Config.Verbose {
		fmt.Println("Starting profiler...")
	}

	// Запуск CPU профилирования
	if p.Config.EnableCPUProfile {
		if err := p.startCPUProfile(); err != nil {
			return fmt.Errorf("failed to start CPU profile: %w", err)
		}
		if p.Config.Verbose {
			fmt.Printf("CPU profiling started: %s\n", p.Config.CPUProfilePath)
		}
	}

	// Запуск трассировки
	if p.Config.EnableTrace {
		if err := p.startTrace(); err != nil {
			return fmt.Errorf("failed to start trace: %w", err)
		}
		if p.Config.Verbose {
			fmt.Printf("Trace started: %s\n", p.Config.TracePath)
		}
	}

	return nil
}

// Stop останавливает профилирование и сохраняет результаты
func (p *Profiler) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.endTime = time.Now()

	if p.Config.Verbose {
		fmt.Println("Stopping profiler...")
	}

	// Остановка CPU профилирования
	if p.cpuActive {
		p.stopCPUProfile()
		if p.Config.Verbose {
			fmt.Println("CPU profiling stopped")
		}
	}

	// Остановка трассировки
	if p.traceActive {
		p.stopTrace()
		if p.Config.Verbose {
			fmt.Println("Trace stopped")
		}
	}

	// Сохранение memory профиля
	if p.Config.EnableMemProfile {
		if err := p.writeMemProfile(); err != nil {
			return fmt.Errorf("failed to write memory profile: %w", err)
		}
		if p.Config.Verbose {
			fmt.Printf("Memory profile saved: %s\n", p.Config.MemProfilePath)
		}
	}

	// Вычисление финальной статистики
	p.Stats.TotalDuration = p.endTime.Sub(p.startTime)

	return nil
}

// startCPUProfile запускает CPU профилирование
func (p *Profiler) startCPUProfile() error {
	f, err := os.Create(p.Config.CPUProfilePath)
	if err != nil {
		return err
	}
	p.cpuFile = f

	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return err
	}

	p.cpuActive = true
	return nil
}

// stopCPUProfile останавливает CPU профилирование
func (p *Profiler) stopCPUProfile() {
	pprof.StopCPUProfile()
	if p.cpuFile != nil {
		p.cpuFile.Close()
		p.cpuFile = nil
	}
	p.cpuActive = false
}

// writeMemProfile записывает memory профиль
func (p *Profiler) writeMemProfile() error {
	f, err := os.Create(p.Config.MemProfilePath)
	if err != nil {
		return err
	}
	defer f.Close()

	runtime.GC() // получить актуальную статистику
	if err := pprof.WriteHeapProfile(f); err != nil {
		return err
	}

	return nil
}

// startTrace запускает трассировку
func (p *Profiler) startTrace() error {
	f, err := os.Create(p.Config.TracePath)
	if err != nil {
		return err
	}
	p.traceFile = f

	if err := trace.Start(f); err != nil {
		f.Close()
		return err
	}

	p.traceActive = true
	return nil
}

// stopTrace останавливает трассировку
func (p *Profiler) stopTrace() {
	trace.Stop()
	if p.traceFile != nil {
		p.traceFile.Close()
		p.traceFile = nil
	}
	p.traceActive = false
}

// GetStats возвращает текущую статистику
func (p *Profiler) GetStats() *Statistics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Обновляем текущую продолжительность, если профилирование активно
	if p.startTime.IsZero() == false && p.endTime.IsZero() {
		p.Stats.TotalDuration = time.Since(p.startTime)
	}

	return p.Stats
}

// PrintReport выводит отчет о профилировании
func (p *Profiler) PrintReport() {
	p.mu.RLock()
	defer p.mu.RUnlock()

	fmt.Println("\n=== Profiling Report ===")
	fmt.Printf("Total Duration: %v\n", p.Stats.TotalDuration)
	fmt.Printf("Start Time: %v\n", p.startTime.Format(time.RFC3339))
	fmt.Printf("End Time: %v\n", p.endTime.Format(time.RFC3339))

	if p.Config.EnableOperationMetrics {
		fmt.Println("\n--- Operation Metrics ---")
		p.OperationMetrics.PrintReport()
	}

	fmt.Println("\n--- Memory Statistics ---")
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc = %v MB\n", bToMb(m.Alloc))
	fmt.Printf("TotalAlloc = %v MB\n", bToMb(m.TotalAlloc))
	fmt.Printf("Sys = %v MB\n", bToMb(m.Sys))
	fmt.Printf("NumGC = %v\n", m.NumGC)

	fmt.Println("\n======================")
}

// bToMb конвертирует байты в мегабайты
func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
