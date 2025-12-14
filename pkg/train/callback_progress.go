package train

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ProgressBar отображает прогресс обучения в консоли.
// Показывает прогресс по эпохам и батчам с метриками и временем.
//
// Пример вывода:
//   Epoch 1/10: 100% |████████████████████| 313/313 [00:12<00:00, 25.42batch/s] loss: 0.4521
type ProgressBar struct {
	BaseCallback

	showEpoch bool // Показывать прогресс эпох
	showBatch bool // Показывать прогресс батчей

	// Внутреннее состояние
	epochStartTime time.Time // Время начала эпохи
	lastUpdate     time.Time // Время последнего обновления
	barWidth       int       // Ширина прогресс-бара
}

// NewProgressBar создает новый колбэк для отображения прогресса.
//
// Параметры:
//   showEpoch - показывать прогресс по эпохам
//   showBatch - показывать прогресс по батчам (прогресс-бар)
//
// Примеры:
//   // Показывать только прогресс эпох
//   NewProgressBar(true, false)
//
//   // Показывать детальный прогресс с батчами
//   NewProgressBar(true, true)
func NewProgressBar(showEpoch, showBatch bool) *ProgressBar {
	return &ProgressBar{
		showEpoch: showEpoch,
		showBatch: showBatch,
		barWidth:  30, // Ширина прогресс-бара по умолчанию
	}
}

// OnEpochBegin вызывается в начале эпохи.
func (pb *ProgressBar) OnEpochBegin(ctx *TrainingContext) error {
	pb.epochStartTime = time.Now()
	pb.lastUpdate = time.Now()

	if pb.showEpoch && !pb.showBatch {
		// Простой вывод начала эпохи (если не показываем батчи)
		fmt.Printf("Epoch %d/%d\n", ctx.Epoch+1, ctx.NumEpochs)
	}

	return nil
}

// OnEpochEnd вызывается в конце эпохи.
func (pb *ProgressBar) OnEpochEnd(ctx *TrainingContext) error {
	if !pb.showEpoch {
		return nil
	}

	elapsed := time.Since(pb.epochStartTime)

	if pb.showBatch {
		// Завершаем прогресс-бар батчей переносом строки
		fmt.Println()
	}

	// Выводим итоговую информацию эпохи
	if pb.showEpoch {
		metricsStr := pb.formatMetrics(ctx.Metrics)
		fmt.Printf("Epoch %d/%d completed in %s - %s\n",
			ctx.Epoch+1, ctx.NumEpochs, pb.formatDuration(elapsed), metricsStr)
	}

	return nil
}

// OnBatchBegin вызывается перед обработкой батча.
func (pb *ProgressBar) OnBatchBegin(ctx *TrainingContext) error {
	// Инициализация если нужно
	if ctx.Batch == 0 && pb.showBatch {
		pb.lastUpdate = time.Now()
	}
	return nil
}

// OnBatchEnd обновляет прогресс-бар после обработки батча.
func (pb *ProgressBar) OnBatchEnd(ctx *TrainingContext) error {
	if !pb.showBatch {
		return nil
	}

	// Обновляем не чаще чем раз в 100мс для производительности
	now := time.Now()
	if now.Sub(pb.lastUpdate) < 100*time.Millisecond && ctx.Batch < ctx.NumBatches-1 {
		return nil
	}
	pb.lastUpdate = now

	// Вычисляем прогресс
	progress := float64(ctx.Batch+1) / float64(ctx.NumBatches)
	filled := int(progress * float64(pb.barWidth))

	// Строим прогресс-бар
	bar := strings.Repeat("█", filled) + strings.Repeat("░", pb.barWidth-filled)

	// Вычисляем время
	elapsed := time.Since(pb.epochStartTime)
	batchesPerSec := float64(ctx.Batch+1) / elapsed.Seconds()
	remaining := time.Duration(0)
	if batchesPerSec > 0 {
		remainingBatches := ctx.NumBatches - (ctx.Batch + 1)
		remaining = time.Duration(float64(remainingBatches)/batchesPerSec) * time.Second
	}

	// Форматируем метрики
	metricsStr := pb.formatMetrics(ctx.Metrics)

	// Выводим прогресс-бар (с возвратом каретки для обновления на месте)
	fmt.Printf("\rEpoch %d/%d: %3.0f%% |%s| %d/%d [%s<%s, %.2fbatch/s] %s",
		ctx.Epoch+1, ctx.NumEpochs,
		progress*100,
		bar,
		ctx.Batch+1, ctx.NumBatches,
		pb.formatDuration(elapsed),
		pb.formatDuration(remaining),
		batchesPerSec,
		metricsStr,
	)

	return nil
}

// formatMetrics форматирует метрики в строку "loss: 0.4521 - acc: 0.8234"
func (pb *ProgressBar) formatMetrics(metrics map[string]float64) string {
	if len(metrics) == 0 {
		return ""
	}

	// Сортируем метрики по имени
	names := make([]string, 0, len(metrics))
	for name := range metrics {
		names = append(names, name)
	}
	sort.Strings(names)

	// Форматируем
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s: %.4f", name, metrics[name]))
	}

	return strings.Join(parts, " - ")
}

// formatDuration форматирует длительность в читаемый вид (MM:SS или HH:MM:SS)
func (pb *ProgressBar) formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second

	if h > 0 {
		return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}

// SetBarWidth устанавливает ширину прогресс-бара (по умолчанию 30).
func (pb *ProgressBar) SetBarWidth(width int) {
	if width > 0 && width <= 100 {
		pb.barWidth = width
	}
}
