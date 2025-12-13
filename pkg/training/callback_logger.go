package training

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// LogFormat определяет формат вывода логов.
type LogFormat string

const (
	LogFormatText LogFormat = "text" // Человеко-читаемый текстовый формат
	LogFormatJSON LogFormat = "json" // JSON формат (по строке на эпоху)
	LogFormatCSV  LogFormat = "csv"  // CSV формат с заголовками
)

// MetricsLogger логирует метрики обучения в файл и/или stdout.
// Поддерживает три формата: text, json, csv.
//
// Форматы вывода:
//
// Text:
//   Epoch 1/10 - loss: 0.4521 - accuracy: 0.8234
//
// JSON:
//   {"epoch": 1, "loss": 0.4521, "accuracy": 0.8234}
//
// CSV:
//   epoch,loss,accuracy
//   1,0.4521,0.8234
type MetricsLogger struct {
	BaseCallback

	logFile string    // Путь к лог-файлу (пустая строка = не записывать в файл)
	format  LogFormat // Формат логов (text, json, csv)
	verbose bool      // Выводить в stdout
	logFreq int       // Частота логирования (1 = каждую эпоху)

	// Внутреннее состояние
	file          *os.File // Открытый файл для записи
	csvWriter     *csv.Writer
	headerWritten bool     // Для CSV: заголовок уже записан
	metricNames   []string // Для CSV: упорядоченные имена метрик
}

// NewMetricsLogger создает новый колбэк для логирования метрик.
//
// Параметры:
//   logFile - путь к лог-файлу (пустая строка = не записывать в файл)
//   format - формат логов: "text", "json", или "csv"
//   verbose - true: выводить в stdout, false: только в файл
//   logFreq - частота логирования в эпохах (1 = каждую эпоху)
//
// Примеры:
//   // Логировать в stdout каждую эпоху (текстовый формат)
//   NewMetricsLogger("", LogFormatText, true, 1)
//
//   // Логировать в файл и stdout (JSON)
//   NewMetricsLogger("training.log", LogFormatJSON, true, 1)
//
//   // Логировать в CSV файл каждые 5 эпох
//   NewMetricsLogger("metrics.csv", LogFormatCSV, false, 5)
func NewMetricsLogger(logFile string, format LogFormat, verbose bool, logFreq int) *MetricsLogger {
	return &MetricsLogger{
		logFile: logFile,
		format:  format,
		verbose: verbose,
		logFreq: logFreq,
	}
}

// OnTrainBegin открывает лог-файл если указан.
func (ml *MetricsLogger) OnTrainBegin(ctx *TrainingContext) error {
	if ml.logFile == "" {
		return nil // Нет файла для записи
	}

	// Открываем файл для записи (создаем если не существует)
	file, err := os.Create(ml.logFile)
	if err != nil {
		return fmt.Errorf("MetricsLogger: failed to create log file: %w", err)
	}

	ml.file = file

	// Для CSV создаем writer
	if ml.format == LogFormatCSV {
		ml.csvWriter = csv.NewWriter(file)
	}

	return nil
}

// OnTrainEnd закрывает лог-файл.
func (ml *MetricsLogger) OnTrainEnd(ctx *TrainingContext) error {
	if ml.file != nil {
		// Для CSV сбрасываем буфер
		if ml.csvWriter != nil {
			ml.csvWriter.Flush()
			if err := ml.csvWriter.Error(); err != nil {
				return fmt.Errorf("MetricsLogger: CSV flush error: %w", err)
			}
		}

		if err := ml.file.Close(); err != nil {
			return fmt.Errorf("MetricsLogger: failed to close log file: %w", err)
		}
		ml.file = nil
	}

	return nil
}

// OnEpochEnd логирует метрики эпохи.
func (ml *MetricsLogger) OnEpochEnd(ctx *TrainingContext) error {
	// Проверяем частоту логирования
	if ml.logFreq > 0 && (ctx.Epoch+1)%ml.logFreq != 0 {
		return nil // Пропускаем эту эпоху
	}

	// Формируем строку для вывода
	var logLine string
	var err error

	switch ml.format {
	case LogFormatText:
		logLine = ml.formatText(ctx)
	case LogFormatJSON:
		logLine, err = ml.formatJSON(ctx)
		if err != nil {
			return fmt.Errorf("MetricsLogger: JSON format error: %w", err)
		}
	case LogFormatCSV:
		err = ml.formatCSV(ctx)
		if err != nil {
			return fmt.Errorf("MetricsLogger: CSV format error: %w", err)
		}
		// Для CSV логирование в файл уже выполнено в formatCSV
		// Выводим в stdout только если verbose=true
		if ml.verbose {
			logLine = ml.formatText(ctx) // Для stdout используем текстовый формат
		}
	default:
		logLine = ml.formatText(ctx) // По умолчанию текстовый
	}

	// Вывод в stdout
	if ml.verbose && logLine != "" {
		fmt.Println(logLine)
	}

	// Запись в файл (для text и json)
	if ml.file != nil && ml.format != LogFormatCSV {
		if _, err := ml.file.WriteString(logLine + "\n"); err != nil {
			return fmt.Errorf("MetricsLogger: failed to write to file: %w", err)
		}
	}

	return nil
}

// formatText форматирует метрики в человеко-читаемый текст.
// Формат: "Epoch 1/10 - loss: 0.4521 - accuracy: 0.8234"
func (ml *MetricsLogger) formatText(ctx *TrainingContext) string {
	var parts []string
	parts = append(parts, fmt.Sprintf("Epoch %d/%d", ctx.Epoch+1, ctx.NumEpochs))

	// Сортируем метрики по имени для консистентности
	names := make([]string, 0, len(ctx.Metrics))
	for name := range ctx.Metrics {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		value := ctx.Metrics[name]
		parts = append(parts, fmt.Sprintf("%s: %.4f", name, value))
	}

	return strings.Join(parts, " - ")
}

// formatJSON форматирует метрики в JSON.
// Формат: {"epoch": 1, "loss": 0.4521, "accuracy": 0.8234}
func (ml *MetricsLogger) formatJSON(ctx *TrainingContext) (string, error) {
	data := make(map[string]interface{})
	data["epoch"] = ctx.Epoch + 1

	// Копируем все метрики
	for name, value := range ctx.Metrics {
		data[name] = value
	}

	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

// formatCSV записывает метрики в CSV формат.
// При первом вызове записывает заголовок.
func (ml *MetricsLogger) formatCSV(ctx *TrainingContext) error {
	if ml.csvWriter == nil {
		return fmt.Errorf("CSV writer not initialized")
	}

	// Первый раз: записываем заголовок
	if !ml.headerWritten {
		// Собираем имена метрик и сортируем для консистентности
		ml.metricNames = make([]string, 0, len(ctx.Metrics))
		for name := range ctx.Metrics {
			ml.metricNames = append(ml.metricNames, name)
		}
		sort.Strings(ml.metricNames)

		// Заголовок: epoch, метрика1, метрика2, ...
		header := []string{"epoch"}
		header = append(header, ml.metricNames...)

		if err := ml.csvWriter.Write(header); err != nil {
			return err
		}

		ml.headerWritten = true
	}

	// Записываем строку данных
	record := []string{fmt.Sprintf("%d", ctx.Epoch+1)}
	for _, name := range ml.metricNames {
		value := ctx.Metrics[name]
		record = append(record, fmt.Sprintf("%.6f", value))
	}

	if err := ml.csvWriter.Write(record); err != nil {
		return err
	}

	// Сбрасываем буфер после каждой эпохи
	ml.csvWriter.Flush()
	return ml.csvWriter.Error()
}
