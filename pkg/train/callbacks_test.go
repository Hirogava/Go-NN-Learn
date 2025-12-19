package train

import (
	"os"
	"testing"
	"time"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// Мок-модель для тестирования
type MockModel struct {
	params []*graph.Node
}

func (m *MockModel) Forward(x *graph.Node) *graph.Node { return x }
func (m *MockModel) Params() []*graph.Node              { return m.params }
func (m *MockModel) Layers() []layers.Layer             { return nil }

// newMockParam создает параметр для тестирования
func newMockParam(vals []float64, shape []int) *graph.Node {
	return &graph.Node{
		Value: &tensor.Tensor{
			Data:    append([]float64(nil), vals...),
			Shape:   append([]int(nil), shape...),
			Strides: nil,
		},
	}
}

// TestMetricsHistory проверяет работу истории метрик
func TestMetricsHistory(t *testing.T) {
	history := NewMetricsHistory()

	// Добавляем метрики
	history.Append(0, map[string]float64{"loss": 0.5, "accuracy": 0.8})
	history.Append(1, map[string]float64{"loss": 0.4, "accuracy": 0.85})
	history.Append(2, map[string]float64{"loss": 0.3, "accuracy": 0.9})

	// Проверяем Get
	lossValues := history.Get("loss")
	if len(lossValues) != 3 {
		t.Errorf("Expected 3 loss values, got %d", len(lossValues))
	}
	if lossValues[0] != 0.5 || lossValues[1] != 0.4 || lossValues[2] != 0.3 {
		t.Errorf("Unexpected loss values: %v", lossValues)
	}

	// Проверяем GetLast
	lastLoss := history.GetLast("loss")
	if lastLoss != 0.3 {
		t.Errorf("Expected last loss 0.3, got %f", lastLoss)
	}

	// Проверяем Best (min)
	epoch, value := history.Best("loss", "min")
	if epoch != 2 || value != 0.3 {
		t.Errorf("Expected best loss at epoch 2 with value 0.3, got epoch %d value %f", epoch, value)
	}

	// Проверяем Best (max)
	epoch, value = history.Best("accuracy", "max")
	if epoch != 2 || value != 0.9 {
		t.Errorf("Expected best accuracy at epoch 2 with value 0.9, got epoch %d value %f", epoch, value)
	}
}

// TestCallbackList проверяет работу списка колбэков
func TestCallbackList(t *testing.T) {
	count := 0
	callbackImpl := &testCallback{count: &count}

	list := NewCallbackList(callbackImpl)

	model := &MockModel{}
	ctx := NewTrainingContext(model, 10)

	// Вызываем OnEpochEnd
	if err := list.OnEpochEnd(ctx); err != nil {
		t.Errorf("OnEpochEnd failed: %v", err)
	}

	if count != 1 {
		t.Errorf("Expected callback to be called once, got %d", count)
	}
}

// testCallback - вспомогательный тип для тестирования
type testCallback struct {
	BaseCallback
	count *int
}

func (tc *testCallback) OnEpochEnd(ctx *TrainingContext) error {
	*tc.count++
	return nil
}

// TestEarlyStopping проверяет досрочную остановку
func TestEarlyStopping(t *testing.T) {
	model := &MockModel{}
	ctx := NewTrainingContext(model, 10)

	// EarlyStopping с patience=2
	es := NewEarlyStopping("loss", 2, "min", 0.001, false)

	// Эпоха 0: loss улучшается
	ctx.Epoch = 0
	ctx.Metrics["loss"] = 0.5
	es.OnEpochEnd(ctx)
	if ctx.StopTraining {
		t.Error("Should not stop on first epoch")
	}

	// Эпоха 1: loss улучшается
	ctx.Epoch = 1
	ctx.Metrics["loss"] = 0.4
	es.OnEpochEnd(ctx)
	if ctx.StopTraining {
		t.Error("Should not stop when improving")
	}

	// Эпоха 2: loss не улучшается (wait=1)
	ctx.Epoch = 2
	ctx.Metrics["loss"] = 0.41
	es.OnEpochEnd(ctx)
	if ctx.StopTraining {
		t.Error("Should not stop on first wait")
	}

	// Эпоха 3: loss не улучшается (wait=2, должна остановиться)
	ctx.Epoch = 3
	ctx.Metrics["loss"] = 0.42
	es.OnEpochEnd(ctx)
	if !ctx.StopTraining {
		t.Error("Should stop after patience epochs")
	}
}

// TestModelCheckpointFilename проверяет форматирование имени файла
func TestModelCheckpointFilename(t *testing.T) {
	mc := NewModelCheckpoint("model_{epoch}.ckpt", "loss", "min", 1, false, false)

	// Проверяем форматирование
	path := mc.formatFilepath(0)
	expected := "model_001.ckpt"
	if path != expected {
		t.Errorf("Expected %s, got %s", expected, path)
	}

	path = mc.formatFilepath(99)
	expected = "model_100.ckpt"
	if path != expected {
		t.Errorf("Expected %s, got %s", expected, path)
	}
}

// TestMetricsLoggerText проверяет текстовое логирование
func TestMetricsLoggerText(t *testing.T) {
	tmpFile := "test_log.txt"
	defer os.Remove(tmpFile)

	logger := NewMetricsLogger(tmpFile, LogFormatText, false, 1)

	model := &MockModel{}
	ctx := NewTrainingContext(model, 2)

	// Начало обучения
	if err := logger.OnTrainBegin(ctx); err != nil {
		t.Fatalf("OnTrainBegin failed: %v", err)
	}

	// Эпоха 0
	ctx.Epoch = 0
	ctx.Metrics["loss"] = 0.5
	ctx.Metrics["accuracy"] = 0.8
	if err := logger.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	// Конец обучения
	if err := logger.OnTrainEnd(ctx); err != nil {
		t.Fatalf("OnTrainEnd failed: %v", err)
	}

	// Проверяем что файл создан
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Log file was not created")
	}
}

// TestMetricsHistoryIsImproved проверяет определение улучшения
func TestMetricsHistoryIsImproved(t *testing.T) {
	history := NewMetricsHistory()

	// Первое значение всегда улучшение
	history.Append(0, map[string]float64{"loss": 0.5})
	if !history.IsImproved("loss", "min", 0.01) {
		t.Error("First value should be considered improved")
	}

	// Значительное улучшение
	history.Append(1, map[string]float64{"loss": 0.3})
	if !history.IsImproved("loss", "min", 0.01) {
		t.Error("Significant improvement should be detected")
	}

	// Незначительное улучшение (меньше minDelta)
	history.Append(2, map[string]float64{"loss": 0.295})
	if history.IsImproved("loss", "min", 0.01) {
		t.Error("Small improvement below minDelta should not be considered")
	}

	// Ухудшение
	history.Append(3, map[string]float64{"loss": 0.35})
	if history.IsImproved("loss", "min", 0.01) {
		t.Error("Deterioration should not be considered improved")
	}
}

// TestModelCheckpointSave проверяет сохранение чекпоинтов
func TestModelCheckpointSave(t *testing.T) {
	tmpDir := t.TempDir()
	ckptPath := tmpDir + "/model_{epoch}.ckpt"

	mc := NewModelCheckpoint(ckptPath, "loss", "min", 1, true, false)

	// Создаем модель с параметрами
	model := &MockModel{
		params: []*graph.Node{
			newMockParam([]float64{1.0, 2.0, 3.0}, []int{3}),
		},
	}
	ctx := NewTrainingContext(model, 3)

	// Эпоха 0: loss=0.5 (первое значение - всегда сохраняем)
	ctx.Epoch = 0
	ctx.Metrics["loss"] = 0.5
	if err := mc.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	// Проверяем что файл создан
	expectedPath := tmpDir + "/model_001.ckpt"
	if _, err := os.Stat(expectedPath); os.IsNotExist(err) {
		t.Error("Checkpoint file was not created")
	}

	// Эпоха 1: loss=0.3 (улучшение - сохраняем)
	ctx.Epoch = 1
	ctx.Metrics["loss"] = 0.3
	if err := mc.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	expectedPath = tmpDir + "/model_002.ckpt"
	if _, err := os.Stat(expectedPath); os.IsNotExist(err) {
		t.Error("Checkpoint file was not created for improved metric")
	}
}

// TestMetricsLoggerJSON проверяет JSON логирование
func TestMetricsLoggerJSON(t *testing.T) {
	tmpFile := t.TempDir() + "/test_log.json"
	defer os.Remove(tmpFile)

	logger := NewMetricsLogger(tmpFile, LogFormatJSON, false, 1)

	model := &MockModel{}
	ctx := NewTrainingContext(model, 2)

	if err := logger.OnTrainBegin(ctx); err != nil {
		t.Fatalf("OnTrainBegin failed: %v", err)
	}

	ctx.Epoch = 0
	ctx.Metrics["loss"] = 0.5
	ctx.Metrics["accuracy"] = 0.8
	if err := logger.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	if err := logger.OnTrainEnd(ctx); err != nil {
		t.Fatalf("OnTrainEnd failed: %v", err)
	}

	// Проверяем что файл создан и содержит JSON
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}

	if len(data) == 0 {
		t.Error("Log file is empty")
	}
}

// TestMetricsLoggerCSV проверяет CSV логирование
func TestMetricsLoggerCSV(t *testing.T) {
	tmpFile := t.TempDir() + "/test_log.csv"
	defer os.Remove(tmpFile)

	logger := NewMetricsLogger(tmpFile, LogFormatCSV, false, 1)

	model := &MockModel{}
	ctx := NewTrainingContext(model, 2)

	if err := logger.OnTrainBegin(ctx); err != nil {
		t.Fatalf("OnTrainBegin failed: %v", err)
	}

	ctx.Epoch = 0
	ctx.Metrics["loss"] = 0.5
	ctx.Metrics["accuracy"] = 0.8
	if err := logger.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	if err := logger.OnTrainEnd(ctx); err != nil {
		t.Fatalf("OnTrainEnd failed: %v", err)
	}

	// Проверяем что файл создан и содержит CSV с заголовками
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}

	content := string(data)
	if len(content) == 0 {
		t.Error("CSV file is empty")
	}

	// Проверяем наличие заголовков
	if !contains(content, "epoch") {
		t.Error("CSV should contain 'epoch' header")
	}
}

// TestProgressBarFormatting проверяет форматирование прогресс-бара
func TestProgressBarFormatting(t *testing.T) {
	pb := NewProgressBar(true, true)

	// Проверяем форматирование метрик
	metrics := map[string]float64{
		"loss":     0.4521,
		"accuracy": 0.8234,
	}
	formatted := pb.formatMetrics(metrics)

	if !contains(formatted, "accuracy") || !contains(formatted, "loss") {
		t.Errorf("Formatted metrics missing keys: %s", formatted)
	}

	// Проверяем форматирование времени
	duration := pb.formatDuration(125 * time.Second)
	expected := "02:05"
	if duration != expected {
		t.Errorf("Expected %s, got %s", expected, duration)
	}

	// Проверяем форматирование длинного времени
	duration = pb.formatDuration(3665 * time.Second)
	expected = "01:01:05"
	if duration != expected {
		t.Errorf("Expected %s, got %s", expected, duration)
	}
}

// TestMultipleCallbacks проверяет работу нескольких колбэков вместе
func TestMultipleCallbacks(t *testing.T) {
	tmpDir := t.TempDir()

	// Создаем несколько колбэков
	es := NewEarlyStopping("loss", 2, "min", 0.001, false)
	mc := NewModelCheckpoint(tmpDir+"/model_{epoch}.ckpt", "loss", "min", 1, true, false)
	pb := NewProgressBar(false, false) // Отключаем вывод для тестов

	list := NewCallbackList(es, mc, pb)

	model := &MockModel{
		params: []*graph.Node{
			newMockParam([]float64{1.0}, []int{1}),
		},
	}
	ctx := NewTrainingContext(model, 5)

	// Эпоха 0
	ctx.Epoch = 0
	ctx.Metrics["loss"] = 0.5
	if err := list.OnEpochBegin(ctx); err != nil {
		t.Fatalf("OnEpochBegin failed: %v", err)
	}
	if err := list.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	if ctx.StopTraining {
		t.Error("Should not stop on first epoch")
	}

	// Эпоха 1: улучшение
	ctx.Epoch = 1
	ctx.Metrics["loss"] = 0.3
	if err := list.OnEpochEnd(ctx); err != nil {
		t.Fatalf("OnEpochEnd failed: %v", err)
	}

	if ctx.StopTraining {
		t.Error("Should not stop when improving")
	}

	// Проверяем что чекпоинт создан
	if _, err := os.Stat(tmpDir + "/model_002.ckpt"); os.IsNotExist(err) {
		t.Error("Checkpoint should be created for improved metric")
	}
}

// contains проверяет содержит ли строка подстроку
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) &&
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
			containsHelper(s, substr)))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
