package train_test

import (
	"fmt"
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/train"
)

// simpleInit - простая функция инициализации для примеров
func simpleInit(data []float64) {
	for i := range data {
		data[i] = rand.Float64()*0.2 - 0.1
	}
}

// SimpleModel - пример простой модели для демонстрации
type SimpleModel struct {
	layer *layers.Dense
}

func (m *SimpleModel) Forward(x *graph.Node) *graph.Node {
	return m.layer.Forward(x)
}

func (m *SimpleModel) Params() []*graph.Node {
	return m.layer.Params()
}

func (m *SimpleModel) Layers() []layers.Layer {
	return []layers.Layer{m.layer}
}

// Example_basicTrainingLoop демонстрирует базовый цикл обучения с колбэками
func Example_basicTrainingLoop() {
	// Создаем модель
	model := &SimpleModel{
		layer: layers.NewDense(10, 5, simpleInit),
	}

	// Создаем контекст обучения
	ctx := train.NewTrainingContext(model, 10)

	// Настраиваем колбэки
	callbacks := train.NewCallbackList(
		train.NewProgressBar(true, true),
		train.NewEarlyStopping("loss", 3, "min", 0.001, true),
	)

	// Начало обучения
	callbacks.OnTrainBegin(ctx)

	// Имитация цикла обучения
	for epoch := 0; epoch < ctx.NumEpochs; epoch++ {
		ctx.Epoch = epoch
		callbacks.OnEpochBegin(ctx)

		// Имитация батчей
		ctx.NumBatches = 100
		for batch := 0; batch < ctx.NumBatches; batch++ {
			ctx.Batch = batch
			callbacks.OnBatchBegin(ctx)

			// Здесь была бы реальная тренировка...
			// loss := trainStep(...)

			callbacks.OnBatchEnd(ctx)
		}

		// Устанавливаем метрики эпохи
		ctx.Metrics["loss"] = 0.5 - float64(epoch)*0.05
		ctx.Metrics["accuracy"] = 0.7 + float64(epoch)*0.02

		callbacks.OnEpochEnd(ctx)

		// Проверяем досрочную остановку
		if ctx.StopTraining {
			fmt.Println("Training stopped early")
			break
		}
	}

	// Конец обучения
	callbacks.OnTrainEnd(ctx)
}

// Example_modelCheckpoint демонстрирует сохранение чекпоинтов
func Example_modelCheckpoint() {
	model := &SimpleModel{
		layer: layers.NewDense(10, 5, simpleInit),
	}

	// Создаем колбэк для сохранения лучшей модели
	checkpoint := train.NewModelCheckpoint(
		"models/best_model.ckpt", // Путь к файлу
		"val_loss",                // Какую метрику отслеживать
		"min",                     // Минимизировать метрику
		0,                         // Не сохранять по частоте (только лучшую)
		true,                      // Сохранять только лучшую
		true,                      // Выводить сообщения
	)

	ctx := train.NewTrainingContext(model, 5)
	checkpoint.OnTrainBegin(ctx)

	for epoch := 0; epoch < 5; epoch++ {
		ctx.Epoch = epoch
		ctx.Metrics["val_loss"] = 0.5 - float64(epoch)*0.1

		checkpoint.OnEpochEnd(ctx)
		// Вывод: Epoch 00001: val_loss improved from inf to 0.50000, saving model to models/best_model.ckpt
		// Вывод: Epoch 00002: val_loss improved from 0.50000 to 0.40000, saving model to models/best_model.ckpt
		// и т.д.
	}
}

// Example_metricsLogger демонстрирует логирование метрик
func Example_metricsLogger() {
	model := &SimpleModel{
		layer: layers.NewDense(10, 5, simpleInit),
	}

	// Создаем логгер в формате CSV
	logger := train.NewMetricsLogger(
		"logs/train.csv",      // Путь к файлу
		train.LogFormatCSV,    // Формат CSV
		true,                     // Выводить в консоль
		1,                        // Логировать каждую эпоху
	)

	ctx := train.NewTrainingContext(model, 3)
	logger.OnTrainBegin(ctx)

	for epoch := 0; epoch < 3; epoch++ {
		ctx.Epoch = epoch
		ctx.Metrics["loss"] = 0.5 - float64(epoch)*0.1
		ctx.Metrics["accuracy"] = 0.7 + float64(epoch)*0.05

		logger.OnEpochEnd(ctx)
		// CSV файл будет содержать:
		// epoch,accuracy,loss
		// 0,0.70,0.50
		// 1,0.75,0.40
		// 2,0.80,0.30
	}

	logger.OnTrainEnd(ctx)
}

// Example_earlyStopping демонстрирует досрочную остановку обучения
func Example_earlyStopping() {
	model := &SimpleModel{
		layer: layers.NewDense(10, 5, simpleInit),
	}

	// Остановка если val_loss не улучшается 3 эпохи подряд
	earlyStopping := train.NewEarlyStopping(
		"val_loss", // Отслеживаемая метрика
		3,          // Patience (терпение) - сколько эпох ждать
		"min",      // Минимизировать метрику
		0.001,      // Минимальное изменение для улучшения
		true,       // Выводить сообщения
	)

	ctx := train.NewTrainingContext(model, 10)
	earlyStopping.OnTrainBegin(ctx)

	losses := []float64{0.5, 0.4, 0.35, 0.34, 0.34, 0.34, 0.34}

	for epoch := 0; epoch < len(losses); epoch++ {
		ctx.Epoch = epoch
		ctx.Metrics["val_loss"] = losses[epoch]

		earlyStopping.OnEpochEnd(ctx)

		if ctx.StopTraining {
			fmt.Printf("Stopped at epoch %d due to no improvement\n", epoch)
			break
		}
	}
	// Вывод: Stopped at epoch 6 due to no improvement
}

// Example_multipleCallbacks демонстрирует использование нескольких колбэков
func Example_multipleCallbacks() {
	model := &SimpleModel{
		layer: layers.NewDense(10, 5, simpleInit),
	}

	// Комбинируем несколько колбэков
	callbacks := train.NewCallbackList(
		// Прогресс-бар
		train.NewProgressBar(true, true),

		// Логирование в JSON
		train.NewMetricsLogger("logs/train.json", train.LogFormatJSON, false, 1),

		// Сохранение чекпоинтов каждые 5 эпох
		train.NewModelCheckpoint("models/model_{epoch}.ckpt", "", "", 5, false, true),

		// Сохранение лучшей модели
		train.NewModelCheckpoint("models/best.ckpt", "val_loss", "min", 0, true, true),

		// Досрочная остановка
		train.NewEarlyStopping("val_loss", 10, "min", 0.0001, true),
	)

	ctx := train.NewTrainingContext(model, 100)

	// Все колбэки будут вызваны автоматически
	callbacks.OnTrainBegin(ctx)

	for epoch := 0; epoch < ctx.NumEpochs; epoch++ {
		ctx.Epoch = epoch
		callbacks.OnEpochBegin(ctx)

		// Обучение...
		ctx.Metrics["loss"] = 0.5
		ctx.Metrics["val_loss"] = 0.6

		callbacks.OnEpochEnd(ctx)

		if ctx.StopTraining {
			break
		}
	}

	callbacks.OnTrainEnd(ctx)
}

// Example_customCallback демонстрирует создание пользовательского колбэка
func Example_customCallback() {
	// Пользовательский колбэк, который печатает сообщение каждые N эпох
	type PrintCallback struct {
		train.BaseCallback
		frequency int
	}

	printCallback := &PrintCallback{frequency: 5}

	// Переопределяем только нужный метод
	printCallback.BaseCallback = train.BaseCallback{}

	model := &SimpleModel{
		layer: layers.NewDense(10, 5, simpleInit),
	}

	ctx := train.NewTrainingContext(model, 20)
	callbacks := train.NewCallbackList(printCallback)

	callbacks.OnTrainBegin(ctx)

	for epoch := 0; epoch < ctx.NumEpochs; epoch++ {
		ctx.Epoch = epoch
		ctx.Metrics["loss"] = 0.5 - float64(epoch)*0.01

		callbacks.OnEpochEnd(ctx)
	}

	callbacks.OnTrainEnd(ctx)
}

// Example_metricsHistory демонстрирует работу с историей метрик
func Example_metricsHistory() {
	history := train.NewMetricsHistory()

	// Добавляем метрики за несколько эпох
	for epoch := 0; epoch < 10; epoch++ {
		metrics := map[string]float64{
			"loss":     0.5 - float64(epoch)*0.04,
			"accuracy": 0.7 + float64(epoch)*0.02,
		}
		history.Append(epoch, metrics)
	}

	// Получаем все значения метрики
	losses := history.Get("loss")
	fmt.Printf("Loss history has %d values\n", len(losses))

	// Получаем последнее значение
	lastLoss := history.GetLast("loss")
	fmt.Printf("Last loss: %.2f\n", lastLoss)

	// Находим лучшее значение
	bestEpoch, bestLoss := history.Best("loss", "min")
	fmt.Printf("Best loss: %.2f at epoch %d\n", bestLoss, bestEpoch)

	// Проверяем улучшение
	improved := history.IsImproved("loss", "min", 0.01)
	fmt.Printf("Improved: %v\n", improved)

	// Output:
	// Loss history has 10 values
	// Last loss: 0.14
	// Best loss: 0.14 at epoch 9
	// Improved: true
}
