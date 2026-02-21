package train

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
)

// ModelCheckpoint сохраняет чекпоинты модели во время обучения.
// Поддерживает:
// - Сохранение каждые N эпох
// - Сохранение только лучшей модели по метрике
// - Шаблоны в пути файла (например, "model_epoch_{epoch}.ckpt")
//
// Использует api.SaveCheckpoint для сохранения модели.
type ModelCheckpoint struct {
	BaseCallback

	filepath string // Путь для сохранения (может содержать "{epoch}")
	monitor  string // Метрика для мониторинга ("loss", "accuracy" и т.д.)
	mode     string // "min" для минимизации, "max" для максимизации
	saveFreq int    // Частота сохранения (каждые N эпох), 0 = только при улучшении
	saveBest bool   // Сохранять только лучшую модель (true) или все (false)
	verbose  bool   // Выводить сообщения о сохранении

	// Внутреннее состояние
	bestValue float64 // Лучшее значение метрики
	bestEpoch int     // Эпоха с лучшим значением
}

// NewModelCheckpoint создает новый колбэк для сохранения чекпоинтов.
//
// Параметры:
//
//	filepath - путь для сохранения, может содержать "{epoch}" для подстановки номера эпохи
//	           Пример: "checkpoints/model_epoch_{epoch}.ckpt" -> "checkpoints/model_epoch_005.ckpt"
//	monitor - метрика для мониторинга ("loss", "val_loss", "accuracy" и т.д.)
//	mode - "min" для минимизации метрики (loss), "max" для максимизации (accuracy)
//	saveFreq - частота сохранения в эпохах (1 = каждую эпоху, 5 = каждую 5-ю эпоху)
//	           0 = сохранять только при улучшении метрики
//	saveBest - true: заменять файл только если метрика улучшилась
//	           false: сохранять всегда (при достижении saveFreq)
//	verbose - true: выводить сообщения о сохранении
//
// Примеры:
//
//	// Сохранять каждую эпоху с номером в имени
//	NewModelCheckpoint("models/model_{epoch}.ckpt", "loss", "min", 1, false, true)
//
//	// Сохранять только лучшую модель (заменяя файл)
//	NewModelCheckpoint("models/best_model.ckpt", "loss", "min", 0, true, true)
//
//	// Сохранять каждые 5 эпох
//	NewModelCheckpoint("models/model_{epoch}.ckpt", "loss", "min", 5, false, true)
func NewModelCheckpoint(filepath, monitor, mode string, saveFreq int, saveBest, verbose bool) *ModelCheckpoint {
	initialValue := 0.0
	if mode == "min" {
		initialValue = math.Inf(1) // +Inf для минимизации
	} else {
		initialValue = math.Inf(-1) // -Inf для максимизации
	}

	return &ModelCheckpoint{
		filepath:  filepath,
		monitor:   monitor,
		mode:      mode,
		saveFreq:  saveFreq,
		saveBest:  saveBest,
		verbose:   verbose,
		bestValue: initialValue,
		bestEpoch: -1,
	}
}

// OnEpochEnd проверяет метрику и сохраняет модель если условия выполнены.
func (mc *ModelCheckpoint) OnEpochEnd(ctx *TrainingContext) error {
	// Получаем текущее значение монitorируемой метрики
	currentValue, exists := ctx.Metrics[mc.monitor]
	if !exists {
		// Метрика не найдена, пропускаем
		if mc.verbose {
			fmt.Printf("ModelCheckpoint: metric '%s' not found in context.Metrics\n", mc.monitor)
		}
		return nil
	}

	shouldSave := false
	improved := false

	// Проверяем улучшение
	if mc.mode == "min" {
		if currentValue < mc.bestValue {
			mc.bestValue = currentValue
			mc.bestEpoch = ctx.Epoch
			improved = true
		}
	} else if mc.mode == "max" {
		if currentValue > mc.bestValue {
			mc.bestValue = currentValue
			mc.bestEpoch = ctx.Epoch
			improved = true
		}
	}

	// Решаем, сохранять ли модель
	if mc.saveBest {
		// Сохраняем только при улучшении
		shouldSave = improved
	} else if mc.saveFreq > 0 {
		// Сохраняем каждые saveFreq эпох
		shouldSave = (ctx.Epoch+1)%mc.saveFreq == 0
	} else {
		// saveFreq=0 и saveBest=false: сохраняем только при улучшении
		shouldSave = improved
	}

	if shouldSave {
		// Формируем путь с подстановкой эпохи
		savePath := mc.formatFilepath(ctx.Epoch)

		// Сохраняем модель
		if err := api.SaveCheckpoint(ctx.Model, savePath); err != nil {
			return fmt.Errorf("ModelCheckpoint: failed to save checkpoint: %w", err)
		}

		if mc.verbose {
			if improved {
				// Нужно вывести старое значение (до обновления), но мы уже обновили bestValue
				// Поэтому выведем просто улучшение
				fmt.Printf("Epoch %05d: %s improved to %.5f, saving model to %s\n",
					ctx.Epoch+1, mc.monitor, currentValue, savePath)
			} else {
				fmt.Printf("Epoch %05d: saving model to %s\n", ctx.Epoch+1, savePath)
			}
		}
	}

	return nil
}

// formatFilepath подставляет номер эпохи в путь файла.
// Поддерживает шаблон "{epoch}" который заменяется на номер эпохи с ведущими нулями.
func (mc *ModelCheckpoint) formatFilepath(epoch int) string {
	path := mc.filepath

	// Подстановка {epoch} с форматированием (3 цифры с ведущими нулями)
	if strings.Contains(path, "{epoch}") {
		epochStr := fmt.Sprintf("%03d", epoch+1) // +1 для human-readable (эпохи с 1)
		path = strings.ReplaceAll(path, "{epoch}", epochStr)
	}

	return filepath.Clean(path)
}

// GetBestEpoch возвращает номер эпохи с лучшим значением метрики.
// Возвращает -1 если лучшее значение еще не установлено.
func (mc *ModelCheckpoint) GetBestEpoch() int {
	return mc.bestEpoch
}

// GetBestValue возвращает лучшее значение монitorируемой метрики.
func (mc *ModelCheckpoint) GetBestValue() float64 {
	return mc.bestValue
}
