package optimizers

// LearningRateScheduler - интерфейс для различных стратегий изменения темпа обучения.
// Scheduler изменяет Learning Rate в процессе тренировки для улучшения сходимости.
type LearningRateScheduler interface {
	// Step вызывается после каждой эпохи или батча для обновления Learning Rate.
	// Возвращает новый Learning Rate.
	Step() float64

	// GetLastLR возвращает последний вычисленный Learning Rate.
	GetLastLR() float64
}

// LearningRateAdjustable - интерфейс для оптимизаторов, которые поддерживают изменение Learning Rate.
type LearningRateAdjustable interface {
	// SetLearningRate устанавливает новый Learning Rate для оптимизатора.
	SetLearningRate(lr float64)
}

