package train

// Callback определяет интерфейс для колбэков тренировочного цикла.
// Колбэки вызываются в ключевые моменты обучения (начало/конец эпохи, начало/конец батча).
// Паттерн Observer позволяет подключать дополнительное поведение без изменения основного цикла.
type Callback interface {
	// OnTrainBegin вызывается один раз в начале всего процесса обучения.
	// Используется для инициализации (открытие файлов, создание директорий и т.д.).
	OnTrainBegin(ctx *TrainingContext) error

	// OnTrainEnd вызывается один раз в конце всего процесса обучения.
	// Используется для финализации (закрытие файлов, сохранение финальной модели и т.д.).
	OnTrainEnd(ctx *TrainingContext) error

	// OnEpochBegin вызывается в начале каждой эпохи.
	// Может использоваться для сброса метрик, изменения learning rate и т.д.
	OnEpochBegin(ctx *TrainingContext) error

	// OnEpochEnd вызывается в конце каждой эпохи.
	// Используется для логирования, сохранения чекпоинтов, проверки early stopping и т.д.
	OnEpochEnd(ctx *TrainingContext) error

	// OnBatchBegin вызывается перед обработкой каждого батча.
	// Может использоваться для динамического изменения параметров обучения.
	OnBatchBegin(ctx *TrainingContext) error

	// OnBatchEnd вызывается после обработки каждого батча.
	// Используется для обновления прогресс-баров, накопления метрик и т.д.
	OnBatchEnd(ctx *TrainingContext) error
}

// CallbackList управляет списком колбэков и вызывает их в правильном порядке.
// Реализует паттерн Observer с поддержкой множественных наблюдателей.
type CallbackList struct {
	callbacks []Callback
}

// NewCallbackList создает новый список колбэков.
// Принимает переменное количество колбэков для удобства инициализации.
//
// Пример:
//   list := NewCallbackList(
//       NewMetricsLogger("", true, 1),
//       NewModelCheckpoint("model.ckpt", "loss", "min", 1, false, true),
//   )
func NewCallbackList(callbacks ...Callback) *CallbackList {
	return &CallbackList{
		callbacks: callbacks,
	}
}

// Add добавляет новый колбэк в список.
// Колбэк будет вызываться после уже существующих.
func (cl *CallbackList) Add(callback Callback) {
	cl.callbacks = append(cl.callbacks, callback)
}

// Len возвращает количество зарегистрированных колбэков.
func (cl *CallbackList) Len() int {
	return len(cl.callbacks)
}

// OnTrainBegin вызывает OnTrainBegin для всех колбэков в порядке добавления.
// При ошибке в любом колбэке прерывает цепочку и возвращает ошибку.
func (cl *CallbackList) OnTrainBegin(ctx *TrainingContext) error {
	for _, cb := range cl.callbacks {
		if err := cb.OnTrainBegin(ctx); err != nil {
			return err
		}
	}
	return nil
}

// OnTrainEnd вызывает OnTrainEnd для всех колбэков в порядке добавления.
// При ошибке в любом колбэке прерывает цепочку и возвращает ошибку.
func (cl *CallbackList) OnTrainEnd(ctx *TrainingContext) error {
	for _, cb := range cl.callbacks {
		if err := cb.OnTrainEnd(ctx); err != nil {
			return err
		}
	}
	return nil
}

// OnEpochBegin вызывает OnEpochBegin для всех колбэков в порядке добавления.
// При ошибке в любом колбэке прерывает цепочку и возвращает ошибку.
func (cl *CallbackList) OnEpochBegin(ctx *TrainingContext) error {
	for _, cb := range cl.callbacks {
		if err := cb.OnEpochBegin(ctx); err != nil {
			return err
		}
	}
	return nil
}

// OnEpochEnd вызывает OnEpochEnd для всех колбэков в порядке добавления.
// При ошибке в любом колбэке прерывает цепочку и возвращает ошибку.
func (cl *CallbackList) OnEpochEnd(ctx *TrainingContext) error {
	for _, cb := range cl.callbacks {
		if err := cb.OnEpochEnd(ctx); err != nil {
			return err
		}
	}
	return nil
}

// OnBatchBegin вызывает OnBatchBegin для всех колбэков в порядке добавления.
// При ошибке в любом колбэке прерывает цепочку и возвращает ошибку.
func (cl *CallbackList) OnBatchBegin(ctx *TrainingContext) error {
	for _, cb := range cl.callbacks {
		if err := cb.OnBatchBegin(ctx); err != nil {
			return err
		}
	}
	return nil
}

// OnBatchEnd вызывает OnBatchEnd для всех колбэков в порядке добавления.
// При ошибке в любом колбэке прерывает цепочку и возвращает ошибку.
func (cl *CallbackList) OnBatchEnd(ctx *TrainingContext) error {
	for _, cb := range cl.callbacks {
		if err := cb.OnBatchEnd(ctx); err != nil {
			return err
		}
	}
	return nil
}
