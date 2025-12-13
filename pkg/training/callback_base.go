package training

// BaseCallback предоставляет базовую реализацию интерфейса Callback.
// Все методы возвращают nil (no-op).
//
// Пользовательские колбэки могут встраивать BaseCallback и переопределять
// только нужные методы, не реализуя весь интерфейс.
//
// Пример:
//   type MyCallback struct {
//       BaseCallback
//       counter int
//   }
//
//   func (c *MyCallback) OnEpochEnd(ctx *TrainingContext) error {
//       c.counter++
//       fmt.Printf("Completed epoch %d\n", ctx.Epoch)
//       return nil
//   }
type BaseCallback struct{}

// OnTrainBegin - no-op реализация.
func (b *BaseCallback) OnTrainBegin(ctx *TrainingContext) error {
	return nil
}

// OnTrainEnd - no-op реализация.
func (b *BaseCallback) OnTrainEnd(ctx *TrainingContext) error {
	return nil
}

// OnEpochBegin - no-op реализация.
func (b *BaseCallback) OnEpochBegin(ctx *TrainingContext) error {
	return nil
}

// OnEpochEnd - no-op реализация.
func (b *BaseCallback) OnEpochEnd(ctx *TrainingContext) error {
	return nil
}

// OnBatchBegin - no-op реализация.
func (b *BaseCallback) OnBatchBegin(ctx *TrainingContext) error {
	return nil
}

// OnBatchEnd - no-op реализация.
func (b *BaseCallback) OnBatchEnd(ctx *TrainingContext) error {
	return nil
}
