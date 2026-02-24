package train

import (
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
)

// TrainerConfig задаёт параметры обучения для создания Trainer.
type TrainerConfig struct {
	Epochs    int
	BatchSize int
	Device    string
	Seed      int64
}

// SetGlobalSeed фиксирует глобальный seed для воспроизводимости.
// Вызывается при создании Trainer из конфига, чтобы все операции,
// использующие math/rand (например, dropout), давали детерминированный результат.
func SetGlobalSeed(seed int64) {
	rand.Seed(seed)
}

// NewTrainerFromConfig создаёт Trainer через конфиг.
// Фиксирует seed глобально (SetGlobalSeed), затем создаёт Trainer с заданным числом эпох.
// BatchSize и Device заданы в конфиге для использования при создании DataLoader и устройств снаружи.
func NewTrainerFromConfig(
	cfg *TrainerConfig,
	model layers.Module,
	dataLoader dataloader.DataLoader,
	opt optimizers.Optimizer,
	lossFn autograd.LossOp,
	lrScheduler optimizers.LearningRateScheduler,
	metric metrics.Metric,
	callbacks *CallbackList,
) *Trainer {
	SetGlobalSeed(cfg.Seed)
	var cb CallbackList
	if callbacks != nil {
		cb = *callbacks
	}
	return NewTrainer(
		model,
		dataLoader,
		opt,
		lossFn,
		lrScheduler,
		metric,
		cb,
		cfg.Epochs,
	)
}
