package train

import (
	"fmt"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// Trainer - основной класс для обучения модели
// Содержит всю необходимую информацию для работы TrainLoop
type Trainer struct {
	model       layers.Module
	dataLoader  dataloader.DataLoader
	opt         optimizers.Optimizer
	lossFn      autograd.LossOp
	lrScheduler optimizers.LearningRateScheduler

	callbacks CallbackList

	context TrainingContext
}

// NewTrainer создает новый экземпляр Trainer
func NewTrainer(
	model layers.Module,
	dataLoader dataloader.DataLoader,
	opt optimizers.Optimizer,
	lossFn autograd.LossOp,
	lrScheduler optimizers.LearningRateScheduler,
	callbacks CallbackList,

	epochNumber int,
) *Trainer {
	return &Trainer{
		model:       model,
		dataLoader:  dataLoader,
		opt:         opt,
		lossFn:      lossFn,
		lrScheduler: lrScheduler,
		callbacks:   callbacks,
		context:     *NewTrainingContext(model, epochNumber),
	}
}

// Train содержит основной TrainLoop для обучения модели
func (t *Trainer) Train() {
	t.callbacks.OnTrainBegin(&t.context)
	for epoch := 0; epoch < t.context.NumEpochs; epoch++ {
		// Обновляем контекст для текущей эпохи и сбрасываем счётчик батчей
		t.resetContext(epoch)

		t.callbacks.OnEpochBegin(&t.context)

		t.dataLoader.Reset() // Сбрасываем итератор батчей

		// Train loop, завершается, когда закончатся батчи
		for {
			if !t.dataLoader.HasNext() {
				break
			}
			batch := t.dataLoader.Next()

			t.callbacks.OnBatchBegin(&t.context)

			err := t.processBatch(batch)
			if err != nil {
				continue
			}

			t.callbacks.OnBatchEnd(&t.context)
		}
		if t.context.Batch == 0 {
			fmt.Println("No batches processed.")
			break
		}
		lr := t.lrScheduler.Step()
		t.opt.SetLearningRate(lr)
		t.callbacks.OnEpochEnd(&t.context)
	}
	t.callbacks.OnTrainEnd(&t.context)
}

func (t *Trainer) resetContext(epoch int) {
	t.context.Epoch = epoch
	t.context.Batch = 0
}

func (t *Trainer) processBatch(batch *dataloader.Batch) error {
	// Берем входные данные
	input := batch.Features
	labels := batch.Targets

	n := graph.NewNode(input, nil, nil) // Засовываем в граф
	pred := t.model.Forward(n)          // Делаем Forward проход

	// Вычисляем потери (loss)
	lossVal := t.calculateLoss(pred, labels)

	// Рассчет Accuracy
	acc := metrics.NewAccuracy()
	err := acc.Update(pred, labels)
	if err != nil {
		return err
	}

	if t.context.Metrics == nil {
		t.context.Metrics = make(map[string]float64)
	}
	t.context.Metrics["loss"] = lossVal
	t.context.Metrics["accuracy"] = acc.Value()
	t.context.History.Append(t.context.Epoch, t.context.Metrics)
	return nil
}

func (t *Trainer) calculateLoss(pred *graph.Node, target *tensor.Tensor) float64 {
	var lossNode autograd.LossOp
	// TODO: проверить все ли функции потерь затронуты.
	switch t.lossFn.(type) {
	case *autograd.MSELossOp:
		lossNode = autograd.NewMSELossOp(pred, target)
	case *autograd.HingeLossOp:
		lossNode = autograd.NewHingeLossOp(pred, target)
	case *autograd.CrossEntropyLogitsOp:
		lossNode = autograd.NewCrossEntropyLogitsOp(pred, target)
	}
	if lossNode == nil {
		return 0
	}
	// Ну хоть убейте, не знаю как обнулить градиенды эффективнее :(
	// Умные люди перепишите с горутинами :)
	for _, layer := range t.model.Layers() {
		node := layer.Params()
		for _, node := range node {
			node.ZeroGrad()
		}
	}

	lossTensor := lossNode.Forward()
	var lossVal float64
	if lossTensor != nil && len(lossTensor.Data) > 0 {
		lossVal = lossTensor.Data[0]
	}

	lossNode.Backward(tensor.Ones(1))
	t.context.Batch++

	t.opt.Step(t.model.Params())

	return lossVal
}
