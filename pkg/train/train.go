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

type Trainer struct {
	model       layers.Module
	dataLoader  dataloader.DataLoader
	opt         optimizers.Optimizer
	lossFn      autograd.LossOp
	lrScheduler optimizers.LearningRateScheduler
	epochs      int

	totalLoss  float64
	numBatches int
	// В будущем возможно добавление callback'ов
}

func NewTrainer(model layers.Module, dataLoader dataloader.DataLoader, opt optimizers.Optimizer, lossFn autograd.LossOp, lrScheduler optimizers.LearningRateScheduler, epochs int) *Trainer {
	return &Trainer{
		model:       model,
		dataLoader:  dataLoader,
		opt:         opt,
		lossFn:      lossFn,
		lrScheduler: lrScheduler,
		epochs:      epochs,
		totalLoss:   0,
		numBatches:  0,
	}
}

func (t *Trainer) Train() {
	for epoch := 0; epoch < t.epochs; epoch++ {
		t.dataLoader.Reset()

		var totalLoss float64
		var numBatches int

		// Train loop, завершается когда закончатся батчи
		for {
			if !t.dataLoader.HasNext() {
				break
			}
			batch := t.dataLoader.Next()
			// TODO: может вылезти паника

			// Берем входные данные
			input := batch.Features
			labels := batch.Targets

			n := graph.NewNode(input, nil, nil) // Засовываем в граф
			pred := t.model.Forward(n)          // Делаем Forward проход

			for _, layer := range t.model.Layers() {
				layer.Params()
			}

			// Вычисляем потери (loss)
			t.calculateLoss(pred, labels)

			// Рассчет Accuracy
			acc := metrics.NewAccuracy()
			err := acc.Update(pred, batch.Features)
			if err != nil {
				continue
			}
			fmt.Printf("Epoch %d, Accuracy: %v\n", epoch, acc.Value())
		}
		if t.numBatches == 0 {
			fmt.Println("No batches processed.")
			break
		}
		lr := t.lrScheduler.Step()
		t.opt.SetLearningRate(lr)
		// Тут дальше коллбэки и тд
	}
}

func (t *Trainer) calculateLoss(pred *graph.Node, target *tensor.Tensor) {
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

	// Ну хоть убейте, не знаю как обнулить градиенды эффективнее :(
	// Умные люди перепишите с горутинами :)
	for _, layer := range t.model.Layers() {
		node := layer.Params()
		for _, node := range node {
			node.ZeroGrad()
		}
	}

	lossNode.Backward(tensor.Ones(1))
	t.numBatches++

	t.opt.Step(t.model.Params())
}
