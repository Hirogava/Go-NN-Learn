# MNIST — практический туториал

## Цель

Показать процесс обучения модели для задачи классификации изображений: загрузка данных, предобработка, модель, обучение, валидация и инференс.

---

## Структура примера

Рекомендуемое расположение:

```
cmd/examples/mnist/
  main.go
  configs/mnist.yaml
  data/         # (опционально) хранилище IDX или CSV
  README.md
```

---

## Конфигурация (пример)

```yaml
model:
  name: "mnist_cnn"
  input_size: 784
  output_size: 10
data:
  path: "./data/mnist"
  batch_size: 64
  shuffle: true
training:
  lr: 0.001
  epochs: 20
  batch: 64
  seed: 12345
checkpoint: "./checkpoints/mnist.ckpt"
```

---

## Подготовка данных

1. Загрузка IDX или CSV формата;
2. Нормализация пикселей (деление на 255.0);
3. Формирование батчей и перемешивание;
4. Разделение на train/validation.

---

## Архитектура модели (рекомендуемая)

### Пример CNN

```
Conv2D(1->16, k=3) -> ReLU -> MaxPool(2)
Conv2D(16->32, k=3) -> ReLU -> MaxPool(2)
Flatten -> Dense(512) -> ReLU -> Dense(10) -> Softmax
```

### Пример MLP (альтернатива)

```
Dense(784 -> 128) -> ReLU -> Dense(128 -> 10) -> Softmax
```

---

## Тренировочный цикл (псевдокод)

```go
cfg, _ := config.LoadAppConfig("configs/mnist.yaml")
model := buildMnistModel(cfg.Model)
opt := optimizers.NewAdam(cfg.Training.LR)
trainLoader := NewDataLoader(cfg.Data.Path, cfg.Data.BatchSize, true)
valLoader := NewDataLoader(cfg.Data.Path, cfg.Data.BatchSize, false)

for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
    for batch := range trainLoader.Batches() {
        x, y := batch.X, batch.Y
        pred := model.Forward(x)
        loss := CrossEntropy(pred, y)
        model.ZeroGrad()
        loss.Backward()
        opt.Step(model.Params())
    }
    acc := EvalAccuracy(model, valLoader)
    fmt.Printf("Epoch %d: val acc=%.4f\n", epoch, acc)
    api.SaveCheckpoint(model, cfg.Checkpoint)
}
```

---

## Инференс

```go
img := loadImageAsTensor("path/to/image.png")
out := api.Predict(model, img)
pred := ArgMax(out.Value.Data)
fmt.Println("predicted:", pred)
```

---

## Рекомендации

* Batch size: 32–128 в зависимости от доступной памяти;
* LR: начните с 1e-3 и подбирайте;
* Используйте валидацию и early stopping;
* При использовании BatchNorm/Dropout не забывайте переключать режим в eval для инференса.
