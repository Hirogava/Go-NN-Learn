# Linear Regression — пример

## Цель

Показать минимальный end-to-end пример: генерация данных, обучение простой модели, сохранение чекпоинта и инференс.

---

## Структура примера

Рекомендуемая структура в репозитории:

```
cmd/examples/linear_regression/
  main.go
  configs/linear.yaml
  README.md
```

---

## Конфигурация (пример)

`configs/linear.yaml`:

```yaml
model:
  name: "linear"
  input_size: 1
  output_size: 1
data:
  batch_size: 16
  shuffle: true
training:
  lr: 0.05
  epochs: 50
  batch: 16
  seed: 42
checkpoint: "./checkpoints/linear.ckpt"
```

---

## Генерация данных

Генерируем 1D-данные по формуле `y = a*x + b + noise`.

Пример функции генерации:

```go
func makeDataset(n int, a, b float64, seed int64) ([]float64, []float64) {
    r := rand.New(rand.NewSource(seed))
    xs := make([]float64, n)
    ys := make([]float64, n)
    for i := 0; i < n; i++ {
        x := r.NormFloat64() * 10.0
        y := a*x + b + r.NormFloat64()*0.1
        xs[i], ys[i] = x, y
    }
    return xs, ys
}
```

---

## Модель

Простейшая модель: `Sequential(Dense(1,1))`.

```go
model := layers.NewSequential(layers.NewDense(1, 1, nil))
```

---

## Тренировка (псевдокод)

```go
cfg, _ := config.LoadAppConfig("configs/linear.yaml")
model := layers.NewSequential(layers.NewDense(1, 1, nil))
opt := optimizers.NewSGD(cfg.Training.LR)
xs, ys := makeDataset(1000, 2.0, -1.0, cfg.Training.Seed)
batchSize := cfg.Data.BatchSize

for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
    for i := 0; i < len(xs); i += batchSize {
        // подготовка батча
        // forward -> loss -> backward -> opt.Step
    }
    // валидация / лог
}

api.SaveCheckpoint(model, cfg.Checkpoint)
```

---

## Инференс

```go
loaded := layers.NewSequential(layers.NewDense(1,1,nil))
api.LoadCheckpoint(loaded, cfg.Checkpoint)
out := api.Predict(loaded, graph.NewNodeFromSlice([]float64{3.14}))
fmt.Println(out.Value.Data[0])
```

---

## Советы

* Начинайте с небольшого набора данных для отладки;
* Подбирайте скорость обучения (lr) и batch size;
* Используйте фиксированный seed для воспроизводимости.
