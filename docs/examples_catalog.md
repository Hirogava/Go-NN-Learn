# Go-NN-Learn

Go-NN-Learn — учебная и исследовательская библиотека для построения и обучения нейросетей на языке Go. Проект ориентирован на прозрачность архитектуры, удобство расширения и практичные примеры использования.

## Быстрый старт

Клонирование и запуск тестов:

```bash
git clone https://github.com/Hirogava/Go-NN-Learn.git
cd Go-NN-Learn
go test ./... -v
```

Пример запуска (линейная регрессия):

```bash
go run ./cmd/examples/linear_regression --config ./cmd/examples/linear_regression/configs/linear.yaml
```

## Структура проекта

```
Go-NN-Learn/
├── cmd/examples/        # runnable примеры (linear_regression, mnist и т.д.)
├── pkg/
│   ├── api/             # публичные функции (Predict, Eval, Save/Load)
│   ├── layers/          # реализации слоёв и моделей
│   ├── optimizers/      # оптимизаторы и оболочки моделей
│   ├── tensor/          # тензоры и вычислительный граф
│   └── config/          # загрузчик конфигов (YAML/JSON + env)
├── docs/                # туториалы и индекс примеров
└── scripts/             # вспомогательные скрипты
```

## Основные API

* `api.SaveCheckpoint(m, path)` — сохранить параметры модели в файл;
* `api.LoadCheckpoint(m, path)` — загрузить параметры из файла;
* `api.Predict(m, x)` — выполнить прямой проход модели;
* `api.Eval(m, inputs, targets, metric)` — вычислить среднюю метрику на наборе данных;
* `optimizers.NewSequential(...)` — собрать последовательную модель (Sequential).

## Примеры и туториалы

Подробные руководства расположены в каталоге `docs/`:

* `docs/linear_regression.md` — пошаговый пример линейной регрессии;
* `docs/mnist.md` — подробный туториал по MNIST (загрузка, модель, обучение).

## Checkpointing

Формат чекпоинта представлен как:

```
[uint32 длина JSON][JSON метаданные][binary float64...]
```

Гарантируется атомарная запись через временный файл и переименование.

## Конфигурация

`pkg/config` поддерживает загрузку JSON/YAML и переопределение через переменные окружения (например `GNN_LR`, `GNN_EPOCHS`, `GNN_BATCH`, `GNN_DATA_PATH`).

## Тесты и CI

* Unit tests: `go test ./...`;
* Примеры (GoDoc Example): `go test ./pkg/... -run Example -v`;
* Benchmarks: `go test ./internal/backend -bench . -benchmem`.

## Контрибьютинг

1. Форкните репозиторий;
2. Сделайте ветку для фичи/исправления;
3. Добавьте тесты и/или примеры;
4. Создайте PR с описанием изменений.

---

Если хотите, могу: сгенерировать git-patch со всеми файлами документации, вставить runnable примеры в `cmd/examples` или добавить CI workflow для автоматического запуска примеров.
