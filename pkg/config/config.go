package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// AppConfig собирает основные настройки приложения
type AppConfig struct {
	// Model — параметры модели (простая декларация)
	Model ModelConfig `json:"model" yaml:"model"`

	// Data — настройки источника данных
	Data DataConfig `json:"data" yaml:"data"`

	// Training — параметры обучения
	Training TrainingConfig `json:"training" yaml:"training"`

	// Checkpoint path (где сохранять/загружать модель)
	Checkpoint string `json:"checkpoint" yaml:"checkpoint"`
}

// ModelConfig задаёт пример конфигурации модели
type ModelConfig struct {
	// Name — человекочитаемое имя модели, например "mlp" или "convnet"
	Name string `json:"name" yaml:"name"`

	// InputSize/OutputSize — чаще всего полезно указывать
	InputSize  int `json:"input_size" yaml:"input_size"`
	OutputSize int `json:"output_size" yaml:"output_size"`

	// HiddenSizes — список скрытых слоёв
	HiddenSizes []int `json:"hidden_sizes" yaml:"hidden_sizes"`
}

// DataConfig описывает где брать и как готовить данные
type DataConfig struct {
	// Path к датасету (файл или папка)
	Path string `json:"path" yaml:"path"`

	// BatchSize
	BatchSize int `json:"batch_size" yaml:"batch_size"`

	// Shuffle включить/выключить
	Shuffle bool `json:"shuffle" yaml:"shuffle"`
}

// TrainingConfig содержит параметры обучения.
type TrainingConfig struct {
	LR     float64 `json:"lr" yaml:"lr"`         // скорость обучения
	Epochs int     `json:"epochs" yaml:"epochs"` // кол-во "эпох" (epochs - число полных проходов по всему набору обучения)
	Batch  int     `json:"batch" yaml:"batch"`   // число примеров, которые модель обрабатывает за одну итерацию
	Seed   int64   `json:"seed" yaml:"seed"`     // начальное число генератора случайных чисел
}

// DefaultAppConfig возвращает конфигурацию с безопасными значениями по умолчанию.
func DefaultAppConfig() AppConfig {
	return AppConfig{
		Model: ModelConfig{
			Name:        "mlp",
			InputSize:   128,
			OutputSize:  10,
			HiddenSizes: []int{64, 32},
		},
		Data: DataConfig{
			Path:      "./data",
			BatchSize: 32,
			Shuffle:   true,
		},
		Training: TrainingConfig{
			LR:     0.01,
			Epochs: 10,
			Batch:  32,
			Seed:   42,
		},
		Checkpoint: "./checkpoints/model.ckpt",
	}
}

// LoadConfig читает конфигурацию из файла path и распарсивает её в out
// Поддерживаются JSON (.json) и YAML (.yaml, .yml)
// Если расширение неизвестно, функция сначала пробует JSON, затем YAML
func LoadConfig(path string, out interface{}) error {
	if path == "" {
		return errors.New("LoadConfig: empty path")
	}
	bs, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("LoadConfig: read file: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".json":
		if err := json.Unmarshal(bs, out); err != nil {
			return fmt.Errorf("LoadConfig: json unmarshal: %w", err)
		}
		return nil
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(bs, out); err != nil {
			return fmt.Errorf("LoadConfig: yaml unmarshal: %w", err)
		}
		return nil
	default:
		// пробуем вызвать JSON
		if err := json.Unmarshal(bs, out); err == nil {
			return nil
		}
		// пробуем вызвать YAML
		if err := yaml.Unmarshal(bs, out); err == nil {
			return nil
		}
		return fmt.Errorf("LoadConfig: unsupported format and parsing failed (json/yaml tried)")
	}
}

// LoadAppConfig загружает AppConfig из файла, заполняет значения по умолчанию
// и позволяет переопределять параметры с помощью переменных окружения
// Возвращает валидированную конфигурацию
func LoadAppConfig(path string) (AppConfig, error) {
	// по умолчанию
	cfg := DefaultAppConfig()

	// если пусто возвращаем значения по умолчанию
	if path == "" {
		applyEnvOverrides(&cfg)
		if err := cfg.Validate(); err != nil {
			return cfg, err
		}
		return cfg, nil
	}

	// переопределяем значения
	if err := LoadConfig(path, &cfg); err != nil {
		return cfg, err
	}

	// переопределяем отдельные поля значениями из переменных окружения (если они заданы)
	applyEnvOverrides(&cfg)

	// валидация
	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// Validate делает базовую валидацию конфигурации и возвращает ошибку, если что-то не так
func (c *AppConfig) Validate() error {
	if c.Model.InputSize <= 0 {
		return errors.New("Model.InputSize must be > 0")
	}
	if c.Model.OutputSize <= 0 {
		return errors.New("Model.OutputSize must be > 0")
	}
	if c.Data.BatchSize <= 0 {
		// allow fallback to training.Batch if set
		if c.Training.Batch > 0 {
			c.Data.BatchSize = c.Training.Batch
		} else {
			return errors.New("Data.BatchSize must be > 0")
		}
	}
	if c.Training.Epochs <= 0 {
		return errors.New("Training.Epochs must be > 0")
	}
	if c.Training.LR <= 0 {
		return errors.New("Training.LR must be > 0")
	}
	if strings.TrimSpace(c.Data.Path) == "" {
		return errors.New("Data.Path must be set")
	}
	return nil
}

// applyEnvOverrides — примитивный механизм переопределения конфигурации через env vars
// Поддерживаем несколько стандартных переменных:
//   GNNCHECKPOINT, GNN_LR, GNN_EPOCHS, GNN_BATCH, GNN_DATA_PATH
func applyEnvOverrides(c *AppConfig) {
	if v := os.Getenv("GNN_CHECKPOINT"); v != "" {
		c.Checkpoint = v
	}
	if v := os.Getenv("GNN_DATA_PATH"); v != "" {
		c.Data.Path = v
	}
	if v := os.Getenv("GNN_LR"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			c.Training.LR = f
		}
	}
	if v := os.Getenv("GNN_EPOCHS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			c.Training.Epochs = i
		}
	}
	if v := os.Getenv("GNN_BATCH"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			c.Training.Batch = i
			c.Data.BatchSize = i
		}
	}
	if v := os.Getenv("GNN_SEED"); v != "" {
		if s, err := strconv.ParseInt(v, 10, 64); err == nil {
			c.Training.Seed = s
		}
	}
}
