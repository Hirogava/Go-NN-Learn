package train

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// TrainState хранит состояние обучения
type TrainState struct {
	CurrentEpoch int     `json:"current_epoch"` // Текущая эпоха обучения
	BestMetric   float64 `json:"best_metric"`   // Лучшая достигнутая метрика (например, accuracy или -loss для минимизации)
	AverageLoss  float64 `json:"average_loss"`  // Средний лосс за последнюю эпоху
}

// SaveTrainState сохраняет состояние обучения в файл
func SaveTrainState(state *TrainState, checkpointPath string) error {
	statePath := filepath.Join(checkpointPath, ".state.json")
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(statePath, data, 0644)
}

// LoadTrainState загружает состояние обучения из файла
func LoadTrainState(checkpointPath string) (*TrainState, error) {
	statePath := filepath.Join(checkpointPath, ".state.json")
	data, err := os.ReadFile(statePath)
	if os.IsNotExist(err) {
		return nil, nil // Нет файла
	}
	if err != nil {
		return nil, err
	}
	var state TrainState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}
	return &state, nil
}
