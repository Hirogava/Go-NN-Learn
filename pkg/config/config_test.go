package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAppConfig_YAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.yaml")
	content := `
model:
  name: "mlp-test"
  input_size: 16
  output_size: 2
  hidden_sizes: [8]
data:
  path: "./data/test"
  batch_size: 8
  shuffle: false
  drop_last: true
  seed: 7
training:
  lr: 0.05
  epochs: 3
  batch: 8
  seed: 7
  loss: "mse"
  metric: "mae"
checkpoint: "./ckpt/test.ckpt"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := LoadAppConfig(path)
	if err != nil {
		t.Fatalf("LoadAppConfig failed: %v", err)
	}
	if cfg.Model.Name != "mlp-test" {
		t.Fatalf("model.name mismatch: %v", cfg.Model.Name)
	}
	if cfg.Data.BatchSize != 8 {
		t.Fatalf("data.batch_size mismatch: %v", cfg.Data.BatchSize)
	}
	if cfg.Data.DropLast != true {
		t.Fatalf("data.drop_last mismatch: %v", cfg.Data.DropLast)
	}
	if cfg.Data.Seed != 7 {
		t.Fatalf("data.seed mismatch: %v", cfg.Data.Seed)
	}
	if cfg.Training.LR != 0.05 {
		t.Fatalf("training.lr mismatch: %v", cfg.Training.LR)
	}
	if cfg.Training.Loss != "mse" {
		t.Fatalf("training.loss mismatch: %v", cfg.Training.Loss)
	}
	if cfg.Training.Metric != "mae" {
		t.Fatalf("training.metric mismatch: %v", cfg.Training.Metric)
	}
	if cfg.Checkpoint != "./ckpt/test.ckpt" {
		t.Fatalf("checkpoint mismatch: %v", cfg.Checkpoint)
	}
}

func TestLoadAppConfig_DefaultsAndEnv(t *testing.T) {
	// set env overrides
	os.Setenv("GNN_LR", "0.123")
	os.Setenv("GNN_EPOCHS", "2")
	os.Setenv("GNN_BATCH", "16")
	os.Setenv("GNN_LOSS", "hinge")
	os.Setenv("GNN_METRIC", "accuracy")
	os.Setenv("GNN_SEED", "99")
	defer func() {
		os.Unsetenv("GNN_LR")
		os.Unsetenv("GNN_EPOCHS")
		os.Unsetenv("GNN_BATCH")
		os.Unsetenv("GNN_LOSS")
		os.Unsetenv("GNN_METRIC")
		os.Unsetenv("GNN_SEED")
	}()

	cfg, err := LoadAppConfig("")
	if err != nil {
		t.Fatalf("LoadAppConfig(default) failed: %v", err)
	}
	if cfg.Training.LR != 0.123 {
		t.Fatalf("env override lr failed: %v", cfg.Training.LR)
	}
	if cfg.Training.Epochs != 2 {
		t.Fatalf("env override epochs failed: %v", cfg.Training.Epochs)
	}
	if cfg.Data.BatchSize != 16 {
		t.Fatalf("env override batch failed: %v", cfg.Data.BatchSize)
	}
	if cfg.Training.Loss != "hinge" {
		t.Fatalf("env override loss failed: %v", cfg.Training.Loss)
	}
	if cfg.Training.Metric != "accuracy" {
		t.Fatalf("env override metric failed: %v", cfg.Training.Metric)
	}
	if cfg.Training.Seed != 99 {
		t.Fatalf("env override seed failed: %v", cfg.Training.Seed)
	}
}
