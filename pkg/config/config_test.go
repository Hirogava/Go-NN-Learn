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
training:
  lr: 0.05
  epochs: 3
  batch: 8
  seed: 7
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
	if cfg.Training.LR != 0.05 {
		t.Fatalf("training.lr mismatch: %v", cfg.Training.LR)
	}
	if cfg.Checkpoint != "./ckpt/test.ckpt" {
		t.Fatalf("checkpoint mismatch: %v", cfg.Checkpoint)
	}
}

func TestLoadAppConfig_DefaultsAndEnv(t *testing.T) {
	// ensure defaults are applied when path == ""
	os.Setenv("GNN_LR", "0.123")
	os.Setenv("GNN_EPOCHS", "2")
	os.Setenv("GNN_BATCH", "16")
	defer os.Unsetenv("GNN_LR")
	defer os.Unsetenv("GNN_EPOCHS")
	defer os.Unsetenv("GNN_BATCH")

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
}
