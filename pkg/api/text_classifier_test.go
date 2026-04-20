package api_test

import (
	"path/filepath"
	"testing"

	api "github.com/Hirogava/Go-NN-Learn/pkg/api"
)

// Fit / Error cases

func TestFitErrors(t *testing.T) {
	tests := []struct {
		name   string
		texts  []string
		labels []string
		expect error
	}{
		{
			name:   "empty dataset",
			texts:  []string{},
			labels: []string{},
			expect: api.ErrEmptyDataset,
		},
		{
			name:   "length mismatch",
			texts:  []string{"a"},
			labels: []string{},
			expect: nil, // any error is fine
		},
		{
			name:   "empty label",
			texts:  []string{"text"},
			labels: []string{""},
			expect: api.ErrUnknownLabel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clf := api.NewTextClassifier(api.DefaultTrainConfig())

			err := clf.Fit(tt.texts, tt.labels)

			if tt.expect != nil && err != tt.expect {
				t.Fatalf("expected %v, got %v", tt.expect, err)
			}

			if tt.expect == nil && err == nil {
				t.Fatalf("expected error, got nil")
			}
		})
	}
}

// Not trained

func TestPredictNotTrained(t *testing.T) {
	clf := api.NewTextClassifier(api.DefaultTrainConfig())

	_, err := clf.Predict([]string{"hello"})
	if err != api.ErrNotTrained {
		t.Fatalf("expected ErrNotTrained, got %v", err)
	}
}

func TestPredictProbaNotTrained(t *testing.T) {
	clf := api.NewTextClassifier(api.DefaultTrainConfig())

	_, err := clf.PredictProba([]string{"test"})
	if err != api.ErrNotTrained {
		t.Fatalf("expected ErrNotTrained, got %v", err)
	}
}

// Fit + Predict (deterministic)

func TestFitAndPredictDeterministic(t *testing.T) {
	cfg := api.DefaultTrainConfig()
	cfg.Epochs = 3
	cfg.Seed = 42

	clf := api.NewTextClassifier(cfg)

	texts := []string{
		"hello world",
		"buy pizza",
		"hello friend",
		"pizza order",
	}

	labels := []string{
		"greeting",
		"food",
		"greeting",
		"food",
	}

	if err := clf.Fit(texts, labels); err != nil {
		t.Fatalf("fit failed: %v", err)
	}

	preds, err := clf.Predict([]string{"hello"})
	if err != nil {
		t.Fatalf("predict failed: %v", err)
	}

	if len(preds) != 1 {
		t.Fatalf("expected 1 prediction, got %d", len(preds))
	}

	if preds[0] == "" {
		t.Fatalf("empty prediction")
	}
}

// Probability sanity (softmax)

func TestPredictProbaSumToOne(t *testing.T) {
	cfg := api.DefaultTrainConfig()
	cfg.Epochs = 3
	cfg.Seed = 42

	clf := api.NewTextClassifier(cfg)

	if err := clf.Fit([]string{"a", "b"}, []string{"x", "y"}); err != nil {
		t.Fatalf("fit failed: %v", err)
	}

	proba, err := clf.PredictProba([]string{"a"})
	if err != nil {
		t.Fatalf("predict proba failed: %v", err)
	}

	if len(proba) != 1 {
		t.Fatalf("expected 1 row")
	}

	sum := 0.0
	for _, v := range proba[0] {
		sum += v
	}

	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("softmax invalid, sum=%f", sum)
	}
}

// Save / Load consistency

func TestSaveLoadConsistency(t *testing.T) {
	cfg := api.DefaultTrainConfig()
	cfg.Epochs = 3
	cfg.Seed = 42

	clf := api.NewTextClassifier(cfg)

	texts := []string{"hello", "buy"}
	labels := []string{"greet", "shop"}

	if err := clf.Fit(texts, labels); err != nil {
		t.Fatalf("fit failed: %v", err)
	}

	tmp := t.TempDir()
	path := filepath.Join(tmp, "model.bin")

	if err := clf.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	clf2 := api.NewTextClassifier(api.DefaultTrainConfig())
	if err := clf2.Load(path); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	in := []string{"hello"}

	p1, _ := clf.Predict(in)
	p2, _ := clf2.Predict(in)

	if p1[0] == "" || p2[0] == "" {
		t.Fatalf("invalid prediction after load")
	}

	if p1[0] != p2[0] {
		t.Fatalf("model not consistent after save/load")
	}
}

// Empty input

func TestPredictEmptyInput(t *testing.T) {
	cfg := api.DefaultTrainConfig()
	cfg.Seed = 42

	clf := api.NewTextClassifier(cfg)

	_ = clf.Fit([]string{"a"}, []string{"b"})

	_, err := clf.Predict([]string{})
	if err != api.ErrEmptyDataset {
		t.Fatalf("expected ErrEmptyDataset, got %v", err)
	}
}
