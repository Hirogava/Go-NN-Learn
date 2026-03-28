package api_test

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/api/text"
)

// 1. базовый Save/Load
func TestArtifact_SaveLoad(t *testing.T) {
	vocab := &text.Vocab{
		TokenToIdx: map[string]int{
			"hello": 0,
			"world": 1,
		},
		IdxToToken: []string{"hello", "world"},
	}

	meta := api.Metadata{
		Classes:   []string{"spam", "ham"},
		Vocab:     vocab,
		HiddenDim: 64,
	}

	weights := []byte{1, 2, 3, 4}

	path := "test_model.art"
	defer os.Remove(path)

	// save
	if err := api.SaveArtifact(path, meta, weights); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// load
	art, err := api.LoadArtifact(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// проверки
	if !reflect.DeepEqual(meta, art.Metadata) {
		t.Fatal("metadata mismatch")
	}

	if !reflect.DeepEqual(weights, art.Weights) {
		t.Fatal("weights mismatch")
	}
}

// 2. проверка version
func TestArtifact_VersionMismatch(t *testing.T) {
	// вручную создаём битый artifact
	art := api.Artifact{
		Version: "v999", // неправильная версия
		Metadata: api.Metadata{
			Classes:   []string{"a"},
			Vocab:     &text.Vocab{},
			HiddenDim: 1,
		},
		Weights: []byte{1},
	}

	data, _ := os.ReadFile(os.DevNull) // временно
	data, _ = json.Marshal(art)

	path := "bad_version.art"
	defer os.Remove(path)

	_ = os.WriteFile(path, data, 0644)

	_, err := api.LoadArtifact(path)
	if err == nil {
		t.Fatal("expected version mismatch error")
	}
}

// 3. пустые classes
func TestArtifact_EmptyClasses(t *testing.T) {
	meta := api.Metadata{
		Classes:   []string{},
		Vocab:     &text.Vocab{},
		HiddenDim: 10,
	}

	err := api.SaveArtifact("bad.art", meta, []byte{1})
	if err == nil {
		t.Fatal("expected error for empty classes")
	}
}

// 4. nil vocab
func TestArtifact_NilVocab(t *testing.T) {
	meta := api.Metadata{
		Classes:   []string{"a"},
		Vocab:     nil,
		HiddenDim: 10,
	}

	err := api.SaveArtifact("bad.art", meta, []byte{1})
	if err == nil {
		t.Fatal("expected error for nil vocab")
	}
}

// 5. пустые weights
func TestArtifact_EmptyWeights(t *testing.T) {
	vocab := &text.Vocab{
		TokenToIdx: map[string]int{"a": 0},
		IdxToToken: []string{"a"},
	}

	meta := api.Metadata{
		Classes:   []string{"a"},
		Vocab:     vocab,
		HiddenDim: 10,
	}

	path := "empty_weights.art"
	defer os.Remove(path)

	err := api.SaveArtifact(path, meta, []byte{})
	if err == nil {
		t.Fatal("expected error for empty weights")
	}
}

// 6. повреждённый JSON
func TestArtifact_CorruptedFile(t *testing.T) {
	path := "corrupted.art"
	defer os.Remove(path)

	_ = os.WriteFile(path, []byte("not a json"), 0644)

	_, err := api.LoadArtifact(path)
	if err == nil {
		t.Fatal("expected error for corrupted file")
	}
}

// 7. полный цикл
func TestArtifact_EndToEnd(t *testing.T) {
	vocab := &text.Vocab{
		TokenToIdx: map[string]int{
			"hello": 0,
		},
		IdxToToken: []string{"hello"},
	}

	meta := api.Metadata{
		Classes:   []string{"pos", "neg"},
		Vocab:     vocab,
		HiddenDim: 32,
	}

	weights := []byte{10, 20, 30}

	path := "full_cycle.art"
	defer os.Remove(path)

	// save
	if err := api.SaveArtifact(path, meta, weights); err != nil {
		t.Fatal(err)
	}

	// load
	art, err := api.LoadArtifact(path)
	if err != nil {
		t.Fatal(err)
	}

	// имитация восстановления модели
	if len(art.Metadata.Classes) != 2 {
		t.Fatal("classes lost")
	}

	if art.Metadata.Vocab.TokenToIdx["hello"] != 0 {
		t.Fatal("vocab corrupted")
	}
}
