package text

import (
	"reflect"
	"testing"
)

// 1. детерминизм vocab

func TestRunPipeline_DeterministicVocab(t *testing.T) {
	texts := []string{
		"Hello world",
		"world hello",
	}

	cfg := DefaultConfig()

	out1, err1 := RunPipeline(texts, cfg)
	out2, err2 := RunPipeline(texts, cfg)

	if err1 != nil || err2 != nil {
		t.Fatalf("unexpected error: %v %v", err1, err2)
	}

	if !reflect.DeepEqual(out1.Vocab, out2.Vocab) {
		t.Fatal("vocab is not deterministic")
	}
}

// 2. пустой датасет

func TestRunPipeline_EmptyDataset(t *testing.T) {
	_, err := RunPipeline([]string{}, DefaultConfig())

	if err == nil {
		t.Fatal("expected error for empty dataset")
	}
}

// 3. пустые строки

func TestRunPipeline_EmptyStrings(t *testing.T) {
	texts := []string{"", "   ", "\n"}

	out, err := RunPipeline(texts, DefaultConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, vec := range out.Features {
		for _, v := range vec {
			if v != 0 {
				t.Fatal("expected zero vector for empty input")
			}
		}
	}
}

// 4. только спецсимволы

func TestRunPipeline_SpecialCharsOnly(t *testing.T) {
	texts := []string{"!!!@@@", "###$$$"}

	out, err := RunPipeline(texts, DefaultConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(out.Vocab.IdxToToken) != 0 {
		t.Fatal("expected empty vocab")
	}
}

// 5. minTokenLen

func TestTokenize_MinTokenLen(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinTokenLen = 3

	tokens := Tokenize("a ab abc abcd", cfg)

	expected := []string{"abc", "abcd"}

	if !reflect.DeepEqual(tokens, expected) {
		t.Fatalf("got %v, want %v", tokens, expected)
	}
}

// 6. lowercase

func TestNormalize_Lowercase(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Lowercase = true

	out := Normalize("HeLLo", cfg)

	if out != "hello" {
		t.Fatalf("got %s, want hello", out)
	}
}

// 7. minTokenFreq

func TestBuildVocab_MinTokenFreq(t *testing.T) {
	tokens := [][]string{
		{"a", "b"},
		{"a"},
	}

	cfg := DefaultConfig()
	cfg.MinTokenFreq = 2

	vocab := BuildVocab(tokens, cfg)

	if len(vocab.IdxToToken) != 1 || vocab.IdxToToken[0] != "a" {
		t.Fatalf("unexpected vocab: %v", vocab.IdxToToken)
	}
}

// 8. maxVocabSize

func TestBuildVocab_MaxVocabSize(t *testing.T) {
	tokens := [][]string{
		{"a", "b", "c"},
		{"a", "b"},
	}

	cfg := DefaultConfig()
	cfg.MaxVocabSize = 1

	vocab := BuildVocab(tokens, cfg)

	if len(vocab.IdxToToken) != 1 {
		t.Fatalf("expected vocab size 1, got %d", len(vocab.IdxToToken))
	}
}

// 9. vectorizeBoW

func TestVectorizeBoW_Counts(t *testing.T) {
	vocab := &Vocab{
		TokenToIdx: map[string]int{
			"a": 0,
			"b": 1,
		},
		IdxToToken: []string{"a", "b"},
	}

	tokens := []string{"a", "a", "b"}

	vec := VectorizeBoW(tokens, vocab)

	expected := []float32{2, 1}

	if !reflect.DeepEqual(vec, expected) {
		t.Fatalf("got %v, want %v", vec, expected)
	}
}

// 10. OOV игнорируется

func TestVectorizeBoW_OOVIgnored(t *testing.T) {
	vocab := &Vocab{
		TokenToIdx: map[string]int{
			"a": 0,
		},
		IdxToToken: []string{"a"},
	}

	tokens := []string{"a", "b", "c"}

	vec := VectorizeBoW(tokens, vocab)

	expected := []float32{1}

	if !reflect.DeepEqual(vec, expected) {
		t.Fatalf("got %v, want %v", vec, expected)
	}
}

// 11. transform использует существующий vocab

func TestTransform_UsesExistingVocab(t *testing.T) {
	vocab := &Vocab{
		TokenToIdx: map[string]int{
			"hello": 0,
		},
		IdxToToken: []string{"hello"},
	}

	texts := []string{"hello world"}

	features := Transform(texts, vocab, DefaultConfig())

	if features[0][0] != 1 {
		t.Fatalf("expected 1, got %v", features[0][0])
	}
}

// 12. transform: пустые строки

func TestTransform_EmptyString(t *testing.T) {
	vocab := &Vocab{
		TokenToIdx: map[string]int{
			"a": 0,
		},
		IdxToToken: []string{"a"},
	}

	features := Transform([]string{""}, vocab, DefaultConfig())

	if features[0][0] != 0 {
		t.Fatal("expected zero vector")
	}
}

// 13. полный pipeline sanity check

func TestRunPipeline_Basic(t *testing.T) {
	texts := []string{
		"hello world",
		"hello",
	}

	out, err := RunPipeline(texts, DefaultConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(out.Features) != 2 {
		t.Fatal("wrong number of feature rows")
	}

	if len(out.Vocab.IdxToToken) == 0 {
		t.Fatal("vocab should not be empty")
	}
}
