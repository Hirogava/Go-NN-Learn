package text

import (
	"errors"
	"strings"
)

var ErrEmptyDataset = errors.New("empty dataset")

func RunPipeline(texts []string, cfg PreprocessConfig) (*Output, error) {
	if len(texts) == 0 {
		return nil, ErrEmptyDataset
	}

	allTokens := make([][]string, 0, len(texts))

	for _, t := range texts {
		if strings.TrimSpace(t) == "" {
			allTokens = append(allTokens, []string{})
			continue
		}

		norm := Normalize(t, cfg)
		tokens := Tokenize(norm, cfg)

		allTokens = append(allTokens, tokens)
	}

	vocab := BuildVocab(allTokens, cfg)

	features := make([][]float32, 0, len(allTokens))

	for _, tokens := range allTokens {
		vec := VectorizeBoW(tokens, vocab)
		features = append(features, vec)
	}

	return &Output{
		Vocab:    vocab,
		Features: features,
	}, nil
}

// Inference pipeline
func Transform(texts []string, vocab *Vocab, cfg PreprocessConfig) [][]float32 {
	features := make([][]float32, 0, len(texts))

	for _, t := range texts {
		if strings.TrimSpace(t) == "" {
			features = append(features, make([]float32, len(vocab.IdxToToken)))
			continue
		}

		norm := Normalize(t, cfg)
		tokens := Tokenize(norm, cfg)

		vec := VectorizeBoW(tokens, vocab)
		features = append(features, vec)
	}

	return features
}
