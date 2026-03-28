package text

func VectorizeBoW(tokens []string, vocab *Vocab) []float32 {
	vec := make([]float32, len(vocab.IdxToToken))

	for _, t := range tokens {
		if idx, ok := vocab.TokenToIdx[t]; ok {
			vec[idx]++
		}
		// OOV -> игнор
	}

	return vec
}
