package text

import "sort"

type Vocab struct {
	TokenToIdx map[string]int `json:"token_to_idx"`
	IdxToToken []string       `json:"idx_to_token"`
}

func BuildVocab(allTokens [][]string, cfg PreprocessConfig) *Vocab {
	freq := make(map[string]int)

	for _, tokens := range allTokens {
		for _, t := range tokens {
			freq[t]++
		}
	}

	type pair struct {
		Token string
		Freq  int
	}

	pairs := make([]pair, 0, len(freq))

	for t, f := range freq {
		if f >= cfg.MinTokenFreq {
			pairs = append(pairs, pair{t, f})
		}
	}

	// детерминизм
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Freq == pairs[j].Freq {
			return pairs[i].Token < pairs[j].Token
		}
		return pairs[i].Freq > pairs[j].Freq
	})

	if cfg.MaxVocabSize > 0 && len(pairs) > cfg.MaxVocabSize {
		pairs = pairs[:cfg.MaxVocabSize]
	}

	vocab := &Vocab{
		TokenToIdx: make(map[string]int),
		IdxToToken: make([]string, 0, len(pairs)),
	}

	for i, p := range pairs {
		vocab.TokenToIdx[p.Token] = i
		vocab.IdxToToken = append(vocab.IdxToToken, p.Token)
	}

	return vocab
}
