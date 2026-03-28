package text

import "regexp"

// убираем спецсимволы
var nonAlphaNum = regexp.MustCompile(`[^a-zA-Z0-9]+`)

func Tokenize(text string, cfg PreprocessConfig) []string {
	tokens := nonAlphaNum.Split(text, -1)

	result := make([]string, 0, len(tokens))

	for _, t := range tokens {
		if len(t) >= cfg.MinTokenLen {
			result = append(result, t)
		}
	}

	return result
}
