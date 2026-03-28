package text

import "strings"

func Normalize(text string, cfg PreprocessConfig) string {
	if cfg.Lowercase {
		text = strings.ToLower(text)
	}
	return text
}
