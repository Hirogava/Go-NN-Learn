package text

type PreprocessConfig struct {
	Lowercase    bool
	MinTokenLen  int
	MinTokenFreq int
	MaxVocabSize int
}

func DefaultConfig() PreprocessConfig {
	return PreprocessConfig{
		Lowercase:    true,
		MinTokenLen:  2,
		MinTokenFreq: 1,
		MaxVocabSize: 0, // без лимита
	}
}
