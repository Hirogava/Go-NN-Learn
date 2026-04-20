package api

import (
	"encoding/json"
	"errors"
	"os"

	"github.com/Hirogava/Go-NN-Learn/pkg/api/text"
)

const CurrentArtifactVersion = "v1"

// ошибки
var (
	ErrInvalidArtifact  = errors.New("invalid artifact")
	ErrArtifactMismatch = errors.New("artifact version mismatch")
)

// главный объект (файл модели)
type Artifact struct {
	Version  string   `json:"version"`
	Metadata Metadata `json:"metadata"`
	Weights  []byte   `json:"weights"`
}

// всё что нужно для восстановления модели
type Metadata struct {
	Classes   []string              `json:"classes"`
	Vocab     *text.Vocab           `json:"vocab"`
	HiddenDim int                   `json:"hidden_dim"`
	TextCfg   text.PreprocessConfig `json:"text_cfg"` // новое
}

// save

func SaveArtifact(path string, meta Metadata, weights []byte) error {
	if len(meta.Classes) == 0 {
		return ErrInvalidArtifact
	}
	if meta.Vocab == nil {
		return ErrInvalidArtifact
	}
	if len(weights) == 0 {
		return ErrInvalidArtifact
	}

	art := Artifact{
		Version:  CurrentArtifactVersion,
		Metadata: meta,
		Weights:  weights,
	}

	data, err := json.Marshal(art)
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// load

func LoadArtifact(path string) (*Artifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var art Artifact
	if err := json.Unmarshal(data, &art); err != nil {
		return nil, err
	}

	if err := validateArtifact(&art); err != nil {
		return nil, err
	}

	return &art, nil
}

// validation

func validateArtifact(a *Artifact) error {
	if a.Version != CurrentArtifactVersion {
		return ErrArtifactMismatch
	}

	if a.Metadata.Vocab == nil {
		return ErrInvalidArtifact
	}

	if len(a.Metadata.Classes) == 0 {
		return ErrInvalidArtifact
	}

	if len(a.Weights) == 0 {
		return ErrInvalidArtifact
	}

	return nil
}
