package api

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
)

type paramMeta struct {
	Shape []int `json:"shape"`
}

type checkpointMeta struct {
	Version int         `json:"version"`
	Params  []paramMeta `json:"params"`
}

// SaveCheckpoint сохраняет все параметры из модуля m в один путь к файлу.
// Формат: [uint32 metaLen][metaJSON][двоичный float64...]
// Meta содержит версию и формы (для каждого параметра в порядке m.Params()).
func SaveCheckpoint(m layers.Module, path string) error {
	params := m.Params()
	meta := checkpointMeta{Version: 1, Params: make([]paramMeta, len(params))}

	for i, p := range params {
		if p == nil || p.Value == nil {
			return fmt.Errorf("param %d is nil", i)
		}
		meta.Params[i] = paramMeta{Shape: append([]int(nil), p.Value.Shape...)}
	}

	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	tmp := filepath.Join(dir, ".tmp_checkpoint")
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer f.Close()

	// длина метаданных
	if len(metaBytes) > (1 << 31) {
		return errors.New("meta too large")
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(len(metaBytes))); err != nil {
		return err
	}
	if _, err := f.Write(metaBytes); err != nil {
		return err
	}

	// запись данных по порядку
	for _, p := range params {
		if p.Value == nil || p.Value.Data == nil {
			return errors.New("nil tensor data")
		}
		for _, v := range p.Value.Data {
			if err := binary.Write(f, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}

	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadCheckpoint загружает параметры из файла и записывает их в модуль m
// Для этого требуется, чтобы len(meta.Params) == len(m.Params()) и shapes совпадали.
func LoadCheckpoint(m layers.Module, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var metaLen uint32
	if err := binary.Read(f, binary.LittleEndian, &metaLen); err != nil {
		return err
	}
	metaBytes := make([]byte, metaLen)
	if _, err := f.Read(metaBytes); err != nil {
		return err
	}
	var meta checkpointMeta
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return err
	}

	params := m.Params()
	if len(meta.Params) != len(params) {
		return fmt.Errorf("params count mismatch: checkpoint=%d model=%d", len(meta.Params), len(params))
	}

	for i, pm := range meta.Params {
		count := 1
		for _, d := range pm.Shape {
			count *= d
		}
		buf := make([]float64, count)
		for j := 0; j < count; j++ {
			if err := binary.Read(f, binary.LittleEndian, &buf[j]); err != nil {
				return err
			}
		}
		// проверка совместимости форм
		target := params[i].Value
		if len(target.Shape) != len(pm.Shape) {
			return fmt.Errorf("shape rank mismatch for param %d: ckpt=%v model=%v", i, pm.Shape, target.Shape)
		}
		for k := range pm.Shape {
			if pm.Shape[k] != target.Shape[k] {
				return fmt.Errorf("shape mismatch for param %d: ckpt=%v model=%v", i, pm.Shape, target.Shape)
			}
		}
		target.Data = make([]float64, count)
		copy(target.Data, buf)
	}
	return nil
}
