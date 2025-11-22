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

// SaveCheckpoint сохраняет все параметры из модуля m в файл path.
// Формат файла:
//   [uint32 metaLen][metaJSON][binary float64...]
//
// JSON-метаданные metaJSON содержат поле "version" и массив "params", где
// для каждого параметра записана его форма (shape). Параметры записываются
// последовательно в том же порядке, как возвращает m.Params().
//
// SaveCheckpoint выполняет запись атомарно: сначала создаётся временный
// файл в той же директории, выполняется запись и fsync (через Close),
// затем временный файл переименовывается в path.
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

// LoadCheckpoint загружает параметры из файла path и записывает их обратно
// в параметры модуля m (в том же порядке, как m.Params()). Требования:
//   - количество параметров в чекпоинте должно совпадать с len(m.Params())
//   - формы (shape) параметров в чекпоинте и в модели должны совпадать.
//
// В случае несовпадения формы или числа параметров функция вернёт ошибку.
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
