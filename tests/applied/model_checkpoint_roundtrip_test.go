// tests/applied/model_checkpoint_roundtrip_test.go
package applied_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

const eps = 1e-9

// deterministic weight init
func initWeights(seed float64) func([]float64) {
	return func(w []float64) {
		for i := range w {
			w[i] = seed + float64(i)*0.001
		}
	}
}

// minimal ReLU layer wrapper (stateless)
type reluLayer struct{}

func (r *reluLayer) Forward(x *graph.Node) *graph.Node {
	in := x.Value
	out := make([]float64, len(in.Data))
	for i, v := range in.Data {
		if v > 0 {
			out[i] = v
		} else {
			out[i] = 0
		}
	}
	return &graph.Node{
		Value: &tensor.Tensor{
			Data:  out,
			Shape: append([]int(nil), in.Shape...),
		},
	}
}
func (r *reluLayer) Params() []*graph.Node { return nil }
func (r *reluLayer) Train()                {}
func (r *reluLayer) Eval()                 {}

// simple MLP: Dense -> ReLU -> Dense
type simpleModel struct {
	l1 *layers.Dense
	r  *reluLayer
	l2 *layers.Dense
}

func newModel(init func([]float64)) *simpleModel {
	return &simpleModel{
		l1: layers.NewDense(4, 8, init),
		r:  &reluLayer{},
		l2: layers.NewDense(8, 1, init),
	}
}

func (m *simpleModel) Layers() []layers.Layer {
	return []layers.Layer{m.l1, m.r, m.l2}
}

func (m *simpleModel) Params() []*graph.Node {
	p := []*graph.Node{}
	p = append(p, m.l1.Params()...)
	p = append(p, m.l2.Params()...)
	return p
}

func (m *simpleModel) Train() {}
func (m *simpleModel) Eval()  {}

func (m *simpleModel) Forward(x *graph.Node) *graph.Node {
	x1 := m.l1.Forward(x)
	x2 := m.r.Forward(x1)
	x3 := m.l2.Forward(x2)
	return x3
}

func TestModelCheckpointRoundtrip(t *testing.T) {
	t.Log("=== Начало теста: проверка roundtrip чекпойнта ===")
	t.Log("=== Start test: model checkpoint roundtrip ===")

	// 1) создаём и инициализируем модель A
	modelA := newModel(initWeights(0.1))
	t.Log("Модель A создана и инициализирована детерминированно / Model A created and deterministically initialized")

	// 2) фиксированный батч входов (например реальные фичи)
	xData := []float64{
		0.2, 0.1, 0.4, 0.7,
		0.3, 0.5, 0.9, 0.2,
		0.8, 0.2, 0.3, 0.1,
	}
	x := &tensor.Tensor{
		Data:  xData,
		Shape: []int{3, 4}, // batch 3, dim 4
	}
	input := graph.NewNode(x, nil, nil)
	t.Log("Формирован фиксированный входной батч / Fixed input batch prepared")

	// 3) forward через модель A
	outA := modelA.Forward(input)
	if outA == nil || outA.Value == nil {
		t.Fatalf("Ошибка: modelA.Forward вернул nil / Error: modelA.Forward returned nil")
	}
	predA := append([]float64(nil), outA.Value.Data...) // копия
	t.Logf("Предсказания до сохранения: %v / Predictions before save: %v", predA, predA)

	// 4) сохраняем checkpoint
	tmpDir, err := os.MkdirTemp("", "checkpoint_test")
	if err != nil {
		t.Fatalf("Не удалось создать временную директорию: %v / Failed to create temp dir: %v", err, err)
	}
	defer os.RemoveAll(tmpDir)

	ckptPath := filepath.Join(tmpDir, "model.ckpt")
	if err := api.SaveCheckpoint(modelA, ckptPath); err != nil {
		t.Fatalf("SaveCheckpoint завершился с ошибкой: %v / SaveCheckpoint failed: %v", err, err)
	}
	t.Logf("Чекпойнт сохранён: %s / Checkpoint saved: %s", ckptPath, ckptPath)

	// 5) создаём новую модель B (как после перезапуска сервиса) с другой инициализацией
	modelB := newModel(initWeights(99.0))
	t.Log("Модель B (новая инстанция) создана с отличной инициализацией / Model B (new instance) created with different init")

	// 6) загружаем checkpoint в модель B
	if err := api.LoadCheckpoint(modelB, ckptPath); err != nil {
		t.Fatalf("LoadCheckpoint завершился с ошибкой: %v / LoadCheckpoint failed: %v", err, err)
	}
	t.Log("Чекпойнт загружен в модель B / Checkpoint loaded into Model B")

	// 7) сравниваем параметры A и B
	pA := modelA.Params()
	pB := modelB.Params()
	if len(pA) != len(pB) {
		t.Fatalf("Количество параметров не совпадает: %d vs %d / Param count mismatch: %d vs %d", len(pA), len(pB), len(pA), len(pB))
	}

	for i := range pA {
		a := pA[i].Value.Data
		b := pB[i].Value.Data
		if len(a) != len(b) {
			t.Fatalf("Размер параметра %d не совпадает: %d vs %d / Param %d length mismatch: %d vs %d", i, len(a), len(b), i, len(a), len(b))
		}
		for j := range a {
			diff := a[j] - b[j]
			if diff < -eps || diff > eps {
				t.Fatalf("Параметр несовпадает: слой %d индекс %d (diff=%.12f) / Param mismatch: layer %d index %d (diff=%.12f)", i, j, diff, i, j, diff)
			}
		}
	}
	t.Log("Параметры модели после загрузки совпадают с исходными / Model parameters after load match the originals")

	// 8) проверяем предсказания после загрузки
	outB := modelB.Forward(graph.NewNode(x, nil, nil))
	if outB == nil || outB.Value == nil {
		t.Fatalf("Ошибка: modelB.Forward вернул nil / Error: modelB.Forward returned nil")
	}
	predB := outB.Value.Data
	t.Logf("Предсказания после загрузки: %v / Predictions after load: %v", predB, predB)

	if len(predA) != len(predB) {
		t.Fatalf("Длина предсказаний не совпадает: %d vs %d / Prediction length mismatch: %d vs %d", len(predA), len(predB), len(predA), len(predB))
	}
	for i := range predA {
		diff := predA[i] - predB[i]
		if diff < -eps || diff > eps {
			t.Fatalf("Предсказание несовпадает для индекса %d: diff=%.12f / Prediction mismatch at index %d: diff=%.12f", i, diff, i, diff)
		}
	}
	t.Log("Предсказания до сохранения и после загрузки совпадают / Predictions before save and after load match")

	// финальный лог
	t.Log("Продуктовый кейс пройден: модель ведёт себя идентично после перезапуска / Product case passed: model behaves identically after restart")
}