package api_test

import (
	"path/filepath"
	"testing"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
	"github.com/Hirogava/Go-NN-Learn/internal/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/api"
)

// mockModule реализует слои.Модуль для тестов.
// ВАЖНО: этот тип должен быть объявлен только один раз в пакете api_test.
type mockModule struct {
	params []*graph.Node
}

// Layers возвращает дочерние слои модуля (не используемые в этих тестах).
func (m *mockModule) Layers() []layers.Layer { return nil }

// Forward реализует пересылку идентификационных данных.
func (m *mockModule) Forward(x *graph.Node) *graph.Node { return x }

// Params возвращает указатели на узлы параметров.
func (m *mockModule) Params() []*graph.Node { return m.params }

// newMockParam удобство для тестов.
func newMockParam(vals []float64, shape []int) *graph.Node {
	return &graph.Node{
		Value: &tensor.Tensor{
			Data:  append([]float64(nil), vals...),
			Shape: append([]int(nil), shape...),
		},
	}
}

func TestSaveLoadCheckpoint_Roundtrip(t *testing.T) {
	// подготовьте модуль с двумя параметрами
	p1 := newMockParam([]float64{1.1, 2.2, 3.3, 4.4}, []int{2, 2})
	p2 := newMockParam([]float64{5.5, 6.6}, []int{2})
	mod := &mockModule{params: []*graph.Node{p1, p2}}

	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt_test.bin")

	// Save
	if err := api.SaveCheckpoint(mod, path); err != nil {
		t.Fatalf("SaveCheckpoint failed: %v", err)
	}

	// Обнулите параметры, чтобы убедиться, что загрузка восстановит их
	for i := range p1.Value.Data {
		p1.Value.Data[i] = 0
	}
	for i := range p2.Value.Data {
		p2.Value.Data[i] = 0
	}

	// Load
	if err := api.LoadCheckpoint(mod, path); err != nil {
		t.Fatalf("LoadCheckpoint failed: %v", err)
	}

	want1 := []float64{1.1, 2.2, 3.3, 4.4}
	for i := range want1 {
		if p1.Value.Data[i] != want1[i] {
			t.Fatalf("p1[%d] = %v, want %v", i, p1.Value.Data[i], want1[i])
		}
	}
	want2 := []float64{5.5, 6.6}
	for i := range want2 {
		if p2.Value.Data[i] != want2[i] {
			t.Fatalf("p2[%d] = %v, want %v", i, p2.Value.Data[i], want2[i])
		}
	}
}

// проверка во время компиляции: mockModule реализует слои.Модуль
func TestMockModuleImplementsLayersModule(t *testing.T) {
	var _ layers.Module = (*mockModule)(nil)
}
