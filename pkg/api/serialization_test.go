package api_test

import (
	"path/filepath"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

type mockModule struct {
	params []*graph.Node
}

func (m *mockModule) Layers() []layers.Layer {
	return nil
}

func (m *mockModule) Forward(x *graph.Node) *graph.Node {
	return x
}

func (m *mockModule) Params() []*graph.Node {
	return m.params
}

func newMockParam(vals []float64, shape []int) *graph.Node {
	return &graph.Node{
		Value: &tensor.Tensor{
			Data:    append([]float64(nil), vals...),
			Shape:   append([]int(nil), shape...),
			Strides: nil,
		},
	}
}

func TestSaveLoadCheckpoint_Roundtrip(t *testing.T) {
	p1 := newMockParam([]float64{1.1, 2.2, 3.3, 4.4}, []int{2, 2})
	p2 := newMockParam([]float64{5.5, 6.6}, []int{2})
	mod := &mockModule{params: []*graph.Node{p1, p2}}

	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt_test.bin")

	if err := api.SaveCheckpoint(mod, path); err != nil {
		t.Fatalf("SaveCheckpoint failed: %v", err)
	}

	for i := range p1.Value.Data {
		p1.Value.Data[i] = 0
	}
	for i := range p2.Value.Data {
		p2.Value.Data[i] = 0
	}

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

func BenchmarkSaveLoadCheckpoint(b *testing.B) {
	const size = 1024
	data1 := make([]float64, size)
	data2 := make([]float64, size/2)
	for i := range data1 {
		data1[i] = float64(i) * 0.001
	}
	for i := range data2 {
		data2[i] = float64(i) * 0.002
	}

	p1 := newMockParam(data1, []int{size})
	p2 := newMockParam(data2, []int{len(data2)})
	mod := &mockModule{params: []*graph.Node{p1, p2}}

	path := filepath.Join(b.TempDir(), "bench_ckpt.bin")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := api.SaveCheckpoint(mod, path); err != nil {
			b.Fatalf("SaveCheckpoint failed: %v", err)
		}
		for j := range p1.Value.Data {
			p1.Value.Data[j] = 0
		}
		for j := range p2.Value.Data {
			p2.Value.Data[j] = 0
		}
		if err := api.LoadCheckpoint(mod, path); err != nil {
			b.Fatalf("LoadCheckpoint failed: %v", err)
		}
	}
}
