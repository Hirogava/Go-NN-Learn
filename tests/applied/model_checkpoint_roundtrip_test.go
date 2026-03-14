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

// Простая локальная обёртка ReLU, реализует layers.Layer
type reluLayer struct{}

func (r *reluLayer) Forward(x *graph.Node) *graph.Node {
	// В библиотеке есть autograd.Engine.ReLU, но для forward достаточно сделать
	// элементный ReLU над x.Value (без графа), чтобы получить детерминированный output.
	in := x.Value
	outData := make([]float64, len(in.Data))
	for i, v := range in.Data {
		if v > 0 {
			outData[i] = v
		} else {
			outData[i] = 0
		}
	}
	out := &graph.Node{
		Value: &tensor.Tensor{
			Data:    outData,
			Shape:   append([]int(nil), in.Shape...),
			Strides: append([]int(nil), in.Strides...),
		},
	}
	return out
}
func (r *reluLayer) Params() []*graph.Node            { return []*graph.Node{} }
func (r *reluLayer) Train()                            {}
func (r *reluLayer) Eval()                             {}
// Конструктор простой MLP: Dense(in, h) -> ReLU -> Dense(h, out)
type simpleMLP struct {
	l1 *layers.Dense
	r  *reluLayer
	l2 *layers.Dense
}

func newSimpleMLP(in, h, out int, init func([]float64)) *simpleMLP {
	return &simpleMLP{
		l1: layers.NewDense(in, h, init),
		r:  &reluLayer{},
		l2: layers.NewDense(h, out, init),
	}
}

// layers.Module interface
func (m *simpleMLP) Layers() []layers.Layer {
	return []layers.Layer{m.l1, m.r, m.l2}
}
func (m *simpleMLP) Forward(x *graph.Node) *graph.Node {
	x1 := m.l1.Forward(x)
	x2 := m.r.Forward(x1)
	x3 := m.l2.Forward(x2)
	return x3
}
func (m *simpleMLP) Params() []*graph.Node {
	// порядок важен — SaveCheckpoint/LoadCheckpoint сохраняют параметры
	// в порядке, который возвращает Params()
	p := []*graph.Node{}
	p = append(p, m.l1.Params()...)
	p = append(p, m.l2.Params()...)
	return p
}
func (m *simpleMLP) Train() { for _, l := range m.Layers() { l.Train() } }
func (m *simpleMLP) Eval()  { for _, l := range m.Layers() { l.Eval() } }

func TestModelCheckpointRoundtrip(t *testing.T) {
	// детерминированная инициализация: заполняем данные последовательностью,
	// чтобы при повторной инициализации (с другой функцией) было другое содержимое
	initSeq := func(seedOffset int64) func([]float64) {
		return func(dst []float64) {
			for i := range dst {
				// простая запоминаемая последовательность (не rand), чтобы быть детерминированным
				dst[i] = float64((i+int(seedOffset))%1000)/1000.0 + float64(seedOffset)*1e-6
			}
		}
	}

	inDim := 4
	hidden := 7
	outDim := 3
	batchSize := 5

	// модель A (инициализация A)
	modelA := newSimpleMLP(inDim, hidden, outDim, initSeq(1))
	// модель B (инициализация B — отличная от A)
	modelB := newSimpleMLP(inDim, hidden, outDim, initSeq(9999))

	// создаём фиксированный входной батч (детерминированный)
	featSize := batchSize * inDim
	featData := make([]float64, featSize)
	for i := 0; i < featSize; i++ {
		featData[i] = float64((i*37+13)%1000) / 1000.0
	}
	featTensor := &tensor.Tensor{
		Data:    featData,
		Shape:   []int{batchSize, inDim},
		Strides: []int{inDim, 1},
	}
	inputNode := graph.NewNode(featTensor, nil, nil)

	// forward через modelA
	outA := modelA.Forward(inputNode)
	if outA == nil || outA.Value == nil {
		t.Fatalf("modelA forward returned nil")
	}
	predsA := append([]float64(nil), outA.Value.Data...) // копия

	// Сохраняем checkpoint модели A во временный файл
	tmpDir, err := os.MkdirTemp("", "ckpt_test")
	if err != nil {
		t.Fatalf("mkdir temp: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	ckptPath := filepath.Join(tmpDir, "modelA.ckpt")

	if err := api.SaveCheckpoint(modelA, ckptPath); err != nil {
		t.Fatalf("SaveCheckpoint failed: %v", err)
	}

	// Убедимся, что модельB стартово отличается от modelA (параметры)
	paramsA := modelA.Params()
	paramsB := modelB.Params()
	if len(paramsA) != len(paramsB) {
		t.Fatalf("params count mismatch before load: A=%d B=%d", len(paramsA), len(paramsB))
	}
	someEqual := true
	for i := range paramsA {
		// если хотя бы один параметр совпадает полностью — ок, но хотим заметить отличие:
		if len(paramsA[i].Value.Data) != len(paramsB[i].Value.Data) {
			t.Fatalf("param %d size mismatch", i)
		}
		eq := true
		for j := range paramsA[i].Value.Data {
			if paramsA[i].Value.Data[j] != paramsB[i].Value.Data[j] {
				eq = false
				break
			}
		}
		if eq {
			// если ровно равны — отмечаем (но не фатально)
			// возможно при такой инициализации несколько параметров совпали случайно
		} else {
			someEqual = false
		}
	}
	// (не фатальная проверка) если все параметры совпали — это необычно
	if someEqual {
		// не аварийно, просто лог
		t.Logf("note: at least one param differs between A and B (expected)")
	}

	// Загружаем чекпоинт в модельB
	if err := api.LoadCheckpoint(modelB, ckptPath); err != nil {
		t.Fatalf("LoadCheckpoint failed: %v", err)
	}

	// Теперь параметры A и B должны совпадать в пределах eps
	const eps = 1e-9
	for i := range paramsA {
		a := paramsA[i].Value.Data
		b := paramsB[i].Value.Data
		if len(a) != len(b) {
			t.Fatalf("param %d len mismatch after load: %d vs %d", i, len(a), len(b))
		}
		for j := range a {
			diff := a[j] - b[j]
			if diff < 0 {
				if -diff > eps {
					t.Fatalf("param %d element %d mismatch: a=%v b=%v (diff=%.12f)", i, j, a[j], b[j], diff)
				}
			} else {
				if diff > eps {
					t.Fatalf("param %d element %d mismatch: a=%v b=%v (diff=%.12f)", i, j, a[j], b[j], diff)
				}
			}
		}
	}

	// Forward через modelB (после загрузки)
	outB := modelB.Forward(graph.NewNode(featTensor, nil, nil))
	if outB == nil || outB.Value == nil {
		t.Fatalf("modelB forward returned nil")
	}
	predsB := outB.Value.Data

	// Сравниваем предсказания
	if len(predsA) != len(predsB) {
		t.Fatalf("predictions length mismatch: %d vs %d", len(predsA), len(predsB))
	}
	for i := range predsA {
		diff := predsA[i] - predsB[i]
		if diff < 0 {
			if -diff > eps {
				t.Fatalf("prediction[%d] mismatch: a=%v b=%v (diff=%.12f)", i, predsA[i], predsB[i], diff)
			}
		} else {
			if diff > eps {
				t.Fatalf("prediction[%d] mismatch: a=%v b=%v (diff=%.12f)", i, predsA[i], predsB[i], diff)
			}
		}
	}
}