package applied

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Product-level test for Train/Eval behavior with BatchNorm + Dropout.
// Logs are produced in the form: "на русском лог / on English log" as requested.

type ProdModel struct {
	d1   *layers.Dense
	bn   *layers.BatchNorm
	drop *layers.Dropout
	d2   *layers.Dense
	eng  *autograd.Engine
}

func initWeights(w []float64) {
	for i := range w {
		w[i] = rand.NormFloat64() * 0.01
	}
}

func NewProdModel(eng *autograd.Engine, in, hidden, out int) *ProdModel {
	d1 := layers.NewDense(in, hidden, initWeights)
	bn := layers.NewBatchNorm(hidden, eng)
	drop := layers.NewDropout(0.5)
	d2 := layers.NewDense(hidden, out, initWeights)
	return &ProdModel{d1: d1, bn: bn, drop: drop, d2: d2, eng: eng}
}

func (m *ProdModel) Forward(x *graph.Node) *graph.Node {
	z1 := m.d1.Forward(x)
	z2 := m.bn.Forward(z1)
	r := m.eng.ReLU(z2)
	rd := m.drop.Forward(r)
	out := m.d2.Forward(rd)
	return out
}

func (m *ProdModel) Train() {
	m.d1.Train()
	m.bn.Train()
	m.drop.Train()
	m.d2.Train()
}

func (m *ProdModel) Eval() {
	m.d1.Eval()
	m.bn.Eval()
	m.drop.Eval()
	m.d2.Eval()
}

func copySlice(src []float64) []float64 {
	if src == nil {
		return nil
	}
	cp := make([]float64, len(src))
	copy(cp, src)
	return cp
}

func floatsEqual(a, b []float64, eps float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}

func floatsDifferent(a, b []float64, eps float64) bool { return !floatsEqual(a, b, eps) }

func TestProductionTrainEvalMode_Product(t *testing.T) {
	// reproducibility for dropout RNG and weight init
	rand.Seed(42)

	engine := autograd.NewEngine()
	model := NewProdModel(engine, 8, 16, 4)

	// create a fixed small batch input [batch, features]
	batch := 4
	features := 8
	data := make([]float64, batch*features)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	inputTensor := &tensor.Tensor{Data: data, Shape: []int{batch, features}, Strides: []int{features, 1}}
	inputNode := graph.NewNode(inputTensor, nil, nil)

	// 1) Eval before training -> baseline inference
	model.Eval()
	outEvalBefore := model.Forward(inputNode)
	vEvalBefore := copySlice(outEvalBefore.Value.Data)
	t.Logf("[инфо] до обучения (Eval) / before training (Eval): %v", vEvalBefore)

	// 2) Training phase (only forward passes to update BN running stats and use Dropout)
	model.Train()
	// do several forward passes in Train mode to update running stats
	for i := 0; i < 20; i++ {
		_ = model.Forward(inputNode)
	}
	// also capture two Train outputs to ensure non-determinism
	outTrain1 := model.Forward(inputNode)
	outTrain2 := model.Forward(inputNode)
	vTrain1 := copySlice(outTrain1.Value.Data)
	vTrain2 := copySlice(outTrain2.Value.Data)
	t.Logf("[инфо] во время обучения - прогон 1 (Train) / during training - run 1 (Train): %v", vTrain1)
	t.Logf("[инфо] во время обучения - прогон 2 (Train) / during training - run 2 (Train): %v", vTrain2)

	if !floatsDifferent(vTrain1, vTrain2, 1e-9) {
		t.Fatalf("ожидалось отличие выходов в Train() из-за Dropout / expected Train() outputs to differ due to Dropout")
	}

	// 3) Eval after training -> uses running stats collected during training
	model.Eval()
	outEvalA := model.Forward(inputNode)
	outEvalB := model.Forward(inputNode)
	vEvalA := copySlice(outEvalA.Value.Data)
	vEvalB := copySlice(outEvalB.Value.Data)
	t.Logf("[инфо] после обучения - прогон 1 (Eval) / after training - run 1 (Eval): %v", vEvalA)
	t.Logf("[инфо] после обучения - прогон 2 (Eval) / after training - run 2 (Eval): %v", vEvalB)

	// Eval outputs must be identical (deterministic inference)
	if !floatsEqual(vEvalA, vEvalB, 1e-9) {
		t.Fatalf("ожидалось совпадение выходов в Eval() / expected Eval() outputs to be identical")
	}

	// 4) Ensure running stats changed by comparing Eval output before vs after training
	if !floatsDifferent(vEvalBefore, vEvalA, 1e-9) {
		t.Fatalf("ожидалось изменение выходов в Eval() до/после обучения (BN running stats должны поменяться) / expected Eval() outputs to change before/after training (BN running stats should be updated)")
	}

	t.Logf("[успех] тест пройден: Train() нестабилен, Eval() детерминирован, BN running stats изменились / [success] test passed: Train() non-deterministic, Eval() deterministic, BN running stats updated")
}
