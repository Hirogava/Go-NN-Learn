package metrics

import (
	"sync"
	"testing"
)

const eps = 1e-9

func almostEqual(a, b float64) bool {
	if a == b {
		return true
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < eps
}

func TestAccuracyFromLabels(t *testing.T) {
	v, err := AccuracyFromLabels([]int{1, 2, 3}, []int{1, 0, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !almostEqual(v, 2.0/3.0) {
		t.Fatalf("expected 2/3, got %v", v)
	}
	_, err = AccuracyFromLabels([]int{1, 2}, []int{1})
	if err == nil {
		t.Fatalf("expected error on length mismatch")
	}
	v, err = AccuracyFromLabels([]int{}, []int{})
	if err != nil {
		t.Fatalf("unexpected error for empty slices: %v", err)
	}
	if v != 0 {
		t.Fatalf("expected 0 for empty slices, got %v", v)
	}
}

func TestMAEFromSlices(t *testing.T) {
	v, err := MAEFromSlices([]float64{1.0, 2.0}, []float64{1.5, 1.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !almostEqual(v, 0.5) {
		t.Fatalf("expected 0.5, got %v", v)
	}
	_, err = MAEFromSlices([]float64{1.0}, []float64{1.0, 2.0})
	if err == nil {
		t.Fatalf("expected error on length mismatch")
	}
}

func TestBinaryPrecisionRecallF1(t *testing.T) {
	p, r, f, err := BinaryPrecisionRecallF1([]int{1, 1, 0, 1}, []int{1, 0, 0, 1}, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !almostEqual(p, 2.0/3.0) {
		t.Fatalf("expected precision 2/3, got %v", p)
	}
	if !almostEqual(r, 1.0) {
		t.Fatalf("expected recall 1, got %v", r)
	}
	if !almostEqual(f, 0.8) {
		t.Fatalf("expected f1 0.8, got %v", f)
	}
}

func TestAccuracyConcurrency(t *testing.T) {
	acc := NewAccuracy()
	var wg sync.WaitGroup
	G := 50
	M := 1000
	K := 700 // correct per batch
	wg.Add(G)
	for g := 0; g < G; g++ {
		go func() {
			defer wg.Done()
			preds := make([]int, M)
			labels := make([]int, M)
			for i := 0; i < M; i++ {
				labels[i] = i
				if i < K {
					preds[i] = labels[i]
				} else {
					preds[i] = -labels[i]
				}
			}
			if err := acc.Update(preds, labels); err != nil {
				t.Errorf("update error: %v", err)
			}
		}()
	}
	wg.Wait()
	expected := float64(G*K) / float64(G*M)
	val := acc.Value()
	if !almostEqual(val, expected) {
		t.Fatalf("expected accuracy %v, got %v", expected, val)
	}
}

func TestMAEConcurrency(t *testing.T) {
	m := NewMAE()
	var wg sync.WaitGroup
	G := 40
	M := 500
	wg.Add(G)
	for g := 0; g < G; g++ {
		go func() {
			defer wg.Done()
			preds := make([]float64, M)
			labels := make([]float64, M)
			for i := 0; i < M; i++ {
				preds[i] = float64(i)
				labels[i] = float64(i + 1) // abs diff == 1
			}
			if err := m.Update(preds, labels); err != nil {
				t.Errorf("update error: %v", err)
			}
		}()
	}
	wg.Wait()
	val := m.Value()
	if !almostEqual(val, 1.0) {
		t.Fatalf("expected MAE 1.0, got %v", val)
	}
}

func computeExpectedFromCounts(counts map[int]map[int]int64) (map[int]float64, map[int]float64, map[int]float64) {
	perP := map[int]float64{}
	perR := map[int]float64{}
	perF := map[int]float64{}
	classes := map[int]struct{}{}
	for t, m := range counts {
		classes[t] = struct{}{}
		for p := range m {
			classes[p] = struct{}{}
		}
	}
	for class := range classes {
		TP := counts[class][class]
		FP := int64(0)
		FN := int64(0)
		for t, m := range counts {
			if t == class {
				continue
			}
			FP += m[class]
		}
		for p, cnt := range counts[class] {
			if p == class {
				continue
			}
			FN += cnt
		}
		p := safeDiv(float64(TP), float64(TP+FP))
		r := safeDiv(float64(TP), float64(TP+FN))
		f := 0.0
		if p+r != 0 {
			f = 2 * p * r / (p + r)
		}
		perP[class] = p
		perR[class] = r
		perF[class] = f
	}
	return perP, perR, perF
}

func TestConfusionMatrixConcurrency(t *testing.T) {
	cm := NewConfusionMatrix()
	var wg sync.WaitGroup
	G := 30
	M := 800
	wg.Add(G)
	expectedCounts := map[int]map[int]int64{}
	var mu sync.Mutex
	for g := 0; g < G; g++ {
		go func(gid int) {
			defer wg.Done()
			preds := make([]int, M)
			labels := make([]int, M)
			localCounts := map[int]map[int]int64{}
			for i := 0; i < M; i++ {
				labels[i] = i % 3
				preds[i] = (labels[i] + (gid % 3)) % 3
				if _, ok := localCounts[labels[i]]; !ok {
					localCounts[labels[i]] = map[int]int64{}
				}
				localCounts[labels[i]][preds[i]]++
			}
			if err := cm.Update(preds, labels); err != nil {
				t.Errorf("update error: %v", err)
			}
			mu.Lock()
			for t, m := range localCounts {
				if _, ok := expectedCounts[t]; !ok {
					expectedCounts[t] = map[int]int64{}
				}
				for p, cnt := range m {
					expectedCounts[t][p] += cnt
				}
			}
			mu.Unlock()
		}(g)
	}
	wg.Wait()
	perP, perR, perF, _, _, _ := cm.PerClassMetrics()
	exP, exR, exF := computeExpectedFromCounts(expectedCounts)
	for cls, ep := range exP {
		pp, ok := perP[cls]
		if !ok {
			t.Fatalf("class %v missing in result", cls)
		}
		if !almostEqual(ep, pp) {
			t.Fatalf("precision mismatch for class %v: expected %v got %v", cls, ep, pp)
		}
	}
	for cls, er := range exR {
		rr, ok := perR[cls]
		if !ok {
			t.Fatalf("class %v missing in recall result", cls)
		}
		if !almostEqual(er, rr) {
			t.Fatalf("recall mismatch for class %v: expected %v got %v", cls, er, rr)
		}
	}
	for cls, ef := range exF {
		ff, ok := perF[cls]
		if !ok {
			t.Fatalf("class %v missing in f1 result", cls)
		}
		if !almostEqual(ef, ff) {
			t.Fatalf("f1 mismatch for class %v: expected %v got %v", cls, ef, ff)
		}
	}
}

func TestResetBehavior(t *testing.T) {
	acc := NewAccuracy()
	_ = acc.Update([]int{1, 2}, []int{1, 0})
	if acc.Value() == 0 {
		t.Fatalf("expected non-zero before reset")
	}
	acc.Reset()
	if acc.Value() != 0 {
		t.Fatalf("expected zero after reset")
	}
	cm := NewConfusionMatrix()
	_ = cm.Update([]int{1}, []int{1})
	pp, _, _, _, _, _ := cm.PerClassMetrics()
	if len(pp) == 0 {
		t.Fatalf("expected entries before reset")
	}
	cm.Reset()
	pp2, _, _, _, _, _ := cm.PerClassMetrics()
	if len(pp2) != 0 {
		t.Fatalf("expected empty after reset")
	}
}

func BenchmarkAccuracyUpdate(b *testing.B) {
	acc := NewAccuracy()
	preds := make([]int, 128)
	labels := make([]int, 128)
	for i := range preds {
		labels[i] = i
		if i%2 == 0 {
			preds[i] = labels[i]
		} else {
			preds[i] = -labels[i]
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = acc.Update(preds, labels)
	}
}

func BenchmarkMAEUpdate(b *testing.B) {
	m := NewMAE()
	preds := make([]float64, 256)
	labels := make([]float64, 256)
	for i := range preds {
		preds[i] = float64(i)
		labels[i] = float64(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Update(preds, labels)
	}
}

func BenchmarkConfusionMatrixUpdate(b *testing.B) {
	cm := NewConfusionMatrix()
	preds := make([]int, 512)
	labels := make([]int, 512)
	for i := range preds {
		labels[i] = i % 10
		preds[i] = (labels[i] + 1) % 10
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cm.Update(preds, labels)
	}
}
