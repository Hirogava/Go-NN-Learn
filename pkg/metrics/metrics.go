package metrics

import (
	"errors"
	"math"
	"sync"
)

type Metric interface {
	Update(preds interface{}, labels interface{}) error
	Value() float64
	Reset()
	Name() string
}

func safeDiv(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return num / den
}

func AccuracyFromLabels(yPred []float64, yTrue []float64) (float64, error) {
	if len(yPred) != len(yTrue) {
		return 0, errors.New("yPred and yTrue must have same length")
	}
	if len(yPred) == 0 {
		return 0, nil
	}
	correct := 0
	for i := range yPred {
		if int(yPred[i]) == int(yTrue[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(yPred)), nil
}

func MAEFromSlices(yPred []float64, yTrue []float64) (float64, error) {
	if len(yPred) != len(yTrue) {
		return 0, errors.New("yPred and yTrue must have same length")
	}
	if len(yPred) == 0 {
		return 0, nil
	}
	var s float64
	for i := range yPred {
		s += math.Abs(yPred[i] - yTrue[i])
	}
	return s / float64(len(yPred)), nil
}

func BinaryPrecisionRecallF1(yPred []float64, yTrue []float64, positiveLabel int) (precision, recall, f1 float64, err error) {
	if len(yPred) != len(yTrue) {
		return 0, 0, 0, errors.New("yPred and yTrue must have same length")
	}
	tp, fp, fn := 0, 0, 0
	for i := range yPred {
		if int(yPred[i]) == positiveLabel {
			if int(yTrue[i]) == positiveLabel {
				tp++
			} else {
				fp++
			}
		} else {
			if int(yTrue[i]) == positiveLabel {
				fn++
			}
		}
	}
	precision = safeDiv(float64(tp), float64(tp+fp))
	recall = safeDiv(float64(tp), float64(tp+fn))
	if precision+recall == 0 {
		f1 = 0
	} else {
		f1 = 2 * precision * recall / (precision + recall)
	}
	return precision, recall, f1, nil
}

type Accuracy struct {
	mu      sync.Mutex
	correct int64
	total   int64
}

func NewAccuracy() *Accuracy { return &Accuracy{} }

func (m *Accuracy) Update(preds interface{}, labels interface{}) error {
	p, ok1 := preds.([]float64)
	l, ok2 := labels.([]float64)
	if !ok1 || !ok2 {
		return errors.New("Accuracy.Update expects []float64 preds and []float64 labels")
	}
	if len(p) != len(l) {
		return errors.New("preds and labels must have same length")
	}
	var localCorrect int64
	for i := range p {
		if int(p[i]) == int(l[i]) {
			localCorrect++
		}
	}
	m.mu.Lock()
	m.correct += localCorrect
	m.total += int64(len(p))
	m.mu.Unlock()
	return nil
}

func (m *Accuracy) Value() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.total == 0 {
		return 0
	}
	return float64(m.correct) / float64(m.total)
}

func (m *Accuracy) Reset() {
	m.mu.Lock()
	m.correct = 0
	m.total = 0
	m.mu.Unlock()
}

func (m *Accuracy) Name() string { return "accuracy" }

type MAE struct {
	mu    sync.Mutex
	sum   float64
	count int64
}

func NewMAE() *MAE { return &MAE{} }

func (m *MAE) Update(preds interface{}, labels interface{}) error {
	p, ok1 := preds.([]float64)
	l, ok2 := labels.([]float64)
	if !ok1 || !ok2 {
		return errors.New("MAE.Update expects []float64 preds and []float64 labels")
	}
	if len(p) != len(l) {
		return errors.New("preds and labels must have same length")
	}
	var localSum float64
	for i := range p {
		localSum += math.Abs(p[i] - l[i])
	}
	m.mu.Lock()
	m.sum += localSum
	m.count += int64(len(p))
	m.mu.Unlock()
	return nil
}

func (m *MAE) Value() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.count == 0 {
		return 0
	}
	return m.sum / float64(m.count)
}

func (m *MAE) Reset() {
	m.mu.Lock()
	m.sum = 0
	m.count = 0
	m.mu.Unlock()
}

func (m *MAE) Name() string { return "mae" }

type ConfusionMatrix struct {
	mu     sync.RWMutex
	counts map[int]map[int]int64
	total  int64
}

func NewConfusionMatrix() *ConfusionMatrix {
	return &ConfusionMatrix{counts: map[int]map[int]int64{}}
}

func (cm *ConfusionMatrix) Update(preds interface{}, labels interface{}) error {
	p, ok1 := preds.([]float64)
	l, ok2 := labels.([]float64)
	if !ok1 || !ok2 {
		return errors.New("ConfusionMatrix.Update expects []float64 preds and []float64 labels")
	}
	if len(p) != len(l) {
		return errors.New("preds and labels must have same length")
	}
	local := map[int]map[int]int64{}
	for i := range p {
		t := int(l[i])
		pr := int(p[i])
		if _, ok := local[t]; !ok {
			local[t] = map[int]int64{}
		}
		local[t][pr]++
	}
	cm.mu.Lock()
	for t, m := range local {
		if _, ok := cm.counts[t]; !ok {
			cm.counts[t] = map[int]int64{}
		}
		for pr, cnt := range m {
			cm.counts[t][pr] += cnt
			cm.total += cnt
		}
	}
	cm.mu.Unlock()
	return nil
}

func (cm *ConfusionMatrix) PerClassMetrics() (perPrecision map[int]float64, perRecall map[int]float64, perF1 map[int]float64, macroP, macroR, macroF1 float64) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	perPrecision = map[int]float64{}
	perRecall = map[int]float64{}
	perF1 = map[int]float64{}
	classes := map[int]struct{}{}
	for t, m := range cm.counts {
		classes[t] = struct{}{}
		for p := range m {
			classes[p] = struct{}{}
		}
	}
	totalP, totalR, totalF := 0.0, 0.0, 0.0
	k := 0
	for class := range classes {
		k++
		TP := cm.counts[class][class]
		FP := int64(0)
		FN := int64(0)
		for t, m := range cm.counts {
			if t == class {
				continue
			}
			FP += m[class]
		}
		for p, cnt := range cm.counts[class] {
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
		perPrecision[class] = p
		perRecall[class] = r
		perF1[class] = f
		totalP += p
		totalR += r
		totalF += f
	}
	if k == 0 {
		return perPrecision, perRecall, perF1, 0, 0, 0
	}
	macroP = totalP / float64(k)
	macroR = totalR / float64(k)
	macroF1 = totalF / float64(k)
	return
}

func (cm *ConfusionMatrix) Reset() {
	cm.mu.Lock()
	cm.counts = map[int]map[int]int64{}
	cm.total = 0
	cm.mu.Unlock()
}

func (cm *ConfusionMatrix) Name() string { return "confusion_matrix" }
