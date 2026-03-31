package api

import (
	"fmt"
	"strconv"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// Predict выполняет прямой проход (forward) модели m по входному узлу x
// и возвращает выходной *graph.Node. Если m или x == nil — возвращает nil.
// Predict НЕ меняет режимы train/eval; если требуется eval-mode для Dropout/BN,
// переключите режим вручную до вызова.
func Predict(m layers.Module, x *graph.Node) *graph.Node {
	if m == nil || x == nil {
		return nil
	}
	return m.Forward(x)
}

// PredictionItem is a standardized top-k prediction entry for client code.
type PredictionItem struct {
	ClassLabel  string
	ClassID     int
	Probability float64
}

// PredictionOptions controls standardized top-k postprocessing for [N, C] logits.
type PredictionOptions struct {
	K         int
	Labels    []string
	Threshold float64
}

// TopKPredictions converts [N, C] logits into top-k prediction items per row.
// It applies row-wise softmax, then returns items sorted by probability descending.
func TopKPredictions(logits *tensor.Tensor, opts PredictionOptions) ([][]PredictionItem, error) {
	if logits == nil {
		return nil, fmt.Errorf("TopKPredictions: logits tensor is nil")
	}
	if opts.K <= 0 {
		return nil, fmt.Errorf("TopKPredictions: k must be > 0, got %d", opts.K)
	}
	if opts.Threshold < 0 || opts.Threshold > 1 {
		return nil, fmt.Errorf("TopKPredictions: threshold must be in [0, 1], got %v", opts.Threshold)
	}
	if len(logits.Shape) != 2 {
		return nil, fmt.Errorf("TopKPredictions: expected logits shape [N, C], got %v", logits.Shape)
	}

	numClasses := logits.Shape[1]
	if len(opts.Labels) > 0 && len(opts.Labels) != numClasses {
		return nil, fmt.Errorf("TopKPredictions: labels length %d does not match class count %d", len(opts.Labels), numClasses)
	}

	probs, err := tensor.SoftmaxByRow(logits)
	if err != nil {
		return nil, err
	}

	topK, err := tensor.TopKByRow(probs, opts.K)
	if err != nil {
		return nil, err
	}

	out := make([][]PredictionItem, len(topK))
	for row := range topK {
		items := make([]PredictionItem, 0, len(topK[row]))
		for _, item := range topK[row] {
			if item.Value < opts.Threshold {
				continue
			}
			items = append(items, PredictionItem{
				ClassID:     item.Index,
				ClassLabel:  classLabel(item.Index, opts.Labels),
				Probability: item.Value,
			})
		}
		out[row] = items
	}

	return out, nil
}

// Eval вычисляет среднюю метрику на парах inputs/targets с использованием
// метрики metric(pred, target) -> float64. Возвращает среднюю по всем парам.
// Проверки: model != nil, metric != nil, len(inputs) == len(targets) > 0.
func Eval(m layers.Module, inputs []*graph.Node, targets []*graph.Node, metric func(*graph.Node, *graph.Node) float64) (float64, error) {
	if m == nil {
		return 0, fmt.Errorf("Eval: model is nil")
	}
	if metric == nil {
		return 0, fmt.Errorf("Eval: metric is nil")
	}
	if len(inputs) != len(targets) {
		return 0, fmt.Errorf("Eval: inputs/targets length mismatch")
	}
	if len(inputs) == 0 {
		return 0, fmt.Errorf("Eval: empty dataset")
	}

	var sum float64
	for i := range inputs {
		if inputs[i] == nil || targets[i] == nil {
			return 0, fmt.Errorf("Eval: nil input or target at index %d", i)
		}

		out := m.Forward(inputs[i])
		if out == nil {
			return 0, fmt.Errorf("Eval: model returned nil output at index %d", i)
		}

		sum += metric(out, targets[i])
	}

	return sum / float64(len(inputs)), nil
}

func classLabel(classID int, labels []string) string {
	if len(labels) > 0 {
		return labels[classID]
	}
	return strconv.Itoa(classID)
}
