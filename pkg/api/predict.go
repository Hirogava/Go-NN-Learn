package api

import (
	"fmt"

	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
	"github.com/Hirogava/Go-NN-Learn/internal/layers"
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
