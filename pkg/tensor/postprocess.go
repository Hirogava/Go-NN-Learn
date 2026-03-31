package tensor

import (
	"fmt"
	"math"
	"sort"
)

// TopKItem describes a single top-k value in a row-major [N, C] tensor.
type TopKItem struct {
	Index int
	Value float64
}

// SoftmaxByRow computes a numerically stable softmax for each row of a [N, C] tensor.
func SoftmaxByRow(logits *Tensor) (*Tensor, error) {
	rows, cols, err := validateRowMatrix(logits, "SoftmaxByRow")
	if err != nil {
		return nil, err
	}

	out := &Tensor{
		Shape:   []int{rows, cols},
		Strides: []int{cols, 1},
		DType:   logits.DType,
	}

	if logits.DType == Float32 {
		out.Data32 = make([]float32, rows*cols)
		for r := 0; r < rows; r++ {
			base := r * cols
			maxVal := logits.Data32[base]
			for c := 1; c < cols; c++ {
				if logits.Data32[base+c] > maxVal {
					maxVal = logits.Data32[base+c]
				}
			}

			var sumExp float64
			for c := 0; c < cols; c++ {
				expVal := math.Exp(float64(logits.Data32[base+c] - maxVal))
				out.Data32[base+c] = float32(expVal)
				sumExp += expVal
			}

			invSum := float32(1.0 / sumExp)
			for c := 0; c < cols; c++ {
				out.Data32[base+c] *= invSum
			}
		}
		return out, nil
	}

	out.Data = make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		base := r * cols
		maxVal := logits.Data[base]
		for c := 1; c < cols; c++ {
			if logits.Data[base+c] > maxVal {
				maxVal = logits.Data[base+c]
			}
		}

		sumExp := 0.0
		for c := 0; c < cols; c++ {
			expVal := math.Exp(logits.Data[base+c] - maxVal)
			out.Data[base+c] = expVal
			sumExp += expVal
		}

		invSum := 1.0 / sumExp
		for c := 0; c < cols; c++ {
			out.Data[base+c] *= invSum
		}
	}

	return out, nil
}

// ArgmaxByRow returns the argmax class index for each row of a [N, C] tensor.
func ArgmaxByRow(scores *Tensor) ([]int, error) {
	rows, cols, err := validateRowMatrix(scores, "ArgmaxByRow")
	if err != nil {
		return nil, err
	}

	indices := make([]int, rows)
	if scores.DType == Float32 {
		for r := 0; r < rows; r++ {
			base := r * cols
			bestIdx := 0
			bestVal := scores.Data32[base]
			for c := 1; c < cols; c++ {
				if scores.Data32[base+c] > bestVal {
					bestVal = scores.Data32[base+c]
					bestIdx = c
				}
			}
			indices[r] = bestIdx
		}
		return indices, nil
	}

	for r := 0; r < rows; r++ {
		base := r * cols
		bestIdx := 0
		bestVal := scores.Data[base]
		for c := 1; c < cols; c++ {
			if scores.Data[base+c] > bestVal {
				bestVal = scores.Data[base+c]
				bestIdx = c
			}
		}
		indices[r] = bestIdx
	}

	return indices, nil
}

// TopKByRow returns the top-k values and indices for each row of a [N, C] tensor.
// Results are sorted by value descending, then by index ascending for stable ties.
func TopKByRow(scores *Tensor, k int) ([][]TopKItem, error) {
	rows, cols, err := validateRowMatrix(scores, "TopKByRow")
	if err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, fmt.Errorf("TopKByRow: k must be > 0, got %d", k)
	}

	if k > cols {
		k = cols
	}

	out := make([][]TopKItem, rows)
	for r := 0; r < rows; r++ {
		base := r * cols
		items := make([]TopKItem, cols)
		for c := 0; c < cols; c++ {
			items[c] = TopKItem{
				Index: c,
				Value: tensorValueAt(scores, base+c),
			}
		}

		sort.Slice(items, func(i, j int) bool {
			if items[i].Value == items[j].Value {
				return items[i].Index < items[j].Index
			}
			return items[i].Value > items[j].Value
		})

		rowTopK := make([]TopKItem, k)
		copy(rowTopK, items[:k])
		out[r] = rowTopK
	}

	return out, nil
}

func validateRowMatrix(t *Tensor, op string) (rows, cols int, err error) {
	if t == nil {
		return 0, 0, fmt.Errorf("%s: tensor is nil", op)
	}
	if len(t.Shape) != 2 {
		return 0, 0, fmt.Errorf("%s: expected tensor shape [N, C], got %v", op, t.Shape)
	}
	rows, cols = t.Shape[0], t.Shape[1]
	if rows <= 0 || cols <= 0 {
		return 0, 0, fmt.Errorf("%s: expected positive shape [N, C], got %v", op, t.Shape)
	}
	if t.DType == Float32 {
		if len(t.Data32) != rows*cols {
			return 0, 0, fmt.Errorf("%s: invalid Float32 data length %d for shape %v", op, len(t.Data32), t.Shape)
		}
		return rows, cols, nil
	}
	if len(t.Data) != rows*cols {
		return 0, 0, fmt.Errorf("%s: invalid Float64 data length %d for shape %v", op, len(t.Data), t.Shape)
	}
	return rows, cols, nil
}

func tensorValueAt(t *Tensor, idx int) float64 {
	if t.DType == Float32 {
		return float64(t.Data32[idx])
	}
	return t.Data[idx]
}
