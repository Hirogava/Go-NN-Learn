package tensor

import (
	"math"
	"testing"
)

func TestSoftmaxByRow(t *testing.T) {
	logits := &Tensor{
		Data:    []float64{1, 2, 3, 1000, 1001, 1002},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
		DType:   Float64,
	}

	out, err := SoftmaxByRow(logits)
	if err != nil {
		t.Fatalf("SoftmaxByRow returned error: %v", err)
	}

	if !shapesEqual(out.Shape, []int{2, 3}) {
		t.Fatalf("unexpected output shape: %v", out.Shape)
	}

	for r := 0; r < 2; r++ {
		base := r * 3
		sum := out.Data[base] + out.Data[base+1] + out.Data[base+2]
		if math.Abs(sum-1.0) > 1e-9 {
			t.Fatalf("row %d softmax sum = %v, want 1", r, sum)
		}
	}

	if math.Abs(out.Data[0]-out.Data[3]) > 1e-9 ||
		math.Abs(out.Data[1]-out.Data[4]) > 1e-9 ||
		math.Abs(out.Data[2]-out.Data[5]) > 1e-9 {
		t.Fatalf("softmax should be shift-invariant, got %v", out.Data)
	}
}

func TestSoftmaxByRowFloat32(t *testing.T) {
	logits := &Tensor{
		Data32:  []float32{0, 1, 2, -1, -2, -3},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
		DType:   Float32,
	}

	out, err := SoftmaxByRow(logits)
	if err != nil {
		t.Fatalf("SoftmaxByRow returned error: %v", err)
	}

	for r := 0; r < 2; r++ {
		base := r * 3
		sum := float64(out.Data32[base] + out.Data32[base+1] + out.Data32[base+2])
		if math.Abs(sum-1.0) > 1e-5 {
			t.Fatalf("row %d softmax sum = %v, want 1", r, sum)
		}
	}
}

func TestArgmaxByRow(t *testing.T) {
	scores := &Tensor{
		Data:    []float64{0.1, 0.8, 0.2, 10, 2, 3},
		Shape:   []int{2, 3},
		Strides: []int{3, 1},
		DType:   Float64,
	}

	indices, err := ArgmaxByRow(scores)
	if err != nil {
		t.Fatalf("ArgmaxByRow returned error: %v", err)
	}

	want := []int{1, 0}
	for i := range want {
		if indices[i] != want[i] {
			t.Fatalf("argmax[%d] = %d, want %d", i, indices[i], want[i])
		}
	}
}

func TestTopKByRow(t *testing.T) {
	scores := &Tensor{
		Data:    []float64{0.5, 0.9, 0.1, 0.9, 0.4, 0.2, 0.8, 0.1},
		Shape:   []int{2, 4},
		Strides: []int{4, 1},
		DType:   Float64,
	}

	topK, err := TopKByRow(scores, 2)
	if err != nil {
		t.Fatalf("TopKByRow returned error: %v", err)
	}

	if len(topK) != 2 {
		t.Fatalf("len(topK) = %d, want 2", len(topK))
	}
	if topK[0][0].Index != 1 || topK[0][1].Index != 3 {
		t.Fatalf("unexpected first row top-k indices: %+v", topK[0])
	}
	if topK[1][0].Index != 2 || topK[1][1].Index != 0 {
		t.Fatalf("unexpected second row top-k indices: %+v", topK[1])
	}
	if topK[0][0].Value != 0.9 || topK[1][0].Value != 0.8 {
		t.Fatalf("unexpected top-k values: %+v", topK)
	}
}

func TestTopKByRowCapsKToColumns(t *testing.T) {
	scores := &Tensor{
		Data:    []float64{0.2, 0.3, 0.4},
		Shape:   []int{1, 3},
		Strides: []int{3, 1},
		DType:   Float64,
	}

	topK, err := TopKByRow(scores, 10)
	if err != nil {
		t.Fatalf("TopKByRow returned error: %v", err)
	}
	if len(topK[0]) != 3 {
		t.Fatalf("len(topK[0]) = %d, want 3", len(topK[0]))
	}
}

func TestPostprocessShapeErrors(t *testing.T) {
	bad := &Tensor{
		Data:    []float64{1, 2, 3},
		Shape:   []int{3},
		Strides: []int{1},
		DType:   Float64,
	}

	if _, err := SoftmaxByRow(bad); err == nil {
		t.Fatal("SoftmaxByRow expected shape error")
	}
	if _, err := ArgmaxByRow(bad); err == nil {
		t.Fatal("ArgmaxByRow expected shape error")
	}
	if _, err := TopKByRow(bad, 2); err == nil {
		t.Fatal("TopKByRow expected shape error")
	}
	if _, err := TopKByRow(&Tensor{
		Data:    []float64{1, 2, 3, 4},
		Shape:   []int{2, 2},
		Strides: []int{2, 1},
		DType:   Float64,
	}, 0); err == nil {
		t.Fatal("TopKByRow expected k validation error")
	}
}
