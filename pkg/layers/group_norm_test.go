package layers

import (
	"fmt"
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

func TestGroupNormForward2D(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	// batch=2, channels=4, groups=2 → по 2 канала в группе
	inputData := tensor.Zeros(2, 4)
	inputData.Data = []float64{
		1, 3, 100, 300,
		5, 7, 200, 400,
	}
	input := graph.NewNode(inputData, nil, nil)

	output := gn.Forward(input)

	if len(output.Value.Shape) != 2 || output.Value.Shape[0] != 2 || output.Value.Shape[1] != 4 {
		t.Fatalf("output shape = %v, want [2 4]", output.Value.Shape)
	}

	// Для каждого примера и группы: mean≈0, var≈1 внутри группы
	channelsPerGroup := 2
	for n := range 2 {
		for g := range 2 {
			cStart := g * channelsPerGroup
			cEnd := cStart + channelsPerGroup
			mean, variance := groupMeanVar2D(output.Value.Data, 2, 4, n, cStart, cEnd)
			if math.Abs(mean) > 1e-5 {
				t.Errorf("sample %d group %d: mean = %v, want ~0", n, g, mean)
			}
			if math.Abs(variance-1.0) > 1e-4 {
				t.Errorf("sample %d group %d: variance = %v, want ~1", n, g, variance)
			}
		}
	}
}

func TestGroupNormForward4D(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	// N=1, C=4, H=2, W=2
	inputData := tensor.Zeros(1, 4, 2, 2)
	for i := range inputData.Data {
		inputData.Data[i] = float64(i + 1)
	}
	input := graph.NewNode(inputData, nil, nil)

	output := gn.Forward(input)

	if len(output.Value.Shape) != 4 {
		t.Fatalf("output shape = %v, want 4D", output.Value.Shape)
	}

	mean, variance := groupMeanVar4D(output.Value.Data, 1, 4, 2, 2, 0, 0, 2)
	if math.Abs(mean) > 1e-5 {
		t.Errorf("group 0 mean = %v, want ~0", mean)
	}
	if math.Abs(variance-1.0) > 1e-4 {
		t.Errorf("group 0 variance = %v, want ~1", variance)
	}
}

func TestGroupNormParams(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(6, 3, engine)

	params := gn.Params()
	if len(params) != 2 {
		t.Fatalf("expected 2 params, got %d", len(params))
	}
	for i := range 6 {
		if params[0].Value.Data[i] != 1.0 {
			t.Errorf("gamma[%d] = %v, want 1", i, params[0].Value.Data[i])
		}
		if params[1].Value.Data[i] != 0.0 {
			t.Errorf("beta[%d] = %v, want 0", i, params[1].Value.Data[i])
		}
	}
}

func TestGroupNormGammaBeta(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(2, 1, engine)
	gn.gamma.Value.Data = []float64{2.0, 3.0}
	gn.beta.Value.Data = []float64{1.0, -1.0}

	inputData := tensor.Zeros(1, 2)
	inputData.Data = []float64{0, 2} // mean=1, var=1 → x_hat=[-1,1]
	input := graph.NewNode(inputData, nil, nil)

	output := gn.Forward(input)

	want0 := 2.0*(-1.0) + 1.0
	want1 := 3.0*1.0 + (-1.0)
	if math.Abs(output.Value.Data[0]-want0) > 1e-4 {
		t.Errorf("output[0] = %v, want %v", output.Value.Data[0], want0)
	}
	if math.Abs(output.Value.Data[1]-want1) > 1e-4 {
		t.Errorf("output[1] = %v, want %v", output.Value.Data[1], want1)
	}
}

func TestGroupNormBackward2D(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	xData := tensor.Zeros(2, 4)
	for i := range xData.Data {
		xData.Data[i] = float64(i) + 0.5
	}
	x := engine.RequireGrad(xData)

	output := gn.Forward(x)
	if output.Operation == nil {
		t.Fatal("expected groupNormOp on output")
	}

	engine.Backward(output)

	if x.Grad == nil {
		t.Fatal("expected gradient on input")
	}
	if gn.gamma.Grad == nil || gn.beta.Grad == nil {
		t.Fatal("expected gradients on gamma and beta")
	}
}

func TestGroupNormBackward4D(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	xData := tensor.Randn([]int{2, 4, 3, 3}, 7)
	x := engine.RequireGrad(xData)

	output := gn.Forward(x)
	engine.Backward(output)

	if x.Grad == nil {
		t.Fatal("expected gradient on 4D input")
	}
}

func TestGroupNormGradCheck2D(t *testing.T) {
	proto := graph.NewNode(tensor.Zeros(2, 4), nil, nil)

	build := func(e *autograd.Engine, inputs []*graph.Node) *graph.Node {
		gn := NewGroupNorm(4, 2, e)
		x := e.RequireGrad(copyTensor(inputs[0].Value))
		return gn.Forward(x)
	}

	if !autograd.CheckGradientEngine(build, []*graph.Node{proto}, 1e-5, 1e-3) {
		t.Error("grad check failed for GroupNorm 2D")
	}
}

func TestGroupNormGradCheck4D(t *testing.T) {
	proto := graph.NewNode(tensor.Zeros(1, 4, 2, 2), nil, nil)

	build := func(e *autograd.Engine, inputs []*graph.Node) *graph.Node {
		gn := NewGroupNorm(4, 2, e)
		x := e.RequireGrad(copyTensor(inputs[0].Value))
		return gn.Forward(x)
	}

	if !autograd.CheckGradientEngine(build, []*graph.Node{proto}, 1e-5, 1e-3) {
		t.Error("grad check failed for GroupNorm 4D")
	}
}

func TestGroupNormInvalidNumGroups(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when numChannels not divisible by numGroups")
		}
	}()
	autograd.NewEngine()
	NewGroupNorm(5, 2, autograd.NewEngine())
}

func TestGroupNormInvalidChannels(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for channel mismatch")
		}
	}()

	input := graph.NewNode(tensor.Zeros(2, 6), nil, nil)
	gn.Forward(input)
}

func TestGroupNormInvalidShape(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for 3D input")
		}
	}()

	input := graph.NewNode(tensor.Zeros(2, 4, 3), nil, nil)
	gn.Forward(input)
}

func TestGroupNormSetEpsilon(t *testing.T) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)
	if gn.eps != 1e-5 {
		t.Fatalf("default eps = %v", gn.eps)
	}
	gn.SetEpsilon(1e-3)
	if gn.eps != 1e-3 {
		t.Errorf("eps = %v, want 1e-3", gn.eps)
	}
}

func BenchmarkGroupNormForward4D(b *testing.B) {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(64, 8, engine)
	input := graph.NewNode(tensor.Randn([]int{8, 64, 28, 28}, 1), nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gn.Forward(input)
	}
}

func ExampleGroupNorm() {
	engine := autograd.NewEngine()
	gn := NewGroupNorm(4, 2, engine)

	inputData := tensor.Zeros(1, 4)
	inputData.Data = []float64{1, 2, 10, 20}
	input := graph.NewNode(inputData, nil, nil)

	output := gn.Forward(input)
	fmt.Printf("output shape: %v\n", output.Value.Shape)
	fmt.Printf("num params: %d\n", len(gn.Params()))

	// Output:
	// output shape: [1 4]
	// num params: 2
}

func copyTensor(src *tensor.Tensor) *tensor.Tensor {
	data := make([]float64, len(src.Data))
	copy(data, src.Data)
	return &tensor.Tensor{
		Data:    data,
		Shape:   append([]int{}, src.Shape...),
		Strides: append([]int{}, src.Strides...),
	}
}

func groupMeanVar2D(data []float64, batchSize, numChannels, n, cStart, cEnd int) (mean, variance float64) {
	count := float64(cEnd - cStart)
	sum := 0.0
	for c := cStart; c < cEnd; c++ {
		sum += data[n*numChannels+c]
	}
	mean = sum / count
	sumSq := 0.0
	for c := cStart; c < cEnd; c++ {
		diff := data[n*numChannels+c] - mean
		sumSq += diff * diff
	}
	variance = sumSq / count
	return mean, variance
}

func groupMeanVar4D(data []float64, batchSize, numChannels, height, width, n, cStart, cEnd int) (mean, variance float64) {
	count := float64((cEnd - cStart) * height * width)
	sum := 0.0
	for c := cStart; c < cEnd; c++ {
		for h := range height {
			for w := range width {
				idx := ((n*numChannels+c)*height+h)*width + w
				sum += data[idx]
			}
		}
	}
	mean = sum / count
	sumSq := 0.0
	for c := cStart; c < cEnd; c++ {
		for h := range height {
			for w := range width {
				idx := ((n*numChannels+c)*height+h)*width + w
				diff := data[idx] - mean
				sumSq += diff * diff
			}
		}
	}
	variance = sumSq / count
	return mean, variance
}
