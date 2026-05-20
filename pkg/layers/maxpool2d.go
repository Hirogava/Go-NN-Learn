package layers

import (
	"fmt"
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type MaxPooling2D struct {
	kernelSize int
	stride     int
}

func NewMaxPooling2D(kernelSize, stride int) *MaxPooling2D {
	if kernelSize <= 0 {
		panic("MaxPooling2D: kernelSize must be positive")
	}
	if stride <= 0 {
		panic("MaxPooling2D: stride must be positive")
	}
	return &MaxPooling2D{
		kernelSize: kernelSize,
		stride:     stride,
	}
}

func (m *MaxPooling2D) Forward(x *graph.Node) *graph.Node {
	if x == nil || x.Value == nil {
		panic("MaxPooling2D.Forward: input is nil")
	}
	if len(x.Value.Shape) != 4 {
		panic(fmt.Sprintf("MaxPooling2D expects 4D input [batch, channels, height, width], got %dD", len(x.Value.Shape)))
	}

	N := x.Value.Shape[0]
	C := x.Value.Shape[1]
	H := x.Value.Shape[2]
	W := x.Value.Shape[3]
	if H < m.kernelSize || W < m.kernelSize {
		panic(fmt.Sprintf("MaxPooling2D: kernelSize %d is larger than input spatial size [%d,%d]", m.kernelSize, H, W))
	}

	outH := (H-m.kernelSize)/m.stride + 1
	outW := (W-m.kernelSize)/m.stride + 1
	output := tensor.Zeros(N, C, outH, outW)
	maxIndices := make([]int, len(output.Data))

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					maxVal := math.Inf(-1)
					maxIdx := -1
					for kh := 0; kh < m.kernelSize; kh++ {
						for kw := 0; kw < m.kernelSize; kw++ {
							h := oh*m.stride + kh
							w := ow*m.stride + kw
							idx := n*C*H*W + c*H*W + h*W + w
							if x.Value.Data[idx] > maxVal {
								maxVal = x.Value.Data[idx]
								maxIdx = idx
							}
						}
					}
					outIdx := n*C*outH*outW + c*outH*outW + oh*outW + ow
					output.Data[outIdx] = maxVal
					maxIndices[outIdx] = maxIdx
				}
			}
		}
	}

	out := &graph.Node{
		Value: output,
	}
	if autograd.GradEnabled() {
		out.Operation = &maxPooling2DOp{
			x:          x,
			inputShape: append([]int{}, x.Value.Shape...),
			maxIndices: maxIndices,
		}
	}
	return out
}

func (m *MaxPooling2D) Params() []*graph.Node {
	return nil
}

func (m *MaxPooling2D) Train() {}
func (m *MaxPooling2D) Eval()  {}

func (m *MaxPooling2D) GetKernelSize() int {
	return m.kernelSize
}

func (m *MaxPooling2D) GetStride() int {
	return m.stride
}

type maxPooling2DOp struct {
	x          *graph.Node
	inputShape []int
	maxIndices []int
}

func (op *maxPooling2DOp) Backward(grad *tensor.Tensor) {
	xGrad := tensor.Zeros(op.inputShape...)
	for i, inputIdx := range op.maxIndices {
		xGrad.Data[inputIdx] += grad.Data[i]
	}

	if op.x.Grad == nil {
		op.x.Grad = xGrad
		return
	}
	for i := range op.x.Grad.Data {
		op.x.Grad.Data[i] += xGrad.Data[i]
	}
}
