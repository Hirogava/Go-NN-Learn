package layers

import (
	"fmt"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type Conv2D struct {
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	padding     int

	weights *graph.Node
	bias    *graph.Node

	inputShape []int
}

func NewConv2D(
	inChannels, outChannels, kernelSize, stride, padding int,
	initFunc func([]float64),
) *Conv2D {
	weightsSize := outChannels * inChannels * kernelSize * kernelSize
	weightsData := make([]float64, weightsSize)
	initFunc(weightsData)

	weights := &graph.Node{
		Value: &tensor.Tensor{
			Data:    weightsData,
			Shape:   []int{outChannels, inChannels, kernelSize, kernelSize},
			Strides: []int{inChannels * kernelSize * kernelSize, kernelSize * kernelSize, kernelSize, 1},
		},
	}

	biasData := make([]float64, outChannels)
	initFunc(biasData)

	bias := &graph.Node{
		Value: &tensor.Tensor{
			Data:    biasData,
			Shape:   []int{outChannels},
			Strides: []int{1},
		},
	}

	return &Conv2D{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		weights:     weights,
		bias:        bias,
	}
}

func (c *Conv2D) Forward(x *graph.Node) *graph.Node {
	if x == nil || x.Value == nil {
		panic("Conv2D.Forward: input is nil")
	}

	if len(x.Value.Shape) != 4 {
		panic(fmt.Sprintf("Conv2D expects 4D input [batch, channels, height, width], got %dD", len(x.Value.Shape)))
	}

	batchSize := x.Value.Shape[0]
	inputChannels := x.Value.Shape[1]
	inputHeight := x.Value.Shape[2]
	inputWidth := x.Value.Shape[3]

	if inputChannels != c.inChannels {
		panic(fmt.Sprintf("Conv2D: input channels mismatch: expected %d, got %d", c.inChannels, inputChannels))
	}

	c.inputShape = append([]int{}, x.Value.Shape...)

	outHeight := (inputHeight+2*c.padding-c.kernelSize)/c.stride + 1
	outWidth := (inputWidth+2*c.padding-c.kernelSize)/c.stride + 1
	padded := pad4D(x.Value.Data, batchSize, inputChannels, inputHeight, inputWidth, c.padding)
	padH := inputHeight + 2*c.padding
	padW := inputWidth + 2*c.padding

	col := im2col(padded, batchSize, inputChannels, padH, padW, c.kernelSize, c.kernelSize, c.stride, outHeight, outWidth)
	colRows := inputChannels * c.kernelSize * c.kernelSize
	colCols := batchSize * outHeight * outWidth

	w := c.weights.Value
	w2D := &tensor.Tensor{
		Data:    w.Data,
		Shape:   []int{c.outChannels, colRows},
		Strides: []int{colRows, 1},
	}

	col2D := &tensor.Tensor{
		Data:    col,
		Shape:   []int{colRows, colCols},
		Strides: []int{colCols, 1},
	}

	outMat, err := tensor.MatMul(w2D, col2D)
	if err != nil {
		panic("Conv2D forward MatMul: " + err.Error())
	}

	outputData := outMat.Data

	biasData := c.bias.Value.Data
	for oc := 0; oc < c.outChannels; oc++ {
		for i := 0; i < colCols; i++ {
			outputData[oc*colCols+i] += biasData[oc]
		}
	}

	outputShape := []int{batchSize, c.outChannels, outHeight, outWidth}
	outputStrides := convStrides(outputShape)

	output := &graph.Node{
		Value: &tensor.Tensor{
			Data:    outputData,
			Shape:   outputShape,
			Strides: outputStrides,
		},
	}

	if autograd.GradEnabled() {
		output.Operation = &conv2dOp{
			x:       x,
			conv2d:  c,
			col:     col,
			colRows: colRows,
			colCols: colCols,
		}
	}

	return output
}

func (c *Conv2D) Params() []*graph.Node {
	return []*graph.Node{c.weights, c.bias}
}

func (c *Conv2D) GetInChannels() int {
	return c.inChannels
}

func (c *Conv2D) GetOutChannels() int {
	return c.outChannels
}

func (c *Conv2D) GetKernelSize() int {
	return c.kernelSize
}

func (c *Conv2D) GetStride() int {
	return c.stride
}

func (c *Conv2D) GetPadding() int {
	return c.padding
}

type conv2dOp struct {
	x       *graph.Node
	conv2d  *Conv2D
	col     []float64
	colRows int
	colCols int
}

func (op *conv2dOp) Backward(grad *tensor.Tensor) {
	c := op.conv2d
	N := c.inputShape[0]
	C := c.inputShape[1]
	H := c.inputShape[2]
	W := c.inputShape[3]
	outH := (H+2*c.padding-c.kernelSize)/c.stride + 1
	outW := (W+2*c.padding-c.kernelSize)/c.stride + 1

	gradReshaped := &tensor.Tensor{
		Data:    grad.Data,
		Shape:   []int{c.outChannels, op.colCols},
		Strides: []int{op.colCols, 1},
	}

	col2D := &tensor.Tensor{
		Data:    op.col,
		Shape:   []int{op.colRows, op.colCols},
		Strides: []int{op.colCols, 1},
	}

	wGradMat, err := tensor.MatMulTransposeB(gradReshaped, col2D)
	if err != nil {
		panic("Conv2D backward dW: " + err.Error())
	}
	op.conv2d.weights.Grad = &tensor.Tensor{
		Data:    wGradMat.Data,
		Shape:   []int{c.outChannels, c.inChannels, c.kernelSize, c.kernelSize},
		Strides: []int{c.inChannels * c.kernelSize * c.kernelSize, c.kernelSize * c.kernelSize, c.kernelSize, 1},
	}

	bGrad := make([]float64, c.outChannels)
	for oc := 0; oc < c.outChannels; oc++ {
		sum := 0.0
		for i := 0; i < op.colCols; i++ {
			sum += grad.Data[oc*op.colCols+i]
		}
		bGrad[oc] = sum
	}
	op.conv2d.bias.Grad = &tensor.Tensor{
		Data:    bGrad,
		Shape:   []int{c.outChannels},
		Strides: []int{1},
	}

	w := c.weights.Value
	w2D := &tensor.Tensor{
		Data:    w.Data,
		Shape:   []int{c.outChannels, op.colRows},
		Strides: []int{op.colRows, 1},
	}
	dcolMat, err := tensor.MatMulTransposeA(w2D, gradReshaped)
	if err != nil {
		panic("Conv2D backward dcol: " + err.Error())
	}

	padH := H + 2*c.padding
	padW := W + 2*c.padding
	dxPadded := col2im(dcolMat.Data, N, C, padH, padW, c.kernelSize, c.kernelSize, c.stride, outH, outW)

	op.x.Grad = unpad4D(dxPadded, N, C, padH, padW, c.padding)
}

func pad4D(data []float64, N, C, H, W, pad int) []float64 {
	if pad <= 0 {
		return data
	}
	padH := H + 2*pad
	padW := W + 2*pad
	out := make([]float64, N*C*padH*padW)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					srcIdx := n*C*H*W + c*H*W + h*W + w
					dstH := h + pad
					dstW := w + pad
					dstIdx := n*C*padH*padW + c*padH*padW + dstH*padW + dstW
					out[dstIdx] = data[srcIdx]
				}
			}
		}
	}
	return out
}

func im2col(padded []float64, N, C, padH, padW, kH, kW, stride, outH, outW int) []float64 {
	colRows := C * kH * kW
	colCols := N * outH * outW
	col := make([]float64, colRows*colCols)

	colIdx := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < C; c++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := oh*stride + kh
							w := ow*stride + kw
							srcIdx := n*C*padH*padW + c*padH*padW + h*padW + w
							col[colIdx] = padded[srcIdx]
							colIdx++
						}
					}
				}
			}
		}
	}
	return col
}

func col2im(col []float64, N, C, padH, padW, kH, kW, stride, outH, outW int) []float64 {
	out := make([]float64, N*C*padH*padW)

	colIdx := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < C; c++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := oh*stride + kh
							w := ow*stride + kw
							dstIdx := n*C*padH*padW + c*padH*padW + h*padW + w
							out[dstIdx] += col[colIdx]
							colIdx++
						}
					}
				}
			}
		}
	}
	return out
}

func unpad4D(padded []float64, N, C, padH, padW, pad int) *tensor.Tensor {
	H := padH - 2*pad
	W := padW - 2*pad
	out := tensor.Zeros(N, C, H, W)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					srcIdx := n*C*padH*padW + c*padH*padW + (h+pad)*padW + (w + pad)
					dstIdx := n*C*H*W + c*H*W + h*W + w
					out.Data[dstIdx] = padded[srcIdx]
				}
			}
		}
	}
	return out
}

func convStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}
