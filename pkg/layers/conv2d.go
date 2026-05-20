package layers

import (
	"fmt"
	"strings"

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
	paddingMode string
	dilation    int

	weights *graph.Node
	bias    *graph.Node

	inputShape []int
}

type Conv2DConfig struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     string
	Dilation    int
	WInit       Initializer
	BInit       Initializer
}

func NewConv2D(
	inChannels, outChannels, kernelSize, stride, padding int,
	wInit, bInit Initializer,
) *Conv2D {
	c := newConv2D(inChannels, outChannels, kernelSize, stride, wInit, bInit)
	c.padding = padding
	c.paddingMode = "explicit"
	return c
}

func NewConv2DWithConfig(cfg Conv2DConfig) *Conv2D {
	c := newConv2D(cfg.InChannels, cfg.OutChannels, cfg.KernelSize, cfg.Stride, cfg.WInit, cfg.BInit)
	c.paddingMode = normalizeConvPadding(cfg.Padding)
	c.dilation = cfg.Dilation
	if c.dilation == 0 {
		c.dilation = 1
	}
	return c
}

func newConv2D(
	inChannels, outChannels, kernelSize, stride int,
	wInit, bInit Initializer,
) *Conv2D {
	if stride <= 0 {
		panic("Conv2D: stride must be > 0")
	}
	if kernelSize <= 0 {
		panic("Conv2D: kernel size must be > 0")
	}
	if wInit == nil {
		wInit = ZeroInit()
	}
	if bInit == nil {
		bInit = ZeroInit()
	}

	weightsSize := outChannels * inChannels * kernelSize * kernelSize
	weightsData := make([]float64, weightsSize)
	wInit(weightsData)

	weights := &graph.Node{
		Value: &tensor.Tensor{
			Data:    weightsData,
			Shape:   []int{outChannels, inChannels, kernelSize, kernelSize},
			Strides: []int{inChannels * kernelSize * kernelSize, kernelSize * kernelSize, kernelSize, 1},
		},
	}

	biasData := make([]float64, outChannels)
	bInit(biasData)

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
		paddingMode: "explicit",
		dilation:    1,
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

	dilation := c.dilation
	if dilation <= 0 {
		dilation = 1
	}
	effKernel := (c.kernelSize-1)*dilation + 1
	outHeight, padTop, padBottom := convOutputAndPadding(inputHeight, c.kernelSize, c.stride, dilation, c.padding, c.paddingMode)
	outWidth, padLeft, padRight := convOutputAndPadding(inputWidth, c.kernelSize, c.stride, dilation, c.padding, c.paddingMode)
	if outHeight <= 0 || outWidth <= 0 {
		panic(fmt.Sprintf("Conv2D: invalid output shape [%d,%d] for input [%d,%d], effective kernel %d, stride %d", outHeight, outWidth, inputHeight, inputWidth, effKernel, c.stride))
	}

	padded := pad4D(x.Value.Data, batchSize, inputChannels, inputHeight, inputWidth, padTop, padBottom, padLeft, padRight)
	padH := inputHeight + padTop + padBottom
	padW := inputWidth + padLeft + padRight

	col := im2col(padded, batchSize, inputChannels, padH, padW, c.kernelSize, c.kernelSize, c.stride, dilation, outHeight, outWidth)
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

	biasData := c.bias.Value.Data
	for oc := 0; oc < c.outChannels; oc++ {
		for i := 0; i < colCols; i++ {
			outMat.Data[oc*colCols+i] += biasData[oc]
		}
	}

	outputShape := []int{batchSize, c.outChannels, outHeight, outWidth}
	outputStrides := convStrides(outputShape)
	outputData := matToNCHW(outMat.Data, batchSize, c.outChannels, outHeight, outWidth)

	outputTensor := &tensor.Tensor{
		Data:    outputData,
		Shape:   outputShape,
		Strides: outputStrides,
	}

	if autograd.GradEnabled() {
		op := &conv2dOp{
			x:       x,
			conv2d:  c,
			col:     col,
			colRows: colRows,
			colCols: colCols,
			outH:    outHeight,
			outW:    outWidth,
			padTop:  padTop,
			padLeft: padLeft,
			padH:    padH,
			padW:    padW,
		}
		return graph.NewNode(outputTensor, []*graph.Node{x, c.weights, c.bias}, op)
	}

	return &graph.Node{Value: outputTensor}
}

func (c *Conv2D) Params() []*graph.Node {
	return []*graph.Node{c.weights, c.bias}
}

func (c *Conv2D) Train() {}
func (c *Conv2D) Eval()  {}

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
	outH    int
	outW    int
	padTop  int
	padLeft int
	padH    int
	padW    int
}

func (op *conv2dOp) Backward(grad *tensor.Tensor) {
	c := op.conv2d
	N := c.inputShape[0]
	C := c.inputShape[1]
	H := c.inputShape[2]
	W := c.inputShape[3]
	dilation := c.dilation
	if dilation <= 0 {
		dilation = 1
	}
	outH := op.outH
	outW := op.outW
	gradMatData := nchwToMat(grad.Data, N, c.outChannels, outH, outW)

	gradReshaped := &tensor.Tensor{
		Data:    gradMatData,
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

	if op.conv2d.weights.Grad == nil {
		op.conv2d.weights.Grad = &tensor.Tensor{
			Data:    wGradMat.Data,
			Shape:   []int{c.outChannels, c.inChannels, c.kernelSize, c.kernelSize},
			Strides: []int{c.inChannels * c.kernelSize * c.kernelSize, c.kernelSize * c.kernelSize, c.kernelSize, 1},
		}
	} else {
		for i := range op.conv2d.weights.Grad.Data {
			op.conv2d.weights.Grad.Data[i] += wGradMat.Data[i]
		}
	}

	if op.conv2d.bias.Grad == nil {
		op.conv2d.bias.Grad = tensor.Zeros(c.outChannels)
	}
	for oc := 0; oc < c.outChannels; oc++ {
		sum := 0.0
		for i := 0; i < op.colCols; i++ {
			sum += gradMatData[oc*op.colCols+i]
		}
		op.conv2d.bias.Grad.Data[oc] += sum
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

	dxPadded := col2im(dcolMat.Data, N, C, op.padH, op.padW, c.kernelSize, c.kernelSize, c.stride, dilation, outH, outW)

	xGrad := unpad4D(dxPadded, N, C, H, W, op.padH, op.padW, op.padTop, op.padLeft)
	if op.x.Grad == nil {
		op.x.Grad = xGrad
	} else {
		for i := range op.x.Grad.Data {
			op.x.Grad.Data[i] += xGrad.Data[i]
		}
	}
}

func normalizeConvPadding(padding string) string {
	padding = strings.ToLower(strings.TrimSpace(padding))
	switch padding {
	case "", "valid":
		return "valid"
	case "same":
		return "same"
	default:
		panic(fmt.Sprintf("Conv2D: unsupported padding mode %q", padding))
	}
}

func convOutputAndPadding(input, kernel, stride, dilation, explicitPad int, mode string) (out, before, after int) {
	effKernel := (kernel-1)*dilation + 1
	switch mode {
	case "same":
		out = (input + stride - 1) / stride
		needed := (out-1)*stride + effKernel - input
		if needed < 0 {
			needed = 0
		}
		before = needed / 2
		after = needed - before
	case "valid":
		out = (input-effKernel)/stride + 1
	default:
		before = explicitPad
		after = explicitPad
		out = (input+before+after-effKernel)/stride + 1
	}
	return out, before, after
}

func pad4D(data []float64, N, C, H, W, padTop, padBottom, padLeft, padRight int) []float64 {
	if padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0 {
		return data
	}
	padH := H + padTop + padBottom
	padW := W + padLeft + padRight
	out := make([]float64, N*C*padH*padW)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					srcIdx := n*C*H*W + c*H*W + h*W + w
					dstH := h + padTop
					dstW := w + padLeft
					dstIdx := n*C*padH*padW + c*padH*padW + dstH*padW + dstW
					out[dstIdx] = data[srcIdx]
				}
			}
		}
	}
	return out
}

func im2col(padded []float64, N, C, padH, padW, kH, kW, stride, dilation, outH, outW int) []float64 {
	colRows := C * kH * kW
	colCols := N * outH * outW
	col := make([]float64, colRows*colCols)

	colCol := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				row := 0
				for c := 0; c < C; c++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := oh*stride + kh*dilation
							w := ow*stride + kw*dilation
							srcIdx := n*C*padH*padW + c*padH*padW + h*padW + w
							col[row*colCols+colCol] = padded[srcIdx]
							row++
						}
					}
				}
				colCol++
			}
		}
	}
	return col
}

func col2im(col []float64, N, C, padH, padW, kH, kW, stride, dilation, outH, outW int) []float64 {
	colCols := N * outH * outW
	out := make([]float64, N*C*padH*padW)

	colCol := 0
	for n := 0; n < N; n++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				row := 0
				for c := 0; c < C; c++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := oh*stride + kh*dilation
							w := ow*stride + kw*dilation
							dstIdx := n*C*padH*padW + c*padH*padW + h*padW + w
							out[dstIdx] += col[row*colCols+colCol]
							row++
						}
					}
				}
				colCol++
			}
		}
	}
	return out
}

func unpad4D(padded []float64, N, C, H, W, padH, padW, padTop, padLeft int) *tensor.Tensor {
	out := tensor.Zeros(N, C, H, W)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					srcIdx := n*C*padH*padW + c*padH*padW + (h+padTop)*padW + (w + padLeft)
					dstIdx := n*C*H*W + c*H*W + h*W + w
					out.Data[dstIdx] = padded[srcIdx]
				}
			}
		}
	}
	return out
}

func matToNCHW(mat []float64, N, C, H, W int) []float64 {
	colCols := N * H * W
	out := make([]float64, N*C*H*W)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					col := n*H*W + h*W + w
					out[n*C*H*W+c*H*W+h*W+w] = mat[c*colCols+col]
				}
			}
		}
	}
	return out
}

func nchwToMat(data []float64, N, C, H, W int) []float64 {
	colCols := N * H * W
	mat := make([]float64, C*colCols)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					col := n*H*W + h*W + w
					mat[c*colCols+col] = data[n*C*H*W+c*H*W+h*W+w]
				}
			}
		}
	}
	return mat
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
