package layers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/matrix"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
<<<<<<< HEAD
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
=======
>>>>>>> origin/main
)

type Dense struct {
	weights *graph.Node
	bias    *graph.Node
	inDim   int
	outDim  int
}

func NewDense(inDim, outDim int, initFunc func([]float64)) *Dense {
	wData := make([]float64, inDim*outDim)
	initFunc(wData)
	weights := &graph.Node{
		Value: &tensor.Tensor{
			Data:    wData,
			Shape:   []int{inDim, outDim},
			Strides: []int{outDim, 1},
		},
	}

	bData := make([]float64, outDim)
	initFunc(bData)
	bias := &graph.Node{
		Value: &tensor.Tensor{
			Data:    bData,
			Shape:   []int{outDim},
			Strides: []int{1},
		},
	}

	return &Dense{
		weights: weights,
		bias:    bias,
		inDim:   inDim,
		outDim:  outDim,
	}
}

func (d *Dense) Forward(x *graph.Node) *graph.Node {
	xTensor := x.Value
	var xMat *tensor.Matrix
	if len(xTensor.Shape) == 1 {
		xMat = &tensor.Matrix{
			Data: xTensor.Data,
			Rows: 1,
			Cols: xTensor.Shape[0],
		}
	} else if len(xTensor.Shape) == 2 {
		xMat = &tensor.Matrix{
			Data: xTensor.Data,
			Rows: xTensor.Shape[0],
			Cols: xTensor.Shape[1],
		}
	} else {
		panic("Dense layer expects 1D or 2D tensor input")
	}

	if xMat.Cols != d.inDim {
		panic("Input dimension mismatch")
	}

	wTensor := d.weights.Value
	wMat := &tensor.Matrix{
		Data: wTensor.Data,
		Rows: wTensor.Shape[0],
		Cols: wTensor.Shape[1],
	}

	outMat, err := matrix.MatMul(xMat, wMat)
	if err != nil {
		panic("Matrix multiplication failed: " + err.Error())
	}

	bTensor := d.bias.Value
	bVec := tensor.Vector(bTensor.Data)
	for i := 0; i < outMat.Rows; i++ {
		for j := 0; j < outMat.Cols; j++ {
			outMat.Data[i*outMat.Cols+j] += bVec[j]
		}
	}

	out := &graph.Node{
		Value: &tensor.Tensor{
			Data:    outMat.Data,
			Shape:   []int{outMat.Rows, outMat.Cols},
			Strides: []int{outMat.Cols, 1},
		},
	}
<<<<<<< HEAD

	dOp := &denseOp{x: x, w: d.weights, b: d.bias}
	out.Operation = &backwardOp{dOp.Backward}

=======
	if autograd.GradEnabled() {
		out.Operation = &denseOp{
			x: x,
			w: d.weights,
			b: d.bias,
		}
	}
>>>>>>> origin/main
	return out
}

func (d *Dense) Params() []*graph.Node {
	return []*graph.Node{d.weights, d.bias}
}

type denseOp struct {
	x *graph.Node
	w *graph.Node
	b *graph.Node
}

func (op *denseOp) Backward(grad *tensor.Tensor) {
	gradMat := &tensor.Matrix{
		Data: grad.Data,
		Rows: grad.Shape[0],
		Cols: grad.Shape[1],
	}

	xTensor := op.x.Value
	xRows := 1
	xCols := xTensor.Shape[0]
	if len(xTensor.Shape) == 2 {
		xRows = xTensor.Shape[0]
		xCols = xTensor.Shape[1]
	}
	xMat := &tensor.Matrix{
		Data: xTensor.Data,
		Rows: xRows,
		Cols: xCols,
	}

	wTensor := op.w.Value
	wMat := &tensor.Matrix{
		Data: wTensor.Data,
		Rows: wTensor.Shape[0],
		Cols: wTensor.Shape[1],
	}

	wMatT, err := matrix.Transposition(wMat)
	if err != nil {
		panic("Transposition failed: " + err.Error())
	}
	xGradMat, err := matrix.MatMul(gradMat, wMatT)
	if err != nil {
		panic("Matrix multiplication failed: " + err.Error())
	}
	op.x.Grad = &tensor.Tensor{
		Data:    xGradMat.Data,
		Shape:   []int{xGradMat.Rows, xGradMat.Cols},
		Strides: []int{xGradMat.Cols, 1},
	}

	xMatT, err := matrix.Transposition(xMat)
	if err != nil {
		panic("Transposition failed: " + err.Error())
	}
	wGradMat, err := matrix.MatMul(xMatT, gradMat)
	if err != nil {
		panic("Matrix multiplication failed: " + err.Error())
	}
	op.w.Grad = &tensor.Tensor{
		Data:    wGradMat.Data,
		Shape:   []int{wGradMat.Rows, wGradMat.Cols},
		Strides: []int{wGradMat.Cols, 1},
	}

	bGrad := make([]float64, gradMat.Cols)
	for j := 0; j < gradMat.Cols; j++ {
		sum := 0.0
		for i := 0; i < gradMat.Rows; i++ {
			sum += gradMat.Data[i*gradMat.Cols+j]
		}
		bGrad[j] = sum
	}
	op.b.Grad = &tensor.Tensor{
		Data:    bGrad,
		Shape:   []int{len(bGrad)},
		Strides: []int{1},
	}
}
