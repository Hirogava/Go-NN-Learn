package autograd_test

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

func feq(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// маленький помощник для запуска backward
func runBackward(e *autograd.Engine, out *graph.Node) {
	if len(out.Value.Shape) == 1 && out.Value.Shape[0] == 1 {
		out.Grad = &tensor.Tensor{Data: []float64{1}, Shape: []int{1}, Strides: []int{1}}
	} else {
		out.Grad = tensor.Ones(out.Value.Shape...)
	}
	// обходим узлы в обратном порядке
	for i := len(e.Nodes) - 1; i >= 0; i-- {
		n := e.Nodes[i]
		if n.Operation != nil && n.Grad != nil {
			n.Operation.Backward(n.Grad)
		}
	}
}

func TestAdd(t *testing.T) {
	e := &autograd.Engine{}
	a := graph.NewNode(&tensor.Tensor{Data: []float64{1, 2}, Shape: []int{2}, Strides: []int{1}}, nil, nil)
	b := graph.NewNode(&tensor.Tensor{Data: []float64{3, 4}, Shape: []int{2}, Strides: []int{1}}, nil, nil)

	y := e.Add(a, b)
	if !feq(y.Value.Data[0], 4) || !feq(y.Value.Data[1], 6) {
		t.Fatal("add forward")
	}

	runBackward(e, y)
	if !feq(a.Grad.Data[0], 1) || !feq(a.Grad.Data[1], 1) {
		t.Fatal("add dA")
	}
	if !feq(b.Grad.Data[0], 1) || !feq(b.Grad.Data[1], 1) {
		t.Fatal("add dB")
	}
}

func TestMul(t *testing.T) {
	e := &autograd.Engine{}
	a := graph.NewNode(&tensor.Tensor{Data: []float64{2, 5}, Shape: []int{2}, Strides: []int{1}}, nil, nil)
	b := graph.NewNode(&tensor.Tensor{Data: []float64{3, 7}, Shape: []int{2}, Strides: []int{1}}, nil, nil)

	y := e.Mul(a, b)
	if !feq(y.Value.Data[0], 6) || !feq(y.Value.Data[1], 35) {
		t.Fatal("mul forward")
	}

	runBackward(e, y)
	// dA = B, dB = A
	if !feq(a.Grad.Data[0], 3) || !feq(a.Grad.Data[1], 7) {
		t.Fatal("mul dA")
	}
	if !feq(b.Grad.Data[0], 2) || !feq(b.Grad.Data[1], 5) {
		t.Fatal("mul dB")
	}
}

func TestTranspose(t *testing.T) {
	e := &autograd.Engine{}
	a := graph.NewNode(&tensor.Tensor{
		Data:    []float64{1, 2, 3, 4},
		Shape:   []int{2, 2},
		Strides: []int{2, 1},
	}, nil, nil)

	y := e.Transpose(a)
	// forward: [[1,2],[3,4]]^T = [[1,3],[2,4]]
	want := []float64{1, 3, 2, 4}
	for i := range want {
		if !feq(y.Value.Data[i], want[i]) {
			t.Fatal("transpose forward")
		}
	}

	runBackward(e, y)
	// upstream ones(2x2) -> grad_in = ones^T = ones(2x2)
	for _, v := range a.Grad.Data {
		if !feq(v, 1) {
			t.Fatal("transpose backward")
		}
	}
}

func TestMatMul(t *testing.T) {
	e := &autograd.Engine{}
	A := graph.NewNode(&tensor.Tensor{
		Data:  []float64{1, 2, 3, 4}, // 2x2
		Shape: []int{2, 2}, Strides: []int{2, 1},
	}, nil, nil)
	B := graph.NewNode(&tensor.Tensor{
		Data:  []float64{5, 6, 7, 8}, // 2x2
		Shape: []int{2, 2}, Strides: []int{2, 1},
	}, nil, nil)

	C := e.MatMul(A, B)
	// forward: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
	want := []float64{19, 22, 43, 50}
	for i := range want {
		if !feq(C.Value.Data[i], want[i]) {
			t.Fatal("matmul forward")
		}
	}

	runBackward(e, C)
	// слабая проверка: формы и ненулевой градиент
	if len(A.Grad.Data) != 4 || len(B.Grad.Data) != 4 {
		t.Fatal("matmul grads size")
	}
}

func TestSum(t *testing.T) {
	e := &autograd.Engine{}
	x := graph.NewNode(&tensor.Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}, Strides: []int{1}}, nil, nil)

	y := e.Sum(x)
	if len(y.Value.Data) != 1 || !feq(y.Value.Data[0], 6) {
		t.Fatal("sum forward")
	}

	// скалярный upstream = 1
	y.Grad = &tensor.Tensor{Data: []float64{1}, Shape: []int{1}, Strides: []int{1}}
	// ручной бэкпроход: только последний узел (sum)
	for i := len(e.Nodes) - 1; i >= 0; i-- {
		n := e.Nodes[i]
		if n == y && n.Operation != nil {
			n.Operation.Backward(n.Grad)
			break
		}
	}
	if len(x.Grad.Data) != 3 {
		t.Fatal("sum dX size")
	}
	for _, v := range x.Grad.Data {
		if !feq(v, 1) {
			t.Fatal("sum backward")
		}
	}
}

func TestExp(t *testing.T) {
	e := &autograd.Engine{}
	x := graph.NewNode(&tensor.Tensor{Data: []float64{0, 1}, Shape: []int{2}, Strides: []int{1}}, nil, nil)

	y := e.Exp(x)
	if !feq(y.Value.Data[0], 1) || !feq(y.Value.Data[1], math.E) {
		t.Fatal("exp forward")
	}

	runBackward(e, y)
	// dX = exp(x)
	if !feq(x.Grad.Data[0], 1) || !feq(x.Grad.Data[1], math.E) {
		t.Fatal("exp backward")
	}
}

func TestLog(t *testing.T) {
	e := &autograd.Engine{}
	x := graph.NewNode(&tensor.Tensor{Data: []float64{1, 4}, Shape: []int{2}, Strides: []int{1}}, nil, nil)

	y := e.Log(x)
	if !feq(y.Value.Data[0], 0) || !feq(y.Value.Data[1], math.Log(4)) {
		t.Fatal("log forward")
	}

	runBackward(e, y)
	// dX = 1/x
	if !feq(x.Grad.Data[0], 1) || !feq(x.Grad.Data[1], 0.25) {
		t.Fatal("log backward")
	}
}

func TestReshape(t *testing.T) {
	e := &autograd.Engine{}
	x := graph.NewNode(&tensor.Tensor{
		Data:  []float64{1, 2, 3, 4, 5, 6},
		Shape: []int{2, 3}, Strides: []int{3, 1},
	}, nil, nil)

	y := e.Reshape(x, []int{3, 2})
	if y == nil {
		t.Fatal("reshape forward nil")
	}
	if y.Value.Shape[0] != 3 || y.Value.Shape[1] != 2 {
		t.Fatal("reshape forward shape")
	}
	// данные те же
	for i, v := range x.Value.Data {
		if !feq(v, y.Value.Data[i]) {
			t.Fatal("reshape data")
		}
	}

	runBackward(e, y)
	// дойдём до reshape и проверим grad формы x
	if x.Grad == nil || x.Grad.Shape[0] != 2 || x.Grad.Shape[1] != 3 {
		t.Fatal("reshape backward shape")
	}
	for _, v := range x.Grad.Data {
		if !feq(v, 1) {
			t.Fatal("reshape backward values")
		}
	}
}
