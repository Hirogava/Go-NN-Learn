package graph

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

type noOp struct{}

func (*noOp) Backward(grad *tensor.Tensor) {}

func TestNewNode_NoGrad_DoesNotBuildGraphOrAllocateGrad(t *testing.T) {
	val := &tensor.Tensor{Data: []float64{1, 2}, Shape: []int{2}}
	parent := &Node{Value: val}
	op := (*noOp)(nil) // ненулевой интерфейс для теста

	EnterNoGrad()
	defer ExitNoGrad()

	n := NewNode(val, []*Node{parent}, op)
	if n == nil {
		t.Fatal("NewNode вернул nil")
	}
	if n.Value != val {
		t.Fatal("Value должен быть установлен")
	}
	// DoD: граф не создаётся
	if n.Parents != nil {
		t.Fatalf("В no_grad Parents должен быть nil, получено %v", n.Parents)
	}
	if n.Operation != nil {
		t.Fatalf("В no_grad Operation должен быть nil, получено %v", n.Operation)
	}
	// DoD: нет аллокаций autograd
	if n.Grad != nil {
		t.Fatalf("В no_grad Grad должен быть nil (без аллокаций autograd), получено %v", n.Grad)
	}
}

func TestNewNode_Normal_BuildsGraphAndAllocatesGrad(t *testing.T) {
	val := &tensor.Tensor{Data: []float64{1, 2}, Shape: []int{2}}
	n := NewNode(val, nil, nil)
	if n.Grad == nil {
		t.Fatal("Вне no_grad Grad должен аллоцироваться")
	}
	if len(n.Grad.Data) != 2 {
		t.Fatalf("Неверная форма Grad: получено %d", len(n.Grad.Data))
	}
}

func TestNoGrad_Nesting(t *testing.T) {
	if IsNoGrad() {
		t.Fatal("IsNoGrad должен быть false до EnterNoGrad")
	}
	EnterNoGrad()
	if !IsNoGrad() {
		t.Fatal("IsNoGrad должен быть true после EnterNoGrad")
	}
	EnterNoGrad()
	if !IsNoGrad() {
		t.Fatal("IsNoGrad должен оставаться true после второго EnterNoGrad")
	}
	ExitNoGrad()
	if !IsNoGrad() {
		t.Fatal("IsNoGrad должен оставаться true после первого ExitNoGrad")
	}
	ExitNoGrad()
	if IsNoGrad() {
		t.Fatal("IsNoGrad должен быть false после двух ExitNoGrad")
	}
}
