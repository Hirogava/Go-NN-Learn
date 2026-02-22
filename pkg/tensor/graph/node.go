package graph

import (
	"sync/atomic"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

type Node struct {
	Value *tensor.Tensor

	Grad *tensor.Tensor

	Parents []*Node

	Operation Operation

	ID string
}

type BackwardFunc func(grad *tensor.Tensor)

type Operation interface {
	Backward(grad *tensor.Tensor)
}

// noGradDepth — счётчик вложенности no_grad для текущей горутины.
// При > 0 NewNode не аллоцирует Grad и не строит граф (нет Parents и Operation).
var noGradDepth atomic.Uint32

// EnterNoGrad увеличивает счётчик no_grad. Вызывается из gnn.NoGrad.
func EnterNoGrad() { noGradDepth.Add(1) }

// ExitNoGrad уменьшает счётчик no_grad.
func ExitNoGrad() { noGradDepth.Add(^uint32(0)) }

// IsNoGrad возвращает true, когда мы внутри блока NoGrad.
func IsNoGrad() bool { return noGradDepth.Load() > 0 }

func NewNode(value *tensor.Tensor, parents []*Node, op Operation) *Node {
	if IsNoGrad() {
		return &Node{
			Value:     value,
			Grad:      nil,
			Parents:   nil,
			Operation: nil,
		}
	}
	return &Node{
		Value:     value,
		Grad:      tensor.Zeros(value.Shape...),
		Parents:   parents,
		Operation: op,
	}
}

func (n *Node) IsLeaf() bool {
	return len(n.Parents) == 0
}

func (n *Node) ZeroGrad() {
	n.Grad = tensor.Zeros(n.Value.Shape...)
}
