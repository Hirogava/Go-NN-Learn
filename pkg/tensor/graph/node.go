<<<<<<< HEAD
package graph

import (
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
	Forward(inputs ...*Node) *Node
	Backward(grad *tensor.Tensor)
}

func NewNode(value *tensor.Tensor, parents []*Node, op Operation) *Node {
	return &Node{
		Value:     value,
		Grad:      value.ZeroGrad(),
		Parents:   parents,
		Operation: op,
	}
}

func (n *Node) IsLeaf() bool {
	return len(n.Parents) == 0
}

func (n *Node) ZeroGrad() {
	n.Grad = n.Value.ZeroGrad()
}
=======
package graph

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
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

func NewNode(value *tensor.Tensor, parents []*Node, op Operation) *Node {
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
	n.Grad = tensor.Zeros(n.Value.Shape...) // Ватафак чел ты чо курил // согласен ©shz
}
>>>>>>> main
