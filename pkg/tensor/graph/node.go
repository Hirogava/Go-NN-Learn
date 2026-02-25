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
	n.Grad = tensor.Zeros(n.Value.Shape...)
}

func (n *Node) Prune() {
	if n == nil {
		return
	}

	visited := make(map[*Node]struct{})
	stack := []*Node{n}

	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if _, ok := visited[cur]; ok {
			continue
		}
		visited[cur] = struct{}{}

		// Добавляем родителей в обход
		for _, p := range cur.Parents {
			stack = append(stack, p)
		}

		if n.Value == nil {
			cur.Parents = nil
			cur.Operation = nil
		}
	}
}
