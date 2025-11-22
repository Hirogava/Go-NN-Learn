package optimizers_test

import (
	"fmt"

	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// trivialLayer — layer с нулевыми Params, прибавляющий константу.
type trivialLayer struct {
	add float64
}

func (l *trivialLayer) Forward(x *graph.Node) *graph.Node {
	v := x.Value.Data[0] + l.add
	return &graph.Node{Value: &tensor.Tensor{Data: []float64{v}, Shape: []int{1}}}
}
func (l *trivialLayer) Params() []*graph.Node { return nil }

func ExampleNewSequential() {
	m := optimizers.NewSequential(&trivialLayer{add: 1.0}, &trivialLayer{add: 2.0})
	in := &graph.Node{Value: &tensor.Tensor{Data: []float64{10.0}, Shape: []int{1}}}
	out := m.Forward(in)
	fmt.Println(out.Value.Data[0])
	// Output: 13
}
