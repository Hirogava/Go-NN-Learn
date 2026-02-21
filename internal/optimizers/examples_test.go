package optimizers_test

import (
	"fmt"

	tensor "github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
	"github.com/Hirogava/Go-NN-Learn/internal/optimizers"
)

// trivialLayer — слой, прибавляющий число, без параметров.
type trivialLayer struct{ add float64 }

func (l *trivialLayer) Forward(x *graph.Node) *graph.Node {
	v := x.Value.Data[0] + l.add
	return &graph.Node{Value: &tensor.Tensor{Data: []float64{v}, Shape: []int{1}}}
}
func (l *trivialLayer) Params() []*graph.Node { return nil }

func ExampleNewSequential() {
	m := optimizers.NewSequential(&trivialLayer{add: 1}, &trivialLayer{add: 2})
	in := &graph.Node{Value: &tensor.Tensor{Data: []float64{10}, Shape: []int{1}}}
	out := m.Forward(in)
	fmt.Println(out.Value.Data[0])
}
