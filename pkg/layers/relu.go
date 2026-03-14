package layers

// import (
// 	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
// 	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
// )

// // ReLU — слой-обёртка, применяющий autograd.Engine().ReLU
// type ReLU struct{}

// // NewReLU возвращает новый ReLU слой
// func NewReLU() *ReLU {
// 	return &ReLU{}
// }

// // Forward применяет ReLU через текущий autograd graph/engine.
// // Требует, чтобы перед вызовом был установлен граф (autograd.SetGraph).
// func (r *ReLU) Forward(x *graph.Node) *graph.Node {
// 	// Если граф не установлен, создаём временный NoGrad граф для безопасного вызова.
// 	g := autograd.GetGraph()
// 	if g == nil {
// 		ctx := autograd.NewGraph()
// 		ctx.NoGrad()
// 		autograd.SetGraph(ctx)
// 		// создаём выход и очищаем граф в конце
// 		out := autograd.GetGraph().Engine().ReLU(x)
// 		autograd.ClearGraph()
// 		return out
// 	}
// 	return g.Engine().ReLU(x)
// }

// // ReLU не имеет параметров
// func (r *ReLU) Params() []*graph.Node { return []*graph.Node{} }

// // Train — no-op
// func (r *ReLU) Train() {}

// // Eval — no-op
// func (r *ReLU) Eval() {}