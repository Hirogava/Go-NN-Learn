package gnn

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// NoGrad выполняет f с отключённым autograd: внутри f вызов graph.NewNode
// не строит граф (нет Parents и Operation) и не аллоцирует тензоры Grad.
// Используйте при инференсе, чтобы не строить граф вычислений и не аллоцировать под autograd:
//
//	gnn.NoGrad(func() {
//	    out := api.Predict(model, x)
//	    // out.Value валиден; out.Grad == nil, out.Parents/Operation == nil
//	})
//
// DoD: граф не создаётся; нет аллокаций autograd.
// Не безопасно для конкурентного вызова из нескольких горутин (состояние no_grad глобальное).
func NoGrad(f func()) {
	graph.EnterNoGrad()
	defer graph.ExitNoGrad()
	f()
}
