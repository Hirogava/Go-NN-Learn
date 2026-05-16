package layers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type ReLULayer struct {
	engine *autograd.Engine
}

func NewReLULayer(engine *autograd.Engine) *ReLULayer {
	return &ReLULayer{engine: engine}
}

func (l *ReLULayer) Forward(x *graph.Node) *graph.Node { return l.engine.ReLU(x) }
func (l *ReLULayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *ReLULayer) Train()                            {}
func (l *ReLULayer) Eval()                             {}

type SigmoidLayer struct {
	engine *autograd.Engine
}

func NewSigmoidLayer(engine *autograd.Engine) *SigmoidLayer {
	return &SigmoidLayer{engine: engine}
}

func (l *SigmoidLayer) Forward(x *graph.Node) *graph.Node { return l.engine.Sigmoid(x) }
func (l *SigmoidLayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *SigmoidLayer) Train()                            {}
func (l *SigmoidLayer) Eval()                             {}

type TanhLayer struct {
	engine *autograd.Engine
}

func NewTanhLayer(engine *autograd.Engine) *TanhLayer {
	return &TanhLayer{engine: engine}
}

func (l *TanhLayer) Forward(x *graph.Node) *graph.Node { return l.engine.Tanh(x) }
func (l *TanhLayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *TanhLayer) Train()                            {}
func (l *TanhLayer) Eval()                             {}

type LeakyReLULayer struct {
	engine *autograd.Engine
	slope  float64
}

func NewLeakyReLULayer(engine *autograd.Engine, slope float64) *LeakyReLULayer {
	return &LeakyReLULayer{engine: engine, slope: slope}
}

func (l *LeakyReLULayer) Forward(x *graph.Node) *graph.Node {
	return l.engine.LeakyReLU(x, l.slope)
}
func (l *LeakyReLULayer) Params() []*graph.Node { return []*graph.Node{} }
func (l *LeakyReLULayer) Train()                {}
func (l *LeakyReLULayer) Eval()                 {}

type ELULayer struct {
	engine *autograd.Engine
	alpha  float64
}

func NewELULayer(engine *autograd.Engine, alpha float64) *ELULayer {
	return &ELULayer{engine: engine, alpha: alpha}
}

func (l *ELULayer) Forward(x *graph.Node) *graph.Node { return l.engine.ELU(x, l.alpha) }
func (l *ELULayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *ELULayer) Train()                            {}
func (l *ELULayer) Eval()                             {}

type SoftPlusLayer struct {
	engine *autograd.Engine
}

func NewSoftPlusLayer(engine *autograd.Engine) *SoftPlusLayer {
	return &SoftPlusLayer{engine: engine}
}

func (l *SoftPlusLayer) Forward(x *graph.Node) *graph.Node { return l.engine.SoftPlus(x) }
func (l *SoftPlusLayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *SoftPlusLayer) Train()                            {}
func (l *SoftPlusLayer) Eval()                             {}

type GELULayer struct {
	engine *autograd.Engine
}

func NewGELULayer(engine *autograd.Engine) *GELULayer {
	return &GELULayer{engine: engine}
}

func (l *GELULayer) Forward(x *graph.Node) *graph.Node { return l.engine.GELU(x) }
func (l *GELULayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *GELULayer) Train()                            {}
func (l *GELULayer) Eval()                             {}

type SoftmaxLayer struct {
	engine *autograd.Engine
}

func NewSoftmaxLayer(engine *autograd.Engine) *SoftmaxLayer {
	return &SoftmaxLayer{engine: engine}
}

func (l *SoftmaxLayer) Forward(x *graph.Node) *graph.Node { return l.engine.Softmax(x) }
func (l *SoftmaxLayer) Params() []*graph.Node             { return []*graph.Node{} }
func (l *SoftmaxLayer) Train()                            {}
func (l *SoftmaxLayer) Eval()                             {}
