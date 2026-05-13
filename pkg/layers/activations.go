package layers

import (
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// ReLULayer оборачивает ReLU активацию в интерфейс Layer
type ReLULayer struct{}

func NewReLU() *ReLULayer {
	return &ReLULayer{}
}

func (l *ReLULayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().ReLU(x)
}

func (l *ReLULayer) Params() []*graph.Node { return nil }
func (l *ReLULayer) Train()                {}
func (l *ReLULayer) Eval()                 {}

// SigmoidLayer оборачивает Sigmoid активацию
type SigmoidLayer struct{}

func NewSigmoid() *SigmoidLayer {
	return &SigmoidLayer{}
}

func (l *SigmoidLayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().Sigmoid(x)
}

func (l *SigmoidLayer) Params() []*graph.Node { return nil }
func (l *SigmoidLayer) Train()                {}
func (l *SigmoidLayer) Eval()                 {}

// TanhLayer оборачивает Tanh активацию
type TanhLayer struct{}

func NewTanh() *TanhLayer {
	return &TanhLayer{}
}

func (l *TanhLayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().Tanh(x)
}

func (l *TanhLayer) Params() []*graph.Node { return nil }
func (l *TanhLayer) Train()                {}
func (l *TanhLayer) Eval()                 {}

// LeakyReLULayer оборачивает LeakyReLU активацию
type LeakyReLULayer struct {
	Slope float64
}

func NewLeakyReLU(slope float64) *LeakyReLULayer {
	return &LeakyReLULayer{Slope: slope}
}

func (l *LeakyReLULayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().LeakyReLU(x, l.Slope)
}

func (l *LeakyReLULayer) Params() []*graph.Node { return nil }
func (l *LeakyReLULayer) Train()                {}
func (l *LeakyReLULayer) Eval()                 {}

// ELULayer оборачивает ELU активацию
type ELULayer struct {
	Alpha float64
}

func NewELU(alpha float64) *ELULayer {
	return &ELULayer{Alpha: alpha}
}

func (l *ELULayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().ELU(x, l.Alpha)
}

func (l *ELULayer) Params() []*graph.Node { return nil }
func (l *ELULayer) Train()                {}
func (l *ELULayer) Eval()                 {}

// SoftPlusLayer оборачивает SoftPlus активацию
type SoftPlusLayer struct{}

func NewSoftPlus() *SoftPlusLayer {
	return &SoftPlusLayer{}
}

func (l *SoftPlusLayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().SoftPlus(x)
}

func (l *SoftPlusLayer) Params() []*graph.Node { return nil }
func (l *SoftPlusLayer) Train()                {}
func (l *SoftPlusLayer) Eval()                 {}

// GELULayer оборачивает GELU активацию
type GELULayer struct{}

func NewGELU() *GELULayer {
	return &GELULayer{}
}

func (l *GELULayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().GELU(x)
}

func (l *GELULayer) Params() []*graph.Node { return nil }
func (l *GELULayer) Train()                {}
func (l *GELULayer) Eval()                 {}

// SoftmaxLayer оборачивает Softmax активацию
type SoftmaxLayer struct{}

func NewSoftmax() *SoftmaxLayer {
	return &SoftmaxLayer{}
}

func (l *SoftmaxLayer) Forward(x *graph.Node) *graph.Node {
	return autograd.GetGraph().Engine().Softmax(x)
}

func (l *SoftmaxLayer) Params() []*graph.Node { return nil }
func (l *SoftmaxLayer) Train()                {}
func (l *SoftmaxLayer) Eval()                 {}
