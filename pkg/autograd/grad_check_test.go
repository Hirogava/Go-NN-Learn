package autograd

import (
	"testing"
)

func TestReLUGradientCheck(t *testing.T) {
	data := []float64{1.0, -0.5, 2.0, -1.7, 0.3}
	if !ReLUGradientCheckOK(data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("ReLU gradient check failed")
	}
}

func TestSigmoidGradientCheck(t *testing.T) {
	data := []float64{-2.0, -0.5, 0.25, 1.3, 2.0}
	if !SigmoidGradientCheckOK(data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("Sigmoid gradient check failed")
	}
}

func TestTanhGradientCheck(t *testing.T) {
	data := []float64{-1.5, -0.2, 0.4, 1.1}
	if !TanhGradientCheckOK(data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("Tanh gradient check failed")
	}
}

func TestSoftPlusGradientCheck(t *testing.T) {
	data := []float64{-4.0, -1.0, 0.0, 1.5, 3.0}
	if !SoftPlusGradientCheckOK(data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("SoftPlus gradient check failed")
	}
}

func TestGELUGradientCheck(t *testing.T) {
	data := []float64{-2.0, -0.5, 0.5, 1.7}
	if !GELUGradientCheckOK(data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("GELU gradient check failed")
	}
}

func TestLeakyReLUGradientCheck(t *testing.T) {
	data := []float64{1.0, -0.5, -2.0, 0.3, -1e-3, 2.5}
	if !LeakyReLUGradientCheckOK(0.01, data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("LeakyReLU gradient check failed")
	}
}

func TestELUGradientCheck(t *testing.T) {
	data := []float64{1.0, -0.5, -2.0, 0.3, -1.2}
	if !ELUGradientCheckOK(1.0, data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("ELU gradient check failed")
	}
}

func TestSoftmaxGradientCheck(t *testing.T) {
	data := []float64{
		0.1, 1.0, -0.2,
		2.0, -1.5, 0.7,
	}
	if !SoftmaxGradientCheckOK(data, []int{2, 3}, 1e-6, 1e-4) {
		t.Fatal("Softmax gradient check failed")
	}
}

func TestSoftmaxCrossEntropyGradientCheck(t *testing.T) {
	logits := []float64{
		1.2, -0.3, 0.4,
		0.1, 1.5, -0.7,
	}
	target := []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
	}
	if !SoftmaxCrossEntropyGradientCheckOK(logits, []int{2, 3}, target, 1e-6, 1e-4) {
		t.Fatal("SoftmaxCrossEntropy gradient check failed")
	}
}