package autograd

import (
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

func TestLeakyReLUGradientCheck_vector(t *testing.T) {
	const slope = 0.01
	// Без x = 0: см. комментарий к LeakyReLUGradientCheckOK.
	data := []float64{1.0, -0.5, -2.0, 0.3, -1e-3, 2.5}
	if !LeakyReLUGradientCheckOK(slope, data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("LeakyReLU gradient check failed (vector, slope=0.01)")
	}
}

func TestLeakyReLUGradientCheck_otherSlope(t *testing.T) {
	const slope = 0.2
	data := []float64{-1.0, 0.5, 3.0, -3e-2}
	if !LeakyReLUGradientCheckOK(slope, data, []int{len(data)}, 1e-6, 1e-4) {
		t.Fatal("LeakyReLU gradient check failed (slope=0.2)")
	}
}

func TestLeakyReLUGradientCheck_matrix(t *testing.T) {
	x := tensor.Randn([]int{4, 5}, 42)
	if !LeakyReLUGradientCheckOK(0.01, x.Data, x.Shape, 1e-6, 1e-4) {
		t.Fatal("LeakyReLU gradient check failed (matrix)")
	}
}
