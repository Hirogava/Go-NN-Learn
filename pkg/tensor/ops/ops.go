package ops

import "github.com/Hirogava/Go-NN-Learn/pkg/tensor"

func Max(a *tensor.Tensor) *tensor.Tensor {
	if len(a.Data) == 0 {
		return tensor.Zeros(1)
	}
	m := a.Data[0]
	for _, v := range a.Data[1:] {
		if v > m {
			m = v
		}
	}
	result := tensor.Zeros(1)
	result.Data[0] = m
	return result
}

func Sub(a, b *tensor.Tensor) *tensor.Tensor {
	result := tensor.Zeros(a.Shape...)
	bVal := b.Data[0]
	for i, v := range a.Data {
		result.Data[i] = v - bVal
	}
	return result
}

func Div(a, b *tensor.Tensor) *tensor.Tensor {
	result := tensor.Zeros(a.Shape...)
	rows := a.Shape[0]
	cols := 1
	if len(a.Shape) > 1 {
		cols = a.Shape[1]
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*a.Strides[0] + j*a.Strides[1]
			result.Data[idx] = a.Data[idx] / b.Data[i]
		}
	}
	return result
}
