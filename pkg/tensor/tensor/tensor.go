package tensor

import "fmt"

type Tensor struct {
	Data  []float64
	Shape []int
}

func New(data []float64, shape []int) *Tensor {
	return &Tensor{
		Data:  data,
		Shape: shape,
	}
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v)", t.Shape)
}

func (t *Tensor) ZeroGrad() *Tensor {
	gradData := make([]float64, len(t.Data))
	return &Tensor{
		Data:  gradData,
		Shape: t.Shape,
	}
}

func (t *Tensor) Clone() *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	return &Tensor{
		Data:  data,
		Shape: append([]int{}, t.Shape...),
	}
}
