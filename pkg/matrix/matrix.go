package matrix

import (
	"errors"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

func NewMatrix(data [][]float64) (*tensor.Matrix, error) {
	if len(data) == 0 {
		return nil, errors.New("матрица пустая")
	}

	rows := len(data)
	cols := len(data[0])

	for i := 1; i < rows; i++ {
		if len(data[i]) != cols {
			return nil, errors.New("матрица имеет столбцы разной длины")
		}
	}

	flatData := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		flatData = append(flatData, data[i]...)
	}

	return &tensor.Matrix{
		Data: flatData,
		Rows: rows,
		Cols: cols,
	}, nil
}

func At(m *tensor.Matrix, i, j int) float64 {
	return m.Data[i*m.Cols+j]
}

func Set(m *tensor.Matrix, i, j int, value float64) {
	m.Data[i*m.Cols+j] = value
}

func MatMul(x1, x2 *tensor.Matrix) (*tensor.Matrix, error) {
	if x1 == nil || x2 == nil {
		return nil, errors.New("матрицы пустые")
	}
	rows, colsx2, colsx1 := x1.Rows, x2.Cols, x1.Cols
	if x1.Cols != x2.Rows {
		return nil, errors.New("матрицы несовместные")
	}
	res := &tensor.Matrix{
		Data: make([]float64, rows*colsx2),
		Rows: rows,
		Cols: colsx2,
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < colsx2; j++ {
			sum := 0.0
			for k := 0; k < colsx1; k++ {
				sum += At(x1, i, k) * At(x2, k, j)
			}
			Set(res, i, j, sum)
		}
	}
	return res, nil
}

func MatMulParallel(x1, x2 *tensor.Matrix) (*tensor.Matrix, error) {
	if x1 == nil || x2 == nil {
		return nil, errors.New("матрицы пустые")
	}
	rows, colsx2, colsx1 := x1.Rows, x2.Cols, x1.Cols
	if x1.Cols != x2.Rows {
		return nil, errors.New("матрицы несовместные")
	}
	res := &tensor.Matrix{
		Data: make([]float64, rows*colsx2),
		Rows: rows,
		Cols: colsx2,
	}
	var wg sync.WaitGroup
	for i := 0; i < rows; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < colsx2; j++ {
				sum := 0.0
				for k := 0; k < colsx1; k++ {
					sum += At(x1, i, k) * At(x2, k, j)
				}
				Set(res, i, j, sum)
			}
		}(i)
	}
	wg.Wait()
	return res, nil
}

func Transposition(x *tensor.Matrix) (*tensor.Matrix, error) {
	if x == nil {
		return nil, errors.New("матрица пустая")
	}
	rows, cols := x.Cols, x.Rows
	res := &tensor.Matrix{
		Data: make([]float64, rows*cols),
		Rows: rows,
		Cols: cols,
	}
	for i := 0; i < cols; i++ {
		for j := 0; j < rows; j++ {
			Set(res, j, i, At(x, i, j))
		}
	}
	return res, nil
}

// Конверторы между Matrix и Tensor
func MatrixToTensor(m *tensor.Matrix) *tensor.Tensor {
	return &tensor.Tensor{
		Data:    m.Data,
		Shape:   []int{m.Rows, m.Cols},
		Strides: []int{m.Cols, 1},
	}
}

func TensorToMatrix(t *tensor.Tensor) *tensor.Matrix {
	return &tensor.Matrix{
		Data: t.Data,
		Rows: t.Shape[0],
		Cols: t.Shape[1],
	}
}
