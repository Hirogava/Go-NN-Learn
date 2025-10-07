package matrix

import (
	"errors"
	"fmt"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

func NewMatrix(data [][]float64) (*tensor.Matrix, error) {
	if len(data) == 0 {
		return nil, errors.New("Матрица пустая")
	}

	rows := len(data)
	cols := len(data[0])

	for i := 1; i < rows; i++ {
		if len(data[i]) != cols {
			return nil, errors.New("Матрица имеет столбцы разной длины")
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
		return nil, errors.New("Матрицы пустые")
	}
	rows, colsx2, colsx1 := x1.Rows, x2.Cols, x1.Cols
	if x1.Cols != x2.Rows {
		return nil, errors.New("Матрицы несовместные")
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
		return nil, errors.New("Матрицы пустые")
	}
	rows, colsx2, colsx1 := x1.Rows, x2.Cols, x1.Cols
	if x1.Cols != x2.Rows {
		return nil, errors.New("Матрицы несовместные")
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
		return nil, errors.New("Матрица пустая")
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

func main() {
	X1, err := NewMatrix([][]float64{{9, 8, 6}, {7, 5, 2}, {2, 3, 3}})
	if err != nil {
		fmt.Println("Ошибка создания первой матрицы")
	}
	X2, err := NewMatrix([][]float64{{4, 1, 8}, {10, 2, 7}, {5, 10, 7}})
	if err != nil {
		fmt.Println("Ошибка создания второй матрицы")
	}
	res1, err := MatMul(X1, X2)
	if err != nil {
		fmt.Println("Ошибка:", err)
	} else {
		fmt.Println(res1.Data)
	}
	res2, err := MatMulParallel(X1, X2)
	if err != nil {
		fmt.Println("Ошибка:", err)
	} else {
		fmt.Println(res2.Data)
	}
	res3, err3 := Transposition(X1)
	if err3 != nil {
		fmt.Println("Ошибка:", err3)
	} else {
		fmt.Println(res3.Data)
	}
}
