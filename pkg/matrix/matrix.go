package matrix

import (
	"errors"
	"runtime"
	"sync"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

const (
	smallMatrixThreshold = 64
	blockSize            = 64
	parallelThreshold    = 128
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

	totalOps := rows * colsx2 * colsx1
	if totalOps < smallMatrixThreshold*smallMatrixThreshold*smallMatrixThreshold {
		return matMulSimple(x1, x2, rows, colsx2, colsx1)
	} else if rows < parallelThreshold && colsx2 < parallelThreshold {
		return matMulBlocked(x1, x2, rows, colsx2, colsx1)
	} else {
		return matMulParallelBlocked(x1, x2, rows, colsx2, colsx1)
	}
}

func matMulSimple(x1, x2 *tensor.Matrix, rows, colsx2, colsx1 int) (*tensor.Matrix, error) {
	res := &tensor.Matrix{
		Data: make([]float64, rows*colsx2),
		Rows: rows,
		Cols: colsx2,
	}

	x1Data := x1.Data
	x2Data := x2.Data
	resData := res.Data
	x1Cols := x1.Cols
	x2Cols := x2.Cols

	for i := 0; i < rows; i++ {
		resRow := i * colsx2
		x1Row := i * x1Cols
		for k := 0; k < colsx1; k++ {
			x1Val := x1Data[x1Row+k]
			x2Row := k * x2Cols
			for j := 0; j < colsx2; j++ {
				resData[resRow+j] += x1Val * x2Data[x2Row+j]
			}
		}
	}

	return res, nil
}

func matMulBlocked(x1, x2 *tensor.Matrix, rows, colsx2, colsx1 int) (*tensor.Matrix, error) {
	res := &tensor.Matrix{
		Data: make([]float64, rows*colsx2),
		Rows: rows,
		Cols: colsx2,
	}

	x1Data := x1.Data
	x2Data := x2.Data
	resData := res.Data
	x1Cols := x1.Cols
	x2Cols := x2.Cols

	for ii := 0; ii < rows; ii += blockSize {
		iEnd := min(ii+blockSize, rows)
		for jj := 0; jj < colsx2; jj += blockSize {
			jEnd := min(jj+blockSize, colsx2)
			for kk := 0; kk < colsx1; kk += blockSize {
				kEnd := min(kk+blockSize, colsx1)

				for i := ii; i < iEnd; i++ {
					resRow := i * colsx2
					x1Row := i * x1Cols
					for k := kk; k < kEnd; k++ {
						x1Val := x1Data[x1Row+k]
						x2Row := k * x2Cols
						for j := jj; j < jEnd; j++ {
							resData[resRow+j] += x1Val * x2Data[x2Row+j]
						}
					}
				}
			}
		}
	}

	return res, nil
}

func matMulParallelBlocked(x1, x2 *tensor.Matrix, rows, colsx2, colsx1 int) (*tensor.Matrix, error) {
	res := &tensor.Matrix{
		Data: make([]float64, rows*colsx2),
		Rows: rows,
		Cols: colsx2,
	}

	x1Data := x1.Data
	x2Data := x2.Data
	resData := res.Data
	x1Cols := x1.Cols
	x2Cols := x2.Cols

	numWorkers := runtime.NumCPU()
	if numWorkers > rows {
		numWorkers = rows
	}

	type task struct {
		startRow, endRow int
	}
	tasks := make(chan task, numWorkers*2)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range tasks {
				for i := t.startRow; i < t.endRow; i++ {
					resRow := i * colsx2
					x1Row := i * x1Cols
					for jj := 0; jj < colsx2; jj += blockSize {
						jEnd := min(jj+blockSize, colsx2)
						for kk := 0; kk < colsx1; kk += blockSize {
							kEnd := min(kk+blockSize, colsx1)
							for k := kk; k < kEnd; k++ {
								x1Val := x1Data[x1Row+k]
								x2Row := k * x2Cols
								for j := jj; j < jEnd; j++ {
									resData[resRow+j] += x1Val * x2Data[x2Row+j]
								}
							}
						}
					}
				}
			}
		}()
	}

	rowsPerTask := max(1, rows/numWorkers)
	for i := 0; i < rows; i += rowsPerTask {
		endRow := min(i+rowsPerTask, rows)
		tasks <- task{startRow: i, endRow: endRow}
	}
	close(tasks)

	wg.Wait()
	return res, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func MatMulParallel(x1, x2 *tensor.Matrix) (*tensor.Matrix, error) {
	return MatMul(x1, x2)
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

	xData := x.Data
	resData := res.Data
	xCols := x.Cols

	for i := 0; i < cols; i++ {
		xRow := i * xCols
		for j := 0; j < rows; j++ {
			resData[j*cols+i] = xData[xRow+j]
		}
	}
	return res, nil
}

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
