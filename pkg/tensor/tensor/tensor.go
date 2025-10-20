package tensor

// Vector представляет одномерный тензор (вектор).
// Это просто слайс float64 для работы с одномерными данными.
type Vector []float64

// Matrix представляет двумерный тензор (матрицу).
// Data хранится в row-major порядке: элементы одной строки идут последовательно.
// Индексация: Data[row*Cols + col]
type Matrix struct {
	Data []float64
	Rows int
	Cols int
}

// Tensor представляет N-мерный тензор.
// Data - плоский массив данных.
// Shape - размерность по каждой оси.
// Strides - количество элементов для перехода к следующему элементу по оси.
// Strides позволяют эффективно работать с транспонированием и подтензорами без копирования.
type Tensor struct {
	Data    []float64
	Shape   []int
	Strides []int
}

// ZeroGrad создает тензор с нулевыми градиентами той же формы
func (t *Tensor) ZeroGrad() *Tensor {
	return &Tensor{
		Data:    make([]float64, len(t.Data)),
		Shape:   append([]int{}, t.Shape...),
		Strides: append([]int{}, t.Strides...),
	}
}

// Size возвращает общее количество элементов в тензоре
func (t *Tensor) Size() int64 {
	return int64(len(t.Data))
}
