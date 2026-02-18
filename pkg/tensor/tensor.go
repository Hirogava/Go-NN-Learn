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
	Data    []float64 // Данные (для Float64) - row-major (C-style)
	Data32  []float32 // Данные (для Float32)
	Shape   []int     // Размерности (форма) тензора
	Strides []int     // Шаги для перехода по размерностям
	DType   DType     // Тип данных
}

// IsFloat32 возвращает true, если тензор использует float32
func (t *Tensor) IsFloat32() bool {
	return t.DType == Float32
}

// DataLen возвращает длину активного среза данных
func (t *Tensor) DataLen() int {
	if t.DType == Float32 {
		return len(t.Data32)
	}
	return len(t.Data)
}

// ZeroGrad создает тензор с нулевыми градиентами той же формы
func (t *Tensor) ZeroGrad() *Tensor {
	out := &Tensor{
		Shape:   append([]int{}, t.Shape...),
		Strides: append([]int{}, t.Strides...),
		DType:   t.DType,
	}
	if t.DType == Float32 {
		out.Data32 = make([]float32, t.DataLen())
	} else {
		out.Data = make([]float64, t.DataLen())
	}
	return out
}

// Size возвращает общее количество элементов в тензоре
func (t *Tensor) Size() int64 {
	return int64(t.DataLen())
}
