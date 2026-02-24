package matrix

import (
	"math"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// Тесты корректности

func TestMatMulCorrectness(t *testing.T) {
	tests := []struct {
		name string
		x1   *tensor.Matrix
		x2   *tensor.Matrix
		want []float64
	}{
		{
			name: "2x2 умножение",
			x1: &tensor.Matrix{
				Data: []float64{1, 2, 3, 4},
				Rows: 2,
				Cols: 2,
			},
			x2: &tensor.Matrix{
				Data: []float64{5, 6, 7, 8},
				Rows: 2,
				Cols: 2,
			},
			want: []float64{19, 22, 43, 50},
		},
		{
			name: "3x2 * 2x4",
			x1: &tensor.Matrix{
				Data: []float64{1, 2, 3, 4, 5, 6},
				Rows: 3,
				Cols: 2,
			},
			x2: &tensor.Matrix{
				Data: []float64{1, 2, 3, 4, 5, 6, 7, 8},
				Rows: 2,
				Cols: 4,
			},
			want: []float64{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68},
		},
		{
			name: "1x3 * 3x1 (скалярное произведение)",
			x1: &tensor.Matrix{
				Data: []float64{1, 2, 3},
				Rows: 1,
				Cols: 3,
			},
			x2: &tensor.Matrix{
				Data: []float64{4, 5, 6},
				Rows: 3,
				Cols: 1,
			},
			want: []float64{32},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MatMul(tt.x1, tt.x2)
			if err != nil {
				t.Fatalf("MatMul() error = %v", err)
			}
			if len(got.Data) != len(tt.want) {
				t.Fatalf("MatMul() размер результата = %d, ожидается %d", len(got.Data), len(tt.want))
			}
			for i := range tt.want {
				if math.Abs(got.Data[i]-tt.want[i]) > 1e-9 {
					t.Errorf("MatMul() результат[%d] = %v, ожидается %v", i, got.Data[i], tt.want[i])
				}
			}
		})
	}
}

func TestMatMulErrors(t *testing.T) {
	tests := []struct {
		name string
		x1   *tensor.Matrix
		x2   *tensor.Matrix
	}{
		{
			name: "nil матрицы",
			x1:   nil,
			x2:   &tensor.Matrix{Data: []float64{1}, Rows: 1, Cols: 1},
		},
		{
			name: "несовместимые размеры",
			x1:   &tensor.Matrix{Data: []float64{1, 2, 3, 4}, Rows: 2, Cols: 2},
			x2:   &tensor.Matrix{Data: []float64{1, 2, 3}, Rows: 3, Cols: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := MatMul(tt.x1, tt.x2)
			if err == nil {
				t.Error("MatMul() ожидается ошибка, но получили nil")
			}
		})
	}
}

// Бенчмарки производительности

func BenchmarkMatMulSmall(b *testing.B) {
	// Маленькие матрицы (используется matMulSimple)
	x1 := &tensor.Matrix{
		Data: make([]float64, 32*32),
		Rows: 32,
		Cols: 32,
	}
	x2 := &tensor.Matrix{
		Data: make([]float64, 32*32),
		Rows: 32,
		Cols: 32,
	}
	// Заполняем тестовыми данными
	for i := range x1.Data {
		x1.Data[i] = float64(i)
		x2.Data[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMul(x1, x2)
	}
}

func BenchmarkMatMulMedium(b *testing.B) {
	// Средние матрицы (используется matMulBlocked)
	x1 := &tensor.Matrix{
		Data: make([]float64, 128*256),
		Rows: 128,
		Cols: 256,
	}
	x2 := &tensor.Matrix{
		Data: make([]float64, 256*128),
		Rows: 256,
		Cols: 128,
	}
	for i := range x1.Data {
		x1.Data[i] = float64(i)
	}
	for i := range x2.Data {
		x2.Data[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMul(x1, x2)
	}
}

func BenchmarkMatMulLarge(b *testing.B) {
	// Большие матрицы (используется matMulParallelBlocked)
	// Типичный размер для нейронных сетей: batch_size=128, features=784
	x1 := &tensor.Matrix{
		Data: make([]float64, 128*784),
		Rows: 128,
		Cols: 784,
	}
	x2 := &tensor.Matrix{
		Data: make([]float64, 784*512),
		Rows: 784,
		Cols: 512,
	}
	for i := range x1.Data {
		x1.Data[i] = float64(i)
	}
	for i := range x2.Data {
		x2.Data[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMul(x1, x2)
	}
}

func BenchmarkMatMulVeryLarge(b *testing.B) {
	// Очень большие матрицы
	x1 := &tensor.Matrix{
		Data: make([]float64, 512*1024),
		Rows: 512,
		Cols: 1024,
	}
	x2 := &tensor.Matrix{
		Data: make([]float64, 1024*512),
		Rows: 1024,
		Cols: 512,
	}
	for i := range x1.Data {
		x1.Data[i] = float64(i)
	}
	for i := range x2.Data {
		x2.Data[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMul(x1, x2)
	}
}

// Бенчмарк для сравнения с транспонированием (альтернативный подход)
func BenchmarkMatMulWithTranspose(b *testing.B) {
	x1 := &tensor.Matrix{
		Data: make([]float64, 128*784),
		Rows: 128,
		Cols: 784,
	}
	x2 := &tensor.Matrix{
		Data: make([]float64, 784*512),
		Rows: 784,
		Cols: 512,
	}
	for i := range x1.Data {
		x1.Data[i] = float64(i)
	}
	for i := range x2.Data {
		x2.Data[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x2T, _ := Transposition(x2)
		_, _ = MatMul(x1, x2T)
	}
}

// Бенчмарк транспонирования
func BenchmarkTransposition(b *testing.B) {
	x := &tensor.Matrix{
		Data: make([]float64, 512*1024),
		Rows: 512,
		Cols: 1024,
	}
	for i := range x.Data {
		x.Data[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Transposition(x)
	}
}

// Тест для проверки, что все алгоритмы дают одинаковый результат
func TestMatMulAlgorithmsConsistency(t *testing.T) {
	// Создаем матрицы разных размеров для проверки всех алгоритмов
	sizes := []struct {
		rows1, cols1, rows2, cols2 int
	}{
		{32, 32, 32, 32},   // Маленькая - matMulSimple
		{100, 100, 100, 100}, // Средняя - matMulBlocked
		{200, 300, 300, 200}, // Большая - matMulParallelBlocked
	}

	for _, size := range sizes {
		x1 := &tensor.Matrix{
			Data: make([]float64, size.rows1*size.cols1),
			Rows: size.rows1,
			Cols: size.cols1,
		}
		x2 := &tensor.Matrix{
			Data: make([]float64, size.rows2*size.cols2),
			Rows: size.rows2,
			Cols: size.cols2,
		}

		// Заполняем случайными данными
		for i := range x1.Data {
			x1.Data[i] = float64(i % 100)
		}
		for i := range x2.Data {
			x2.Data[i] = float64((i + 1) % 100)
		}

		// Выполняем умножение
		result, err := MatMul(x1, x2)
		if err != nil {
			t.Fatalf("MatMul() error для размера %dx%d * %dx%d: %v", 
				size.rows1, size.cols1, size.rows2, size.cols2, err)
		}

		// Проверяем размеры результата
		if result.Rows != size.rows1 || result.Cols != size.cols2 {
			t.Errorf("Неверные размеры результата: ожидается %dx%d, получено %dx%d",
				size.rows1, size.cols2, result.Rows, result.Cols)
		}

		// Проверяем, что результат не пустой
		hasNonZero := false
		for _, v := range result.Data {
			if v != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Errorf("Результат умножения пустой для размера %dx%d * %dx%d",
				size.rows1, size.cols1, size.rows2, size.cols2)
		}
	}
}

