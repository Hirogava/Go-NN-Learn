package backend

import (
	"math"
	"testing"
)

func TestZeros(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{"1D vector", []int{5}},
		{"2D matrix", []int{3, 4}},
		{"3D tensor", []int{2, 3, 4}},
		{"empty shape", []int{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Zeros(tt.shape...)

			// Проверка формы
			if len(result.Shape) != len(tt.shape) {
				t.Errorf("Shape length = %d, want %d", len(result.Shape), len(tt.shape))
			}
			for i := range tt.shape {
				if result.Shape[i] != tt.shape[i] {
					t.Errorf("Shape[%d] = %d, want %d", i, result.Shape[i], tt.shape[i])
				}
			}

			// Проверка что все элементы = 0
			for i, val := range result.Data {
				if val != 0.0 {
					t.Errorf("Data[%d] = %v, want 0.0", i, val)
				}
			}

			// Проверка размера данных
			expectedSize := calculateSize(tt.shape)
			if len(result.Data) != expectedSize {
				t.Errorf("Data length = %d, want %d", len(result.Data), expectedSize)
			}

			// Проверка strides
			expectedStrides := calculateStrides(tt.shape)
			if len(result.Strides) != len(expectedStrides) {
				t.Errorf("Strides length = %d, want %d", len(result.Strides), len(expectedStrides))
			}
			for i := range expectedStrides {
				if result.Strides[i] != expectedStrides[i] {
					t.Errorf("Strides[%d] = %d, want %d", i, result.Strides[i], expectedStrides[i])
				}
			}
		})
	}
}

func TestOnes(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{"1D vector", []int{5}},
		{"2D matrix", []int{3, 4}},
		{"3D tensor", []int{2, 3, 4}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Ones(tt.shape...)

			// Проверка формы
			if len(result.Shape) != len(tt.shape) {
				t.Errorf("Shape length = %d, want %d", len(result.Shape), len(tt.shape))
			}

			// Проверка что все элементы = 1.0
			for i, val := range result.Data {
				if val != 1.0 {
					t.Errorf("Data[%d] = %v, want 1.0", i, val)
				}
			}

			// Проверка размера данных
			expectedSize := calculateSize(tt.shape)
			if len(result.Data) != expectedSize {
				t.Errorf("Data length = %d, want %d", len(result.Data), expectedSize)
			}
		})
	}
}

func TestRandn(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		seed  int64
	}{
		{"1D vector", []int{5}, 42},
		{"2D matrix", []int{3, 4}, 123},
		{"3D tensor", []int{2, 3, 4}, 456},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Randn(tt.shape, tt.seed)

			// Проверка формы
			if len(result.Shape) != len(tt.shape) {
				t.Errorf("Shape length = %d, want %d", len(result.Shape), len(tt.shape))
			}

			// Проверка размера данных
			expectedSize := calculateSize(tt.shape)
			if len(result.Data) != expectedSize {
				t.Errorf("Data length = %d, want %d", len(result.Data), expectedSize)
			}

			// Проверка что не все элементы равны
			allSame := true
			first := result.Data[0]
			for _, val := range result.Data[1:] {
				if val != first {
					allSame = false
					break
				}
			}
			if allSame && len(result.Data) > 1 {
				t.Error("All elements are the same - random initialization failed")
			}

			// Проверка что значения из разумного диапазона для N(0,1)
			// ~99.7% значений должны быть в пределах [-3, 3]
			for i, val := range result.Data {
				if math.Abs(val) > 5.0 {
					t.Errorf("Data[%d] = %v is too far from normal distribution", i, val)
				}
			}
		})
	}
}

func TestRandnReproducibility(t *testing.T) {
	// Проверка воспроизводимости: одинаковый seed должен давать одинаковый результат
	shape := []int{3, 4}
	seed := int64(42)

	t1 := Randn(shape, seed)
	t2 := Randn(shape, seed)

	for i := range t1.Data {
		if t1.Data[i] != t2.Data[i] {
			t.Errorf("Reproducibility failed: Data[%d] = %v vs %v", i, t1.Data[i], t2.Data[i])
		}
	}
}

func TestRandnDifferentSeeds(t *testing.T) {
	// Проверка что разные seed дают разные результаты
	shape := []int{3, 4}

	t1 := Randn(shape, 42)
	t2 := Randn(shape, 123)

	allSame := true
	for i := range t1.Data {
		if t1.Data[i] != t2.Data[i] {
			allSame = false
			break
		}
	}

	if allSame {
		t.Error("Different seeds produced identical results")
	}
}

func TestCalculateSize(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  int
	}{
		{"1D", []int{5}, 5},
		{"2D", []int{3, 4}, 12},
		{"3D", []int{2, 3, 4}, 24},
		{"4D", []int{2, 2, 2, 2}, 16},
		{"empty", []int{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateSize(tt.shape)
			if got != tt.want {
				t.Errorf("calculateSize() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCalculateStrides(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  []int
	}{
		{"1D", []int{5}, []int{1}},
		{"2D", []int{3, 4}, []int{4, 1}},
		{"3D", []int{2, 3, 4}, []int{12, 4, 1}},
		{"4D", []int{2, 2, 2, 2}, []int{8, 4, 2, 1}},
		{"empty", []int{}, []int{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateStrides(tt.shape)
			if len(got) != len(tt.want) {
				t.Errorf("calculateStrides() length = %v, want %v", len(got), len(tt.want))
				return
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("calculateStrides()[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}
