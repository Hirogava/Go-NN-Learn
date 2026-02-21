package backend

import (
	"testing"
)

func TestMatMul(t *testing.T) {
	tests := []struct {
		name      string
		a         *Tensor
		b         *Tensor
		wantShape []int
		wantData  []float64
		wantErr   bool
	}{
		{
			name: "2x2 * 2x2",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			b: &Tensor{
				Data:    []float64{5, 6, 7, 8},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			wantShape: []int{2, 2},
			wantData:  []float64{19, 22, 43, 50},
			wantErr:   false,
		},
		{
			name: "2x3 * 3x2",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6},
				Shape:   []int{2, 3},
				Strides: []int{3, 1},
			},
			b: &Tensor{
				Data:    []float64{7, 8, 9, 10, 11, 12},
				Shape:   []int{3, 2},
				Strides: []int{2, 1},
			},
			wantShape: []int{2, 2},
			wantData:  []float64{58, 64, 139, 154},
			wantErr:   false,
		},
		{
			name: "3x1 * 1x3 (outer product)",
			a: &Tensor{
				Data:    []float64{1, 2, 3},
				Shape:   []int{3, 1},
				Strides: []int{1, 1},
			},
			b: &Tensor{
				Data:    []float64{4, 5, 6},
				Shape:   []int{1, 3},
				Strides: []int{3, 1},
			},
			wantShape: []int{3, 3},
			wantData:  []float64{4, 5, 6, 8, 10, 12, 12, 15, 18},
			wantErr:   false,
		},
		{
			name: "incompatible shapes error",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			b: &Tensor{
				Data:    []float64{1, 2, 3},
				Shape:   []int{3, 1},
				Strides: []int{1, 1},
			},
			wantErr: true,
		},
		{
			name: "1D tensor error",
			a: &Tensor{
				Data:    []float64{1, 2, 3},
				Shape:   []int{3},
				Strides: []int{1},
			},
			b: &Tensor{
				Data:    []float64{4, 5, 6},
				Shape:   []int{3},
				Strides: []int{1},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MatMul(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("MatMul() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !shapesEqual(got.Shape, tt.wantShape) {
					t.Errorf("MatMul() shape = %v, want %v", got.Shape, tt.wantShape)
				}
				for i := range tt.wantData {
					if got.Data[i] != tt.wantData[i] {
						t.Errorf("MatMul() data[%d] = %v, want %v", i, got.Data[i], tt.wantData[i])
					}
				}
			}
		})
	}
}

func TestMatMulTransposeB(t *testing.T) {
	tests := []struct {
		name      string
		a         *Tensor
		b         *Tensor
		wantShape []int
		wantData  []float64
	}{
		{
			name: "2x2 * 2x2^T",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			b: &Tensor{
				Data:    []float64{5, 6, 7, 8},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			wantShape: []int{2, 2},
			wantData:  []float64{17, 23, 39, 53},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MatMulTransposeB(tt.a, tt.b)
			if err != nil {
				t.Errorf("MatMulTransposeB() error = %v", err)
				return
			}
			if !shapesEqual(got.Shape, tt.wantShape) {
				t.Errorf("MatMulTransposeB() shape = %v, want %v", got.Shape, tt.wantShape)
			}
			for i := range tt.wantData {
				if got.Data[i] != tt.wantData[i] {
					t.Errorf("MatMulTransposeB() data[%d] = %v, want %v", i, got.Data[i], tt.wantData[i])
				}
			}
		})
	}
}

func TestMatMulTransposeA(t *testing.T) {
	tests := []struct {
		name      string
		a         *Tensor
		b         *Tensor
		wantShape []int
		wantData  []float64
	}{
		{
			name: "2x2^T * 2x2",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			b: &Tensor{
				Data:    []float64{5, 6, 7, 8},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			wantShape: []int{2, 2},
			wantData:  []float64{26, 30, 38, 44},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MatMulTransposeA(tt.a, tt.b)
			if err != nil {
				t.Errorf("MatMulTransposeA() error = %v", err)
				return
			}
			if !shapesEqual(got.Shape, tt.wantShape) {
				t.Errorf("MatMulTransposeA() shape = %v, want %v", got.Shape, tt.wantShape)
			}
			for i := range tt.wantData {
				if got.Data[i] != tt.wantData[i] {
					t.Errorf("MatMulTransposeA() data[%d] = %v, want %v", i, got.Data[i], tt.wantData[i])
				}
			}
		})
	}
}

// Benchmarks

func BenchmarkMatMulSmall(b *testing.B) {
	// 8x8 matrices
	a := &Tensor{
		Data:    make([]float64, 64),
		Shape:   []int{8, 8},
		Strides: []int{8, 1},
	}
	c := &Tensor{
		Data:    make([]float64, 64),
		Shape:   []int{8, 8},
		Strides: []int{8, 1},
	}
	for i := range a.Data {
		a.Data[i] = float64(i)
		c.Data[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(a, c)
	}
}

func BenchmarkMatMulMedium(b *testing.B) {
	// 128x128 matrices
	a := &Tensor{
		Data:    make([]float64, 128*128),
		Shape:   []int{128, 128},
		Strides: []int{128, 1},
	}
	c := &Tensor{
		Data:    make([]float64, 128*128),
		Shape:   []int{128, 128},
		Strides: []int{128, 1},
	}
	for i := range a.Data {
		a.Data[i] = float64(i % 100)
		c.Data[i] = float64(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(a, c)
	}
}

func BenchmarkMatMulLarge(b *testing.B) {
	// 512x512 matrices
	a := &Tensor{
		Data:    make([]float64, 512*512),
		Shape:   []int{512, 512},
		Strides: []int{512, 1},
	}
	c := &Tensor{
		Data:    make([]float64, 512*512),
		Shape:   []int{512, 512},
		Strides: []int{512, 1},
	}
	for i := range a.Data {
		a.Data[i] = float64(i % 100)
		c.Data[i] = float64(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(a, c)
	}
}

func BenchmarkMatMulPooled(b *testing.B) {
	// 128x128 matrices with pooling
	a := &Tensor{
		Data:    make([]float64, 128*128),
		Shape:   []int{128, 128},
		Strides: []int{128, 1},
	}
	c := &Tensor{
		Data:    make([]float64, 128*128),
		Shape:   []int{128, 128},
		Strides: []int{128, 1},
	}
	for i := range a.Data {
		a.Data[i] = float64(i % 100)
		c.Data[i] = float64(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, _ := MatMulPooled(a, c)
		PutTensor(result)
	}
}

func BenchmarkMatMulOptimized(b *testing.B) {
	// 64x64 для сравнения оптимизированного алгоритма
	size := 64
	a := make([]float64, size*size)
	bm := make([]float64, size*size)
	c := make([]float64, size*size)
	for i := range a {
		a[i] = float64(i % 100)
		bm[i] = float64(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulOptimized(a, bm, c, size, size, size)
	}
}

func BenchmarkMatMulBlocked(b *testing.B) {
	// 128x128 для блочного умножения
	size := 128
	a := make([]float64, size*size)
	bm := make([]float64, size*size)
	c := make([]float64, size*size)
	for i := range a {
		a[i] = float64(i % 100)
		bm[i] = float64(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBlocked(a, bm, c, size, size, size)
	}
}

func BenchmarkMatMulParallel(b *testing.B) {
	// 256x256 для параллельного умножения
	size := 256
	a := make([]float64, size*size)
	bm := make([]float64, size*size)
	c := make([]float64, size*size)
	for i := range a {
		a[i] = float64(i % 100)
		bm[i] = float64(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulParallelBlocked(a, bm, c, size, size, size)
	}
}
