package tensor

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
)

func benchmarkTensorSquare(size int) (*Tensor, *Tensor) {
	a := &Tensor{
		Data:    make([]float64, size*size),
		Shape:   []int{size, size},
		Strides: []int{size, 1},
	}
	b := &Tensor{
		Data:    make([]float64, size*size),
		Shape:   []int{size, size},
		Strides: []int{size, 1},
	}
	for i := range a.Data {
		a.Data[i] = float64((i % 97) - 48)
		b.Data[i] = float64((i % 89) - 44)
	}
	return a, b
}

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

func TestMatMulLargeMatchesAcrossGOMAXPROCS(t *testing.T) {
	a, b := benchmarkTensorSquare(192)

	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)

	gotSingle, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul() with GOMAXPROCS=1 error = %v", err)
	}

	runtime.GOMAXPROCS(4)
	gotParallel, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul() with GOMAXPROCS=4 error = %v", err)
	}

	if !shapesEqual(gotSingle.Shape, gotParallel.Shape) {
		t.Fatalf("shape mismatch: single=%v parallel=%v", gotSingle.Shape, gotParallel.Shape)
	}

	for i := range gotSingle.Data {
		if gotSingle.Data[i] != gotParallel.Data[i] {
			t.Fatalf("data mismatch at %d: single=%v parallel=%v", i, gotSingle.Data[i], gotParallel.Data[i])
		}
	}
}

func TestMatMulConcurrentLarge(t *testing.T) {
	a, b := benchmarkTensorSquare(192)

	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)

	expected, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul() expected result error = %v", err)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 8)

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			got, err := MatMul(a, b)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}

			for idx := range expected.Data {
				if got.Data[idx] != expected.Data[idx] {
					select {
					case errCh <- &matmulMismatchError{index: idx, got: got.Data[idx], want: expected.Data[idx]}:
					default:
					}
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)

	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
}

type matmulMismatchError struct {
	index int
	got   float64
	want  float64
}

func (e *matmulMismatchError) Error() string {
	return fmt.Sprintf("matmul mismatch at %d: got %v want %v", e.index, e.got, e.want)
}

func BenchmarkMatMulBlockedV2(b *testing.B) {
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
		matmulBlockedV2(a, bm, c, size, size, size, chooseBlockSize(size, size, size))
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

func BenchmarkMatMulBlockedV1_1024(b *testing.B) {
	size := 1024
	a, c := benchmarkTensorSquare(size)
	out := make([]float64, size*size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBlocked(a.Data, c.Data, out, size, size, size)
	}
}

func BenchmarkMatMulBlockedV2_1024(b *testing.B) {
	size := 1024
	a, c := benchmarkTensorSquare(size)
	out := make([]float64, size*size)
	blockSize := chooseBlockSize(size, size, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBlockedV2(a.Data, c.Data, out, size, size, size, blockSize)
	}
}

func BenchmarkMatMulBlockedV2_1024_Block32(b *testing.B) {
	size := 1024
	a, c := benchmarkTensorSquare(size)
	out := make([]float64, size*size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBlockedV2(a.Data, c.Data, out, size, size, size, 32)
	}
}

func BenchmarkMatMulBlockedV2_1024_Block64(b *testing.B) {
	size := 1024
	a, c := benchmarkTensorSquare(size)
	out := make([]float64, size*size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBlockedV2(a.Data, c.Data, out, size, size, size, 64)
	}
}

func BenchmarkMatMulParallelBlockedV2_1024(b *testing.B) {
	size := 1024
	a, c := benchmarkTensorSquare(size)
	out := make([]float64, size*size)
	blockSize := chooseBlockSize(size, size, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulParallelBlockedV2(a.Data, c.Data, out, size, size, size, blockSize)
	}
}

func BenchmarkMatMulParallelBlockedV2_1024_GOMAXPROCS(b *testing.B) {
	size := 1024
	a, c := benchmarkTensorSquare(size)
	blockSize := chooseBlockSize(size, size, size)

	for _, procs := range []int{1, 2, 4} {
		b.Run(fmt.Sprintf("procs=%d", procs), func(b *testing.B) {
			previous := runtime.GOMAXPROCS(procs)
			defer runtime.GOMAXPROCS(previous)

			out := make([]float64, size*size)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmulParallelBlockedV2(a.Data, c.Data, out, size, size, size, blockSize)
			}
		})
	}
}
