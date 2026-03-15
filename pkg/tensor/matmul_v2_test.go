package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestMatMulV2Correctness(t *testing.T) {
	sizes := []struct{ m, n, p int }{
		{1, 1, 1},
		{16, 16, 16},
		{32, 32, 32},
		{63, 63, 63},
		{64, 64, 64},
		{65, 65, 65},
		{128, 128, 128},
		{129, 64, 129},
		{100, 200, 300},
	}

	for _, s := range sizes {
		t.Run(fmt.Sprintf("%dx%dx%d", s.m, s.n, s.p), func(t *testing.T) {
			a := make([]float64, s.m*s.n)
			b := make([]float64, s.n*s.p)
			cRef := make([]float64, s.m*s.p)
			cV2 := make([]float64, s.m*s.p)

			rng := rand.New(rand.NewSource(time.Now().UnixNano()))
			for i := range a {
				a[i] = rng.Float64()
			}
			for i := range b {
				b[i] = rng.Float64()
			}

			// Reference: simple loops
			for i := 0; i < s.m; i++ {
				for k := 0; k < s.n; k++ {
					valA := a[i*s.n+k]
					for j := 0; j < s.p; j++ {
						cRef[i*s.p+j] += valA * b[k*s.p+j]
					}
				}
			}

			matmulV2(a, b, cV2, s.m, s.n, s.p)

			for i := range cRef {
				if math.Abs(cRef[i]-cV2[i]) > 1e-9 {
					t.Fatalf("mismatch at %d: ref=%v v2=%v", i, cRef[i], cV2[i])
				}
			}
		})
	}
}

func BenchmarkMatMulV2_1024(b *testing.B) {
	n := 1024
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)

	for i := range a {
		a[i] = 0.5
	}
	for i := range bm {
		bm[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulV2(a, bm, c, n, n, n)
	}
}

func BenchmarkMatMulBaseline_1024(b *testing.B) {
	n := 1024
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)

	for i := range a {
		a[i] = 0.5
	}
	for i := range bm {
		bm[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Replica of old adaptive logic
		matmulBaseline(a, bm, c, n, n, n, 128)
	}
}

func BenchmarkMatMulV2_64(b *testing.B) {
	n := 64
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulV2(a, bm, c, n, n, n)
	}
}

func BenchmarkMatMulBaseline_64(b *testing.B) {
	n := 64
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBaseline(a, bm, c, n, n, n, 32)
	}
}

func BenchmarkMatMulV2_32(b *testing.B) {
	n := 32
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulV2(a, bm, c, n, n, n)
	}
}

func BenchmarkMatMulBaseline_32(b *testing.B) {
	n := 32
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBaseline(a, bm, c, n, n, n, 32)
	}
}

// -----------------------------------------------------------------------------
// Baseline Implementation (Replica of v1 logic for accurate comparison)
// -----------------------------------------------------------------------------

func matmulBaseline(a, b, c []float64, m, n, p int, blockSize int) {
	for i := range c {
		c[i] = 0.0
	}
	// Simplified adaptive parallel logic using ParallelFor
	ParallelFor(m, blockSize, func(start, end int) {
		for kk := 0; kk < n; kk += blockSize {
			kEnd := minWait(kk+blockSize, n)
			for ii := start; ii < end; ii += blockSize {
				iEnd := minWait(ii+blockSize, end)
				for jj := 0; jj < p; jj += blockSize {
					jEnd := minWait(jj+blockSize, p)
					matmulSIMDKernelBaseline(a, b, c, 0, n, p, ii, iEnd, kk, kEnd, jj, jEnd)
				}
			}
		}
	})
}

func matmulSIMDKernelBaseline(a, b, c []float64, m, n, p, iStart, iEnd, kStart, kEnd, jStart, jEnd int) {
	// Calling the existing kernel from package tensor
	MatMulSIMDKernel(a, b, c, m, n, p, iStart, iEnd, kStart, kEnd, jStart, jEnd)
}

func minWait(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestMatMulV2AtLeast40PercentFasterThanBaseline1024(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping speed comparison in short mode")
	}
	n := 1024
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	cV2 := make([]float64, n*n)
	cBaseline := make([]float64, n*n)
	for i := range a {
		a[i] = 0.5
		bm[i] = 0.5
	}
	const iterations = 5
	var sumV2, sumBaseline int64
	for i := 0; i < iterations; i++ {
		start := time.Now()
		matmulV2(a, bm, cV2, n, n, n)
		sumV2 += time.Since(start).Nanoseconds()
	}
	for i := 0; i < iterations; i++ {
		start := time.Now()
		matmulBaseline(a, bm, cBaseline, n, n, n, 128)
		sumBaseline += time.Since(start).Nanoseconds()
	}
	ratio := float64(sumBaseline) / float64(sumV2)
	if ratio < 1.4 {
		t.Errorf("expected matmulV2 at least 40%% faster than baseline (ratio baseline/v2 >= 1.4), got ratio=%.2f", ratio)
	}
}

func BenchmarkMatMulAPI_1024(b *testing.B) {
	SetDefaultDType(Float64)
	aT := Randn([]int{1024, 1024}, 1)
	bT := Randn([]int{1024, 1024}, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMul(aT, bT)
	}
}

func BenchmarkMatMulPooled_1024(b *testing.B) {
	SetDefaultDType(Float64)
	aT := Randn([]int{1024, 1024}, 1)
	bT := Randn([]int{1024, 1024}, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, _ := MatMulPooled(aT, bT)
		PutTensor(res)
	}
}

func BenchmarkMatMulAllocComparison(b *testing.B) {
	SetDefaultDType(Float64)
	aT := Randn([]int{256, 256}, 1)
	bT := Randn([]int{256, 256}, 2)
	b.Run("MatMul_alloc_new", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = MatMul(aT, bT)
		}
	})
	b.Run("MatMulPooled_reuse_result", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			res, _ := MatMulPooled(aT, bT)
			PutTensor(res)
		}
	})
}

func BenchmarkMatMulV2AllocBeforeAfter(b *testing.B) {
	n := 1024
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)
	for i := range a {
		a[i], bm[i] = 0.5, 0.5
	}
	b.Run("matmulV2_1024_alloc_after_workspace", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matmulV2(a, bm, c, n, n, n)
		}
	})
}
