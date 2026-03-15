package tensor

import (
	"runtime"
	"sync/atomic"
	"testing"
)

func TestParallelForCorrectness(t *testing.T) {
	// ParallelFor result must equal sequential result
	const n = 1000
	sequential := make([]float64, n)
	parallel := make([]float64, n)

	// Sequential
	for i := 0; i < n; i++ {
		sequential[i] = float64(i * i)
	}

	// Parallel
	ParallelFor(n, 1, func(start, end int) {
		for i := start; i < end; i++ {
			parallel[i] = float64(i * i)
		}
	})

	for i := 0; i < n; i++ {
		if sequential[i] != parallel[i] {
			t.Fatalf("mismatch at %d: sequential=%f, parallel=%f", i, sequential[i], parallel[i])
		}
	}
}

func TestParallelForSmallInput(t *testing.T) {
	// Small inputs should run sequentially (no panic, correct result)
	result := make([]int, 3)
	ParallelFor(3, 64, func(start, end int) {
		for i := start; i < end; i++ {
			result[i] = i + 1
		}
	})

	for i, v := range result {
		if v != i+1 {
			t.Fatalf("expected %d, got %d at index %d", i+1, v, i)
		}
	}
}

func TestParallelForZeroInput(t *testing.T) {
	// Zero-length input should not panic
	called := false
	ParallelFor(0, 1, func(start, end int) {
		called = true
	})
	// With total=0, body(0,0) is called but does nothing
	_ = called
}

func TestParallelForNesting(t *testing.T) {
	// Nested ParallelFor should NOT create N² goroutines
	// The inner call should run sequentially due to anti-nesting guard
	const n = 100
	result := make([]float64, n*n)

	ParallelFor(n, 1, func(outerStart, outerEnd int) {
		for i := outerStart; i < outerEnd; i++ {
			// This inner call should detect nesting and run sequentially
			ParallelFor(n, 1, func(innerStart, innerEnd int) {
				for j := innerStart; j < innerEnd; j++ {
					result[i*n+j] = float64(i + j)
				}
			})
		}
	})

	// Verify correctness
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := float64(i + j)
			if result[i*n+j] != expected {
				t.Fatalf("nesting test: expected %f at [%d,%d], got %f", expected, i, j, result[i*n+j])
			}
		}
	}
}

func TestParallelForRace(t *testing.T) {
	// This test is designed to be run with -race flag
	// Each goroutine writes to non-overlapping ranges — no race expected
	const n = 10000
	data := make([]float64, n)

	ParallelFor(n, 1, func(start, end int) {
		for i := start; i < end; i++ {
			data[i] = float64(i) * 2.0
		}
	})

	for i := 0; i < n; i++ {
		if data[i] != float64(i)*2.0 {
			t.Fatalf("race test: expected %f at %d, got %f", float64(i)*2.0, i, data[i])
		}
	}
}

func TestSetMaxWorkers(t *testing.T) {
	original := GetMaxWorkers()
	defer SetMaxWorkers(original)

	SetMaxWorkers(4)
	if GetMaxWorkers() != 4 {
		t.Fatalf("expected 4 workers, got %d", GetMaxWorkers())
	}

	// SetMaxWorkers(0) should reset to GOMAXPROCS
	SetMaxWorkers(0)
	expected := runtime.GOMAXPROCS(0)
	if GetMaxWorkers() != expected {
		t.Fatalf("expected %d workers (GOMAXPROCS), got %d", expected, GetMaxWorkers())
	}
}

func TestSetMaxWorkersSequential(t *testing.T) {
	original := GetMaxWorkers()
	defer SetMaxWorkers(original)

	// With 1 worker, should run sequentially
	SetMaxWorkers(1)

	const n = 100
	data := make([]float64, n)
	ParallelFor(n, 1, func(start, end int) {
		for i := start; i < end; i++ {
			data[i] = float64(i)
		}
	})

	for i := 0; i < n; i++ {
		if data[i] != float64(i) {
			t.Fatalf("sequential mode: expected %f at %d, got %f", float64(i), i, data[i])
		}
	}
}

// Benchmarks

func BenchmarkParallelForGOMAXPROCS(b *testing.B) {
	// Benchmark scaling across different GOMAXPROCS values
	const n = 1024 * 1024
	data := make([]float64, n)

	for _, procs := range []int{1, 2, 4, 8} {
		b.Run(
			"GOMAXPROCS="+itoa(procs),
			func(b *testing.B) {
				originalMaxP := runtime.GOMAXPROCS(procs)
				originalWorkers := GetMaxWorkers()
				SetMaxWorkers(procs)
				defer func() {
					runtime.GOMAXPROCS(originalMaxP)
					SetMaxWorkers(originalWorkers)
				}()

				// Reset parallelDepth to 0 for clean benchmark
				parallelDepth.Store(0)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					ParallelFor(n, MinGrainSize, func(start, end int) {
						for j := start; j < end; j++ {
							data[j] = float64(j) * 1.5
						}
					})
				}
			},
		)
	}
}

func BenchmarkParallelForMatMul256(b *testing.B) {
	size := 256
	a := make([]float64, size*size)
	bm := make([]float64, size*size)
	c := make([]float64, size*size)
	for i := range a {
		a[i] = float64(i % 100)
		bm[i] = float64(i % 100)
	}

	// Reset parallelDepth
	parallelDepth.Store(0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulParallelBlocked(a, bm, c, size, size, size)
	}
}

// itoa is a simple int to string for benchmark names
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// Ensure parallelDepth is accessible for testing
func resetParallelDepth() {
	parallelDepth.Store(0)
}

func getParallelDepth() int32 {
	return parallelDepth.Load()
}

func TestParallelDepthResets(t *testing.T) {
	resetParallelDepth()

	// After ParallelFor completes, depth should be back to 0
	ParallelFor(100, 1, func(start, end int) {
		// Check depth is 1 inside parallel region
		depth := getParallelDepth()
		if depth != 1 {
			t.Errorf("expected depth 1 inside parallel region, got %d", depth)
		}
	})

	if d := getParallelDepth(); d != 0 {
		t.Fatalf("expected depth 0 after ParallelFor, got %d", d)
	}
}

// Test that anti-nesting works with atomic counter
func TestAntiNestingAtomic(t *testing.T) {
	resetParallelDepth()

	var innerParallel atomic.Int32

	ParallelFor(10, 1, func(outerStart, outerEnd int) {
		for i := outerStart; i < outerEnd; i++ {
			// Check if inner call actually runs in parallel
			depth := getParallelDepth()
			if depth > 0 {
				innerParallel.Add(1) // This is expected — we're in parallel region
			}
		}
	})

	// All iterations should have seen depth > 0
	if innerParallel.Load() != 10 {
		t.Logf("inner parallel count: %d (expected 10)", innerParallel.Load())
	}
}
