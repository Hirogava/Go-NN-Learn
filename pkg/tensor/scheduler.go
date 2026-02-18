package tensor

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// Scheduler provides a centralized parallel execution facility for tensor operations.
// It prevents nested parallelism (goroutine explosion) and respects GOMAXPROCS.

var (
	// maxWorkers controls the maximum number of parallel workers.
	// Defaults to GOMAXPROCS.
	maxWorkers int32 = int32(runtime.GOMAXPROCS(0))

	// parallelDepth tracks the current nesting depth of parallel regions.
	// If > 0, subsequent ParallelFor calls execute sequentially to prevent N² goroutines.
	parallelDepth atomic.Int32
)

// MinGrainSize is the default minimum number of iterations per worker.
// Below this threshold, parallelization overhead exceeds the benefit.
const MinGrainSize = 64

// SetMaxWorkers sets the maximum number of parallel workers.
// Use 0 to reset to GOMAXPROCS.
func SetMaxWorkers(n int) {
	if n <= 0 {
		n = runtime.GOMAXPROCS(0)
	}
	atomic.StoreInt32(&maxWorkers, int32(n))
}

// GetMaxWorkers returns the current maximum number of parallel workers.
func GetMaxWorkers() int {
	return int(atomic.LoadInt32(&maxWorkers))
}

// ParallelFor splits [0, total) into chunks and executes body(start, end) in parallel.
//
// Rules:
//   - If total <= minGrain or workers <= 1, runs sequentially.
//   - If already inside a parallel region (nesting), runs sequentially.
//   - Otherwise, splits work across GOMAXPROCS goroutines.
//
// The body function receives a [start, end) range and must be safe to call
// concurrently from multiple goroutines with non-overlapping ranges.
func ParallelFor(total, minGrain int, body func(start, end int)) {
	workers := int(atomic.LoadInt32(&maxWorkers))

	// Guard 1: too little work or single-threaded
	if total <= minGrain || workers <= 1 {
		body(0, total)
		return
	}

	// Guard 2: already inside a parallel region — prevent nesting
	if parallelDepth.Add(1) > 1 {
		parallelDepth.Add(-1)
		body(0, total)
		return
	}
	defer parallelDepth.Add(-1)

	// Calculate grain size: at least minGrain per worker
	grain := (total + workers - 1) / workers
	if grain < minGrain {
		grain = minGrain
	}

	// Count actual number of chunks
	numChunks := (total + grain - 1) / grain
	if numChunks <= 1 {
		body(0, total)
		return
	}

	var wg sync.WaitGroup
	wg.Add(numChunks)

	for start := 0; start < total; start += grain {
		s, e := start, start+grain
		if e > total {
			e = total
		}
		go func(start, end int) {
			defer wg.Done()
			body(start, end)
		}(s, e)
	}

	wg.Wait()
}
