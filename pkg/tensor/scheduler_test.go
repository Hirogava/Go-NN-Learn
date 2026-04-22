package tensor

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
)

func TestMatmulWorkerCountUsesGOMAXPROCS(t *testing.T) {
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)

	if got := matmulWorkerCount(1024, 64); got != 1 {
		t.Fatalf("matmulWorkerCount() with GOMAXPROCS=1 = %d, want 1", got)
	}

	runtime.GOMAXPROCS(4)
	if got := matmulWorkerCount(1024, 64); got != 4 {
		t.Fatalf("matmulWorkerCount() with GOMAXPROCS=4 = %d, want 4", got)
	}

	runtime.GOMAXPROCS(32)
	if got := matmulWorkerCount(128, 64); got != 2 {
		t.Fatalf("matmulWorkerCount() should cap workers by row chunks, got %d want 2", got)
	}
}

func TestMatmulRowSchedulerProducesNonOverlappingRanges(t *testing.T) {
	scheduler := newMatmulRowScheduler(130, 32)

	seen := make([]bool, 130)
	var mu sync.Mutex
	var wg sync.WaitGroup
	errCh := make(chan error, 1)

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				start, end, ok := scheduler.next()
				if !ok {
					return
				}

				mu.Lock()
				for row := start; row < end; row++ {
					if seen[row] {
						mu.Unlock()
						select {
						case errCh <- fmt.Errorf("row %d scheduled more than once", row):
						default:
						}
						return
					}
					seen[row] = true
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	close(errCh)

	if err := <-errCh; err != nil {
		t.Fatal(err)
	}

	for row, ok := range seen {
		if !ok {
			t.Fatalf("row %d was not scheduled", row)
		}
	}
}
