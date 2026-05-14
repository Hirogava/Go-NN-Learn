package tensor

import (
	"runtime"
	"sync/atomic"
)

// matmulRowScheduler раздаёт непересекающиеся диапазоны строк C.
// Каждый worker получает эксклюзивный row-range, поэтому запись в результат не конфликтует.
type matmulRowScheduler struct {
	totalRows int
	chunkRows int
	nextRow   atomic.Int64
}

func newMatmulRowScheduler(totalRows, chunkRows int) *matmulRowScheduler {
	if chunkRows < 1 {
		chunkRows = 1
	}

	return &matmulRowScheduler{
		totalRows: totalRows,
		chunkRows: chunkRows,
	}
}

func (s *matmulRowScheduler) next() (start, end int, ok bool) {
	start = int(s.nextRow.Add(int64(s.chunkRows))) - s.chunkRows
	if start >= s.totalRows {
		return 0, 0, false
	}

	end = start + s.chunkRows
	if end > s.totalRows {
		end = s.totalRows
	}

	return start, end, true
}

func matmulWorkerCount(totalRows, blockSize int) int {
	if totalRows <= 0 {
		return 1
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}

	rowChunks := ceilDiv(totalRows, max(1, blockSize))
	if rowChunks > 0 && workers > rowChunks {
		workers = rowChunks
	}
	if workers > totalRows {
		workers = totalRows
	}
	if workers < 1 {
		workers = 1
	}

	return workers
}

func matmulChunkRows(totalRows, workers, blockSize int) int {
	if totalRows <= 0 {
		return 1
	}
	if workers < 1 {
		workers = 1
	}

	chunkRows := ceilDiv(totalRows, workers)
	if blockSize > 1 {
		chunkRows = ceilDiv(chunkRows, blockSize) * blockSize
	}
	if chunkRows < 1 {
		chunkRows = 1
	}

	return chunkRows
}

func ceilDiv(x, y int) int {
	if y <= 0 {
		return 0
	}
	return (x + y - 1) / y
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
