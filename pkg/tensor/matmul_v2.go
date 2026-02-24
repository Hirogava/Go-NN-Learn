package tensor

import (
	"sync"
	"sync/atomic"
)

// BLIS-style cache-blocked matrix multiplication v2.
// Uses panel packing + 4×4 micro-kernel for optimal cache utilization.

// Cache hierarchy: block sizes 32/64 for L1; 4×4 micro-kernel.
const (
	mc = 64  // panel height A (L1), block size 64
	kc = 128 // panel depth (L2)
	nc = 512 // panel width B (L3)
	mr = 4   // micro-kernel rows (tile 4)
	nr = 4   // micro-kernel cols (tile 4)
	// BlockSize32 is used for small matrices in matmul.go (threshold 32)
	BlockSize32 = 32
)

var (
	packedAPool = sync.Pool{
		New: func() any { return make([]float64, mc*kc) },
	}
	packedBPool = sync.Pool{
		New: func() any { return make([]float64, kc*nc) },
	}
)

type matMulWorkspace struct {
	packedA [][]float64
	packedB []float64
	nextA   atomic.Int32
}

const maxMatMulChunks = 64

func newMatMulWorkspace() *matMulWorkspace {
	n := GetMaxWorkers()
	if n < 1 {
		n = 1
	}
	if n < maxMatMulChunks {
		n = maxMatMulChunks
	}
	ws := &matMulWorkspace{
		packedA: make([][]float64, n),
		packedB: make([]float64, kc*nc),
	}
	for i := range ws.packedA {
		ws.packedA[i] = make([]float64, mc*kc)
	}
	return ws
}

var matMulWorkspacePool = sync.Pool{
	New: func() any { return newMatMulWorkspace() },
}

func getMatMulWorkspace() *matMulWorkspace {
	return matMulWorkspacePool.Get().(*matMulWorkspace)
}

func putMatMulWorkspace(ws *matMulWorkspace) {
	ws.nextA.Store(0)
	matMulWorkspacePool.Put(ws)
}

func matmulV2(a, b, c []float64, m, n, p int) {
	if len(c) < m*p || len(a) < m*n || len(b) < n*p {
		return
	}
	for i := range c {
		c[i] = 0
	}
	ws := getMatMulWorkspace()
	defer putMatMulWorkspace(ws)

	for jc := 0; jc < p; jc += nc {
		jcEnd := min(jc+nc, p)
		ncCur := jcEnd - jc

		for pc := 0; pc < n; pc += kc {
			pcEnd := min(pc+kc, n)
			kcCur := pcEnd - pc

			colsPadded := (ncCur + nr - 1) / nr * nr
			neededB := kcCur * colsPadded
			numChunks := (m + mc - 1) / mc
			useWorkspaceA := neededB <= len(ws.packedB) && numChunks <= len(ws.packedA)
			pB := ws.packedB
			if !useWorkspaceA || neededB > len(pB) {
				pB = getPackedB(kcCur, ncCur)
				packB(b, pB, n, p, pc, pcEnd, jc, jcEnd)
				ParallelFor(m, mc, func(icStart, icEnd int) {
					if icEnd <= icStart {
						return
					}
					pA := getPackedA(min(icEnd-icStart, mc), kcCur)
					for ic := icStart; ic < icEnd; ic += mc {
						icActualEnd := min(ic+mc, icEnd)
						mcCur := icActualEnd - ic
						packA(a, pA, n, ic, icActualEnd, pc, pcEnd)
						gebp(pA, pB, c, p, ic, mcCur, ncCur, kcCur, jc)
					}
					putPackedA(pA)
				})
				putPackedB(pB)
				continue
			}
			pB = pB[:neededB]
			packB(b, pB, n, p, pc, pcEnd, jc, jcEnd)

			ws.nextA.Store(0)
			ParallelFor(m, mc, func(icStart, icEnd int) {
				if icEnd <= icStart {
					return
				}
				idx := int(ws.nextA.Add(1) - 1)
				mcCur := min(icEnd-icStart, mc)
				rowsPadded := (mcCur + mr - 1) / mr * mr
				neededA := rowsPadded * kcCur
				if neededA <= 0 {
					return
				}
				pA := ws.packedA[idx%len(ws.packedA)][:neededA]
				for ic := icStart; ic < icEnd; ic += mc {
					icActualEnd := min(ic+mc, icEnd)
					mcCur := icActualEnd - ic
					packA(a, pA, n, ic, icActualEnd, pc, pcEnd)
					gebp(pA, pB, c, p, ic, mcCur, ncCur, kcCur, jc)
				}
			})
		}
	}
}

// gebp computes C[ic:ic+mcCur, jc:jc+ncCur] += packedA × packedB
// using the 4×4 micro-kernel.
func gebp(packedA, packedB, c []float64, ldc, ic, mcCur, ncCur, kcCur, jc int) {
	if len(packedA) == 0 || len(packedB) == 0 {
		return
	}
	for jr := 0; jr < ncCur; jr += nr { // Loop 2
		nrCur := min(nr, ncCur-jr)

		for ir := 0; ir < mcCur; ir += mr { // Loop 1
			mrCur := min(mr, mcCur-ir)

			if mrCur == mr && nrCur == nr {
				// Fast path: full 4×4 micro-kernel
				microKernel4x4(
					packedA[ir*kcCur:],
					packedB[jr*kcCur:],
					c, ldc,
					ic+ir, jc+jr, kcCur,
				)
			} else {
				// Edge case: partial tile
				microKernelGeneric(
					packedA[ir*kcCur:],
					packedB[jr*kcCur:],
					c, ldc,
					ic+ir, jc+jr, kcCur,
					mrCur, nrCur,
				)
			}
		}
	}
}

// microKernel4x4 computes a 4×4 block of C using 16 scalar accumulators.
// packedA is in column-major layout: [mr] elements per k step.
// packedB is in row-major layout: [nr] elements per k step.
func microKernel4x4(packedA, packedB, c []float64, ldc, iBase, jBase, kLen int) {
	if len(packedA) < mr*kLen || len(packedB) < nr*kLen {
		return
	}
	var c00, c01, c02, c03 float64
	var c10, c11, c12, c13 float64
	var c20, c21, c22, c23 float64
	var c30, c31, c32, c33 float64

	aOff := 0
	bOff := 0

	for p := 0; p < kLen; p++ {
		// Load mr=4 elements from packed A
		a0 := packedA[aOff]
		a1 := packedA[aOff+1]
		a2 := packedA[aOff+2]
		a3 := packedA[aOff+3]

		// Load nr=4 elements from packed B
		b0 := packedB[bOff]
		b1 := packedB[bOff+1]
		b2 := packedB[bOff+2]
		b3 := packedB[bOff+3]

		// Rank-1 update: outer product of a × b
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3

		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3

		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3

		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		aOff += mr
		bOff += nr
	}

	// Store accumulators back to C (accumulate, C was zero-initialized)
	r0 := iBase*ldc + jBase
	c[r0] += c00
	c[r0+1] += c01
	c[r0+2] += c02
	c[r0+3] += c03

	r1 := r0 + ldc
	c[r1] += c10
	c[r1+1] += c11
	c[r1+2] += c12
	c[r1+3] += c13

	r2 := r1 + ldc
	c[r2] += c20
	c[r2+1] += c21
	c[r2+2] += c22
	c[r2+3] += c23

	r3 := r2 + ldc
	c[r3] += c30
	c[r3+1] += c31
	c[r3+2] += c32
	c[r3+3] += c33
}

// microKernelGeneric handles edge-case tiles smaller than mr×nr.
func microKernelGeneric(packedA, packedB, c []float64, ldc, iBase, jBase, kLen, mrCur, nrCur int) {
	for p := 0; p < kLen; p++ {
		for i := 0; i < mrCur; i++ {
			aVal := packedA[i+p*mr]
			for j := 0; j < nrCur; j++ {
				bVal := packedB[j+p*nr]
				c[(iBase+i)*ldc+(jBase+j)] += aVal * bVal
			}
		}
	}
}

// packA packs a panel of A[rowStart:rowEnd, colStart:colEnd] into
// column-major layout optimized for the micro-kernel.
//
// Output layout: sequence of strips of height mr.
// Strip r (rows r*mr ... r*mr+mr-1):
//
//	K=0: A[r*mr, 0] ... A[r*mr+mr-1, 0]
//	K=1: A[r*mr, 1] ... A[r*mr+mr-1, 1]
//	...
func packA(a []float64, packed []float64, n, rowStart, rowEnd, colStart, colEnd int) {
	rows := rowEnd - rowStart
	cols := colEnd - colStart // kcCur

	idx := 0
	// Iterate over strips of height mr
	for ir := 0; ir < rows; ir += mr {
		// Minimum of mr or remaining rows
		mrCur := mr
		if ir+mr > rows {
			mrCur = rows - ir
		}

		// For each column k in the panel
		for k := 0; k < cols; k++ {
			// Copy column segment of height mrCur
			baseA := (rowStart+ir)*n + (colStart + k)
			for i := 0; i < mrCur; i++ {
				packed[idx] = a[baseA+i*n]
				idx++
			}
			// Pad with zeros if strip is partial (mrCur < mr)
			for i := mrCur; i < mr; i++ {
				packed[idx] = 0
				idx++
			}
		}
	}
}

// packB packs a panel of B[rowStart:rowEnd, colStart:colEnd] into
// row-major layout optimized for the micro-kernel.
//
// Output layout: sequence of strips of width nr.
// Strip c (cols c*nr ... c*nr+nr-1):
//
//	K=0: B[0, c*nr] ... B[0, c*nr+nr-1]
//	K=1: B[1, c*nr] ... B[1, c*nr+nr-1]
//	...
func packB(b []float64, packed []float64, _, p, rowStart, rowEnd, colStart, colEnd int) {
	rows := rowEnd - rowStart // kcCur
	cols := colEnd - colStart

	idx := 0
	// Iterate over strips of width nr
	for jr := 0; jr < cols; jr += nr {
		// Minimum of nr or remaining cols
		nrCur := nr
		if jr+nr > cols {
			nrCur = cols - jr
		}

		// For each row k in the panel
		for k := 0; k < rows; k++ {
			// Copy row segment of width nrCur
			baseB := (rowStart+k)*p + (colStart + jr)
			for j := 0; j < nrCur; j++ {
				packed[idx] = b[baseB+j]
				idx++
			}
			// Pad with zeros if strip is partial (nrCur < nr)
			for j := nrCur; j < nr; j++ {
				packed[idx] = 0
				idx++
			}
		}
	}
}

// Workspace management — get/put packed buffers from pools.

func getPackedA(mcCur, kcCur int) []float64 {
	// Calculate size with padding: we pack strips of height mr
	rowsPadded := (mcCur + mr - 1) / mr * mr
	needed := rowsPadded * kcCur

	if needed <= mc*kc {
		buf := packedAPool.Get().([]float64)
		if cap(buf) < needed {
			// Should ensure capacity, but pool items are fixed size mc*kc.
			// If needed > mc*kc (should not happen if mc is multiple of mr), we alloc.
			// But wait, mc=64, mr=4. mc is multiple.
			// If mcCur <= mc, then rowsPadded <= mc.
			// So needed <= mc*kc.
			// Just reslice buf to needed.
			return buf[:needed]
		}
		// If buf is nil or we just got it, ensure len is needed
		return buf[:needed]
	}
	return make([]float64, needed)
}

func putPackedA(buf []float64) {
	// Only put back if capacity is standard
	if cap(buf) == mc*kc {
		packedAPool.Put(buf)
	}
}

func getPackedB(kcCur, ncCur int) []float64 {
	// Calculate size with padding: we pack strips of width nr
	colsPadded := (ncCur + nr - 1) / nr * nr
	needed := kcCur * colsPadded

	if needed <= kc*nc {
		buf := packedBPool.Get().([]float64)
		// Ensure capacity check properly (standard pool items are kc*nc)
		if cap(buf) < needed {
			return buf[:needed] // Should not happen for standard sizes
		}
		return buf[:needed]
	}
	return make([]float64, needed)
}

func putPackedB(buf []float64) {
	if cap(buf) == kc*nc {
		packedBPool.Put(buf)
	}
}

var (
	packedAPool32 = sync.Pool{
		New: func() any { return make([]float32, mc*kc) },
	}
	packedBPool32 = sync.Pool{
		New: func() any { return make([]float32, kc*nc) },
	}
)

func matmulV2Float32(a, b, c []float32, m, n, p int) {
	for i := range c {
		c[i] = 0
	}
	for jc := 0; jc < p; jc += nc {
		jcEnd := min(jc+nc, p)
		ncCur := jcEnd - jc
		for pc := 0; pc < n; pc += kc {
			pcEnd := min(pc+kc, n)
			kcCur := pcEnd - pc
			pB := getPackedB32(kcCur, ncCur)
			packB32(b, pB, n, p, pc, pcEnd, jc, jcEnd)
			ParallelFor(m, mc, func(icStart, icEnd int) {
				pA := getPackedA32(min(icEnd-icStart, mc), kcCur)
				for ic := icStart; ic < icEnd; ic += mc {
					icActualEnd := min(ic+mc, icEnd)
					mcCur := icActualEnd - ic
					packA32(a, pA, n, ic, icActualEnd, pc, pcEnd)
					gebp32(pA, pB, c, p, ic, mcCur, ncCur, kcCur, jc)
				}
				putPackedA32(pA)
			})
			putPackedB32(pB)
		}
	}
}

func gebp32(packedA, packedB, c []float32, ldc, ic, mcCur, ncCur, kcCur, jc int) {
	for jr := 0; jr < ncCur; jr += nr {
		nrCur := min(nr, ncCur-jr)
		for ir := 0; ir < mcCur; ir += mr {
			mrCur := min(mr, mcCur-ir)
			if mrCur == mr && nrCur == nr {
				microKernel4x4Float32(
					packedA[ir*kcCur:], packedB[jr*kcCur:], c, ldc,
					ic+ir, jc+jr, kcCur,
				)
			} else {
				microKernelGenericFloat32(
					packedA[ir*kcCur:], packedB[jr*kcCur:], c, ldc,
					ic+ir, jc+jr, kcCur, mrCur, nrCur,
				)
			}
		}
	}
}

func microKernel4x4Float32(packedA, packedB, c []float32, ldc, iBase, jBase, kLen int) {
	var c00, c01, c02, c03 float32
	var c10, c11, c12, c13 float32
	var c20, c21, c22, c23 float32
	var c30, c31, c32, c33 float32
	aOff, bOff := 0, 0
	for p := 0; p < kLen; p++ {
		a0 := packedA[aOff]
		a1 := packedA[aOff+1]
		a2 := packedA[aOff+2]
		a3 := packedA[aOff+3]
		b0 := packedB[bOff]
		b1 := packedB[bOff+1]
		b2 := packedB[bOff+2]
		b3 := packedB[bOff+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
		aOff += mr
		bOff += nr
	}
	r0 := iBase*ldc + jBase
	c[r0] += c00
	c[r0+1] += c01
	c[r0+2] += c02
	c[r0+3] += c03
	r1 := r0 + ldc
	c[r1] += c10
	c[r1+1] += c11
	c[r1+2] += c12
	c[r1+3] += c13
	r2 := r1 + ldc
	c[r2] += c20
	c[r2+1] += c21
	c[r2+2] += c22
	c[r2+3] += c23
	r3 := r2 + ldc
	c[r3] += c30
	c[r3+1] += c31
	c[r3+2] += c32
	c[r3+3] += c33
}

func microKernelGenericFloat32(packedA, packedB, c []float32, ldc, iBase, jBase, kLen, mrCur, nrCur int) {
	for p := 0; p < kLen; p++ {
		for i := 0; i < mrCur; i++ {
			aVal := packedA[i+p*mr]
			for j := 0; j < nrCur; j++ {
				c[(iBase+i)*ldc+(jBase+j)] += aVal * packedB[j+p*nr]
			}
		}
	}
}

func packA32(a []float32, packed []float32, n, rowStart, rowEnd, colStart, colEnd int) {
	rows := rowEnd - rowStart
	cols := colEnd - colStart
	idx := 0
	for ir := 0; ir < rows; ir += mr {
		mrCur := mr
		if ir+mr > rows {
			mrCur = rows - ir
		}
		for k := 0; k < cols; k++ {
			baseA := (rowStart+ir)*n + (colStart + k)
			for i := 0; i < mrCur; i++ {
				packed[idx] = a[baseA+i*n]
				idx++
			}
			for i := mrCur; i < mr; i++ {
				packed[idx] = 0
				idx++
			}
		}
	}
}

func packB32(b []float32, packed []float32, _, p, rowStart, rowEnd, colStart, colEnd int) {
	rows := rowEnd - rowStart
	cols := colEnd - colStart
	idx := 0
	for jr := 0; jr < cols; jr += nr {
		nrCur := nr
		if jr+nr > cols {
			nrCur = cols - jr
		}
		for k := 0; k < rows; k++ {
			baseB := (rowStart+k)*p + (colStart + jr)
			for j := 0; j < nrCur; j++ {
				packed[idx] = b[baseB+j]
				idx++
			}
			for j := nrCur; j < nr; j++ {
				packed[idx] = 0
				idx++
			}
		}
	}
}

func getPackedA32(mcCur, kcCur int) []float32 {
	rowsPadded := (mcCur + mr - 1) / mr * mr
	needed := rowsPadded * kcCur
	if needed <= mc*kc {
		buf := packedAPool32.Get().([]float32)
		return buf[:needed]
	}
	return make([]float32, needed)
}

func putPackedA32(buf []float32) {
	if cap(buf) == mc*kc {
		packedAPool32.Put(buf)
	}
}

func getPackedB32(kcCur, ncCur int) []float32 {
	colsPadded := (ncCur + nr - 1) / nr * nr
	needed := kcCur * colsPadded
	if needed <= kc*nc {
		buf := packedBPool32.Get().([]float32)
		return buf[:needed]
	}
	return make([]float32, needed)
}

func putPackedB32(buf []float32) {
	if cap(buf) == kc*nc {
		packedBPool32.Put(buf)
	}
}
