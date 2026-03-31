package tensor_test

import (
	"math/rand"
	"os"
	"testing"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

type matMulBenchmarkCase struct {
	name string
	m    int
	n    int
	p    int
}

type conv2DBenchmarkCase struct {
	name        string
	batchSize   int
	inChannels  int
	outChannels int
	height      int
	width       int
	kernelSize  int
	stride      int
	padding     int
}

func BenchmarkMatMulSizes(b *testing.B) {
	tensor.SetDefaultDType(tensor.Float64)

	cases := []matMulBenchmarkCase{
		{name: "32x32x32", m: 32, n: 32, p: 32},
		{name: "64x64x64", m: 64, n: 64, p: 64},
		{name: "128x128x128", m: 128, n: 128, p: 128},
		{name: "256x256x256", m: 256, n: 256, p: 256},
		{name: "512x512x512", m: 512, n: 512, p: 512},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			a := randomMatrix(tc.m, tc.n, 1)
			c := randomMatrix(tc.n, tc.p, 2)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := tensor.MatMul(a, c)
				if err != nil {
					b.Fatalf("MatMul failed: %v", err)
				}
				if out.Shape[0] != tc.m || out.Shape[1] != tc.p {
					b.Fatalf("unexpected result shape: %v", out.Shape)
				}
			}
		})
	}
}

func BenchmarkConv2DForward(b *testing.B) {
	cases := []conv2DBenchmarkCase{
		{
			name:        "small_1x3x28x28_k3",
			batchSize:   1,
			inChannels:  3,
			outChannels: 8,
			height:      28,
			width:       28,
			kernelSize:  3,
			stride:      1,
			padding:     1,
		},
		{
			name:        "medium_8x3x32x32_k3",
			batchSize:   8,
			inChannels:  3,
			outChannels: 16,
			height:      32,
			width:       32,
			kernelSize:  3,
			stride:      1,
			padding:     1,
		},
		{
			name:        "large_16x16x64x64_k3",
			batchSize:   16,
			inChannels:  16,
			outChannels: 32,
			height:      64,
			width:       64,
			kernelSize:  3,
			stride:      1,
			padding:     1,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			layer := layers.NewConv2D(
				tc.inChannels,
				tc.outChannels,
				tc.kernelSize,
				tc.stride,
				tc.padding,
				func(w []float64) {
					for i := range w {
						w[i] = 0.01
					}
				},
			)

			input := tensor.Zeros(tc.batchSize, tc.inChannels, tc.height, tc.width)
			fillTensorWithRandom(input, 7)
			node := graph.NewNode(input, nil, nil)

			restoreStdout := muteStdout(b)
			defer restoreStdout()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out := layer.Forward(node)
				if out == nil || out.Value == nil {
					b.Fatal("Conv2D.Forward returned nil output")
				}
			}
		})
	}
}

func BenchmarkSIMDVsCPU(b *testing.B) {
	sizes := []int{1 << 10, 1 << 14, 1 << 18}

	for _, size := range sizes {
		size := size
		b.Run("AddSIMD_"+itoa(size), func(b *testing.B) {
			a, c := randomVectorPair(size, 11, 13)
			out := make([]float64, size)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tensor.AddSIMD(a, c, out)
			}
		})

		b.Run("AddCPU_"+itoa(size), func(b *testing.B) {
			a, c := randomVectorPair(size, 11, 13)
			out := make([]float64, size)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				addCPU(a, c, out)
			}
		})

		b.Run("MulSIMD_"+itoa(size), func(b *testing.B) {
			a, c := randomVectorPair(size, 17, 19)
			out := make([]float64, size)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tensor.MulSIMD(a, c, out)
			}
		})

		b.Run("MulCPU_"+itoa(size), func(b *testing.B) {
			a, c := randomVectorPair(size, 17, 19)
			out := make([]float64, size)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mulCPU(a, c, out)
			}
		})

		b.Run("DotSIMD_"+itoa(size), func(b *testing.B) {
			a, c := randomVectorPair(size, 23, 29)
			var sink float64

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = tensor.DotProductSIMD(a, c)
			}
			_ = sink
		})

		b.Run("DotCPU_"+itoa(size), func(b *testing.B) {
			a, c := randomVectorPair(size, 23, 29)
			var sink float64

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = dotCPU(a, c)
			}
			_ = sink
		})
	}
}

func randomMatrix(rows, cols int, seed int64) *tensor.Tensor {
	rng := rand.New(rand.NewSource(seed))
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rng.Float64()*2 - 1
	}
	return &tensor.Tensor{
		Data:    data,
		Shape:   []int{rows, cols},
		Strides: []int{cols, 1},
		DType:   tensor.Float64,
	}
}

func fillTensorWithRandom(t *tensor.Tensor, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	for i := range t.Data {
		t.Data[i] = rng.Float64()*2 - 1
	}
}

func randomVectorPair(size int, seedA, seedB int64) ([]float64, []float64) {
	a := make([]float64, size)
	c := make([]float64, size)
	rngA := rand.New(rand.NewSource(seedA))
	rngB := rand.New(rand.NewSource(seedB))
	for i := 0; i < size; i++ {
		a[i] = rngA.Float64()*2 - 1
		c[i] = rngB.Float64()*2 - 1
	}
	return a, c
}

func addCPU(a, b, out []float64) {
	for i := range a {
		out[i] = a[i] + b[i]
	}
}

func mulCPU(a, b, out []float64) {
	for i := range a {
		out[i] = a[i] * b[i]
	}
}

func dotCPU(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func muteStdout(tb testing.TB) func() {
	tb.Helper()

	oldStdout := os.Stdout
	nullFile, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		tb.Fatalf("failed to open %s: %v", os.DevNull, err)
	}

	os.Stdout = nullFile
	return func() {
		os.Stdout = oldStdout
		_ = nullFile.Close()
	}
}

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
