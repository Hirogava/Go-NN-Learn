package tensor

import (
	"math"
	"testing"
)

func TestFloat32HalfMemory(t *testing.T) {
	shape := []int{100, 200}
	SetDefaultDType(Float64)
	t64 := Zeros(shape...)
	bytes64 := t64.DataLen() * 8
	SetDefaultDType(Float32)
	t32 := Zeros(shape...)
	bytes32 := t32.DataLen() * 4
	if bytes32*2 != bytes64 {
		t.Fatalf("Float32 should use half memory: Float64=%d bytes Float32=%d bytes", bytes64, bytes32)
	}
	SetDefaultDType(Float64)
}

func TestFloat32ZerosOnesRandn(t *testing.T) {
	defer SetDefaultDType(GetDefaultDType())
	SetDefaultDType(Float32)

	z := Zeros(2, 3)
	if z.DType != Float32 || z.Data32 == nil || len(z.Data32) != 6 {
		t.Fatalf("Zeros Float32: DType=%v len(Data32)=%v", z.DType, len(z.Data32))
	}
	for i := range z.Data32 {
		if z.Data32[i] != 0 {
			t.Errorf("Zeros Data32[%d]=%v", i, z.Data32[i])
		}
	}

	o := Ones(2, 3)
	if o.DType != Float32 || len(o.Data32) != 6 {
		t.Fatalf("Ones Float32")
	}
	for i := range o.Data32 {
		if o.Data32[i] != 1.0 {
			t.Errorf("Ones Data32[%d]=%v", i, o.Data32[i])
		}
	}

	r := Randn([]int{2, 3}, 42)
	if r.DType != Float32 || len(r.Data32) != 6 {
		t.Fatalf("Randn Float32")
	}
}

func TestFloat32AddMulMatMul(t *testing.T) {
	defer SetDefaultDType(GetDefaultDType())
	SetDefaultDType(Float32)

	a := Zeros(2, 3)
	b := Zeros(2, 3)
	for i := range a.Data32 {
		a.Data32[i] = float32(i + 1)
		b.Data32[i] = float32(i + 2)
	}
	sum, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if sum.DType != Float32 {
		t.Fatalf("Add DType=%v", sum.DType)
	}
	for i := range sum.Data32 {
		want := float32(i+1) + float32(i+2)
		if math.Abs(float64(sum.Data32[i]-want)) > 1e-5 {
			t.Errorf("Add Data32[%d]=%v want %v", i, sum.Data32[i], want)
		}
	}

	mul, err := Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if mul.DType != Float32 {
		t.Fatalf("Mul DType=%v", mul.DType)
	}

	am := Randn([]int{4, 8}, 1)
	bm := Randn([]int{8, 2}, 2)
	cm, err := MatMul(am, bm)
	if err != nil {
		t.Fatal(err)
	}
	if cm.DType != Float32 || cm.Shape[0] != 4 || cm.Shape[1] != 2 {
		t.Fatalf("MatMul Float32 shape=%v", cm.Shape)
	}
}

func TestFloat32ModelTrainStep(t *testing.T) {
	defer SetDefaultDType(GetDefaultDType())
	SetDefaultDType(Float32)

	x := Randn([]int{32, 64}, 10)
	w := Randn([]int{64, 128}, 20)
	y, err := MatMul(x, w)
	if err != nil {
		t.Fatal(err)
	}
	if y.DType != Float32 || y.Shape[0] != 32 || y.Shape[1] != 128 {
		t.Fatalf("forward shape=%v", y.Shape)
	}
	bias := Ones(128)
	for i := 0; i < 32; i++ {
		for j := 0; j < 128; j++ {
			y.Data32[i*128+j] += bias.Data32[j]
		}
	}
	relu := Apply(y, func(f float64) float64 {
		if f < 0 {
			return 0
		}
		return f
	})
	if relu.DType != Float32 || relu.DataLen() != 32*128 {
		t.Fatalf("relu DType=%v len=%v", relu.DType, relu.DataLen())
	}
}

func BenchmarkFloat32VsFloat64Memory(b *testing.B) {
	shape := []int{128, 256}
	b.Run("Float64", func(b *testing.B) {
		SetDefaultDType(Float64)
		for i := 0; i < b.N; i++ {
			_ = Zeros(shape...)
		}
	})
	b.Run("Float32", func(b *testing.B) {
		SetDefaultDType(Float32)
		for i := 0; i < b.N; i++ {
			_ = Zeros(shape...)
		}
	})
}
