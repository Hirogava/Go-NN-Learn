package tensor

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	tests := []struct {
		name    string
		a       *Tensor
		b       *Tensor
		want    []float64
		wantErr bool
	}{
		{
			name: "simple 1D addition",
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
			want:    []float64{5, 7, 9},
			wantErr: false,
		},
		{
			name: "2D tensor addition",
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
			want:    []float64{6, 8, 10, 12},
			wantErr: false,
		},
		{
			name: "shape mismatch error",
			a: &Tensor{
				Data:    []float64{1, 2, 3},
				Shape:   []int{3},
				Strides: []int{1},
			},
			b: &Tensor{
				Data:    []float64{1, 2},
				Shape:   []int{2},
				Strides: []int{1},
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Add(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for i := range tt.want {
					if got.Data[i] != tt.want[i] {
						t.Errorf("Add() got[%d] = %v, want %v", i, got.Data[i], tt.want[i])
					}
				}
			}
		})
	}
}

func TestMul(t *testing.T) {
	tests := []struct {
		name    string
		a       *Tensor
		b       *Tensor
		want    []float64
		wantErr bool
	}{
		{
			name: "simple 1D multiplication (Hadamard)",
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
			want:    []float64{4, 10, 18},
			wantErr: false,
		},
		{
			name: "2D tensor multiplication",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			b: &Tensor{
				Data:    []float64{2, 3, 4, 5},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			want:    []float64{2, 6, 12, 20},
			wantErr: false,
		},
		{
			name: "shape mismatch error",
			a: &Tensor{
				Data:    []float64{1, 2, 3},
				Shape:   []int{3},
				Strides: []int{1},
			},
			b: &Tensor{
				Data:    []float64{1, 2},
				Shape:   []int{2},
				Strides: []int{1},
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Mul(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Mul() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for i := range tt.want {
					if got.Data[i] != tt.want[i] {
						t.Errorf("Mul() got[%d] = %v, want %v", i, got.Data[i], tt.want[i])
					}
				}
			}
		})
	}
}

func TestApply(t *testing.T) {
	tests := []struct {
		name string
		a    *Tensor
		f    func(float64) float64
		want []float64
	}{
		{
			name: "square function",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{4},
				Strides: []int{1},
			},
			f:    func(x float64) float64 { return x * x },
			want: []float64{1, 4, 9, 16},
		},
		{
			name: "ReLU activation",
			a: &Tensor{
				Data:    []float64{-1, 0, 1, 2},
				Shape:   []int{4},
				Strides: []int{1},
			},
			f: func(x float64) float64 {
				if x < 0 {
					return 0
				}
				return x
			},
			want: []float64{0, 0, 1, 2},
		},
		{
			name: "sigmoid approximation",
			a: &Tensor{
				Data:    []float64{0},
				Shape:   []int{1},
				Strides: []int{1},
			},
			f:    func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) },
			want: []float64{0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Apply(tt.a, tt.f)
			for i := range tt.want {
				if math.Abs(got.Data[i]-tt.want[i]) > 1e-10 {
					t.Errorf("Apply() got[%d] = %v, want %v", i, got.Data[i], tt.want[i])
				}
			}
		})
	}
}

func TestShapesEqual(t *testing.T) {
	tests := []struct {
		name string
		a    []int
		b    []int
		want bool
	}{
		{"equal shapes", []int{2, 3}, []int{2, 3}, true},
		{"different values", []int{2, 3}, []int{3, 2}, false},
		{"different lengths", []int{2, 3}, []int{2, 3, 4}, false},
		{"empty shapes", []int{}, []int{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shapesEqual(tt.a, tt.b); got != tt.want {
				t.Errorf("shapesEqual() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name      string
		a         *Tensor
		newShape  []int
		wantShape []int
		wantData  []float64
		wantErr   bool
	}{
		{
			name: "reshape 1D to 2D",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6},
				Shape:   []int{6},
				Strides: []int{1},
			},
			newShape:  []int{2, 3},
			wantShape: []int{2, 3},
			wantData:  []float64{1, 2, 3, 4, 5, 6},
			wantErr:   false,
		},
		{
			name: "reshape 2D to 1D",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			newShape:  []int{4},
			wantShape: []int{4},
			wantData:  []float64{1, 2, 3, 4},
			wantErr:   false,
		},
		{
			name: "reshape 2D to 3D",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6, 7, 8},
				Shape:   []int{2, 4},
				Strides: []int{4, 1},
			},
			newShape:  []int{2, 2, 2},
			wantShape: []int{2, 2, 2},
			wantData:  []float64{1, 2, 3, 4, 5, 6, 7, 8},
			wantErr:   false,
		},
		{
			name: "size mismatch error",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{4},
				Strides: []int{1},
			},
			newShape: []int{2, 3},
			wantErr:  true,
		},
		{
			name: "invalid dimension error",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{4},
				Strides: []int{1},
			},
			newShape: []int{2, 0},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Reshape(tt.a, tt.newShape)
			if (err != nil) != tt.wantErr {
				t.Errorf("Reshape() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !shapesEqual(got.Shape, tt.wantShape) {
					t.Errorf("Reshape() shape = %v, want %v", got.Shape, tt.wantShape)
				}
				for i := range tt.wantData {
					if got.Data[i] != tt.wantData[i] {
						t.Errorf("Reshape() data[%d] = %v, want %v", i, got.Data[i], tt.wantData[i])
					}
				}
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	tests := []struct {
		name      string
		a         *Tensor
		wantShape []int
		wantData  []float64
		wantErr   bool
	}{
		{
			name: "transpose 2x3 matrix",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6},
				Shape:   []int{2, 3},
				Strides: []int{3, 1},
			},
			wantShape: []int{3, 2},
			wantData:  []float64{1, 4, 2, 5, 3, 6},
			wantErr:   false,
		},
		{
			name: "transpose 3x2 matrix",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6},
				Shape:   []int{3, 2},
				Strides: []int{2, 1},
			},
			wantShape: []int{2, 3},
			wantData:  []float64{1, 3, 5, 2, 4, 6},
			wantErr:   false,
		},
		{
			name: "transpose 2x2 matrix",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
			wantShape: []int{2, 2},
			wantData:  []float64{1, 3, 2, 4},
			wantErr:   false,
		},
		{
			name: "error on 1D tensor",
			a: &Tensor{
				Data:    []float64{1, 2, 3},
				Shape:   []int{3},
				Strides: []int{1},
			},
			wantErr: true,
		},
		{
			name: "error on 3D tensor",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6, 7, 8},
				Shape:   []int{2, 2, 2},
				Strides: []int{4, 2, 1},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Transpose(tt.a)
			if (err != nil) != tt.wantErr {
				t.Errorf("Transpose() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !shapesEqual(got.Shape, tt.wantShape) {
					t.Errorf("Transpose() shape = %v, want %v", got.Shape, tt.wantShape)
				}
				for i := range tt.wantData {
					if got.Data[i] != tt.wantData[i] {
						t.Errorf("Transpose() data[%d] = %v, want %v", i, got.Data[i], tt.wantData[i])
					}
				}
			}
		})
	}
}

func TestSum(t *testing.T) {
	tests := []struct {
		name string
		a    *Tensor
		want float64
	}{
		{
			name: "sum of 1D tensor",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				Shape:   []int{4},
				Strides: []int{1},
			},
			want: 10,
		},
		{
			name: "sum of 2D tensor",
			a: &Tensor{
				Data:    []float64{1, 2, 3, 4, 5, 6},
				Shape:   []int{2, 3},
				Strides: []int{3, 1},
			},
			want: 21,
		},
		{
			name: "sum with negative values",
			a: &Tensor{
				Data:    []float64{-1, -2, 3, 4},
				Shape:   []int{4},
				Strides: []int{1},
			},
			want: 4,
		},
		{
			name: "sum of zeros",
			a: &Tensor{
				Data:    []float64{0, 0, 0},
				Shape:   []int{3},
				Strides: []int{1},
			},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sum(tt.a)
			if got.Data[0] != tt.want {
				t.Errorf("Sum() = %v, want %v", got.Data[0], tt.want)
			}
			if len(got.Shape) != 1 || got.Shape[0] != 1 {
				t.Errorf("Sum() shape = %v, want [1]", got.Shape)
			}
		})
	}
}

func TestExp(t *testing.T) {
	tests := []struct {
		name string
		a    *Tensor
		want []float64
	}{
		{
			name: "exp of zeros",
			a: &Tensor{
				Data:    []float64{0, 0, 0},
				Shape:   []int{3},
				Strides: []int{1},
			},
			want: []float64{1, 1, 1},
		},
		{
			name: "exp of ones",
			a: &Tensor{
				Data:    []float64{1, 1},
				Shape:   []int{2},
				Strides: []int{1},
			},
			want: []float64{math.E, math.E},
		},
		{
			name: "exp of various values",
			a: &Tensor{
				Data:    []float64{0, 1, 2},
				Shape:   []int{3},
				Strides: []int{1},
			},
			want: []float64{1, math.E, math.Exp(2)},
		},
		{
			name: "exp of negative values",
			a: &Tensor{
				Data:    []float64{-1, -2},
				Shape:   []int{2},
				Strides: []int{1},
			},
			want: []float64{math.Exp(-1), math.Exp(-2)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Exp(tt.a)
			for i := range tt.want {
				if math.Abs(got.Data[i]-tt.want[i]) > 1e-10 {
					t.Errorf("Exp() data[%d] = %v, want %v", i, got.Data[i], tt.want[i])
				}
			}
		})
	}
}

func TestLog(t *testing.T) {
	tests := []struct {
		name string
		a    *Tensor
		want []float64
	}{
		{
			name: "log of ones",
			a: &Tensor{
				Data:    []float64{1, 1, 1},
				Shape:   []int{3},
				Strides: []int{1},
			},
			want: []float64{0, 0, 0},
		},
		{
			name: "log of e",
			a: &Tensor{
				Data:    []float64{math.E, math.E},
				Shape:   []int{2},
				Strides: []int{1},
			},
			want: []float64{1, 1},
		},
		{
			name: "log of various values",
			a: &Tensor{
				Data:    []float64{1, math.E, math.Exp(2)},
				Shape:   []int{3},
				Strides: []int{1},
			},
			want: []float64{0, 1, 2},
		},
		{
			name: "log of powers of 10",
			a: &Tensor{
				Data:    []float64{10, 100},
				Shape:   []int{2},
				Strides: []int{1},
			},
			want: []float64{math.Log(10), math.Log(100)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Log(tt.a)
			for i := range tt.want {
				if math.Abs(got.Data[i]-tt.want[i]) > 1e-10 {
					t.Errorf("Log() data[%d] = %v, want %v", i, got.Data[i], tt.want[i])
				}
			}
		})
	}
}
