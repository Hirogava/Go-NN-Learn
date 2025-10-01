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
