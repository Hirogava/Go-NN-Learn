package tensor

import "sync/atomic"

// DType represents the data type of the tensor elements.
type DType uint8

const (
	// Float64 is the default double-precision floating point type (8 bytes).
	Float64 DType = iota
	// Float32 is single-precision floating point type (4 bytes).
	Float32
)

var (
	// defaultDType stores the default data type for new tensors.
	// Accessed atomically to ensure thread safety.
	defaultDType atomic.Int32 // Stores DType cast to int32
)

func init() {
	// Initialize default to Float64 explicitly (though 0 is Float64)
	SetDefaultDType(Float64)
}

// SetDefaultDType sets the default data type for new tensors.
func SetDefaultDType(dt DType) {
	defaultDType.Store(int32(dt))
}

// GetDefaultDType returns the current default data type.
func GetDefaultDType() DType {
	return DType(defaultDType.Load())
}

// String returns the string representation of the DType.
func (dt DType) String() string {
	switch dt {
	case Float64:
		return "Float64"
	case Float32:
		return "Float32"
	default:
		return "Unknown"
	}
}

// Size returns the size in bytes of a single element of DType.
func (dt DType) Size() int {
	switch dt {
	case Float64:
		return 8
	case Float32:
		return 4
	default:
		return 0
	}
}
