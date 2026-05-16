package layers

import (
	"math"
	"testing"
)

func TestInitializers(t *testing.T) {
	size := 10000
	data := make([]float64, size)

	t.Run("ZeroInit", func(t *testing.T) {
		ZeroInit()(data)
		for _, v := range data {
			if v != 0 {
				t.Errorf("expected 0, got %f", v)
			}
		}
	})

	t.Run("HeInit", func(t *testing.T) {
		fanIn := 100
		HeInit(fanIn)(data)
		
		sum := 0.0
		for _, v := range data {
			sum += v
		}
		mean := sum / float64(size)
		if math.Abs(mean) > 0.1 {
			t.Errorf("expected mean near 0, got %f", mean)
		}

		sumSq := 0.0
		for _, v := range data {
			sumSq += v * v
		}
		variance := sumSq / float64(size)
		expectedVariance := 2.0 / float64(fanIn)
		if math.Abs(variance-expectedVariance) > 0.05 {
			t.Errorf("expected variance near %f, got %f", expectedVariance, variance)
		}
	})

	t.Run("XavierInit", func(t *testing.T) {
		fanIn, fanOut := 100, 100
		XavierInit(fanIn, fanOut)(data)
		
		sumSq := 0.0
		for _, v := range data {
			sumSq += v * v
		}
		variance := sumSq / float64(size)
		expectedVariance := 2.0 / float64(fanIn+fanOut)
		if math.Abs(variance-expectedVariance) > 0.05 {
			t.Errorf("expected variance near %f, got %f", expectedVariance, variance)
		}
	})
}
