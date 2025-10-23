package layers

import (
	"fmt"
	"testing"
)

func TestExampleDotSumGradient(t *testing.T) {
	batch := []float64{-3, 0, 100, 100000}
	norm := BatchNormVector(batch, 0, 0)

	fmt.Println(batch)
	fmt.Println(norm)

	sum := 0.0
	for _, el := range norm {
		sum += el
	}

	fmt.Println(sum)
	fmt.Println(sum / 4)
}
