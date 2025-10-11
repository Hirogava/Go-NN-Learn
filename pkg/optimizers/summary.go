package optimizers

import (
	"fmt"
	"reflect"

	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// Summary выводит краткую табличную информацию о слоях модели
func Summary(m layers.Module, sample *tensor.Tensor) {
	if m == nil {
		fmt.Println("Summary: model is nil")
		return
	}
	if sample == nil {
		fmt.Println("Summary: sample input is nil — provide sample *tensor.Tensor")
		return
	}

	// Входной узел
	input := graph.NewNode(sample, nil, nil)
	out := input

	fmt.Printf("%-4s %-24s %-18s %s\n", "#", "Layer (type)", "Output shape", "Param #")
	fmt.Printf("-------------------------------------------------------------\n")

	totalParams := 0
	for i, l := range m.Layers() {
		out = l.Forward(out)

		var shape []int
		if out != nil && out.Value != nil {
			shape = out.Value.Shape
		}

		params := l.Params()
		paramCount := 0
		for _, p := range params {
			if p == nil || p.Value == nil {
				continue
			}
			paramCount += numel(p.Value.Shape)
		}
		totalParams += paramCount

		layerType := prettyTypeName(l)

		fmt.Printf("%-4d %-24s %-18v %d\n", i, layerType, shape, paramCount)
	}

	fmt.Printf("-------------------------------------------------------------\n")
	fmt.Printf("Total params: %d\n", totalParams)
}

// numel — считает количество элементов по форме
func numel(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func prettyTypeName(v interface{}) string {
	if v == nil {
		return "<nil>"
	}
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		if t.Elem() != nil {
			return t.Elem().Name()
		}
	}
	if t.Name() == "" {
		return fmt.Sprintf("%T", v)
	}
	return t.Name()
}
