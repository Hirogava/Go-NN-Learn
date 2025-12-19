package api_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/tensor"
)

// ExampleSaveLoad демонстрирует сохранение и загрузку чекпоинта.
func ExampleSaveLoad() {
	mod := &mockModule{
		params: []*graph.Node{
			{Value: &tensor.Tensor{Data: []float64{1, 2}, Shape: []int{2}}},
		},
	}
	path := filepath.Join(os.TempDir(), "gonn_example_ckpt.bin")
	_ = os.Remove(path)

	// Сохраняем
	if err := api.SaveCheckpoint(mod, path); err != nil {
		fmt.Println("save error:", err)
		return
	}
	// обнулим
	mod.params[0].Value.Data = []float64{0, 0}
	// Загружаем
	if err := api.LoadCheckpoint(mod, path); err != nil {
		fmt.Println("load error:", err)
		return
	}
	fmt.Println(mod.params[0].Value.Data)
	_ = os.Remove(path)
	// Output: [1 2]
}

// ExamplePredict демонстрирует вызов Predict (identity модель).
func ExamplePredict() {
	mod := &mockModule{}
	in := &graph.Node{
		Value: &tensor.Tensor{
			Data:  []float64{3.14},
			Shape: []int{1},
		},
	}
	out := api.Predict(mod, in)
	fmt.Println(out.Value.Data[0])
	// Output: 3.14
}
