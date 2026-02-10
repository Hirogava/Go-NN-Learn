package autograd

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

// CheckGradientEngine проверяет градиенты для готовых структур Engine/Node.
// build — функция, которая получает новый Engine и слайс входных узлов (leaf nodes) и должна
// построить вычислительный граф, вернув конечный узел (обычно скалярный loss).
// inputs — слайс узлов, представляющих параметры (leaf nodes). Для каждого вызова build
// внутри чекера будет создан новый Engine и новые input nodes с значениями из вектора x.
// eps/tol — параметры конечных разностей и допустимой относительной погрешности по умолчанию 1e-6, 1e-4 соотв..
func CheckGradientEngine(build func(e *Engine, inputs []*graph.Node) *graph.Node, inputs []*graph.Node, eps float64, tol float64) bool {
	// Вычисление общего размера и shapes.
	sizes := make([]int, len(inputs))
	total := 0
	for i, inp := range inputs {
		size := 1
		for _, d := range inp.Value.Shape {
			size *= d
		}
		sizes[i] = size
		total += size
	}

	// Положить все входные значения в вектор
	pack := func(nodes []*graph.Node) []float64 {
		x := make([]float64, total)
		pos := 0
		for i, n := range nodes {
			for j := 0; j < sizes[i]; j++ {
				x[pos] = n.Value.Data[j]
				pos++
			}
		}
		return x
	}

	// Раскидать вектор в только что созданные ноды
	makeInputNodes := func(x []float64) []*graph.Node {
		nodes := make([]*graph.Node, len(inputs))
		pos := 0
		for i, orig := range inputs {
			// Скопировать массив данных для этих входных данных
			data := make([]float64, sizes[i])
			copy(data, x[pos:pos+sizes[i]])
			pos += sizes[i]
			// Создать тензор с такой же формой как в ориге
			t := &tensor.Tensor{Data: data, Shape: append([]int{}, orig.Value.Shape...), Strides: append([]int{}, orig.Value.Strides...)}
			nodes[i] = graph.NewNode(t, nil, nil)
		}
		return nodes
	}

	// Функция для вычисления скалярного выходного значения для заданного x с использованием нового движка
	eval := func(x []float64) float64 {
		e := NewEngine()
		inNodes := makeInputNodes(x)
		out := build(e, inNodes)
		// Eсли out не скаляр, приводим к скаляру, суммируя все элементы
		if len(out.Value.Data) == 1 {
			return out.Value.Data[0]
		}
		s := 0.0
		for _, v := range out.Value.Data {
			s += v
		}
		return s
	}

	// Аналитический градиент
	x0 := pack(inputs)

	eAnal := NewEngine() // :)
	inAnal := makeInputNodes(x0)
	outAnal := build(eAnal, inAnal)
	// Запускаем backward
	eAnal.Backward(outAnal)

	// Собираем аналитический вектор градиента
	analytic := make([]float64, total)
	pos := 0
	for i, n := range inAnal {
		if n.Grad == nil {
			// Если дифференциирование не было произведено, считать нулями
			for j := 0; j < sizes[i]; j++ {
				analytic[pos] = 0
				pos++
			}
			continue
		}
		for j := 0; j < sizes[i]; j++ {
			analytic[pos] = n.Grad.Data[j]
			pos++
		}
	}

	// Числовой градиент через центральные различия
	numeric := make([]float64, total)
	for i := 0; i < total; i++ {
		xInc := make([]float64, total)
		xDec := make([]float64, total)
		copy(xInc, x0)
		copy(xDec, x0)
		xInc[i] += eps
		xDec[i] -= eps
		numeric[i] = (eval(xInc) - eval(xDec)) / (2 * eps)
	}

	// Сравнить относительные ошибки
	for i := 0; i < total; i++ {
		absErr := math.Abs(analytic[i] - numeric[i])
		m := math.Max(1.0, math.Max(math.Abs(analytic[i]), math.Abs(numeric[i])))
		relErr := absErr / m
		if relErr > tol {
			return false
		}
	}
	return true
}
