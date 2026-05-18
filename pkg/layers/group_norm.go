package layers

import (
	"fmt"
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// GroupNorm нормализует активации по группам каналов внутри каждого примера.
// Между BatchNorm (по батчу) и LayerNorm (по всем признакам примера):
// статистики считаются по каждой группе каналов отдельно для каждого элемента батча.
//
// Формулы (для каждого n и группы g):
//
//	μ = mean(x_g)
//	σ² = var(x_g)
//	x̂ = (x - μ) / sqrt(σ² + ε)
//	y = γ * x̂ + β
//
// γ и β — обучаемые параметры длины numChannels.
// В отличие от BatchNorm, running mean/var не используются.
type GroupNorm struct {
	numChannels int
	numGroups   int
	eps         float64

	gamma *graph.Node
	beta  *graph.Node

	engine *autograd.Engine
}

// NewGroupNorm создаёт слой GroupNorm.
// numChannels должно делиться на numGroups без остатка.
func NewGroupNorm(numChannels, numGroups int, engine *autograd.Engine) *GroupNorm {
	if numGroups <= 0 {
		panic("GroupNorm: numGroups must be positive")
	}
	if numChannels%numGroups != 0 {
		panic(fmt.Sprintf("GroupNorm: numChannels (%d) must be divisible by numGroups (%d)", numChannels, numGroups))
	}

	gamma := engine.RequireGrad(tensor.Ones(numChannels))
	beta := engine.RequireGrad(tensor.Zeros(numChannels))

	return &GroupNorm{
		numChannels: numChannels,
		numGroups:   numGroups,
		eps:         1e-5,
		gamma:       gamma,
		beta:        beta,
		engine:      engine,
	}
}

// Forward выполняет прямой проход.
// Поддерживаются формы [batch, channels] и [batch, channels, height, width].
func (gn *GroupNorm) Forward(x *graph.Node) *graph.Node {
	if x == nil || x.Value == nil {
		panic("GroupNorm.Forward: input is nil")
	}

	shape := x.Value.Shape
	switch len(shape) {
	case 2:
		return gn.forward2D(x)
	case 4:
		return gn.forward4D(x)
	default:
		panic(fmt.Sprintf("GroupNorm expects 2D [N,C] or 4D [N,C,H,W] input, got %dD", len(shape)))
	}
}

func (gn *GroupNorm) forward2D(x *graph.Node) *graph.Node {
	batchSize := x.Value.Shape[0]
	numChannels := x.Value.Shape[1]
	gn.validateChannels(numChannels)

	output := tensor.Zeros(batchSize, numChannels)
	xHat := tensor.Zeros(batchSize, numChannels)
	invStd := make([]float64, batchSize*gn.numGroups)

	channelsPerGroup := gn.numChannels / gn.numGroups

	for n := range batchSize {
		for g := range gn.numGroups {
			cStart := g * channelsPerGroup
			cEnd := cStart + channelsPerGroup

			mean, variance := gn.stats2D(x.Value.Data, batchSize, numChannels, n, cStart, cEnd)
			std := math.Sqrt(variance + gn.eps)
			invStd[n*gn.numGroups+g] = 1.0 / std

			for c := cStart; c < cEnd; c++ {
				idx := n*numChannels + c
				xHat.Data[idx] = (x.Value.Data[idx] - mean) / std
				output.Data[idx] = gn.gamma.Value.Data[c]*xHat.Data[idx] + gn.beta.Value.Data[c]
			}
		}
	}

	op := &groupNormOp{
		x:                x,
		gamma:            gn.gamma,
		beta:             gn.beta,
		xHat:             xHat,
		invStd:           invStd,
		shape:            append([]int{}, x.Value.Shape...),
		numGroups:        gn.numGroups,
		channelsPerGroup: channelsPerGroup,
	}

	return graph.NewNode(output, []*graph.Node{x, gn.gamma, gn.beta}, op)
}

func (gn *GroupNorm) forward4D(x *graph.Node) *graph.Node {
	batchSize := x.Value.Shape[0]
	numChannels := x.Value.Shape[1]
	height := x.Value.Shape[2]
	width := x.Value.Shape[3]
	gn.validateChannels(numChannels)

	output := tensor.Zeros(batchSize, numChannels, height, width)
	xHat := tensor.Zeros(batchSize, numChannels, height, width)
	invStd := make([]float64, batchSize*gn.numGroups)

	channelsPerGroup := gn.numChannels / gn.numGroups

	for n := range batchSize {
		for g := range gn.numGroups {
			cStart := g * channelsPerGroup
			cEnd := cStart + channelsPerGroup

			mean, variance := gn.stats4D(x.Value.Data, batchSize, numChannels, height, width, n, cStart, cEnd)
			std := math.Sqrt(variance + gn.eps)
			invStd[n*gn.numGroups+g] = 1.0 / std

			for c := cStart; c < cEnd; c++ {
				for h := range height {
					for w := range width {
						idx := gn.index4D(batchSize, numChannels, height, width, n, c, h, w)
						xHat.Data[idx] = (x.Value.Data[idx] - mean) / std
						output.Data[idx] = gn.gamma.Value.Data[c]*xHat.Data[idx] + gn.beta.Value.Data[c]
					}
				}
			}
		}
	}

	op := &groupNormOp{
		x:                x,
		gamma:            gn.gamma,
		beta:             gn.beta,
		xHat:             xHat,
		invStd:           invStd,
		shape:            append([]int{}, x.Value.Shape...),
		numGroups:        gn.numGroups,
		channelsPerGroup: channelsPerGroup,
	}

	return graph.NewNode(output, []*graph.Node{x, gn.gamma, gn.beta}, op)
}

func (gn *GroupNorm) validateChannels(numChannels int) {
	if numChannels != gn.numChannels {
		panic(fmt.Sprintf("GroupNorm: expected %d channels, got %d", gn.numChannels, numChannels))
	}
}

func (gn *GroupNorm) stats2D(data []float64, batchSize, numChannels, n, cStart, cEnd int) (mean, variance float64) {
	count := float64(cEnd - cStart)
	sum := 0.0
	for c := cStart; c < cEnd; c++ {
		sum += data[n*numChannels+c]
	}
	mean = sum / count

	sumSq := 0.0
	for c := cStart; c < cEnd; c++ {
		diff := data[n*numChannels+c] - mean
		sumSq += diff * diff
	}
	variance = sumSq / count
	return mean, variance
}

func (gn *GroupNorm) stats4D(data []float64, batchSize, numChannels, height, width, n, cStart, cEnd int) (mean, variance float64) {
	count := float64((cEnd - cStart) * height * width)
	sum := 0.0
	for c := cStart; c < cEnd; c++ {
		for h := range height {
			for w := range width {
				idx := gn.index4D(batchSize, numChannels, height, width, n, c, h, w)
				sum += data[idx]
			}
		}
	}
	mean = sum / count

	sumSq := 0.0
	for c := cStart; c < cEnd; c++ {
		for h := range height {
			for w := range width {
				idx := gn.index4D(batchSize, numChannels, height, width, n, c, h, w)
				diff := data[idx] - mean
				sumSq += diff * diff
			}
		}
	}
	variance = sumSq / count
	return mean, variance
}

func (gn *GroupNorm) index4D(batchSize, numChannels, height, width, n, c, h, w int) int {
	_ = batchSize
	return ((n*numChannels+c)*height + h)*width + w
}

// Params возвращает обучаемые параметры (gamma, beta).
func (gn *GroupNorm) Params() []*graph.Node {
	return []*graph.Node{gn.gamma, gn.beta}
}

// Train и Eval сохранены для совместимости с Layer; поведение GroupNorm одинаково.
func (gn *GroupNorm) Train() {}
func (gn *GroupNorm) Eval()  {}

// SetEpsilon устанавливает ε для численной стабильности.
func (gn *GroupNorm) SetEpsilon(eps float64) {
	gn.eps = eps
}

type groupNormOp struct {
	x                *graph.Node
	gamma            *graph.Node
	beta             *graph.Node
	xHat             *tensor.Tensor
	invStd           []float64
	shape            []int
	numGroups        int
	channelsPerGroup int
}

func (op *groupNormOp) Backward(grad *tensor.Tensor) {
	switch len(op.shape) {
	case 2:
		op.backward2D(grad)
	case 4:
		op.backward4D(grad)
	default:
		panic("groupNormOp: unsupported shape")
	}
}

func (op *groupNormOp) backward2D(grad *tensor.Tensor) {
	batchSize := op.shape[0]
	numChannels := op.shape[1]

	if op.x.Grad == nil {
		op.x.Grad = tensor.Zeros(op.shape...)
	}
	if op.gamma.Grad == nil {
		op.gamma.Grad = tensor.Zeros(op.gamma.Value.Shape...)
	}
	if op.beta.Grad == nil {
		op.beta.Grad = tensor.Zeros(op.beta.Value.Shape...)
	}

	for n := range batchSize {
		for g := range op.numGroups {
			cStart := g * op.channelsPerGroup
			cEnd := cStart + op.channelsPerGroup
			m := float64(cEnd - cStart)
			invStd := op.invStd[n*op.numGroups+g]

			var sumGradXHat, sumGradXHatXHat float64
			for c := cStart; c < cEnd; c++ {
				idx := n*numChannels + c
				gradXHat := grad.Data[idx] * op.gamma.Value.Data[c]
				sumGradXHat += gradXHat
				sumGradXHatXHat += gradXHat * op.xHat.Data[idx]

				op.gamma.Grad.Data[c] += grad.Data[idx] * op.xHat.Data[idx]
				op.beta.Grad.Data[c] += grad.Data[idx]
			}

			for c := cStart; c < cEnd; c++ {
				idx := n*numChannels + c
				gradXHat := grad.Data[idx] * op.gamma.Value.Data[c]
				op.x.Grad.Data[idx] += invStd / m * (m*gradXHat - sumGradXHat - op.xHat.Data[idx]*sumGradXHatXHat)
			}
		}
	}
}

func (op *groupNormOp) backward4D(grad *tensor.Tensor) {
	batchSize := op.shape[0]
	numChannels := op.shape[1]
	height := op.shape[2]
	width := op.shape[3]

	if op.x.Grad == nil {
		op.x.Grad = tensor.Zeros(op.shape...)
	}
	if op.gamma.Grad == nil {
		op.gamma.Grad = tensor.Zeros(op.gamma.Value.Shape...)
	}
	if op.beta.Grad == nil {
		op.beta.Grad = tensor.Zeros(op.beta.Value.Shape...)
	}

	index4D := func(n, c, h, w int) int {
		return ((n*numChannels+c)*height+h)*width + w
	}

	for n := range batchSize {
		for g := range op.numGroups {
			cStart := g * op.channelsPerGroup
			cEnd := cStart + op.channelsPerGroup
			m := float64((cEnd - cStart) * height * width)
			invStd := op.invStd[n*op.numGroups+g]

			var sumGradXHat, sumGradXHatXHat float64
			for c := cStart; c < cEnd; c++ {
				for h := range height {
					for w := range width {
						idx := index4D(n, c, h, w)
						gradXHat := grad.Data[idx] * op.gamma.Value.Data[c]
						sumGradXHat += gradXHat
						sumGradXHatXHat += gradXHat * op.xHat.Data[idx]

						op.gamma.Grad.Data[c] += grad.Data[idx] * op.xHat.Data[idx]
						op.beta.Grad.Data[c] += grad.Data[idx]
					}
				}
			}

			for c := cStart; c < cEnd; c++ {
				for h := range height {
					for w := range width {
						idx := index4D(n, c, h, w)
						gradXHat := grad.Data[idx] * op.gamma.Value.Data[c]
						op.x.Grad.Data[idx] += invStd / m * (m*gradXHat - sumGradXHat - op.xHat.Data[idx]*sumGradXHatXHat)
					}
				}
			}
		}
	}
}
