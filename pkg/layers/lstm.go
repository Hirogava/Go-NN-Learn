package layers

import (
	"math"

	"github.com/Hirogava/Go-NN-Learn/pkg/matrix"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

// lstmDirection хранит обучаемые параметры LSTM для одного направления (forward/backward).
type lstmDirection struct {
	Wii, Wif, Wig, Wio *graph.Node // input-to-hidden [hidden_size, input_size]
	Whi, Whf, Whg, Who *graph.Node // hidden-to-hidden [hidden_size, hidden_size]
	bi, bf, bg, bo     *graph.Node // [hidden_size]
}

// LSTMCell выполняет один временной шаг LSTM.
// Веса передаются из слоя LSTM; ячейка не владеет параметрами.
type LSTMCell struct {
	dir        *lstmDirection
	hiddenSize int
}

// lstmStepState кэш промежуточных значений одного шага для BPTT.
type lstmStepState struct {
	i, f, g, o       *tensor.Tensor
	c, h             *tensor.Tensor
	tanhC            *tensor.Tensor
	hPrev, cPrev     *tensor.Tensor
}

// NewLSTMCell создаёт ячейку LSTM для заданного набора весов.
func NewLSTMCell(dir *lstmDirection, hiddenSize int) *LSTMCell {
	return &LSTMCell{dir: dir, hiddenSize: hiddenSize}
}

// Forward вычисляет один шаг:
//
//	i_t = σ(W_ii·x_t + W_hi·h_{t-1} + b_i)
//	f_t = σ(W_if·x_t + W_hf·h_{t-1} + b_f)
//	g_t = tanh(W_ig·x_t + W_hg·h_{t-1} + b_g)
//	o_t = σ(W_io·x_t + W_ho·h_{t-1} + b_o)
//	c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//	h_t = o_t ⊙ tanh(c_t)
func (cell *LSTMCell) Forward(x, hPrev, cPrev *tensor.Tensor, batchSize int) *lstmStepState {
	dir := cell.dir

	preI := lstmLinear(x, hPrev, dir.Wii, dir.Whi, dir.bi, batchSize)
	i := sigmoidTensor(preI)

	preF := lstmLinear(x, hPrev, dir.Wif, dir.Whf, dir.bf, batchSize)
	f := sigmoidTensor(preF)

	preG := lstmLinear(x, hPrev, dir.Wig, dir.Whg, dir.bg, batchSize)
	g := tanhTensor(preG)

	preO := lstmLinear(x, hPrev, dir.Wio, dir.Who, dir.bo, batchSize)
	o := sigmoidTensor(preO)

	fc, _ := tensor.Mul(f, cPrev)
	ig, _ := tensor.Mul(i, g)
	c, _ := tensor.Add(fc, ig)

	tanhC := tanhTensor(c)
	h, _ := tensor.Mul(o, tanhC)

	return &lstmStepState{i: i, f: f, g: g, o: o, c: c, h: h, tanhC: tanhC}
}

// LSTM — слой Long Short-Term Memory для последовательностей.
//
// Вход: [batch_size, sequence_length, input_size]
// Выход: [batch_size, sequence_length, hidden_size] (×2 при bidirectional)
type LSTM struct {
	inputSize     int
	hiddenSize    int
	bidirectional bool
	training      bool

	fwd *lstmDirection
	bwd *lstmDirection // nil, если не bidirectional

	hiddenState *tensor.Tensor
	cellState   *tensor.Tensor
}

// NewLSTM создаёт LSTM-слой.
func NewLSTM(
	inputSize, hiddenSize int,
	bidirectional bool,
	initFunc func([]float64),
) *LSTM {
	l := &LSTM{
		inputSize:     inputSize,
		hiddenSize:    hiddenSize,
		bidirectional: bidirectional,
		training:      true,
		fwd:           newLSTMDirection(inputSize, hiddenSize, initFunc),
	}
	if bidirectional {
		l.bwd = newLSTMDirection(inputSize, hiddenSize, initFunc)
	}
	return l
}

func newLSTMDirection(inputSize, hiddenSize int, initFunc func([]float64)) *lstmDirection {
	return &lstmDirection{
		Wii: newWeightMatrix(hiddenSize, inputSize, initFunc),
		Wif: newWeightMatrix(hiddenSize, inputSize, initFunc),
		Wig: newWeightMatrix(hiddenSize, inputSize, initFunc),
		Wio: newWeightMatrix(hiddenSize, inputSize, initFunc),
		Whi: newWeightMatrix(hiddenSize, hiddenSize, initFunc),
		Whf: newWeightMatrix(hiddenSize, hiddenSize, initFunc),
		Whg: newWeightMatrix(hiddenSize, hiddenSize, initFunc),
		Who: newWeightMatrix(hiddenSize, hiddenSize, initFunc),
		bi:  newBiasVector(hiddenSize, initFunc),
		bf:  newBiasVector(hiddenSize, initFunc),
		bg:  newBiasVector(hiddenSize, initFunc),
		bo:  newBiasVector(hiddenSize, initFunc),
	}
}

func newWeightMatrix(rows, cols int, initFunc func([]float64)) *graph.Node {
	size := rows * cols
	data := make([]float64, size)
	initFunc(data)
	return &graph.Node{
		Value: &tensor.Tensor{
			Data:    data,
			Shape:   []int{rows, cols},
			Strides: []int{cols, 1},
		},
	}
}

func newBiasVector(size int, initFunc func([]float64)) *graph.Node {
	data := make([]float64, size)
	initFunc(data)
	return &graph.Node{
		Value: &tensor.Tensor{
			Data:    data,
			Shape:   []int{size},
			Strides: []int{1},
		},
	}
}

// Params возвращает все обучаемые параметры.
func (l *LSTM) Params() []*graph.Node {
	params := l.fwd.params()
	if l.bidirectional {
		params = append(params, l.bwd.params()...)
	}
	return params
}

func (d *lstmDirection) params() []*graph.Node {
	return []*graph.Node{
		d.Wii, d.Wif, d.Wig, d.Wio,
		d.Whi, d.Whf, d.Whg, d.Who,
		d.bi, d.bf, d.bg, d.bo,
	}
}

func (l *LSTM) Train() { l.training = true }
func (l *LSTM) Eval()  { l.training = false }

func (l *LSTM) ResetState() {
	l.hiddenState = nil
	l.cellState = nil
}

func (l *LSTM) GetHiddenState() *tensor.Tensor { return l.hiddenState }
func (l *LSTM) GetCellState() *tensor.Tensor   { return l.cellState }

func (l *LSTM) SetHiddenState(h *tensor.Tensor) { l.hiddenState = h }
func (l *LSTM) SetCellState(c *tensor.Tensor)   { l.cellState = c }

func (l *LSTM) GetInputSize() int     { return l.inputSize }
func (l *LSTM) GetHiddenSize() int    { return l.hiddenSize }
func (l *LSTM) IsBidirectional() bool { return l.bidirectional }

// lstmOp хранит состояние для BPTT.
type lstmOp struct {
	x       *graph.Node
	lstm    *LSTM
	fwd     *lstmBPTTCache
	bwd     *lstmBPTTCache
}

type lstmBPTTCache struct {
	hStates []*tensor.Tensor // len seqLen+1
	cStates []*tensor.Tensor
	steps   []*lstmStepState // len seqLen
}

// Forward выполняет проход по последовательности [batch, seq, input].
func (l *LSTM) Forward(x *graph.Node) *graph.Node {
	batchSize := x.Value.Shape[0]
	seqLen := x.Value.Shape[1]
	outHidden := l.hiddenSize
	if l.bidirectional {
		outHidden *= 2
	}

	outputVal := tensor.Zeros(batchSize, seqLen, outHidden)

	var op *lstmOp
	if l.training {
		op = &lstmOp{x: x, lstm: l}
		op.fwd = l.runDirection(x, batchSize, seqLen, l.fwd, true, outputVal, 0)
		if l.bidirectional {
			op.bwd = l.runDirection(x, batchSize, seqLen, l.bwd, false, outputVal, l.hiddenSize)
		}
	} else {
		l.runDirectionInference(x, batchSize, seqLen, l.fwd, true, outputVal, 0)
		if l.bidirectional {
			l.runDirectionInference(x, batchSize, seqLen, l.bwd, false, outputVal, l.hiddenSize)
		}
	}

	if !l.training {
		return graph.NewNode(outputVal, nil, nil)
	}

	parents := append([]*graph.Node{x}, l.Params()...)
	return graph.NewNode(outputVal, parents, op)
}

func (l *LSTM) runDirection(
	x *graph.Node,
	batchSize, seqLen int,
	dir *lstmDirection,
	forward bool,
	output *tensor.Tensor,
	hiddenOffset int,
) *lstmBPTTCache {
	cell := NewLSTMCell(dir, l.hiddenSize)
	cache := &lstmBPTTCache{
		hStates: make([]*tensor.Tensor, seqLen+1),
		cStates: make([]*tensor.Tensor, seqLen+1),
		steps:   make([]*lstmStepState, seqLen),
	}

	hPrev := l.initialHidden(batchSize)
	cPrev := tensor.Zeros(batchSize, l.hiddenSize)
	cache.hStates[0] = hPrev
	cache.cStates[0] = cPrev

	if forward {
		for t := 0; t < seqLen; t++ {
			xt := extractSlice(x.Value, t)
			step := cell.Forward(xt, hPrev, cPrev, batchSize)
			step.hPrev, step.cPrev = hPrev, cPrev
			cache.steps[t] = step
			cache.hStates[t+1] = step.h
			cache.cStates[t+1] = step.c
			copySliceOffset(output, step.h, t, hiddenOffset)
			hPrev, cPrev = step.h, step.c
		}
		if !l.bidirectional || dir == l.fwd {
			l.hiddenState = hPrev
			l.cellState = cPrev
		}
	} else {
		for t := seqLen - 1; t >= 0; t-- {
			xt := extractSlice(x.Value, t)
			step := cell.Forward(xt, hPrev, cPrev, batchSize)
			step.hPrev, step.cPrev = hPrev, cPrev
			cache.steps[t] = step
			cache.hStates[t+1] = step.h
			cache.cStates[t+1] = step.c
			copySliceOffset(output, step.h, t, hiddenOffset)
			hPrev, cPrev = step.h, step.c
		}
	}

	return cache
}

func (l *LSTM) runDirectionInference(
	x *graph.Node,
	batchSize, seqLen int,
	dir *lstmDirection,
	forward bool,
	output *tensor.Tensor,
	hiddenOffset int,
) {
	cell := NewLSTMCell(dir, l.hiddenSize)
	hPrev := l.initialHidden(batchSize)
	cPrev := tensor.Zeros(batchSize, l.hiddenSize)

	if forward {
		for t := 0; t < seqLen; t++ {
			xt := extractSlice(x.Value, t)
			step := cell.Forward(xt, hPrev, cPrev, batchSize)
			copySliceOffset(output, step.h, t, hiddenOffset)
			hPrev, cPrev = step.h, step.c
		}
		if !l.bidirectional || dir == l.fwd {
			l.hiddenState = hPrev
			l.cellState = cPrev
		}
	} else {
		for t := seqLen - 1; t >= 0; t-- {
			xt := extractSlice(x.Value, t)
			step := cell.Forward(xt, hPrev, cPrev, batchSize)
			copySliceOffset(output, step.h, t, hiddenOffset)
			hPrev, cPrev = step.h, step.c
		}
	}
}

func (l *LSTM) initialHidden(batchSize int) *tensor.Tensor {
	if l.hiddenState != nil && l.hiddenState.Shape[0] == batchSize {
		return l.hiddenState
	}
	return tensor.Zeros(batchSize, l.hiddenSize)
}

// Backward выполняет BPTT по обоим направлениям (если bidirectional).
func (op *lstmOp) Backward(gradOutput *tensor.Tensor) {
	seqLen := op.x.Value.Shape[1]
	batchSize := op.x.Value.Shape[0]
	dx := tensor.Zeros(op.x.Value.Shape...)

	lstmBackwardDirection(op.lstm.fwd, op.fwd, op.x, gradOutput, batchSize, seqLen, 0, dx)

	if op.lstm.bidirectional {
		lstmBackwardDirection(op.lstm.bwd, op.bwd, op.x, gradOutput, batchSize, seqLen, op.lstm.hiddenSize, dx)
	}

	accumulate(op.x, dx)
}

type lstmGradBuf struct {
	dWii, dWif, dWig, dWio *tensor.Tensor
	dWhi, dWhf, dWhg, dWho *tensor.Tensor
	dbi, dbf, dbg, dbo     *tensor.Tensor
}

func newLSTMGradBuf(dir *lstmDirection) *lstmGradBuf {
	return &lstmGradBuf{
		dWii: tensor.Zeros(dir.Wii.Value.Shape...),
		dWif: tensor.Zeros(dir.Wif.Value.Shape...),
		dWig: tensor.Zeros(dir.Wig.Value.Shape...),
		dWio: tensor.Zeros(dir.Wio.Value.Shape...),
		dWhi: tensor.Zeros(dir.Whi.Value.Shape...),
		dWhf: tensor.Zeros(dir.Whf.Value.Shape...),
		dWhg: tensor.Zeros(dir.Whg.Value.Shape...),
		dWho: tensor.Zeros(dir.Who.Value.Shape...),
		dbi:  tensor.Zeros(dir.bi.Value.Shape...),
		dbf:  tensor.Zeros(dir.bf.Value.Shape...),
		dbg:  tensor.Zeros(dir.bg.Value.Shape...),
		dbo:  tensor.Zeros(dir.bo.Value.Shape...),
	}
}

func (g *lstmGradBuf) accumulateTo(dir *lstmDirection) {
	accumulate(dir.Wii, g.dWii)
	accumulate(dir.Wif, g.dWif)
	accumulate(dir.Wig, g.dWig)
	accumulate(dir.Wio, g.dWio)
	accumulate(dir.Whi, g.dWhi)
	accumulate(dir.Whf, g.dWhf)
	accumulate(dir.Whg, g.dWhg)
	accumulate(dir.Who, g.dWho)
	accumulate(dir.bi, g.dbi)
	accumulate(dir.bf, g.dbf)
	accumulate(dir.bg, g.dbg)
	accumulate(dir.bo, g.dbo)
}

func lstmBackwardDirection(
	dir *lstmDirection,
	cache *lstmBPTTCache,
	x *graph.Node,
	gradOutput *tensor.Tensor,
	batchSize, seqLen, hiddenOffset int,
	dx *tensor.Tensor,
) {
	buf := newLSTMGradBuf(dir)
	hiddenSize := cache.hStates[0].Shape[1]
	dhNext := tensor.Zeros(batchSize, hiddenSize)
	dcNext := tensor.Zeros(batchSize, hiddenSize)

	for t := seqLen - 1; t >= 0; t-- {
		stepGrad := extractSliceOffset(gradOutput, t, hiddenOffset, hiddenSize)
		dh, _ := tensor.Add(stepGrad, dhNext)
		step := cache.steps[t]
		dxT, dhPrev, dcPrev := lstmBackwardStep(
			dir, step, extractSlice(x.Value, t), dh, dcNext, batchSize, buf,
		)
		copySlice(dx, dxT, t)
		dhNext = dhPrev
		dcNext = dcPrev
	}

	buf.accumulateTo(dir)
}

func lstmBackwardStep(
	dir *lstmDirection,
	step *lstmStepState,
	x, dh, dcNext *tensor.Tensor,
	batchSize int,
	buf *lstmGradBuf,
) (dx, dhPrev, dcPrev *tensor.Tensor) {
	i, f, g, o := step.i, step.f, step.g, step.o
	tanhC := step.tanhC
	cPrev := step.cPrev

	dc := cloneTensor(dcNext)
	dtanhC, _ := tensor.Mul(dh, o)
	dtanhCPart := tanhGradFromOutput(tanhC, dtanhC)
	dc, _ = tensor.Add(dc, dtanhCPart)

	dcPrev, _ = tensor.Mul(dc, f)
	df, _ := tensor.Mul(dc, cPrev)
	di, _ := tensor.Mul(dc, g)
	dg, _ := tensor.Mul(dc, i)

	dpreI := sigmoidGradFromOutput(i, di)
	dpreF := sigmoidGradFromOutput(f, df)
	dpreG := tanhGradFromOutput(g, dg)
	do, _ := tensor.Mul(dh, tanhC)
	dpreO := sigmoidGradFromOutput(o, do)

	lstmAccumGateGrad(dpreI, x, step.hPrev, dir.Wii, dir.Whi, dir.bi, buf.dWii, buf.dWhi, buf.dbi)
	lstmAccumGateGrad(dpreF, x, step.hPrev, dir.Wif, dir.Whf, dir.bf, buf.dWif, buf.dWhf, buf.dbf)
	lstmAccumGateGrad(dpreG, x, step.hPrev, dir.Wig, dir.Whg, dir.bg, buf.dWig, buf.dWhg, buf.dbg)
	lstmAccumGateGrad(dpreO, x, step.hPrev, dir.Wio, dir.Who, dir.bo, buf.dWio, buf.dWho, buf.dbo)

	dhPrev = tensor.Zeros(batchSize, dh.Shape[1])
	dhPrev = lstmAddDhPrev(dhPrev, dpreI, dir.Whi)
	dhPrev = lstmAddDhPrev(dhPrev, dpreF, dir.Whf)
	dhPrev = lstmAddDhPrev(dhPrev, dpreG, dir.Whg)
	dhPrev = lstmAddDhPrev(dhPrev, dpreO, dir.Who)

	dx = tensor.Zeros(x.Shape...)
	dx = lstmAddDxGate(dx, dpreI, dir.Wii)
	dx = lstmAddDxGate(dx, dpreF, dir.Wif)
	dx = lstmAddDxGate(dx, dpreG, dir.Wig)
	dx = lstmAddDxGate(dx, dpreO, dir.Wio)

	return dx, dhPrev, dcPrev
}

func lstmLinear(x, hPrev *tensor.Tensor, Wx, Wh, b *graph.Node, batchSize int) *tensor.Tensor {
	wxT, _ := matrix.Transposition(matrix.TensorToMatrix(Wx.Value))
	whT, _ := matrix.Transposition(matrix.TensorToMatrix(Wh.Value))

	xM := matrix.TensorToMatrix(x)
	ixM, _ := matrix.MatMul(xM, wxT)
	ix := matrix.MatrixToTensor(ixM)

	hM := matrix.TensorToMatrix(hPrev)
	hhM, _ := matrix.MatMul(hM, whT)
	hh := matrix.MatrixToTensor(hhM)

	sum, _ := tensor.Add(ix, hh)
	sum, _ = tensor.Add(sum, broadcastBias(b.Value, batchSize))
	return sum
}

func lstmAccumGateGrad(
	dPre, x, hPrev *tensor.Tensor,
	Wx, Wh, _ *graph.Node,
	dWx, dWh, db *tensor.Tensor,
) {
	dPreM := matrix.TensorToMatrix(dPre)
	dPreT, _ := matrix.Transposition(dPreM)

	xM := matrix.TensorToMatrix(x)
	localWx, _ := matrix.MatMul(dPreT, xM)
	accumulateTensor(dWx, matrix.MatrixToTensor(localWx))

	hM := matrix.TensorToMatrix(hPrev)
	localWh, _ := matrix.MatMul(dPreT, hM)
	accumulateTensor(dWh, matrix.MatrixToTensor(localWh))

	accumulateTensor(db, sumAlongBatch(dPre))
}

func accumulateTensor(dst, src *tensor.Tensor) {
	for i := range dst.Data {
		dst.Data[i] += src.Data[i]
	}
}

func lstmAddDhPrev(acc, dPre *tensor.Tensor, Wh *graph.Node) *tensor.Tensor {
	dPreM := matrix.TensorToMatrix(dPre)
	whM := matrix.TensorToMatrix(Wh.Value)
	dhM, _ := matrix.MatMul(dPreM, whM)
	part, _ := tensor.Add(acc, matrix.MatrixToTensor(dhM))
	return part
}

func lstmAddDxGate(acc, dPre *tensor.Tensor, Wx *graph.Node) *tensor.Tensor {
	dPreM := matrix.TensorToMatrix(dPre)
	wxM := matrix.TensorToMatrix(Wx.Value)
	dxM, _ := matrix.MatMul(dPreM, wxM)
	part, _ := tensor.Add(acc, matrix.MatrixToTensor(dxM))
	return part
}

func sigmoidTensor(t *tensor.Tensor) *tensor.Tensor {
	return tensor.Apply(t, sigmoid)
}

func tanhTensor(t *tensor.Tensor) *tensor.Tensor {
	return tensor.Apply(t, math.Tanh)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidGradFromOutput(out, dOut *tensor.Tensor) *tensor.Tensor {
	deriv, _ := tensor.Mul(out, tensor.Apply(out, func(v float64) float64 { return 1 - v }))
	res, _ := tensor.Mul(deriv, dOut)
	return res
}

func tanhGradFromOutput(out, dOut *tensor.Tensor) *tensor.Tensor {
	deriv := tensor.Apply(out, func(v float64) float64 { return 1 - v*v })
	res, _ := tensor.Mul(deriv, dOut)
	return res
}

func cloneTensor(t *tensor.Tensor) *tensor.Tensor {
	data := append([]float64(nil), t.Data...)
	return &tensor.Tensor{
		Data:    data,
		Shape:   append([]int{}, t.Shape...),
		Strides: append([]int{}, t.Strides...),
	}
}

func copySliceOffset(dest *tensor.Tensor, src *tensor.Tensor, step, featureOffset int) {
	b, s, f := dest.Shape[0], dest.Shape[1], dest.Shape[2]
	h := src.Shape[1]
	for i := 0; i < b; i++ {
		destStart := i*s*f + step*f + featureOffset
		copy(dest.Data[destStart:destStart+h], src.Data[i*h:(i+1)*h])
	}
}

func extractSliceOffset(t *tensor.Tensor, step, featureOffset, hiddenSize int) *tensor.Tensor {
	b, s, f := t.Shape[0], t.Shape[1], t.Shape[2]
	_ = f
	res := make([]float64, b*hiddenSize)
	for i := 0; i < b; i++ {
		srcStart := i*s*f + step*f + featureOffset
		copy(res[i*hiddenSize:(i+1)*hiddenSize], t.Data[srcStart:srcStart+hiddenSize])
	}
	return &tensor.Tensor{Data: res, Shape: []int{b, hiddenSize}, Strides: []int{hiddenSize, 1}}
}
