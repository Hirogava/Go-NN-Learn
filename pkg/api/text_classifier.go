package api

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/Hirogava/Go-NN-Learn/pkg/api/text"
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/gnn"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

var (
	ErrEmptyDataset  = errors.New("empty dataset")
	ErrUnknownLabel  = errors.New("unknown label")
	ErrNotTrained    = errors.New("text classifier is not trained")
	ErrArtifactShape = errors.New("artifact mismatch")
)

type TrainConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	Seed         int64
	HiddenDim    int
	TextConfig   text.PreprocessConfig
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Epochs:       10,
		BatchSize:    8,
		LearningRate: 0.01,
		Seed:         42,
		HiddenDim:    32,
		TextConfig:   text.DefaultConfig(),
	}
}

type FitEvent struct {
	Epoch     int
	Epochs    int
	BatchSize int
	Batches   int
	Loss      float64
	Accuracy  float64
}

type Callback func(FitEvent)

func NewConsoleProgressCallback(w io.Writer) Callback {
	if w == nil {
		w = os.Stdout
	}
	return func(e FitEvent) {
		fmt.Fprintf(w, "epoch %d/%d | loss=%.4f | acc=%.4f\n", e.Epoch+1, e.Epochs, e.Loss, e.Accuracy)
	}
}

type TextClassifier struct {
	cfg       TrainConfig
	callbacks []Callback

	model      *textClassifierModel
	vocab      *text.Vocab
	classes    []string
	classToIdx map[string]int

	trained bool
}

func NewTextClassifier(cfg TrainConfig, callbacks ...Callback) *TextClassifier {
	cfg = normalizeTrainConfig(cfg)

	if len(callbacks) == 0 {
		callbacks = []Callback{NewConsoleProgressCallback(os.Stdout)}
	}

	return &TextClassifier{
		cfg:       cfg,
		callbacks: callbacks,
	}
}

func (c *TextClassifier) Fit(texts []string, labels []string) error {
	if len(texts) == 0 || len(labels) == 0 {
		return ErrEmptyDataset
	}
	if len(texts) != len(labels) {
		return fmt.Errorf("fit: texts/labels length mismatch: %d != %d", len(texts), len(labels))
	}

	for _, label := range labels {
		if strings.TrimSpace(label) == "" {
			return ErrUnknownLabel
		}
	}

	featuresOut, err := text.RunPipeline(texts, c.cfg.TextConfig)
	if err != nil {
		return err
	}

	classes, classToIdx, err := buildLabelMapping(labels)
	if err != nil {
		return err
	}

	featuresTensor, err := featuresToTensor(featuresOut.Features)
	if err != nil {
		return err
	}

	targetsTensor, err := oneHotTargets(labels, classToIdx)
	if err != nil {
		return err
	}

	c.vocab = featuresOut.Vocab
	c.classes = classes
	c.classToIdx = classToIdx
	c.model = newTextClassifierModel(len(c.vocab.IdxToToken), c.cfg.HiddenDim, len(c.classes), c.cfg.Seed)

	batchSize := c.cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	if batchSize > len(texts) {
		batchSize = len(texts)
	}

	ds := dataloader.NewSimpleDataset(featuresTensor, targetsTensor)
	loader := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: batchSize,
		Shuffle:    true,
		Seed:       c.cfg.Seed,
	})

	opt := optimizers.NewAdam(c.cfg.LearningRate, 0.9, 0.999, 1e-8)

	for epoch := 0; epoch < c.cfg.Epochs; epoch++ {
		loader.Reset()

		var sumLoss float64
		var sumCorrect float64
		var seen int
		var batches int

		for loader.HasNext() {
			batch := loader.Next()
			batches++

			ctx := autograd.NewGraph()
			ctx.WithGrad()
			autograd.SetGraph(ctx)

			input := graph.NewNode(batch.Features, nil, nil)
			logits := c.model.Forward(input)
			if logits == nil {
				return fmt.Errorf("fit: model returned nil logits")
			}

			lossNode := ctx.Engine().SoftmaxCrossEntropy(logits, batch.Targets)
			if lossNode == nil || lossNode.Value == nil {
				return fmt.Errorf("fit: loss is nil")
			}

			ctx.Backward(lossNode)
			opt.Step(c.model.Params())
			opt.ZeroGrad(c.model.Params())

			batchLoss := meanTensor(lossNode.Value)
			batchCorrect := batchAccuracy(logits.Value, batch.Targets)

			batchSizeNow := batch.Features.Shape[0]
			sumLoss += batchLoss * float64(batchSizeNow)
			sumCorrect += batchCorrect * float64(batchSizeNow)
			seen += batchSizeNow
		}

		if seen == 0 {
			return ErrEmptyDataset
		}

		c.notify(FitEvent{
			Epoch:     epoch,
			Epochs:    c.cfg.Epochs,
			BatchSize: batchSize,
			Batches:   batches,
			Loss:      sumLoss / float64(seen),
			Accuracy:  sumCorrect / float64(seen),
		})
	}

	c.trained = true
	return nil
}

func (c *TextClassifier) Predict(texts []string) ([]string, error) {
	proba, err := c.PredictProba(texts)
	if err != nil {
		return nil, err
	}

	labels := make([]string, len(proba))
	for i, row := range proba {
		labels[i] = c.classes[argmax(row)]
	}
	return labels, nil
}

func (c *TextClassifier) PredictProba(texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, ErrEmptyDataset
	}
	if err := c.ensureReady(); err != nil {
		return nil, err
	}

	features := text.Transform(texts, c.vocab, c.cfg.TextConfig)
	featuresTensor, err := featuresToTensor(features)
	if err != nil {
		return nil, err
	}

	var logits *graph.Node
	gnn.NoGrad(func() {
		input := graph.NewNode(featuresTensor, nil, nil)
		logits = c.model.Forward(input)
	})
	if logits == nil || logits.Value == nil {
		return nil, fmt.Errorf("predict: model returned nil logits")
	}

	return softmaxRows(logits.Value)
}

func (c *TextClassifier) Save(path string) error {
	if err := c.ensureReady(); err != nil {
		return err
	}

	weights, err := SaveCheckpointToBytes(c.model)
	if err != nil {
		return err
	}

	meta := Metadata{
		Classes:   append([]string(nil), c.classes...),
		Vocab:     c.vocab,
		HiddenDim: c.cfg.HiddenDim,
		TextCfg:   c.cfg.TextConfig,
	}

	return SaveArtifact(path, meta, weights)
}

func (c *TextClassifier) Load(path string) error {
	art, err := LoadArtifact(path)
	if err != nil {
		return err
	}

	if art.Metadata.Vocab == nil || len(art.Metadata.Classes) == 0 {
		return ErrInvalidArtifact
	}

	c.vocab = art.Metadata.Vocab
	c.classes = append([]string(nil), art.Metadata.Classes...)
	c.classToIdx = make(map[string]int, len(c.classes))
	for i, className := range c.classes {
		c.classToIdx[className] = i
	}
	c.cfg.HiddenDim = art.Metadata.HiddenDim
	c.cfg.TextConfig = art.Metadata.TextCfg
	c.model = newTextClassifierModel(len(c.vocab.IdxToToken), c.cfg.HiddenDim, len(c.classes), c.cfg.Seed)

	if err := LoadCheckpointFromBytes(c.model, art.Weights); err != nil {
		return err
	}

	c.trained = true
	return nil
}

func (c *TextClassifier) notify(e FitEvent) {
	for _, cb := range c.callbacks {
		if cb != nil {
			cb(e)
		}
	}
}

func (c *TextClassifier) ensureReady() error {
	if !c.trained || c.model == nil || c.vocab == nil || len(c.classes) == 0 {
		return ErrNotTrained
	}
	return nil
}

func normalizeTrainConfig(cfg TrainConfig) TrainConfig {
	def := DefaultTrainConfig()

	if cfg.Epochs <= 0 {
		cfg.Epochs = def.Epochs
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = def.BatchSize
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = def.LearningRate
	}
	if cfg.HiddenDim <= 0 {
		cfg.HiddenDim = def.HiddenDim
	}
	if cfg.Seed == 0 {
		cfg.Seed = def.Seed
	}
	if cfg.TextConfig == (text.PreprocessConfig{}) {
		cfg.TextConfig = def.TextConfig
	}

	return cfg
}

func buildLabelMapping(labels []string) ([]string, map[string]int, error) {
	set := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		trimmed := strings.TrimSpace(label)
		if trimmed == "" {
			return nil, nil, ErrUnknownLabel
		}
		set[trimmed] = struct{}{}
	}

	classes := make([]string, 0, len(set))
	for label := range set {
		classes = append(classes, label)
	}
	sort.Strings(classes)

	mapping := make(map[string]int, len(classes))
	for i, label := range classes {
		mapping[label] = i
	}

	return classes, mapping, nil
}

func oneHotTargets(labels []string, mapping map[string]int) (*tensor.Tensor, error) {
	rows := len(labels)
	cols := len(mapping)
	if rows == 0 || cols == 0 {
		return nil, ErrEmptyDataset
	}

	targets := tensor.Zeros(rows, cols)
	for i, label := range labels {
		idx, ok := mapping[strings.TrimSpace(label)]
		if !ok {
			return nil, ErrUnknownLabel
		}
		targets.Data[i*targets.Strides[0]+idx] = 1.0
	}

	return targets, nil
}

func featuresToTensor(features [][]float32) (*tensor.Tensor, error) {
	if len(features) == 0 {
		return nil, ErrEmptyDataset
	}

	cols := len(features[0])
	for i := range features {
		if len(features[i]) != cols {
			return nil, fmt.Errorf("feature shape mismatch at row %d", i)
		}
	}

	out := tensor.Zeros(len(features), cols)
	for i, row := range features {
		for j, v := range row {
			out.Data[i*out.Strides[0]+j] = float64(v)
		}
	}

	return out, nil
}

func meanTensor(t *tensor.Tensor) float64 {
	if t == nil || len(t.Data) == 0 {
		return 0
	}
	var sum float64
	for _, v := range t.Data {
		sum += v
	}
	return sum / float64(len(t.Data))
}

func batchAccuracy(logits *tensor.Tensor, targets *tensor.Tensor) float64 {
	if logits == nil || targets == nil || len(logits.Shape) != 2 || len(targets.Shape) != 2 {
		return 0
	}

	rows := logits.Shape[0]
	cols := logits.Shape[1]
	if rows == 0 || cols == 0 || targets.Shape[0] != rows || targets.Shape[1] != cols {
		return 0
	}

	var correct int
	for i := 0; i < rows; i++ {
		logitBase := i * logits.Strides[0]
		targetBase := i * targets.Strides[0]

		predIdx := 0
		predMax := logits.Data[logitBase]
		for j := 1; j < cols; j++ {
			if logits.Data[logitBase+j] > predMax {
				predMax = logits.Data[logitBase+j]
				predIdx = j
			}
		}

		targetIdx := 0
		for j := 1; j < cols; j++ {
			if targets.Data[targetBase+j] > targets.Data[targetBase+targetIdx] {
				targetIdx = j
			}
		}

		if predIdx == targetIdx {
			correct++
		}
	}

	return float64(correct) / float64(rows)
}

func softmaxRows(t *tensor.Tensor) ([][]float64, error) {
	if t == nil || len(t.Shape) != 2 {
		return nil, fmt.Errorf("softmax: expected 2D tensor")
	}

	rows, cols := t.Shape[0], t.Shape[1]
	out := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		base := i * t.Strides[0]
		row := make([]float64, cols)

		maxVal := t.Data[base]
		for j := 1; j < cols; j++ {
			if t.Data[base+j] > maxVal {
				maxVal = t.Data[base+j]
			}
		}

		var sum float64
		for j := 0; j < cols; j++ {
			row[j] = math.Exp(t.Data[base+j] - maxVal)
			sum += row[j]
		}

		if sum == 0 {
			for j := range row {
				row[j] = 1.0 / float64(cols)
			}
		} else {
			for j := range row {
				row[j] /= sum
			}
		}

		out[i] = row
	}

	return out, nil
}

func argmax(values []float64) int {
	if len(values) == 0 {
		return 0
	}
	bestIdx := 0
	bestVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > bestVal {
			bestVal = values[i]
			bestIdx = i
		}
	}
	return bestIdx
}

type textClassifierModel struct {
	hidden *layers.Dense
	output *layers.Dense
}

func newTextClassifierModel(inputDim, hiddenDim, numClasses int, seed int64) *textClassifierModel {
	rng := rand.New(rand.NewSource(seed))
	initWeights := func(data []float64) {
		for i := range data {
			data[i] = rng.NormFloat64() * 0.01
		}
	}

	return &textClassifierModel{
		hidden: layers.NewDense(inputDim, hiddenDim, initWeights),
		output: layers.NewDense(hiddenDim, numClasses, initWeights),
	}
}

func (m *textClassifierModel) Forward(x *graph.Node) *graph.Node {
	if m == nil || x == nil {
		return nil
	}
	out := m.hidden.Forward(x)
	out = reluNode(out)
	return m.output.Forward(out)
}

func (m *textClassifierModel) Params() []*graph.Node {
	if m == nil {
		return nil
	}
	params := m.hidden.Params()
	params = append(params, m.output.Params()...)
	return params
}

func (m *textClassifierModel) Layers() []layers.Layer { return nil }
func (m *textClassifierModel) Train()                 {}
func (m *textClassifierModel) Eval()                  {}

type reluOp struct {
	input *graph.Node
}

func reluNode(x *graph.Node) *graph.Node {
	if x == nil {
		return nil
	}

	out := tensor.Zeros(x.Value.Shape...)
	for i, v := range x.Value.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}

	return graph.NewNode(out, []*graph.Node{x}, &reluOp{input: x})
}

func (op *reluOp) Backward(grad *tensor.Tensor) {
	if op == nil || op.input == nil || op.input.Value == nil {
		return
	}
	gradInput := tensor.Zeros(op.input.Value.Shape...)
	for i, v := range op.input.Value.Data {
		if v > 0 {
			gradInput.Data[i] = grad.Data[i]
		}
	}
	op.input.Grad = gradInput
}