package api

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/Hirogava/Go-NN-Learn/pkg/api/text"
	"github.com/Hirogava/Go-NN-Learn/pkg/autograd"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/layers"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/optimizers"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
	"github.com/Hirogava/Go-NN-Learn/pkg/tensor/graph"
)

var (
	ErrEmptyDataset = errors.New("empty dataset")
	ErrNotFitted    = errors.New("model not fitted")
	ErrUnknownLabel = errors.New("unknown label")
)

// CONFIG

type TrainConfig struct {
	Epochs    int
	BatchSize int
	LR        float64
	Seed      int64
}

// CLASSIFIER

type TextClassifier struct {
	textCfg  text.PreprocessConfig
	trainCfg TrainConfig

	vocab *text.Vocab
	le    *labelEncoder

	model layers.Module
}

// LABEL ENCODER

type labelEncoder struct {
	classToIdx map[string]int
	idxToClass []string
}

func newLabelEncoder(labels []string) *labelEncoder {
	m := map[string]int{}
	classes := []string{}

	for _, l := range labels {
		if _, ok := m[l]; !ok {
			m[l] = len(classes)
			classes = append(classes, l)
		}
	}

	return &labelEncoder{
		classToIdx: m,
		idxToClass: classes,
	}
}

func (le *labelEncoder) encode(labels []string) ([]int, error) {
	out := make([]int, len(labels))

	for i, l := range labels {
		idx, ok := le.classToIdx[l]
		if !ok {
			return nil, ErrUnknownLabel
		}
		out[i] = idx
	}

	return out, nil
}

// CONSTRUCTOR

func NewTextClassifier(cfg text.PreprocessConfig, trainCfg TrainConfig) *TextClassifier {
	return &TextClassifier{
		textCfg:  cfg,
		trainCfg: trainCfg,
	}
}

// HELPERS

func featuresToTensor(features [][]float32) *tensor.Tensor {
	n := len(features)
	d := len(features[0])

	data := make([]float64, 0, n*d)

	for _, row := range features {
		for _, v := range row {
			data = append(data, float64(v))
		}
	}

	return &tensor.Tensor{
		Data:  data,
		Shape: []int{n, d},
	}
}

func labelsToOneHot(labels []int, numClasses int) *tensor.Tensor {
	n := len(labels)
	data := make([]float64, n*numClasses)

	for i, l := range labels {
		data[i*numClasses+l] = 1.0
	}

	return &tensor.Tensor{
		Data:  data,
		Shape: []int{n, numClasses},
	}
}

// FIT

func (tc *TextClassifier) Fit(texts []string, labels []string) error {
	if len(texts) == 0 {
		return ErrEmptyDataset
	}

	out, err := text.RunPipeline(texts, tc.textCfg)
	if err != nil {
		return err
	}

	tc.vocab = out.Vocab
	tc.le = newLabelEncoder(labels)

	yIdx, err := tc.le.encode(labels)
	if err != nil {
		return err
	}

	X := featuresToTensor(out.Features)
	Y := labelsToOneHot(yIdx, len(tc.le.idxToClass))

	ds := dataloader.NewSimpleDataset(X, Y)

	loader := dataloader.NewDataLoader(ds, dataloader.DataLoaderConfig{
		BatchSize: tc.trainCfg.BatchSize,
		Shuffle:   true,
		Seed:      tc.trainCfg.Seed,
	})

	inputDim := len(tc.vocab.IdxToToken)
	numClasses := len(tc.le.idxToClass)

	tc.model = optimizers.NewSequential(
		Dense(inputDim, numClasses, tc.trainCfg.Seed),
	)

	opt := optimizers.NewAdam(tc.trainCfg.LR, 0.9, 0.999, 1e-8)
	acc := metrics.NewAccuracy()

	for epoch := 0; epoch < tc.trainCfg.Epochs; epoch++ {

		loader.Reset()
		acc.Reset()

		for loader.HasNext() {
			batch := loader.Next()

			ctx := autograd.NewGraph()
			ctx.WithGrad()
			autograd.SetGraph(ctx)

			xNode := graph.NewNode(batch.Features, nil, nil)
			logits := tc.model.Forward(xNode)

			loss := ctx.Engine().SoftmaxCrossEntropy(logits, batch.Targets)

			ctx.Backward(loss)

			opt.Step(tc.model.Params())
			opt.ZeroGrad(tc.model.Params())

			// accuracy
			numClasses := len(tc.le.idxToClass)
			preds := make([]float64, len(batch.Targets.Data)/numClasses)
			labelsIdx := make([]float64, len(preds))

			for i := 0; i < len(preds); i++ {
				maxIdx := 0
				maxVal := logits.Value.Data[i*numClasses]

				for j := 1; j < numClasses; j++ {
					v := logits.Value.Data[i*numClasses+j]
					if v > maxVal {
						maxVal = v
						maxIdx = j
					}
				}

				preds[i] = float64(maxIdx)

				for j := 0; j < numClasses; j++ {
					if batch.Targets.Data[i*numClasses+j] == 1 {
						labelsIdx[i] = float64(j)
						break
					}
				}
			}

			acc.Update(preds, labelsIdx)
		}

		fmt.Printf("epoch %d/%d acc=%.4f\n",
			epoch+1,
			tc.trainCfg.Epochs,
			acc.Value(),
		)
	}

	return nil
}

// PREDICT PROBA

func (tc *TextClassifier) PredictProba(texts []string) ([][]float64, error) {
	if tc.model == nil || tc.vocab == nil {
		return nil, ErrNotFitted
	}

	features := text.Transform(texts, tc.vocab, tc.textCfg)
	X := featuresToTensor(features)

	xNode := graph.NewNode(X, nil, nil)
	logits := tc.model.Forward(xNode)

	numClasses := len(tc.le.idxToClass)

	out := make([][]float64, len(texts))

	for i := 0; i < len(texts); i++ {
		row := logits.Value.Data[i*numClasses : (i+1)*numClasses]

		expVals := make([]float64, numClasses)
		sum := 0.0

		for j, v := range row {
			e := math.Exp(v)
			expVals[j] = e
			sum += e
		}

		probs := make([]float64, numClasses)
		for j := range expVals {
			probs[j] = expVals[j] / sum
		}

		out[i] = probs
	}

	return out, nil
}

// PREDICT

func (tc *TextClassifier) Predict(texts []string) ([]string, error) {
	probs, err := tc.PredictProba(texts)
	if err != nil {
		return nil, err
	}

	out := make([]string, len(probs))

	for i, p := range probs {
		maxIdx := 0
		maxVal := p[0]

		for j := 1; j < len(p); j++ {
			if p[j] > maxVal {
				maxVal = p[j]
				maxIdx = j
			}
		}

		out[i] = tc.le.idxToClass[maxIdx]
	}

	return out, nil
}

// SAVE

func (tc *TextClassifier) Save(path string) error {
	if tc.model == nil || tc.vocab == nil {
		return ErrNotFitted
	}

	weights, err := SaveCheckpointToBytes(tc.model)
	if err != nil {
		return err
	}

	meta := Metadata{
		Classes: tc.le.idxToClass,
		Vocab:   tc.vocab,
		TextCfg: tc.textCfg,
	}

	return SaveArtifact(path, meta, weights)
}

// LOAD

func LoadTextClassifier(path string) (*TextClassifier, error) {
	art, err := LoadArtifact(path)
	if err != nil {
		return nil, err
	}

	tc := &TextClassifier{
		textCfg: art.Metadata.TextCfg,
		vocab:   art.Metadata.Vocab,
	}

	tc.le = &labelEncoder{
		classToIdx: map[string]int{},
		idxToClass: art.Metadata.Classes,
	}

	for i, c := range art.Metadata.Classes {
		tc.le.classToIdx[c] = i
	}

	inputDim := len(tc.vocab.IdxToToken)
	numClasses := len(tc.le.idxToClass)

	tc.model = optimizers.NewSequential(
		Dense(inputDim, numClasses, 42), // seed не важен при загрузке весов
	)

	if err := LoadCheckpointFromBytes(tc.model, art.Weights); err != nil {
		return nil, err
	}

	return tc, nil
}

// Dense — удобная обёртка над layers.NewDense с seed и нормальной инициализацией
func Dense(inputDim, outputDim int, seed int64) *layers.Dense {
	return layers.NewDense(inputDim, outputDim, func(w []float64) {
		r := rand.New(rand.NewSource(seed))
		for i := range w {
			w[i] = r.NormFloat64() * 0.01
		}
	})
}
