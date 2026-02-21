package gnn

import (
	"github.com/Hirogava/Go-NN-Learn/internal/autograd"
	"github.com/Hirogava/Go-NN-Learn/internal/backend"
	"github.com/Hirogava/Go-NN-Learn/internal/backend/graph"
	"github.com/Hirogava/Go-NN-Learn/internal/layers"
	"github.com/Hirogava/Go-NN-Learn/internal/optimizers"

	"github.com/Hirogava/Go-NN-Learn/pkg/api"
	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
	"github.com/Hirogava/Go-NN-Learn/pkg/metrics"
	"github.com/Hirogava/Go-NN-Learn/pkg/train"
)

type Sequential = optimizers.Sequential
type Dense = layers.Dense

// type ReLU отстутствует в проекте, есть только ReLuOp

type Adam = optimizers.Adam
type SGD = optimizers.StochasticGradientDescent

type Train = train.Trainer

var Predict = api.Predict

type Dataset = dataloader.Dataset
type DataLoader = dataloader.DataLoader

// для MNIST
type Tensor = backend.Tensor
type DataLoaderConfig = dataloader.DataLoaderConfig

var NewSimpleDataset = dataloader.NewSimpleDataset
var NewDataLoader = dataloader.NewDataLoader
var NewTrainer = train.NewTrainer
var Zeros = backend.Zeros
var NewDense = layers.NewDense
var NewAdam = optimizers.NewAdam
var NewEngine = autograd.NewEngine
var NewAccuracy = metrics.NewAccuracy
var NewNode = graph.NewNode
