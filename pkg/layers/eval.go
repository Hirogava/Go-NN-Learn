package layers

type Trainable interface {
	Train()
	Eval()
}

func SetEvalMode(module Module) {
	for _, layer := range module.Layers() {
		if t, ok := layer.(Trainable); ok {
			t.Eval()
		}
	}
}

func SetTrainMode(module Module) {
	for _, layer := range module.Layers() {
		if t, ok := layer.(Trainable); ok {
			t.Train()
		}
	}
}
