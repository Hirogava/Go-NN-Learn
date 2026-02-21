package layers

func SetEvalMode(module Module) {
	for _, layer := range module.Layers() {
		if bn, ok := layer.(*BatchNorm); ok {
			bn.Eval()
		}

		if do, ok := layer.(*Dropout); ok {
			do.training = false // или do.Eval(), если добавишь метод
		}
	}
}
