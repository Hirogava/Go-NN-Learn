package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/Hirogava/Go-NN-Learn/pkg/dataloader"
)

func main() {
	dataDir := "./data/mnist"

	fmt.Println("Checking for MNIST data...")
	err := dataloader.DownloadMNIST(dataDir)
	if err != nil {
		log.Fatalf("Failed to download MNIST: %v", err)
	}

	fmt.Println("Loading training data...")
	trainImg := filepath.Join(dataDir, "train-images-idx3-ubyte.gz")
	trainLbl := filepath.Join(dataDir, "train-labels-idx1-ubyte.gz")

	trainDS, err := dataloader.LoadMNIST(trainImg, trainLbl, dataloader.DefaultMNISTConfig())
	if err != nil {
		log.Fatalf("Failed to load MNIST training data: %v", err)
	}

	fmt.Printf("Successfully loaded %d training samples\n", trainDS.Len())

	// Пример получения первого сэмпла
	x, y := trainDS.Get(0)
	fmt.Printf("Image shape: %v, Label shape: %v\n", x.Shape, y.Shape)
	
	// Вывод первой метки (One-Hot)
	fmt.Printf("First label (One-Hot): %v\n", y.Data)
}
