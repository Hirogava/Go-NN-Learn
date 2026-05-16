package dataloader

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/Hirogava/Go-NN-Learn/pkg/tensor"
)

const (
	mnistImageMagic = 2051
	mnistLabelMagic = 2049
)

var mnistUrls = map[string]string{
	"train-images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
	"train-labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
	"t10k-images":  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
	"t10k-labels":  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

// MNISTConfig содержит настройки загрузки MNIST.
type MNISTConfig struct {
	Normalize bool // Нормализовать значения пикселей в диапазон [0, 1]
	OneHot    bool // Преобразовать метки в One-Hot encoding (10 классов)
}

// DefaultMNISTConfig возвращает конфигурацию по умолчанию.
func DefaultMNISTConfig() MNISTConfig {
	return MNISTConfig{
		Normalize: true,
		OneHot:    true,
	}
}

// LoadMNIST загружает датасет MNIST из файлов формата IDX.
// Поддерживает как обычные файлы, так и сжатые (.gz).
func LoadMNIST(imagesPath, labelsPath string, cfg MNISTConfig) (Dataset, error) {
	images, err := readIdxFile(imagesPath, mnistImageMagic)
	if err != nil {
		return nil, fmt.Errorf("failed to read images: %w", err)
	}

	labels, err := readIdxFile(labelsPath, mnistLabelMagic)
	if err != nil {
		return nil, fmt.Errorf("failed to read labels: %w", err)
	}

	if images.Shape[0] != labels.Shape[0] {
		return nil, fmt.Errorf("images and labels count mismatch: %d != %d", images.Shape[0], labels.Shape[0])
	}

	// Нормализация изображений [0, 255] -> [0, 1]
	if cfg.Normalize {
		for i := range images.Data {
			images.Data[i] /= 255.0
		}
	}

	// One-Hot кодирование меток
	var finalLabels *tensor.Tensor
	if cfg.OneHot {
		numSamples := labels.Shape[0]
		finalLabels = tensor.Zeros(numSamples, 10)
		for i := range numSamples {
			labelIdx := int(labels.Data[i])
			if labelIdx < 0 || labelIdx >= 10 {
				return nil, fmt.Errorf("invalid label at index %d: %d", i, labelIdx)
			}
			finalLabels.Data[i*10+labelIdx] = 1.0
		}
	} else {
		finalLabels = labels
	}

	return NewSimpleDataset(images, finalLabels), nil
}

// DownloadMNIST скачивает файлы MNIST в указанную директорию, если они отсутствуют.
func DownloadMNIST(targetDir string) error {
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return err
	}

	for _, url := range mnistUrls {
		filename := filepath.Base(url)
		path := filepath.Join(targetDir, filename)

		if _, err := os.Stat(path); err == nil {
			continue // Файл уже есть
		}

		fmt.Printf("Downloading %s...\n", url)
		resp, err := http.Get(url)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("bad status: %s", resp.Status)
		}

		out, err := os.Create(path)
		if err != nil {
			return err
		}
		defer out.Close()

		if _, err := io.Copy(out, resp.Body); err != nil {
			return err
		}
	}
	return nil
}

func readIdxFile(path string, expectedMagic int32) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gzr, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer gzr.Close()
		r = gzr
	}

	var magic int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != expectedMagic {
		return nil, fmt.Errorf("invalid magic number: %d (expected %d)", magic, expectedMagic)
	}

	var numItems int32
	if err := binary.Read(r, binary.BigEndian, &numItems); err != nil {
		return nil, err
	}

	var rows, cols int32 = 1, 1
	if expectedMagic == mnistImageMagic {
		binary.Read(r, binary.BigEndian, &rows)
		binary.Read(r, binary.BigEndian, &cols)
	}

	size := int(numItems * rows * cols)
	data := make([]byte, size)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, err
	}

	floatData := make([]float64, size)
	for i, b := range data {
		floatData[i] = float64(b)
	}

	shape := []int{int(numItems), int(rows * cols)}
	if expectedMagic == mnistLabelMagic {
		shape = []int{int(numItems), 1}
	}

	return &tensor.Tensor{
		Data:    floatData,
		Shape:   shape,
		Strides: []int{int(rows * cols), 1},
	}, nil
}
