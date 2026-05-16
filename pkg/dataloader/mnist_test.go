package dataloader

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"os"
	"testing"
)

func TestLoadMNIST(t *testing.T) {
	numSamples := 10
	rows, cols := 28, 28

	// Хелпер для создания dummy IDX данных
	createIdxData := func(magic int32, counts ...int32) []byte {
		buf := new(bytes.Buffer)
		binary.Write(buf, binary.BigEndian, magic)
		for _, count := range counts {
			binary.Write(buf, binary.BigEndian, count)
		}
		size := int32(1)
		for _, count := range counts {
			size *= count
		}
		buf.Write(make([]byte, size))
		return buf.Bytes()
	}

	t.Run("PlainFiles", func(t *testing.T) {
		imgFile, lblFile := "test_imgs.idx", "test_lbls.idx"
		os.WriteFile(imgFile, createIdxData(mnistImageMagic, int32(numSamples), int32(rows), int32(cols)), 0644)
		os.WriteFile(lblFile, createIdxData(mnistLabelMagic, int32(numSamples)), 0644)
		defer os.Remove(imgFile)
		defer os.Remove(lblFile)

		ds, err := LoadMNIST(imgFile, lblFile, DefaultMNISTConfig())
		if err != nil {
			t.Fatal(err)
		}
		if ds.Len() != numSamples {
			t.Errorf("expected %d, got %d", numSamples, ds.Len())
		}
	})

	t.Run("GzipFiles", func(t *testing.T) {
		imgFile, lblFile := "test_imgs.gz", "test_lbls.gz"
		
		writeGz := func(path string, data []byte) {
			f, _ := os.Create(path)
			gw := gzip.NewWriter(f)
			gw.Write(data)
			gw.Close()
			f.Close()
		}

		writeGz(imgFile, createIdxData(mnistImageMagic, int32(numSamples), int32(rows), int32(cols)))
		writeGz(lblFile, createIdxData(mnistLabelMagic, int32(numSamples)))
		defer os.Remove(imgFile)
		defer os.Remove(lblFile)

		ds, err := LoadMNIST(imgFile, lblFile, DefaultMNISTConfig())
		if err != nil {
			t.Fatal(err)
		}
		if ds.Len() != numSamples {
			t.Errorf("expected %d, got %d", numSamples, ds.Len())
		}
	})
}
