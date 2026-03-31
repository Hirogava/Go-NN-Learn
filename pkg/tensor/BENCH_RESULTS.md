Добавлено:

- [`pkg/tensor/benchmark_test.go`](/mnt/c/Users/mygeetoq/repos/Go_projects/Go-NN-Learn/pkg/tensor/benchmark_test.go)
- Benchmark `MatMul` для размеров `32x32x32`, `64x64x64`, `128x128x128`, `256x256x256`, `512x512x512`
- Benchmark `Conv2D.Forward` для нескольких входных конфигураций
- Benchmark сравнения `SIMD` и обычных CPU-циклов для `Add`, `Mul`, `DotProduct`

Команды запуска:
прогон с BLAS/OpenBLAS:

- Установлен пакет `libopenblas-dev`
- Для сборки BLAS-пути дополнительно исправлены build tags и include path в [`pkg/tensor/blas_nocgo.go`](/mnt/c/Users/mygeetoq/repos/Go_projects/Go-NN-Learn/pkg/tensor/blas_nocgo.go) и [`pkg/tensor/blas_cgo.go`](/mnt/c/Users/mygeetoq/repos/Go_projects/Go-NN-Learn/pkg/tensor/blas_cgo.go)

Команды запуска:

```bash
GOCACHE=/tmp/go-build-cache GOMODCACHE=/tmp/go-mod-cache go test ./pkg/tensor -run '^$' -bench '^BenchmarkMatMulSizes$' -benchmem
GOCACHE=/tmp/go-build-cache GOMODCACHE=/tmp/go-mod-cache go test ./pkg/tensor -run '^$' -bench '^BenchmarkConv2DForward$' -benchmem
GOCACHE=/tmp/go-build-cache GOMODCACHE=/tmp/go-mod-cache go test ./pkg/tensor -run '^$' -bench '^BenchmarkSIMDVsCPU$' -benchmem
```

Результаты `MatMul` с BLAS:

- `32x32x32`: `20972 ns/op`, `9751 B/op`, `8 allocs/op`
- `64x64x64`: `145150 ns/op`, `34787 B/op`, `8 allocs/op`
- `128x128x128`: `570164 ns/op`, `179477 B/op`, `13 allocs/op`
- `256x256x256`: `2741472 ns/op`, `700832 B/op`, `31 allocs/op`
- `512x512x512`: `7782247 ns/op`, `2097313 B/op`, `4 allocs/op`

Сравнение pure Go vs BLAS для `MatMul`:

- `32x32x32`: `21888 ns/op` vs `20972 ns/op`
- `64x64x64`: `149701 ns/op` vs `145150 ns/op`
- `128x128x128`: `544220 ns/op` vs `570164 ns/op`
- `256x256x256`: `2530425 ns/op` vs `2741472 ns/op`
- `512x512x512`: `16545049 ns/op` vs `7782247 ns/op`

Результаты `Conv2D.Forward` с BLAS:

- `small_1x3x28x28_k3`: `7747 ns/op`, `57679 B/op`, `6 allocs/op`
- `medium_8x3x32x32_k3`: `88635 ns/op`, `1049133 B/op`, `6 allocs/op`
- `large_16x16x64x64_k3`: `809203 ns/op`, `16778679 B/op`, `7 allocs/op`

Результаты `SIMD vs CPU` с BLAS-сборкой:

- `AddSIMD_1024`: `372.6 ns/op`
- `AddCPU_1024`: `503.2 ns/op`
- `MulSIMD_1024`: `374.1 ns/op`
- `MulCPU_1024`: `483.4 ns/op`
- `DotSIMD_1024`: `259.8 ns/op`
- `DotCPU_1024`: `259.0 ns/op`
- `AddSIMD_16384`: `7047 ns/op`
- `AddCPU_16384`: `7420 ns/op`
- `MulSIMD_16384`: `7082 ns/op`
- `MulCPU_16384`: `7526 ns/op`
- `DotSIMD_16384`: `5828 ns/op`
- `DotCPU_16384`: `4412 ns/op`
- `AddSIMD_262144`: `146019 ns/op`
- `AddCPU_262144`: `145339 ns/op`
- `MulSIMD_262144`: `160716 ns/op`
- `MulCPU_262144`: `144170 ns/op`
- `DotSIMD_262144`: `103009 ns/op`
- `DotCPU_262144`: `64902 ns/op`

Обновленные выводы:

- BLAS/OpenBLAS дает заметный выигрыш на больших матрицах, особенно на `512x512x512`, где время снизилось примерно с `16.5 ms` до `7.8 ms`.
- На малых и части средних размеров BLAS не дает гарантированного выигрыша, а иногда оказывается медленнее pure Go пути из-за overhead вызова внешней библиотеки.
- Это означает, что порог переключения на BLAS должен подбираться эмпирически по машине и не обязан быть выгодным на всех размерах подряд.
