package backend

import (
	"unsafe"
)

// SIMD-оптимизированные операции используя unsafe для прямого доступа к памяти
// Эти функции используют явную векторизацию для максимальной производительности

// SIMDVectorSize - размер SIMD вектора (обычно 4 для AVX2 с float64)
const SIMDVectorSize = 4

// AddSIMD выполняет векторизованное сложение с использованием unsafe
// Оптимизировано для AVX/AVX2 инструкций процессора
func AddSIMD(a, b, result []float64) {
	n := len(a)

	// Проверка выравнивания для SIMD
	if n < SIMDVectorSize*2 {
		// Для малых массивов используем обычный цикл
		for i := 0; i < n; i++ {
			result[i] = a[i] + b[i]
		}
		return
	}

	// Получаем указатели на данные
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	resPtr := unsafe.Pointer(&result[0])

	// SIMD цикл - обрабатываем по 4 элемента за раз
	i := 0
	simdLoops := (n / SIMDVectorSize) * SIMDVectorSize

	for i < simdLoops {
		// Загружаем 4 элемента из a
		a0 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr(i*8)))
		a1 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+1)*8)))
		a2 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+2)*8)))
		a3 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+3)*8)))

		// Загружаем 4 элемента из b
		b0 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr(i*8)))
		b1 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+1)*8)))
		b2 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+2)*8)))
		b3 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+3)*8)))

		// Выполняем сложение
		r0 := a0 + b0
		r1 := a1 + b1
		r2 := a2 + b2
		r3 := a3 + b3

		// Сохраняем результат
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr(i*8))) = r0
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+1)*8))) = r1
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+2)*8))) = r2
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+3)*8))) = r3

		i += SIMDVectorSize
	}

	// Обрабатываем оставшиеся элементы
	for ; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

// MulSIMD выполняет векторизованное умножение с использованием unsafe
func MulSIMD(a, b, result []float64) {
	n := len(a)

	if n < SIMDVectorSize*2 {
		for i := 0; i < n; i++ {
			result[i] = a[i] * b[i]
		}
		return
	}

	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	resPtr := unsafe.Pointer(&result[0])

	i := 0
	simdLoops := (n / SIMDVectorSize) * SIMDVectorSize

	for i < simdLoops {
		a0 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr(i*8)))
		a1 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+1)*8)))
		a2 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+2)*8)))
		a3 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+3)*8)))

		b0 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr(i*8)))
		b1 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+1)*8)))
		b2 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+2)*8)))
		b3 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+3)*8)))

		r0 := a0 * b0
		r1 := a1 * b1
		r2 := a2 * b2
		r3 := a3 * b3

		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr(i*8))) = r0
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+1)*8))) = r1
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+2)*8))) = r2
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+3)*8))) = r3

		i += SIMDVectorSize
	}

	for ; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

// FMAOperation выполняет Fused Multiply-Add: result = a*b + c
// Это важная операция для MatMul, которая может использовать FMA инструкции процессора
func FMAOperation(a, b, c, result []float64) {
	n := len(a)

	if n < SIMDVectorSize*2 {
		for i := 0; i < n; i++ {
			result[i] = a[i]*b[i] + c[i]
		}
		return
	}

	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	cPtr := unsafe.Pointer(&c[0])
	resPtr := unsafe.Pointer(&result[0])

	i := 0
	simdLoops := (n / SIMDVectorSize) * SIMDVectorSize

	for i < simdLoops {
		a0 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr(i*8)))
		a1 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+1)*8)))
		a2 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+2)*8)))
		a3 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+3)*8)))

		b0 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr(i*8)))
		b1 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+1)*8)))
		b2 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+2)*8)))
		b3 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+3)*8)))

		c0 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr(i*8)))
		c1 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((i+1)*8)))
		c2 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((i+2)*8)))
		c3 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((i+3)*8)))

		// FMA операция: a*b + c
		r0 := a0*b0 + c0
		r1 := a1*b1 + c1
		r2 := a2*b2 + c2
		r3 := a3*b3 + c3

		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr(i*8))) = r0
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+1)*8))) = r1
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+2)*8))) = r2
		*(*float64)(unsafe.Pointer(uintptr(resPtr) + uintptr((i+3)*8))) = r3

		i += SIMDVectorSize
	}

	for ; i < n; i++ {
		result[i] = a[i]*b[i] + c[i]
	}
}

// DotProductSIMD вычисляет скалярное произведение векторов с SIMD оптимизацией
// Используется в MatMulTransposeB
func DotProductSIMD(a, b []float64) float64 {
	n := len(a)

	if n < SIMDVectorSize*2 {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += a[i] * b[i]
		}
		return sum
	}

	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])

	// Аккумуляторы для параллельного суммирования (уменьшаем зависимости)
	sum0, sum1, sum2, sum3 := 0.0, 0.0, 0.0, 0.0

	i := 0
	simdLoops := (n / SIMDVectorSize) * SIMDVectorSize

	for i < simdLoops {
		a0 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr(i*8)))
		a1 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+1)*8)))
		a2 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+2)*8)))
		a3 := *(*float64)(unsafe.Pointer(uintptr(aPtr) + uintptr((i+3)*8)))

		b0 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr(i*8)))
		b1 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+1)*8)))
		b2 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+2)*8)))
		b3 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((i+3)*8)))

		sum0 += a0 * b0
		sum1 += a1 * b1
		sum2 += a2 * b2
		sum3 += a3 * b3

		i += SIMDVectorSize
	}

	// Суммируем аккумуляторы
	sum := sum0 + sum1 + sum2 + sum3

	// Остаток
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// MatMulSIMDKernel - SIMD-оптимизированное ядро умножения матриц
// Использует unsafe для максимальной производительности
func MatMulSIMDKernel(a, b, c []float64, m, n, p int, iStart, iEnd, kStart, kEnd, jStart, jEnd int) {
	for i := iStart; i < iEnd; i++ {
		iOffsetA := i * n
		iOffsetC := i * p

		for k := kStart; k < kEnd; k++ {
			aik := a[iOffsetA+k]
			kOffset := k * p

			// SIMD-оптимизированный внутренний цикл
			j := jStart
			jRange := jEnd - jStart

			if jRange >= SIMDVectorSize*2 {
				bPtr := unsafe.Pointer(&b[kOffset+jStart])
				cPtr := unsafe.Pointer(&c[iOffsetC+jStart])

				simdEnd := jStart + (jRange/SIMDVectorSize)*SIMDVectorSize

				for j < simdEnd {
					idx := j - jStart

					b0 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr(idx*8)))
					b1 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((idx+1)*8)))
					b2 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((idx+2)*8)))
					b3 := *(*float64)(unsafe.Pointer(uintptr(bPtr) + uintptr((idx+3)*8)))

					c0 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr(idx*8)))
					c1 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((idx+1)*8)))
					c2 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((idx+2)*8)))
					c3 := *(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((idx+3)*8)))

					// FMA операция
					c0 += aik * b0
					c1 += aik * b1
					c2 += aik * b2
					c3 += aik * b3

					*(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr(idx*8))) = c0
					*(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((idx+1)*8))) = c1
					*(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((idx+2)*8))) = c2
					*(*float64)(unsafe.Pointer(uintptr(cPtr) + uintptr((idx+3)*8))) = c3

					j += SIMDVectorSize
				}
			}

			// Остаток
			for ; j < jEnd; j++ {
				c[iOffsetC+j] += aik * b[kOffset+j]
			}
		}
	}
}
