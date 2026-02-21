package optimizers

import "math"

// StepLR - уменьшает темп обучения на фиксированный коэффициент через определенное количество эпох.
// Формула: lr = initial_lr * gamma ^ floor(epoch / step_size)
type StepLR struct {
	initialLR float64 // Начальный Learning Rate
	gamma     float64 // Коэффициент уменьшения
	stepSize  int     // Количество эпох между уменьшениями
	lastEpoch int     // Номер последней эпохи
	lastLR    float64 // Последний вычисленный Learning Rate
}

// NewStepLR создает новый экземпляр StepLR scheduler.
// initialLR - начальный Learning Rate
// gamma - коэффициент уменьшения (обычно 0.1)
// stepSize - количество эпох между уменьшениями
func NewStepLR(initialLR, gamma float64, stepSize int) *StepLR {
	return &StepLR{
		initialLR: initialLR,
		gamma:     gamma,
		stepSize:  stepSize,
		lastEpoch: 0,
		lastLR:    initialLR,
	}
}

// Step вызывается после каждой эпохи для обновления Learning Rate.
func (s *StepLR) Step() float64 {
	s.lastEpoch++
	// lr = initial_lr * gamma ^ floor(epoch / step_size)
	steps := s.lastEpoch / s.stepSize
	s.lastLR = s.initialLR * math.Pow(s.gamma, float64(steps))
	return s.lastLR
}

// GetLastLR возвращает последний вычисленный Learning Rate.
func (s *StepLR) GetLastLR() float64 {
	return s.lastLR
}

// ExponentialLR - уменьшает темп обучения экспоненциально с каждой эпохой.
// Формула: lr = initial_lr * gamma ^ epoch
type ExponentialLR struct {
	initialLR float64 // Начальный Learning Rate
	gamma     float64 // Коэффициент уменьшения (обычно 0.95-0.99)
	lastEpoch int     // Номер последней эпохи
	lastLR    float64 // Последний вычисленный Learning Rate
}

// NewExponentialLR создает новый экземпляр ExponentialLR scheduler.
// initialLR - начальный Learning Rate
// gamma - коэффициент уменьшения (обычно 0.95-0.99)
func NewExponentialLR(initialLR, gamma float64) *ExponentialLR {
	return &ExponentialLR{
		initialLR: initialLR,
		gamma:     gamma,
		lastEpoch: 0,
		lastLR:    initialLR,
	}
}

// Step вызывается после каждой эпохи для обновления Learning Rate.
func (e *ExponentialLR) Step() float64 {
	e.lastEpoch++
	// lr = initial_lr * gamma ^ epoch
	e.lastLR = e.initialLR * math.Pow(e.gamma, float64(e.lastEpoch))
	return e.lastLR
}

// GetLastLR возвращает последний вычисленный Learning Rate.
func (e *ExponentialLR) GetLastLR() float64 {
	return e.lastLR
}

// CosineAnnealingLR - изменяет темп обучения по косинусоидальной кривой.
// Формула: lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
type CosineAnnealingLR struct {
	initialLR float64 // Начальный Learning Rate
	etaMin    float64 // Минимальный Learning Rate
	TMax      int     // Максимальное количество эпох
	lastEpoch int     // Номер последней эпохи
	lastLR    float64 // Последний вычисленный Learning Rate
}

// NewCosineAnnealingLR создает новый экземпляр CosineAnnealingLR scheduler.
// initialLR - начальный Learning Rate
// TMax - максимальное количество эпох
// etaMin - минимальный Learning Rate (по умолчанию 0)
func NewCosineAnnealingLR(initialLR float64, TMax int, etaMin float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{
		initialLR: initialLR,
		etaMin:    etaMin,
		TMax:      TMax,
		lastEpoch: 0,
		lastLR:    initialLR,
	}
}

// Step вызывается после каждой эпохи для обновления Learning Rate.
func (c *CosineAnnealingLR) Step() float64 {
	c.lastEpoch++
	// lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
	cosine := math.Cos(math.Pi * float64(c.lastEpoch) / float64(c.TMax))
	c.lastLR = c.etaMin + (c.initialLR-c.etaMin)*(1+cosine)/2
	return c.lastLR
}

// GetLastLR возвращает последний вычисленный Learning Rate.
func (c *CosineAnnealingLR) GetLastLR() float64 {
	return c.lastLR
}

// OneCycleLR - стратегия, которая сначала линейно увеличивает темп обучения,
// а затем линейно уменьшает его, часто с очень низким темпом обучения в конце.
// Формула:
//   - Фаза увеличения (0 <= epoch < maxEpochs/2): lr = min_lr + (max_lr - min_lr) * (2 * epoch / maxEpochs)
//   - Фаза уменьшения (maxEpochs/2 <= epoch <= maxEpochs): lr = max_lr - (max_lr - final_lr) * (2 * (epoch - maxEpochs/2) / maxEpochs)
type OneCycleLR struct {
	minLR      float64 // Минимальный Learning Rate (начальный)
	maxLR      float64 // Максимальный Learning Rate
	finalLR    float64 // Финальный Learning Rate
	maxEpochs  int     // Максимальное количество эпох
	lastEpoch  int     // Номер последней эпохи
	lastLR     float64 // Последний вычисленный Learning Rate
}

// NewOneCycleLR создает новый экземпляр OneCycleLR scheduler.
// minLR - минимальный Learning Rate (начальный)
// maxLR - максимальный Learning Rate
// maxEpochs - максимальное количество эпох
// finalLR - финальный Learning Rate (по умолчанию minLR)
func NewOneCycleLR(minLR, maxLR float64, maxEpochs int, finalLR float64) *OneCycleLR {
	if finalLR == 0 {
		finalLR = minLR
	}
	return &OneCycleLR{
		minLR:     minLR,
		maxLR:     maxLR,
		finalLR:   finalLR,
		maxEpochs: maxEpochs,
		lastEpoch: 0,
		lastLR:    minLR,
	}
}

// Step вызывается после каждой эпохи для обновления Learning Rate.
func (o *OneCycleLR) Step() float64 {
	o.lastEpoch++
	
	if o.lastEpoch > o.maxEpochs {
		o.lastLR = o.finalLR
		return o.lastLR
	}

	midPoint := o.maxEpochs / 2
	
	if o.lastEpoch <= midPoint {
		// Фаза увеличения: линейно от minLR до maxLR
		progress := float64(o.lastEpoch) / float64(midPoint)
		o.lastLR = o.minLR + (o.maxLR-o.minLR)*progress
	} else {
		// Фаза уменьшения: линейно от maxLR до finalLR
		progress := float64(o.lastEpoch-midPoint) / float64(o.maxEpochs-midPoint)
		o.lastLR = o.maxLR - (o.maxLR-o.finalLR)*progress
	}
	
	return o.lastLR
}

// GetLastLR возвращает последний вычисленный Learning Rate.
func (o *OneCycleLR) GetLastLR() float64 {
	return o.lastLR
}

