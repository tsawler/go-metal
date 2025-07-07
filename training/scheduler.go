package training

import (
	"math"
)

// LRScheduler defines the interface for learning rate scheduling strategies
// All schedulers must be stateless and pure functions to maintain GPU-resident principles
type LRScheduler interface {
	// GetLR returns the learning rate for the current epoch/step
	// This is a pure function - no state modifications
	GetLR(epoch int, step int, baseLR float64) float64
	
	// GetName returns the scheduler name for logging
	GetName() string
}

// StepLRScheduler reduces learning rate by a factor every stepSize epochs
type StepLRScheduler struct {
	StepSize int     // Epochs between LR reductions
	Gamma    float64 // Multiplicative factor of LR decay
}

// NewStepLRScheduler creates a step learning rate scheduler
func NewStepLRScheduler(stepSize int, gamma float64) *StepLRScheduler {
	if stepSize <= 0 {
		stepSize = 30 // Default: reduce every 30 epochs
	}
	if gamma <= 0 || gamma >= 1 {
		gamma = 0.1 // Default: reduce by 10x
	}
	return &StepLRScheduler{
		StepSize: stepSize,
		Gamma:    gamma,
	}
}

func (s *StepLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	// Calculate how many times to apply gamma
	times := epoch / s.StepSize
	return baseLR * math.Pow(s.Gamma, float64(times))
}

func (s *StepLRScheduler) GetName() string {
	return "StepLR"
}

// ExponentialLRScheduler decays learning rate exponentially
type ExponentialLRScheduler struct {
	Gamma float64 // Multiplicative factor of LR decay per epoch
}

// NewExponentialLRScheduler creates an exponential learning rate scheduler
func NewExponentialLRScheduler(gamma float64) *ExponentialLRScheduler {
	if gamma <= 0 || gamma >= 1 {
		gamma = 0.95 // Default: 5% reduction per epoch
	}
	return &ExponentialLRScheduler{
		Gamma: gamma,
	}
}

func (s *ExponentialLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	return baseLR * math.Pow(s.Gamma, float64(epoch))
}

func (s *ExponentialLRScheduler) GetName() string {
	return "ExponentialLR"
}

// CosineAnnealingLRScheduler implements cosine annealing schedule
type CosineAnnealingLRScheduler struct {
	TMax int     // Maximum number of epochs
	EtaMin float64 // Minimum learning rate
}

// NewCosineAnnealingLRScheduler creates a cosine annealing scheduler
func NewCosineAnnealingLRScheduler(tMax int, etaMin float64) *CosineAnnealingLRScheduler {
	if tMax <= 0 {
		tMax = 100 // Default: 100 epochs
	}
	if etaMin < 0 {
		etaMin = 0 // Default: anneal to 0
	}
	return &CosineAnnealingLRScheduler{
		TMax: tMax,
		EtaMin: etaMin,
	}
}

func (s *CosineAnnealingLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	if epoch >= s.TMax {
		return s.EtaMin
	}
	
	// Cosine annealing formula
	return s.EtaMin + (baseLR-s.EtaMin) * (1 + math.Cos(math.Pi*float64(epoch)/float64(s.TMax))) / 2
}

func (s *CosineAnnealingLRScheduler) GetName() string {
	return "CosineAnnealingLR"
}

// ReduceLROnPlateauScheduler reduces LR when a metric has stopped improving
// This scheduler requires state tracking, so it's handled differently
type ReduceLROnPlateauScheduler struct {
	Factor    float64 // Factor by which the learning rate will be reduced
	Patience  int     // Number of epochs with no improvement after which LR will be reduced
	Threshold float64 // Threshold for measuring the new optimum
	Mode      string  // One of "min" or "max"
	
	// Internal state - these are CPU-side only for scheduling decisions
	bestMetric   float64
	badEpochs    int
	currentLR    float64
	initialized  bool
}

// NewReduceLROnPlateauScheduler creates a plateau-based scheduler
func NewReduceLROnPlateauScheduler(factor float64, patience int, threshold float64, mode string) *ReduceLROnPlateauScheduler {
	if factor <= 0 || factor >= 1 {
		factor = 0.1
	}
	if patience <= 0 {
		patience = 10
	}
	if threshold < 0 {
		threshold = 1e-4
	}
	if mode != "min" && mode != "max" {
		mode = "min" // Default: minimize loss
	}
	
	return &ReduceLROnPlateauScheduler{
		Factor:    factor,
		Patience:  patience,
		Threshold: threshold,
		Mode:      mode,
	}
}

// Step checks if LR should be reduced based on metric
// This is called once per epoch with the validation metric
func (s *ReduceLROnPlateauScheduler) Step(metric float64, currentLR float64) float64 {
	if !s.initialized {
		s.bestMetric = metric
		s.currentLR = currentLR
		s.initialized = true
		return currentLR
	}
	
	improved := false
	if s.Mode == "min" {
		improved = metric < s.bestMetric - s.Threshold
	} else {
		improved = metric > s.bestMetric + s.Threshold
	}
	
	if improved {
		s.bestMetric = metric
		s.badEpochs = 0
	} else {
		s.badEpochs++
		if s.badEpochs >= s.Patience {
			s.currentLR *= s.Factor
			s.badEpochs = 0
		}
	}
	
	return s.currentLR
}

func (s *ReduceLROnPlateauScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	// For plateau scheduler, we return the internally tracked LR
	// The actual reduction happens in Step() based on metrics
	if s.initialized {
		return s.currentLR
	}
	return baseLR
}

func (s *ReduceLROnPlateauScheduler) GetName() string {
	return "ReduceLROnPlateau"
}

// NoOpScheduler maintains constant learning rate (default behavior)
type NoOpScheduler struct{}

func (s *NoOpScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	return baseLR
}

func (s *NoOpScheduler) GetName() string {
	return "ConstantLR"
}