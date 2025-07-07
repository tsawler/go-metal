package training

import (
	"math"
	"testing"
)

func TestStepLRScheduler(t *testing.T) {
	scheduler := NewStepLRScheduler(2, 0.1)
	baseLR := 0.1
	
	tests := []struct {
		epoch      int
		expectedLR float64
	}{
		{0, 0.1},      // Initial
		{1, 0.1},      // No change yet
		{2, 0.01},     // First reduction
		{3, 0.01},     // Same
		{4, 0.001},    // Second reduction
		{5, 0.001},    // Same
		{6, 0.0001},   // Third reduction
	}
	
	for _, tt := range tests {
		lr := scheduler.GetLR(tt.epoch, 0, baseLR)
		if math.Abs(lr-tt.expectedLR) > 1e-8 {
			t.Errorf("Epoch %d: expected LR %f, got %f", tt.epoch, tt.expectedLR, lr)
		}
	}
}

func TestExponentialLRScheduler(t *testing.T) {
	scheduler := NewExponentialLRScheduler(0.9)
	baseLR := 0.1
	
	tests := []struct {
		epoch      int
		expectedLR float64
	}{
		{0, 0.1},        // Initial
		{1, 0.09},       // 0.1 * 0.9
		{2, 0.081},      // 0.1 * 0.9^2
		{3, 0.0729},     // 0.1 * 0.9^3
		{4, 0.06561},    // 0.1 * 0.9^4
		{5, 0.059049},   // 0.1 * 0.9^5
	}
	
	for _, tt := range tests {
		lr := scheduler.GetLR(tt.epoch, 0, baseLR)
		if math.Abs(lr-tt.expectedLR) > 1e-8 {
			t.Errorf("Epoch %d: expected LR %f, got %f", tt.epoch, tt.expectedLR, lr)
		}
	}
}

func TestCosineAnnealingLRScheduler(t *testing.T) {
	scheduler := NewCosineAnnealingLRScheduler(5, 0.0001)
	baseLR := 0.01
	
	// Test specific points in the cosine curve
	tests := []struct {
		epoch      int
		expectedLR float64
		tolerance  float64
	}{
		{0, 0.01, 1e-6},      // Initial (max)
		{5, 0.0001, 1e-6},    // Final (min)
		{2, 0.006580, 1e-6}, // Midpoint calculation
	}
	
	for _, tt := range tests {
		lr := scheduler.GetLR(tt.epoch, 0, baseLR)
		if math.Abs(lr-tt.expectedLR) > tt.tolerance {
			t.Errorf("Epoch %d: expected LR %f, got %f", tt.epoch, tt.expectedLR, lr)
		}
	}
	
	// Test beyond TMax
	lr := scheduler.GetLR(10, 0, baseLR)
	if lr != 0.0001 {
		t.Errorf("Beyond TMax: expected LR %f, got %f", 0.0001, lr)
	}
}

func TestReduceLROnPlateauScheduler(t *testing.T) {
	scheduler := NewReduceLROnPlateauScheduler(0.5, 2, 0.01, "min")
	
	// Test basic functionality
	currentLR := scheduler.Step(1.0, 0.1) // Initial
	if currentLR != 0.1 {
		t.Errorf("Initial: expected LR %f, got %f", 0.1, currentLR)
	}
	
	currentLR = scheduler.Step(0.98, currentLR) // Improvement
	if currentLR != 0.1 {
		t.Errorf("After improvement: expected LR %f, got %f", 0.1, currentLR)
	}
	
	currentLR = scheduler.Step(0.99, currentLR) // No improvement
	if currentLR != 0.1 {
		t.Errorf("No improvement 1: expected LR %f, got %f", 0.1, currentLR)
	}
	
	currentLR = scheduler.Step(0.99, currentLR) // No improvement - should reduce
	if currentLR != 0.05 {
		t.Errorf("No improvement 2: expected LR %f, got %f", 0.05, currentLR)
	}
}

func TestSchedulerNames(t *testing.T) {
	tests := []struct {
		scheduler LRScheduler
		expected  string
	}{
		{NewStepLRScheduler(10, 0.1), "StepLR"},
		{NewExponentialLRScheduler(0.95), "ExponentialLR"},
		{NewCosineAnnealingLRScheduler(100, 0.0), "CosineAnnealingLR"},
		{NewReduceLROnPlateauScheduler(0.1, 10, 0.001, "min"), "ReduceLROnPlateau"},
		{&NoOpScheduler{}, "ConstantLR"},
	}
	
	for _, tt := range tests {
		name := tt.scheduler.GetName()
		if name != tt.expected {
			t.Errorf("Expected name %s, got %s", tt.expected, name)
		}
	}
}