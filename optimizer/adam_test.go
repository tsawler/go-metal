package optimizer

import (
	"testing"
	"unsafe"
)

// TestAdamOptimizer tests the Adam optimizer implementation
func TestAdamOptimizer(t *testing.T) {
	// Skip this test for now as it requires Metal device integration
	t.Skip("Skipping Adam optimizer test - requires Metal CGO integration")
}

// TestAdamConfig tests the Adam configuration
func TestAdamConfig(t *testing.T) {
	config := DefaultAdamConfig()

	if config.LearningRate != 0.001 {
		t.Errorf("Expected learning rate 0.001, got %f", config.LearningRate)
	}

	if config.Beta1 != 0.9 {
		t.Errorf("Expected beta1 0.9, got %f", config.Beta1)
	}

	if config.Beta2 != 0.999 {
		t.Errorf("Expected beta2 0.999, got %f", config.Beta2)
	}

	if config.Epsilon != 1e-8 {
		t.Errorf("Expected epsilon 1e-8, got %f", config.Epsilon)
	}

	if config.WeightDecay != 0.0 {
		t.Errorf("Expected weight decay 0.0, got %f", config.WeightDecay)
	}

	t.Log("Adam config test passed")
}

// TestAdamBasicStructures tests the basic Adam data structures
func TestAdamBasicStructures(t *testing.T) {
	// Test weight shapes calculation
	shapes := [][]int{
		{8, 2},  // FC weights
		{2},     // FC bias
	}

	for i, shape := range shapes {
		size := calculateTensorSize(shape)
		expectedSize := 1
		for _, dim := range shape {
			expectedSize *= dim
		}

		if size != expectedSize {
			t.Errorf("Shape %d: expected size %d, got %d", i, expectedSize, size)
		}
	}

	// Test Adam stats structure
	stats := AdamStats{
		StepCount:       100,
		LearningRate:    0.001,
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.01,
		NumParameters:   2,
		TotalBufferSize: 1024,
	}

	if stats.StepCount != 100 {
		t.Errorf("Expected step count 100, got %d", stats.StepCount)
	}

	if stats.NumParameters != 2 {
		t.Errorf("Expected 2 parameters, got %d", stats.NumParameters)
	}

	t.Log("Adam basic structures test passed")
}

// MockAdamOptimizer creates a mock Adam optimizer for testing without Metal device
func MockAdamOptimizer() *AdamOptimizerState {
	config := DefaultAdamConfig()
	
	// Create mock optimizer state
	adam := &AdamOptimizerState{
		LearningRate:    config.LearningRate,
		Beta1:           config.Beta1,
		Beta2:           config.Beta2,
		Epsilon:         config.Epsilon,
		WeightDecay:     config.WeightDecay,
		MomentumBuffers: make([]unsafe.Pointer, 2),
		VarianceBuffers: make([]unsafe.Pointer, 2),
		WeightBuffers:   make([]unsafe.Pointer, 2),
		StepCount:       0,
		memoryManager:   nil, // Mock
		device:          nil, // Mock
		bufferSizes:     []int{32, 8}, // FC weights: 8*2*4=64 bytes, FC bias: 2*4=8 bytes
	}

	return adam
}

// TestAdamMockOperations tests Adam operations with mock data
func TestAdamMockOperations(t *testing.T) {
	adam := MockAdamOptimizer()

	// Test initial state
	if adam.GetStep() != 0 {
		t.Errorf("Expected initial step count 0, got %d", adam.GetStep())
	}

	// Test learning rate update
	newLR := float32(0.0005)
	adam.UpdateLearningRate(newLR)

	if adam.LearningRate != newLR {
		t.Errorf("Expected learning rate %f, got %f", newLR, adam.LearningRate)
	}

	// Test stats
	stats := adam.GetStats()
	if stats.LearningRate != newLR {
		t.Errorf("Stats learning rate mismatch: expected %f, got %f", newLR, stats.LearningRate)
	}

	if stats.NumParameters != 2 {
		t.Errorf("Expected 2 parameters in stats, got %d", stats.NumParameters)
	}

	expectedBufferSize := (32 + 8) * 2 // (momentum + variance) for each weight
	if stats.TotalBufferSize != expectedBufferSize {
		t.Errorf("Expected total buffer size %d, got %d", expectedBufferSize, stats.TotalBufferSize)
	}

	t.Log("Adam mock operations test passed")
}