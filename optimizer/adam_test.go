package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// TestAdamOptimizer tests the Adam optimizer implementation
func TestAdamOptimizer(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for Adam optimizer test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test configuration
	config := AdamConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.001,
	}

	// Test weight shapes
	weightShapes := [][]int{
		{10, 5},
		{5},
		{5, 3},
		{3},
	}

	// Create optimizer
	optimizer, err := NewAdamOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}
	defer optimizer.Cleanup()

	// Verify initial state
	if optimizer.GetStep() != 0 {
		t.Errorf("Expected initial step count to be 0, got %d", optimizer.GetStep())
	}

	// Create weight buffers
	weightBuffers := make([]unsafe.Pointer, len(weightShapes))
	for i, shape := range weightShapes {
		size := calculateTensorSize(shape) * 4 // 4 bytes per float32
		buffer := memoryManager.AllocateBuffer(size)
		if buffer == nil {
			t.Fatalf("Failed to allocate weight buffer %d", i)
		}
		defer memoryManager.ReleaseBuffer(buffer)
		weightBuffers[i] = buffer
	}

	// Set weight buffers
	err = optimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		t.Fatalf("Failed to set weight buffers: %v", err)
	}

	// Create gradient buffers
	gradientBuffers := make([]unsafe.Pointer, len(weightShapes))
	for i, shape := range weightShapes {
		size := calculateTensorSize(shape) * 4
		buffer := memoryManager.AllocateBuffer(size)
		if buffer == nil {
			t.Fatalf("Failed to allocate gradient buffer %d", i)
		}
		defer memoryManager.ReleaseBuffer(buffer)
		gradientBuffers[i] = buffer
	}

	// Test optimization steps
	for step := 1; step <= 3; step++ {
		err = optimizer.Step(gradientBuffers)
		if err != nil {
			t.Fatalf("Adam step %d failed: %v", step, err)
		}

		if optimizer.GetStep() != uint64(step) {
			t.Errorf("Expected step count to be %d, got %d", step, optimizer.GetStep())
		}
	}

	// Test stats
	stats := optimizer.GetStats()
	if stats.StepCount != 3 {
		t.Errorf("Expected step count in stats to be 3, got %d", stats.StepCount)
	}
	if stats.LearningRate != config.LearningRate {
		t.Errorf("Expected learning rate in stats to be %f, got %f", config.LearningRate, stats.LearningRate)
	}

	// Test learning rate update
	newLR := float32(0.0005)
	optimizer.UpdateLearningRate(newLR)
	stats = optimizer.GetStats()
	if stats.LearningRate != newLR {
		t.Errorf("Expected updated learning rate to be %f, got %f", newLR, stats.LearningRate)
	}

	// Test state save/restore
	state, err := optimizer.GetState()
	if err != nil {
		t.Fatalf("Failed to get optimizer state: %v", err)
	}
	if state.Type != "Adam" {
		t.Errorf("Expected state type 'Adam', got '%s'", state.Type)
	}

	// Test loading state
	err = optimizer.LoadState(state)
	if err != nil {
		t.Fatalf("Failed to load optimizer state: %v", err)
	}
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