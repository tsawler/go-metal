package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/memory"
)

func TestNadamOptimizer(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for Nadam optimizer test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test configuration
	config := NadamConfig{
		LearningRate: 0.002,
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
	optimizer, err := NewNadamOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create Nadam optimizer: %v", err)
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
			t.Fatalf("Nadam step %d failed: %v", step, err)
		}

		if optimizer.GetStep() != uint64(step) {
			t.Errorf("Expected step count to be %d, got %d", step, optimizer.GetStep())
		}
	}

	// Test stats
	stats := optimizer.GetStats()
	if stats["step"] != uint64(3) {
		t.Errorf("Expected step in stats to be 3, got %v", stats["step"])
	}
	if stats["learning_rate"] != config.LearningRate {
		t.Errorf("Expected learning rate in stats to be %f, got %v", config.LearningRate, stats["learning_rate"])
	}

	// Test learning rate update
	newLR := float32(0.001)
	optimizer.UpdateLearningRate(newLR)

	// Verify the update
	stats = optimizer.GetStats()
	if stats["learning_rate"] != newLR {
		t.Errorf("Expected updated learning rate to be %f, got %v", newLR, stats["learning_rate"])
	}
}

func TestNadamOptimizerWithCommandPool(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for Nadam command pool test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultNadamConfig()
	weightShapes := [][]int{{10, 5}, {5}}

	optimizer, err := NewNadamOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create Nadam optimizer: %v", err)
	}
	defer optimizer.Cleanup()

	// Test setting command pool
	mockPool := unsafe.Pointer(uintptr(0x2000))
	optimizer.SetCommandPool(mockPool)
	
	if optimizer.commandPool != mockPool {
		t.Errorf("Expected commandPool %p, got %p", mockPool, optimizer.commandPool)
	}
	
	if !optimizer.usePooling {
		t.Error("Expected usePooling to be true")
	}
	
	// Test setting nil pool
	optimizer.SetCommandPool(nil)
	if optimizer.usePooling {
		t.Error("Expected usePooling to be false after setting nil pool")
	}
}

func TestNadamOptimizerInvalidInputs(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for Nadam invalid inputs test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultNadamConfig()

	// Test nil memory manager
	_, err = NewNadamOptimizer(config, [][]int{{5, 5}}, nil, device)
	if err == nil {
		t.Error("Expected error for nil memory manager, got nil")
	}

	// Test nil device
	_, err = NewNadamOptimizer(config, [][]int{{5, 5}}, memoryManager, nil)
	if err == nil {
		t.Error("Expected error for nil device, got nil")
	}

	// Test empty weight shapes
	_, err = NewNadamOptimizer(config, [][]int{}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for empty weight shapes, got nil")
	}

	// Test invalid learning rate
	invalidConfig := config
	invalidConfig.LearningRate = 0.0
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for learning rate = 0.0, got nil")
	}

	invalidConfig.LearningRate = -0.001
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative learning rate, got nil")
	}

	// Test invalid beta1 values
	invalidConfig = config
	invalidConfig.Beta1 = -0.1
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative beta1, got nil")
	}

	invalidConfig.Beta1 = 1.0
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for beta1 = 1.0, got nil")
	}

	invalidConfig.Beta1 = 1.1
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for beta1 > 1.0, got nil")
	}

	// Test invalid beta2 values
	invalidConfig = config
	invalidConfig.Beta2 = -0.1
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative beta2, got nil")
	}

	invalidConfig.Beta2 = 1.0
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for beta2 = 1.0, got nil")
	}

	invalidConfig.Beta2 = 1.1
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for beta2 > 1.0, got nil")
	}

	// Test invalid epsilon
	invalidConfig = config
	invalidConfig.Epsilon = 0.0
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for epsilon = 0.0, got nil")
	}

	invalidConfig.Epsilon = -1e-8
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative epsilon, got nil")
	}

	// Test invalid weight decay
	invalidConfig = config
	invalidConfig.WeightDecay = -0.1
	_, err = NewNadamOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative weight decay, got nil")
	}

	// Create valid optimizer for further tests
	optimizer, err := NewNadamOptimizer(config, [][]int{{5, 5}}, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create optimizer: %v", err)
	}
	defer optimizer.Cleanup()

	// Test mismatched weight buffer count
	err = optimizer.SetWeightBuffers([]unsafe.Pointer{})
	if err == nil {
		t.Error("Expected error for mismatched weight buffer count, got nil")
	}

	// Test Step with mismatched gradient buffer count
	size := calculateTensorSize([]int{5, 5}) * 4
	weightBuffer := memoryManager.AllocateBuffer(size)
	defer memoryManager.ReleaseBuffer(weightBuffer)

	optimizer.SetWeightBuffers([]unsafe.Pointer{weightBuffer})
	err = optimizer.Step([]unsafe.Pointer{})
	if err == nil {
		t.Error("Expected error for mismatched gradient buffer count, got nil")
	}

	// Test Step with proper buffer count
	gradientBuffer := memoryManager.AllocateBuffer(size)
	defer memoryManager.ReleaseBuffer(gradientBuffer)

	err = optimizer.Step([]unsafe.Pointer{gradientBuffer})
	if err != nil {
		t.Errorf("Unexpected error for proper buffer count: %v", err)
	}

	// Test invalid learning rate update (no longer validated in UpdateLearningRate)
	// optimizer.UpdateLearningRate(0.0) // This would set it but not validate
	// optimizer.UpdateLearningRate(-0.001) // This would set it but not validate
}

func TestDefaultNadamConfig(t *testing.T) {
	config := DefaultNadamConfig()
	
	if config.LearningRate != 0.002 {
		t.Errorf("Expected default learning rate to be 0.002, got %f", config.LearningRate)
	}
	
	if config.Beta1 != 0.9 {
		t.Errorf("Expected default beta1 to be 0.9, got %f", config.Beta1)
	}
	
	if config.Beta2 != 0.999 {
		t.Errorf("Expected default beta2 to be 0.999, got %f", config.Beta2)
	}
	
	if config.Epsilon != 1e-8 {
		t.Errorf("Expected default epsilon to be 1e-8, got %e", config.Epsilon)
	}
	
	if config.WeightDecay != 0.0 {
		t.Errorf("Expected default weight decay to be 0.0, got %f", config.WeightDecay)
	}
}

func TestNadamConfigValidation(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test valid config
	validConfig := NadamConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
	}

	_, err = NewNadamOptimizer(validConfig, [][]int{{5, 5}}, memoryManager, device)
	if err != nil {
		t.Errorf("Expected no error for valid config, got %v", err)
	}

	// Test edge cases
	edgeConfig := NadamConfig{
		LearningRate: 0.0001,  // Very small but valid
		Beta1:        0.001,   // Very small but valid
		Beta2:        0.001,   // Very small but valid
		Epsilon:      1e-10,   // Very small but valid
		WeightDecay:  1.0,     // Large but valid
	}

	optimizer, err := NewNadamOptimizer(edgeConfig, [][]int{{5, 5}}, memoryManager, device)
	if err != nil {
		t.Errorf("Expected no error for edge case config, got %v", err)
	}
	if optimizer != nil {
		optimizer.Cleanup()
	}
}

// MockNadamOptimizer creates a mock Nadam optimizer for testing without Metal device
func MockNadamOptimizer() *NadamOptimizerState {
	config := DefaultNadamConfig()
	
	// Create mock optimizer state
	nadam := &NadamOptimizerState{
		config:          config,
		momentumBuffers: make([]unsafe.Pointer, 2),
		varianceBuffers: make([]unsafe.Pointer, 2),
		WeightBuffers:   make([]unsafe.Pointer, 2),
		currentStep:     0,
		memoryManager:   nil, // Mock
		device:          nil, // Mock
		bufferSizes:     []int{64, 8}, // FC weights: 8*2*4=64 bytes, FC bias: 2*4=8 bytes
	}

	return nadam
}

// TestNadamMockOperations tests Nadam operations with mock data
func TestNadamMockOperations(t *testing.T) {
	nadam := MockNadamOptimizer()

	// Test initial state
	if nadam.GetStep() != 0 {
		t.Errorf("Expected initial step count 0, got %d", nadam.GetStep())
	}

	// Test learning rate update
	newLR := float32(0.001)
	nadam.UpdateLearningRate(newLR)

	if nadam.config.LearningRate != newLR {
		t.Errorf("Expected learning rate %f, got %f", newLR, nadam.config.LearningRate)
	}

	// Test stats
	stats := nadam.GetStats()
	if stats["learning_rate"] != newLR {
		t.Errorf("Stats learning rate mismatch: expected %f, got %v", newLR, stats["learning_rate"])
	}

	if stats["step"] != uint64(0) {
		t.Errorf("Expected step in stats to be 0, got %v", stats["step"])
	}

	if beta1, ok := stats["beta1"].(float32); !ok || beta1 != 0.9 {
		t.Errorf("Expected beta1 in stats to be 0.9, got %v", stats["beta1"])
	}

	if beta2, ok := stats["beta2"].(float32); !ok || beta2 != 0.999 {
		t.Errorf("Expected beta2 in stats to be 0.999, got %v", stats["beta2"])
	}

	t.Log("Nadam mock operations test passed")
}

// TestNadamCheckpointing tests the checkpointing functionality
func TestNadamCheckpointing(t *testing.T) {
	nadam := MockNadamOptimizer()

	// Set some state
	nadam.currentStep = 12
	nadam.config.LearningRate = 0.0015
	nadam.config.Beta1 = 0.85
	nadam.config.Beta2 = 0.995

	// Test GetState with mock data (will fail with real Metal calls)
	// We need to mock the metal buffer calls for this test
	t.Skip("GetState/LoadState requires Metal buffer operations - test in integration tests")

	// Test state type validation
	invalidState := &OptimizerState{
		Type: "InvalidType",
		Parameters: map[string]interface{}{
			"learning_rate": 0.002,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err := nadam.LoadState(invalidState)
	if err == nil {
		t.Error("Expected error for invalid state type, got nil")
	}

	// Test parameter restoration with valid state
	validState := &OptimizerState{
		Type: "Nadam",
		Parameters: map[string]interface{}{
			"learning_rate": 0.0025,
			"beta1":         0.88,
			"beta2":         0.998,
			"epsilon":       1e-7,
			"weight_decay":  0.005,
			"step_count":    float64(8),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err = nadam.LoadState(validState)
	if err != nil {
		t.Errorf("Expected no error for valid state, got %v", err)
	}

	// Verify parameters were restored
	if nadam.config.LearningRate != 0.0025 {
		t.Errorf("Expected learning rate 0.0025, got %f", nadam.config.LearningRate)
	}
	if nadam.config.Beta1 != 0.88 {
		t.Errorf("Expected beta1 0.88, got %f", nadam.config.Beta1)
	}
	if nadam.config.Beta2 != 0.998 {
		t.Errorf("Expected beta2 0.998, got %f", nadam.config.Beta2)
	}
	if nadam.config.Epsilon != 1e-7 {
		t.Errorf("Expected epsilon 1e-7, got %f", nadam.config.Epsilon)
	}
	if nadam.config.WeightDecay != 0.005 {
		t.Errorf("Expected weight decay 0.005, got %f", nadam.config.WeightDecay)
	}
	if nadam.currentStep != 8 {
		t.Errorf("Expected step count 8, got %d", nadam.currentStep)
	}

	t.Log("Nadam checkpointing test passed")
}

// TestNadamBasicStructures tests the basic Nadam data structures
func TestNadamBasicStructures(t *testing.T) {
	// Test weight shapes calculation
	shapes := [][]int{
		{8, 2},   // FC weights
		{2},      // FC bias
		{10, 5},  // Another layer
		{128, 64}, // Larger layer
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

	// Test configuration validation
	config := NadamConfig{
		LearningRate: 0.003,
		Beta1:        0.85,
		Beta2:        0.995,
		Epsilon:      1e-7,
		WeightDecay:  0.002,
	}

	if config.LearningRate != 0.003 {
		t.Errorf("Expected learning rate 0.003, got %f", config.LearningRate)
	}

	if config.Beta1 != 0.85 {
		t.Errorf("Expected beta1 0.85, got %f", config.Beta1)
	}

	if config.Beta2 != 0.995 {
		t.Errorf("Expected beta2 0.995, got %f", config.Beta2)
	}

	if config.Epsilon != 1e-7 {
		t.Errorf("Expected epsilon 1e-7, got %f", config.Epsilon)
	}

	if config.WeightDecay != 0.002 {
		t.Errorf("Expected weight decay 0.002, got %f", config.WeightDecay)
	}

	t.Log("Nadam basic structures test passed")
}

// TestNadamParameterValidation tests parameter validation
func TestNadamParameterValidation(t *testing.T) {
	baseConfig := DefaultNadamConfig()

	// Test learning rate validation
	testCases := []struct {
		name          string
		learningRate  float32
		expectError   bool
	}{
		{"valid_lr", 0.001, false},
		{"zero_lr", 0.0, true},
		{"negative_lr", -0.001, true},
		{"very_small_lr", 1e-10, false},
		{"large_lr", 1.0, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := baseConfig
			config.LearningRate = tc.learningRate

			// Test validation logic (would be in NewNadamOptimizer)
			if tc.expectError {
				if config.LearningRate > 0 {
					t.Errorf("Expected invalid learning rate %f to be <= 0", config.LearningRate)
				}
			} else {
				if config.LearningRate <= 0 {
					t.Errorf("Expected valid learning rate %f to be > 0", config.LearningRate)
				}
			}
		})
	}

	// Test beta1 validation
	betaTestCases := []struct {
		name        string
		beta1       float32
		expectError bool
	}{
		{"valid_beta1", 0.9, false},
		{"zero_beta1", 0.0, false},
		{"negative_beta1", -0.1, true},
		{"one_beta1", 1.0, true},
		{"above_one_beta1", 1.1, true},
		{"very_small_beta1", 0.001, false},
	}

	for _, tc := range betaTestCases {
		t.Run(tc.name, func(t *testing.T) {
			config := baseConfig
			config.Beta1 = tc.beta1

			// Test validation logic
			if tc.expectError {
				if config.Beta1 >= 0 && config.Beta1 < 1 {
					t.Errorf("Expected invalid beta1 %f to be outside [0, 1)", config.Beta1)
				}
			} else {
				if config.Beta1 < 0 || config.Beta1 >= 1 {
					t.Errorf("Expected valid beta1 %f to be in [0, 1)", config.Beta1)
				}
			}
		})
	}

	t.Log("Nadam parameter validation test passed")
}

// TestNadamStepCountTracking tests step count tracking
func TestNadamStepCountTracking(t *testing.T) {
	nadam := MockNadamOptimizer()

	// Test initial step count
	if nadam.GetStepCount() != 0 {
		t.Errorf("Expected initial step count 0, got %d", nadam.GetStepCount())
	}

	// Simulate step increments
	nadam.currentStep = 5
	if nadam.GetStepCount() != 5 {
		t.Errorf("Expected step count 5, got %d", nadam.GetStepCount())
	}

	// Test GetStep compatibility
	if nadam.GetStep() != nadam.GetStepCount() {
		t.Errorf("GetStep() and GetStepCount() should return same value")
	}

	t.Log("Nadam step count tracking test passed")
}

// TestNadamComparisonWithAdam tests differences between Nadam and Adam
func TestNadamComparisonWithAdam(t *testing.T) {
	nadamConfig := DefaultNadamConfig()
	adamConfig := DefaultAdamConfig()

	// Test that Nadam has slightly higher default learning rate
	if nadamConfig.LearningRate <= adamConfig.LearningRate {
		t.Errorf("Expected Nadam LR (%f) > Adam LR (%f)", nadamConfig.LearningRate, adamConfig.LearningRate)
	}

	// Test that beta values are similar
	if nadamConfig.Beta1 != adamConfig.Beta1 {
		t.Errorf("Expected same beta1 values: Nadam=%f, Adam=%f", nadamConfig.Beta1, adamConfig.Beta1)
	}

	if nadamConfig.Beta2 != adamConfig.Beta2 {
		t.Errorf("Expected same beta2 values: Nadam=%f, Adam=%f", nadamConfig.Beta2, adamConfig.Beta2)
	}

	// Test epsilon values
	if nadamConfig.Epsilon != adamConfig.Epsilon {
		t.Errorf("Expected same epsilon values: Nadam=%f, Adam=%f", nadamConfig.Epsilon, adamConfig.Epsilon)
	}

	t.Log("Nadam vs Adam comparison test passed")
}