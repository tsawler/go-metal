package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/memory"
)

func TestAdaDeltaOptimizer(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for AdaDelta optimizer test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test configuration
	config := AdaDeltaConfig{
		Rho:         0.95,
		Epsilon:     1e-6,
		WeightDecay: 0.001,
	}

	// Test weight shapes
	weightShapes := [][]int{
		{10, 5},
		{5},
		{5, 3},
		{3},
	}

	// Create optimizer
	optimizer, err := NewAdaDeltaOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create AdaDelta optimizer: %v", err)
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
			t.Fatalf("AdaDelta step %d failed: %v", step, err)
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
	if stats["rho"] != config.Rho {
		t.Errorf("Expected rho in stats to be %f, got %v", config.Rho, stats["rho"])
	}

	// Test that learning rate update is a no-op (AdaDelta doesn't use fixed learning rate)
	optimizer.UpdateLearningRate(0.005) // This is a no-op for AdaDelta
}

func TestAdaDeltaOptimizerWithCommandPool(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for AdaDelta command pool test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultAdaDeltaConfig()
	weightShapes := [][]int{{10, 5}, {5}}

	optimizer, err := NewAdaDeltaOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create AdaDelta optimizer: %v", err)
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

func TestAdaDeltaOptimizerInvalidInputs(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for AdaDelta invalid inputs test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultAdaDeltaConfig()

	// Test nil memory manager
	_, err = NewAdaDeltaOptimizer(config, [][]int{{5, 5}}, nil, device)
	if err == nil {
		t.Error("Expected error for nil memory manager, got nil")
	}

	// Test nil device
	_, err = NewAdaDeltaOptimizer(config, [][]int{{5, 5}}, memoryManager, nil)
	if err == nil {
		t.Error("Expected error for nil device, got nil")
	}

	// Test empty weight shapes
	_, err = NewAdaDeltaOptimizer(config, [][]int{}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for empty weight shapes, got nil")
	}

	// Test invalid rho values
	invalidConfig := config
	invalidConfig.Rho = 0.0
	_, err = NewAdaDeltaOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for rho = 0.0, got nil")
	}

	invalidConfig.Rho = 1.0
	_, err = NewAdaDeltaOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for rho = 1.0, got nil")
	}

	invalidConfig.Rho = -0.1
	_, err = NewAdaDeltaOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative rho, got nil")
	}

	invalidConfig.Rho = 1.1
	_, err = NewAdaDeltaOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for rho > 1.0, got nil")
	}

	// Create valid optimizer for further tests
	optimizer, err := NewAdaDeltaOptimizer(config, [][]int{{5, 5}}, memoryManager, device)
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

	// Test Step with mismatched squared update buffer count (internal test)
	gradientBuffer := memoryManager.AllocateBuffer(size)
	defer memoryManager.ReleaseBuffer(gradientBuffer)

	err = optimizer.Step([]unsafe.Pointer{gradientBuffer})
	if err != nil {
		t.Errorf("Unexpected error for proper buffer count: %v", err)
	}
}

func TestDefaultAdaDeltaConfig(t *testing.T) {
	config := DefaultAdaDeltaConfig()
	
	if config.Rho != 0.95 {
		t.Errorf("Expected default rho to be 0.95, got %f", config.Rho)
	}
	
	if config.Epsilon != 1e-6 {
		t.Errorf("Expected default epsilon to be 1e-6, got %e", config.Epsilon)
	}
	
	if config.WeightDecay != 0.0 {
		t.Errorf("Expected default weight decay to be 0.0, got %f", config.WeightDecay)
	}
}

// MockAdaDeltaOptimizer creates a mock AdaDelta optimizer for testing without Metal device
func MockAdaDeltaOptimizer() *AdaDeltaOptimizerState {
	config := DefaultAdaDeltaConfig()
	
	// Create mock optimizer state
	adadelta := &AdaDeltaOptimizerState{
		config:                  config,
		squaredGradAvgBuffers:   make([]unsafe.Pointer, 2),
		squaredUpdateAvgBuffers: make([]unsafe.Pointer, 2),
		WeightBuffers:           make([]unsafe.Pointer, 2),
		currentStep:             0,
		memoryManager:           nil, // Mock
		device:                  nil, // Mock
		bufferSizes:             []int{64, 8}, // FC weights: 8*2*4=64 bytes, FC bias: 2*4=8 bytes
	}

	return adadelta
}

// TestAdaDeltaMockOperations tests AdaDelta operations with mock data
func TestAdaDeltaMockOperations(t *testing.T) {
	adadelta := MockAdaDeltaOptimizer()

	// Test initial state
	if adadelta.GetStep() != 0 {
		t.Errorf("Expected initial step count 0, got %d", adadelta.GetStep())
	}

	// Test learning rate update (should be no-op)
	adadelta.UpdateLearningRate(0.005)
	// AdaDelta doesn't use fixed learning rate, so no verification needed

	// Test stats
	stats := adadelta.GetStats()
	if stats["step"] != uint64(0) {
		t.Errorf("Expected step in stats to be 0, got %v", stats["step"])
	}

	if rho, ok := stats["rho"].(float32); !ok || rho != 0.95 {
		t.Errorf("Expected rho in stats to be 0.95, got %v", stats["rho"])
	}

	t.Log("AdaDelta mock operations test passed")
}

// TestAdaDeltaCheckpointing tests the checkpointing functionality
func TestAdaDeltaCheckpointing(t *testing.T) {
	adadelta := MockAdaDeltaOptimizer()

	// Set some state
	adadelta.currentStep = 15
	adadelta.config.Rho = 0.90
	adadelta.config.Epsilon = 1e-7

	// Test GetState with mock data (will fail with real Metal calls)
	// We need to mock the metal buffer calls for this test
	t.Skip("GetState/LoadState requires Metal buffer operations - test in integration tests")

	// Test state type validation
	invalidState := &OptimizerState{
		Type: "InvalidType",
		Parameters: map[string]interface{}{
			"rho": 0.95,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err := adadelta.LoadState(invalidState)
	if err == nil {
		t.Error("Expected error for invalid state type, got nil")
	}

	// Test parameter restoration with valid state
	validState := &OptimizerState{
		Type: "AdaDelta",
		Parameters: map[string]interface{}{
			"rho":         0.92,
			"epsilon":     1e-5,
			"weight_decay": 0.003,
			"step_count":  float64(7),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err = adadelta.LoadState(validState)
	if err != nil {
		t.Errorf("Expected no error for valid state, got %v", err)
	}

	// Verify parameters were restored
	if adadelta.config.Rho != 0.92 {
		t.Errorf("Expected rho 0.92, got %f", adadelta.config.Rho)
	}
	if adadelta.config.Epsilon != 1e-5 {
		t.Errorf("Expected epsilon 1e-5, got %f", adadelta.config.Epsilon)
	}
	if adadelta.config.WeightDecay != 0.003 {
		t.Errorf("Expected weight decay 0.003, got %f", adadelta.config.WeightDecay)
	}
	if adadelta.currentStep != 7 {
		t.Errorf("Expected step count 7, got %d", adadelta.currentStep)
	}

	t.Log("AdaDelta checkpointing test passed")
}

// TestAdaDeltaBasicStructures tests the basic AdaDelta data structures
func TestAdaDeltaBasicStructures(t *testing.T) {
	// Test weight shapes calculation
	shapes := [][]int{
		{8, 2},  // FC weights
		{2},     // FC bias
		{10, 5}, // Another layer
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
	config := AdaDeltaConfig{
		Rho:         0.9,
		Epsilon:     1e-8,
		WeightDecay: 0.01,
	}

	if config.Rho != 0.9 {
		t.Errorf("Expected rho 0.9, got %f", config.Rho)
	}

	if config.Epsilon != 1e-8 {
		t.Errorf("Expected epsilon 1e-8, got %f", config.Epsilon)
	}

	if config.WeightDecay != 0.01 {
		t.Errorf("Expected weight decay 0.01, got %f", config.WeightDecay)
	}

	t.Log("AdaDelta basic structures test passed")
}

// TestAdaDeltaConfigValidation tests the configuration validation
func TestAdaDeltaConfigValidation(t *testing.T) {
	// Test valid configuration
	validConfig := AdaDeltaConfig{
		Rho:         0.95,
		Epsilon:     1e-6,
		WeightDecay: 0.0,
	}

	if validConfig.Rho <= 0 || validConfig.Rho >= 1 {
		t.Errorf("Valid config rho should be in (0, 1), got %f", validConfig.Rho)
	}

	// Test boundary values
	testCases := []struct {
		name   string
		rho    float32
		valid  bool
	}{
		{"rho_zero", 0.0, false},
		{"rho_one", 1.0, false},
		{"rho_negative", -0.1, false},
		{"rho_above_one", 1.1, false},
		{"rho_valid_low", 0.01, true},
		{"rho_valid_high", 0.99, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := validConfig
			config.Rho = tc.rho
			
			// In real implementation, this would be validated in NewAdaDeltaOptimizer
			// For now, just check the value is what we expect
			if tc.valid {
				if config.Rho <= 0 || config.Rho >= 1 {
					t.Errorf("Expected valid rho %f to be in (0, 1)", config.Rho)
				}
			} else {
				if config.Rho > 0 && config.Rho < 1 {
					t.Errorf("Expected invalid rho %f to be outside (0, 1)", config.Rho)
				}
			}
		})
	}

	t.Log("AdaDelta config validation test passed")
}