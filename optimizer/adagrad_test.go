package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/memory"
)

func TestAdaGradOptimizer(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for AdaGrad optimizer test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test configuration
	config := AdaGradConfig{
		LearningRate: 0.01,
		Epsilon:      1e-10,
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
	optimizer, err := NewAdaGradOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create AdaGrad optimizer: %v", err)
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
			t.Fatalf("AdaGrad step %d failed: %v", step, err)
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
		t.Errorf("Expected learning_rate in stats to be %f, got %v", config.LearningRate, stats["learning_rate"])
	}

	// Test learning rate update
	newLR := float32(0.005)
	optimizer.UpdateLearningRate(newLR)

	stats = optimizer.GetStats()
	if stats["learning_rate"] != newLR {
		t.Errorf("Expected updated learning_rate to be %f, got %v", newLR, stats["learning_rate"])
	}

	// Test invalid learning rate (no longer validated in UpdateLearningRate)
	// optimizer.UpdateLearningRate(-0.001) // This would set it but not validate
}

func TestAdaGradOptimizerWithCommandPool(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for AdaGrad command pool test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultAdaGradConfig()
	weightShapes := [][]int{{10, 5}, {5}}

	optimizer, err := NewAdaGradOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create AdaGrad optimizer: %v", err)
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

func TestAdaGradOptimizerInvalidInputs(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for AdaGrad invalid inputs test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultAdaGradConfig()

	// Test nil memory manager
	_, err = NewAdaGradOptimizer(config, [][]int{{5, 5}}, nil, device)
	if err == nil {
		t.Error("Expected error for nil memory manager, got nil")
	}

	// Test nil device
	_, err = NewAdaGradOptimizer(config, [][]int{{5, 5}}, memoryManager, nil)
	if err == nil {
		t.Error("Expected error for nil device, got nil")
	}

	// Test empty weight shapes
	_, err = NewAdaGradOptimizer(config, [][]int{}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for empty weight shapes, got nil")
	}

	// Create valid optimizer for further tests
	optimizer, err := NewAdaGradOptimizer(config, [][]int{{5, 5}}, memoryManager, device)
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
}

// TestAdaGradConfig tests the AdaGrad configuration
func TestAdaGradConfig(t *testing.T) {
	config := DefaultAdaGradConfig()

	if config.LearningRate != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", config.LearningRate)
	}

	if config.Epsilon != 1e-10 {
		t.Errorf("Expected epsilon 1e-10, got %f", config.Epsilon)
	}

	if config.WeightDecay != 0.0 {
		t.Errorf("Expected weight decay 0.0, got %f", config.WeightDecay)
	}

	t.Log("AdaGrad config test passed")
}

// MockAdaGradOptimizer creates a mock AdaGrad optimizer for testing without Metal device
func MockAdaGradOptimizer() *AdaGradOptimizerState {
	config := DefaultAdaGradConfig()
	
	// Create mock optimizer state
	adagrad := &AdaGradOptimizerState{
		config:                config,
		squaredGradAvgBuffers: make([]unsafe.Pointer, 2),
		WeightBuffers:         make([]unsafe.Pointer, 2),
		currentStep:           0,
		memoryManager:         nil, // Mock
		device:                nil, // Mock
		bufferSizes:           []int{64, 8}, // FC weights: 8*2*4=64 bytes, FC bias: 2*4=8 bytes
	}

	return adagrad
}

// TestAdaGradMockOperations tests AdaGrad operations with mock data
func TestAdaGradMockOperations(t *testing.T) {
	adagrad := MockAdaGradOptimizer()

	// Test initial state
	if adagrad.GetStep() != 0 {
		t.Errorf("Expected initial step count 0, got %d", adagrad.GetStep())
	}

	// Test learning rate update
	newLR := float32(0.005)
	adagrad.UpdateLearningRate(newLR)

	if adagrad.config.LearningRate != newLR {
		t.Errorf("Expected learning rate %f, got %f", newLR, adagrad.config.LearningRate)
	}

	// Test stats
	stats := adagrad.GetStats()
	if stats["learning_rate"] != newLR {
		t.Errorf("Stats learning rate mismatch: expected %f, got %v", newLR, stats["learning_rate"])
	}

	if stats["step"] != uint64(0) {
		t.Errorf("Expected step in stats to be 0, got %v", stats["step"])
	}

	t.Log("AdaGrad mock operations test passed")
}

// TestAdaGradCheckpointing tests the checkpointing functionality
func TestAdaGradCheckpointing(t *testing.T) {
	adagrad := MockAdaGradOptimizer()

	// Set some state
	adagrad.currentStep = 10
	adagrad.config.LearningRate = 0.005

	// Test GetState with mock data (will fail with real Metal calls)
	// We need to mock the metal buffer calls for this test
	t.Skip("GetState/LoadState requires Metal buffer operations - test in integration tests")

	// Test state type validation
	invalidState := &OptimizerState{
		Type: "InvalidType",
		Parameters: map[string]interface{}{
			"learning_rate": 0.01,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err := adagrad.LoadState(invalidState)
	if err == nil {
		t.Error("Expected error for invalid state type, got nil")
	}

	// Test parameter restoration with valid state
	validState := &OptimizerState{
		Type: "AdaGrad",
		Parameters: map[string]interface{}{
			"learning_rate": 0.008,
			"epsilon":       1e-8,
			"weight_decay":  0.002,
			"step_count":    float64(5),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err = adagrad.LoadState(validState)
	if err != nil {
		t.Errorf("Expected no error for valid state, got %v", err)
	}

	// Verify parameters were restored
	if adagrad.config.LearningRate != 0.008 {
		t.Errorf("Expected learning rate 0.008, got %f", adagrad.config.LearningRate)
	}
	if adagrad.config.Epsilon != 1e-8 {
		t.Errorf("Expected epsilon 1e-8, got %f", adagrad.config.Epsilon)
	}
	if adagrad.config.WeightDecay != 0.002 {
		t.Errorf("Expected weight decay 0.002, got %f", adagrad.config.WeightDecay)
	}
	if adagrad.currentStep != 5 {
		t.Errorf("Expected step count 5, got %d", adagrad.currentStep)
	}

	t.Log("AdaGrad checkpointing test passed")
}

// TestAdaGradBasicStructures tests the basic AdaGrad data structures
func TestAdaGradBasicStructures(t *testing.T) {
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

	// Test configuration validation
	config := AdaGradConfig{
		LearningRate: 0.02,
		Epsilon:      1e-9,
		WeightDecay:  0.01,
	}

	if config.LearningRate != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", config.LearningRate)
	}

	if config.Epsilon != 1e-9 {
		t.Errorf("Expected epsilon 1e-9, got %f", config.Epsilon)
	}

	if config.WeightDecay != 0.01 {
		t.Errorf("Expected weight decay 0.01, got %f", config.WeightDecay)
	}

	t.Log("AdaGrad basic structures test passed")
}

// TestAdaGradGetStepCount tests the GetStepCount method
func TestAdaGradGetStepCount(t *testing.T) {
	adagrad := MockAdaGradOptimizer()
	
	// Test initial step count
	if adagrad.GetStepCount() != 0 {
		t.Errorf("Expected initial step count 0, got %d", adagrad.GetStepCount())
	}
	
	// Test after incrementing step count
	adagrad.currentStep = 25
	if adagrad.GetStepCount() != 25 {
		t.Errorf("Expected step count 25, got %d", adagrad.GetStepCount())
	}
	
	// Test that GetStep and GetStepCount return same value
	if adagrad.GetStep() != adagrad.GetStepCount() {
		t.Errorf("GetStep() and GetStepCount() should return same value: %d != %d", 
			adagrad.GetStep(), adagrad.GetStepCount())
	}
	
	t.Log("AdaGrad GetStepCount test passed")
}