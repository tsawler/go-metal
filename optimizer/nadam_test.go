package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

func TestNadamOptimizer(t *testing.T) {
	// Initialize device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
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
	err = optimizer.UpdateLearningRate(newLR)
	if err != nil {
		t.Errorf("Failed to update learning rate: %v", err)
	}

	// Verify the update
	stats = optimizer.GetStats()
	if stats["learning_rate"] != newLR {
		t.Errorf("Expected updated learning rate to be %f, got %v", newLR, stats["learning_rate"])
	}
}

func TestNadamOptimizerWithCommandPool(t *testing.T) {
	// Skip command pool test for now since pooling is not yet implemented
	t.Skip("Command buffer pooling not yet implemented for Nadam")
}

func TestNadamOptimizerInvalidInputs(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
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

	// Test invalid learning rate update
	err = optimizer.UpdateLearningRate(0.0)
	if err == nil {
		t.Error("Expected error for learning rate = 0.0 in update, got nil")
	}

	err = optimizer.UpdateLearningRate(-0.001)
	if err == nil {
		t.Error("Expected error for negative learning rate in update, got nil")
	}
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