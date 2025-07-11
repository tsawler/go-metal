package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

func TestAdaDeltaOptimizer(t *testing.T) {
	// Initialize device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
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

	// Test that learning rate update fails (AdaDelta doesn't use fixed learning rate)
	err = optimizer.UpdateLearningRate(0.005)
	if err == nil {
		t.Error("Expected error for learning rate update (AdaDelta doesn't use fixed LR), got nil")
	}
}

func TestAdaDeltaOptimizerWithCommandPool(t *testing.T) {
	// Skip command pool test for now since pooling is not yet implemented
	t.Skip("Command buffer pooling not yet implemented for AdaDelta")
}

func TestAdaDeltaOptimizerInvalidInputs(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
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