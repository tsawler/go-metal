package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

func TestAdaGradOptimizer(t *testing.T) {
	// Initialize device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
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
	err = optimizer.UpdateLearningRate(newLR)
	if err != nil {
		t.Fatalf("Failed to update learning rate: %v", err)
	}

	stats = optimizer.GetStats()
	if stats["learning_rate"] != newLR {
		t.Errorf("Expected updated learning_rate to be %f, got %v", newLR, stats["learning_rate"])
	}

	// Test invalid learning rate
	err = optimizer.UpdateLearningRate(-0.001)
	if err == nil {
		t.Error("Expected error for negative learning rate, got nil")
	}
}

func TestAdaGradOptimizerWithCommandPool(t *testing.T) {
	// Skip command pool test for now since pooling is not yet implemented
	t.Skip("Command buffer pooling not yet implemented for AdaGrad")
}

func TestAdaGradOptimizerInvalidInputs(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Fatalf("Failed to create Metal device: %v", err)
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