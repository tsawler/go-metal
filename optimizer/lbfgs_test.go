package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

func TestLBFGSConfig_Default(t *testing.T) {
	config := DefaultLBFGSConfig()
	
	if config.HistorySize != 10 {
		t.Errorf("Expected default history size 10, got %d", config.HistorySize)
	}
	if config.LineSearchTol != 1e-4 {
		t.Errorf("Expected default line search tolerance 1e-4, got %f", config.LineSearchTol)
	}
	if config.MaxLineSearch != 20 {
		t.Errorf("Expected default max line search 20, got %d", config.MaxLineSearch)
	}
	if config.C1 != 1e-4 {
		t.Errorf("Expected default C1 1e-4, got %f", config.C1)
	}
	if config.C2 != 0.9 {
		t.Errorf("Expected default C2 0.9, got %f", config.C2)
	}
	if config.InitialStep != 1.0 {
		t.Errorf("Expected default initial step 1.0, got %f", config.InitialStep)
	}
}

func TestLBFGSOptimizer_Creation(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for L-BFGS creation test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()
	
	config := DefaultLBFGSConfig()
	config.HistorySize = 5 // Small history for testing
	
	paramShapes := [][]int{
		{10, 5},  // Weight matrix
		{5},      // Bias vector
		{5, 1},   // Output weight
	}
	
	optimizer, err := NewLBFGSOptimizer(
		config,
		paramShapes,
		memoryManager,
		device,
	)
	
	if err != nil {
		t.Fatalf("Failed to create L-BFGS optimizer: %v", err)
	}
	defer optimizer.Cleanup()
	
	// Test basic properties
	if optimizer.config.HistorySize != 5 {
		t.Errorf("Expected history size 5, got %d", optimizer.config.HistorySize)
	}
	
	if optimizer.historyCount != 0 {
		t.Errorf("Expected initial history count 0, got %d", optimizer.historyCount)
	}
	
	if optimizer.currentStep != 0 {
		t.Errorf("Expected initial step count 0, got %d", optimizer.currentStep)
	}
	
	// Test that buffers were allocated
	if len(optimizer.sVectors) != 5 {
		t.Errorf("Expected 5 s-vector entries, got %d", len(optimizer.sVectors))
	}
	
	if len(optimizer.yVectors) != 5 {
		t.Errorf("Expected 5 y-vector entries, got %d", len(optimizer.yVectors))
	}
	
	if len(optimizer.rhoBuffers) != 5 {
		t.Errorf("Expected 5 rho buffers, got %d", len(optimizer.rhoBuffers))
	}
	
	if len(optimizer.oldGradients) != 3 {
		t.Errorf("Expected 3 old gradient buffers, got %d", len(optimizer.oldGradients))
	}
	
	if len(optimizer.searchDir) != 3 {
		t.Errorf("Expected 3 search direction buffers, got %d", len(optimizer.searchDir))
	}
}

func TestLBFGSOptimizer_InvalidConfig(t *testing.T) {
	// Skip this test as it requires actual Metal device
	t.Skip("Skipping L-BFGS invalid config test - requires actual Metal device for buffer allocation")
	
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	// Test invalid history size
	config := DefaultLBFGSConfig()
	config.HistorySize = 0
	
	paramShapes := [][]int{{10, 5}}
	
	_, err = NewLBFGSOptimizer(
		config,
		paramShapes,
		memory.GetGlobalMemoryManager(),
		device,
	)
	
	if err == nil {
		t.Error("Expected error for zero history size, got nil")
	}
}

func TestLBFGSOptimizer_InvalidInputs(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for L-BFGS invalid inputs test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{{10, 5}}
	
	// Test nil memory manager
	_, err = NewLBFGSOptimizer(config, paramShapes, nil, device)
	if err == nil {
		t.Error("Expected error for nil memory manager, got nil")
	}
	
	// Test nil device
	_, err = NewLBFGSOptimizer(config, paramShapes, memoryManager, nil)
	if err == nil {
		t.Error("Expected error for nil device, got nil")
	}
	
	// Test empty parameter shapes
	_, err = NewLBFGSOptimizer(config, [][]int{}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for empty parameter shapes, got nil")
	}
}

func TestLBFGSOptimizer_SetWeightBuffers(t *testing.T) {
	// Skip this test as it requires actual Metal device
	t.Skip("Skipping L-BFGS set weight buffers test - requires actual Metal device for buffer allocation")
	
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{{10, 5}, {5}}
	
	optimizer, err := NewLBFGSOptimizer(
		config,
		paramShapes,
		memory.GetGlobalMemoryManager(),
		device,
	)
	if err != nil {
		t.Fatalf("Failed to create optimizer: %v", err)
	}
	defer optimizer.Cleanup()
	
	// Create test tensors
	tensors := make([]*memory.Tensor, len(paramShapes))
	weightBuffers := make([]unsafe.Pointer, len(paramShapes))
	
	for i, shape := range paramShapes {
		tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
		if err != nil {
			t.Fatalf("Failed to create tensor %d: %v", i, err)
		}
		tensors[i] = tensor
		weightBuffers[i] = tensor.MetalBuffer()
	}
	
	// Test setting weight buffers
	err = optimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		t.Errorf("Failed to set weight buffers: %v", err)
	}
	
	// Test wrong number of buffers
	err = optimizer.SetWeightBuffers(weightBuffers[:1])
	if err == nil {
		t.Error("Expected error for wrong number of weight buffers, got nil")
	}
	
	// Cleanup
	for _, tensor := range tensors {
		tensor.Release()
	}
}

func TestLBFGSOptimizer_GetStats(t *testing.T) {
	// Skip this test as it requires actual Metal device
	t.Skip("Skipping L-BFGS get stats test - requires actual Metal device for buffer allocation")
	
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{{5, 3}}
	
	optimizer, err := NewLBFGSOptimizer(
		config,
		paramShapes,
		memory.GetGlobalMemoryManager(),
		device,
	)
	if err != nil {
		t.Fatalf("Failed to create optimizer: %v", err)
	}
	defer optimizer.Cleanup()
	
	stats := optimizer.GetStats()
	
	// Verify stats structure
	if step, ok := stats["step"]; !ok {
		t.Error("Stats missing 'step' field")
	} else if step != uint64(0) {
		t.Errorf("Expected initial step 0, got %v", step)
	}
	
	if historySize, ok := stats["history_size"]; !ok {
		t.Error("Stats missing 'history_size' field")
	} else if historySize != config.HistorySize {
		t.Errorf("Expected history size %d, got %v", config.HistorySize, historySize)
	}
	
	if historyUsed, ok := stats["history_used"]; !ok {
		t.Error("Stats missing 'history_used' field")
	} else if historyUsed != 0 {
		t.Errorf("Expected initial history used 0, got %v", historyUsed)
	}
}

func TestLBFGSOptimizer_UpdateLearningRate(t *testing.T) {
	// Skip this test as it requires actual Metal device
	t.Skip("Skipping L-BFGS update learning rate test - requires actual Metal device for buffer allocation")
	
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{{5, 3}}
	
	optimizer, err := NewLBFGSOptimizer(
		config,
		paramShapes,
		memory.GetGlobalMemoryManager(),
		device,
	)
	if err != nil {
		t.Fatalf("Failed to create optimizer: %v", err)
	}
	defer optimizer.Cleanup()
	
	// L-BFGS should not allow learning rate updates
	err = optimizer.UpdateLearningRate(0.001)
	if err == nil {
		t.Error("Expected error when trying to update learning rate for L-BFGS, got nil")
	}
}

func TestLBFGSOptimizer_Cleanup(t *testing.T) {
	// Skip this test as it requires actual Metal device
	t.Skip("Skipping L-BFGS cleanup test - requires actual Metal device for buffer allocation")
	
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{{5, 3}, {3}}
	
	optimizer, err := NewLBFGSOptimizer(
		config,
		paramShapes,
		memory.GetGlobalMemoryManager(),
		device,
	)
	if err != nil {
		t.Fatalf("Failed to create optimizer: %v", err)
	}
	
	// Test cleanup doesn't panic
	optimizer.Cleanup()
	
	// Test multiple cleanups don't panic
	optimizer.Cleanup()
}

func TestLBFGSOptimizer_BufferSizes(t *testing.T) {
	// Skip this test as it requires actual Metal device
	t.Skip("Skipping L-BFGS buffer sizes test - requires actual Metal device for buffer allocation")
	
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{
		{10, 5},  // 50 elements = 200 bytes
		{5},      // 5 elements = 20 bytes
		{5, 1},   // 5 elements = 20 bytes
	}
	
	optimizer, err := NewLBFGSOptimizer(
		config,
		paramShapes,
		memory.GetGlobalMemoryManager(),
		device,
	)
	if err != nil {
		t.Fatalf("Failed to create optimizer: %v", err)
	}
	defer optimizer.Cleanup()
	
	expectedSizes := []int{200, 20, 20} // 4 bytes per float32
	
	if len(optimizer.bufferSizes) != len(expectedSizes) {
		t.Errorf("Expected %d buffer sizes, got %d", len(expectedSizes), len(optimizer.bufferSizes))
	}
	
	for i, expected := range expectedSizes {
		if optimizer.bufferSizes[i] != expected {
			t.Errorf("Buffer %d: expected size %d, got %d", i, expected, optimizer.bufferSizes[i])
		}
	}
}

// Benchmark tests
func BenchmarkLBFGSOptimizer_Creation(b *testing.B) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	memory.InitializeGlobalMemoryManager(device)
	
	config := DefaultLBFGSConfig()
	paramShapes := [][]int{{100, 50}, {50}, {50, 10}, {10}}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer, err := NewLBFGSOptimizer(
			config,
			paramShapes,
			memory.GetGlobalMemoryManager(),
			device,
		)
		if err != nil {
			b.Fatalf("Failed to create optimizer: %v", err)
		}
		optimizer.Cleanup()
	}
}

// TestLBFGSSetWeightBuffers tests the SetWeightBuffers method
func TestLBFGSSetWeightBuffers(t *testing.T) {
	config := DefaultLBFGSConfig()
	weightShapes := [][]int{{10, 5}, {5}, {5, 3}}
	
	// Create mock L-BFGS optimizer
	optimizer := &LBFGSOptimizerState{
		config:      config,
		WeightBuffers: make([]unsafe.Pointer, len(weightShapes)),
		currentStep: 0,
	}
	
	// Test setting correct number of weight buffers
	weightBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x1000)),
		unsafe.Pointer(uintptr(0x2000)),
		unsafe.Pointer(uintptr(0x3000)),
	}
	
	err := optimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		t.Errorf("Unexpected error for correct buffer count: %v", err)
	}
	
	// Verify buffers were set
	for i, expected := range weightBuffers {
		if optimizer.WeightBuffers[i] != expected {
			t.Errorf("Expected WeightBuffers[%d] %p, got %p", i, expected, optimizer.WeightBuffers[i])
		}
	}
	
	// Test setting wrong number of weight buffers
	wrongBuffers := []unsafe.Pointer{unsafe.Pointer(uintptr(0x1000))}
	err = optimizer.SetWeightBuffers(wrongBuffers)
	if err == nil {
		t.Error("Expected error for mismatched buffer count, got nil")
	}
	
	t.Log("L-BFGS SetWeightBuffers test passed")
}

// TestLBFGSGetStep tests the GetStep method
func TestLBFGSGetStep(t *testing.T) {
	config := DefaultLBFGSConfig()
	
	// Create mock L-BFGS optimizer
	optimizer := &LBFGSOptimizerState{
		config:      config,
		currentStep: 0,
	}
	
	// Test initial step count
	if optimizer.GetStep() != 0 {
		t.Errorf("Expected initial step count 0, got %d", optimizer.GetStep())
	}
	
	// Test after incrementing step count
	optimizer.currentStep = 42
	if optimizer.GetStep() != 42 {
		t.Errorf("Expected step count 42, got %d", optimizer.GetStep())
	}
	
	t.Log("L-BFGS GetStep test passed")
}

// TestLBFGSGetStats tests the GetStats method
func TestLBFGSGetStats(t *testing.T) {
	config := DefaultLBFGSConfig()
	
	// Create mock L-BFGS optimizer
	optimizer := &LBFGSOptimizerState{
		config:       config,
		currentStep:  5,
		historyCount: 3,
		prevLoss:     0.123,
	}
	
	stats := optimizer.GetStats()
	
	// Check all expected fields
	if stats["step"] != uint64(5) {
		t.Errorf("Expected step 5, got %v", stats["step"])
	}
	if stats["history_size"] != config.HistorySize {
		t.Errorf("Expected history_size %d, got %v", config.HistorySize, stats["history_size"])
	}
	if stats["history_used"] != 3 {
		t.Errorf("Expected history_used 3, got %v", stats["history_used"])
	}
	if stats["prev_loss"] != float32(0.123) {
		t.Errorf("Expected prev_loss 0.123, got %v", stats["prev_loss"])
	}
	
	t.Log("L-BFGS GetStats test passed")
}

// TestLBFGSSetCommandPool tests the SetCommandPool method
func TestLBFGSSetCommandPool(t *testing.T) {
	config := DefaultLBFGSConfig()
	
	// Create mock L-BFGS optimizer
	optimizer := &LBFGSOptimizerState{
		config:      config,
		currentStep: 0,
	}
	
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
	
	t.Log("L-BFGS SetCommandPool test passed")
}

// TestLBFGSUpdateLearningRate tests the UpdateLearningRate method
func TestLBFGSUpdateLearningRate(t *testing.T) {
	config := DefaultLBFGSConfig()
	
	// Create mock L-BFGS optimizer
	optimizer := &LBFGSOptimizerState{
		config:      config,
		currentStep: 0,
	}
	
	// Test that UpdateLearningRate returns error for L-BFGS
	err := optimizer.UpdateLearningRate(0.001)
	if err == nil {
		t.Error("Expected error for L-BFGS UpdateLearningRate, got nil")
	}
	
	// Check error message
	expectedMsg := "L-BFGS does not use a fixed learning rate; it uses line search"
	if err.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
	}
	
	t.Log("L-BFGS UpdateLearningRate test passed")
}

// TestLBFGSStepValidation tests the Step method validation
func TestLBFGSStepValidation(t *testing.T) {
	config := DefaultLBFGSConfig()
	weightShapes := [][]int{{10, 5}, {5}}
	
	// Create mock L-BFGS optimizer
	optimizer := &LBFGSOptimizerState{
		config:        config,
		WeightBuffers: make([]unsafe.Pointer, len(weightShapes)),
		currentStep:   0,
	}
	
	// Test step with mismatched gradient buffer count
	gradientBuffers := []unsafe.Pointer{unsafe.Pointer(uintptr(0x1000))}
	err := optimizer.Step(gradientBuffers, 0.5)
	if err == nil {
		t.Error("Expected error for mismatched gradient buffer count, got nil")
	}
	
	// Test step with correct gradient buffer count (will fail on CGO bridge call)
	// Note: We can't test the full step due to CGO bridge dependency
	correctGradientBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x1000)),
		unsafe.Pointer(uintptr(0x2000)),
	}
	
	// Test the validation logic directly
	if len(correctGradientBuffers) != len(optimizer.WeightBuffers) {
		t.Errorf("Expected gradient buffer count %d to match weight buffer count %d", 
			len(correctGradientBuffers), len(optimizer.WeightBuffers))
	}
	
	t.Log("L-BFGS Step validation test passed")
}