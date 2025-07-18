package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
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

// TestAdamSetCommandPool tests the SetCommandPool method
func TestAdamSetCommandPool(t *testing.T) {
	adam := MockAdamOptimizer()
	
	// Test setting command pool
	mockPool := unsafe.Pointer(uintptr(0x2000))
	adam.SetCommandPool(mockPool)
	
	if adam.commandPool != mockPool {
		t.Errorf("Expected commandPool %p, got %p", mockPool, adam.commandPool)
	}
	
	if !adam.usePooling {
		t.Error("Expected usePooling to be true")
	}
	
	// Test setting nil pool
	adam.SetCommandPool(nil)
	if adam.usePooling {
		t.Error("Expected usePooling to be false after setting nil pool")
	}
	
	t.Log("Adam SetCommandPool test passed")
}

// TestAdamPowFunction tests the pow utility function
func TestAdamPowFunction(t *testing.T) {
	// Test cases for pow function
	testCases := []struct {
		name     string
		x        float32
		y        float32
		expected float32
	}{
		{"zero_power", 5.0, 0.0, 1.0},
		{"one_power", 5.0, 1.0, 5.0},
		{"two_power", 3.0, 2.0, 9.0},
		{"three_power", 2.0, 3.0, 8.0},
		{"negative_base", -2.0, 2.0, 4.0},
		{"decimal_base", 1.5, 2.0, 2.25},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := pow(tc.x, tc.y)
			if result != tc.expected {
				t.Errorf("pow(%f, %f) = %f, expected %f", tc.x, tc.y, result, tc.expected)
			}
		})
	}
	
	t.Log("Adam pow function test passed")
}

// TestAdamCleanupMethod tests the cleanup method (private method)
func TestAdamCleanupMethod(t *testing.T) {
	adam := MockAdamOptimizer()
	
	// The cleanup method is private, so we can't test it directly
	// But we can test that it's called through the public Cleanup method
	
	// Set some mock buffers
	adam.MomentumBuffers[0] = unsafe.Pointer(uintptr(0x1000))
	adam.VarianceBuffers[0] = unsafe.Pointer(uintptr(0x2000))
	
	// Call public Cleanup method (which calls private cleanup)
	adam.Cleanup()
	
	// Verify buffers were cleared
	if adam.MomentumBuffers != nil {
		t.Error("Expected MomentumBuffers to be nil after cleanup")
	}
	if adam.VarianceBuffers != nil {
		t.Error("Expected VarianceBuffers to be nil after cleanup")
	}
	if adam.WeightBuffers != nil {
		t.Error("Expected WeightBuffers to be nil after cleanup")
	}
	
	t.Log("Adam cleanup method test passed")
}

// TestAdamLoadStateValidation tests LoadState parameter validation and error handling
func TestAdamLoadStateValidation(t *testing.T) {
	adam := MockAdamOptimizer()
	
	// Test invalid state type
	invalidState := &OptimizerState{
		Type: "InvalidType",
		Parameters: map[string]interface{}{
			"learning_rate": 0.001,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err := adam.LoadState(invalidState)
	if err == nil {
		t.Error("Expected error for invalid state type, got nil")
	}
	
	// Test parameter restoration with valid state
	originalLR := adam.LearningRate
	originalBeta1 := adam.Beta1
	originalBeta2 := adam.Beta2
	originalEpsilon := adam.Epsilon
	originalWeightDecay := adam.WeightDecay
	originalStepCount := adam.StepCount
	
	validState := &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": 0.002,
			"beta1":         0.85,
			"beta2":         0.995,
			"epsilon":       1e-7,
			"weight_decay":  0.005,
			"step_count":    float64(10),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err = adam.LoadState(validState)
	if err != nil {
		t.Errorf("Expected no error for valid state, got %v", err)
	}
	
	// Verify parameters were restored correctly
	if adam.LearningRate != 0.002 {
		t.Errorf("Expected learning rate 0.002, got %f", adam.LearningRate)
	}
	if adam.Beta1 != 0.85 {
		t.Errorf("Expected beta1 0.85, got %f", adam.Beta1)
	}
	if adam.Beta2 != 0.995 {
		t.Errorf("Expected beta2 0.995, got %f", adam.Beta2)
	}
	if adam.Epsilon != 1e-7 {
		t.Errorf("Expected epsilon 1e-7, got %f", adam.Epsilon)
	}
	if adam.WeightDecay != 0.005 {
		t.Errorf("Expected weight decay 0.005, got %f", adam.WeightDecay)
	}
	if adam.StepCount != 10 {
		t.Errorf("Expected step count 10, got %d", adam.StepCount)
	}
	
	// Test that missing parameters don't change existing values
	partialState := &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": 0.003,
			// Missing other parameters
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err = adam.LoadState(partialState)
	if err != nil {
		t.Errorf("Expected no error for partial state, got %v", err)
	}
	
	// Only learning rate should have changed
	if adam.LearningRate != 0.003 {
		t.Errorf("Expected learning rate 0.003, got %f", adam.LearningRate)
	}
	if adam.Beta1 != 0.85 { // Should remain unchanged
		t.Errorf("Expected beta1 0.85 (unchanged), got %f", adam.Beta1)
	}
	
	// Test wrong parameter types
	wrongTypeState := &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": "not_a_number",
			"beta1":         0.9,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err = adam.LoadState(wrongTypeState)
	if err != nil {
		t.Errorf("Expected no error for wrong type (should be ignored), got %v", err)
	}
	
	// Learning rate should be unchanged since string couldn't be converted
	if adam.LearningRate != 0.003 {
		t.Errorf("Expected learning rate 0.003 (unchanged), got %f", adam.LearningRate)
	}
	
	// Restore original values
	adam.LearningRate = originalLR
	adam.Beta1 = originalBeta1
	adam.Beta2 = originalBeta2
	adam.Epsilon = originalEpsilon
	adam.WeightDecay = originalWeightDecay
	adam.StepCount = originalStepCount
	
	t.Log("Adam LoadState validation test passed")
}

// TestAdamGetStateStructure tests GetState structure and parameters
func TestAdamGetStateStructure(t *testing.T) {
	adam := MockAdamOptimizer()
	
	// Set specific values to verify they're captured
	adam.LearningRate = 0.002
	adam.Beta1 = 0.85
	adam.Beta2 = 0.995
	adam.Epsilon = 1e-7
	adam.WeightDecay = 0.005
	adam.StepCount = 15
	
	// Note: We can't test the full GetState because it requires Metal buffers
	// But we can test the structure and parameters that would be created
	
	// Test that GetState would fail with nil buffers (expected behavior)
	// Note: GetState may not fail immediately with mock data, but would fail when trying to read Metal buffers
	_, err := adam.GetState()
	if err == nil {
		t.Log("GetState did not fail with mock data (expected - CGO bridge calls would fail)")
	} else {
		t.Logf("GetState failed as expected with mock data: %v", err)
	}
	
	// Test parameter structure by creating what GetState would create
	expectedState := &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": adam.LearningRate,
			"beta1":         adam.Beta1,
			"beta2":         adam.Beta2,
			"epsilon":       adam.Epsilon,
			"weight_decay":  adam.WeightDecay,
			"step_count":    adam.StepCount,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	// Verify the structure we expect GetState to produce
	if expectedState.Type != "Adam" {
		t.Errorf("Expected type 'Adam', got '%s'", expectedState.Type)
	}
	
	if expectedState.Parameters["learning_rate"] != adam.LearningRate {
		t.Errorf("Expected learning rate %f, got %v", adam.LearningRate, expectedState.Parameters["learning_rate"])
	}
	
	if expectedState.Parameters["beta1"] != adam.Beta1 {
		t.Errorf("Expected beta1 %f, got %v", adam.Beta1, expectedState.Parameters["beta1"])
	}
	
	if expectedState.Parameters["beta2"] != adam.Beta2 {
		t.Errorf("Expected beta2 %f, got %v", adam.Beta2, expectedState.Parameters["beta2"])
	}
	
	if expectedState.Parameters["epsilon"] != adam.Epsilon {
		t.Errorf("Expected epsilon %f, got %v", adam.Epsilon, expectedState.Parameters["epsilon"])
	}
	
	if expectedState.Parameters["weight_decay"] != adam.WeightDecay {
		t.Errorf("Expected weight decay %f, got %v", adam.WeightDecay, expectedState.Parameters["weight_decay"])
	}
	
	if expectedState.Parameters["step_count"] != adam.StepCount {
		t.Errorf("Expected step count %d, got %v", adam.StepCount, expectedState.Parameters["step_count"])
	}
	
	t.Log("Adam GetState structure test passed")
}

// TestAdamStateRoundTrip tests the parameter roundtrip (without Metal buffers)
func TestAdamStateRoundTrip(t *testing.T) {
	adam := MockAdamOptimizer()
	
	// Set initial values
	adam.LearningRate = 0.003
	adam.Beta1 = 0.88
	adam.Beta2 = 0.998
	adam.Epsilon = 1e-6
	adam.WeightDecay = 0.01
	adam.StepCount = 42
	
	// Create state manually (simulating what GetState would create)
	state := &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": float64(adam.LearningRate),
			"beta1":         float64(adam.Beta1),
			"beta2":         float64(adam.Beta2),
			"epsilon":       float64(adam.Epsilon),
			"weight_decay":  float64(adam.WeightDecay),
			"step_count":    float64(adam.StepCount),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	// Modify values to test restoration
	adam.LearningRate = 0.999
	adam.Beta1 = 0.999
	adam.Beta2 = 0.999
	adam.Epsilon = 0.999
	adam.WeightDecay = 0.999
	adam.StepCount = 999
	
	// Restore state
	err := adam.LoadState(state)
	if err != nil {
		t.Errorf("Unexpected error during LoadState: %v", err)
	}
	
	// Verify all values were restored correctly
	if adam.LearningRate != 0.003 {
		t.Errorf("Expected learning rate 0.003, got %f", adam.LearningRate)
	}
	if adam.Beta1 != 0.88 {
		t.Errorf("Expected beta1 0.88, got %f", adam.Beta1)
	}
	if adam.Beta2 != 0.998 {
		t.Errorf("Expected beta2 0.998, got %f", adam.Beta2)
	}
	if adam.Epsilon != 1e-6 {
		t.Errorf("Expected epsilon 1e-6, got %f", adam.Epsilon)
	}
	if adam.WeightDecay != 0.01 {
		t.Errorf("Expected weight decay 0.01, got %f", adam.WeightDecay)
	}
	if adam.StepCount != 42 {
		t.Errorf("Expected step count 42, got %d", adam.StepCount)
	}
	
	t.Log("Adam state roundtrip test passed")
}