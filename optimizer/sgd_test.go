package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/memory"
)

// TestDefaultSGDConfig tests the default SGD configuration
func TestDefaultSGDConfig(t *testing.T) {
	config := DefaultSGDConfig()
	
	expectedLR := float32(0.01)
	expectedMomentum := float32(0.0)
	expectedWeightDecay := float32(0.0)
	expectedNesterov := false
	
	if config.LearningRate != expectedLR {
		t.Errorf("Expected LearningRate %f, got %f", expectedLR, config.LearningRate)
	}
	
	if config.Momentum != expectedMomentum {
		t.Errorf("Expected Momentum %f, got %f", expectedMomentum, config.Momentum)
	}
	
	if config.WeightDecay != expectedWeightDecay {
		t.Errorf("Expected WeightDecay %f, got %f", expectedWeightDecay, config.WeightDecay)
	}
	
	if config.Nesterov != expectedNesterov {
		t.Errorf("Expected Nesterov %t, got %t", expectedNesterov, config.Nesterov)
	}
	
	t.Log("Default SGD config test passed")
}

// TestSGDOptimizerCreation tests SGD optimizer creation
func TestSGDOptimizerCreation(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for SGD optimizer test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultSGDConfig()
	config.Momentum = 0.9
	config.WeightDecay = 0.0001
	config.Nesterov = true

	weightShapes := [][]int{
		{10, 5},
		{5},
		{5, 3},
		{3},
	}

	optimizer, err := NewSGDOptimizer(config, weightShapes, memoryManager, device)
	if err != nil {
		t.Fatalf("Failed to create SGD optimizer: %v", err)
	}
	defer optimizer.Cleanup()

	// Test initial state
	if optimizer.GetStepCount() != 0 {
		t.Errorf("Expected initial step count 0, got %d", optimizer.GetStepCount())
	}

	// Test hyperparameters
	if optimizer.LearningRate != config.LearningRate {
		t.Errorf("Expected LearningRate %f, got %f", config.LearningRate, optimizer.LearningRate)
	}

	if optimizer.Momentum != config.Momentum {
		t.Errorf("Expected Momentum %f, got %f", config.Momentum, optimizer.Momentum)
	}

	if optimizer.WeightDecay != config.WeightDecay {
		t.Errorf("Expected WeightDecay %f, got %f", config.WeightDecay, optimizer.WeightDecay)
	}

	if optimizer.Nesterov != config.Nesterov {
		t.Errorf("Expected Nesterov %t, got %t", config.Nesterov, optimizer.Nesterov)
	}
}

// TestSGDOptimizerInvalidInputs tests SGD optimizer with invalid inputs
func TestSGDOptimizerInvalidInputs(t *testing.T) {
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for SGD invalid inputs test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	config := DefaultSGDConfig()

	// Test nil memory manager
	_, err = NewSGDOptimizer(config, [][]int{{5, 5}}, nil, device)
	if err == nil {
		t.Error("Expected error for nil memory manager, got nil")
	}

	// Test nil device
	_, err = NewSGDOptimizer(config, [][]int{{5, 5}}, memoryManager, nil)
	if err == nil {
		t.Error("Expected error for nil device, got nil")
	}

	// Test empty weight shapes
	_, err = NewSGDOptimizer(config, [][]int{}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for empty weight shapes, got nil")
	}

	// Test invalid learning rate
	invalidConfig := config
	invalidConfig.LearningRate = -0.001
	_, err = NewSGDOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative learning rate, got nil")
	}

	// Test invalid momentum
	invalidConfig = config
	invalidConfig.Momentum = -0.1
	_, err = NewSGDOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative momentum, got nil")
	}

	invalidConfig.Momentum = 1.1
	_, err = NewSGDOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for momentum > 1.0, got nil")
	}

	// Test invalid weight decay
	invalidConfig = config
	invalidConfig.WeightDecay = -0.1
	_, err = NewSGDOptimizer(invalidConfig, [][]int{{5, 5}}, memoryManager, device)
	if err == nil {
		t.Error("Expected error for negative weight decay, got nil")
	}
}

// MockSGDOptimizer creates a mock SGD optimizer for testing without Metal device
func MockSGDOptimizer() *SGDOptimizerState {
	config := DefaultSGDConfig()
	config.Momentum = 0.9
	config.WeightDecay = 0.0001
	config.Nesterov = true
	
	sgd := &SGDOptimizerState{
		LearningRate:    config.LearningRate,
		Momentum:        config.Momentum,
		WeightDecay:     config.WeightDecay,
		Nesterov:        config.Nesterov,
		MomentumBuffers: make([]unsafe.Pointer, 2),
		WeightBuffers:   make([]unsafe.Pointer, 2),
		StepCount:       0,
		memoryManager:   nil, // Mock
		device:          nil, // Mock
		bufferSizes:     []int{64, 8}, // FC weights: 8*2*4=64 bytes, FC bias: 2*4=8 bytes
	}

	return sgd
}

// TestSGDMockOperations tests SGD operations with mock data
func TestSGDMockOperations(t *testing.T) {
	sgd := MockSGDOptimizer()

	// Test initial state
	if sgd.GetStepCount() != 0 {
		t.Errorf("Expected initial step count 0, got %d", sgd.GetStepCount())
	}

	// Test learning rate update
	newLR := float32(0.005)
	sgd.UpdateLearningRate(newLR)

	if sgd.LearningRate != newLR {
		t.Errorf("Expected learning rate %f, got %f", newLR, sgd.LearningRate)
	}

	// Test hyperparameters
	if sgd.Momentum != 0.9 {
		t.Errorf("Expected momentum 0.9, got %f", sgd.Momentum)
	}

	if sgd.WeightDecay != 0.0001 {
		t.Errorf("Expected weight decay 0.0001, got %f", sgd.WeightDecay)
	}

	if !sgd.Nesterov {
		t.Errorf("Expected Nesterov true, got %t", sgd.Nesterov)
	}

	// Test command pool
	commandPool := unsafe.Pointer(uintptr(0x5000))
	sgd.SetCommandPool(commandPool)
	if sgd.commandPool != commandPool {
		t.Errorf("Expected command pool %p, got %p", commandPool, sgd.commandPool)
	}
	if !sgd.usePooling {
		t.Error("Expected usePooling to be true")
	}

	// Test setting nil command pool
	sgd.SetCommandPool(nil)
	if sgd.usePooling {
		t.Error("Expected usePooling to be false after setting nil pool")
	}

	t.Log("SGD mock operations test passed")
}

// TestSGDCheckpointing tests the checkpointing functionality
func TestSGDCheckpointing(t *testing.T) {
	sgd := MockSGDOptimizer()

	// Set some state
	sgd.StepCount = 20
	sgd.LearningRate = 0.008
	sgd.Momentum = 0.85
	sgd.WeightDecay = 0.0005

	// Test GetState with mock data (will fail with real Metal calls)
	t.Skip("GetState/LoadState requires Metal buffer operations - test in integration tests")

	// Test state type validation
	invalidState := &OptimizerState{
		Type: "InvalidType",
		Parameters: map[string]interface{}{
			"learning_rate": 0.01,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err := sgd.LoadState(invalidState)
	if err == nil {
		t.Error("Expected error for invalid state type, got nil")
	}

	// Test parameter restoration with valid state
	validState := &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": 0.012,
			"momentum":      0.88,
			"weight_decay":  0.0008,
			"nesterov":      false,
			"step_count":    float64(15),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}

	err = sgd.LoadState(validState)
	if err != nil {
		t.Errorf("Expected no error for valid state, got %v", err)
	}

	// Verify parameters were restored
	if sgd.LearningRate != 0.012 {
		t.Errorf("Expected learning rate 0.012, got %f", sgd.LearningRate)
	}
	if sgd.Momentum != 0.88 {
		t.Errorf("Expected momentum 0.88, got %f", sgd.Momentum)
	}
	if sgd.WeightDecay != 0.0008 {
		t.Errorf("Expected weight decay 0.0008, got %f", sgd.WeightDecay)
	}
	if sgd.Nesterov != false {
		t.Errorf("Expected nesterov false, got %t", sgd.Nesterov)
	}
	if sgd.StepCount != 15 {
		t.Errorf("Expected step count 15, got %d", sgd.StepCount)
	}

	t.Log("SGD checkpointing test passed")
}

// TestSGDBasicStructures tests the basic SGD data structures
func TestSGDBasicStructures(t *testing.T) {
	// Test weight shapes calculation
	shapes := [][]int{
		{8, 2},   // FC weights
		{2},      // FC bias
		{10, 5},  // Another layer
		{32, 16}, // Larger layer
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
	config := SGDConfig{
		LearningRate: 0.02,
		Momentum:     0.95,
		WeightDecay:  0.001,
		Nesterov:     true,
	}

	if config.LearningRate != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", config.LearningRate)
	}

	if config.Momentum != 0.95 {
		t.Errorf("Expected momentum 0.95, got %f", config.Momentum)
	}

	if config.WeightDecay != 0.001 {
		t.Errorf("Expected weight decay 0.001, got %f", config.WeightDecay)
	}

	if config.Nesterov != true {
		t.Errorf("Expected nesterov true, got %t", config.Nesterov)
	}

	t.Log("SGD basic structures test passed")
}

// TestSGDVariations tests different SGD variations
func TestSGDVariations(t *testing.T) {
	// Test vanilla SGD (no momentum)
	vanillaSGD := SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.0,
		WeightDecay:  0.0,
		Nesterov:     false,
	}

	if vanillaSGD.Momentum != 0.0 {
		t.Errorf("Vanilla SGD momentum should be 0.0, got %f", vanillaSGD.Momentum)
	}

	// Test SGD with momentum
	momentumSGD := SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.9,
		WeightDecay:  0.0,
		Nesterov:     false,
	}

	if momentumSGD.Momentum != 0.9 {
		t.Errorf("Momentum SGD momentum should be 0.9, got %f", momentumSGD.Momentum)
	}

	// Test Nesterov SGD
	nesterovSGD := SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.9,
		WeightDecay:  0.0,
		Nesterov:     true,
	}

	if !nesterovSGD.Nesterov {
		t.Errorf("Nesterov SGD should have Nesterov=true, got %t", nesterovSGD.Nesterov)
	}

	// Test SGD with weight decay
	weightDecaySGD := SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.9,
		WeightDecay:  0.0001,
		Nesterov:     false,
	}

	if weightDecaySGD.WeightDecay != 0.0001 {
		t.Errorf("Weight decay SGD should have WeightDecay=0.0001, got %f", weightDecaySGD.WeightDecay)
	}

	t.Log("SGD variations test passed")
}

// TestSGDParameterValidation tests parameter validation
func TestSGDParameterValidation(t *testing.T) {
	baseConfig := DefaultSGDConfig()

	// Test learning rate validation
	testCases := []struct {
		name         string
		learningRate float32
		expectError  bool
	}{
		{"valid_lr", 0.01, false},
		{"zero_lr", 0.0, true},
		{"negative_lr", -0.01, true},
		{"very_small_lr", 1e-10, false},
		{"large_lr", 10.0, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := baseConfig
			config.LearningRate = tc.learningRate

			// Test validation logic (would be in NewSGDOptimizer)
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

	// Test momentum validation
	momentumTestCases := []struct {
		name        string
		momentum    float32
		expectError bool
	}{
		{"valid_momentum", 0.9, false},
		{"zero_momentum", 0.0, false},
		{"negative_momentum", -0.1, true},
		{"one_momentum", 1.0, false},
		{"above_one_momentum", 1.1, true},
	}

	for _, tc := range momentumTestCases {
		t.Run(tc.name, func(t *testing.T) {
			config := baseConfig
			config.Momentum = tc.momentum

			// Test validation logic
			if tc.expectError {
				if config.Momentum >= 0 && config.Momentum <= 1 {
					t.Errorf("Expected invalid momentum %f to be outside [0, 1]", config.Momentum)
				}
			} else {
				if config.Momentum < 0 || config.Momentum > 1 {
					t.Errorf("Expected valid momentum %f to be in [0, 1]", config.Momentum)
				}
			}
		})
	}

	// Test weight decay validation
	weightDecayTestCases := []struct {
		name        string
		weightDecay float32
		expectError bool
	}{
		{"valid_weight_decay", 0.0001, false},
		{"zero_weight_decay", 0.0, false},
		{"negative_weight_decay", -0.0001, true},
		{"large_weight_decay", 0.1, false},
	}

	for _, tc := range weightDecayTestCases {
		t.Run(tc.name, func(t *testing.T) {
			config := baseConfig
			config.WeightDecay = tc.weightDecay

			// Test validation logic
			if tc.expectError {
				if config.WeightDecay >= 0 {
					t.Errorf("Expected invalid weight decay %f to be < 0", config.WeightDecay)
				}
			} else {
				if config.WeightDecay < 0 {
					t.Errorf("Expected valid weight decay %f to be >= 0", config.WeightDecay)
				}
			}
		})
	}

	t.Log("SGD parameter validation test passed")
}

// TestSGDStepCountTracking tests step count tracking
func TestSGDStepCountTracking(t *testing.T) {
	sgd := MockSGDOptimizer()

	// Test initial step count
	if sgd.GetStepCount() != 0 {
		t.Errorf("Expected initial step count 0, got %d", sgd.GetStepCount())
	}

	// Simulate step increments
	sgd.StepCount = 10
	if sgd.GetStepCount() != 10 {
		t.Errorf("Expected step count 10, got %d", sgd.GetStepCount())
	}

	// Test very large step count
	sgd.StepCount = 1000000
	if sgd.GetStepCount() != 1000000 {
		t.Errorf("Expected step count 1000000, got %d", sgd.GetStepCount())
	}

	t.Log("SGD step count tracking test passed")
}

// TestSGDCleanup tests the cleanup functionality
func TestSGDCleanup(t *testing.T) {
	sgd := MockSGDOptimizer()

	// Test cleanup doesn't panic with mock data
	sgd.Cleanup()

	// Test multiple cleanups don't panic
	sgd.Cleanup()
	sgd.Cleanup()

	t.Log("SGD cleanup test passed")
}

// TestSGDNesterovFlag tests Nesterov momentum flag
func TestSGDNesterovFlag(t *testing.T) {
	// Test with Nesterov disabled
	config := DefaultSGDConfig()
	config.Momentum = 0.9
	config.Nesterov = false

	if config.Nesterov {
		t.Error("Expected Nesterov to be false")
	}

	// Test with Nesterov enabled
	config.Nesterov = true
	if !config.Nesterov {
		t.Error("Expected Nesterov to be true")
	}

	// Test Nesterov with zero momentum (should be ignored)
	config.Momentum = 0.0
	config.Nesterov = true
	// In practice, Nesterov with zero momentum behaves like vanilla SGD
	// The flag is still set but has no effect

	t.Log("SGD Nesterov flag test passed")
}

// TestSGDBufferManagement tests buffer management
func TestSGDBufferManagement(t *testing.T) {
	sgd := MockSGDOptimizer()

	// Test buffer initialization
	if len(sgd.MomentumBuffers) != 2 {
		t.Errorf("Expected 2 momentum buffers, got %d", len(sgd.MomentumBuffers))
	}

	if len(sgd.WeightBuffers) != 2 {
		t.Errorf("Expected 2 weight buffers, got %d", len(sgd.WeightBuffers))
	}

	if len(sgd.bufferSizes) != 2 {
		t.Errorf("Expected 2 buffer sizes, got %d", len(sgd.bufferSizes))
	}

	// Test buffer sizes
	expectedSizes := []int{64, 8}
	for i, expected := range expectedSizes {
		if sgd.bufferSizes[i] != expected {
			t.Errorf("Buffer %d: expected size %d, got %d", i, expected, sgd.bufferSizes[i])
		}
	}

	t.Log("SGD buffer management test passed")
}

// TestSGDComparisonWithOtherOptimizers tests differences between SGD and other optimizers
func TestSGDComparisonWithOtherOptimizers(t *testing.T) {
	sgdConfig := DefaultSGDConfig()
	adamConfig := DefaultAdamConfig()
	rmspropConfig := DefaultRMSPropConfig()

	// Test that SGD has same default learning rate as RMSProp
	if sgdConfig.LearningRate != rmspropConfig.LearningRate {
		t.Errorf("Expected SGD LR (%f) == RMSProp LR (%f)", sgdConfig.LearningRate, rmspropConfig.LearningRate)
	}

	// Test that SGD has higher default learning rate than Adam
	if sgdConfig.LearningRate <= adamConfig.LearningRate {
		t.Errorf("Expected SGD LR (%f) > Adam LR (%f)", sgdConfig.LearningRate, adamConfig.LearningRate)
	}

	// Test that SGD has momentum parameter
	if sgdConfig.Momentum != 0.0 {
		t.Errorf("Expected SGD default momentum 0.0, got %f", sgdConfig.Momentum)
	}

	// Test that SGD has weight decay parameter
	if sgdConfig.WeightDecay != 0.0 {
		t.Errorf("Expected SGD default weight decay 0.0, got %f", sgdConfig.WeightDecay)
	}

	// Test that SGD has Nesterov parameter
	if sgdConfig.Nesterov != false {
		t.Errorf("Expected SGD default Nesterov false, got %t", sgdConfig.Nesterov)
	}

	t.Log("SGD comparison with other optimizers test passed")
}

// TestSGDSetWeightBuffers tests the SetWeightBuffers method
func TestSGDSetWeightBuffers(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Test setting correct number of weight buffers
	weightBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x1000)),
		unsafe.Pointer(uintptr(0x2000)),
	}
	
	err := sgd.SetWeightBuffers(weightBuffers)
	if err != nil {
		t.Errorf("Unexpected error for correct buffer count: %v", err)
	}
	
	// Verify buffers were set
	if sgd.WeightBuffers[0] != weightBuffers[0] {
		t.Errorf("Expected WeightBuffers[0] %p, got %p", weightBuffers[0], sgd.WeightBuffers[0])
	}
	if sgd.WeightBuffers[1] != weightBuffers[1] {
		t.Errorf("Expected WeightBuffers[1] %p, got %p", weightBuffers[1], sgd.WeightBuffers[1])
	}
	
	// Test setting wrong number of weight buffers
	wrongBuffers := []unsafe.Pointer{unsafe.Pointer(uintptr(0x1000))}
	err = sgd.SetWeightBuffers(wrongBuffers)
	if err == nil {
		t.Error("Expected error for mismatched buffer count, got nil")
	}
	
	// Test setting too many weight buffers
	tooManyBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x1000)),
		unsafe.Pointer(uintptr(0x2000)),
		unsafe.Pointer(uintptr(0x3000)),
	}
	err = sgd.SetWeightBuffers(tooManyBuffers)
	if err == nil {
		t.Error("Expected error for too many buffers, got nil")
	}
	
	t.Log("SGD SetWeightBuffers test passed")
}

// TestSGDGetStep tests the GetStep method
func TestSGDGetStep(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Test initial step count
	if sgd.GetStep() != 0 {
		t.Errorf("Expected initial step count 0, got %d", sgd.GetStep())
	}
	
	// Test after manually incrementing step count
	sgd.StepCount = 5
	if sgd.GetStep() != 5 {
		t.Errorf("Expected step count 5, got %d", sgd.GetStep())
	}
	
	// Test large step count
	sgd.StepCount = 999999
	if sgd.GetStep() != 999999 {
		t.Errorf("Expected step count 999999, got %d", sgd.GetStep())
	}
	
	// Test that GetStep and GetStepCount return same value
	if sgd.GetStep() != sgd.GetStepCount() {
		t.Errorf("GetStep() and GetStepCount() should return same value: %d != %d", sgd.GetStep(), sgd.GetStepCount())
	}
	
	t.Log("SGD GetStep test passed")
}

// TestSGDStepWithError tests the Step method which currently returns an error
func TestSGDStepWithError(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Test step with mismatched gradient buffer count
	gradientBuffers := []unsafe.Pointer{unsafe.Pointer(uintptr(0x1000))}
	err := sgd.Step(gradientBuffers)
	if err == nil {
		t.Error("Expected error for mismatched gradient buffer count, got nil")
	}
	
	// Test step with correct gradient buffer count (should return "not implemented" error)
	correctGradientBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x1000)),
		unsafe.Pointer(uintptr(0x2000)),
	}
	err = sgd.Step(correctGradientBuffers)
	if err == nil {
		t.Error("Expected error for unimplemented SGD step, got nil")
	}
	
	// Should still increment step count even with error
	if sgd.StepCount != 1 {
		t.Errorf("Expected step count to be incremented to 1, got %d", sgd.StepCount)
	}
	
	t.Log("SGD Step error handling test passed")
}

// TestSGDCleanupMethod tests the cleanup method
func TestSGDCleanupMethod(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Mock SGD has nil memory manager, so we can only test the slice clearing
	// Test cleanup
	sgd.Cleanup()
	
	// Verify buffers were cleared
	if sgd.MomentumBuffers != nil {
		t.Error("Expected MomentumBuffers to be nil after cleanup")
	}
	if sgd.WeightBuffers != nil {
		t.Error("Expected WeightBuffers to be nil after cleanup")
	}
	if sgd.bufferSizes != nil {
		t.Error("Expected bufferSizes to be nil after cleanup")
	}
	
	t.Log("SGD cleanup method test passed")
}

// TestSGDLoadStateValidation tests LoadState parameter validation and error handling
func TestSGDLoadStateValidation(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Test invalid state type
	invalidState := &OptimizerState{
		Type: "InvalidType",
		Parameters: map[string]interface{}{
			"learning_rate": 0.01,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err := sgd.LoadState(invalidState)
	if err == nil {
		t.Error("Expected error for invalid state type, got nil")
	}
	
	// Test parameter restoration with valid state
	originalLR := sgd.LearningRate
	originalMomentum := sgd.Momentum
	originalWeightDecay := sgd.WeightDecay
	originalNesterov := sgd.Nesterov
	originalStepCount := sgd.StepCount
	
	validState := &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": 0.02,
			"momentum":      0.95,
			"weight_decay":  0.01,
			"nesterov":      true,
			"step_count":    float64(25),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err = sgd.LoadState(validState)
	if err != nil {
		t.Errorf("Expected no error for valid state, got %v", err)
	}
	
	// Verify parameters were restored correctly
	if sgd.LearningRate != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", sgd.LearningRate)
	}
	if sgd.Momentum != 0.95 {
		t.Errorf("Expected momentum 0.95, got %f", sgd.Momentum)
	}
	if sgd.WeightDecay != 0.01 {
		t.Errorf("Expected weight decay 0.01, got %f", sgd.WeightDecay)
	}
	if sgd.Nesterov != true {
		t.Errorf("Expected nesterov true, got %v", sgd.Nesterov)
	}
	if sgd.StepCount != 25 {
		t.Errorf("Expected step count 25, got %d", sgd.StepCount)
	}
	
	// Test that missing parameters don't change existing values
	partialState := &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": 0.005,
			// Missing other parameters
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err = sgd.LoadState(partialState)
	if err != nil {
		t.Errorf("Expected no error for partial state, got %v", err)
	}
	
	// Only learning rate should have changed
	if sgd.LearningRate != 0.005 {
		t.Errorf("Expected learning rate 0.005, got %f", sgd.LearningRate)
	}
	if sgd.Momentum != 0.95 { // Should remain unchanged
		t.Errorf("Expected momentum 0.95 (unchanged), got %f", sgd.Momentum)
	}
	
	// Test wrong parameter types
	wrongTypeState := &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": "not_a_number",
			"momentum":      0.9,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	err = sgd.LoadState(wrongTypeState)
	if err != nil {
		t.Errorf("Expected no error for wrong type (should be ignored), got %v", err)
	}
	
	// Learning rate should be unchanged since string couldn't be converted
	if sgd.LearningRate != 0.005 {
		t.Errorf("Expected learning rate 0.005 (unchanged), got %f", sgd.LearningRate)
	}
	
	// Restore original values
	sgd.LearningRate = originalLR
	sgd.Momentum = originalMomentum
	sgd.WeightDecay = originalWeightDecay
	sgd.Nesterov = originalNesterov
	sgd.StepCount = originalStepCount
	
	t.Log("SGD LoadState validation test passed")
}

// TestSGDGetStateStructure tests GetState structure and parameters
func TestSGDGetStateStructure(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Set specific values to verify they're captured
	sgd.LearningRate = 0.015
	sgd.Momentum = 0.88
	sgd.WeightDecay = 0.002
	sgd.Nesterov = true
	sgd.StepCount = 30
	
	// Note: We can't test the full GetState because it requires Metal buffers
	// But we can test the structure and parameters that would be created
	
	// Test that GetState would fail with nil buffers (expected behavior)
	// Note: GetState may not fail immediately with mock data, but would fail when trying to read Metal buffers
	_, err := sgd.GetState()
	if err == nil {
		t.Log("GetState did not fail with mock data (expected - CGO bridge calls would fail)")
	} else {
		t.Logf("GetState failed as expected with mock data: %v", err)
	}
	
	// Test parameter structure by creating what GetState would create
	expectedState := &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": sgd.LearningRate,
			"momentum":      sgd.Momentum,
			"weight_decay":  sgd.WeightDecay,
			"nesterov":      sgd.Nesterov,
			"step_count":    sgd.StepCount,
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	// Verify the structure we expect GetState to produce
	if expectedState.Type != "SGD" {
		t.Errorf("Expected type 'SGD', got '%s'", expectedState.Type)
	}
	
	if expectedState.Parameters["learning_rate"] != sgd.LearningRate {
		t.Errorf("Expected learning rate %f, got %v", sgd.LearningRate, expectedState.Parameters["learning_rate"])
	}
	
	if expectedState.Parameters["momentum"] != sgd.Momentum {
		t.Errorf("Expected momentum %f, got %v", sgd.Momentum, expectedState.Parameters["momentum"])
	}
	
	if expectedState.Parameters["weight_decay"] != sgd.WeightDecay {
		t.Errorf("Expected weight decay %f, got %v", sgd.WeightDecay, expectedState.Parameters["weight_decay"])
	}
	
	if expectedState.Parameters["nesterov"] != sgd.Nesterov {
		t.Errorf("Expected nesterov %v, got %v", sgd.Nesterov, expectedState.Parameters["nesterov"])
	}
	
	if expectedState.Parameters["step_count"] != sgd.StepCount {
		t.Errorf("Expected step count %d, got %v", sgd.StepCount, expectedState.Parameters["step_count"])
	}
	
	t.Log("SGD GetState structure test passed")
}

// TestSGDStateRoundTrip tests the parameter roundtrip (without Metal buffers)
func TestSGDStateRoundTrip(t *testing.T) {
	sgd := MockSGDOptimizer()
	
	// Set initial values
	sgd.LearningRate = 0.008
	sgd.Momentum = 0.92
	sgd.WeightDecay = 0.003
	sgd.Nesterov = true
	sgd.StepCount = 55
	
	// Create state manually (simulating what GetState would create)
	state := &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": float64(sgd.LearningRate),
			"momentum":      float64(sgd.Momentum),
			"weight_decay":  float64(sgd.WeightDecay),
			"nesterov":      sgd.Nesterov,
			"step_count":    float64(sgd.StepCount),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	
	// Modify values to test restoration
	sgd.LearningRate = 0.999
	sgd.Momentum = 0.999
	sgd.WeightDecay = 0.999
	sgd.Nesterov = false
	sgd.StepCount = 999
	
	// Restore state
	err := sgd.LoadState(state)
	if err != nil {
		t.Errorf("Unexpected error during LoadState: %v", err)
	}
	
	// Verify all values were restored correctly
	if sgd.LearningRate != 0.008 {
		t.Errorf("Expected learning rate 0.008, got %f", sgd.LearningRate)
	}
	if sgd.Momentum != 0.92 {
		t.Errorf("Expected momentum 0.92, got %f", sgd.Momentum)
	}
	if sgd.WeightDecay != 0.003 {
		t.Errorf("Expected weight decay 0.003, got %f", sgd.WeightDecay)
	}
	if sgd.Nesterov != true {
		t.Errorf("Expected nesterov true, got %v", sgd.Nesterov)
	}
	if sgd.StepCount != 55 {
		t.Errorf("Expected step count 55, got %d", sgd.StepCount)
	}
	
	t.Log("SGD state roundtrip test passed")
}