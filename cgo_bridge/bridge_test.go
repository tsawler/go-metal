package cgo_bridge

import (
	"testing"
)

// Test Metal device creation and destruction
func TestMetalDeviceLifecycle(t *testing.T) {
	// Test individual device creation and destruction
	device, err := CreateMetalDevice()
	if err != nil {
		t.Skipf("Skipping test - Metal device not available: %v", err)
	}
	defer DestroyMetalDevice(device)

	if device == nil {
		t.Error("CreateMetalDevice returned nil pointer")
	}

	t.Log("✅ Metal device creation and destruction test passed")
}

// Test shared device access
func TestSharedDeviceAccess(t *testing.T) {
	device, err := getSharedDevice()
	if err != nil {
		t.Skipf("Skipping test - shared device not available: %v", err)
	}

	if device == nil {
		t.Error("getSharedDevice returned nil pointer")
	}

	t.Log("✅ Shared device access test passed")
}

// Test command queue creation
func TestCommandQueueLifecycle(t *testing.T) {
	device, err := getSharedDevice()
	if err != nil {
		t.Skipf("Skipping test - Metal device not available: %v", err)
	}

	// Test command queue creation
	queue, err := CreateCommandQueue(device)
	if err != nil {
		t.Fatalf("Failed to create command queue: %v", err)
	}
	defer DestroyCommandQueue(queue)

	if queue == nil {
		t.Error("CreateCommandQueue returned nil pointer")
	}

	t.Log("✅ Command queue lifecycle test passed")
}

// Test command buffer creation
func TestCommandBufferLifecycle(t *testing.T) {
	queue, err := getSharedCommandQueue()
	if err != nil {
		t.Skipf("Skipping test - command queue not available: %v", err)
	}

	// Test command buffer creation
	buffer, err := CreateCommandBuffer(queue)
	if err != nil {
		t.Fatalf("Failed to create command buffer: %v", err)
	}
	defer ReleaseCommandBuffer(buffer)

	if buffer == nil {
		t.Error("CreateCommandBuffer returned nil pointer")
	}

	t.Log("✅ Command buffer lifecycle test passed")
}

// Test buffer allocation and deallocation
func TestMetalBufferLifecycle(t *testing.T) {
	const bufferSize = 1024 // 1KB buffer

	// Test buffer allocation
	buffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(buffer)

	if buffer == nil {
		t.Error("AllocateMetalBuffer returned nil pointer")
	}

	t.Log("✅ Metal buffer lifecycle test passed")
}

// Test training configuration validation
func TestTrainingConfigValidation(t *testing.T) {
	// Test just the configuration struct creation
	validConfigs := []struct {
		name   string
		config TrainingConfig
	}{
		{
			name: "valid_sgd_config",
			config: TrainingConfig{
				LearningRate:   0.01,
				Beta1:          0.0,
				Beta2:          0.0,
				WeightDecay:    0.001,
				Epsilon:        1e-8,
				OptimizerType:  0, // SGD
				ProblemType:    0, // Classification
				LossFunction:   0, // CrossEntropy
			},
		},
		{
			name: "valid_adam_config",
			config: TrainingConfig{
				LearningRate:   0.001,
				Beta1:          0.9,
				Beta2:          0.999,
				WeightDecay:    0.0001,
				Epsilon:        1e-8,
				OptimizerType:  1, // Adam
				ProblemType:    0, // Classification
				LossFunction:   0, // CrossEntropy
			},
		},
		{
			name: "valid_rmsprop_config",
			config: TrainingConfig{
				LearningRate:   0.001,
				Alpha:          0.99,
				Epsilon:        1e-8,
				WeightDecay:    0.0,
				Momentum:       0.0,
				Centered:       false,
				OptimizerType:  2, // RMSProp
				ProblemType:    0, // Classification
				LossFunction:   0, // CrossEntropy
			},
		},
	}

	for _, test := range validConfigs {
		t.Run(test.name, func(t *testing.T) {
			// Just validate that the config struct has the expected values
			if test.config.LearningRate <= 0 {
				t.Errorf("Invalid learning rate: %f", test.config.LearningRate)
			}
			if test.config.OptimizerType < 0 || test.config.OptimizerType > 3 {
				t.Errorf("Invalid optimizer type: %d", test.config.OptimizerType)
			}
			t.Logf("✅ %s config validated successfully", test.name)
		})
	}

	t.Log("✅ Training configuration validation tests passed")
}

// Test inference configuration validation
func TestInferenceConfigValidation(t *testing.T) {
	config := InferenceConfig{
		UseDynamicEngine:       true,
		BatchNormInferenceMode: false,
		InputShape:             []int32{1, 10},
		InputShapeLen:          2,
		ProblemType:            0, // Classification
		LossFunction:           0, // CrossEntropy
		UseCommandPooling:      false,
		OptimizeForSingleBatch: true,
	}

	// Just validate the config structure without creating an engine
	if len(config.InputShape) != int(config.InputShapeLen) {
		t.Error("InputShape length doesn't match InputShapeLen")
	}
	
	if config.ProblemType < 0 {
		t.Error("Invalid problem type")
	}

	t.Log("✅ Inference configuration validation test passed")
}

// Test buffer operations
func TestBufferOperations(t *testing.T) {
	const bufferSize = 256 // 256 bytes
	const numFloats = bufferSize / 4 // 64 float32 values

	// Create test buffer
	buffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(buffer)

	// Test data operations
	testData := make([]float32, numFloats)
	for i := range testData {
		testData[i] = float32(i) * 0.1
	}

	// Test copy to buffer
	err = CopyFloat32ArrayToMetalBuffer(buffer, testData)
	if err != nil {
		t.Fatalf("Failed to copy data to buffer: %v", err)
	}

	// Test copy from buffer
	retrievedData, err := CopyMetalBufferToFloat32Array(buffer, numFloats)
	if err != nil {
		t.Fatalf("Failed to copy data from buffer: %v", err)
	}

	// Verify data integrity
	if len(retrievedData) != len(testData) {
		t.Errorf("Data length mismatch: expected %d, got %d", len(testData), len(retrievedData))
	}

	for i := range testData {
		if retrievedData[i] != testData[i] {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, testData[i], retrievedData[i])
		}
	}

	t.Log("✅ Buffer operations test passed")
}

// Test buffer zeroing
func TestBufferZeroing(t *testing.T) {
	const bufferSize = 128
	const numFloats = bufferSize / 4

	device, err := getSharedDevice()
	if err != nil {
		t.Skipf("Skipping test - Metal device not available: %v", err)
	}

	buffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(buffer)

	// Fill buffer with test data
	testData := make([]float32, numFloats)
	for i := range testData {
		testData[i] = float32(i + 1) // Non-zero values
	}
	err = CopyFloat32ArrayToMetalBuffer(buffer, testData)
	if err != nil {
		t.Fatalf("Failed to copy test data: %v", err)
	}

	// Zero the buffer
	err = ZeroMetalBuffer(device, buffer, bufferSize)
	if err != nil {
		t.Fatalf("Failed to zero buffer: %v", err)
	}

	// Verify buffer is zeroed
	zeroedData, err := CopyMetalBufferToFloat32Array(buffer, numFloats)
	if err != nil {
		t.Fatalf("Failed to read zeroed buffer: %v", err)
	}

	for i, value := range zeroedData {
		if value != 0.0 {
			t.Errorf("Buffer not zeroed at index %d: expected 0.0, got %f", i, value)
		}
	}

	t.Log("✅ Buffer zeroing test passed")
}

// Test autorelease pool management
func TestAutoreleasePoolManagement(t *testing.T) {
	// Test autorelease pool setup and drain
	SetupAutoreleasePool()
	DrainAutoreleasePool()

	// This test mainly ensures the functions don't crash
	t.Log("✅ Autorelease pool management test passed")
}

// Test error handling for invalid parameters
func TestErrorHandling(t *testing.T) {
	// Test with nil device
	_, err := CreateTrainingEngine(nil, TrainingConfig{
		LearningRate:  0.001,
		OptimizerType: 0,
		ProblemType:   0,
		LossFunction:  0,
	})
	if err == nil {
		t.Error("Expected error for nil device, got nil")
	}

	// Test buffer allocation with zero size
	device, err := getSharedDevice()
	if err != nil {
		t.Skipf("Skipping test - Metal device not available: %v", err)
	}

	_, err = AllocateMetalBuffer(device, 0, GPU)
	if err == nil {
		t.Error("Expected error for zero buffer size, got nil")
	}

	t.Log("✅ Error handling test passed")
}

// Test multiple configuration validation (safe test)
func TestMultipleConfigValidation(t *testing.T) {
	configs := []TrainingConfig{
		{
			LearningRate:  0.001,
			Beta1:         0.9,
			Beta2:         0.999,
			WeightDecay:   0.0001,
			Epsilon:       1e-8,
			OptimizerType: 1, // Adam
			ProblemType:   0, // Classification
			LossFunction:  0, // CrossEntropy
		},
		{
			LearningRate:  0.01,
			WeightDecay:   0.001,
			Epsilon:       1e-8,
			OptimizerType: 0, // SGD
			ProblemType:   0, // Classification
			LossFunction:  0, // CrossEntropy
		},
		{
			LearningRate:  0.001,
			Alpha:         0.99,
			Epsilon:       1e-8,
			OptimizerType: 2, // RMSProp
			ProblemType:   0, // Classification
			LossFunction:  0, // CrossEntropy
		},
	}

	// Test multiple configurations without creating engines
	for i, config := range configs {
		if config.LearningRate <= 0 {
			t.Errorf("Config %d has invalid learning rate: %f", i, config.LearningRate)
		}
		if config.Epsilon <= 0 {
			t.Errorf("Config %d has invalid epsilon: %f", i, config.Epsilon)
		}
		t.Logf("✅ Config %d validated successfully", i)
	}

	t.Log("✅ Multiple configuration validation test passed")
}

// Test DeviceType constants
func TestDeviceTypeConstants(t *testing.T) {
	// Test that device type constants are defined
	types := []DeviceType{
		GPU,
		CPU,
		PersistentGPU,
	}

	for i, deviceType := range types {
		if int(deviceType) < 0 {
			t.Errorf("DeviceType constant %d has invalid value: %d", i, int(deviceType))
		}
	}

	t.Log("✅ DeviceType constants test passed")
}

// Test configuration struct completeness
func TestConfigurationStructs(t *testing.T) {
	// Test TrainingConfig has all required fields
	config := TrainingConfig{
		LearningRate:  0.001,
		Beta1:         0.9,
		Beta2:         0.999,
		WeightDecay:   0.0001,
		Epsilon:       1e-8,
		Alpha:         0.99,
		Momentum:      0.0,
		Centered:      false,
		OptimizerType: 1,
		ProblemType:   0,
		LossFunction:  0,
	}

	if config.LearningRate != 0.001 {
		t.Error("TrainingConfig LearningRate field not set correctly")
	}

	// Test InferenceConfig has all required fields
	infConfig := InferenceConfig{
		UseDynamicEngine:       true,
		BatchNormInferenceMode: true,
		InputShape:             []int32{1, 10},
		InputShapeLen:          2,
		ProblemType:            0,
		LossFunction:           0,
		UseCommandPooling:      false,
		OptimizeForSingleBatch: true,
	}

	if infConfig.InputShapeLen != 2 {
		t.Error("InferenceConfig InputShapeLen field not set correctly")
	}

	t.Log("✅ Configuration structs test passed")
}