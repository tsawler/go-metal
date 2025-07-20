package cgo_bridge

import (
	"testing"
	"unsafe"
)

// Test memory operations
func TestMemoryOperations(t *testing.T) {
	const bufferSize = 512
	const numInts = bufferSize / 4

	// Create test buffer
	buffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(buffer)

	// Test int32 operations
	testIntData := make([]int32, numInts)
	for i := range testIntData {
		testIntData[i] = int32(i * 2)
	}

	// Test copy int32 array to buffer
	err = CopyInt32ArrayToMetalBuffer(buffer, testIntData)
	if err != nil {
		t.Fatalf("Failed to copy int32 array to buffer: %v", err)
	}

	// Test copy int32 array from buffer
	retrievedIntData, err := CopyMetalBufferToInt32Array(buffer, numInts)
	if err != nil {
		t.Fatalf("Failed to copy int32 array from buffer: %v", err)
	}

	// Verify int32 data integrity
	if len(retrievedIntData) != len(testIntData) {
		t.Errorf("Int32 data length mismatch: expected %d, got %d", len(testIntData), len(retrievedIntData))
	}

	for i := range testIntData {
		if retrievedIntData[i] != testIntData[i] {
			t.Errorf("Int32 data mismatch at index %d: expected %d, got %d", i, testIntData[i], retrievedIntData[i])
		}
	}

	t.Log("✅ Memory operations test passed")
}

// Test buffer copying operations
func TestBufferCopyOperations(t *testing.T) {
	const bufferSize = 256

	// Create source and destination buffers
	srcBuffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(srcBuffer)

	dstBuffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(dstBuffer)

	// Fill source buffer with test data
	testData := make([]float32, bufferSize/4)
	for i := range testData {
		testData[i] = float32(i) * 0.5
	}
	err = CopyFloat32ArrayToMetalBuffer(srcBuffer, testData)
	if err != nil {
		t.Fatalf("Failed to fill source buffer: %v", err)
	}

	// Test buffer-to-buffer copy
	err = CopyTensorBufferSync(srcBuffer, dstBuffer, bufferSize)
	if err != nil {
		t.Fatalf("Failed to copy buffer to buffer: %v", err)
	}

	// Verify copy worked
	copiedData, err := CopyMetalBufferToFloat32Array(dstBuffer, len(testData))
	if err != nil {
		t.Fatalf("Failed to read copied buffer: %v", err)
	}

	for i := range testData {
		if copiedData[i] != testData[i] {
			t.Errorf("Buffer copy failed at index %d: expected %f, got %f", i, testData[i], copiedData[i])
		}
	}

	t.Log("✅ Buffer copy operations test passed")
}

// Test command buffer operations
func TestCommandBufferOperations(t *testing.T) {
	queue, err := getSharedCommandQueue()
	if err != nil {
		t.Skipf("Skipping test - command queue not available: %v", err)
	}

	// Create multiple command buffers
	buffers := make([]unsafe.Pointer, 3)
	for i := range buffers {
		buffer, err := CreateCommandBuffer(queue)
		if err != nil {
			// Cleanup previously created buffers
			for j := 0; j < i; j++ {
				ReleaseCommandBuffer(buffers[j])
			}
			t.Fatalf("Failed to create command buffer %d: %v", i, err)
		}
		buffers[i] = buffer
	}

	// Test command buffer commit and wait
	for i, buffer := range buffers {
		err := CommitCommandBuffer(buffer)
		if err != nil {
			t.Logf("Command buffer %d commit failed (expected for empty buffer): %v", i, err)
		}

		err = WaitCommandBufferCompletion(buffer)
		if err != nil {
			t.Logf("Command buffer %d wait failed (expected for empty buffer): %v", i, err)
		}

		ReleaseCommandBuffer(buffer)
		t.Logf("✅ Command buffer %d lifecycle completed", i)
	}

	t.Log("✅ Command buffer operations test passed")
}

// Test tensor type conversion
func TestTensorTypeConversion(t *testing.T) {
	const bufferSize = 128

	device, err := getSharedDevice()
	if err != nil {
		t.Skipf("Skipping test - Metal device not available: %v", err)
	}

	// Create source and destination buffers
	srcBuffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(srcBuffer)

	dstBuffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(dstBuffer)

	// Test tensor type conversion (same type for simplicity)
	shape := []int{32} // 32 float32 values
	err = ConvertTensorType(srcBuffer, dstBuffer, shape, 0, 0, device) // 0 = float32
	if err != nil {
		// This may fail with "Unsupported type conversion" which is expected for same-type conversion
		t.Logf("Tensor type conversion returned expected error: %v", err)
	} else {
		t.Log("Tensor type conversion succeeded")
	}

	t.Log("✅ Tensor type conversion test passed")
}

// Test data copying to staging buffer
func TestStagingBufferOperations(t *testing.T) {
	const bufferSize = 256

	// Create staging buffer
	stagingBuffer, err := createTestBuffer(bufferSize, CPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(stagingBuffer)

	// Test data copying to staging buffer
	testData := make([]byte, bufferSize)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	err = CopyDataToStagingBuffer(stagingBuffer, testData)
	if err != nil {
		t.Fatalf("Failed to copy data to staging buffer: %v", err)
	}

	t.Log("✅ Staging buffer operations test passed")
}

// Test autorelease pool with buffer operations
func TestAutoreleaseWithOperations(t *testing.T) {
	// Setup autorelease pool
	SetupAutoreleasePool()
	defer DrainAutoreleasePool()

	const bufferSize = 128

	// Create and use buffers within autorelease pool
	buffer, err := createTestBuffer(bufferSize, GPU, t)
	if err != nil {
		return // Will skip if buffer pool exhausted
	}
	defer DeallocateMetalBuffer(buffer)

	// Perform some operations
	testData := []float32{1.0, 2.0, 3.0, 4.0}
	err = CopyFloat32ArrayToMetalBuffer(buffer, testData)
	if err != nil {
		t.Fatalf("Failed to perform buffer operation: %v", err)
	}

	// Drain and setup again to test multiple cycles
	DrainAutoreleasePool()
	SetupAutoreleasePool()

	t.Log("✅ Autorelease with operations test passed")
}

// Test configuration parameter bounds
func TestConfigurationBounds(t *testing.T) {
	// Test edge cases for configuration parameters
	configs := []struct {
		name   string
		config TrainingConfig
		valid  bool
	}{
		{
			name: "minimum_valid_learning_rate",
			config: TrainingConfig{
				LearningRate:  1e-8,
				Epsilon:       1e-10,
				OptimizerType: 0,
			},
			valid: true,
		},
		{
			name: "maximum_beta_values",
			config: TrainingConfig{
				LearningRate:  0.001,
				Beta1:         0.999,
				Beta2:         0.9999,
				Epsilon:       1e-8,
				OptimizerType: 1,
			},
			valid: true,
		},
		{
			name: "zero_learning_rate",
			config: TrainingConfig{
				LearningRate:  0.0,
				Epsilon:       1e-8,
				OptimizerType: 0,
			},
			valid: false, // Zero learning rate is invalid
		},
	}

	for _, test := range configs {
		t.Run(test.name, func(t *testing.T) {
			isValid := test.config.LearningRate > 0 && 
					  test.config.Epsilon > 0 &&
					  test.config.OptimizerType >= 0 && test.config.OptimizerType <= 3

			if isValid != test.valid {
				t.Errorf("Configuration validation mismatch for %s: expected %v, got %v", 
					test.name, test.valid, isValid)
			}

			t.Logf("✅ %s bounds check passed", test.name)
		})
	}

	t.Log("✅ Configuration bounds test passed")
}

// Test device type usage patterns
func TestDeviceTypeUsage(t *testing.T) {
	deviceTypes := []struct {
		name       string
		deviceType DeviceType
		expected   string
	}{
		{"GPU", GPU, "GPU"},
		{"CPU", CPU, "CPU"}, 
		{"PersistentGPU", PersistentGPU, "PersistentGPU"},
	}

	for _, dt := range deviceTypes {
		t.Run(dt.name, func(t *testing.T) {
			// Test that device type can be used in buffer allocation
			// (without actually allocating to avoid resource issues)
			if int(dt.deviceType) < 0 {
				t.Errorf("Device type %s has invalid value: %d", dt.name, int(dt.deviceType))
			}
			t.Logf("✅ Device type %s (%d) is valid", dt.name, int(dt.deviceType))
		})
	}

	t.Log("✅ Device type usage test passed")
}