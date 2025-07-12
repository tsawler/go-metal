package memory

import (
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

// Setup mock functions for testing
func init() {
	// Initialize mock bridge functions for testing
	SetupBridge(
		mockToFloat32Slice,
		mockCopyFloat32Data,
		mockCopyInt32Data,
	)
	
	SetupBridgeWithConvert(
		mockToFloat32Slice,
		mockCopyFloat32Data,
		mockCopyInt32Data,
		mockConvertTensorType,
		mockCopyTensor,
	)
}

// Mock bridge functions for testing
func mockToFloat32Slice(buffer unsafe.Pointer, numElements int) ([]float32, error) {
	// Return a mock float32 slice filled with test data
	result := make([]float32, numElements)
	for i := range result {
		result[i] = float32(i) * 0.1
	}
	return result, nil
}

func mockCopyFloat32Data(buffer unsafe.Pointer, data []float32) error {
	// Mock copy operation - just return success
	return nil
}

func mockCopyInt32Data(buffer unsafe.Pointer, data []int32) error {
	// Mock copy operation - just return success
	return nil
}

func mockConvertTensorType(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType int) error {
	// Mock conversion operation - just return success
	return nil
}

func mockCopyTensor(srcBuffer, dstBuffer unsafe.Pointer, size int) error {
	// Mock tensor copy operation - just return success
	return nil
}

// TestDataType tests DataType constants and behavior
func TestDataType(t *testing.T) {
	// Test DataType values
	if Float32 != 0 {
		t.Errorf("Expected Float32 to be 0, got %d", Float32)
	}
	if Int32 != 1 {
		t.Errorf("Expected Int32 to be 1, got %d", Int32)
	}
	if Float16 != 2 {
		t.Errorf("Expected Float16 to be 2, got %d", Float16)
	}
	if Int8 != 3 {
		t.Errorf("Expected Int8 to be 3, got %d", Int8)
	}
	
	t.Log("DataType constants tests passed")
}

// TestDeviceType tests DeviceType constants and behavior
func TestDeviceType(t *testing.T) {
	// Test DeviceType values
	if CPU != 0 {
		t.Errorf("Expected CPU to be 0, got %d", CPU)
	}
	if GPU != 1 {
		t.Errorf("Expected GPU to be 1, got %d", GPU)
	}
	if PersistentGPU != 2 {
		t.Errorf("Expected PersistentGPU to be 2, got %d", PersistentGPU)
	}
	
	t.Log("DeviceType constants tests passed")
}

// TestCalculateSize tests the calculateSize function
func TestCalculateSize(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		dtype    DataType
		expected int
	}{
		{"empty_shape", []int{}, Float32, 0},
		{"scalar", []int{1}, Float32, 4},
		{"vector", []int{10}, Float32, 40},
		{"matrix", []int{3, 4}, Float32, 48},
		{"tensor_3d", []int{2, 3, 4}, Float32, 96},
		{"int32_vector", []int{10}, Int32, 40},
		{"float16_vector", []int{10}, Float16, 20},
		{"int8_vector", []int{10}, Int8, 10},
	}
	
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := calculateSize(test.shape, test.dtype)
			if result != test.expected {
				t.Errorf("calculateSize(%v, %d) = %d; expected %d", 
					test.shape, test.dtype, result, test.expected)
			}
		})
	}
	
	t.Log("calculateSize tests passed")
}

// TestCalculateSizePanic tests panic behavior for unsupported data types
func TestCalculateSizePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for unsupported data type")
		} else {
			t.Log("Correctly panicked for unsupported data type")
		}
	}()
	
	// This should panic
	calculateSize([]int{10}, DataType(99))
}

// TestMemoryManagerCreation tests memory manager creation
func TestMemoryManagerCreation(t *testing.T) {
	device := CreateMockDevice()
	mm := NewMemoryManager(device)
	
	if mm == nil {
		t.Fatal("Memory manager should not be nil")
	}
	
	if mm.device != device {
		t.Error("Memory manager should have correct device")
	}
	
	if mm.pools == nil {
		t.Error("Memory manager pools should be initialized")
	}
	
	if mm.bufferSizes == nil {
		t.Error("Memory manager bufferSizes should be initialized")
	}
	
	if len(mm.poolSizes) == 0 {
		t.Error("Memory manager should have default pool sizes")
	}
	
	t.Log("Memory manager creation tests passed")
}

// TestBufferPoolCreation tests buffer pool creation
func TestBufferPoolCreation(t *testing.T) {
	pool := NewBufferPool(1024, 10, GPU)
	
	if pool == nil {
		t.Fatal("Buffer pool should not be nil")
	}
	
	if pool.bufferSize != 1024 {
		t.Errorf("Expected buffer size 1024, got %d", pool.bufferSize)
	}
	
	if pool.maxSize != 10 {
		t.Errorf("Expected max size 10, got %d", pool.maxSize)
	}
	
	if pool.device != GPU {
		t.Errorf("Expected device GPU, got %d", pool.device)
	}
	
	if pool.allocated != 0 {
		t.Errorf("Expected allocated count 0, got %d", pool.allocated)
	}
	
	t.Log("Buffer pool creation tests passed")
}

// TestBufferPoolStats tests buffer pool statistics
func TestBufferPoolStats(t *testing.T) {
	pool := NewBufferPool(1024, 10, GPU)
	
	available, allocated, maxSize := pool.Stats()
	
	if available != 0 {
		t.Errorf("Expected 0 available buffers, got %d", available)
	}
	
	if allocated != 0 {
		t.Errorf("Expected 0 allocated buffers, got %d", allocated)
	}
	
	if maxSize != 10 {
		t.Errorf("Expected max size 10, got %d", maxSize)
	}
	
	t.Log("Buffer pool stats tests passed")
}

// TestMemoryManagerFindPoolSize tests pool size finding logic
func TestMemoryManagerFindPoolSize(t *testing.T) {
	device := CreateMockDevice()
	mm := NewMemoryManager(device)
	
	tests := []struct {
		requested int
		expected  int
	}{
		{100, 1024},      // Should use 1KB pool
		{1024, 1024},     // Exact match
		{5000, 16384},    // Should use 16KB pool
		{100000000, 100000000}, // Larger than largest pool
	}
	
	for _, test := range tests {
		result := mm.findPoolSize(test.requested)
		if result != test.expected {
			t.Errorf("findPoolSize(%d) = %d; expected %d", 
				test.requested, result, test.expected)
		}
	}
	
	t.Log("Memory manager findPoolSize tests passed")
}

// TestCalculateMaxPoolSize tests max pool size calculation
func TestCalculateMaxPoolSize(t *testing.T) {
	tests := []struct {
		bufferSize int
		expected   int
	}{
		{1024, 100},     // <= 4KB
		{4096, 100},     // <= 4KB
		{16384, 50},     // <= 64KB
		{65536, 50},     // <= 64KB
		{262144, 20},    // <= 1MB
		{1048576, 20},   // <= 1MB
		{4194304, 20},   // <= 16MB
		{16777216, 20},  // <= 16MB
		{67108864, 15},  // > 16MB
	}
	
	for _, test := range tests {
		result := calculateMaxPoolSize(test.bufferSize)
		if result != test.expected {
			t.Errorf("calculateMaxPoolSize(%d) = %d; expected %d", 
				test.bufferSize, result, test.expected)
		}
	}
	
	t.Log("calculateMaxPoolSize tests passed")
}

// TestGlobalMemoryManager tests global memory manager functionality
func TestGlobalMemoryManager(t *testing.T) {
	// Reset global state for testing
	globalMemoryManager = nil
	globalMemoryManagerOnce = sync.Once{}
	
	// Test panic when not initialized
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when accessing uninitialized global manager")
		}
	}()
	
	GetGlobalMemoryManager()
}

// TestGlobalMemoryManagerInitialization tests global memory manager initialization
func TestGlobalMemoryManagerInitialization(t *testing.T) {
	// Reset global state for testing
	globalMemoryManager = nil
	globalMemoryManagerOnce = sync.Once{}
	
	device := CreateMockDevice()
	InitializeGlobalMemoryManager(device)
	
	mm := GetGlobalMemoryManager()
	if mm == nil {
		t.Fatal("Global memory manager should not be nil after initialization")
	}
	
	if mm.device != device {
		t.Error("Global memory manager should have correct device")
	}
	
	// Test that subsequent calls return the same instance
	mm2 := GetGlobalMemoryManager()
	if mm != mm2 {
		t.Error("Subsequent calls should return the same instance")
	}
	
	t.Log("Global memory manager initialization tests passed")
}

// TestTensorCreation tests tensor creation with various configurations
func TestTensorCreation(t *testing.T) {
	// Skip tensor creation tests that require actual Metal buffer allocation
	// In a real testing environment, these would be tested with proper Metal mocking
	t.Skip("Tensor creation requires Metal buffer allocation - skipping for unit tests")
	
	tests := []struct {
		name   string
		shape  []int
		dtype  DataType
		device DeviceType
	}{
		{"float32_vector", []int{10}, Float32, GPU},
		{"int32_matrix", []int{3, 4}, Int32, CPU},
		{"float16_tensor", []int{2, 3, 4}, Float16, GPU},
		{"int8_scalar", []int{1}, Int8, GPU},
	}
	
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tensor, err := NewTensor(test.shape, test.dtype, test.device)
			if err != nil {
				t.Fatalf("Failed to create tensor: %v", err)
			}
			defer tensor.Release()
			
			if tensor.DType() != test.dtype {
				t.Errorf("Expected dtype %d, got %d", test.dtype, tensor.DType())
			}
			
			if tensor.Device() != test.device {
				t.Errorf("Expected device %d, got %d", test.device, tensor.Device())
			}
			
			shape := tensor.Shape()
			if len(shape) != len(test.shape) {
				t.Errorf("Expected shape length %d, got %d", len(test.shape), len(shape))
			}
			
			for i, dim := range test.shape {
				if i < len(shape) && shape[i] != dim {
					t.Errorf("Expected shape[%d] = %d, got %d", i, dim, shape[i])
				}
			}
			
			if tensor.RefCount() != 1 {
				t.Errorf("Expected initial ref count 1, got %d", tensor.RefCount())
			}
		})
	}
	
	t.Log("Tensor creation tests passed")
}

// TestTensorReferenceCountingbasic tests basic reference counting
func TestTensorReferenceCountingBasic(t *testing.T) {
	// Skip tensor reference counting tests that require actual Metal buffer allocation
	t.Skip("Tensor reference counting requires Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{10}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	
	// Test initial reference count
	if tensor.RefCount() != 1 {
		t.Errorf("Expected initial ref count 1, got %d", tensor.RefCount())
	}
	
	// Test retain
	tensor2 := tensor.Retain()
	if tensor2 != tensor {
		t.Error("Retain should return the same tensor")
	}
	if tensor.RefCount() != 2 {
		t.Errorf("Expected ref count 2 after retain, got %d", tensor.RefCount())
	}
	
	// Test clone
	tensor3 := tensor.Clone()
	if tensor3 != tensor {
		t.Error("Clone should return the same tensor")
	}
	if tensor.RefCount() != 3 {
		t.Errorf("Expected ref count 3 after clone, got %d", tensor.RefCount())
	}
	
	// Test release
	tensor2.Release()
	if tensor.RefCount() != 2 {
		t.Errorf("Expected ref count 2 after first release, got %d", tensor.RefCount())
	}
	
	tensor3.Release()
	if tensor.RefCount() != 1 {
		t.Errorf("Expected ref count 1 after second release, got %d", tensor.RefCount())
	}
	
	// Final release
	tensor.Release()
	// After final release, tensor should be cleaned up
	
	t.Log("Tensor reference counting basic tests passed")
}

// TestTensorReferenceCountingConcurrent tests concurrent reference counting
func TestTensorReferenceCountingConcurrent(t *testing.T) {
	// Skip concurrent tensor tests that require actual Metal buffer allocation
	t.Skip("Concurrent tensor tests require Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{100}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	
	numGoroutines := 50
	numOperations := 100
	
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	
	// Start multiple goroutines that retain and release the tensor
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				retained := tensor.Retain()
				// Do some work
				runtime.Gosched()
				retained.Release()
			}
		}()
	}
	
	wg.Wait()
	
	// Should be back to 1 reference
	if tensor.RefCount() != 1 {
		t.Errorf("Expected ref count 1 after concurrent operations, got %d", tensor.RefCount())
	}
	
	tensor.Release()
	
	t.Log("Tensor reference counting concurrent tests passed")
}

// TestTensorDataOperations tests tensor data copy operations
func TestTensorDataOperations(t *testing.T) {
	// Skip tensor data operations that require actual Metal buffer allocation
	t.Skip("Tensor data operations require Metal buffer allocation - skipping for unit tests")
	
	// Test Float32 data operations
	tensor, err := NewTensor([]int{5}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	err = tensor.CopyFloat32Data(data)
	if err != nil {
		t.Errorf("Failed to copy float32 data: %v", err)
	}
	
	// Test wrong data length
	wrongData := []float32{1.0, 2.0, 3.0}
	err = tensor.CopyFloat32Data(wrongData)
	if err == nil {
		t.Error("Expected error for wrong data length")
	}
	
	// Test wrong data type
	intTensor, err := NewTensor([]int{5}, Int32, GPU)
	if err != nil {
		t.Fatalf("Failed to create int tensor: %v", err)
	}
	defer intTensor.Release()
	
	err = intTensor.CopyFloat32Data(data)
	if err == nil {
		t.Error("Expected error for wrong data type")
	}
	
	// Test Int32 data operations
	intData := []int32{1, 2, 3, 4, 5}
	err = intTensor.CopyInt32Data(intData)
	if err != nil {
		t.Errorf("Failed to copy int32 data: %v", err)
	}
	
	t.Log("Tensor data operations tests passed")
}

// TestTensorToFloat32Slice tests conversion to float32 slice
func TestTensorToFloat32Slice(t *testing.T) {
	// Skip tensor conversion tests that require actual Metal buffer allocation
	t.Skip("Tensor conversion tests require Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{5}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	data, err := tensor.ToFloat32Slice()
	if err != nil {
		t.Errorf("Failed to convert to float32 slice: %v", err)
	}
	
	if len(data) != 5 {
		t.Errorf("Expected slice length 5, got %d", len(data))
	}
	
	// Test wrong data type
	intTensor, err := NewTensor([]int{5}, Int32, GPU)
	if err != nil {
		t.Fatalf("Failed to create int tensor: %v", err)
	}
	defer intTensor.Release()
	
	_, err = intTensor.ToFloat32Slice()
	if err == nil {
		t.Error("Expected error for wrong data type")
	}
	
	t.Log("Tensor ToFloat32Slice tests passed")
}

// TestTensorConvertTo tests tensor type conversion
func TestTensorConvertTo(t *testing.T) {
	// Skip tensor type conversion tests that require actual Metal buffer allocation
	t.Skip("Tensor type conversion tests require Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{5}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	// Test conversion to same type (should just retain)
	converted, err := tensor.ConvertTo(Float32)
	if err != nil {
		t.Errorf("Failed to convert to same type: %v", err)
	}
	defer converted.Release()
	
	if converted != tensor {
		t.Error("Converting to same type should return same tensor")
	}
	
	if tensor.RefCount() != 2 {
		t.Errorf("Expected ref count 2 after same-type conversion, got %d", tensor.RefCount())
	}
	
	// Test conversion to different type
	int32Tensor, err := tensor.ConvertTo(Int32)
	if err != nil {
		t.Errorf("Failed to convert to int32: %v", err)
	}
	defer int32Tensor.Release()
	
	if int32Tensor.DType() != Int32 {
		t.Errorf("Expected converted tensor dtype Int32, got %d", int32Tensor.DType())
	}
	
	t.Log("Tensor ConvertTo tests passed")
}

// TestTensorMetalBuffer tests metal buffer access
func TestTensorMetalBuffer(t *testing.T) {
	// Skip metal buffer access tests that require actual Metal buffer allocation
	t.Skip("Metal buffer access tests require Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{10}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	
	buffer := tensor.MetalBuffer()
	if buffer == nil {
		t.Error("Metal buffer should not be nil")
	}
	
	// Release tensor and test panic
	tensor.Release()
	
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when accessing released tensor buffer")
		}
	}()
	
	// This should panic
	tensor.MetalBuffer()
}

// TestTensorString tests string representation
func TestTensorString(t *testing.T) {
	// Skip tensor string tests that require actual Metal buffer allocation
	t.Skip("Tensor string tests require Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{2, 3}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	str := tensor.String()
	if len(str) == 0 {
		t.Error("String representation should not be empty")
	}
	
	// Should contain tensor information
	if !contains(str, "Tensor") {
		t.Error("String should contain 'Tensor'")
	}
	
	t.Log("Tensor String tests passed")
}

// TestMemoryManagerStats tests memory manager statistics
func TestMemoryManagerStats(t *testing.T) {
	// Skip memory manager stats tests that require actual Metal buffer allocation
	t.Skip("Memory manager stats tests require Metal buffer allocation - skipping for unit tests")
	
	mm := GetGlobalMemoryManager()
	
	// Initially should have no pools
	stats := mm.Stats()
	if len(stats) != 0 {
		t.Errorf("Expected 0 pools initially, got %d", len(stats))
	}
	
	// Create a tensor to force pool creation
	tensor, err := NewTensor([]int{100}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	// Should have created a pool
	stats = mm.Stats()
	if len(stats) == 0 {
		t.Error("Expected at least one pool after tensor creation")
	}
	
	t.Log("Memory manager stats tests passed")
}

// TestMockFunctions tests mock device and queue creation
func TestMockFunctions(t *testing.T) {
	device := CreateMockDevice()
	if device == nil {
		t.Error("Mock device should not be nil")
	}
	
	queue := CreateMockCommandQueue()
	if queue == nil {
		t.Error("Mock command queue should not be nil")
	}
	
	// Should have different addresses
	if uintptr(device) == uintptr(queue) {
		t.Error("Mock device and queue should have different addresses")
	}
	
	t.Log("Mock functions tests passed")
}

// TestMemoryManagerAllocateRelease tests simple allocate/release interface
func TestMemoryManagerAllocateRelease(t *testing.T) {
	// Skip allocate/release tests that require actual Metal buffer allocation
	t.Skip("Allocate/release tests require Metal buffer allocation - skipping for unit tests")
	
	// Test is skipped - requires actual buffer allocation
}

// TestGetDevice tests device getter
func TestGetDevice(t *testing.T) {
	// Skip get device tests that require global memory manager initialization
	t.Skip("GetDevice tests require global memory manager - skipping for unit tests")
	
	// Test is skipped - no actual device to compare
}

// TestTensorZeroShape tests tensor with zero-sized dimensions
func TestTensorZeroShape(t *testing.T) {
	// Skip zero shape tensor tests that require actual Metal buffer allocation
	t.Skip("Zero shape tensor tests require Metal buffer allocation - skipping for unit tests")
	
	// Test tensor with zero dimension
	tensor, err := NewTensor([]int{0, 5}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor with zero dimension: %v", err)
	}
	defer tensor.Release()
	
	if tensor.Size() != 0 {
		t.Errorf("Expected size 0 for tensor with zero dimension, got %d", tensor.Size())
	}
	
	// Test ToFloat32Slice with invalid shape
	_, err = tensor.ToFloat32Slice()
	if err == nil {
		t.Error("Expected error for ToFloat32Slice with invalid shape")
	}
	
	t.Log("Tensor zero shape tests passed")
}

// TestBridgeFunctionErrors tests error handling when bridge functions are not set
func TestBridgeFunctionErrors(t *testing.T) {
	// Skip bridge function error tests that require actual Metal buffer allocation
	t.Skip("Bridge function error tests require Metal buffer allocation - skipping for unit tests")
	
	// Save current bridge functions
	oldToFloat32 := ToFloat32SliceFunc
	oldCopyFloat32 := CopyFloat32DataFunc
	oldCopyInt32 := CopyInt32DataFunc
	oldConvert := ConvertTensorTypeFunc
	
	// Clear bridge functions
	ToFloat32SliceFunc = nil
	CopyFloat32DataFunc = nil
	CopyInt32DataFunc = nil
	ConvertTensorTypeFunc = nil
	
	defer func() {
		// Restore bridge functions
		ToFloat32SliceFunc = oldToFloat32
		CopyFloat32DataFunc = oldCopyFloat32
		CopyInt32DataFunc = oldCopyInt32
		ConvertTensorTypeFunc = oldConvert
	}()
	
	tensor, err := NewTensor([]int{5}, Float32, GPU)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	// Test ToFloat32Slice without bridge function
	_, err = tensor.ToFloat32Slice()
	if err == nil {
		t.Error("Expected error when ToFloat32SliceFunc is nil")
	}
	
	// Test CopyFloat32Data without bridge function
	data := []float32{1, 2, 3, 4, 5}
	err = tensor.CopyFloat32Data(data)
	if err == nil {
		t.Error("Expected error when CopyFloat32DataFunc is nil")
	}
	
	// Test CopyInt32Data without bridge function
	intTensor, err := NewTensor([]int{5}, Int32, GPU)
	if err != nil {
		t.Fatalf("Failed to create int tensor: %v", err)
	}
	defer intTensor.Release()
	
	intData := []int32{1, 2, 3, 4, 5}
	err = intTensor.CopyInt32Data(intData)
	if err == nil {
		t.Error("Expected error when CopyInt32DataFunc is nil")
	}
	
	// Test ConvertTo without bridge function
	_, err = tensor.ConvertTo(Int32)
	if err == nil {
		t.Error("Expected error when ConvertTensorTypeFunc is nil")
	}
	
	t.Log("Bridge function error tests passed")
}

// Helper function for string contains check
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || 
		    (len(s) > len(substr) && 
		     (s[:len(substr)] == substr || 
		      s[len(s)-len(substr):] == substr || 
		      containsAtIndex(s, substr))))
}

func containsAtIndex(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Benchmarks

// BenchmarkTensorCreation benchmarks tensor creation
func BenchmarkTensorCreation(b *testing.B) {
	// Skip tensor creation benchmarks that require actual Metal buffer allocation
	b.Skip("Tensor creation benchmarks require Metal buffer allocation - skipping for unit tests")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor, err := NewTensor([]int{100}, Float32, GPU)
		if err != nil {
			b.Fatalf("Failed to create tensor: %v", err)
		}
		tensor.Release()
	}
}

// BenchmarkTensorRetainRelease benchmarks retain/release operations
func BenchmarkTensorRetainRelease(b *testing.B) {
	// Skip tensor retain/release benchmarks that require actual Metal buffer allocation
	b.Skip("Tensor retain/release benchmarks require Metal buffer allocation - skipping for unit tests")
	
	tensor, err := NewTensor([]int{100}, Float32, GPU)
	if err != nil {
		b.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Release()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		retained := tensor.Retain()
		retained.Release()
	}
}

// BenchmarkMemoryManagerGetReturn benchmarks buffer get/return operations
func BenchmarkMemoryManagerGetReturn(b *testing.B) {
	// Skip memory manager get/return benchmarks that require actual Metal buffer allocation
	b.Skip("Memory manager get/return benchmarks require Metal buffer allocation - skipping for unit tests")
	// Benchmark is skipped - requires actual buffer allocation
}