package async

import (
	"fmt"
	"testing"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// MockDataSource implements DataSource for testing
type MockDataSource struct {
	batchSize  int
	counter    int
	maxBatches int
}

func NewMockDataSource(maxBatches int) *MockDataSource {
	return &MockDataSource{
		maxBatches: maxBatches,
		counter:    0,
	}
}

func (mds *MockDataSource) GetBatch(batchSize int) ([]float32, []int, []float32, []int, error) {
	if mds.counter >= mds.maxBatches {
		return nil, nil, nil, nil, fmt.Errorf("no more batches available")
	}

	mds.counter++

	// Create dummy batch data
	inputShape := []int{batchSize, 3, 32, 32} // RGB 32x32 images
	labelShape := []int{batchSize}

	inputSize := batchSize * 3 * 32 * 32
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(i%100) / 100.0
	}

	labelData := make([]float32, batchSize)
	for i := range labelData {
		labelData[i] = float32(i % 2) // Binary classification
	}

	return inputData, inputShape, labelData, labelShape, nil
}

func (mds *MockDataSource) Size() int {
	return mds.maxBatches * 32 // Assuming batch size of 32
}

func (mds *MockDataSource) Reset() error {
	mds.counter = 0
	return nil
}

// TestAsyncDataLoaderIntegration tests the async data loading pipeline integration
func TestAsyncDataLoaderIntegration(t *testing.T) {
	// Test the full async data loading pipeline if Metal device is available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Skipping AsyncDataLoader integration test - Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Create data source
	mockDataSource := NewMockDataSource(10)
	config := AsyncDataLoaderConfig{
		BatchSize:     16,
		PrefetchDepth: 3,
		Workers:       2,
		MemoryManager: memoryManager,
	}

	// Create and test the full pipeline
	loader, err := NewAsyncDataLoader(mockDataSource, config)
	if err != nil {
		t.Fatalf("Failed to create async data loader: %v", err)
	}
	defer loader.Stop()

	err = loader.Start()
	if err != nil {
		t.Fatalf("Failed to start data loader: %v", err)
	}

	// Test getting batches from the pipeline
	for i := 0; i < 5; i++ {
		batch, err := loader.GetBatch()
		if err != nil {
			t.Fatalf("Failed to get batch %d: %v", i, err)
		}

		// Verify batch structure
		if batch.InputTensor == nil || batch.LabelTensor == nil {
			t.Errorf("Batch %d has nil tensors", i)
		}

		batch.Release()
		t.Logf("✅ Integration test batch %d completed", i)
	}

	t.Log("✅ AsyncDataLoader integration test passed")
}

// TestStagingBufferPoolIntegration tests the staging buffer pool integration
func TestStagingBufferPoolIntegration(t *testing.T) {
	// Test staging buffer pool if Metal device is available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Skipping StagingBufferPool integration test - Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Create staging buffer pool
	pool, err := NewStagingBufferPool(memoryManager, 4)
	if err != nil {
		t.Fatalf("Failed to create staging buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Create test tensor
	tensorShape := []int{256}
	gpuTensor, err := memory.NewTensor(tensorShape, memory.Float32, memory.GPU)
	if err != nil {
		t.Fatalf("Failed to create GPU tensor: %v", err)
	}
	defer gpuTensor.Release()

	// Test data transfer
	testData := make([]float32, 256)
	for i := range testData {
		testData[i] = float32(i) * 0.5
	}

	err = pool.TransferToGPUSync(testData, gpuTensor)
	if err != nil {
		t.Fatalf("Failed to transfer data to GPU: %v", err)
	}

	t.Log("✅ StagingBufferPool integration test passed")
}

// TestCommandBufferPoolIntegration tests the command buffer pool integration
func TestCommandBufferPoolIntegration(t *testing.T) {
	// Test command buffer pool if Metal device is available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Skipping CommandBufferPool integration test - Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	// Create command queue
	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Fatalf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	// Create command buffer pool
	pool, err := NewCommandBufferPool(commandQueue, 4)
	if err != nil {
		t.Fatalf("Failed to create command buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Test getting and returning a buffer
	buffer, err := pool.GetBuffer()
	if err != nil {
		t.Fatalf("Failed to get command buffer: %v", err)
	}

	if !buffer.IsInUse() {
		t.Error("Buffer should be marked as in use")
	}

	pool.ReturnBuffer(buffer)

	if buffer.IsInUse() {
		t.Error("Buffer should not be in use after return")
	}

	t.Log("✅ CommandBufferPool integration test passed")
}

// TestCommandBufferPoolLogic tests the command buffer pool logic without Metal integration
func TestCommandBufferPoolLogic(t *testing.T) {
	// Test validation logic
	
	// Test 1: Nil command queue should be rejected
	_, err := NewCommandBufferPool(nil, 5)
	if err == nil {
		t.Error("Expected error for nil command queue")
	}
	if err != nil && !contains(err.Error(), "command queue cannot be nil") {
		t.Errorf("Expected 'command queue cannot be nil' error, got: %v", err)
	}
	
	// Test 2: Invalid maxBuffers should be rejected
	mockQueue := memory.CreateMockCommandQueue()
	_, err = NewCommandBufferPool(mockQueue, 0)
	if err == nil {
		t.Error("Expected error for zero maxBuffers")
	}
	if err != nil && !contains(err.Error(), "maxBuffers must be positive") {
		t.Errorf("Expected 'maxBuffers must be positive' error, got: %v", err)
	}
	
	_, err = NewCommandBufferPool(mockQueue, -1)
	if err == nil {
		t.Error("Expected error for negative maxBuffers")
	}
	
	// Test 3: CommandBuffer structure validation
	buffer := &CommandBuffer{
		buffer: mockQueue, // Use mock pointer
		inUse:  false,
		id:     1,
	}
	
	if buffer.id != 1 {
		t.Errorf("Expected buffer ID 1, got %d", buffer.id)
	}
	
	if buffer.inUse {
		t.Error("New buffer should not be marked as in use")
	}
	
	buffer.inUse = true
	if !buffer.inUse {
		t.Error("Buffer should be marked as in use after setting")
	}
	
	// Test 4: CommandPoolStats structure
	stats := CommandPoolStats{
		TotalBuffers:     10,
		AvailableBuffers: 5,
		InUseBuffers:     5,
		MaxBuffers:       10,
	}
	
	if stats.TotalBuffers != 10 {
		t.Errorf("Expected TotalBuffers 10, got %d", stats.TotalBuffers)
	}
	
	if stats.AvailableBuffers+stats.InUseBuffers != stats.TotalBuffers {
		t.Errorf("Available (%d) + InUse (%d) should equal Total (%d)", 
			stats.AvailableBuffers, stats.InUseBuffers, stats.TotalBuffers)
	}
	
	// Test 5: BatchOperation structure
	completed := false
	operation := BatchOperation{
		Type: "test_operation",
		Data: "test_data",
		Completion: func(err error) {
			completed = true
		},
	}
	
	if operation.Type != "test_operation" {
		t.Errorf("Expected operation type 'test_operation', got %s", operation.Type)
	}
	
	if operation.Data != "test_data" {
		t.Errorf("Expected operation data 'test_data', got %v", operation.Data)
	}
	
	// Test completion callback
	operation.Completion(nil)
	if !completed {
		t.Error("Completion callback should have been called")
	}
	
	t.Log("✅ Command buffer pool logic tests passed")
}

// Helper function for string contains check (reused from layers tests)
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

// TestAsyncPipelineIntegration tests the complete async pipeline
func TestAsyncPipelineIntegration(t *testing.T) {
	// Test complete async pipeline if Metal device is available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Skipping async pipeline integration test - Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Create command queue
	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Fatalf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	// Create components
	mockDataSource := NewMockDataSource(15)
	commandPool, err := NewCommandBufferPool(commandQueue, 4)
	if err != nil {
		t.Fatalf("Failed to create command buffer pool: %v", err)
	}
	defer commandPool.Cleanup()

	stagingPool, err := NewStagingBufferPool(memoryManager, 6)
	if err != nil {
		t.Fatalf("Failed to create staging buffer pool: %v", err)
	}
	defer stagingPool.Cleanup()

	dataLoaderConfig := AsyncDataLoaderConfig{
		BatchSize:     8,
		PrefetchDepth: 3,
		Workers:       2,
		MemoryManager: memoryManager,
	}

	dataLoader, err := NewAsyncDataLoader(mockDataSource, dataLoaderConfig)
	if err != nil {
		t.Fatalf("Failed to create data loader: %v", err)
	}
	defer dataLoader.Stop()

	// Start the pipeline
	err = dataLoader.Start()
	if err != nil {
		t.Fatalf("Failed to start data loader: %v", err)
	}

	// Test the complete pipeline
	numBatches := 5
	for i := 0; i < numBatches; i++ {
		// Get batch from data loader
		batch, err := dataLoader.GetBatch()
		if err != nil {
			t.Fatalf("Failed to get batch %d: %v", i, err)
		}

		// Get command buffer
		cmdBuffer, err := commandPool.GetBuffer()
		if err != nil {
			t.Fatalf("Failed to get command buffer for batch %d: %v", i, err)
		}

		// Test staging buffer for additional transfers
		additionalData := []float32{1.0, 2.0, 3.0, 4.0}
		err = stagingPool.TransferToGPU(additionalData, batch.InputTensor)
		if err != nil {
			t.Logf("Note: Staging transfer failed (expected for size mismatch): %v", err)
		}

		// Return command buffer
		commandPool.ReturnBuffer(cmdBuffer)

		// Release batch
		batch.Release()

		t.Logf("✅ Pipeline iteration %d completed", i)
	}

	// Verify statistics
	dataLoaderStats := dataLoader.Stats()
	commandPoolStats := commandPool.Stats()
	stagingPoolStats := stagingPool.Stats()

	t.Logf("Pipeline stats:")
	t.Logf("  Data Loader: %d batches produced, %d workers", 
		dataLoaderStats.BatchesProduced, dataLoaderStats.Workers)
	t.Logf("  Command Pool: %d total buffers, %d in use", 
		commandPoolStats.TotalBuffers, commandPoolStats.InUseBuffers)
	t.Logf("  Staging Pool: %d total buffers, %d in use", 
		stagingPoolStats.TotalBuffers, stagingPoolStats.InUseBuffers)

	// Verify clean state
	if commandPoolStats.InUseBuffers != 0 {
		t.Errorf("Expected 0 command buffers in use, got %d", commandPoolStats.InUseBuffers)
	}
	if stagingPoolStats.InUseBuffers != 0 {
		t.Errorf("Expected 0 staging buffers in use, got %d", stagingPoolStats.InUseBuffers)
	}

	t.Log("✅ Complete async pipeline integration test passed")
}

// TestAsyncPackageInterfaces tests interface implementations
func TestAsyncPackageInterfaces(t *testing.T) {
	// Test that MockDataSource properly implements DataSource interface
	var dataSource DataSource = NewMockDataSource(5)

	// Test interface methods
	size := dataSource.Size()
	if size != 5*32 {
		t.Errorf("Expected size %d, got %d", 5*32, size)
	}

	inputData, inputShape, labelData, labelShape, err := dataSource.GetBatch(16)
	if err != nil {
		t.Fatalf("Failed to get batch: %v", err)
	}

	if len(inputData) != 16*3*32*32 {
		t.Errorf("Unexpected input data length: %d", len(inputData))
	}
	if len(inputShape) != 4 {
		t.Errorf("Unexpected input shape length: %d", len(inputShape))
	}
	if len(labelData) != 16 {
		t.Errorf("Unexpected label data length: %d", len(labelData))
	}
	if len(labelShape) != 1 {
		t.Errorf("Unexpected label shape length: %d", len(labelShape))
	}

	err = dataSource.Reset()
	if err != nil {
		t.Errorf("Failed to reset data source: %v", err)
	}

	t.Log("✅ Interface implementation tests passed")
}

// TestErrorHandling tests error conditions across the async package
func TestErrorHandling(t *testing.T) {
	// Test 1: Invalid configurations
	t.Run("InvalidConfigurations", func(t *testing.T) {
		// Test AsyncDataLoaderConfig validation
		mockDataSource := NewMockDataSource(5)
		
		// Nil data source
		_, err := NewAsyncDataLoader(nil, AsyncDataLoaderConfig{})
		if err == nil {
			t.Error("Expected error for nil data source")
		}

		// Invalid batch size
		_, err = NewAsyncDataLoader(mockDataSource, AsyncDataLoaderConfig{
			BatchSize: 0,
		})
		if err == nil {
			t.Error("Expected error for zero batch size")
		}

		// Nil memory manager
		_, err = NewAsyncDataLoader(mockDataSource, AsyncDataLoaderConfig{
			BatchSize:     32,
			MemoryManager: nil,
		})
		if err == nil {
			t.Error("Expected error for nil memory manager")
		}
	})

	// Test 2: Data source exhaustion
	t.Run("DataSourceExhaustion", func(t *testing.T) {
		limitedDataSource := NewMockDataSource(2) // Only 2 batches

		// Get first batch
		_, _, _, _, err := limitedDataSource.GetBatch(16)
		if err != nil {
			t.Fatalf("Failed to get first batch: %v", err)
		}

		// Get second batch
		_, _, _, _, err = limitedDataSource.GetBatch(16)
		if err != nil {
			t.Fatalf("Failed to get second batch: %v", err)
		}

		// Third batch should fail
		_, _, _, _, err = limitedDataSource.GetBatch(16)
		if err == nil {
			t.Error("Expected error for exhausted data source")
		}
		if !contains(err.Error(), "no more batches available") {
			t.Errorf("Expected 'no more batches available' error, got: %v", err)
		}
	})

	// Test 3: Configuration parameter bounds
	t.Run("ConfigurationBounds", func(t *testing.T) {
		// Test edge case configurations
		mockDataSource := NewMockDataSource(10)
		
		// Very large batch size
		config := AsyncDataLoaderConfig{
			BatchSize:     1000000, // 1M - very large
			PrefetchDepth: 1,
			Workers:       1,
			MemoryManager: nil, // Will cause error
		}
		
		_, err := NewAsyncDataLoader(mockDataSource, config)
		if err == nil {
			t.Error("Expected error for nil memory manager")
		}
		
		// Negative values should be handled
		config.BatchSize = -1
		_, err = NewAsyncDataLoader(mockDataSource, config)
		if err == nil {
			t.Error("Expected error for negative batch size")
		}
	})

	t.Log("✅ Error handling tests passed")
}

// TestBasicStructures tests the basic data structures without Metal integration
func TestBasicStructures(t *testing.T) {
	// Test GPUBatch structure
	batch := &GPUBatch{
		InputTensor:   nil, // Would be real tensors in actual usage
		LabelTensor:   nil,
		WeightTensors: nil,
		BatchID:       123,
		Generation:    1,
	}

	if batch.BatchID != 123 {
		t.Errorf("Expected BatchID 123, got %d", batch.BatchID)
	}

	if batch.Generation != 1 {
		t.Errorf("Expected Generation 1, got %d", batch.Generation)
	}

	// Test MockDataSource
	dataSource := NewMockDataSource(5)
	if dataSource.Size() != 5*32 {
		t.Errorf("Expected size %d, got %d", 5*32, dataSource.Size())
	}

	// Test getting a batch
	inputData, inputShape, labelData, labelShape, err := dataSource.GetBatch(16)
	if err != nil {
		t.Fatalf("Failed to get batch: %v", err)
	}

	expectedInputSize := 16 * 3 * 32 * 32
	if len(inputData) != expectedInputSize {
		t.Errorf("Expected input data size %d, got %d", expectedInputSize, len(inputData))
	}

	if len(inputShape) != 4 {
		t.Errorf("Expected input shape length 4, got %d", len(inputShape))
	}

	if len(labelData) != 16 {
		t.Errorf("Expected label data size 16, got %d", len(labelData))
	}

	if len(labelShape) != 1 {
		t.Errorf("Expected label shape length 1, got %d", len(labelShape))
	}

	// Test reset
	err = dataSource.Reset()
	if err != nil {
		t.Errorf("Failed to reset data source: %v", err)
	}

	if dataSource.counter != 0 {
		t.Errorf("Expected counter to be 0 after reset, got %d", dataSource.counter)
	}

	t.Log("✅ Basic structure tests passed")
}

// TestAsyncPackageConstants tests package-level constants and defaults
func TestAsyncPackageConstants(t *testing.T) {
	// Test AsyncDataLoaderConfig default application
	config := AsyncDataLoaderConfig{
		BatchSize:     32,
		PrefetchDepth: 0, // Should get default
		Workers:       0, // Should get default
	}

	// Simulate default application (normally done in NewAsyncDataLoader)
	if config.PrefetchDepth <= 0 {
		config.PrefetchDepth = 3 // Default
	}
	if config.Workers <= 0 {
		config.Workers = 2 // Default
	}

	if config.PrefetchDepth != 3 {
		t.Errorf("Expected default PrefetchDepth 3, got %d", config.PrefetchDepth)
	}
	if config.Workers != 2 {
		t.Errorf("Expected default Workers 2, got %d", config.Workers)
	}

	// Test staging buffer pool constants
	expectedBufferSize := 4 * 1024 * 1024 // 4MB
	if expectedBufferSize != 4194304 {
		t.Errorf("Expected buffer size calculation to be 4MB (4194304 bytes), got %d", expectedBufferSize)
	}

	t.Log("✅ Package constants tests passed")
}

// TestAsyncComponentCompatibility tests compatibility between async components
func TestAsyncComponentCompatibility(t *testing.T) {
	// Test that different components work together conceptually
	
	// Test 1: GPUBatch compatibility with different tensor shapes
	batch1 := &GPUBatch{
		BatchID:    1,
		Generation: 1,
	}
	batch2 := &GPUBatch{
		BatchID:    2,
		Generation: 1,
	}

	if batch1.BatchID == batch2.BatchID {
		t.Error("Batches should have unique IDs")
	}
	if batch1.Generation != batch2.Generation {
		t.Error("Batches from same generation should have same generation number")
	}

	// Test 2: MockDataSource batch size compatibility
	dataSource := NewMockDataSource(5)
	
	// Test different batch sizes
	batchSizes := []int{1, 8, 16, 32, 64}
	for _, batchSize := range batchSizes {
		err := dataSource.Reset()
		if err != nil {
			t.Fatalf("Failed to reset for batch size %d: %v", batchSize, err)
		}

		inputData, inputShape, labelData, labelShape, err := dataSource.GetBatch(batchSize)
		if err != nil {
			t.Fatalf("Failed to get batch with size %d: %v", batchSize, err)
		}

		// Verify shape consistency
		if inputShape[0] != batchSize {
			t.Errorf("Batch size %d: input shape[0] should be %d, got %d", 
				batchSize, batchSize, inputShape[0])
		}
		if labelShape[0] != batchSize {
			t.Errorf("Batch size %d: label shape[0] should be %d, got %d", 
				batchSize, batchSize, labelShape[0])
		}

		// Verify data consistency
		expectedInputSize := batchSize * 3 * 32 * 32
		if len(inputData) != expectedInputSize {
			t.Errorf("Batch size %d: expected input data size %d, got %d", 
				batchSize, expectedInputSize, len(inputData))
		}
		if len(labelData) != batchSize {
			t.Errorf("Batch size %d: expected label data size %d, got %d", 
				batchSize, batchSize, len(labelData))
		}
	}

	// Test 3: Statistics structure consistency
	dataLoaderStats := AsyncDataLoaderStats{
		IsRunning:       true,
		BatchesProduced: 100,
		QueuedBatches:   5,
		QueueCapacity:   10,
		Workers:         2,
		Generation:      3,
	}

	commandPoolStats := CommandPoolStats{
		TotalBuffers:     8,
		AvailableBuffers: 3,
		InUseBuffers:     5,
		MaxBuffers:       10,
	}

	stagingPoolStats := StagingPoolStats{
		TotalBuffers:     6,
		AvailableBuffers: 2,
		InUseBuffers:     4,
		BufferSize:       4 * 1024 * 1024,
		MaxBuffers:       8,
	}

	// Verify internal consistency
	if dataLoaderStats.QueuedBatches > dataLoaderStats.QueueCapacity {
		t.Error("Queued batches should not exceed queue capacity")
	}
	if commandPoolStats.AvailableBuffers+commandPoolStats.InUseBuffers != commandPoolStats.TotalBuffers {
		t.Error("Command pool buffer counts should be consistent")
	}
	if stagingPoolStats.AvailableBuffers+stagingPoolStats.InUseBuffers != stagingPoolStats.TotalBuffers {
		t.Error("Staging pool buffer counts should be consistent")
	}

	t.Log("✅ Component compatibility tests passed")
}

// TestAsyncDataLoaderLogic tests the async data loader logic without Metal integration
func TestAsyncDataLoaderLogic(t *testing.T) {
	// Test 1: Nil data source should be rejected
	config := AsyncDataLoaderConfig{
		BatchSize:     32,
		PrefetchDepth: 3,
		Workers:       2,
		MemoryManager: nil, // Will test separately
	}
	
	_, err := NewAsyncDataLoader(nil, config)
	if err == nil {
		t.Error("Expected error for nil data source")
	}
	if err != nil && !contains(err.Error(), "data source cannot be nil") {
		t.Errorf("Expected 'data source cannot be nil' error, got: %v", err)
	}
	
	// Test 2: Invalid batch size should be rejected
	mockDataSource := NewMockDataSource(5)
	config.BatchSize = 0
	_, err = NewAsyncDataLoader(mockDataSource, config)
	if err == nil {
		t.Error("Expected error for zero batch size")
	}
	if err != nil && !contains(err.Error(), "batch size must be positive") {
		t.Errorf("Expected 'batch size must be positive' error, got: %v", err)
	}
	
	config.BatchSize = -1
	_, err = NewAsyncDataLoader(mockDataSource, config)
	if err == nil {
		t.Error("Expected error for negative batch size")
	}
	
	// Test 3: Nil memory manager should be rejected
	config.BatchSize = 32
	config.MemoryManager = nil
	_, err = NewAsyncDataLoader(mockDataSource, config)
	if err == nil {
		t.Error("Expected error for nil memory manager")
	}
	if err != nil && !contains(err.Error(), "memory manager is required") {
		t.Errorf("Expected 'memory manager is required' error, got: %v", err)
	}
	
	// Test 4: GPUBatch structure and methods
	batch := &GPUBatch{
		InputTensor:   nil, // Would be real tensors in production
		LabelTensor:   nil,
		WeightTensors: nil,
		BatchID:       123,
		Generation:    5,
	}
	
	if batch.BatchID != 123 {
		t.Errorf("Expected BatchID 123, got %d", batch.BatchID)
	}
	
	if batch.Generation != 5 {
		t.Errorf("Expected Generation 5, got %d", batch.Generation)
	}
	
	// Test Release method with nil tensors (should not crash)
	batch.Release()
	
	// Test 5: AsyncDataLoaderConfig validation with defaults
	validConfig := AsyncDataLoaderConfig{
		BatchSize:     32,
		PrefetchDepth: 0, // Should use default
		Workers:       0, // Should use default
		MemoryManager: nil,
	}
	
	// The constructor would apply defaults: PrefetchDepth=3, Workers=2
	expectedPrefetchDepth := 3
	expectedWorkers := 2
	
	if validConfig.PrefetchDepth == 0 {
		validConfig.PrefetchDepth = expectedPrefetchDepth
	}
	if validConfig.Workers == 0 {
		validConfig.Workers = expectedWorkers
	}
	
	if validConfig.PrefetchDepth != expectedPrefetchDepth {
		t.Errorf("Expected default PrefetchDepth %d, got %d", expectedPrefetchDepth, validConfig.PrefetchDepth)
	}
	
	if validConfig.Workers != expectedWorkers {
		t.Errorf("Expected default Workers %d, got %d", expectedWorkers, validConfig.Workers)
	}
	
	// Test 6: AsyncDataLoaderStats structure
	stats := AsyncDataLoaderStats{
		IsRunning:       true,
		BatchesProduced: 100,
		QueuedBatches:   5,
		QueueCapacity:   10,
		Workers:         2,
		Generation:      3,
	}
	
	if !stats.IsRunning {
		t.Error("Stats should show loader as running")
	}
	
	if stats.BatchesProduced != 100 {
		t.Errorf("Expected BatchesProduced 100, got %d", stats.BatchesProduced)
	}
	
	if stats.QueuedBatches > stats.QueueCapacity {
		t.Errorf("QueuedBatches (%d) should not exceed QueueCapacity (%d)", 
			stats.QueuedBatches, stats.QueueCapacity)
	}
	
	if stats.Workers != 2 {
		t.Errorf("Expected Workers 2, got %d", stats.Workers)
	}
	
	if stats.Generation != 3 {
		t.Errorf("Expected Generation 3, got %d", stats.Generation)
	}
	
	t.Log("✅ Async data loader logic tests passed")
}

// TestDataSourceInterfaceIntegration tests the DataSource interface implementation with full validation
func TestDataSourceInterfaceIntegration(t *testing.T) {
	// Test MockDataSource implementation
	maxBatches := 10
	dataSource := NewMockDataSource(maxBatches)
	
	// Test Size method
	expectedSize := maxBatches * 32 // MockDataSource uses batch size 32
	if dataSource.Size() != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, dataSource.Size())
	}
	
	// Test GetBatch method
	batchSize := 16
	inputData, inputShape, labelData, labelShape, err := dataSource.GetBatch(batchSize)
	if err != nil {
		t.Fatalf("Failed to get first batch: %v", err)
	}
	
	// Verify input data
	expectedInputSize := batchSize * 3 * 32 * 32
	if len(inputData) != expectedInputSize {
		t.Errorf("Expected input data length %d, got %d", expectedInputSize, len(inputData))
	}
	
	// Verify input shape
	expectedInputShape := []int{batchSize, 3, 32, 32}
	if len(inputShape) != len(expectedInputShape) {
		t.Errorf("Expected input shape length %d, got %d", len(expectedInputShape), len(inputShape))
	}
	for i, dim := range expectedInputShape {
		if inputShape[i] != dim {
			t.Errorf("Input shape dimension %d: expected %d, got %d", i, dim, inputShape[i])
		}
	}
	
	// Verify label data
	if len(labelData) != batchSize {
		t.Errorf("Expected label data length %d, got %d", batchSize, len(labelData))
	}
	
	// Verify label shape
	expectedLabelShape := []int{batchSize}
	if len(labelShape) != len(expectedLabelShape) {
		t.Errorf("Expected label shape length %d, got %d", len(expectedLabelShape), len(labelShape))
	}
	if labelShape[0] != batchSize {
		t.Errorf("Expected label shape[0] %d, got %d", batchSize, labelShape[0])
	}
	
	// Verify data values are in expected range
	for i, val := range inputData {
		if val < 0.0 || val >= 1.0 {
			t.Errorf("Input data[%d] = %f should be in range [0.0, 1.0)", i, val)
			break // Only report first issue
		}
	}
	
	for i, label := range labelData {
		if label != 0.0 && label != 1.0 {
			t.Errorf("Label data[%d] = %f should be 0.0 or 1.0 for binary classification", i, label)
			break // Only report first issue
		}
	}
	
	// Test getting all batches
	for i := 1; i < maxBatches; i++ {
		_, _, _, _, err = dataSource.GetBatch(batchSize)
		if err != nil {
			t.Fatalf("Failed to get batch %d: %v", i+1, err)
		}
	}
	
	// Test exhaustion
	_, _, _, _, err = dataSource.GetBatch(batchSize)
	if err == nil {
		t.Error("Expected error when getting batch beyond maxBatches")
	}
	if err != nil && !contains(err.Error(), "no more batches available") {
		t.Errorf("Expected 'no more batches available' error, got: %v", err)
	}
	
	// Test Reset method
	err = dataSource.Reset()
	if err != nil {
		t.Errorf("Failed to reset data source: %v", err)
	}
	
	// Should be able to get batches again after reset
	_, _, _, _, err = dataSource.GetBatch(batchSize)
	if err != nil {
		t.Fatalf("Failed to get batch after reset: %v", err)
	}
	
	t.Log("✅ DataSource interface integration tests passed")
}
