package async

import (
	"sync"
	"testing"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// TestGPUBatchStructure tests the GPUBatch data structure
func TestGPUBatchStructure(t *testing.T) {
	// Test 1: Basic structure validation
	batch := &GPUBatch{
		InputTensor:   nil, // Will be nil in this test
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

	// Test 2: Release method with nil tensors (should not crash)
	batch.Release()

	// Test 3: Release method with real tensors if Metal device available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Logf("Metal device not available, testing with nil tensors only: %v", err)
		t.Log("✅ GPUBatch structure tests passed (without Metal)")
		return
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	// Create real tensors
	inputTensor, err := memory.NewTensor([]int{32, 3, 224, 224}, memory.Float32, memory.GPU)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	labelTensor, err := memory.NewTensor([]int{32}, memory.Float32, memory.GPU)
	if err != nil {
		inputTensor.Release()
		t.Fatalf("Failed to create label tensor: %v", err)
	}

	weightTensor, err := memory.NewTensor([]int{512, 256}, memory.Float32, memory.GPU)
	if err != nil {
		inputTensor.Release()
		labelTensor.Release()
		t.Fatalf("Failed to create weight tensor: %v", err)
	}

	batchWithTensors := &GPUBatch{
		InputTensor:   inputTensor,
		LabelTensor:   labelTensor,
		WeightTensors: []*memory.Tensor{weightTensor},
		BatchID:       456,
		Generation:    10,
	}

	// Release should clean up all tensors
	batchWithTensors.Release()

	// Verify tensors are set to nil
	if batchWithTensors.InputTensor != nil {
		t.Error("InputTensor should be nil after Release()")
	}
	if batchWithTensors.LabelTensor != nil {
		t.Error("LabelTensor should be nil after Release()")
	}
	if batchWithTensors.WeightTensors != nil {
		t.Error("WeightTensors should be nil after Release()")
	}

	t.Log("✅ GPUBatch structure tests passed (with Metal)")
}

// TestDataSourceInterface tests the DataSource interface implementation
func TestDataSourceInterface(t *testing.T) {
	maxBatches := 10
	dataSource := NewMockDataSource(maxBatches)

	// Test 1: Size method
	expectedSize := maxBatches * 32 // MockDataSource uses batch size 32
	if dataSource.Size() != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, dataSource.Size())
	}

	// Test 2: GetBatch method with different batch sizes
	testBatchSizes := []int{16, 32, 64}
	
	for _, batchSize := range testBatchSizes {
		err := dataSource.Reset()
		if err != nil {
			t.Fatalf("Failed to reset data source: %v", err)
		}

		inputData, inputShape, labelData, labelShape, err := dataSource.GetBatch(batchSize)
		if err != nil {
			t.Fatalf("Failed to get batch with size %d: %v", batchSize, err)
		}

		// Verify input data
		expectedInputSize := batchSize * 3 * 32 * 32
		if len(inputData) != expectedInputSize {
			t.Errorf("Batch size %d: expected input data length %d, got %d", 
				batchSize, expectedInputSize, len(inputData))
		}

		// Verify input shape
		expectedInputShape := []int{batchSize, 3, 32, 32}
		if len(inputShape) != len(expectedInputShape) {
			t.Errorf("Batch size %d: expected input shape length %d, got %d", 
				batchSize, len(expectedInputShape), len(inputShape))
		}
		for i, dim := range expectedInputShape {
			if inputShape[i] != dim {
				t.Errorf("Batch size %d: input shape dimension %d: expected %d, got %d", 
					batchSize, i, dim, inputShape[i])
			}
		}

		// Verify label data
		if len(labelData) != batchSize {
			t.Errorf("Batch size %d: expected label data length %d, got %d", 
				batchSize, batchSize, len(labelData))
		}

		// Verify label shape
		expectedLabelShape := []int{batchSize}
		if len(labelShape) != len(expectedLabelShape) {
			t.Errorf("Batch size %d: expected label shape length %d, got %d", 
				batchSize, len(expectedLabelShape), len(labelShape))
		}
		if labelShape[0] != batchSize {
			t.Errorf("Batch size %d: expected label shape[0] %d, got %d", 
				batchSize, batchSize, labelShape[0])
		}
	}

	// Test 3: Data value validation
	err := dataSource.Reset()
	if err != nil {
		t.Fatalf("Failed to reset data source: %v", err)
	}

	inputData, _, labelData, _, err := dataSource.GetBatch(16)
	if err != nil {
		t.Fatalf("Failed to get batch for validation: %v", err)
	}

	// Verify input data values are in expected range [0.0, 1.0)
	for i, val := range inputData {
		if val < 0.0 || val >= 1.0 {
			t.Errorf("Input data[%d] = %f should be in range [0.0, 1.0)", i, val)
			break
		}
	}

	// Verify label data values are 0.0 or 1.0 (binary classification)
	for i, label := range labelData {
		if label != 0.0 && label != 1.0 {
			t.Errorf("Label data[%d] = %f should be 0.0 or 1.0 for binary classification", i, label)
			break
		}
	}

	// Test 4: Exhaustion and reset
	err = dataSource.Reset()
	if err != nil {
		t.Fatalf("Failed to reset data source: %v", err)
	}

	// Get all batches
	for i := 0; i < maxBatches; i++ {
		_, _, _, _, err = dataSource.GetBatch(16)
		if err != nil {
			t.Fatalf("Failed to get batch %d: %v", i, err)
		}
	}

	// Should be exhausted now
	_, _, _, _, err = dataSource.GetBatch(16)
	if err == nil {
		t.Error("Expected error when getting batch beyond maxBatches")
	}
	if err != nil && !contains(err.Error(), "no more batches available") {
		t.Errorf("Expected 'no more batches available' error, got: %v", err)
	}

	// Reset should allow getting batches again
	err = dataSource.Reset()
	if err != nil {
		t.Errorf("Failed to reset data source: %v", err)
	}

	_, _, _, _, err = dataSource.GetBatch(16)
	if err != nil {
		t.Fatalf("Failed to get batch after reset: %v", err)
	}

	t.Log("✅ DataSource interface tests passed")
}

// TestAsyncDataLoaderConfig tests configuration validation
func TestAsyncDataLoaderConfig(t *testing.T) {
	mockDataSource := NewMockDataSource(5)

	// Test 1: Nil data source should be rejected
	config := AsyncDataLoaderConfig{
		BatchSize:     32,
		PrefetchDepth: 3,
		Workers:       2,
		MemoryManager: nil,
	}

	_, err := NewAsyncDataLoader(nil, config)
	if err == nil {
		t.Error("Expected error for nil data source")
	}
	if err != nil && !contains(err.Error(), "data source cannot be nil") {
		t.Errorf("Expected 'data source cannot be nil' error, got: %v", err)
	}

	// Test 2: Invalid batch size should be rejected
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

	// Test 4: Default value application
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Logf("Metal device not available for default values test: %v", err)
		t.Log("✅ Config validation tests passed (without Metal)")
		return
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test with zero values that should get defaults
	config = AsyncDataLoaderConfig{
		BatchSize:     32,
		PrefetchDepth: 0, // Should use default 3
		Workers:       0, // Should use default 2
		MemoryManager: memoryManager,
	}

	loader, err := NewAsyncDataLoader(mockDataSource, config)
	if err != nil {
		t.Fatalf("Failed to create data loader with defaults: %v", err)
	}
	defer loader.Stop()

	stats := loader.Stats()
	if stats.Workers != 2 {
		t.Errorf("Expected default Workers 2, got %d", stats.Workers)
	}

	if cap(loader.batchChannel) != 3 { // PrefetchDepth becomes channel capacity
		t.Errorf("Expected default PrefetchDepth 3 (channel capacity), got %d", cap(loader.batchChannel))
	}

	t.Log("✅ Config validation tests passed (with Metal)")
}

// TestAsyncDataLoaderLifecycle tests start/stop lifecycle
func TestAsyncDataLoaderLifecycle(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for lifecycle test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	mockDataSource := NewMockDataSource(10)
	config := AsyncDataLoaderConfig{
		BatchSize:     16,
		PrefetchDepth: 2,
		Workers:       1,
		MemoryManager: memoryManager,
	}

	loader, err := NewAsyncDataLoader(mockDataSource, config)
	if err != nil {
		t.Fatalf("Failed to create data loader: %v", err)
	}

	// Test 1: Initial state
	stats := loader.Stats()
	if stats.IsRunning {
		t.Error("Data loader should not be running initially")
	}

	// Test 2: Start should succeed
	err = loader.Start()
	if err != nil {
		t.Fatalf("Failed to start data loader: %v", err)
	}

	stats = loader.Stats()
	if !stats.IsRunning {
		t.Error("Data loader should be running after Start()")
	}

	// Test 3: Second start should fail
	err = loader.Start()
	if err == nil {
		t.Error("Expected error for second Start() call")
	}
	if err != nil && !contains(err.Error(), "already running") {
		t.Errorf("Expected 'already running' error, got: %v", err)
	}

	// Test 4: Stop should succeed
	err = loader.Stop()
	if err != nil {
		t.Errorf("Failed to stop data loader: %v", err)
	}

	stats = loader.Stats()
	if stats.IsRunning {
		t.Error("Data loader should not be running after Stop()")
	}

	// Test 5: Second stop should be safe
	err = loader.Stop()
	if err != nil {
		t.Errorf("Second Stop() should be safe: %v", err)
	}

	t.Log("✅ Data loader lifecycle tests passed")
}

// TestAsyncDataLoaderBatchProduction tests batch production functionality
func TestAsyncDataLoaderBatchProduction(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for batch production test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	mockDataSource := NewMockDataSource(15) // Enough batches to avoid premature exhaustion
	config := AsyncDataLoaderConfig{
		BatchSize:     8,
		PrefetchDepth: 3,
		Workers:       2,
		MemoryManager: memoryManager,
	}

	loader, err := NewAsyncDataLoader(mockDataSource, config)
	if err != nil {
		t.Fatalf("Failed to create data loader: %v", err)
	}
	defer loader.Stop()

	err = loader.Start()
	if err != nil {
		t.Fatalf("Failed to start data loader: %v", err)
	}

	// Test 1: Get several batches
	numBatches := 3
	batches := make([]*GPUBatch, 0, numBatches)

	for i := 0; i < numBatches; i++ {
		batch, err := loader.GetBatch()
		if err != nil {
			t.Fatalf("Failed to get batch %d: %v", i, err)
		}

		if batch == nil {
			t.Fatalf("Batch %d is nil", i)
		}

		// Verify batch structure
		if batch.InputTensor == nil {
			t.Errorf("Batch %d InputTensor is nil", i)
		}
		if batch.LabelTensor == nil {
			t.Errorf("Batch %d LabelTensor is nil", i)
		}
		if batch.WeightTensors == nil {
			t.Errorf("Batch %d WeightTensors is nil", i)
		}

		batches = append(batches, batch)
		t.Logf("✅ Got batch %d: ID=%d, Generation=%d", i, batch.BatchID, batch.Generation)
	}

	// Test 2: Verify batch IDs are unique and sequential
	for i, batch := range batches {
		expectedID := uint64(i)
		if batch.BatchID != expectedID {
			t.Errorf("Batch %d: expected ID %d, got %d", i, expectedID, batch.BatchID)
		}
	}

	// Test 3: Test non-blocking get
	nonBlockingBatch, err := loader.TryGetBatch()
	if err != nil {
		t.Errorf("TryGetBatch should not error: %v", err)
	}
	if nonBlockingBatch != nil {
		batches = append(batches, nonBlockingBatch)
		t.Logf("✅ Got non-blocking batch: ID=%d", nonBlockingBatch.BatchID)
	} else {
		t.Log("✅ TryGetBatch correctly returned nil (no batch ready)")
	}

	// Test 4: Verify statistics
	stats := loader.Stats()
	if stats.BatchesProduced < uint64(numBatches) {
		t.Errorf("Expected at least %d batches produced, got %d", numBatches, stats.BatchesProduced)
	}

	t.Logf("Final stats: Produced=%d, Queued=%d, Workers=%d", 
		stats.BatchesProduced, stats.QueuedBatches, stats.Workers)

	// Test 5: Clean up batches
	for i, batch := range batches {
		batch.Release()
		t.Logf("✅ Released batch %d", i)
	}

	t.Log("✅ Batch production tests passed")
}

// TestAsyncDataLoaderWorkerManagement tests worker behavior
func TestAsyncDataLoaderWorkerManagement(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for worker management test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test 1: Multiple workers
	t.Run("MultipleWorkers", func(t *testing.T) {
		mockDataSource := NewMockDataSource(20)
		config := AsyncDataLoaderConfig{
			BatchSize:     4,
			PrefetchDepth: 5,
			Workers:       4, // Multiple workers
			MemoryManager: memoryManager,
		}

		loader, err := NewAsyncDataLoader(mockDataSource, config)
		if err != nil {
			t.Fatalf("Failed to create data loader: %v", err)
		}
		defer loader.Stop()

		err = loader.Start()
		if err != nil {
			t.Fatalf("Failed to start data loader: %v", err)
		}

		// Get batches quickly to stress test workers
		startTime := time.Now()
		numBatches := 10
		var batches []*GPUBatch

		for i := 0; i < numBatches; i++ {
			batch, err := loader.GetBatch()
			if err != nil {
				t.Fatalf("Failed to get batch %d: %v", i, err)
			}
			batches = append(batches, batch)
		}

		elapsed := time.Since(startTime)
		t.Logf("✅ Got %d batches in %v with %d workers", numBatches, elapsed, config.Workers)

		// Cleanup
		for _, batch := range batches {
			batch.Release()
		}

		stats := loader.Stats()
		if stats.Workers != 4 {
			t.Errorf("Expected 4 workers, got %d", stats.Workers)
		}
	})

	// Test 2: Worker error handling
	t.Run("WorkerErrorHandling", func(t *testing.T) {
		// Create a data source that will fail after a few batches
		failingDataSource := NewMockDataSource(5) // More batches to account for prefetching
		config := AsyncDataLoaderConfig{
			BatchSize:     4,
			PrefetchDepth: 1, // Reduced prefetch to make test more predictable
			Workers:       1, // Single worker for predictability
			MemoryManager: memoryManager,
		}

		loader, err := NewAsyncDataLoader(failingDataSource, config)
		if err != nil {
			t.Fatalf("Failed to create data loader: %v", err)
		}
		defer loader.Stop()

		err = loader.Start()
		if err != nil {
			t.Fatalf("Failed to start data loader: %v", err)
		}

		// Get the available batches (should succeed)
		var batches []*GPUBatch
		for i := 0; i < 3; i++ {
			batch, err := loader.GetBatch()
			if err != nil {
				t.Logf("Got error at batch %d (expected): %v", i, err)
				break
			}
			batches = append(batches, batch)
		}

		// Continue getting batches until we exhaust the data source
		for {
			batch, err := loader.GetBatch()
			if err != nil {
				t.Logf("✅ Got expected error from worker: %v", err)
				break
			}
			batches = append(batches, batch)
			if len(batches) > 10 { // Safety limit
				t.Error("Got too many batches, data source should have been exhausted")
				break
			}
		}

		// Cleanup
		for _, batch := range batches {
			batch.Release()
		}
	})
}

// TestAsyncDataLoaderConcurrency tests concurrent access patterns
func TestAsyncDataLoaderConcurrency(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for concurrency test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	mockDataSource := NewMockDataSource(50)
	config := AsyncDataLoaderConfig{
		BatchSize:     8,
		PrefetchDepth: 5,
		Workers:       3,
		MemoryManager: memoryManager,
	}

	loader, err := NewAsyncDataLoader(mockDataSource, config)
	if err != nil {
		t.Fatalf("Failed to create data loader: %v", err)
	}
	defer loader.Stop()

	err = loader.Start()
	if err != nil {
		t.Fatalf("Failed to start data loader: %v", err)
	}

	// Test concurrent GetBatch calls
	numConsumers := 3
	batchesPerConsumer := 5
	var wg sync.WaitGroup
	var mu sync.Mutex
	var allBatches []*GPUBatch
	var errors []error

	t.Logf("Starting %d concurrent consumers, %d batches each", numConsumers, batchesPerConsumer)

	for consumerID := 0; consumerID < numConsumers; consumerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			var consumerBatches []*GPUBatch
			for i := 0; i < batchesPerConsumer; i++ {
				batch, err := loader.GetBatch()
				if err != nil {
					mu.Lock()
					errors = append(errors, err)
					mu.Unlock()
					return
				}
				consumerBatches = append(consumerBatches, batch)
			}

			mu.Lock()
			allBatches = append(allBatches, consumerBatches...)
			mu.Unlock()

			t.Logf("✅ Consumer %d completed %d batches", id, len(consumerBatches))
		}(consumerID)
	}

	wg.Wait()

	// Check for errors
	if len(errors) > 0 {
		t.Errorf("Got %d errors during concurrent access: %v", len(errors), errors[0])
	}

	// Verify we got the expected number of batches
	expectedBatches := numConsumers * batchesPerConsumer
	if len(allBatches) != expectedBatches {
		t.Errorf("Expected %d total batches, got %d", expectedBatches, len(allBatches))
	}

	// Verify batch IDs are unique
	batchIDs := make(map[uint64]bool)
	for _, batch := range allBatches {
		if batchIDs[batch.BatchID] {
			t.Errorf("Duplicate batch ID found: %d", batch.BatchID)
		}
		batchIDs[batch.BatchID] = true
	}

	// Cleanup
	for _, batch := range allBatches {
		batch.Release()
	}

	finalStats := loader.Stats()
	t.Logf("✅ Concurrency test completed: %d batches produced", finalStats.BatchesProduced)
}

// TestAsyncDataLoaderMemoryManagement tests memory allocation and cleanup
func TestAsyncDataLoaderMemoryManagement(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for memory management test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	mockDataSource := NewMockDataSource(25) // Increased for stress test
	config := AsyncDataLoaderConfig{
		BatchSize:     4,
		PrefetchDepth: 2,
		Workers:       1,
		MemoryManager: memoryManager,
	}

	// Test 1: Memory allocation during normal operation
	t.Run("NormalOperation", func(t *testing.T) {
		loader, err := NewAsyncDataLoader(mockDataSource, config)
		if err != nil {
			t.Fatalf("Failed to create data loader: %v", err)
		}

		err = loader.Start()
		if err != nil {
			t.Fatalf("Failed to start data loader: %v", err)
		}

		// Get and immediately release batches to test memory lifecycle
		for i := 0; i < 5; i++ {
			batch, err := loader.GetBatch()
			if err != nil {
				t.Fatalf("Failed to get batch %d: %v", i, err)
			}

			// Verify tensors are allocated
			if batch.InputTensor == nil {
				t.Errorf("Batch %d InputTensor is nil", i)
			}
			if batch.LabelTensor == nil {
				t.Errorf("Batch %d LabelTensor is nil", i)
			}

			// Release immediately
			batch.Release()
		}

		err = loader.Stop()
		if err != nil {
			t.Errorf("Failed to stop loader: %v", err)
		}

		t.Log("✅ Normal operation memory management test passed")
	})

	// Test 2: Memory cleanup on stop
	t.Run("MemoryCleanupOnStop", func(t *testing.T) {
		loader, err := NewAsyncDataLoader(mockDataSource, config)
		if err != nil {
			t.Fatalf("Failed to create data loader: %v", err)
		}

		err = loader.Start()
		if err != nil {
			t.Fatalf("Failed to start data loader: %v", err)
		}

		// Let some batches accumulate in the channel
		time.Sleep(100 * time.Millisecond)

		// Stop should clean up any queued batches
		err = loader.Stop()
		if err != nil {
			t.Errorf("Failed to stop loader: %v", err)
		}

		// Verify channel is closed
		batch, err := loader.GetBatch()
		if batch != nil {
			batch.Release()
		}
		if err == nil {
			t.Error("Expected error getting batch after stop")
		}

		t.Log("✅ Memory cleanup on stop test passed")
	})

	// Test 3: Stress test memory allocation
	t.Run("StressTestMemoryAllocation", func(t *testing.T) {
		loader, err := NewAsyncDataLoader(mockDataSource, config)
		if err != nil {
			t.Fatalf("Failed to create data loader: %v", err)
		}
		defer loader.Stop()

		err = loader.Start()
		if err != nil {
			t.Fatalf("Failed to start data loader: %v", err)
		}

		// Rapidly allocate and release batches
		numIterations := 5 // Reduced to avoid exhausting data source
		for i := 0; i < numIterations; i++ {
			batch, err := loader.GetBatch()
			if err != nil {
				t.Fatalf("Failed to get batch %d: %v", i, err)
			}

			// Verify tensors have correct shapes
			inputShape := batch.InputTensor.Shape()
			if len(inputShape) != 4 || inputShape[0] != config.BatchSize {
				t.Errorf("Batch %d: incorrect input shape %v", i, inputShape)
			}

			labelShape := batch.LabelTensor.Shape()
			if len(labelShape) != 1 || labelShape[0] != config.BatchSize {
				t.Errorf("Batch %d: incorrect label shape %v", i, labelShape)
			}

			batch.Release()
		}

		stats := loader.Stats()
		t.Logf("✅ Stress test completed: %d batches produced", stats.BatchesProduced)
	})
}

// TestAsyncDataLoaderStats tests statistics reporting
func TestAsyncDataLoaderStats(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for stats test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	mockDataSource := NewMockDataSource(15)
	config := AsyncDataLoaderConfig{
		BatchSize:     8,
		PrefetchDepth: 4,
		Workers:       2,
		MemoryManager: memoryManager,
	}

	loader, err := NewAsyncDataLoader(mockDataSource, config)
	if err != nil {
		t.Fatalf("Failed to create data loader: %v", err)
	}
	defer loader.Stop()

	// Test 1: Initial stats
	stats := loader.Stats()
	if stats.IsRunning {
		t.Error("Loader should not be running initially")
	}
	if stats.BatchesProduced != 0 {
		t.Errorf("Expected 0 batches produced initially, got %d", stats.BatchesProduced)
	}
	if stats.QueuedBatches != 0 {
		t.Errorf("Expected 0 queued batches initially, got %d", stats.QueuedBatches)
	}
	if stats.QueueCapacity != config.PrefetchDepth {
		t.Errorf("Expected queue capacity %d, got %d", config.PrefetchDepth, stats.QueueCapacity)
	}
	if stats.Workers != config.Workers {
		t.Errorf("Expected %d workers, got %d", config.Workers, stats.Workers)
	}
	if stats.Generation != 1 {
		t.Errorf("Expected generation 1, got %d", stats.Generation)
	}

	// Test 2: Stats after starting
	err = loader.Start()
	if err != nil {
		t.Fatalf("Failed to start data loader: %v", err)
	}

	stats = loader.Stats()
	if !stats.IsRunning {
		t.Error("Loader should be running after start")
	}

	// Test 3: Stats during operation
	var batches []*GPUBatch
	numBatches := 5

	for i := 0; i < numBatches; i++ {
		batch, err := loader.GetBatch()
		if err != nil {
			t.Fatalf("Failed to get batch %d: %v", i, err)
		}
		batches = append(batches, batch)

		stats = loader.Stats()
		if stats.BatchesProduced < uint64(i+1) {
			t.Errorf("After getting %d batches, expected at least %d produced, got %d", 
				i+1, i+1, stats.BatchesProduced)
		}
	}

	// Test 4: Stats progression
	finalStats := loader.Stats()
	t.Logf("Final stats: Running=%t, Produced=%d, Queued=%d, Capacity=%d, Workers=%d, Gen=%d",
		finalStats.IsRunning, finalStats.BatchesProduced, finalStats.QueuedBatches,
		finalStats.QueueCapacity, finalStats.Workers, finalStats.Generation)

	if finalStats.BatchesProduced < uint64(numBatches) {
		t.Errorf("Expected at least %d batches produced, got %d", 
			numBatches, finalStats.BatchesProduced)
	}

	// Cleanup batches
	for _, batch := range batches {
		batch.Release()
	}

	// Test 5: Stats after stopping
	err = loader.Stop()
	if err != nil {
		t.Errorf("Failed to stop loader: %v", err)
	}

	stats = loader.Stats()
	if stats.IsRunning {
		t.Error("Loader should not be running after stop")
	}

	t.Log("✅ Stats tests passed")
}

// BenchmarkAsyncDataLoader benchmarks data loader performance
func BenchmarkAsyncDataLoader(b *testing.B) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available for benchmark: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	b.Run("BatchProduction", func(b *testing.B) {
		mockDataSource := NewMockDataSource(b.N * 2) // Ensure enough data
		config := AsyncDataLoaderConfig{
			BatchSize:     32,
			PrefetchDepth: 8,
			Workers:       4,
			MemoryManager: memoryManager,
		}

		loader, err := NewAsyncDataLoader(mockDataSource, config)
		if err != nil {
			b.Fatalf("Failed to create data loader: %v", err)
		}
		defer loader.Stop()

		err = loader.Start()
		if err != nil {
			b.Fatalf("Failed to start data loader: %v", err)
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			batch, err := loader.GetBatch()
			if err != nil {
				b.Fatalf("Failed to get batch: %v", err)
			}
			batch.Release()
		}

		b.StopTimer()

		stats := loader.Stats()
		samplesPerSecond := float64(stats.BatchesProduced*uint64(config.BatchSize)) / b.Elapsed().Seconds()
		b.Logf("Throughput: %.1f samples/second", samplesPerSecond)
	})

	b.Run("TryGetBatch", func(b *testing.B) {
		mockDataSource := NewMockDataSource(b.N * 2)
		config := AsyncDataLoaderConfig{
			BatchSize:     16,
			PrefetchDepth: 4,
			Workers:       2,
			MemoryManager: memoryManager,
		}

		loader, err := NewAsyncDataLoader(mockDataSource, config)
		if err != nil {
			b.Fatalf("Failed to create data loader: %v", err)
		}
		defer loader.Stop()

		err = loader.Start()
		if err != nil {
			b.Fatalf("Failed to start data loader: %v", err)
		}

		// Let some batches accumulate
		time.Sleep(100 * time.Millisecond)

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			batch, err := loader.TryGetBatch()
			if err != nil {
				b.Fatalf("TryGetBatch failed: %v", err)
			}
			if batch != nil {
				batch.Release()
			}
		}
	})
}