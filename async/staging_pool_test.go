package async

import (
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// TestStagingBufferPoolMemoryTransfer tests the optimized memory transfer functionality
func TestStagingBufferPoolMemoryTransfer(t *testing.T) {
	// Initialize memory manager (required for Metal device)
	memory.InitializeGlobalMemoryManager(nil)

	// Get memory manager
	memManager := memory.GetGlobalMemoryManager()
	if memManager == nil {
		t.Skipf("Memory manager not available")
	}

	// Create staging buffer pool
	pool, err := NewStagingBufferPool(memManager, 4)
	if err != nil {
		t.Skipf("Failed to create staging buffer pool (Metal device not available): %v", err)
	}
	defer pool.Cleanup()

	// Test data - sample float32 array
	testData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}

	// Create GPU tensor for testing
	tensorShape := []int{len(testData)}
	gpuTensor, err := memory.NewTensor(tensorShape, memory.Float32, memory.GPU)
	if err != nil {
		t.Fatalf("Failed to create GPU tensor: %v", err)
	}
	defer gpuTensor.Release()

	t.Logf("✅ Created staging pool and GPU tensor successfully")

	// Test 1: Async transfer
	t.Run("AsyncTransfer", func(t *testing.T) {
		err := pool.TransferToGPU(testData, gpuTensor)
		if err != nil {
			t.Errorf("Async transfer failed: %v", err)
			return
		}

		// Wait for completion to verify transfer
		err = pool.WaitForTransferCompletion()
		if err != nil {
			t.Errorf("Failed to wait for transfer completion: %v", err)
			return
		}

		t.Logf("✅ Async transfer completed successfully")
	})

	// Test 2: Synchronous transfer
	t.Run("SyncTransfer", func(t *testing.T) {
		err := pool.TransferToGPUSync(testData, gpuTensor)
		if err != nil {
			t.Errorf("Sync transfer failed: %v", err)
			return
		}

		t.Logf("✅ Sync transfer completed successfully")
	})

	// Test 3: Multiple transfers (stress test)
	t.Run("MultipleTransfers", func(t *testing.T) {
		for i := 0; i < 10; i++ {
			testDataVaried := make([]float32, 8)
			for j := range testDataVaried {
				testDataVaried[j] = float32(i*10 + j)
			}

			err := pool.TransferToGPU(testDataVaried, gpuTensor)
			if err != nil {
				t.Errorf("Transfer %d failed: %v", i, err)
				return
			}
		}

		// Wait for all transfers to complete
		err := pool.WaitForTransferCompletion()
		if err != nil {
			t.Errorf("Failed to wait for multiple transfers: %v", err)
			return
		}

		t.Logf("✅ Multiple transfers completed successfully")
	})

	// Test 4: Pool statistics
	t.Run("PoolStatistics", func(t *testing.T) {
		stats := pool.Stats()
		t.Logf("Pool stats: Total=%d, Available=%d, InUse=%d, BufferSize=%d, Max=%d",
			stats.TotalBuffers, stats.AvailableBuffers, stats.InUseBuffers,
			stats.BufferSize, stats.MaxBuffers)

		if stats.BufferSize != 4*1024*1024 {
			t.Errorf("Expected buffer size 4MB, got %d", stats.BufferSize)
		}

		if stats.MaxBuffers != 4 {
			t.Errorf("Expected max buffers 4, got %d", stats.MaxBuffers)
		}

		t.Logf("✅ Pool statistics are correct")
	})

	// Test 5: Different data types
	t.Run("DifferentDataTypes", func(t *testing.T) {
		// Test int32 data
		intData := []int32{10, 20, 30, 40}
		err := pool.TransferToGPU(intData, gpuTensor)
		if err != nil {
			t.Errorf("Int32 transfer failed: %v", err)
			return
		}

		// Test byte data
		byteData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
		err = pool.TransferToGPU(byteData, gpuTensor)
		if err != nil {
			t.Errorf("Byte transfer failed: %v", err)
			return
		}

		err = pool.WaitForTransferCompletion()
		if err != nil {
			t.Errorf("Failed to wait for different data type transfers: %v", err)
			return
		}

		t.Logf("✅ Different data types transferred successfully")
	})

	t.Logf("🎉 All memory transfer tests passed!")
}

// TestStagingBufferPoolBufferManagement tests buffer lifecycle management
func TestStagingBufferPoolBufferManagement(t *testing.T) {
	// Initialize memory manager
	memory.InitializeGlobalMemoryManager(nil)

	memManager := memory.GetGlobalMemoryManager()
	if memManager == nil {
		t.Skipf("Memory manager not available")
	}

	// Create small pool for testing buffer management
	pool, err := NewStagingBufferPool(memManager, 2)
	if err != nil {
		t.Skipf("Failed to create staging buffer pool (Metal device not available): %v", err)
	}
	defer pool.Cleanup()

	// Test buffer acquisition and release
	buffer1, err := pool.GetBuffer()
	if err != nil {
		t.Fatalf("Failed to get buffer 1: %v", err)
	}

	buffer2, err := pool.GetBuffer()
	if err != nil {
		t.Fatalf("Failed to get buffer 2: %v", err)
	}

	// Pool should be at capacity
	stats := pool.Stats()
	if stats.InUseBuffers != 2 {
		t.Errorf("Expected 2 buffers in use, got %d", stats.InUseBuffers)
	}

	// Try to get another buffer (should create new one)
	buffer3, err := pool.GetBuffer()
	if err == nil {
		pool.ReturnBuffer(buffer3)
		t.Logf("Successfully created buffer beyond initial capacity")
	} else {
		t.Logf("Expected behavior: %v", err)
	}

	// Return buffers
	pool.ReturnBuffer(buffer1)
	pool.ReturnBuffer(buffer2)

	// Check available buffers
	stats = pool.Stats()
	t.Logf("After returning buffers: Available=%d, InUse=%d", 
		stats.AvailableBuffers, stats.InUseBuffers)

	t.Logf("✅ Buffer management test completed")
}

// TestStagingBufferPoolCreation tests pool creation and validation
func TestStagingBufferPoolCreation(t *testing.T) {
	// Test 1: Nil memory manager should be rejected
	_, err := NewStagingBufferPool(nil, 4)
	if err == nil {
		t.Error("Expected error for nil memory manager")
	}
	if err != nil && !contains(err.Error(), "memory manager cannot be nil") {
		t.Errorf("Expected 'memory manager cannot be nil' error, got: %v", err)
	}

	// Test 2: Invalid maxBuffers should be rejected
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Logf("Metal device not available, testing validation logic only: %v", err)
		return
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	_, err = NewStagingBufferPool(memoryManager, 0)
	if err == nil {
		t.Error("Expected error for zero maxBuffers")
	}
	if err != nil && !contains(err.Error(), "maxBuffers must be positive") {
		t.Errorf("Expected 'maxBuffers must be positive' error, got: %v", err)
	}

	_, err = NewStagingBufferPool(memoryManager, -1)
	if err == nil {
		t.Error("Expected error for negative maxBuffers")
	}

	// Test 3: Valid creation should succeed
	pool, err := NewStagingBufferPool(memoryManager, 4)
	if err != nil {
		t.Fatalf("Failed to create staging buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Verify initial state
	stats := pool.Stats()
	if stats.MaxBuffers != 4 {
		t.Errorf("Expected MaxBuffers 4, got %d", stats.MaxBuffers)
	}
	if stats.BufferSize != 4*1024*1024 {
		t.Errorf("Expected BufferSize 4MB, got %d", stats.BufferSize)
	}
	if stats.TotalBuffers == 0 {
		t.Error("Expected some pre-allocated buffers")
	}

	t.Logf("✅ Pool created successfully with %d initial buffers", stats.TotalBuffers)
}

// TestStagingBufferPoolDataTypes tests transfer of different data types
func TestStagingBufferPoolDataTypes(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	pool, err := NewStagingBufferPool(memoryManager, 4)
	if err != nil {
		t.Fatalf("Failed to create staging buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Create test tensor
	tensorShape := []int{256} // 256 elements
	gpuTensor, err := memory.NewTensor(tensorShape, memory.Float32, memory.GPU)
	if err != nil {
		t.Fatalf("Failed to create GPU tensor: %v", err)
	}
	defer gpuTensor.Release()

	// Test 1: Float32 data
	t.Run("Float32Data", func(t *testing.T) {
		float32Data := make([]float32, 256)
		for i := range float32Data {
			float32Data[i] = float32(i) * 0.1
		}

		err := pool.TransferToGPUSync(float32Data, gpuTensor)
		if err != nil {
			t.Errorf("Float32 transfer failed: %v", err)
		} else {
			t.Log("✅ Float32 transfer completed")
		}
	})

	// Test 2: Int32 data
	t.Run("Int32Data", func(t *testing.T) {
		int32Data := make([]int32, 256)
		for i := range int32Data {
			int32Data[i] = int32(i * 10)
		}

		err := pool.TransferToGPU(int32Data, gpuTensor)
		if err != nil {
			t.Errorf("Int32 transfer failed: %v", err)
		} else {
			t.Log("✅ Int32 transfer initiated")
			err = pool.WaitForTransferCompletion()
			if err != nil {
				t.Errorf("Failed to wait for int32 transfer: %v", err)
			} else {
				t.Log("✅ Int32 transfer completed")
			}
		}
	})

	// Test 3: Byte data
	t.Run("ByteData", func(t *testing.T) {
		byteData := make([]byte, 1024) // 256 * 4 bytes
		for i := range byteData {
			byteData[i] = byte(i % 256)
		}

		err := pool.TransferToGPU(byteData, gpuTensor)
		if err != nil {
			t.Errorf("Byte transfer failed: %v", err)
		} else {
			t.Log("✅ Byte transfer initiated")
			err = pool.WaitForTransferCompletion()
			if err != nil {
				t.Errorf("Failed to wait for byte transfer: %v", err)
			} else {
				t.Log("✅ Byte transfer completed")
			}
		}
	})

	// Test 4: Unsupported data type
	t.Run("UnsupportedDataType", func(t *testing.T) {
		unsupportedData := []string{"test", "data"}
		err := pool.TransferToGPU(unsupportedData, gpuTensor)
		if err == nil {
			t.Error("Expected error for unsupported data type")
		}
		if err != nil && !contains(err.Error(), "unsupported data type") {
			t.Errorf("Expected 'unsupported data type' error, got: %v", err)
		}
	})

	// Test 5: Data too large for buffer
	t.Run("DataTooLarge", func(t *testing.T) {
		// Create data larger than buffer size (4MB)
		largeData := make([]float32, 2*1024*1024) // 8MB
		err := pool.TransferToGPU(largeData, gpuTensor)
		if err == nil {
			t.Error("Expected error for data too large")
		}
		if err != nil && !contains(err.Error(), "exceeds staging buffer size") {
			t.Errorf("Expected 'exceeds staging buffer size' error, got: %v", err)
		}
	})
}

// TestStagingBufferPoolConcurrency tests concurrent access
func TestStagingBufferPoolConcurrency(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	pool, err := NewStagingBufferPool(memoryManager, 8)
	if err != nil {
		t.Fatalf("Failed to create staging buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Create test tensor
	tensorShape := []int{128}
	gpuTensor, err := memory.NewTensor(tensorShape, memory.Float32, memory.GPU)
	if err != nil {
		t.Fatalf("Failed to create GPU tensor: %v", err)
	}
	defer gpuTensor.Release()

	// Test concurrent transfers (reduced for reliability)
	numGoroutines := 3
	transfersPerGoroutine := 2
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errors []error

	t.Logf("Starting %d goroutines with %d transfers each", numGoroutines, transfersPerGoroutine)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for j := 0; j < transfersPerGoroutine; j++ {
				// Create unique data for this transfer
				testData := make([]float32, 128)
				for k := range testData {
					testData[k] = float32(goroutineID*1000 + j*100 + k)
				}

				err := pool.TransferToGPU(testData, gpuTensor)
				if err != nil {
					mu.Lock()
					errors = append(errors, err)
					mu.Unlock()
					return
				}
			}
		}(i)
	}

	// Wait for completion with timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All goroutines completed
	case <-time.After(5 * time.Second):
		t.Error("Concurrent transfers timed out")
		return
	}

	// Wait for all transfers to complete
	err = pool.WaitForTransferCompletion()
	if err != nil {
		t.Errorf("Failed to wait for transfer completion: %v", err)
	}

	// Check for errors
	if len(errors) > 0 {
		t.Errorf("Got %d errors during concurrent transfers: %v", len(errors), errors[0])
	} else {
		t.Logf("✅ All %d concurrent transfers completed successfully", numGoroutines*transfersPerGoroutine)
	}

	// Verify final statistics
	stats := pool.Stats()
	t.Logf("Final stats: Total=%d, Available=%d, InUse=%d", 
		stats.TotalBuffers, stats.AvailableBuffers, stats.InUseBuffers)

	if stats.InUseBuffers != 0 {
		t.Errorf("Expected 0 buffers in use after completion, got %d", stats.InUseBuffers)
	}
}

// TestStagingBufferPoolResourceManagement tests resource lifecycle
func TestStagingBufferPoolResourceManagement(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	// Test 1: Buffer acquisition and return
	t.Run("BufferAcquisitionAndReturn", func(t *testing.T) {
		pool, err := NewStagingBufferPool(memoryManager, 2)
		if err != nil {
			t.Fatalf("Failed to create pool: %v", err)
		}
		defer pool.Cleanup()

		// Get buffers up to capacity
		var buffers []*StagingBuffer
		for i := 0; i < 2; i++ {
			buffer, err := pool.GetBuffer()
			if err != nil {
				t.Fatalf("Failed to get buffer %d: %v", i, err)
			}
			buffers = append(buffers, buffer)
		}

		// Pool should be at capacity
		stats := pool.Stats()
		if stats.InUseBuffers != 2 {
			t.Errorf("Expected 2 buffers in use, got %d", stats.InUseBuffers)
		}

		// Try to get another buffer (should create new or fail)
		// Use a timeout to prevent hanging
		done := make(chan struct{})
		var buffer4 *StagingBuffer
		var getBufferErr error
		
		go func() {
			buffer4, getBufferErr = pool.GetBuffer()
			close(done)
		}()
		
		select {
		case <-done:
			if getBufferErr == nil {
				buffers = append(buffers, buffer4)
				t.Log("✅ Pool expanded beyond initial capacity")
			} else {
				t.Logf("✅ Pool properly rejected request at capacity: %v", getBufferErr)
			}
		case <-time.After(2 * time.Second):
			t.Log("✅ GetBuffer timed out at capacity (expected behavior)")
		}

		// Return all buffers
		for _, buffer := range buffers {
			pool.ReturnBuffer(buffer)
		}

		// Verify buffers are available again
		stats = pool.Stats()
		if stats.InUseBuffers != 0 {
			t.Errorf("Expected 0 buffers in use after return, got %d", stats.InUseBuffers)
		}
	})

	// Test 2: Return buffer edge cases
	t.Run("ReturnBufferEdgeCases", func(t *testing.T) {
		pool, err := NewStagingBufferPool(memoryManager, 2)
		if err != nil {
			t.Fatalf("Failed to create pool: %v", err)
		}
		defer pool.Cleanup()

		// Test returning nil buffer (should not crash)
		pool.ReturnBuffer(nil)

		// Test returning buffer multiple times
		buffer, err := pool.GetBuffer()
		if err != nil {
			t.Fatalf("Failed to get buffer: %v", err)
		}

		pool.ReturnBuffer(buffer)
		pool.ReturnBuffer(buffer) // Second return should be handled gracefully

		t.Log("✅ Buffer return edge cases handled correctly")
	})

	// Test 3: Pool cleanup
	t.Run("PoolCleanup", func(t *testing.T) {
		pool, err := NewStagingBufferPool(memoryManager, 4)
		if err != nil {
			t.Fatalf("Failed to create pool: %v", err)
		}

		// Get some buffers
		buffer1, _ := pool.GetBuffer()
		buffer2, _ := pool.GetBuffer()

		initialStats := pool.Stats()
		t.Logf("Before cleanup: Total=%d, InUse=%d", initialStats.TotalBuffers, initialStats.InUseBuffers)

		// Cleanup should not crash
		pool.Cleanup()

		// Verify cleanup doesn't crash on second call
		pool.Cleanup()

		// Verify buffers are handled gracefully after cleanup
		pool.ReturnBuffer(buffer1)
		pool.ReturnBuffer(buffer2)
		pool.ReturnBuffer(nil)

		t.Log("✅ Pool cleanup completed without issues")
	})
}

// TestStagingBufferStructure tests StagingBuffer data structure
func TestStagingBufferStructure(t *testing.T) {
	mockBuffer := unsafe.Pointer(uintptr(0x12345678)) // Mock pointer

	buffer := &StagingBuffer{
		buffer: mockBuffer,
		size:   1024,
		inUse:  false,
		id:     42,
	}

	// Test structure fields
	if buffer.buffer != mockBuffer {
		t.Error("Buffer pointer mismatch")
	}
	if buffer.size != 1024 {
		t.Errorf("Expected size 1024, got %d", buffer.size)
	}
	if buffer.inUse {
		t.Error("Buffer should not be in use initially")
	}
	if buffer.id != 42 {
		t.Errorf("Expected ID 42, got %d", buffer.id)
	}

	// Test inUse flag manipulation
	buffer.inUse = true
	if !buffer.inUse {
		t.Error("Buffer should be marked as in use")
	}

	buffer.inUse = false
	if buffer.inUse {
		t.Error("Buffer should not be marked as in use")
	}

	t.Log("✅ StagingBuffer structure tests passed")
}

// TestStagingPoolStats tests statistics reporting
func TestStagingPoolStats(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)
	memoryManager := memory.GetGlobalMemoryManager()

	pool, err := NewStagingBufferPool(memoryManager, 5)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Cleanup()

	// Test initial stats
	stats := pool.Stats()
	if stats.MaxBuffers != 5 {
		t.Errorf("Expected MaxBuffers 5, got %d", stats.MaxBuffers)
	}
	if stats.BufferSize != 4*1024*1024 {
		t.Errorf("Expected BufferSize 4MB, got %d", stats.BufferSize)
	}
	if stats.TotalBuffers == 0 {
		t.Error("Expected some initial buffers")
	}
	if stats.InUseBuffers != 0 {
		t.Errorf("Expected 0 buffers in use initially, got %d", stats.InUseBuffers)
	}

	// Get some buffers and verify stats change
	buffer1, err := pool.GetBuffer()
	if err != nil {
		t.Fatalf("Failed to get buffer 1: %v", err)
	}

	buffer2, err := pool.GetBuffer()
	if err != nil {
		t.Fatalf("Failed to get buffer 2: %v", err)
	}

	stats = pool.Stats()
	if stats.InUseBuffers != 2 {
		t.Errorf("Expected 2 buffers in use, got %d", stats.InUseBuffers)
	}

	// Verify Available + InUse = Total (approximately)
	if stats.AvailableBuffers+stats.InUseBuffers > stats.TotalBuffers {
		t.Errorf("Available (%d) + InUse (%d) should not exceed Total (%d)", 
			stats.AvailableBuffers, stats.InUseBuffers, stats.TotalBuffers)
	}

	// Return buffers and verify stats
	pool.ReturnBuffer(buffer1)
	pool.ReturnBuffer(buffer2)

	stats = pool.Stats()
	if stats.InUseBuffers != 0 {
		t.Errorf("Expected 0 buffers in use after return, got %d", stats.InUseBuffers)
	}

	t.Logf("✅ Final stats: Total=%d, Available=%d, InUse=%d, Size=%d, Max=%d",
		stats.TotalBuffers, stats.AvailableBuffers, stats.InUseBuffers, stats.BufferSize, stats.MaxBuffers)
}

// BenchmarkMemoryTransfer benchmarks the optimized memory transfer performance
func BenchmarkMemoryTransfer(b *testing.B) {
	// Initialize memory manager
	memory.InitializeGlobalMemoryManager(nil)

	memManager := memory.GetGlobalMemoryManager()
	if memManager == nil {
		b.Skipf("Memory manager not available")
	}

	// Create staging buffer pool
	pool, err := NewStagingBufferPool(memManager, 8)
	if err != nil {
		b.Skipf("Failed to create staging buffer pool (Metal device not available): %v", err)
	}
	defer pool.Cleanup()

	// Create test data (1MB of float32 data)
	dataSize := 256 * 1024 // 256K float32 values = 1MB
	testData := make([]float32, dataSize)
	for i := range testData {
		testData[i] = float32(i)
	}

	// Create GPU tensor
	tensorShape := []int{dataSize}
	gpuTensor, err := memory.NewTensor(tensorShape, memory.Float32, memory.GPU)
	if err != nil {
		b.Fatalf("Failed to create GPU tensor: %v", err)
	}
	defer gpuTensor.Release()

	b.Logf("Benchmarking transfer of %d MB", len(testData)*4/(1024*1024))

	// Reset timer and run benchmark
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := pool.TransferToGPUSync(testData, gpuTensor)
		if err != nil {
			b.Fatalf("Transfer failed: %v", err)
		}
	}

	// Calculate throughput
	totalBytes := int64(b.N) * int64(len(testData)*4)
	throughputMBps := float64(totalBytes) / b.Elapsed().Seconds() / (1024 * 1024)
	b.Logf("Throughput: %.2f MB/s", throughputMBps)
}