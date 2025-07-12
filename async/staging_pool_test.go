package async

import (
	"testing"

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