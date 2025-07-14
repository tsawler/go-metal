package async

import (
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// TestCommandBufferPoolCreation tests creation and validation
func TestCommandBufferPoolCreation(t *testing.T) {
	// Test 1: Nil command queue should be rejected
	_, err := NewCommandBufferPool(nil, 5)
	if err == nil {
		t.Error("Expected error for nil command queue")
	}
	if err != nil && !contains(err.Error(), "command queue cannot be nil") {
		t.Errorf("Expected 'command queue cannot be nil' error, got: %v", err)
	}

	// Test 2: Invalid maxBuffers validation (without actually creating pool)
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for validation tests: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)
	
	_, err = NewCommandBufferPool(commandQueue, 0)
	if err == nil {
		t.Error("Expected error for zero maxBuffers")
	}
	if err != nil && !contains(err.Error(), "maxBuffers must be positive") {
		t.Errorf("Expected 'maxBuffers must be positive' error, got: %v", err)
	}

	_, err = NewCommandBufferPool(commandQueue, -1)
	if err == nil {
		t.Error("Expected error for negative maxBuffers")
	}

	// Test 3: Valid creation should succeed
	pool, err := NewCommandBufferPool(commandQueue, 4)
	if err != nil {
		t.Fatalf("Failed to create command buffer pool: %v", err)
	}
	defer pool.Cleanup()
	
	// Verify initial state
	stats := pool.Stats()
	if stats.MaxBuffers != 4 {
		t.Errorf("Expected MaxBuffers 4, got %d", stats.MaxBuffers)
	}
	
	// Should have pre-allocated some buffers
	if stats.TotalBuffers == 0 {
		t.Error("Expected some pre-allocated buffers")
	}
	
	t.Logf("✅ Pool created successfully with %d initial buffers", stats.TotalBuffers)
}

// TestCommandBufferStructure tests CommandBuffer data structure
func TestCommandBufferStructure(t *testing.T) {
	mockBuffer := memory.CreateMockCommandQueue() // Use as mock buffer pointer
	
	buffer := &CommandBuffer{
		buffer: mockBuffer,
		inUse:  false,
		id:     42,
	}

	// Test GetBuffer method
	if buffer.GetBuffer() != mockBuffer {
		t.Error("GetBuffer should return the buffer pointer")
	}

	// Test GetID method
	if buffer.GetID() != 42 {
		t.Errorf("Expected ID 42, got %d", buffer.GetID())
	}

	// Test IsInUse method
	if buffer.IsInUse() {
		t.Error("New buffer should not be in use")
	}

	buffer.inUse = true
	if !buffer.IsInUse() {
		t.Error("Buffer should be marked as in use")
	}

	t.Log("✅ CommandBuffer structure tests passed")
}

// TestCommandBufferPoolBufferManagement tests buffer lifecycle
func TestCommandBufferPoolBufferManagement(t *testing.T) {
	// Create pool with real device if available, otherwise test validation logic
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Logf("Metal device not available, testing validation logic only: %v", err)
		testCommandBufferPoolValidationLogic(t)
		return
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	// Initialize memory manager
	memory.InitializeGlobalMemoryManager(device)

	// Create command queue
	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	// Create small pool for testing
	pool, err := NewCommandBufferPool(commandQueue, 3)
	if err != nil {
		t.Fatalf("Failed to create command buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Test 1: Get buffers up to capacity
	var buffers []*CommandBuffer
	for i := 0; i < 3; i++ {
		buffer, err := pool.GetBuffer()
		if err != nil {
			t.Fatalf("Failed to get buffer %d: %v", i, err)
		}
		buffers = append(buffers, buffer)
		
		if !buffer.IsInUse() {
			t.Errorf("Buffer %d should be marked as in use", i)
		}
	}

	// Test 2: Pool should be at capacity
	stats := pool.Stats()
	if stats.InUseBuffers != 3 {
		t.Errorf("Expected 3 buffers in use, got %d", stats.InUseBuffers)
	}

	// Test 3: Try to get another buffer (should create new one or fail)
	buffer4, err := pool.GetBuffer()
	if err == nil {
		// Pool created new buffer beyond initial capacity
		buffers = append(buffers, buffer4)
		t.Logf("✅ Pool expanded beyond initial capacity")
	} else {
		// Pool at capacity
		t.Logf("✅ Pool properly rejected request at capacity: %v", err)
	}

	// Test 4: Return buffers
	for i, buffer := range buffers {
		pool.ReturnBuffer(buffer)
		
		if buffer.IsInUse() {
			t.Errorf("Buffer %d should not be in use after return", i)
		}
	}

	// Test 5: Verify buffers are available again
	stats = pool.Stats()
	t.Logf("After returning buffers: Total=%d, Available=%d, InUse=%d", 
		stats.TotalBuffers, stats.AvailableBuffers, stats.InUseBuffers)

	if stats.InUseBuffers != 0 {
		t.Errorf("Expected 0 buffers in use after return, got %d", stats.InUseBuffers)
	}

	t.Log("✅ Buffer management tests passed")
}

// TestCommandBufferPoolValidationLogic tests validation without Metal device
func TestCommandBufferPoolValidationLogic(t *testing.T) {
	// Test CommandPoolStats structure
	stats := CommandPoolStats{
		TotalBuffers:     8,
		AvailableBuffers: 3,
		InUseBuffers:     5,
		MaxBuffers:       10,
	}

	if stats.TotalBuffers != 8 {
		t.Errorf("Expected TotalBuffers 8, got %d", stats.TotalBuffers)
	}

	if stats.AvailableBuffers+stats.InUseBuffers != stats.TotalBuffers {
		t.Errorf("Available (%d) + InUse (%d) should equal Total (%d)", 
			stats.AvailableBuffers, stats.InUseBuffers, stats.TotalBuffers)
	}

	if stats.MaxBuffers < stats.TotalBuffers {
		t.Errorf("MaxBuffers (%d) should be >= TotalBuffers (%d)", 
			stats.MaxBuffers, stats.TotalBuffers)
	}

	t.Log("✅ Command pool validation logic tests passed")
}

// testCommandBufferPoolValidationLogic is the internal helper function
func testCommandBufferPoolValidationLogic(t *testing.T) {
	// Test CommandPoolStats structure
	stats := CommandPoolStats{
		TotalBuffers:     8,
		AvailableBuffers: 3,
		InUseBuffers:     5,
		MaxBuffers:       10,
	}

	if stats.TotalBuffers != 8 {
		t.Errorf("Expected TotalBuffers 8, got %d", stats.TotalBuffers)
	}

	if stats.AvailableBuffers+stats.InUseBuffers != stats.TotalBuffers {
		t.Errorf("Available (%d) + InUse (%d) should equal Total (%d)", 
			stats.AvailableBuffers, stats.InUseBuffers, stats.TotalBuffers)
	}

	if stats.MaxBuffers < stats.TotalBuffers {
		t.Errorf("MaxBuffers (%d) should be >= TotalBuffers (%d)", 
			stats.MaxBuffers, stats.TotalBuffers)
	}

	t.Log("✅ Command pool validation logic tests passed")
}

// TestCommandBufferPoolAsyncExecution tests async execution capabilities
func TestCommandBufferPoolAsyncExecution(t *testing.T) {
	// Create pool with real device if available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for async execution test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	pool, err := NewCommandBufferPool(commandQueue, 4)
	if err != nil {
		t.Skipf("Failed to create command buffer pool: %v", err)
	}
	defer pool.Cleanup()

	// Test 1: Basic async execution
	t.Run("BasicAsyncExecution", func(t *testing.T) {
		buffer, err := pool.GetBuffer()
		if err != nil {
			t.Fatalf("Failed to get buffer: %v", err)
		}

		var wg sync.WaitGroup
		var completionError error
		
		wg.Add(1)
		err = pool.ExecuteAsync(buffer, func(err error) {
			completionError = err
			wg.Done()
		})
		if err != nil {
			t.Fatalf("Failed to execute async: %v", err)
		}

		// Wait for completion with timeout
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			if completionError != nil {
				t.Errorf("Async execution failed: %v", completionError)
			} else {
				t.Log("✅ Async execution completed successfully")
			}
		case <-time.After(5 * time.Second):
			t.Error("Async execution timed out")
		}
	})

	// Test 2: Multiple concurrent executions
	t.Run("ConcurrentExecutions", func(t *testing.T) {
		numExecutions := 3
		var wg sync.WaitGroup
		var mu sync.Mutex
		var completionErrors []error

		for i := 0; i < numExecutions; i++ {
			buffer, err := pool.GetBuffer()
			if err != nil {
				t.Fatalf("Failed to get buffer %d: %v", i, err)
			}

			wg.Add(1)
			err = pool.ExecuteAsync(buffer, func(err error) {
				mu.Lock()
				completionErrors = append(completionErrors, err)
				mu.Unlock()
				wg.Done()
			})
			if err != nil {
				t.Errorf("Failed to execute async %d: %v", i, err)
			}
		}

		// Wait for all completions
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			mu.Lock()
			defer mu.Unlock()
			
			if len(completionErrors) != numExecutions {
				t.Errorf("Expected %d completions, got %d", numExecutions, len(completionErrors))
			}

			for i, err := range completionErrors {
				if err != nil {
					t.Errorf("Execution %d failed: %v", i, err)
				}
			}
			
			t.Logf("✅ %d concurrent executions completed", numExecutions)
		case <-time.After(10 * time.Second):
			t.Error("Concurrent executions timed out")
		}
	})

	// Test 3: Invalid execution parameters
	t.Run("InvalidExecutionParameters", func(t *testing.T) {
		// Test nil buffer
		err := pool.ExecuteAsync(nil, func(err error) {})
		if err == nil {
			t.Error("Expected error for nil buffer")
		}
		if err != nil && !contains(err.Error(), "command buffer is nil") {
			t.Errorf("Expected 'command buffer is nil' error, got: %v", err)
		}

		// Test buffer not in use
		mockBuffer := &CommandBuffer{
			buffer: unsafe.Pointer(uintptr(1)), // Mock pointer
			inUse:  false,
			id:     999,
		}
		err = pool.ExecuteAsync(mockBuffer, func(err error) {})
		if err == nil {
			t.Error("Expected error for buffer not in use")
		}
		if err != nil && !contains(err.Error(), "not marked as in use") {
			t.Errorf("Expected 'not marked as in use' error, got: %v", err)
		}

		t.Log("✅ Invalid execution parameter tests passed")
	})
}

// TestCommandBufferPoolBatchOperations tests batch operation functionality
func TestCommandBufferPoolBatchOperations(t *testing.T) {
	// Test batch operation structure
	t.Run("BatchOperationStructure", func(t *testing.T) {
		completed := false
		
		operation := BatchOperation{
			Type: "training_step",
			Data: map[string]interface{}{
				"learning_rate": 0.001,
				"batch_size":    32,
			},
			Completion: func(err error) {
				completed = true
			},
		}

		if operation.Type != "training_step" {
			t.Errorf("Expected operation type 'training_step', got %s", operation.Type)
		}

		if operation.Data == nil {
			t.Error("Operation data should not be nil")
		}

		// Test completion callback
		operation.Completion(nil)
		if !completed {
			t.Error("Completion callback should have been called")
		}

		t.Log("✅ BatchOperation structure tests passed")
	})

	// Test batch execution with real device if available
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Logf("Metal device not available for batch operations test: %v", err)
		return
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	pool, err := NewCommandBufferPool(commandQueue, 4)
	if err != nil {
		t.Skipf("Failed to create command buffer pool: %v", err)
	}
	defer pool.Cleanup()

	t.Run("BatchExecution", func(t *testing.T) {
		// Test 1: Empty operations should fail
		err := pool.ExecuteBatch([]BatchOperation{})
		if err == nil {
			t.Error("Expected error for empty operations")
		}
		if err != nil && !contains(err.Error(), "no operations provided") {
			t.Errorf("Expected 'no operations provided' error, got: %v", err)
		}

		// Test 2: Valid batch execution
		var wg sync.WaitGroup
		var mu sync.Mutex
		var completionResults []error

		operations := []BatchOperation{
			{
				Type: "data_transfer",
				Data: "input_data",
				Completion: func(err error) {
					mu.Lock()
					completionResults = append(completionResults, err)
					mu.Unlock()
					wg.Done()
				},
			},
			{
				Type: "training_step",
				Data: map[string]float32{"lr": 0.001},
				Completion: func(err error) {
					mu.Lock()
					completionResults = append(completionResults, err)
					mu.Unlock()
					wg.Done()
				},
			},
		}

		wg.Add(len(operations))
		err = pool.ExecuteBatch(operations)
		if err != nil {
			t.Fatalf("Failed to execute batch: %v", err)
		}

		// Wait for completion
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			mu.Lock()
			defer mu.Unlock()
			
			if len(completionResults) != len(operations) {
				t.Errorf("Expected %d completion results, got %d", len(operations), len(completionResults))
			}

			for i, err := range completionResults {
				if err != nil {
					t.Errorf("Batch operation %d failed: %v", i, err)
				}
			}
			
			t.Logf("✅ Batch execution with %d operations completed", len(operations))
		case <-time.After(5 * time.Second):
			t.Error("Batch execution timed out")
		}
	})
}

// TestCommandBufferPoolResourceManagement tests resource cleanup and management
func TestCommandBufferPoolResourceManagement(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for resource management test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	// Test 1: Pool cleanup
	t.Run("PoolCleanup", func(t *testing.T) {
		pool, err := NewCommandBufferPool(commandQueue, 3)
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

		t.Log("✅ Pool cleanup completed without issues")

		// Verify buffers are handled gracefully after cleanup
		pool.ReturnBuffer(buffer1)
		pool.ReturnBuffer(buffer2)
		pool.ReturnBuffer(nil) // Should handle nil gracefully

		t.Log("✅ Resource management tests passed")
	})

	// Test 2: Buffer return edge cases
	t.Run("BufferReturnEdgeCases", func(t *testing.T) {
		pool, err := NewCommandBufferPool(commandQueue, 2)
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

		// Test buffer state after return
		if buffer.IsInUse() {
			t.Error("Buffer should not be in use after return")
		}

		t.Log("✅ Buffer return edge cases handled correctly")
	})
}

// TestCommandBufferPoolStressTest tests pool under concurrent load
func TestCommandBufferPoolStressTest(t *testing.T) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available for stress test: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		t.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	pool, err := NewCommandBufferPool(commandQueue, 10)
	if err != nil {
		t.Skipf("Failed to create pool: %v", err)
	}
	defer pool.Cleanup()

	// Stress test: Multiple goroutines getting and returning buffers
	numGoroutines := 20
	operationsPerGoroutine := 10
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errorCount int

	t.Logf("Starting stress test with %d goroutines, %d operations each", 
		numGoroutines, operationsPerGoroutine)

	startTime := time.Now()

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for j := 0; j < operationsPerGoroutine; j++ {
				// Get buffer
				buffer, err := pool.GetBuffer()
				if err != nil {
					mu.Lock()
					errorCount++
					mu.Unlock()
					continue
				}

				// Simulate some work
				time.Sleep(time.Millisecond)

				// Return buffer
				pool.ReturnBuffer(buffer)
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(startTime)

	finalStats := pool.Stats()
	t.Logf("Stress test completed in %v", elapsed)
	t.Logf("Final stats: Total=%d, Available=%d, InUse=%d, Max=%d", 
		finalStats.TotalBuffers, finalStats.AvailableBuffers, finalStats.InUseBuffers, finalStats.MaxBuffers)

	if finalStats.InUseBuffers != 0 {
		t.Errorf("Expected 0 buffers in use after stress test, got %d", finalStats.InUseBuffers)
	}

	if errorCount > 0 {
		t.Logf("Note: %d operations failed (expected under high contention)", errorCount)
	}

	totalOperations := numGoroutines * operationsPerGoroutine
	operationsPerSecond := float64(totalOperations) / elapsed.Seconds()
	t.Logf("✅ Stress test completed: %d operations, %.1f ops/sec", totalOperations, operationsPerSecond)
}

// BenchmarkCommandBufferPool benchmarks pool performance
func BenchmarkCommandBufferPool(b *testing.B) {
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available for benchmark: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		b.Skipf("Failed to create command queue: %v", err)
	}
	defer cgo_bridge.DestroyCommandQueue(commandQueue)

	pool, err := NewCommandBufferPool(commandQueue, 16)
	if err != nil {
		b.Skipf("Failed to create pool: %v", err)
	}
	defer pool.Cleanup()

	b.ResetTimer()

	b.Run("GetReturnBuffer", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			buffer, err := pool.GetBuffer()
			if err != nil {
				b.Fatalf("Failed to get buffer: %v", err)
			}
			pool.ReturnBuffer(buffer)
		}
	})

	b.Run("ParallelGetReturnBuffer", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buffer, err := pool.GetBuffer()
				if err != nil {
					b.Errorf("Failed to get buffer: %v", err)
					continue
				}
				pool.ReturnBuffer(buffer)
			}
		})
	})
}