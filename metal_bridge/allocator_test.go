package metal_bridge

import (
	"testing"
)

func TestBufferAllocator(t *testing.T) {
	// Create device and allocator
	device := CreateSystemDefaultDevice()
	if device == nil {
		t.Skip("Metal device not available")
		return
	}
	
	config := DefaultAllocatorConfig()
	allocator := NewBufferAllocator(device, config)
	defer allocator.Shutdown()
	
	t.Run("Basic allocation and release", func(t *testing.T) {
		// Allocate a buffer
		buffer, err := allocator.Allocate(1024, 0)
		if err != nil {
			t.Fatalf("Failed to allocate buffer: %v", err)
		}
		
		if buffer.length != 1024 {
			t.Errorf("Expected buffer length 1024, got %d", buffer.length)
		}
		
		if !buffer.inUse {
			t.Error("Buffer should be marked as in use")
		}
		
		if buffer.refCount != 1 {
			t.Errorf("Expected refCount 1, got %d", buffer.refCount)
		}
		
		// Release the buffer
		allocator.Release(buffer)
		
		if buffer.inUse {
			t.Error("Buffer should not be marked as in use after release")
		}
	})
	
	t.Run("Pool reuse", func(t *testing.T) {
		// Allocate and release a buffer
		buffer1, err := allocator.Allocate(2048, 0)
		if err != nil {
			t.Fatalf("Failed to allocate first buffer: %v", err)
		}
		
		allocator.Release(buffer1)
		
		// Allocate another buffer of the same size
		buffer2, err := allocator.Allocate(2048, 0)
		if err != nil {
			t.Fatalf("Failed to allocate second buffer: %v", err)
		}
		
		// Should reuse the same buffer from pool
		if buffer1 != buffer2 {
			t.Error("Expected to reuse buffer from pool")
		}
		
		allocator.Release(buffer2)
	})
	
	t.Run("Size categorization", func(t *testing.T) {
		// Test different sizes are handled properly
		sizes := []uint64{512, 1024, 2048, 4096, 8192}
		buffers := make([]*Buffer, len(sizes))
		
		// Allocate buffers of different sizes
		for i, size := range sizes {
			buffer, err := allocator.Allocate(size, 0)
			if err != nil {
				t.Fatalf("Failed to allocate buffer of size %d: %v", size, err)
			}
			buffers[i] = buffer
		}
		
		// Check that buffers were allocated with rounded-up sizes
		for i, buffer := range buffers {
			expectedSize := roundUpToPowerOf2(sizes[i])
			if uint64(buffer.length) != expectedSize {
				t.Errorf("Expected buffer length %d, got %d for input size %d", 
					expectedSize, buffer.length, sizes[i])
			}
		}
		
		// Release all buffers
		for _, buffer := range buffers {
			allocator.Release(buffer)
		}
	})
	
	t.Run("Memory statistics", func(t *testing.T) {
		// Get initial stats
		initialStats := allocator.GetMemoryStats()
		
		// Allocate some buffers
		var buffers []*Buffer
		for i := 0; i < 5; i++ {
			buffer, err := allocator.Allocate(1024, 0)
			if err != nil {
				t.Fatalf("Failed to allocate buffer %d: %v", i, err)
			}
			buffers = append(buffers, buffer)
		}
		
		// Check stats after allocation
		afterAllocStats := allocator.GetMemoryStats()
		
		if afterAllocStats.NumAllocations <= initialStats.NumAllocations {
			t.Error("Allocation count should have increased")
		}
		
		if afterAllocStats.TotalAllocated <= initialStats.TotalAllocated {
			t.Error("Total allocated memory should have increased")
		}
		
		// Release buffers
		for _, buffer := range buffers {
			allocator.Release(buffer)
		}
		
		// Check stats after release
		afterReleaseStats := allocator.GetMemoryStats()
		
		if afterReleaseStats.NumDeallocations <= initialStats.NumDeallocations {
			t.Error("Deallocation count should have increased")
		}
		
		if afterReleaseStats.TotalFree <= initialStats.TotalFree {
			t.Error("Total free memory should have increased")
		}
	})
	
	t.Run("Reference counting", func(t *testing.T) {
		buffer, err := allocator.Allocate(1024, 0)
		if err != nil {
			t.Fatalf("Failed to allocate buffer: %v", err)
		}
		
		// Test retain/release
		buffer.Retain()
		if buffer.RefCount() != 2 {
			t.Errorf("Expected refCount 2 after retain, got %d", buffer.RefCount())
		}
		
		buffer.Release()
		if buffer.RefCount() != 1 {
			t.Errorf("Expected refCount 1 after first release, got %d", buffer.RefCount())
		}
		
		buffer.Release()
		if buffer.RefCount() != 0 {
			t.Errorf("Expected refCount 0 after final release, got %d", buffer.RefCount())
		}
	})
	
	t.Run("Pool size limits", func(t *testing.T) {
		// Create allocator with small pool size for testing
		smallConfig := AllocatorConfig{
			MaxPoolSize:    2,
			MaxTotalMemory: 1024 * 1024, // 1MB
			MinBufferSize:  1024,
		}
		smallAllocator := NewBufferAllocator(device, smallConfig)
		defer smallAllocator.Shutdown()
		
		// Allocate more buffers than pool size
		var buffers []*Buffer
		for i := 0; i < 5; i++ {
			buffer, err := smallAllocator.Allocate(1024, 0)
			if err != nil {
				t.Fatalf("Failed to allocate buffer %d: %v", i, err)
			}
			buffers = append(buffers, buffer)
		}
		
		// Release all buffers - only some should go to pool
		for _, buffer := range buffers {
			smallAllocator.Release(buffer)
		}
		
		// Check that pool didn't exceed limits
		stats := smallAllocator.GetMemoryStats()
		if stats.TotalFree > uint64(smallConfig.MaxPoolSize*1024) {
			t.Errorf("Pool exceeded maximum size limit")
		}
	})
}

func TestRoundUpToPowerOf2(t *testing.T) {
	tests := []struct {
		input    uint64
		expected uint64
	}{
		{0, 1},
		{1, 1},
		{2, 2},
		{3, 4},
		{4, 4},
		{5, 8},
		{1023, 1024},
		{1024, 1024},
		{1025, 2048},
	}
	
	for _, test := range tests {
		result := roundUpToPowerOf2(test.input)
		if result != test.expected {
			t.Errorf("roundUpToPowerOf2(%d) = %d, expected %d", 
				test.input, result, test.expected)
		}
	}
}

func TestSizeToPoolKey(t *testing.T) {
	tests := []struct {
		size     uint64
		expected int
	}{
		{1, 0},     // 2^0 = 1
		{2, 1},     // 2^1 = 2
		{4, 2},     // 2^2 = 4
		{1024, 10}, // 2^10 = 1024
		{1023, 10}, // rounds up to 1024 = 2^10
	}
	
	for _, test := range tests {
		result := sizeToPoolKey(test.size)
		if result != test.expected {
			t.Errorf("sizeToPoolKey(%d) = %d, expected %d", 
				test.size, result, test.expected)
		}
	}
}

func BenchmarkBufferAllocation(b *testing.B) {
	device := CreateSystemDefaultDevice()
	if device == nil {
		b.Skip("Metal device not available")
		return
	}
	
	config := DefaultAllocatorConfig()
	allocator := NewBufferAllocator(device, config)
	defer allocator.Shutdown()
	
	b.Run("Allocate and Release", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buffer, err := allocator.Allocate(1024, 0)
			if err != nil {
				b.Fatalf("Failed to allocate: %v", err)
			}
			allocator.Release(buffer)
		}
	})
	
	b.Run("Pool Reuse", func(b *testing.B) {
		// Pre-allocate and release buffers to populate pool
		for i := 0; i < 10; i++ {
			buffer, _ := allocator.Allocate(1024, 0)
			allocator.Release(buffer)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buffer, err := allocator.Allocate(1024, 0)
			if err != nil {
				b.Fatalf("Failed to allocate: %v", err)
			}
			allocator.Release(buffer)
		}
	})
}

func TestLongRunningAllocation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}
	
	device := CreateSystemDefaultDevice()
	if device == nil {
		t.Skip("Metal device not available")
		return
	}
	
	config := DefaultAllocatorConfig()
	allocator := NewBufferAllocator(device, config)
	defer allocator.Shutdown()
	
	// Simulate long-running allocation/deallocation pattern
	const iterations = 1000
	const maxBuffers = 50
	
	var activeBuffers []*Buffer
	
	for i := 0; i < iterations; i++ {
		// Randomly allocate or release buffers
		if len(activeBuffers) == 0 || (len(activeBuffers) < maxBuffers && i%3 != 0) {
			// Allocate new buffer
			size := uint64(1024 * (1 + (i % 10))) // Vary sizes
			buffer, err := allocator.Allocate(size, 0)
			if err != nil {
				t.Fatalf("Failed to allocate buffer at iteration %d: %v", i, err)
			}
			activeBuffers = append(activeBuffers, buffer)
		} else {
			// Release a buffer
			if len(activeBuffers) > 0 {
				idx := i % len(activeBuffers)
				allocator.Release(activeBuffers[idx])
				// Remove from slice
				activeBuffers = append(activeBuffers[:idx], activeBuffers[idx+1:]...)
			}
		}
		
		// Check stats periodically
		if i%100 == 0 {
			stats := allocator.GetMemoryStats()
			t.Logf("Iteration %d: Allocated=%d, Free=%d, Pools=%d", 
				i, stats.TotalAllocated, stats.TotalFree, stats.NumPools)
		}
	}
	
	// Release all remaining buffers
	for _, buffer := range activeBuffers {
		allocator.Release(buffer)
	}
	
	finalStats := allocator.GetMemoryStats()
	t.Logf("Final stats: Allocations=%d, Deallocations=%d, Pool hits=%d, Pool misses=%d", 
		finalStats.NumAllocations, finalStats.NumDeallocations, 
		finalStats.NumPoolHits, finalStats.NumPoolMisses)
	
	// Verify no memory leaks
	if finalStats.NumAllocations != finalStats.NumDeallocations {
		t.Errorf("Memory leak detected: allocations=%d, deallocations=%d", 
			finalStats.NumAllocations, finalStats.NumDeallocations)
	}
}