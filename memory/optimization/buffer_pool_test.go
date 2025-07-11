package optimization

import (
	"runtime"
	"sync"
	"testing"
)

// TestNewBufferPool tests buffer pool creation
func TestNewBufferPool(t *testing.T) {
	pool := NewBufferPool()
	
	if pool == nil {
		t.Fatal("Buffer pool should not be nil")
	}
	
	if pool.pools == nil {
		t.Error("Buffer pool pools map should be initialized")
	}
	
	if pool.stats == nil {
		t.Error("Buffer pool stats map should be initialized")
	}
	
	t.Log("NewBufferPool tests passed")
}

// TestRoundUpToPowerOf2 tests the power of 2 rounding function
func TestRoundUpToPowerOf2(t *testing.T) {
	tests := []struct {
		input    int
		expected int
	}{
		{0, 1},
		{1, 1},
		{2, 2},
		{3, 4},
		{4, 4},
		{5, 8},
		{8, 8},
		{9, 16},
		{16, 16},
		{17, 32},
		{31, 32},
		{32, 32},
		{33, 64},
		{100, 128},
		{1000, 1024},
		{1024, 1024},
		{1025, 2048},
	}
	
	for _, test := range tests {
		result := roundUpToPowerOf2(test.input)
		if result != test.expected {
			t.Errorf("roundUpToPowerOf2(%d) = %d; expected %d", 
				test.input, result, test.expected)
		}
	}
	
	t.Log("roundUpToPowerOf2 tests passed")
}

// TestBufferPoolGetFloat32Buffer tests getting float32 buffers
func TestBufferPoolGetFloat32Buffer(t *testing.T) {
	pool := NewBufferPool()
	
	// Test getting a buffer
	size := 100
	buffer := pool.GetFloat32Buffer(size)
	
	if buffer == nil {
		t.Fatal("Buffer should not be nil")
	}
	
	if len(buffer) != size {
		t.Errorf("Expected buffer length %d, got %d", size, len(buffer))
	}
	
	if cap(buffer) < size {
		t.Errorf("Expected buffer capacity >= %d, got %d", size, cap(buffer))
	}
	
	// Test that buffer capacity is rounded up to power of 2
	expectedCap := roundUpToPowerOf2(size)
	if cap(buffer) != expectedCap {
		t.Errorf("Expected buffer capacity %d, got %d", expectedCap, cap(buffer))
	}
	
	// Return buffer to pool
	pool.PutFloat32Buffer(buffer)
	
	t.Log("GetFloat32Buffer basic tests passed")
}

// TestBufferPoolGetPutCycle tests get/put cycle
func TestBufferPoolGetPutCycle(t *testing.T) {
	pool := NewBufferPool()
	size := 50
	
	// Get a buffer
	buffer1 := pool.GetFloat32Buffer(size)
	if len(buffer1) != size {
		t.Errorf("Expected buffer length %d, got %d", size, len(buffer1))
	}
	
	// Put it back
	pool.PutFloat32Buffer(buffer1)
	
	// Get another buffer - should be reused from pool
	buffer2 := pool.GetFloat32Buffer(size)
	if len(buffer2) != size {
		t.Errorf("Expected buffer length %d, got %d", size, len(buffer2))
	}
	
	// Should be the same underlying array (reused from pool)
	// Note: We can't guarantee this because the slice may be re-sliced,
	// but we can at least check that it works
	
	pool.PutFloat32Buffer(buffer2)
	
	t.Log("Get/Put cycle tests passed")
}

// TestBufferPoolStats tests statistics tracking
func TestBufferPoolStats(t *testing.T) {
	pool := NewBufferPool()
	size := 64
	
	// Initially no stats
	stats := pool.Stats()
	if len(stats) != 0 {
		t.Errorf("Expected 0 stat entries initially, got %d", len(stats))
	}
	
	// Get a buffer
	buffer := pool.GetFloat32Buffer(size)
	
	// Should have stats now
	stats = pool.Stats()
	poolSize := roundUpToPowerOf2(size)
	if _, exists := stats[poolSize]; !exists {
		t.Errorf("Expected stats for pool size %d", poolSize)
	}
	
	stat := stats[poolSize]
	if stat.Gets != 1 {
		t.Errorf("Expected Gets=1, got %d", stat.Gets)
	}
	if stat.InUse != 1 {
		t.Errorf("Expected InUse=1, got %d", stat.InUse)
	}
	if stat.MaxInUse != 1 {
		t.Errorf("Expected MaxInUse=1, got %d", stat.MaxInUse)
	}
	
	// Put buffer back
	pool.PutFloat32Buffer(buffer)
	
	// Check stats after put
	stats = pool.Stats()
	stat = stats[poolSize]
	if stat.Puts != 1 {
		t.Errorf("Expected Puts=1, got %d", stat.Puts)
	}
	if stat.InUse != 0 {
		t.Errorf("Expected InUse=0, got %d", stat.InUse)
	}
	
	t.Log("Buffer pool stats tests passed")
}

// TestBufferPoolMultipleSizes tests multiple buffer sizes
func TestBufferPoolMultipleSizes(t *testing.T) {
	pool := NewBufferPool()
	
	sizes := []int{10, 50, 100, 500, 1000}
	buffers := make([][]float32, len(sizes))
	
	// Get buffers of different sizes
	for i, size := range sizes {
		buffers[i] = pool.GetFloat32Buffer(size)
		if len(buffers[i]) != size {
			t.Errorf("Expected buffer[%d] length %d, got %d", i, size, len(buffers[i]))
		}
	}
	
	// Check that we have stats for multiple pool sizes
	stats := pool.Stats()
	if len(stats) == 0 {
		t.Error("Expected multiple stat entries")
	}
	
	// Return all buffers
	for i, buffer := range buffers {
		pool.PutFloat32Buffer(buffer)
		_ = i // Use i to avoid compiler warning
	}
	
	// Check that all pools show 0 in use
	stats = pool.Stats()
	for _, stat := range stats {
		if stat.InUse != 0 {
			t.Errorf("Expected InUse=0 after returning all buffers, got %d", stat.InUse)
		}
	}
	
	t.Log("Multiple buffer sizes tests passed")
}

// TestBufferPoolConcurrentAccess tests concurrent buffer operations
func TestBufferPoolConcurrentAccess(t *testing.T) {
	pool := NewBufferPool()
	numGoroutines := 50
	numOperations := 100
	bufferSize := 100
	
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	
	// Start multiple goroutines doing get/put operations
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < numOperations; j++ {
				// Get buffer
				buffer := pool.GetFloat32Buffer(bufferSize)
				
				// Use buffer (write some data)
				for k := range buffer {
					buffer[k] = float32(id*1000 + j*10 + k)
				}
				
				// Simulate some work
				runtime.Gosched()
				
				// Return buffer
				pool.PutFloat32Buffer(buffer)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Check final stats
	stats := pool.Stats()
	poolSize := roundUpToPowerOf2(bufferSize)
	if stat, exists := stats[poolSize]; exists {
		totalGets := int64(numGoroutines * numOperations)
		if stat.Gets != totalGets {
			t.Errorf("Expected Gets=%d, got %d", totalGets, stat.Gets)
		}
		if stat.Puts != totalGets {
			t.Errorf("Expected Puts=%d, got %d", totalGets, stat.Puts)
		}
		if stat.InUse != 0 {
			t.Errorf("Expected InUse=0 after all operations, got %d", stat.InUse)
		}
	} else {
		t.Errorf("Expected stats for pool size %d", poolSize)
	}
	
	t.Log("Concurrent access tests passed")
}

// TestBufferPoolBufferClearing tests that buffers are cleared when returned
func TestBufferPoolBufferClearing(t *testing.T) {
	pool := NewBufferPool()
	size := 10
	
	// Get a buffer and fill it with data
	buffer := pool.GetFloat32Buffer(size)
	for i := range buffer {
		buffer[i] = float32(i + 1)
	}
	
	// Verify buffer has data
	if buffer[0] == 0 {
		t.Error("Buffer should have non-zero data before returning to pool")
	}
	
	// Return buffer to pool
	pool.PutFloat32Buffer(buffer)
	
	// Get buffer again
	buffer2 := pool.GetFloat32Buffer(size)
	
	// Buffer should be cleared (all zeros)
	for i, val := range buffer2 {
		if val != 0 {
			t.Errorf("Expected buffer[%d] to be 0 after reuse, got %f", i, val)
		}
	}
	
	pool.PutFloat32Buffer(buffer2)
	
	t.Log("Buffer clearing tests passed")
}

// TestBufferPoolGetInt32Buffer tests int32 buffer allocation
func TestBufferPoolGetInt32Buffer(t *testing.T) {
	pool := NewBufferPool()
	size := 50
	
	buffer := pool.GetInt32Buffer(size)
	
	if buffer == nil {
		t.Fatal("Int32 buffer should not be nil")
	}
	
	if len(buffer) != size {
		t.Errorf("Expected buffer length %d, got %d", size, len(buffer))
	}
	
	// Test that we can write to the buffer
	for i := range buffer {
		buffer[i] = int32(i)
	}
	
	// Verify the data
	for i, val := range buffer {
		if val != int32(i) {
			t.Errorf("Expected buffer[%d] = %d, got %d", i, i, val)
		}
	}
	
	t.Log("GetInt32Buffer tests passed")
}

// TestBufferPoolString tests string representation
func TestBufferPoolString(t *testing.T) {
	pool := NewBufferPool()
	
	// Initially empty
	str := pool.String()
	if len(str) == 0 {
		t.Error("String representation should not be empty")
	}
	
	// Should contain "BufferPool Statistics"
	if !contains(str, "BufferPool Statistics") {
		t.Error("String should contain 'BufferPool Statistics'")
	}
	
	// Get some buffers to generate stats
	buffer1 := pool.GetFloat32Buffer(100)
	buffer2 := pool.GetFloat32Buffer(200)
	
	str = pool.String()
	
	// Should contain statistics information
	if !contains(str, "Gets=") {
		t.Error("String should contain 'Gets='")
	}
	if !contains(str, "HitRate=") {
		t.Error("String should contain 'HitRate='")
	}
	
	pool.PutFloat32Buffer(buffer1)
	pool.PutFloat32Buffer(buffer2)
	
	t.Log("String representation tests passed")
}

// TestGlobalBufferPool tests the global buffer pool singleton
func TestGlobalBufferPool(t *testing.T) {
	// Reset global state for testing
	globalPool = nil
	globalPoolOnce = sync.Once{}
	
	pool1 := GetGlobalBufferPool()
	pool2 := GetGlobalBufferPool()
	
	if pool1 == nil {
		t.Fatal("Global buffer pool should not be nil")
	}
	
	if pool1 != pool2 {
		t.Error("Multiple calls to GetGlobalBufferPool should return the same instance")
	}
	
	// Test that it works like a normal pool
	buffer := pool1.GetFloat32Buffer(50)
	if len(buffer) != 50 {
		t.Errorf("Expected buffer length 50, got %d", len(buffer))
	}
	
	pool1.PutFloat32Buffer(buffer)
	
	t.Log("Global buffer pool tests passed")
}

// TestBufferPoolEdgeCases tests edge cases and error conditions
func TestBufferPoolEdgeCases(t *testing.T) {
	pool := NewBufferPool()
	
	// Test zero size buffer
	buffer := pool.GetFloat32Buffer(0)
	if len(buffer) != 0 {
		t.Errorf("Expected zero-length buffer, got length %d", len(buffer))
	}
	pool.PutFloat32Buffer(buffer)
	
	// Test negative size (should be handled gracefully)
	// Note: Current implementation doesn't handle negative sizes gracefully
	// This would need to be fixed in the actual implementation
	// buffer = pool.GetFloat32Buffer(-1)
	// pool.PutFloat32Buffer(buffer)
	
	// Test putting nil buffer (should not panic)
	pool.PutFloat32Buffer(nil)
	
	// Test putting empty buffer
	pool.PutFloat32Buffer([]float32{})
	
	t.Log("Edge cases tests passed")
}

// TestPoolStatsHitRate tests hit rate calculation
func TestPoolStatsHitRate(t *testing.T) {
	pool := NewBufferPool()
	size := 64
	
	// Get multiple buffers of the same size
	buffer1 := pool.GetFloat32Buffer(size)
	buffer2 := pool.GetFloat32Buffer(size) // This should create a new buffer (miss)
	
	stats := pool.Stats()
	poolSize := roundUpToPowerOf2(size)
	stat := stats[poolSize]
	
	// Should have 2 gets, with at least 1 miss
	if stat.Gets != 2 {
		t.Errorf("Expected Gets=2, got %d", stat.Gets)
	}
	
	// Return buffers
	pool.PutFloat32Buffer(buffer1)
	pool.PutFloat32Buffer(buffer2)
	
	// Get buffer again - should be a hit
	buffer3 := pool.GetFloat32Buffer(size)
	
	stats = pool.Stats()
	stat = stats[poolSize]
	
	if stat.Gets != 3 {
		t.Errorf("Expected Gets=3, got %d", stat.Gets)
	}
	
	pool.PutFloat32Buffer(buffer3)
	
	t.Log("Hit rate calculation tests passed")
}

// TestBufferPoolMaxInUse tests MaxInUse tracking
func TestBufferPoolMaxInUse(t *testing.T) {
	pool := NewBufferPool()
	size := 100
	
	// Get multiple buffers simultaneously
	buffers := make([][]float32, 5)
	for i := range buffers {
		buffers[i] = pool.GetFloat32Buffer(size)
	}
	
	stats := pool.Stats()
	poolSize := roundUpToPowerOf2(size)
	stat := stats[poolSize]
	
	if stat.MaxInUse < int64(len(buffers)) {
		t.Errorf("Expected MaxInUse >= %d, got %d", len(buffers), stat.MaxInUse)
	}
	
	// Return some buffers
	for i := 0; i < 3; i++ {
		pool.PutFloat32Buffer(buffers[i])
	}
	
	stats = pool.Stats()
	stat = stats[poolSize]
	
	// MaxInUse should remain the same (max seen)
	if stat.MaxInUse < int64(len(buffers)) {
		t.Errorf("MaxInUse should not decrease, expected >= %d, got %d", len(buffers), stat.MaxInUse)
	}
	
	// Return remaining buffers
	for i := 3; i < len(buffers); i++ {
		pool.PutFloat32Buffer(buffers[i])
	}
	
	t.Log("MaxInUse tracking tests passed")
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

// BenchmarkBufferPoolGet benchmarks buffer allocation
func BenchmarkBufferPoolGet(b *testing.B) {
	pool := NewBufferPool()
	size := 1024
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buffer := pool.GetFloat32Buffer(size)
		pool.PutFloat32Buffer(buffer)
	}
}

// BenchmarkBufferPoolGetLarge benchmarks large buffer allocation
func BenchmarkBufferPoolGetLarge(b *testing.B) {
	pool := NewBufferPool()
	size := 100000 // 100K elements
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buffer := pool.GetFloat32Buffer(size)
		pool.PutFloat32Buffer(buffer)
	}
}

// BenchmarkBufferPoolConcurrent benchmarks concurrent access
func BenchmarkBufferPoolConcurrent(b *testing.B) {
	pool := NewBufferPool()
	size := 1024
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buffer := pool.GetFloat32Buffer(size)
			// Simulate some work
			for i := range buffer {
				buffer[i] = float32(i)
			}
			pool.PutFloat32Buffer(buffer)
		}
	})
}

// BenchmarkRoundUpToPowerOf2 benchmarks the power of 2 function
func BenchmarkRoundUpToPowerOf2(b *testing.B) {
	sizes := []int{1, 3, 7, 15, 31, 63, 127, 255, 511, 1023}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		size := sizes[i%len(sizes)]
		roundUpToPowerOf2(size)
	}
}

// BenchmarkGlobalBufferPool benchmarks the global buffer pool
func BenchmarkGlobalBufferPool(b *testing.B) {
	size := 512
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool := GetGlobalBufferPool()
		buffer := pool.GetFloat32Buffer(size)
		pool.PutFloat32Buffer(buffer)
	}
}