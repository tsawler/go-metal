package dataloader

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestNewCacheManager tests cache manager creation
func TestNewCacheManager(t *testing.T) {
	maxSize := 100
	itemSize := 1000

	cm := NewCacheManager(maxSize, itemSize)

	if cm.maxSize != maxSize {
		t.Errorf("Expected max size %d, got %d", maxSize, cm.maxSize)
	}

	if cm.itemSize != itemSize {
		t.Errorf("Expected item size %d, got %d", itemSize, cm.itemSize)
	}

	if cm.currentSize != 0 {
		t.Errorf("Expected initial current size 0, got %d", cm.currentSize)
	}

	if cm.cache == nil {
		t.Error("Cache map should be initialized")
	}

	if cm.lru == nil {
		t.Error("LRU list should be initialized")
	}

	if cm.lruMap == nil {
		t.Error("LRU map should be initialized")
	}

	if cm.hits != 0 || cm.misses != 0 {
		t.Error("Statistics should be initialized to zero")
	}
}

// TestCacheManagerBasicOperations tests basic get/put operations
func TestCacheManagerBasicOperations(t *testing.T) {
	cm := NewCacheManager(5, 100)

	// Test initial get on empty cache
	data, exists := cm.Get("nonexistent")
	if exists || data != nil {
		t.Error("Get should return false and nil for nonexistent key")
	}

	stats := cm.Stats()
	if stats.Misses != 1 {
		t.Errorf("Expected 1 miss, got %d", stats.Misses)
	}

	// Test put and get
	testData := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	cm.Put("test_key", testData)

	stats = cm.Stats()
	if stats.Size != 1 {
		t.Errorf("Expected cache size 1, got %d", stats.Size)
	}

	retrievedData, exists := cm.Get("test_key")
	if !exists {
		t.Error("Get should return true for existing key")
	}

	if len(retrievedData) != len(testData) {
		t.Errorf("Expected data length %d, got %d", len(testData), len(retrievedData))
	}

	for i, val := range retrievedData {
		if val != testData[i] {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, testData[i], val)
		}
	}

	stats = cm.Stats()
	if stats.Hits != 1 {
		t.Errorf("Expected 1 hit, got %d", stats.Hits)
	}
}

// TestCacheManagerLRUEviction tests LRU eviction policy
func TestCacheManagerLRUEviction(t *testing.T) {
	cm := NewCacheManager(3, 100) // Cache with max size 3

	// Add 3 items
	cm.Put("key1", []float32{1.0})
	cm.Put("key2", []float32{2.0})
	cm.Put("key3", []float32{3.0})

	stats := cm.Stats()
	if stats.Size != 3 {
		t.Errorf("Expected cache size 3, got %d", stats.Size)
	}

	// Verify all items exist
	if _, exists := cm.Get("key1"); !exists {
		t.Error("key1 should exist")
	}
	if _, exists := cm.Get("key2"); !exists {
		t.Error("key2 should exist")
	}
	if _, exists := cm.Get("key3"); !exists {
		t.Error("key3 should exist")
	}

	// Add fourth item - should evict key1 (least recently used)
	cm.Put("key4", []float32{4.0})

	stats = cm.Stats()
	if stats.Size != 3 {
		t.Errorf("Expected cache size 3 after eviction, got %d", stats.Size)
	}

	// key1 should be evicted
	if _, exists := cm.Get("key1"); exists {
		t.Error("key1 should have been evicted")
	}

	// Other keys should still exist
	if _, exists := cm.Get("key2"); !exists {
		t.Error("key2 should still exist")
	}
	if _, exists := cm.Get("key3"); !exists {
		t.Error("key3 should still exist")
	}
	if _, exists := cm.Get("key4"); !exists {
		t.Error("key4 should exist")
	}
}

// TestCacheManagerLRUOrder tests LRU ordering with access patterns
func TestCacheManagerLRUOrder(t *testing.T) {
	cm := NewCacheManager(3, 100)

	// Add 3 items
	cm.Put("key1", []float32{1.0})
	cm.Put("key2", []float32{2.0})
	cm.Put("key3", []float32{3.0})

	// Access key1 to make it most recently used
	cm.Get("key1")

	// Add fourth item - should evict key2 (oldest unused)
	cm.Put("key4", []float32{4.0})

	// key2 should be evicted, key1 should still exist
	if _, exists := cm.Get("key2"); exists {
		t.Error("key2 should have been evicted")
	}
	if _, exists := cm.Get("key1"); !exists {
		t.Error("key1 should still exist (was accessed recently)")
	}
}

// TestCacheManagerPutExisting tests putting to existing keys
func TestCacheManagerPutExisting(t *testing.T) {
	cm := NewCacheManager(3, 100)

	// Add initial item
	cm.Put("key1", []float32{1.0})
	initialSize := cm.Stats().Size

	// Put same key again - should not increase size
	cm.Put("key1", []float32{1.0})
	
	stats := cm.Stats()
	if stats.Size != initialSize {
		t.Errorf("Expected cache size to remain %d, got %d", initialSize, stats.Size)
	}

	// Key should still be accessible
	if _, exists := cm.Get("key1"); !exists {
		t.Error("key1 should still exist after re-put")
	}
}

// TestCacheManagerStats tests statistics calculation
func TestCacheManagerStats(t *testing.T) {
	cm := NewCacheManager(5, 100)

	// Initial stats
	stats := cm.Stats()
	if stats.Size != 0 || stats.MaxSize != 5 || stats.Hits != 0 || stats.Misses != 0 {
		t.Error("Initial stats incorrect")
	}

	if stats.HitRate != 0 {
		t.Errorf("Expected initial hit rate 0, got %f", stats.HitRate)
	}

	// Add some items and test operations
	cm.Put("key1", []float32{1.0})
	cm.Put("key2", []float32{2.0})

	// Generate some hits and misses
	cm.Get("key1")      // hit
	cm.Get("key2")      // hit
	cm.Get("key3")      // miss
	cm.Get("nonexist")  // miss

	stats = cm.Stats()
	if stats.Size != 2 {
		t.Errorf("Expected size 2, got %d", stats.Size)
	}
	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}
	if stats.Misses != 2 {
		t.Errorf("Expected 2 misses, got %d", stats.Misses)
	}
	
	expectedHitRate := 50.0 // 2 hits out of 4 total = 50%
	if stats.HitRate != expectedHitRate {
		t.Errorf("Expected hit rate %f, got %f", expectedHitRate, stats.HitRate)
	}
}

// TestCacheManagerClear tests cache clearing
func TestCacheManagerClear(t *testing.T) {
	cm := NewCacheManager(5, 100)

	// Add some items
	cm.Put("key1", []float32{1.0})
	cm.Put("key2", []float32{2.0})
	cm.Get("key1") // Generate some stats

	// Verify items exist
	stats := cm.Stats()
	if stats.Size != 2 {
		t.Errorf("Expected size 2 before clear, got %d", stats.Size)
	}

	// Clear cache
	cm.Clear()

	// Verify cache is empty but stats are preserved
	stats = cm.Stats()
	if stats.Size != 0 {
		t.Errorf("Expected size 0 after clear, got %d", stats.Size)
	}

	if stats.Hits == 0 {
		t.Error("Expected stats to be preserved after clear")
	}

	// Verify items no longer exist
	if _, exists := cm.Get("key1"); exists {
		t.Error("key1 should not exist after clear")
	}
	if _, exists := cm.Get("key2"); exists {
		t.Error("key2 should not exist after clear")
	}
}

// TestCacheManagerResetStats tests statistics reset
func TestCacheManagerResetStats(t *testing.T) {
	cm := NewCacheManager(5, 100)

	// Generate some stats
	cm.Put("key1", []float32{1.0})
	cm.Get("key1")      // hit
	cm.Get("nonexist")  // miss

	stats := cm.Stats()
	if stats.Hits == 0 || stats.Misses == 0 {
		t.Error("Expected some hits and misses before reset")
	}

	// Reset stats
	cm.ResetStats()

	stats = cm.Stats()
	if stats.Hits != 0 || stats.Misses != 0 {
		t.Errorf("Expected zero stats after reset, got hits: %d, misses: %d", stats.Hits, stats.Misses)
	}

	if stats.HitRate != 0 {
		t.Errorf("Expected zero hit rate after reset, got %f", stats.HitRate)
	}

	// Cache contents should remain
	if stats.Size != 1 {
		t.Errorf("Expected cache size to remain 1, got %d", stats.Size)
	}
}

// TestCacheManagerConcurrency tests thread safety
func TestCacheManagerConcurrency(t *testing.T) {
	cm := NewCacheManager(100, 100)
	numGoroutines := 50
	numOperations := 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines performing concurrent operations
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				data := []float32{float32(id), float32(j)}

				// Put operation
				cm.Put(key, data)

				// Get operation
				if retrievedData, exists := cm.Get(key); exists {
					if len(retrievedData) != 2 || retrievedData[0] != float32(id) || retrievedData[1] != float32(j) {
						t.Errorf("Data corruption detected for key %s", key)
					}
				}

				// Stats operation
				cm.Stats()
			}
		}(i)
	}

	// Wait for all operations to complete
	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()

	select {
	case <-done:
		// Test completed successfully
	case <-time.After(10 * time.Second):
		t.Fatal("Timeout waiting for concurrent operations")
	}

	// Verify final state
	stats := cm.Stats()
	if stats.Size == 0 {
		t.Error("Expected non-zero cache size after concurrent operations")
	}

	if stats.Hits+stats.Misses == 0 {
		t.Error("Expected some cache operations to have occurred")
	}
}

// TestCacheStatsString tests the string representation of cache stats
func TestCacheStatsString(t *testing.T) {
	stats := CacheStats{
		Size:    10,
		MaxSize: 100,
		Hits:    75,
		Misses:  25,
		HitRate: 75.0,
	}

	str := stats.String()
	
	// Check that the string contains expected values
	expectedSubstrings := []string{"10/100", "75", "25", "75.0%"}
	for _, substr := range expectedSubstrings {
		if !containsString(str, substr) {
			t.Errorf("Expected stats string to contain '%s', got: %s", substr, str)
		}
	}
}

// TestCacheManagerHitRateCalculation tests hit rate calculation edge cases
func TestCacheManagerHitRateCalculation(t *testing.T) {
	cm := NewCacheManager(5, 100)

	// Test with no operations
	hitRate := cm.calculateHitRate()
	if hitRate != 0 {
		t.Errorf("Expected hit rate 0 with no operations, got %f", hitRate)
	}

	// Test with only hits
	cm.hits = 10
	cm.misses = 0
	hitRate = cm.calculateHitRate()
	if hitRate != 100.0 {
		t.Errorf("Expected hit rate 100.0 with only hits, got %f", hitRate)
	}

	// Test with only misses
	cm.hits = 0
	cm.misses = 5
	hitRate = cm.calculateHitRate()
	if hitRate != 0.0 {
		t.Errorf("Expected hit rate 0.0 with only misses, got %f", hitRate)
	}

	// Test with mixed hits and misses
	cm.hits = 7
	cm.misses = 3
	hitRate = cm.calculateHitRate()
	expected := 70.0
	if hitRate != expected {
		t.Errorf("Expected hit rate %f, got %f", expected, hitRate)
	}
}

// TestCacheManagerEvictionOrder tests that eviction happens in correct LRU order
func TestCacheManagerEvictionOrder(t *testing.T) {
	cm := NewCacheManager(2, 100) // Very small cache

	// Add first item
	cm.Put("oldest", []float32{1.0})
	
	// Add second item
	cm.Put("middle", []float32{2.0})
	
	// Access first item to make it recently used
	cm.Get("oldest")
	
	// Add third item - should evict "middle"
	cm.Put("newest", []float32{3.0})
	
	// "middle" should be gone, "oldest" and "newest" should remain
	if _, exists := cm.Get("middle"); exists {
		t.Error("middle item should have been evicted")
	}
	
	if _, exists := cm.Get("oldest"); !exists {
		t.Error("oldest item should still exist (was accessed)")
	}
	
	if _, exists := cm.Get("newest"); !exists {
		t.Error("newest item should exist")
	}
}

// BenchmarkCacheManagerPut benchmarks put operations
func BenchmarkCacheManagerPut(b *testing.B) {
	cm := NewCacheManager(1000, 1000)
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key_%d", i%500) // Some key reuse
		cm.Put(key, data)
	}
}

// BenchmarkCacheManagerGet benchmarks get operations
func BenchmarkCacheManagerGet(b *testing.B) {
	cm := NewCacheManager(1000, 1000)
	data := make([]float32, 1000)
	
	// Pre-populate cache
	for i := 0; i < 500; i++ {
		key := fmt.Sprintf("key_%d", i)
		cm.Put(key, data)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key_%d", i%500)
		cm.Get(key)
	}
}

// BenchmarkCacheManagerMixed benchmarks mixed operations
func BenchmarkCacheManagerMixed(b *testing.B) {
	cm := NewCacheManager(1000, 1000)
	data := make([]float32, 1000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key_%d", i%1000)
		
		if i%3 == 0 {
			cm.Put(key, data)
		} else {
			cm.Get(key)
		}
	}
}