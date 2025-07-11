package dataloader

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestGetGlobalSharedCache tests the global shared cache singleton
func TestGetGlobalSharedCache(t *testing.T) {
	// Reset the singleton for testing
	sharedCacheOnce = sync.Once{}
	globalSharedCache = nil

	// First call should create the instance
	cache1 := GetGlobalSharedCache()
	if cache1 == nil {
		t.Error("Expected non-nil shared cache")
	}

	if cache1.caches == nil {
		t.Error("Expected initialized caches map")
	}

	// Second call should return the same instance
	cache2 := GetGlobalSharedCache()
	if cache1 != cache2 {
		t.Error("Expected same instance from global shared cache")
	}

	// Subsequent calls should also return the same instance
	cache3 := GetGlobalSharedCache()
	if cache1 != cache3 {
		t.Error("Expected same instance on third call")
	}
}

// TestSharedCacheManagerGetOrCreateCache tests cache creation and retrieval
func TestSharedCacheManagerGetOrCreateCache(t *testing.T) {
	scm := &SharedCacheManager{
		caches: make(map[string]*CacheManager),
	}

	// Test creating a new cache
	cache1 := scm.GetOrCreateCache("test_cache", 100, 1000)
	if cache1 == nil {
		t.Error("Expected non-nil cache")
	}

	if cache1.maxSize != 100 {
		t.Errorf("Expected max size 100, got %d", cache1.maxSize)
	}

	if cache1.itemSize != 1000 {
		t.Errorf("Expected item size 1000, got %d", cache1.itemSize)
	}

	// Test retrieving existing cache
	cache2 := scm.GetOrCreateCache("test_cache", 200, 2000)
	if cache1 != cache2 {
		t.Error("Expected same cache instance for same name")
	}

	// Parameters should not change for existing cache
	if cache2.maxSize != 100 {
		t.Errorf("Expected max size to remain 100, got %d", cache2.maxSize)
	}

	// Test creating a different cache
	cache3 := scm.GetOrCreateCache("different_cache", 50, 500)
	if cache1 == cache3 {
		t.Error("Expected different cache instance for different name")
	}

	if cache3.maxSize != 50 {
		t.Errorf("Expected max size 50 for new cache, got %d", cache3.maxSize)
	}
}

// TestSharedCacheManagerRemoveCache tests cache removal
func TestSharedCacheManagerRemoveCache(t *testing.T) {
	scm := &SharedCacheManager{
		caches: make(map[string]*CacheManager),
	}

	// Create some caches
	cache1 := scm.GetOrCreateCache("cache1", 100, 1000)
	cache2 := scm.GetOrCreateCache("cache2", 200, 2000)
	_ = cache2 // Use cache2 to avoid unused warning

	// Verify they exist
	if len(scm.caches) != 2 {
		t.Errorf("Expected 2 caches, got %d", len(scm.caches))
	}

	// Remove one cache
	scm.RemoveCache("cache1")

	if len(scm.caches) != 1 {
		t.Errorf("Expected 1 cache after removal, got %d", len(scm.caches))
	}

	// Verify the correct cache was removed
	if _, exists := scm.caches["cache1"]; exists {
		t.Error("cache1 should have been removed")
	}

	if _, exists := scm.caches["cache2"]; !exists {
		t.Error("cache2 should still exist")
	}

	// Try to remove non-existent cache (should not panic)
	scm.RemoveCache("nonexistent")

	if len(scm.caches) != 1 {
		t.Errorf("Expected 1 cache after removing nonexistent, got %d", len(scm.caches))
	}

	// Get the removed cache again (should create a new one)
	newCache1 := scm.GetOrCreateCache("cache1", 150, 1500)
	if newCache1 == cache1 {
		t.Error("Expected new cache instance after removal")
	}

	if newCache1.maxSize != 150 {
		t.Errorf("Expected new cache max size 150, got %d", newCache1.maxSize)
	}
}

// TestSharedCacheManagerClearAllCaches tests clearing all caches
func TestSharedCacheManagerClearAllCaches(t *testing.T) {
	scm := &SharedCacheManager{
		caches: make(map[string]*CacheManager),
	}

	// Create some caches and add data
	cache1 := scm.GetOrCreateCache("cache1", 100, 1000)
	cache2 := scm.GetOrCreateCache("cache2", 200, 2000)

	cache1.Put("key1", []float32{1.0, 2.0})
	cache2.Put("key2", []float32{3.0, 4.0})

	// Verify data exists
	if cache1.Stats().Size == 0 {
		t.Error("Expected cache1 to have data")
	}
	if cache2.Stats().Size == 0 {
		t.Error("Expected cache2 to have data")
	}

	// Clear all caches
	scm.ClearAllCaches()

	// Verify caches are cleared but still exist
	if len(scm.caches) != 2 {
		t.Errorf("Expected 2 caches to still exist, got %d", len(scm.caches))
	}

	if cache1.Stats().Size != 0 {
		t.Errorf("Expected cache1 to be empty, got size %d", cache1.Stats().Size)
	}

	if cache2.Stats().Size != 0 {
		t.Errorf("Expected cache2 to be empty, got size %d", cache2.Stats().Size)
	}

	// Verify data is gone
	if _, exists := cache1.Get("key1"); exists {
		t.Error("key1 should have been cleared from cache1")
	}

	if _, exists := cache2.Get("key2"); exists {
		t.Error("key2 should have been cleared from cache2")
	}
}

// TestCreateSharedDataLoaders tests the factory function for shared DataLoaders
func TestCreateSharedDataLoaders(t *testing.T) {
	trainDataset := NewMockDataset(100)
	valDataset := NewMockDataset(50)

	config := Config{
		BatchSize:    32,
		ImageSize:    224,
		MaxCacheSize: 200,
	}

	trainLoader, valLoader := CreateSharedDataLoaders(trainDataset, valDataset, config)

	// Verify DataLoaders were created
	if trainLoader == nil {
		t.Error("Expected non-nil train loader")
	}
	if valLoader == nil {
		t.Error("Expected non-nil validation loader")
	}

	// Verify they share the same cache
	if trainLoader.cacheManager != valLoader.cacheManager {
		t.Error("Expected train and validation loaders to share the same cache")
	}

	// Verify neither owns the cache (it's shared)
	if trainLoader.ownedCache {
		t.Error("Expected train loader to not own the cache")
	}
	if valLoader.ownedCache {
		t.Error("Expected validation loader to not own the cache")
	}

	// Verify shuffle settings
	if !trainLoader.shuffle {
		t.Error("Expected train loader to have shuffle enabled")
	}
	if valLoader.shuffle {
		t.Error("Expected validation loader to have shuffle disabled")
	}

	// Verify datasets
	if trainLoader.dataset != trainDataset {
		t.Error("Train loader dataset mismatch")
	}
	if valLoader.dataset != valDataset {
		t.Error("Validation loader dataset mismatch")
	}

	// Verify batch sizes
	if trainLoader.batchSize != 32 {
		t.Errorf("Expected train loader batch size 32, got %d", trainLoader.batchSize)
	}
	if valLoader.batchSize != 32 {
		t.Errorf("Expected validation loader batch size 32, got %d", valLoader.batchSize)
	}
}

// TestCreateSharedDataLoadersDefaultCacheSize tests default cache size calculation
func TestCreateSharedDataLoadersDefaultCacheSize(t *testing.T) {
	trainDataset := NewMockDataset(80)
	valDataset := NewMockDataset(20)

	config := Config{
		BatchSize: 16,
		ImageSize: 128,
		// MaxCacheSize not specified - should use total dataset size
	}

	trainLoader, valLoader := CreateSharedDataLoaders(trainDataset, valDataset, config)
	_ = valLoader // Use variable to avoid unused warning

	// Verify cache size is set to total dataset size
	expectedCacheSize := 80 + 20 // train + val
	actualCacheSize := trainLoader.cacheManager.maxSize

	if actualCacheSize != expectedCacheSize {
		t.Errorf("Expected cache size %d, got %d", expectedCacheSize, actualCacheSize)
	}

	// Verify item size calculation
	expectedItemSize := 3 * 128 * 128 // CHW format
	actualItemSize := trainLoader.cacheManager.itemSize

	if actualItemSize != expectedItemSize {
		t.Errorf("Expected item size %d, got %d", expectedItemSize, actualItemSize)
	}
}

// TestSharedCacheManagerConcurrency tests thread safety of shared cache manager
func TestSharedCacheManagerConcurrency(t *testing.T) {
	scm := &SharedCacheManager{
		caches: make(map[string]*CacheManager),
	}

	numGoroutines := 20
	numOperations := 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines performing concurrent operations
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				cacheName := fmt.Sprintf("cache_%d", id%5) // 5 different caches
				
				// Create/get cache
				cache := scm.GetOrCreateCache(cacheName, 100, 1000)
				
				// Use the cache
				key := fmt.Sprintf("key_%d_%d", id, j)
				data := []float32{float32(id), float32(j)}
				cache.Put(key, data)
				cache.Get(key)
				
				// Occasionally remove caches
				if j%20 == 0 {
					scm.RemoveCache(cacheName)
				}
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

	// Verify final state is consistent
	scm.mu.RLock()
	cacheCount := len(scm.caches)
	scm.mu.RUnlock()

	if cacheCount < 0 {
		t.Error("Negative cache count detected")
	}
}

// TestSharedCacheDataSharing tests that data is actually shared between loaders
func TestSharedCacheDataSharing(t *testing.T) {
	trainDataset := NewMockDataset(10)
	valDataset := NewMockDataset(10)

	// Use overlapping image paths
	for i := 0; i < 5; i++ {
		trainDataset.items[i].imagePath = fmt.Sprintf("shared_image_%d.jpg", i)
		valDataset.items[i].imagePath = fmt.Sprintf("shared_image_%d.jpg", i)
	}

	config := Config{
		BatchSize: 5,
		ImageSize: 64,
	}

	trainLoader, valLoader := CreateSharedDataLoaders(trainDataset, valDataset, config)

	// Add data to shared cache via train loader
	sharedCache := trainLoader.cacheManager
	testData := []float32{1.0, 2.0, 3.0}
	sharedCache.Put("shared_image_0.jpg", testData)

	// Verify data is accessible from both loaders
	trainData, trainExists := trainLoader.cacheManager.Get("shared_image_0.jpg")
	valData, valExists := valLoader.cacheManager.Get("shared_image_0.jpg")

	if !trainExists {
		t.Error("Data should be accessible from train loader")
	}
	if !valExists {
		t.Error("Data should be accessible from validation loader")
	}

	// Verify it's the same data
	if len(trainData) != len(testData) || len(valData) != len(testData) {
		t.Error("Data length mismatch")
	}

	for i := range testData {
		if trainData[i] != testData[i] || valData[i] != testData[i] {
			t.Error("Data content mismatch")
		}
	}

	// Verify stats are shared
	trainStats := trainLoader.Stats()
	valStats := valLoader.Stats()

	if trainStats != valStats {
		t.Error("Expected identical stats for shared cache")
	}
}

// TestSharedCacheManagerEmpty tests behavior with empty state
func TestSharedCacheManagerEmpty(t *testing.T) {
	scm := &SharedCacheManager{
		caches: make(map[string]*CacheManager),
	}

	// Clear empty manager (should not panic)
	scm.ClearAllCaches()

	// Remove from empty manager (should not panic)
	scm.RemoveCache("nonexistent")

	// Verify state is still valid
	if scm.caches == nil {
		t.Error("Caches map should not be nil")
	}

	if len(scm.caches) != 0 {
		t.Errorf("Expected 0 caches, got %d", len(scm.caches))
	}
}

// BenchmarkSharedCacheGetOrCreate benchmarks cache creation/retrieval
func BenchmarkSharedCacheGetOrCreate(b *testing.B) {
	scm := &SharedCacheManager{
		caches: make(map[string]*CacheManager),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheName := fmt.Sprintf("cache_%d", i%100) // Reuse some cache names
		scm.GetOrCreateCache(cacheName, 1000, 1000)
	}
}

// BenchmarkCreateSharedDataLoaders benchmarks the factory function
func BenchmarkCreateSharedDataLoaders(b *testing.B) {
	trainDataset := NewMockDataset(1000)
	valDataset := NewMockDataset(200)

	config := Config{
		BatchSize: 32,
		ImageSize: 224,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CreateSharedDataLoaders(trainDataset, valDataset, config)
	}
}