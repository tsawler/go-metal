package dataloader

import (
	"sync"
)

// SharedCacheManager manages a global cache that can be shared across multiple DataLoaders
type SharedCacheManager struct {
	mu     sync.RWMutex
	caches map[string]*CacheManager
}

var (
	globalSharedCache *SharedCacheManager
	sharedCacheOnce   sync.Once
)

// GetGlobalSharedCache returns the global shared cache manager
func GetGlobalSharedCache() *SharedCacheManager {
	sharedCacheOnce.Do(func() {
		globalSharedCache = &SharedCacheManager{
			caches: make(map[string]*CacheManager),
		}
	})
	return globalSharedCache
}

// GetOrCreateCache gets or creates a cache with the given name and parameters
func (scm *SharedCacheManager) GetOrCreateCache(name string, maxSize int, itemSize int) *CacheManager {
	scm.mu.Lock()
	defer scm.mu.Unlock()

	if cache, exists := scm.caches[name]; exists {
		return cache
	}

	cache := NewCacheManager(maxSize, itemSize)
	scm.caches[name] = cache
	return cache
}

// RemoveCache removes a cache by name
func (scm *SharedCacheManager) RemoveCache(name string) {
	scm.mu.Lock()
	defer scm.mu.Unlock()
	delete(scm.caches, name)
}

// ClearAllCaches clears all managed caches
func (scm *SharedCacheManager) ClearAllCaches() {
	scm.mu.Lock()
	defer scm.mu.Unlock()
	
	for _, cache := range scm.caches {
		cache.Clear()
	}
}

// CreateSharedDataLoaders creates train and validation DataLoaders with a shared cache
func CreateSharedDataLoaders(trainDataset, valDataset Dataset, config Config) (*DataLoader, *DataLoader) {
	// Calculate cache parameters
	itemSize := 3 * config.ImageSize * config.ImageSize // CHW format
	totalImages := trainDataset.Len() + valDataset.Len()
	
	// Use the total dataset size for cache size if not specified
	cacheSize := config.MaxCacheSize
	if cacheSize == 0 {
		cacheSize = totalImages
	}
	
	// Create or get shared cache
	sharedCache := NewCacheManager(cacheSize, itemSize)
	
	// Create train loader
	trainConfig := config
	trainConfig.CacheManager = sharedCache
	trainConfig.Shuffle = true
	trainLoader := NewDataLoader(trainDataset, trainConfig)
	
	// Create validation loader
	valConfig := config
	valConfig.CacheManager = sharedCache
	valConfig.Shuffle = false
	valLoader := NewDataLoader(valDataset, valConfig)
	
	return trainLoader, valLoader
}