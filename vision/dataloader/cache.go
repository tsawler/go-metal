package dataloader

import (
	"container/list"
	"fmt"
	"sync"
)

// CacheManager manages a shared cache for preprocessed data
type CacheManager struct {
	mu           sync.RWMutex
	cache        map[string][]float32
	lru          *list.List
	lruMap       map[string]*list.Element
	maxSize      int
	currentSize  int
	itemSize     int // Size of each item in float32 elements
	
	// Statistics
	hits   int64
	misses int64
}

// NewCacheManager creates a new cache manager
func NewCacheManager(maxSize int, itemSize int) *CacheManager {
	return &CacheManager{
		cache:    make(map[string][]float32),
		lru:      list.New(),
		lruMap:   make(map[string]*list.Element),
		maxSize:  maxSize,
		itemSize: itemSize,
	}
}

// Get retrieves an item from the cache
func (cm *CacheManager) Get(key string) ([]float32, bool) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if data, exists := cm.cache[key]; exists {
		// Move to front (most recently used)
		if elem, ok := cm.lruMap[key]; ok {
			cm.lru.MoveToFront(elem)
		}
		cm.hits++
		return data, true
	}

	cm.misses++
	return nil, false
}

// Put adds an item to the cache
func (cm *CacheManager) Put(key string, data []float32) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Check if already exists
	if _, exists := cm.cache[key]; exists {
		// Move to front
		if elem, ok := cm.lruMap[key]; ok {
			cm.lru.MoveToFront(elem)
		}
		return
	}

	// Add new item
	elem := cm.lru.PushFront(key)
	cm.lruMap[key] = elem
	cm.cache[key] = data
	cm.currentSize++

	// Evict if necessary
	for cm.currentSize > cm.maxSize && cm.lru.Len() > 0 {
		oldest := cm.lru.Back()
		if oldest != nil {
			cm.removeElement(oldest)
		}
	}
}

// removeElement removes an element from the cache
func (cm *CacheManager) removeElement(elem *list.Element) {
	key := elem.Value.(string)
	cm.lru.Remove(elem)
	delete(cm.lruMap, key)
	delete(cm.cache, key)
	cm.currentSize--
}

// Stats returns cache statistics
func (cm *CacheManager) Stats() CacheStats {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return CacheStats{
		Size:     cm.currentSize,
		MaxSize:  cm.maxSize,
		Hits:     cm.hits,
		Misses:   cm.misses,
		HitRate:  cm.calculateHitRate(),
	}
}

// calculateHitRate calculates the hit rate percentage
func (cm *CacheManager) calculateHitRate() float64 {
	total := cm.hits + cm.misses
	if total == 0 {
		return 0
	}
	return float64(cm.hits) / float64(total) * 100
}

// Clear clears the cache
func (cm *CacheManager) Clear() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.cache = make(map[string][]float32)
	cm.lru = list.New()
	cm.lruMap = make(map[string]*list.Element)
	cm.currentSize = 0
	// Don't reset statistics - keep them cumulative
}

// ResetStats resets the statistics
func (cm *CacheManager) ResetStats() {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.hits = 0
	cm.misses = 0
}

// CacheStats holds cache statistics
type CacheStats struct {
	Size    int
	MaxSize int
	Hits    int64
	Misses  int64
	HitRate float64
}

// String returns a string representation of cache stats
func (cs CacheStats) String() string {
	return fmt.Sprintf("Cache: %d/%d items, Hits: %d, Misses: %d, Hit Rate: %.1f%%",
		cs.Size, cs.MaxSize, cs.Hits, cs.Misses, cs.HitRate)
}