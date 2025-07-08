package optimization

import (
	"fmt"
	"sync"
)

// BufferPool provides GPU-optimized buffer pooling for various data types
type BufferPool struct {
	mu    sync.Mutex
	pools map[int]*sync.Pool // Pools indexed by buffer size
	stats map[int]*PoolStats
}

// PoolStats tracks statistics for a buffer pool
type PoolStats struct {
	Gets     int64
	Puts     int64
	Misses   int64
	InUse    int64
	MaxInUse int64
}

// NewBufferPool creates a new buffer pool
func NewBufferPool() *BufferPool {
	return &BufferPool{
		pools: make(map[int]*sync.Pool),
		stats: make(map[int]*PoolStats),
	}
}

// GetFloat32Buffer gets a float32 buffer of at least the requested size
func (bp *BufferPool) GetFloat32Buffer(size int) []float32 {
	bp.mu.Lock()
	
	// Find the appropriate pool size (round up to power of 2)
	poolSize := roundUpToPowerOf2(size)
	
	pool, exists := bp.pools[poolSize]
	if !exists {
		pool = &sync.Pool{
			New: func() interface{} {
				return make([]float32, poolSize)
			},
		}
		bp.pools[poolSize] = pool
		bp.stats[poolSize] = &PoolStats{}
	}
	
	stats := bp.stats[poolSize]
	stats.Gets++
	stats.InUse++
	if stats.InUse > stats.MaxInUse {
		stats.MaxInUse = stats.InUse
	}
	
	bp.mu.Unlock()
	
	// Get buffer from pool
	buf := pool.Get().([]float32)
	
	// Ensure it's the right size
	if len(buf) < size {
		stats.Misses++
		buf = make([]float32, size)
	}
	
	return buf[:size]
}

// PutFloat32Buffer returns a float32 buffer to the pool
func (bp *BufferPool) PutFloat32Buffer(buf []float32) {
	if len(buf) == 0 {
		return
	}
	
	bp.mu.Lock()
	poolSize := roundUpToPowerOf2(cap(buf))
	
	pool, exists := bp.pools[poolSize]
	if exists {
		stats := bp.stats[poolSize]
		stats.Puts++
		stats.InUse--
		bp.mu.Unlock()
		
		// Zero out the buffer before returning to pool (optional, for security)
		for i := range buf {
			buf[i] = 0
		}
		
		pool.Put(buf[:cap(buf)])
	} else {
		bp.mu.Unlock()
	}
}

// GetInt32Buffer gets an int32 buffer of at least the requested size
func (bp *BufferPool) GetInt32Buffer(size int) []int32 {
	// For simplicity, we'll allocate int32 buffers directly
	// In a real implementation, you'd want separate pools for different types
	return make([]int32, size)
}

// Stats returns statistics for all buffer pools
func (bp *BufferPool) Stats() map[int]*PoolStats {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	
	// Return a copy to avoid race conditions
	statsCopy := make(map[int]*PoolStats)
	for size, stats := range bp.stats {
		statsCopy[size] = &PoolStats{
			Gets:     stats.Gets,
			Puts:     stats.Puts,
			Misses:   stats.Misses,
			InUse:    stats.InUse,
			MaxInUse: stats.MaxInUse,
		}
	}
	
	return statsCopy
}

// String returns a string representation of pool statistics
func (bp *BufferPool) String() string {
	stats := bp.Stats()
	result := "BufferPool Statistics:\n"
	
	for size, stat := range stats {
		hitRate := float64(0)
		if stat.Gets > 0 {
			hitRate = float64(stat.Gets-stat.Misses) / float64(stat.Gets) * 100
		}
		
		result += fmt.Sprintf("  Size %d: Gets=%d, Puts=%d, InUse=%d, MaxInUse=%d, HitRate=%.1f%%\n",
			size, stat.Gets, stat.Puts, stat.InUse, stat.MaxInUse, hitRate)
	}
	
	return result
}

// roundUpToPowerOf2 rounds a number up to the nearest power of 2
func roundUpToPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}
	
	// If n is already a power of 2, return it
	if n&(n-1) == 0 {
		return n
	}
	
	// Find the next power of 2
	power := 1
	for power < n {
		power <<= 1
	}
	
	return power
}

// GlobalBufferPool is a singleton buffer pool for the entire application
var (
	globalPool     *BufferPool
	globalPoolOnce sync.Once
)

// GetGlobalBufferPool returns the global buffer pool instance
func GetGlobalBufferPool() *BufferPool {
	globalPoolOnce.Do(func() {
		globalPool = NewBufferPool()
	})
	return globalPool
}