package metal_bridge

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation
#include "metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// BufferPool represents a pool of buffers for a specific size category
type BufferPool struct {
	buffers []*Buffer
	mutex   sync.Mutex
}

// BufferAllocator manages a pool of MTLBuffers for efficient memory management
// It categorizes buffers by size to minimize fragmentation and allocation overhead
type BufferAllocator struct {
	device *Device
	
	// Size-based pools using power-of-2 bins
	pools map[int]*BufferPool // key is the power of 2 (e.g., 10 for 1KB, 20 for 1MB)
	
	// Global mutex for pool management
	mutex sync.RWMutex
	
	// Statistics for diagnostics
	stats struct {
		totalAllocated   uint64 // Total bytes allocated from Metal
		totalFree        uint64 // Total bytes available in pools
		numAllocations   uint64 // Number of allocation requests
		numDeallocations uint64 // Number of deallocation requests
		numPoolHits      uint64 // Number of successful pool retrievals
		numPoolMisses    uint64 // Number of new allocations needed
		mutex            sync.RWMutex
	}
	
	// Configuration
	maxPoolSize     int     // Maximum number of buffers per pool
	maxTotalMemory  uint64  // Maximum total memory to cache (bytes)
	minBufferSize   uint64  // Minimum buffer size to pool (smaller buffers are not pooled)
}

// AllocatorConfig holds configuration for the BufferAllocator
type AllocatorConfig struct {
	MaxPoolSize    int    // Maximum buffers per size pool (default: 32)
	MaxTotalMemory uint64 // Maximum total cached memory in bytes (default: 1GB)
	MinBufferSize  uint64 // Minimum buffer size to pool in bytes (default: 1KB)
}

// DefaultAllocatorConfig returns sensible defaults for the allocator
func DefaultAllocatorConfig() AllocatorConfig {
	return AllocatorConfig{
		MaxPoolSize:    32,
		MaxTotalMemory: 1024 * 1024 * 1024, // 1GB
		MinBufferSize:  1024,                // 1KB
	}
}

// Global allocator instance
var globalAllocator *BufferAllocator
var allocatorOnce sync.Once

// GetGlobalAllocator returns the singleton BufferAllocator instance
func GetGlobalAllocator() *BufferAllocator {
	allocatorOnce.Do(func() {
		device := CreateSystemDefaultDevice()
		config := DefaultAllocatorConfig()
		globalAllocator = NewBufferAllocator(device, config)
	})
	return globalAllocator
}

// NewBufferAllocator creates a new BufferAllocator with the given device and configuration
func NewBufferAllocator(device *Device, config AllocatorConfig) *BufferAllocator {
	allocator := &BufferAllocator{
		device:         device,
		pools:          make(map[int]*BufferPool),
		maxPoolSize:    config.MaxPoolSize,
		maxTotalMemory: config.MaxTotalMemory,
		minBufferSize:  config.MinBufferSize,
	}
	
	// Set finalizer to clean up when allocator is garbage collected
	runtime.SetFinalizer(allocator, (*BufferAllocator).cleanup)
	
	return allocator
}

// roundUpToPowerOf2 rounds a size up to the next power of 2
func roundUpToPowerOf2(size uint64) uint64 {
	if size == 0 {
		return 1
	}
	
	// If already power of 2, return as is
	if size&(size-1) == 0 {
		return size
	}
	
	// Find the next power of 2
	size--
	size |= size >> 1
	size |= size >> 2
	size |= size >> 4
	size |= size >> 8
	size |= size >> 16
	size |= size >> 32
	size++
	
	return size
}

// sizeToPoolKey converts a buffer size to a pool key (power of 2 exponent)
func sizeToPoolKey(size uint64) int {
	if size == 0 {
		return 0
	}
	
	roundedSize := roundUpToPowerOf2(size)
	
	// Find the exponent (log2)
	key := 0
	temp := roundedSize
	for temp > 1 {
		temp >>= 1
		key++
	}
	
	return key
}

// poolKeyToSize converts a pool key back to buffer size
func poolKeyToSize(key int) uint64 {
	if key < 0 {
		return 0
	}
	return uint64(1) << uint64(key)
}

// getOrCreatePool gets an existing pool or creates a new one for the given key
func (a *BufferAllocator) getOrCreatePool(key int) *BufferPool {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	
	pool, exists := a.pools[key]
	if !exists {
		pool = &BufferPool{
			buffers: make([]*Buffer, 0, a.maxPoolSize),
		}
		a.pools[key] = pool
	}
	
	return pool
}

// Allocate allocates a buffer of the specified size from the pool or creates a new one
func (a *BufferAllocator) Allocate(sizeInBytes uint64, options C.NSUInteger) (*Buffer, error) {
	if sizeInBytes == 0 {
		return nil, fmt.Errorf("cannot allocate buffer with zero size")
	}
	
	// Update statistics
	a.stats.mutex.Lock()
	a.stats.numAllocations++
	a.stats.mutex.Unlock()
	
	// For very small buffers, don't use pooling - allocate directly
	if sizeInBytes < a.minBufferSize {
		return a.allocateNew(sizeInBytes, options)
	}
	
	// Determine pool key based on rounded size
	poolKey := sizeToPoolKey(sizeInBytes)
	actualSize := poolKeyToSize(poolKey)
	
	// Try to get buffer from pool first
	pool := a.getOrCreatePool(poolKey)
	
	pool.mutex.Lock()
	if len(pool.buffers) > 0 {
		// Reuse buffer from pool
		buffer := pool.buffers[len(pool.buffers)-1]
		pool.buffers = pool.buffers[:len(pool.buffers)-1]
		pool.mutex.Unlock()
		
		// Update statistics
		a.stats.mutex.Lock()
		a.stats.numPoolHits++
		a.stats.totalFree -= actualSize
		a.stats.mutex.Unlock()
		
		// Mark buffer as in use and reset any reference counting
		buffer.inUse = true
		buffer.refCount = 1
		
		return buffer, nil
	}
	pool.mutex.Unlock()
	
	// No buffer available in pool, allocate new one
	a.stats.mutex.Lock()
	a.stats.numPoolMisses++
	a.stats.mutex.Unlock()
	
	return a.allocateNew(actualSize, options)
}

// allocateNew creates a new MTLBuffer via Metal bridge
func (a *BufferAllocator) allocateNew(sizeInBytes uint64, options C.NSUInteger) (*Buffer, error) {
	// Create new MTLBuffer using Metal bridge
	c_buf := C.CreateBufferWithLength(a.device.c_device, C.size_t(sizeInBytes), options)
	if c_buf == nil {
		return nil, fmt.Errorf("failed to create MTLBuffer of size %d", sizeInBytes)
	}
	
	// Retain the buffer for Go ownership
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_buf)))
	
	// Create Go wrapper
	buffer := &Buffer{
		c_buffer:  c_buf,
		length:    uintptr(sizeInBytes),
		inUse:     true,
		refCount:  1,
		allocator: a,
	}
	
	// Set finalizer for the buffer
	runtime.SetFinalizer(buffer, (*Buffer).finalize)
	
	// Update statistics
	a.stats.mutex.Lock()
	a.stats.totalAllocated += sizeInBytes
	a.stats.mutex.Unlock()
	
	return buffer, nil
}

// Release returns a buffer to the pool for reuse
func (a *BufferAllocator) Release(buffer *Buffer) {
	if buffer == nil || !buffer.inUse {
		return
	}
	
	// Update statistics
	a.stats.mutex.Lock()
	a.stats.numDeallocations++
	a.stats.mutex.Unlock()
	
	// Check if buffer is too small for pooling
	if uint64(buffer.length) < a.minBufferSize {
		// Release directly without pooling
		buffer.releaseNow()
		return
	}
	
	// Determine which pool this buffer belongs to
	poolKey := sizeToPoolKey(uint64(buffer.length))
	pool := a.getOrCreatePool(poolKey)
	
	pool.mutex.Lock()
	
	// Check if pool is full
	if len(pool.buffers) >= a.maxPoolSize {
		pool.mutex.Unlock()
		// Pool is full, release buffer directly
		buffer.releaseNow()
		return
	}
	
	// Check total memory limit
	a.stats.mutex.RLock()
	totalFree := a.stats.totalFree
	a.stats.mutex.RUnlock()
	
	if totalFree+uint64(buffer.length) > a.maxTotalMemory {
		pool.mutex.Unlock()
		// Would exceed memory limit, release buffer directly
		buffer.releaseNow()
		return
	}
	
	// Add buffer to pool
	buffer.inUse = false
	buffer.refCount = 0
	pool.buffers = append(pool.buffers, buffer)
	pool.mutex.Unlock()
	
	// Update statistics
	a.stats.mutex.Lock()
	a.stats.totalFree += uint64(buffer.length)
	a.stats.mutex.Unlock()
}

// MemoryStats holds memory usage statistics
type MemoryStats struct {
	TotalAllocated   uint64 // Total bytes allocated from Metal
	TotalFree        uint64 // Total bytes available in pools
	NumAllocations   uint64 // Number of allocation requests
	NumDeallocations uint64 // Number of deallocation requests
	NumPoolHits      uint64 // Number of successful pool retrievals
	NumPoolMisses    uint64 // Number of new allocations needed
	FragmentedMemory uint64 // Estimated fragmented memory
	NumPools         int    // Number of active pools
}

// GetMemoryStats returns current memory usage statistics
func (a *BufferAllocator) GetMemoryStats() MemoryStats {
	a.stats.mutex.RLock()
	stats := MemoryStats{
		TotalAllocated:   a.stats.totalAllocated,
		TotalFree:        a.stats.totalFree,
		NumAllocations:   a.stats.numAllocations,
		NumDeallocations: a.stats.numDeallocations,
		NumPoolHits:      a.stats.numPoolHits,
		NumPoolMisses:    a.stats.numPoolMisses,
	}
	a.stats.mutex.RUnlock()
	
	// Calculate fragmentation and pool count
	a.mutex.RLock()
	stats.NumPools = len(a.pools)
	
	// Estimate fragmentation based on pool memory
	for _, pool := range a.pools {
		pool.mutex.Lock()
		if len(pool.buffers) > 0 {
			poolMemory := uint64(len(pool.buffers)) * poolKeyToSize(sizeToPoolKey(uint64(pool.buffers[0].length)))
			stats.FragmentedMemory += poolMemory
		}
		pool.mutex.Unlock()
	}
	a.mutex.RUnlock()
	
	return stats
}

// cleanup releases all pooled buffers and cleans up resources
func (a *BufferAllocator) cleanup() {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	
	// Release all buffers in all pools
	for _, pool := range a.pools {
		pool.mutex.Lock()
		for _, buffer := range pool.buffers {
			buffer.releaseNow()
		}
		pool.buffers = nil
		pool.mutex.Unlock()
	}
	
	// Clear pools
	a.pools = make(map[int]*BufferPool)
	
	// Reset statistics
	a.stats.mutex.Lock()
	a.stats.totalAllocated = 0
	a.stats.totalFree = 0
	a.stats.mutex.Unlock()
}

// Shutdown performs cleanup and releases all resources
func (a *BufferAllocator) Shutdown() {
	a.cleanup()
}