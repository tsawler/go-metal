package memory

import (
	"fmt"
	"sync"
	"unsafe"
	
	"github.com/tsawler/go-metal/cgo_bridge"
)

// BufferPool manages a pool of Metal buffers of a specific size
type BufferPool struct {
	buffers    chan unsafe.Pointer // Available MTLBuffers
	maxSize    int                 // Pool size limit
	bufferSize int                 // Fixed buffer size for this pool
	device     DeviceType          // Device type for this pool
	allocated  int                 // Current number of allocated buffers
	mutex      sync.RWMutex        // Protects allocated counter
}

// NewBufferPool creates a new buffer pool
func NewBufferPool(bufferSize int, maxSize int, device DeviceType) *BufferPool {
	return &BufferPool{
		buffers:    make(chan unsafe.Pointer, maxSize),
		maxSize:    maxSize,
		bufferSize: bufferSize,
		device:     device,
		allocated:  0,
	}
}

// Get retrieves a buffer from the pool or allocates a new one
func (bp *BufferPool) Get() (unsafe.Pointer, error) {
	select {
	case buffer := <-bp.buffers:
		return buffer, nil
	default:
		// Pool is empty, allocate new buffer
		bp.mutex.Lock()
		canAllocate := bp.allocated < bp.maxSize
		if canAllocate {
			bp.allocated++
		}
		bp.mutex.Unlock()
		
		if !canAllocate {
			return nil, fmt.Errorf("buffer pool at capacity (%d)", bp.maxSize)
		}
		
		// Allocate new Metal buffer via CGO
		buffer, err := allocateMetalBuffer(bp.bufferSize, bp.device)
		if err != nil {
			bp.mutex.Lock()
			bp.allocated--
			bp.mutex.Unlock()
			return nil, fmt.Errorf("failed to allocate Metal buffer: %v", err)
		}
		
		return buffer, nil
	}
}

// Return puts a buffer back into the pool
func (bp *BufferPool) Return(buffer unsafe.Pointer) {
	if buffer == nil {
		return
	}
	
	select {
	case bp.buffers <- buffer:
		// Successfully returned to pool
	default:
		// Pool is full, deallocate the buffer
		deallocateMetalBuffer(buffer)
		bp.mutex.Lock()
		bp.allocated--
		bp.mutex.Unlock()
	}
}

// Stats returns pool statistics
func (bp *BufferPool) Stats() (available int, allocated int, maxSize int) {
	bp.mutex.RLock()
	defer bp.mutex.RUnlock()
	return len(bp.buffers), bp.allocated, bp.maxSize
}

// MemoryManager manages GPU buffer lifecycle and pooling
type MemoryManager struct {
	pools       map[PoolKey]*BufferPool // Pools by size and device
	poolsMutex  sync.RWMutex            // Protects pools map
	device      unsafe.Pointer          // MTLDevice
	
	// Pool size tiers (in bytes)
	poolSizes   []int
	
	// Buffer size tracking
	bufferSizes     map[unsafe.Pointer]int // Maps buffer pointer to its allocated size
	bufferSizesMutex sync.RWMutex         // Protects bufferSizes map
}

// PoolKey represents a key for the buffer pool map
type PoolKey struct {
	Size   int
	Device DeviceType
}

// Default pool sizes: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB
var defaultPoolSizes = []int{
	1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
}

// NewMemoryManager creates a new memory manager
func NewMemoryManager(device unsafe.Pointer) *MemoryManager {
	return &MemoryManager{
		pools:       make(map[PoolKey]*BufferPool),
		device:      device,
		poolSizes:   defaultPoolSizes,
		bufferSizes: make(map[unsafe.Pointer]int),
	}
}

// GetBuffer gets a buffer of at least the specified size
func (mm *MemoryManager) GetBuffer(size int, device DeviceType) (unsafe.Pointer, error) {
	// Find the smallest pool that can accommodate this size
	poolSize := mm.findPoolSize(size)
	key := PoolKey{Size: poolSize, Device: device}
	
	// Get or create the pool
	pool := mm.getOrCreatePool(key)
	
	buffer, err := pool.Get()
	if err != nil {
		return nil, err
	}
	
	// Track the buffer size
	mm.bufferSizesMutex.Lock()
	mm.bufferSizes[buffer] = poolSize
	mm.bufferSizesMutex.Unlock()
	
	return buffer, nil
}

// ReturnBuffer returns a buffer to the appropriate pool
func (mm *MemoryManager) ReturnBuffer(buffer unsafe.Pointer, size int, device DeviceType) {
	if buffer == nil {
		return
	}
	
	poolSize := mm.findPoolSize(size)
	key := PoolKey{Size: poolSize, Device: device}
	
	mm.poolsMutex.RLock()
	pool, exists := mm.pools[key]
	mm.poolsMutex.RUnlock()
	
	if exists {
		pool.Return(buffer)
	} else {
		// No pool exists, just deallocate
		deallocateMetalBuffer(buffer)
	}
	
	// Clean up buffer size tracking
	mm.bufferSizesMutex.Lock()
	delete(mm.bufferSizes, buffer)
	mm.bufferSizesMutex.Unlock()
}

// findPoolSize finds the smallest pool size that can accommodate the request
func (mm *MemoryManager) findPoolSize(size int) int {
	for _, poolSize := range mm.poolSizes {
		if poolSize >= size {
			return poolSize
		}
	}
	// If size is larger than largest pool, use the requested size
	return size
}

// getOrCreatePool gets an existing pool or creates a new one
func (mm *MemoryManager) getOrCreatePool(key PoolKey) *BufferPool {
	mm.poolsMutex.RLock()
	pool, exists := mm.pools[key]
	mm.poolsMutex.RUnlock()
	
	if exists {
		return pool
	}
	
	// Create new pool
	mm.poolsMutex.Lock()
	defer mm.poolsMutex.Unlock()
	
	// Double-check after acquiring write lock
	if pool, exists := mm.pools[key]; exists {
		return pool
	}
	
	// Determine max pool size based on buffer size
	maxPoolSize := calculateMaxPoolSize(key.Size)
	pool = NewBufferPool(key.Size, maxPoolSize, key.Device)
	mm.pools[key] = pool
	
	return pool
}

// calculateMaxPoolSize determines the maximum number of buffers for a pool
func calculateMaxPoolSize(bufferSize int) int {
	// Smaller buffers get larger pools
	switch {
	case bufferSize <= 4096:     // <= 4KB
		return 100
	case bufferSize <= 65536:    // <= 64KB  
		return 50
	case bufferSize <= 1048576:  // <= 1MB
		return 20
	case bufferSize <= 16777216: // <= 16MB
		return 10
	default:                     // > 16MB
		return 5
	}
}

// Stats returns memory manager statistics
func (mm *MemoryManager) Stats() map[PoolKey]string {
	mm.poolsMutex.RLock()
	defer mm.poolsMutex.RUnlock()
	
	stats := make(map[PoolKey]string)
	for key, pool := range mm.pools {
		available, allocated, maxSize := pool.Stats()
		stats[key] = fmt.Sprintf("available=%d, allocated=%d, max=%d", 
			available, allocated, maxSize)
	}
	
	return stats
}

// Global memory manager instance
var globalMemoryManager *MemoryManager
var globalMemoryManagerOnce sync.Once

// InitializeGlobalMemoryManager initializes the global memory manager
func InitializeGlobalMemoryManager(device unsafe.Pointer) {
	globalMemoryManagerOnce.Do(func() {
		globalMemoryManager = NewMemoryManager(device)
	})
}

// GetGlobalMemoryManager returns the global memory manager instance
func GetGlobalMemoryManager() *MemoryManager {
	if globalMemoryManager == nil {
		panic("global memory manager not initialized - call InitializeGlobalMemoryManager first")
	}
	return globalMemoryManager
}

// CGO bridge functions

// allocateMetalBuffer allocates a Metal buffer via CGO
func allocateMetalBuffer(size int, device DeviceType) (unsafe.Pointer, error) {
	// Get the device from global memory manager
	metalDevice := GetGlobalMemoryManager().device
	
	// Convert device type
	var cgoDev cgo_bridge.DeviceType
	switch device {
	case CPU:
		cgoDev = cgo_bridge.CPU
	case GPU:
		cgoDev = cgo_bridge.GPU
	case PersistentGPU:
		cgoDev = cgo_bridge.PersistentGPU
	default:
		cgoDev = cgo_bridge.GPU
	}
	
	return cgo_bridge.AllocateMetalBuffer(metalDevice, size, cgoDev)
}

// deallocateMetalBuffer deallocates a Metal buffer via CGO  
func deallocateMetalBuffer(buffer unsafe.Pointer) {
	cgo_bridge.DeallocateMetalBuffer(buffer)
}

// AllocateBuffer is a simple interface for external packages
func (mm *MemoryManager) AllocateBuffer(size int) unsafe.Pointer {
	buffer, err := mm.GetBuffer(size, GPU)
	if err != nil {
		return nil
	}
	// Size tracking is already done in GetBuffer
	return buffer
}

// ReleaseBuffer is a simple interface for external packages
func (mm *MemoryManager) ReleaseBuffer(buffer unsafe.Pointer) {
	if buffer == nil {
		return
	}
	
	// Look up the actual buffer size
	mm.bufferSizesMutex.RLock()
	size, exists := mm.bufferSizes[buffer]
	mm.bufferSizesMutex.RUnlock()
	
	if !exists {
		// Buffer not tracked - this shouldn't happen in normal operation
		// Use a reasonable default size and log a warning
		size = 4194304 // 4MB default as fallback
		fmt.Printf("Warning: releasing untracked buffer %p, using default size %d\n", buffer, size)
	}
	
	mm.ReturnBuffer(buffer, size, GPU)
}

// Mock functions for testing when Metal device is not available

// CreateMockDevice creates a mock Metal device for testing
func CreateMockDevice() unsafe.Pointer {
	// Return a non-nil pointer for testing
	mockDevice := uintptr(0x1000) // Arbitrary non-zero value
	return unsafe.Pointer(mockDevice)
}

// CreateMockCommandQueue creates a mock Metal command queue for testing
func CreateMockCommandQueue() unsafe.Pointer {
	// Return a non-nil pointer for testing
	mockQueue := uintptr(0x2000) // Arbitrary non-zero value
	return unsafe.Pointer(mockQueue)
}