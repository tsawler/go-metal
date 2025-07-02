package async

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/tsawler/go-metal/memory"
)

// StagingBuffer represents a CPU-accessible buffer for GPU transfers
type StagingBuffer struct {
	buffer    unsafe.Pointer // MTLBuffer with CPU access
	size      int            // Buffer size in bytes
	inUse     bool           // Whether buffer is currently in use
	id        int            // Unique identifier for debugging
}

// StagingBufferPool manages a pool of staging buffers for async CPU→GPU transfers
type StagingBufferPool struct {
	buffers       []*StagingBuffer
	available     chan *StagingBuffer
	memoryManager *memory.MemoryManager
	maxBuffers    int
	bufferSize    int // Fixed size for all buffers in this pool
	mutex         sync.Mutex
	nextID        int
}

// NewStagingBufferPool creates a new staging buffer pool
func NewStagingBufferPool(memoryManager *memory.MemoryManager, maxBuffers int) (*StagingBufferPool, error) {
	if memoryManager == nil {
		return nil, fmt.Errorf("memory manager cannot be nil")
	}
	
	if maxBuffers <= 0 {
		return nil, fmt.Errorf("maxBuffers must be positive, got %d", maxBuffers)
	}
	
	// Use 4MB buffers - good for typical batch sizes
	bufferSize := 4 * 1024 * 1024 // 4MB
	
	pool := &StagingBufferPool{
		buffers:       make([]*StagingBuffer, 0, maxBuffers),
		available:     make(chan *StagingBuffer, maxBuffers),
		memoryManager: memoryManager,
		maxBuffers:    maxBuffers,
		bufferSize:    bufferSize,
		nextID:        1,
	}
	
	// Pre-allocate some staging buffers
	initialBuffers := maxBuffers / 2
	if initialBuffers < 2 {
		initialBuffers = 2
	}
	
	for i := 0; i < initialBuffers; i++ {
		buffer, err := pool.createBuffer()
		if err != nil {
			pool.Cleanup()
			return nil, fmt.Errorf("failed to create initial staging buffer %d: %v", i, err)
		}
		pool.buffers = append(pool.buffers, buffer)
		pool.available <- buffer
	}
	
	return pool, nil
}

// createBuffer creates a new staging buffer
func (sbp *StagingBufferPool) createBuffer() (*StagingBuffer, error) {
	// Create CPU-accessible buffer for staging
	// Note: We'll use the memory manager's device to create MTLBuffer with shared storage mode
	buffer := sbp.memoryManager.AllocateBuffer(sbp.bufferSize)
	if buffer == nil {
		return nil, fmt.Errorf("failed to allocate staging buffer")
	}
	
	sbp.mutex.Lock()
	id := sbp.nextID
	sbp.nextID++
	sbp.mutex.Unlock()
	
	return &StagingBuffer{
		buffer: buffer,
		size:   sbp.bufferSize,
		inUse:  false,
		id:     id,
	}, nil
}

// GetBuffer gets an available staging buffer (creates new one if needed and under limit)
func (sbp *StagingBufferPool) GetBuffer() (*StagingBuffer, error) {
	select {
	case buffer := <-sbp.available:
		buffer.inUse = true
		return buffer, nil
	default:
		// No buffer available, try to create new one if under limit
		sbp.mutex.Lock()
		defer sbp.mutex.Unlock()
		
		if len(sbp.buffers) < sbp.maxBuffers {
			buffer, err := sbp.createBuffer()
			if err != nil {
				return nil, fmt.Errorf("failed to create new staging buffer: %v", err)
			}
			sbp.buffers = append(sbp.buffers, buffer)
			buffer.inUse = true
			return buffer, nil
		}
		
		return nil, fmt.Errorf("no staging buffers available and pool is at capacity (%d)", sbp.maxBuffers)
	}
}

// ReturnBuffer returns a staging buffer to the pool
func (sbp *StagingBufferPool) ReturnBuffer(buffer *StagingBuffer) {
	if buffer == nil {
		return
	}
	
	buffer.inUse = false
	
	select {
	case sbp.available <- buffer:
		// Successfully returned to pool
	default:
		// Pool channel is full, deallocate the buffer to prevent a leak
		if buffer.buffer != nil { // Check if the underlying Metal buffer pointer is valid
			sbp.memoryManager.ReleaseBuffer(buffer.buffer) // This calls cgo_bridge.DeallocateMetalBuffer
			buffer.buffer = nil // Clear the pointer to prevent double-free/use-after-free
		}
	}
}

// TransferToGPU transfers data from CPU slice to GPU tensor using staging buffers
func (sbp *StagingBufferPool) TransferToGPU(data interface{}, gpuTensor *memory.Tensor) error {
	var dataBytes []byte
	var err error
	
	// Convert data to bytes based on type
	switch d := data.(type) {
	case []float32:
		dataBytes = make([]byte, len(d)*4) // 4 bytes per float32
		for i, f := range d {
			bits := *(*uint32)(unsafe.Pointer(&f))
			dataBytes[i*4] = byte(bits)
			dataBytes[i*4+1] = byte(bits >> 8)
			dataBytes[i*4+2] = byte(bits >> 16)
			dataBytes[i*4+3] = byte(bits >> 24)
		}
	case []int32:
		dataBytes = make([]byte, len(d)*4) // 4 bytes per int32
		for i, val := range d {
			dataBytes[i*4] = byte(val)
			dataBytes[i*4+1] = byte(val >> 8)
			dataBytes[i*4+2] = byte(val >> 16)
			dataBytes[i*4+3] = byte(val >> 24)
		}
	case []byte:
		dataBytes = d
	default:
		return fmt.Errorf("unsupported data type for GPU transfer: %T", data)
	}
	
	if len(dataBytes) > sbp.bufferSize {
		return fmt.Errorf("data size %d exceeds staging buffer size %d", len(dataBytes), sbp.bufferSize)
	}
	
	// Get staging buffer
	stagingBuffer, err := sbp.GetBuffer()
	if err != nil {
		return fmt.Errorf("failed to get staging buffer: %v", err)
	}
	defer sbp.ReturnBuffer(stagingBuffer)
	
	// For now, we'll rely on the memory manager to handle the actual transfer
	// In a complete implementation, this would:
	// 1. Copy data to staging buffer (CPU-accessible)
	// 2. Issue Metal blit command to copy from staging to GPU tensor
	// 3. Wait for completion or handle async with callbacks
	
	// TODO: Implement actual Metal buffer copy operations
	// This is a placeholder that shows the structure
	_ = dataBytes
	_ = stagingBuffer
	_ = gpuTensor
	
	return nil
}

// Stats returns statistics about the staging buffer pool
func (sbp *StagingBufferPool) Stats() StagingPoolStats {
	sbp.mutex.Lock()
	defer sbp.mutex.Unlock()
	
	inUseCount := 0
	for _, buffer := range sbp.buffers {
		if buffer.inUse {
			inUseCount++
		}
	}
	
	return StagingPoolStats{
		TotalBuffers:     len(sbp.buffers),
		AvailableBuffers: len(sbp.available),
		InUseBuffers:     inUseCount,
		BufferSize:       sbp.bufferSize,
		MaxBuffers:       sbp.maxBuffers,
	}
}

// StagingPoolStats provides statistics about the staging buffer pool
type StagingPoolStats struct {
	TotalBuffers     int
	AvailableBuffers int
	InUseBuffers     int
	BufferSize       int
	MaxBuffers       int
}

// Cleanup releases all staging buffers
func (sbp *StagingBufferPool) Cleanup() {
	sbp.mutex.Lock()
	defer sbp.mutex.Unlock()
	
	// Drain available channel
	close(sbp.available)
	for range sbp.available {
		// Drain remaining buffers
	}
	
	// Release all buffers
	for _, buffer := range sbp.buffers {
		if buffer.buffer != nil {
			sbp.memoryManager.ReleaseBuffer(buffer.buffer)
			buffer.buffer = nil
		}
	}
	
	sbp.buffers = nil
}

