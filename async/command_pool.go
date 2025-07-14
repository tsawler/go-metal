package async

import (
	"fmt"
	"sync"
	"unsafe"
	
	"github.com/tsawler/go-metal/cgo_bridge"
)

// CommandBuffer represents a Metal command buffer wrapper
type CommandBuffer struct {
	buffer unsafe.Pointer // MTLCommandBuffer
	inUse  bool           // Whether buffer is currently in use
	id     int            // Unique identifier for debugging
}

// GetBuffer returns the underlying Metal command buffer pointer
func (cb *CommandBuffer) GetBuffer() unsafe.Pointer {
	return cb.buffer
}

// GetID returns the unique identifier for debugging
func (cb *CommandBuffer) GetID() int {
	return cb.id
}

// IsInUse returns whether the buffer is currently in use
func (cb *CommandBuffer) IsInUse() bool {
	return cb.inUse
}

// CommandBufferPool manages a pool of Metal command buffers for reuse
type CommandBufferPool struct {
	commandQueue  unsafe.Pointer       // MTLCommandQueue
	buffers       []*CommandBuffer
	available     chan *CommandBuffer
	maxBuffers    int
	mutex         sync.Mutex
	nextID        int
	closed        bool                  // Track if pool has been cleaned up
}

// NewCommandBufferPool creates a new command buffer pool
func NewCommandBufferPool(commandQueue unsafe.Pointer, maxBuffers int) (*CommandBufferPool, error) {
	if commandQueue == nil {
		return nil, fmt.Errorf("command queue cannot be nil")
	}
	
	if maxBuffers <= 0 {
		return nil, fmt.Errorf("maxBuffers must be positive, got %d", maxBuffers)
	}
	
	pool := &CommandBufferPool{
		commandQueue: commandQueue,
		buffers:      make([]*CommandBuffer, 0, maxBuffers),
		available:    make(chan *CommandBuffer, maxBuffers),
		maxBuffers:   maxBuffers,
		nextID:       1,
	}
	
	// Pre-allocate some command buffers
	initialBuffers := maxBuffers / 2
	if initialBuffers < 2 {
		initialBuffers = 2
	}
	
	for i := 0; i < initialBuffers; i++ {
		pool.mutex.Lock()
		buffer, err := pool.createBuffer()
		pool.mutex.Unlock()
		if err != nil {
			pool.Cleanup()
			return nil, fmt.Errorf("failed to create initial command buffer %d: %v", i, err)
		}
		pool.buffers = append(pool.buffers, buffer)
		pool.available <- buffer
	}
	
	return pool, nil
}

// createBuffer creates a new command buffer
// NOTE: This function expects the caller to hold cbp.mutex
func (cbp *CommandBufferPool) createBuffer() (*CommandBuffer, error) {
	id := cbp.nextID
	cbp.nextID++
	
	// IMPLEMENTED: Create actual MTLCommandBuffer using CGO bridge
	buffer, err := cgo_bridge.CreateCommandBuffer(cbp.commandQueue)
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal command buffer: %v", err)
	}
	
	return &CommandBuffer{
		buffer: buffer,
		inUse:  false,
		id:     id,
	}, nil
}

// GetBuffer gets an available command buffer (creates new one if needed and under limit)
func (cbp *CommandBufferPool) GetBuffer() (*CommandBuffer, error) {
	select {
	case buffer := <-cbp.available:
		buffer.inUse = true
		return buffer, nil
	default:
		// No buffer available, try to create new one if under limit
		cbp.mutex.Lock()
		defer cbp.mutex.Unlock()
		
		if len(cbp.buffers) < cbp.maxBuffers {
			buffer, err := cbp.createBuffer()
			if err != nil {
				return nil, fmt.Errorf("failed to create new command buffer: %v", err)
			}
			cbp.buffers = append(cbp.buffers, buffer)
			buffer.inUse = true
			return buffer, nil
		}
		
		return nil, fmt.Errorf("no command buffers available and pool is at capacity (%d)", cbp.maxBuffers)
	}
}

// ReturnBuffer returns a command buffer to the pool after completion
func (cbp *CommandBufferPool) ReturnBuffer(buffer *CommandBuffer) {
	if buffer == nil {
		return
	}
	
	buffer.inUse = false
	
	cbp.mutex.Lock()
	closed := cbp.closed
	cbp.mutex.Unlock()
	
	// Don't try to return to a closed pool
	if closed {
		return
	}
	
	select {
	case cbp.available <- buffer:
		// Successfully returned to pool
	default:
		// Pool channel is full, this shouldn't happen but handle gracefully
		// Buffer will be garbage collected when pool is cleaned up
	}
}

// ExecuteAsync submits a command buffer for async execution
func (cbp *CommandBufferPool) ExecuteAsync(buffer *CommandBuffer, completion func(error)) error {
	if buffer == nil {
		return fmt.Errorf("command buffer is nil")
	}
	
	if !buffer.inUse {
		return fmt.Errorf("command buffer is not marked as in use")
	}
	
	// IMPLEMENTED: Actual Metal command buffer execution
	go func() {
		var err error
		
		// Setup autorelease pool for Metal resource management
		cgo_bridge.SetupAutoreleasePool()
		defer cgo_bridge.DrainAutoreleasePool()
		
		// Commit command buffer for execution (Metal command buffers are one-time use)
		if commitErr := cgo_bridge.CommitCommandBuffer(buffer.buffer); commitErr != nil {
			err = fmt.Errorf("failed to commit command buffer: %v", commitErr)
		} else {
			// Wait for completion
			if waitErr := cgo_bridge.WaitCommandBufferCompletion(buffer.buffer); waitErr != nil {
				err = fmt.Errorf("command buffer execution failed: %v", waitErr)
			}
		}
		
		// After commit, the Metal command buffer is no longer usable
		// We need to create a new one for this buffer wrapper
		cbp.mutex.Lock()
		if buffer.buffer != nil {
			// Release the old committed buffer
			cgo_bridge.ReleaseCommandBuffer(buffer.buffer)
			// Create a new Metal command buffer for reuse
			newBuffer, createErr := cgo_bridge.CreateCommandBuffer(cbp.commandQueue)
			if createErr != nil {
				// If we can't create a new buffer, mark this one as unusable
				buffer.buffer = nil
			} else {
				buffer.buffer = newBuffer
			}
		}
		cbp.mutex.Unlock()
		
		// Call completion handler
		if completion != nil {
			completion(err)
		}
		
		// Return buffer to pool
		cbp.ReturnBuffer(buffer)
	}()
	
	return nil
}

// Batch operation support for multiple operations in single command buffer
type BatchOperation struct {
	Type        string // "training_step", "data_transfer", etc.
	Data        interface{}
	Completion  func(error)
}

// ExecuteBatch executes multiple operations in a single command buffer for efficiency
func (cbp *CommandBufferPool) ExecuteBatch(operations []BatchOperation) error {
	if len(operations) == 0 {
		return fmt.Errorf("no operations provided")
	}
	
	buffer, err := cbp.GetBuffer()
	if err != nil {
		return fmt.Errorf("failed to get command buffer: %v", err)
	}
	
	// TODO: Implement actual batch operation encoding
	// This would involve encoding all operations into the command buffer
	// before committing for execution
	
	// Aggregate completion handlers
	allCompletions := make([]func(error), 0, len(operations))
	for _, op := range operations {
		if op.Completion != nil {
			allCompletions = append(allCompletions, op.Completion)
		}
	}
	
	// Execute with combined completion handler
	return cbp.ExecuteAsync(buffer, func(err error) {
		for _, completion := range allCompletions {
			completion(err)
		}
	})
}

// Stats returns statistics about the command buffer pool
func (cbp *CommandBufferPool) Stats() CommandPoolStats {
	cbp.mutex.Lock()
	defer cbp.mutex.Unlock()
	
	inUseCount := 0
	for _, buffer := range cbp.buffers {
		if buffer.inUse {
			inUseCount++
		}
	}
	
	return CommandPoolStats{
		TotalBuffers:     len(cbp.buffers),
		AvailableBuffers: len(cbp.available),
		InUseBuffers:     inUseCount,
		MaxBuffers:       cbp.maxBuffers,
	}
}

// CommandPoolStats provides statistics about the command buffer pool
type CommandPoolStats struct {
	TotalBuffers     int
	AvailableBuffers int
	InUseBuffers     int
	MaxBuffers       int
}

// Cleanup releases all command buffers
func (cbp *CommandBufferPool) Cleanup() {
	cbp.mutex.Lock()
	defer cbp.mutex.Unlock()
	
	// Prevent double cleanup
	if cbp.closed {
		return
	}
	cbp.closed = true
	
	// Drain available channel
	close(cbp.available)
	for range cbp.available {
		// Drain remaining buffers
	}
	
	// Release all buffers
	for _, buffer := range cbp.buffers {
		if buffer.buffer != nil {
			// IMPLEMENTED: Release MTLCommandBuffer using CGO bridge
			cgo_bridge.ReleaseCommandBuffer(buffer.buffer)
			buffer.buffer = nil
		}
	}
	
	cbp.buffers = nil
}