package metal_bridge

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// OperationID represents a unique identifier for GPU operations
type OperationID uint64

// CommandBufferManager manages command buffer queuing and dependency tracking
type CommandBufferManager struct {
	device       *Device
	commandQueue *CommandQueue
	
	// Operation tracking
	nextOpID     uint64
	pendingOps   map[OperationID]*PendingOperation
	dependencies map[OperationID][]OperationID
	completedOps map[OperationID]bool
	
	// Resource management
	resourceMutex sync.RWMutex
	bufferPool    map[*Buffer]bool // Track active buffers
	
	// Synchronization
	mutex        sync.RWMutex
	operationCh  chan *PendingOperation
	shutdownCh   chan struct{}
	
	// Metrics for debugging
	operationsQueued   int64
	operationsExecuted int64
}

// PendingOperation represents an operation waiting to be executed
type PendingOperation struct {
	ID           OperationID
	Dependencies []OperationID
	Execute      func() error
	Cleanup      func() error
	OnComplete   func(error)
	
	// Resource tracking for memory safety
	InputBuffers  []*Buffer
	OutputBuffers []*Buffer
	TempBuffers   []*Buffer
	
	// State tracking
	isReady     bool
	isExecuting bool
	isCompleted bool
	error       error
}

// NewCommandBufferManager creates a new command buffer manager
func NewCommandBufferManager(device *Device, commandQueue *CommandQueue) *CommandBufferManager {
	manager := &CommandBufferManager{
		device:       device,
		commandQueue: commandQueue,
		pendingOps:   make(map[OperationID]*PendingOperation),
		dependencies: make(map[OperationID][]OperationID),
		completedOps: make(map[OperationID]bool),
		bufferPool:   make(map[*Buffer]bool),
		operationCh:  make(chan *PendingOperation, 100), // Buffered channel for queuing
		shutdownCh:   make(chan struct{}),
	}
	
	// Start the operation processing goroutine
	go manager.processOperations()
	
	return manager
}

// GenerateOperationID creates a unique operation ID
func (m *CommandBufferManager) GenerateOperationID() OperationID {
	return OperationID(atomic.AddUint64(&m.nextOpID, 1))
}

// QueueOperation adds an operation to the queue with dependency tracking
func (m *CommandBufferManager) QueueOperation(op *PendingOperation) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Validate dependencies exist or are completed
	for _, depID := range op.Dependencies {
		if !m.completedOps[depID] {
			if _, exists := m.pendingOps[depID]; !exists {
				return fmt.Errorf("dependency operation %d does not exist", depID)
			}
		}
	}
	
	// Register the operation
	m.pendingOps[op.ID] = op
	m.dependencies[op.ID] = op.Dependencies
	
	// Track input/output buffers for resource management
	m.resourceMutex.Lock()
	for _, buf := range op.InputBuffers {
		m.bufferPool[buf] = true
	}
	for _, buf := range op.OutputBuffers {
		m.bufferPool[buf] = true
	}
	for _, buf := range op.TempBuffers {
		m.bufferPool[buf] = true
	}
	m.resourceMutex.Unlock()
	
	// Check if operation is ready to execute
	op.isReady = m.checkOperationReady(op.ID)
	
	if op.isReady {
		// Queue for immediate execution
		select {
		case m.operationCh <- op:
			atomic.AddInt64(&m.operationsQueued, 1)
		default:
			return fmt.Errorf("operation queue is full")
		}
	}
	
	return nil
}

// checkOperationReady determines if an operation's dependencies are satisfied
func (m *CommandBufferManager) checkOperationReady(opID OperationID) bool {
	deps := m.dependencies[opID]
	for _, depID := range deps {
		if !m.completedOps[depID] {
			return false
		}
	}
	return true
}

// processOperations is the main goroutine that processes queued operations
func (m *CommandBufferManager) processOperations() {
	for {
		select {
		case op := <-m.operationCh:
			m.executeOperation(op)
		case <-m.shutdownCh:
			return
		}
	}
}

// executeOperation executes a single operation asynchronously
func (m *CommandBufferManager) executeOperation(op *PendingOperation) {
	m.mutex.Lock()
	op.isExecuting = true
	m.mutex.Unlock()
	
	// Execute the operation
	err := op.Execute()
	
	// Handle completion
	m.handleOperationCompletion(op, err)
}

// handleOperationCompletion processes operation completion and triggers dependent operations
func (m *CommandBufferManager) handleOperationCompletion(op *PendingOperation, err error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Mark operation as completed
	op.isCompleted = true
	op.error = err
	m.completedOps[op.ID] = true
	delete(m.pendingOps, op.ID)
	
	atomic.AddInt64(&m.operationsExecuted, 1)
	
	// Resource cleanup
	m.cleanupOperationResources(op)
	
	// Notify completion callback
	if op.OnComplete != nil {
		go op.OnComplete(err)
	}
	
	// Check if any pending operations are now ready
	m.checkAndQueueReadyOperations()
}

// cleanupOperationResources performs resource cleanup according to README specifications
func (m *CommandBufferManager) cleanupOperationResources(op *PendingOperation) {
	m.resourceMutex.Lock()
	defer m.resourceMutex.Unlock()
	
	// Execute custom cleanup if provided
	if op.Cleanup != nil {
		if cleanupErr := op.Cleanup(); cleanupErr != nil {
			// Log cleanup error but don't fail the operation
			fmt.Printf("Warning: cleanup error for operation %d: %v\n", op.ID, cleanupErr)
		}
	}
	
	// Release temporary buffers back to allocator
	for _, buf := range op.TempBuffers {
		delete(m.bufferPool, buf)
		// Note: Actual buffer release should be handled by BufferAllocator
		// This just removes from our tracking
	}
	
	// Keep input/output buffers tracked until explicitly released
	// They may be used by other operations
}

// checkAndQueueReadyOperations finds operations that are now ready to execute
func (m *CommandBufferManager) checkAndQueueReadyOperations() {
	for opID, op := range m.pendingOps {
		if !op.isReady && !op.isExecuting && m.checkOperationReady(opID) {
			op.isReady = true
			
			// Queue for execution
			select {
			case m.operationCh <- op:
				atomic.AddInt64(&m.operationsQueued, 1)
			default:
				// Queue is full, operation will be checked again later
				fmt.Printf("Warning: operation queue full, operation %d delayed\n", opID)
			}
		}
	}
}

// WaitForOperation blocks until a specific operation completes
func (m *CommandBufferManager) WaitForOperation(opID OperationID) error {
	for {
		m.mutex.RLock()
		if m.completedOps[opID] {
			m.mutex.RUnlock()
			if op, exists := m.pendingOps[opID]; exists {
				return op.error
			}
			return nil
		}
		m.mutex.RUnlock()
		
		// Small delay to avoid busy waiting
		// In production, this could use condition variables
		select {
		case <-m.shutdownCh:
			return fmt.Errorf("manager shutting down")
		default:
			// Continue checking
		}
	}
}

// GetStats returns operation statistics for monitoring
func (m *CommandBufferManager) GetStats() (queued, executed int64, pending int) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	return atomic.LoadInt64(&m.operationsQueued), 
		   atomic.LoadInt64(&m.operationsExecuted), 
		   len(m.pendingOps)
}

// Shutdown gracefully shuts down the command buffer manager
func (m *CommandBufferManager) Shutdown() {
	close(m.shutdownCh)
	
	// Wait for pending operations to complete
	for {
		m.mutex.RLock()
		pendingCount := len(m.pendingOps)
		m.mutex.RUnlock()
		
		if pendingCount == 0 {
			break
		}
	}
	
	// Final resource cleanup
	m.resourceMutex.Lock()
	m.bufferPool = make(map[*Buffer]bool)
	m.resourceMutex.Unlock()
}