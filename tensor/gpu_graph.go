package tensor

import (
	"fmt"
	"sync"
	"github.com/tsawler/go-metal/metal_bridge"
)

// GPUComputationGraph manages a graph of GPU operations with dependency tracking
type GPUComputationGraph struct {
	manager    *metal_bridge.CommandBufferManager
	engine     *metal_bridge.ComputeEngine
	
	// Graph state
	mutex      sync.RWMutex
	operations map[metal_bridge.OperationID]*GraphOperation
	
	// Resource tracking for memory safety
	tensorRefs map[*Tensor]int32 // Reference counting for tensor lifetime
}

// GraphOperation represents a single operation in the computation graph
type GraphOperation struct {
	ID           metal_bridge.OperationID
	Type         string
	InputTensors []*Tensor
	OutputTensor *Tensor
	Dependencies []metal_bridge.OperationID
	
	// Operation-specific data
	Params       map[string]interface{}
	
	// Completion tracking
	completed    bool
	result       *Tensor
	error        error
	completionCh chan struct{}
}

// NewGPUComputationGraph creates a new GPU computation graph
func NewGPUComputationGraph() (*GPUComputationGraph, error) {
	engine, err := getComputeEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get compute engine: %v", err)
	}
	
	// Create command buffer manager
	device := engine.GetDevice()
	queue := engine.GetCommandQueue()
	manager := metal_bridge.NewCommandBufferManager(device, queue)
	
	return &GPUComputationGraph{
		manager:    manager,
		engine:     engine,
		operations: make(map[metal_bridge.OperationID]*GraphOperation),
		tensorRefs: make(map[*Tensor]int32),
	}, nil
}

// AddOperation adds an operation to the computation graph
func (g *GPUComputationGraph) AddOperation(opType string, inputs []*Tensor, dependencies []metal_bridge.OperationID, params map[string]interface{}) (metal_bridge.OperationID, error) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	opID := g.manager.GenerateOperationID()
	
	// Create graph operation
	graphOp := &GraphOperation{
		ID:           opID,
		Type:         opType,
		InputTensors: inputs,
		Dependencies: dependencies,
		Params:       params,
		completionCh: make(chan struct{}),
	}
	
	// Increment reference counts for input tensors (skip nil placeholders)
	for _, tensor := range inputs {
		if tensor != nil {
			g.tensorRefs[tensor]++
			tensor.Retain() // Use the tensor's built-in reference counting
		}
	}
	
	g.operations[opID] = graphOp
	
	// Create the pending operation for the command buffer manager
	pendingOp, err := g.createPendingOperation(graphOp)
	if err != nil {
		// Cleanup on error (skip nil placeholders)
		for _, tensor := range inputs {
			if tensor != nil {
				g.tensorRefs[tensor]--
				tensor.Release()
			}
		}
		delete(g.operations, opID)
		return 0, fmt.Errorf("failed to create pending operation: %v", err)
	}
	
	// Queue the operation
	err = g.manager.QueueOperation(pendingOp)
	if err != nil {
		// Cleanup on error
		for _, tensor := range inputs {
			g.tensorRefs[tensor]--
			tensor.Release()
		}
		delete(g.operations, opID)
		return 0, fmt.Errorf("failed to queue operation: %v", err)
	}
	
	return opID, nil
}

// createPendingOperation creates a PendingOperation for the command buffer manager
func (g *GPUComputationGraph) createPendingOperation(graphOp *GraphOperation) (*metal_bridge.PendingOperation, error) {
	var inputBuffers, outputBuffers, tempBuffers []*metal_bridge.Buffer
	
	// Convert input tensors to buffers (this may need buffer creation)
	for _, tensor := range graphOp.InputTensors {
		if tensor != nil && tensor.Device == GPU && tensor.GetGPUBuffer() != nil {
			if buf, ok := tensor.GetGPUBuffer().(*metal_bridge.Buffer); ok {
				inputBuffers = append(inputBuffers, buf)
			}
		}
	}
	
	// Create the execution function based on operation type
	executeFunc, err := g.createExecuteFunction(graphOp)
	if err != nil {
		return nil, err
	}
	
	// Create cleanup function
	cleanupFunc := func() error {
		return g.cleanupOperation(graphOp)
	}
	
	// Create completion callback
	completionFunc := func(err error) {
		g.handleOperationCompletion(graphOp, err)
	}
	
	return &metal_bridge.PendingOperation{
		ID:            graphOp.ID,
		Dependencies:  graphOp.Dependencies,
		Execute:       executeFunc,
		Cleanup:       cleanupFunc,
		OnComplete:    completionFunc,
		InputBuffers:  inputBuffers,
		OutputBuffers: outputBuffers,
		TempBuffers:   tempBuffers,
	}, nil
}

// createExecuteFunction creates the execution function for a specific operation type
func (g *GPUComputationGraph) createExecuteFunction(graphOp *GraphOperation) (func() error, error) {
	switch graphOp.Type {
	case "MatMul":
		return g.createMatMulExecute(graphOp), nil
	case "Add":
		return g.createAddExecute(graphOp), nil
	case "ReLU":
		return g.createReLUExecute(graphOp), nil
	default:
		return nil, fmt.Errorf("unsupported operation type: %s", graphOp.Type)
	}
}

// createMatMulExecute creates the execution function for matrix multiplication
func (g *GPUComputationGraph) createMatMulExecute(graphOp *GraphOperation) func() error {
	return func() error {
		if len(graphOp.InputTensors) != 2 {
			return fmt.Errorf("MatMul requires exactly 2 input tensors")
		}
		
		t1, t2 := graphOp.InputTensors[0], graphOp.InputTensors[1]
		
		// Check for nil tensors which indicate unresolved dependencies
		if t1 == nil || t2 == nil {
			return fmt.Errorf("MatMul operation has nil input tensors - dependency resolution not implemented for async execution")
		}
		
		// Use synchronous MatMul for now to avoid hanging issues
		// This can be optimized later once the basic infrastructure is stable
		var result *Tensor
		var err error
		
		if t1.Device == GPU || t2.Device == GPU {
			result, err = MatMulGPU(t1, t2)
		} else {
			result, err = MatMul(t1, t2)
		}
		
		if err != nil {
			return err
		}
		
		graphOp.result = result
		return nil
	}
}

// createAddExecute creates the execution function for tensor addition
func (g *GPUComputationGraph) createAddExecute(graphOp *GraphOperation) func() error {
	return func() error {
		if len(graphOp.InputTensors) != 2 {
			return fmt.Errorf("Add requires exactly 2 input tensors")
		}
		
		t1, t2 := graphOp.InputTensors[0], graphOp.InputTensors[1]
		
		// Check for nil tensors which indicate unresolved dependencies
		if t1 == nil || t2 == nil {
			return fmt.Errorf("Add operation has nil input tensors - dependency resolution not implemented for async execution")
		}
		
		// Use synchronous Add for now to avoid hanging issues
		var result *Tensor
		var err error
		
		if t1.Device == GPU || t2.Device == GPU {
			result, err = AddGPU(t1, t2)
		} else {
			result, err = Add(t1, t2)
		}
		
		if err != nil {
			return err
		}
		
		graphOp.result = result
		return nil
	}
}

// createReLUExecute creates the execution function for ReLU activation
func (g *GPUComputationGraph) createReLUExecute(graphOp *GraphOperation) func() error {
	return func() error {
		if len(graphOp.InputTensors) != 1 {
			return fmt.Errorf("ReLU requires exactly 1 input tensor")
		}
		
		input := graphOp.InputTensors[0]
		
		// Create a simple ReLU operation using the existing GPU infrastructure
		resultCh := make(chan *Tensor, 1)
		errorCh := make(chan error, 1)
		
		// Use CPU fallback for ReLU for now, can be optimized later
		go func() {
			result, err := ReLU(input)
			if err != nil {
				errorCh <- err
			} else {
				resultCh <- result
			}
		}()
		
		// Wait for completion
		select {
		case result := <-resultCh:
			graphOp.result = result
			return nil
		case err := <-errorCh:
			return err
		}
	}
}

// cleanupOperation performs cleanup for a completed operation
func (g *GPUComputationGraph) cleanupOperation(graphOp *GraphOperation) error {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	// Decrement reference counts for input tensors (skip nil placeholders)
	for _, tensor := range graphOp.InputTensors {
		if tensor != nil {
			g.tensorRefs[tensor]--
			tensor.Release()
			
			// If reference count reaches zero, tensor can be garbage collected
			if g.tensorRefs[tensor] <= 0 {
				delete(g.tensorRefs, tensor)
			}
		}
	}
	
	return nil
}

// handleOperationCompletion handles the completion of an operation
func (g *GPUComputationGraph) handleOperationCompletion(graphOp *GraphOperation, err error) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	graphOp.completed = true
	graphOp.error = err
	
	// Signal completion
	close(graphOp.completionCh)
}

// WaitForOperation waits for a specific operation to complete
func (g *GPUComputationGraph) WaitForOperation(opID metal_bridge.OperationID) (*Tensor, error) {
	g.mutex.RLock()
	graphOp, exists := g.operations[opID]
	g.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("operation %d not found", opID)
	}
	
	// Wait for completion
	<-graphOp.completionCh
	
	if graphOp.error != nil {
		return nil, graphOp.error
	}
	
	return graphOp.result, nil
}

// ExecuteSequence executes a sequence of operations and returns the final result
func (g *GPUComputationGraph) ExecuteSequence(operations []OperationDesc) (*Tensor, error) {
	var lastOpID metal_bridge.OperationID
	var dependencies []metal_bridge.OperationID
	var lastResult *Tensor
	
	for i, opDesc := range operations {
		// Resolve nil placeholders with previous result
		actualInputs := make([]*Tensor, len(opDesc.Inputs))
		copy(actualInputs, opDesc.Inputs)
		
		for j, input := range actualInputs {
			if input == nil && lastResult != nil {
				// Replace nil placeholder with the previous result
				actualInputs[j] = lastResult
			}
		}
		
		if i > 0 {
			dependencies = []metal_bridge.OperationID{lastOpID}
		} else {
			dependencies = nil
		}
		
		opID, err := g.AddOperation(opDesc.Type, actualInputs, dependencies, opDesc.Params)
		if err != nil {
			return nil, fmt.Errorf("failed to add operation %d: %v", i, err)
		}
		
		lastOpID = opID
		
		// Get the result to use for the next operation's nil placeholder resolution
		lastResult, err = g.WaitForOperation(opID)
		if err != nil {
			return nil, fmt.Errorf("operation %d failed: %v", i, err)
		}
	}
	
	// Return the final result
	return lastResult, nil
}

// OperationDesc describes an operation to be added to the graph
type OperationDesc struct {
	Type   string
	Inputs []*Tensor
	Params map[string]interface{}
}

// GetStats returns statistics about the computation graph
func (g *GPUComputationGraph) GetStats() (queued, executed int64, pending int) {
	return g.manager.GetStats()
}

// Shutdown gracefully shuts down the computation graph
func (g *GPUComputationGraph) Shutdown() {
	g.manager.Shutdown()
	
	// Release all remaining tensor references
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	for tensor, refCount := range g.tensorRefs {
		for i := int32(0); i < refCount; i++ {
			tensor.Release()
		}
	}
	g.tensorRefs = make(map[*Tensor]int32)
}