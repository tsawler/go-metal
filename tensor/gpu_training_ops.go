package tensor

import (
	"fmt"
	"sync"
	"github.com/tsawler/go-metal/metal_bridge"
)

// GPUTrainingContext manages GPU operations for neural network training
type GPUTrainingContext struct {
	graph   *GPUComputationGraph
	mutex   sync.RWMutex
	
	// Operation batching
	batchOps    []OperationDesc
	batchMutex  sync.Mutex
	batchSize   int
	
	// Performance metrics
	totalOps       int64
	batchedOps     int64
	avgBatchSize   float64
}

// NewGPUTrainingContext creates a new GPU training context
func NewGPUTrainingContext() (*GPUTrainingContext, error) {
	graph, err := NewGPUComputationGraph()
	if err != nil {
		return nil, fmt.Errorf("failed to create GPU computation graph: %v", err)
	}
	
	return &GPUTrainingContext{
		graph:     graph,
		batchSize: 16, // Default batch size for operation queuing
	}, nil
}

// LinearLayerForwardAsync performs a linear layer forward pass asynchronously
// This combines MatMul + Bias addition + optional activation in a dependency chain
func (ctx *GPUTrainingContext) LinearLayerForwardAsync(input, weight, bias *Tensor, activation string) (*Tensor, error) {
	// Create operation sequence with dependencies
	operations := []OperationDesc{
		{
			Type:   "MatMul",
			Inputs: []*Tensor{input, weight},
			Params: nil,
		},
	}
	
	// Add bias if provided
	if bias != nil {
		operations = append(operations, OperationDesc{
			Type:   "Add", 
			Inputs: []*Tensor{nil, bias}, // nil will be replaced with previous result
			Params: nil,
		})
	}
	
	// Add activation if specified
	if activation != "" {
		operations = append(operations, OperationDesc{
			Type:   activation,
			Inputs: []*Tensor{nil}, // nil will be replaced with previous result
			Params: nil,
		})
	}
	
	// Execute the sequence
	return ctx.graph.ExecuteSequence(operations)
}

// BatchOperationsAsync batches multiple operations for efficient GPU execution
// This version includes automatic operation fusion optimization
func (ctx *GPUTrainingContext) BatchOperationsAsync(ops []OperationDesc) ([]*Tensor, error) {
	ctx.batchMutex.Lock()
	defer ctx.batchMutex.Unlock()
	
	// First, optimize the operation sequence with fusion detection
	optimizedOps, err := OptimizeOperationSequence(ops)
	if err != nil {
		return nil, fmt.Errorf("failed to optimize operation sequence: %v", err)
	}
	
	var results []*Tensor
	
	// Execute optimized operations
	for i, op := range optimizedOps {
		var result *Tensor
		
		// Check if this is a fused operation
		if IsFusedOperation(op.Type) {
			// Execute fused operation directly
			result, err = ExecuteFusedOperation(op)
			if err != nil {
				return nil, fmt.Errorf("failed to execute fused operation %d (%s): %v", i, op.Type, err)
			}
		} else {
			// Execute through computation graph for regular operations
			var deps []metal_bridge.OperationID
			if i > 0 && len(results) > 0 {
				// For now, we'll execute sequentially for non-fused operations
				// In the future, this could be optimized with better dependency tracking
			}
			
			opID, err := ctx.graph.AddOperation(op.Type, op.Inputs, deps, op.Params)
			if err != nil {
				return nil, fmt.Errorf("failed to add operation %d: %v", i, err)
			}
			
			result, err = ctx.graph.WaitForOperation(opID)
			if err != nil {
				return nil, fmt.Errorf("operation %d failed: %v", opID, err)
			}
		}
		
		results = append(results, result)
	}
	
	ctx.batchedOps += int64(len(optimizedOps))
	ctx.totalOps += int64(len(ops)) // Original operation count for metrics
	
	return results, nil
}

// QueueOperation adds an operation to the batch queue
func (ctx *GPUTrainingContext) QueueOperation(op OperationDesc) {
	ctx.batchMutex.Lock()
	defer ctx.batchMutex.Unlock()
	
	ctx.batchOps = append(ctx.batchOps, op)
	
	// Execute batch when it reaches the batch size
	if len(ctx.batchOps) >= ctx.batchSize {
		go ctx.executeBatch()
	}
}

// executeBatch executes the current batch of operations
func (ctx *GPUTrainingContext) executeBatch() {
	ctx.batchMutex.Lock()
	ops := make([]OperationDesc, len(ctx.batchOps))
	copy(ops, ctx.batchOps)
	ctx.batchOps = nil
	ctx.batchMutex.Unlock()
	
	if len(ops) == 0 {
		return
	}
	
	// Execute batched operations
	_, err := ctx.BatchOperationsAsync(ops)
	if err != nil {
		fmt.Printf("Error executing operation batch: %v\n", err)
	}
}

// FlushBatch executes any remaining operations in the batch
func (ctx *GPUTrainingContext) FlushBatch() error {
	ctx.batchMutex.Lock()
	if len(ctx.batchOps) == 0 {
		ctx.batchMutex.Unlock()
		return nil
	}
	
	ops := make([]OperationDesc, len(ctx.batchOps))
	copy(ops, ctx.batchOps)
	ctx.batchOps = nil
	ctx.batchMutex.Unlock()
	
	_, err := ctx.BatchOperationsAsync(ops)
	return err
}

// ConvolutionForwardAsync performs a convolution forward pass with dependency tracking
func (ctx *GPUTrainingContext) ConvolutionForwardAsync(input, weights, bias *Tensor, stride, padding int) (*Tensor, error) {
	// Use MPS convolution operation
	return Conv2DMPS(input, weights, bias, stride, stride, padding, padding, padding, padding)
}

// TrainingStepAsync performs a complete training step with batched operations
func (ctx *GPUTrainingContext) TrainingStepAsync(forward, backward []OperationDesc) error {
	// Execute forward pass
	_, err := ctx.BatchOperationsAsync(forward)
	if err != nil {
		return fmt.Errorf("forward pass failed: %v", err)
	}
	
	// Execute backward pass
	_, err = ctx.BatchOperationsAsync(backward)
	if err != nil {
		return fmt.Errorf("backward pass failed: %v", err)
	}
	
	return nil
}

// OptimizedMatMulChain performs a chain of matrix multiplications with minimal memory transfers
func (ctx *GPUTrainingContext) OptimizedMatMulChain(tensors []*Tensor) (*Tensor, error) {
	if len(tensors) < 2 {
		return nil, fmt.Errorf("need at least 2 tensors for matrix multiplication chain")
	}
	
	// Create dependency chain for sequential matrix multiplications
	var operations []OperationDesc
	
	// First operation: multiply first two tensors
	operations = append(operations, OperationDesc{
		Type:   "MatMul",
		Inputs: []*Tensor{tensors[0], tensors[1]},
		Params: nil,
	})
	
	// Subsequent operations: multiply result with next tensor
	for i := 2; i < len(tensors); i++ {
		operations = append(operations, OperationDesc{
			Type:   "MatMul",
			Inputs: []*Tensor{nil, tensors[i]}, // nil will be replaced with previous result
			Params: nil,
		})
	}
	
	return ctx.graph.ExecuteSequence(operations)
}

// GetGPUStats returns GPU operation statistics
func (ctx *GPUTrainingContext) GetGPUStats() (queued, executed int64, pending int, batchEfficiency float64) {
	q, e, p := ctx.graph.GetStats()
	
	var efficiency float64
	if ctx.totalOps > 0 {
		efficiency = float64(ctx.batchedOps) / float64(ctx.totalOps) * 100
	}
	
	return q, e, p, efficiency
}

// SetBatchSize sets the operation batch size for GPU operations
func (ctx *GPUTrainingContext) SetBatchSize(size int) {
	ctx.batchMutex.Lock()
	defer ctx.batchMutex.Unlock()
	
	ctx.batchSize = size
}

// Shutdown gracefully shuts down the training context
func (ctx *GPUTrainingContext) Shutdown() {
	// Flush any remaining batched operations
	ctx.FlushBatch()
	
	// Shutdown the computation graph
	ctx.graph.Shutdown()
}

// Global training context instance for easy access
var globalGPUTrainingContext *GPUTrainingContext
var gpuContextOnce sync.Once

// GetGlobalGPUTrainingContext returns the global GPU training context
func GetGlobalGPUTrainingContext() (*GPUTrainingContext, error) {
	var err error
	gpuContextOnce.Do(func() {
		globalGPUTrainingContext, err = NewGPUTrainingContext()
	})
	
	if err != nil {
		return nil, err
	}
	
	return globalGPUTrainingContext, nil
}

// Helper function for creating operation descriptors
func NewMatMulOp(a, b *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "MatMul", 
		Inputs: []*Tensor{a, b},
		Params: nil,
	}
}

func NewAddOp(a, b *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "Add",
		Inputs: []*Tensor{a, b},
		Params: nil,
	}
}

func NewReLUOp(input *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "ReLU",
		Inputs: []*Tensor{input},
		Params: nil,
	}
}

// IsFusedOperation checks if an operation type is a fused operation
func IsFusedOperation(opType string) bool {
	fusedOps := map[string]bool{
		"LinearForward":  true,
		"LinearReLU":     true,
		"LinearSigmoid":  true,
		"BatchMatMul":    true,
	}
	return fusedOps[opType]
}

// Helper functions for creating fused operation descriptors
func NewLinearForwardOp(input, weight, bias *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "LinearForward",
		Inputs: []*Tensor{input, weight, bias},
		Params: nil,
	}
}

func NewLinearReLUOp(input, weight, bias *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "LinearReLU",
		Inputs: []*Tensor{input, weight, bias},
		Params: nil,
	}
}

func NewLinearSigmoidOp(input, weight, bias *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "LinearSigmoid",
		Inputs: []*Tensor{input, weight, bias},
		Params: nil,
	}
}

func NewBatchMatMulOp(a, b *Tensor) OperationDesc {
	return OperationDesc{
		Type:   "BatchMatMul",
		Inputs: []*Tensor{a, b},
		Params: nil,
	}
}