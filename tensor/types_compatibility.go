package tensor

// Compatibility types for files that reference the old compute engine types
// These are minimal definitions to allow compilation while using MPSGraph operations

// OperationDesc describes an operation to be added to the graph
type OperationDesc struct {
	Type   string
	Inputs []*Tensor
	Params map[string]interface{}
}

// GPUComputationGraph stub for compatibility
// This is a minimal stub to allow compilation of gpu_training_ops.go
// The actual implementation should use MPSGraph operations directly
type GPUComputationGraph struct {
	// Empty struct for compatibility
}

// NewGPUComputationGraph creates a stub computation graph
func NewGPUComputationGraph() (*GPUComputationGraph, error) {
	return &GPUComputationGraph{}, nil
}

// AddOperation is a stub for compatibility
func (g *GPUComputationGraph) AddOperation(opType string, inputs []*Tensor, dependencies []interface{}, params map[string]interface{}) (interface{}, error) {
	// This should not be used - instead use MPSGraph operations directly
	return nil, nil
}

// WaitForOperation is a stub for compatibility
func (g *GPUComputationGraph) WaitForOperation(opID interface{}) (*Tensor, error) {
	// This should not be used - instead use MPSGraph operations directly
	return nil, nil
}

// ExecuteSequence is a stub for compatibility
func (g *GPUComputationGraph) ExecuteSequence(operations []OperationDesc) (*Tensor, error) {
	// This should not be used - instead use MPSGraph operations directly
	return nil, nil
}

// GetStats is a stub for compatibility
func (g *GPUComputationGraph) GetStats() (queued, executed int64, pending int) {
	return 0, 0, 0
}

// Shutdown is a stub for compatibility
func (g *GPUComputationGraph) Shutdown() {
	// No-op for compatibility
}