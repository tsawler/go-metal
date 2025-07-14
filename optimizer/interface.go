package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/checkpoints"
)

// Optimizer defines the common interface for all optimizers
// This interface enables state save/restore for checkpoint functionality
// while maintaining GPU-resident state and minimizing CGO calls
type Optimizer interface {
	// Step performs a single optimization step
	// gradientBuffers must match the number of weight buffers
	Step(gradientBuffers []unsafe.Pointer) error

	// GetState extracts optimizer state for checkpointing
	// This transfers GPU state to CPU only when needed for saving
	// The implementation should batch reads to minimize CGO calls
	GetState() (*OptimizerState, error)

	// LoadState restores optimizer state from checkpoint
	// This transfers CPU state back to GPU buffers
	// The implementation should batch writes to minimize CGO calls
	LoadState(state *OptimizerState) error

	// GetStepCount returns the current optimization step number
	GetStepCount() uint64

	// UpdateLearningRate updates the learning rate
	UpdateLearningRate(lr float32)

	// SetWeightBuffers sets the current weight buffer pointers
	// Must be called before each optimization step
	SetWeightBuffers(weightBuffers []unsafe.Pointer) error

	// SetCommandPool enables command buffer pooling for Metal operations
	SetCommandPool(commandPool unsafe.Pointer)

	// Cleanup releases all GPU resources
	Cleanup()
}

// OptimizerState represents the complete state of an optimizer
// Compatible with checkpoints.OptimizerState for serialization
type OptimizerState struct {
	Type       string                          `json:"type"`       // "Adam", "SGD", etc.
	Parameters map[string]interface{}          `json:"parameters"` // Hyperparameters
	StateData  []checkpoints.OptimizerTensor   `json:"state_data"` // GPU state tensors
}

// Common helper functions for state extraction

// extractBufferIndex extracts the buffer index from state tensor names like "momentum_0", "variance_1", "squared_grad_avg_0"
func extractBufferIndex(name string) int {
	var idx int
	// Find the last underscore in the name
	lastUnderscoreIdx := -1
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '_' {
			lastUnderscoreIdx = i
			break
		}
	}
	
	if lastUnderscoreIdx == -1 {
		return -1
	}
	
	// Try to parse the number after the last underscore
	if n, err := fmt.Sscanf(name[lastUnderscoreIdx+1:], "%d", &idx); n == 1 && err == nil {
		return idx
	}
	return -1
}

// validateStateType ensures the state type matches the optimizer
func validateStateType(optimizerType string, state *OptimizerState) error {
	if state.Type != optimizerType {
		return fmt.Errorf("state type mismatch: expected %s, got %s", optimizerType, state.Type)
	}
	return nil
}