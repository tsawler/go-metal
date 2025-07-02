package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// AdamOptimizerState represents GPU-resident Adam optimizer state
type AdamOptimizerState struct {
	// Hyperparameters
	LearningRate float32
	Beta1        float32 // Momentum decay (typically 0.9)
	Beta2        float32 // Variance decay (typically 0.999)
	Epsilon      float32 // Small constant to prevent division by zero (typically 1e-8)
	WeightDecay  float32 // L2 regularization coefficient

	// GPU-resident state buffers
	MomentumBuffers []unsafe.Pointer // First moment (momentum) for each weight tensor
	VarianceBuffers []unsafe.Pointer // Second moment (variance) for each weight tensor
	WeightBuffers   []unsafe.Pointer // Current weight tensors

	// Step tracking for bias correction
	StepCount uint64

	// Buffer management
	memoryManager *memory.MemoryManager
	device        unsafe.Pointer

	// Buffer sizes for proper cleanup
	bufferSizes []int
	
	// RESOURCE LEAK FIX: Command buffer pooling
	commandPool unsafe.Pointer  // Optional command buffer pool for Metal operations
	usePooling  bool           // Whether to use command buffer pooling
}

// AdamConfig holds configuration for Adam optimizer
type AdamConfig struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	WeightDecay  float32
}

// DefaultAdamConfig returns default Adam optimizer configuration
func DefaultAdamConfig() AdamConfig {
	return AdamConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
	}
}

// NewAdamOptimizer creates a new GPU-resident Adam optimizer
func NewAdamOptimizer(
	config AdamConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*AdamOptimizerState, error) {
	if memoryManager == nil {
		return nil, fmt.Errorf("memory manager cannot be nil")
	}

	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}

	if len(weightShapes) == 0 {
		return nil, fmt.Errorf("no weight shapes provided")
	}

	numWeights := len(weightShapes)

	adam := &AdamOptimizerState{
		LearningRate:    config.LearningRate,
		Beta1:           config.Beta1,
		Beta2:           config.Beta2,
		Epsilon:         config.Epsilon,
		WeightDecay:     config.WeightDecay,
		MomentumBuffers: make([]unsafe.Pointer, numWeights),
		VarianceBuffers: make([]unsafe.Pointer, numWeights),
		WeightBuffers:   make([]unsafe.Pointer, numWeights),
		StepCount:       0,
		memoryManager:   memoryManager,
		device:          device,
		bufferSizes:     make([]int, numWeights),
	}

	// Allocate GPU buffers for momentum and variance
	for i, shape := range weightShapes {
		// Calculate buffer size (assume float32)
		size := calculateTensorSize(shape) * 4 // 4 bytes per float32
		adam.bufferSizes[i] = size

		// Allocate momentum buffer
		momentumBuffer := adam.memoryManager.AllocateBuffer(size)
		if momentumBuffer == nil {
			adam.cleanup(i) // Cleanup previously allocated buffers
			return nil, fmt.Errorf("failed to allocate momentum buffer for weight %d", i)
		}
		adam.MomentumBuffers[i] = momentumBuffer

		// Allocate variance buffer
		varianceBuffer := adam.memoryManager.AllocateBuffer(size)
		if varianceBuffer == nil {
			adam.memoryManager.ReleaseBuffer(momentumBuffer)
			adam.cleanup(i) // Cleanup previously allocated buffers
			return nil, fmt.Errorf("failed to allocate variance buffer for weight %d", i)
		}
		adam.VarianceBuffers[i] = varianceBuffer

		// Initialize buffers to zero (momentum and variance start at 0)
		err := cgo_bridge.ZeroMetalBuffer(adam.device, momentumBuffer, size)
		if err != nil {
			adam.memoryManager.ReleaseBuffer(momentumBuffer)
			adam.memoryManager.ReleaseBuffer(varianceBuffer)
			adam.cleanup(i)
			return nil, fmt.Errorf("failed to zero momentum buffer for weight %d: %v", i, err)
		}

		err = cgo_bridge.ZeroMetalBuffer(adam.device, varianceBuffer, size)
		if err != nil {
			adam.memoryManager.ReleaseBuffer(momentumBuffer)
			adam.memoryManager.ReleaseBuffer(varianceBuffer)
			adam.cleanup(i)
			return nil, fmt.Errorf("failed to zero variance buffer for weight %d: %v", i, err)
		}
	}

	return adam, nil
}

// calculateTensorSize calculates the number of elements in a tensor
func calculateTensorSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

// cleanup releases previously allocated buffers in case of partial initialization failure
func (adam *AdamOptimizerState) cleanup(upToIndex int) {
	for i := 0; i < upToIndex; i++ {
		if adam.MomentumBuffers[i] != nil {
			adam.memoryManager.ReleaseBuffer(adam.MomentumBuffers[i])
			adam.MomentumBuffers[i] = nil
		}
		if adam.VarianceBuffers[i] != nil {
			adam.memoryManager.ReleaseBuffer(adam.VarianceBuffers[i])
			adam.VarianceBuffers[i] = nil
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
// This should be called before each optimization step
func (adam *AdamOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(adam.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(adam.WeightBuffers), len(weightBuffers))
	}

	copy(adam.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single Adam optimization step
// This will be implemented as a Metal compute kernel for maximum performance
func (adam *AdamOptimizerState) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(adam.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(adam.WeightBuffers))
	}

	adam.StepCount++

	// RESOURCE LEAK FIX: Use pooled version if command pooling is enabled
	var err error
	if adam.usePooling && adam.commandPool != nil {
		// Use pooled command buffers to prevent Metal resource accumulation
		err = cgo_bridge.ExecuteAdamStepMPSGraphPooled(
			adam.device,
			adam.WeightBuffers,
			gradientBuffers,
			adam.MomentumBuffers,
			adam.VarianceBuffers,
			adam.bufferSizes,
			adam.LearningRate,
			adam.Beta1,
			adam.Beta2,
			adam.Epsilon,
			adam.WeightDecay,
			int(adam.StepCount),
			adam.commandPool,
		)
	} else {
		// Fallback to non-pooled version
		err = cgo_bridge.ExecuteAdamStepMPSGraph(
			adam.device,
			adam.WeightBuffers,
			gradientBuffers,
			adam.MomentumBuffers,
			adam.VarianceBuffers,
			adam.bufferSizes,
			adam.LearningRate,
			adam.Beta1,
			adam.Beta2,
			adam.Epsilon,
			adam.WeightDecay,
			int(adam.StepCount),
		)
	}

	if err != nil {
		return fmt.Errorf("Adam step execution failed: %v", err)
	}

	return nil
}

// pow computes x^y for float32
func pow(x, y float32) float32 {
	if y == 0 {
		return 1.0
	}
	if y == 1 {
		return x
	}
	
	// Simple implementation for small integer powers
	result := float32(1.0)
	for i := 0; i < int(y); i++ {
		result *= x
	}
	return result
}

// UpdateLearningRate updates the learning rate (useful for learning rate scheduling)
func (adam *AdamOptimizerState) UpdateLearningRate(newLR float32) {
	adam.LearningRate = newLR
}

// SetCommandPool enables command buffer pooling for Metal operations
// RESOURCE LEAK FIX: Allows Adam optimizer to use pooled command buffers
func (adam *AdamOptimizerState) SetCommandPool(commandPool unsafe.Pointer) {
	adam.commandPool = commandPool
	adam.usePooling = (commandPool != nil)
}

// GetStep returns the current step count
func (adam *AdamOptimizerState) GetStep() uint64 {
	return adam.StepCount
}

// GetStats returns optimizer statistics
func (adam *AdamOptimizerState) GetStats() AdamStats {
	return AdamStats{
		StepCount:       adam.StepCount,
		LearningRate:    adam.LearningRate,
		Beta1:           adam.Beta1,
		Beta2:           adam.Beta2,
		Epsilon:         adam.Epsilon,
		WeightDecay:     adam.WeightDecay,
		NumParameters:   len(adam.WeightBuffers),
		TotalBufferSize: adam.getTotalBufferSize(),
	}
}

// AdamStats provides statistics about the Adam optimizer
type AdamStats struct {
	StepCount       uint64
	LearningRate    float32
	Beta1           float32
	Beta2           float32
	Epsilon         float32
	WeightDecay     float32
	NumParameters   int
	TotalBufferSize int
}

// getTotalBufferSize calculates total memory used by optimizer state
func (adam *AdamOptimizerState) getTotalBufferSize() int {
	total := 0
	for _, size := range adam.bufferSizes {
		total += size * 2 // momentum + variance buffers
	}
	return total
}

// Cleanup releases all GPU buffers
func (adam *AdamOptimizerState) Cleanup() {
	for i := range adam.MomentumBuffers {
		if adam.MomentumBuffers[i] != nil {
			adam.memoryManager.ReleaseBuffer(adam.MomentumBuffers[i])
			adam.MomentumBuffers[i] = nil
		}
		if adam.VarianceBuffers[i] != nil {
			adam.memoryManager.ReleaseBuffer(adam.VarianceBuffers[i])
			adam.VarianceBuffers[i] = nil
		}
	}
	
	// Clear slices
	adam.MomentumBuffers = nil
	adam.VarianceBuffers = nil
	adam.WeightBuffers = nil
	adam.bufferSizes = nil
}