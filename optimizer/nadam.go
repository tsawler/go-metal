package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// NadamOptimizerState represents GPU-resident Nadam optimizer state
// Nadam combines Adam's adaptive learning rates with Nesterov momentum
type NadamOptimizerState struct {
	config NadamConfig

	// GPU-resident state buffers
	momentumBuffers []unsafe.Pointer // First moment (momentum) for each weight tensor
	varianceBuffers []unsafe.Pointer // Second moment (variance) for each weight tensor
	WeightBuffers   []unsafe.Pointer // Current weight tensors

	// Step tracking for bias correction
	currentStep uint64

	// Buffer management
	memoryManager *memory.MemoryManager
	device        unsafe.Pointer

	// Buffer sizes for proper cleanup
	bufferSizes []int

	// Command buffer pooling
	commandPool unsafe.Pointer // Optional command buffer pool for Metal operations
	usePooling  bool           // Whether to use command buffer pooling
}

// NadamConfig holds configuration for Nadam optimizer
type NadamConfig struct {
	LearningRate float32 // Base learning rate (typically 0.002)
	Beta1        float32 // Exponential decay rate for first moment estimates (typically 0.9)
	Beta2        float32 // Exponential decay rate for second moment estimates (typically 0.999)
	Epsilon      float32 // Small constant for numerical stability (typically 1e-8)
	WeightDecay  float32 // L2 regularization coefficient (typically 0.0)
}

// DefaultNadamConfig returns default Nadam optimizer configuration
func DefaultNadamConfig() NadamConfig {
	return NadamConfig{
		LearningRate: 0.002, // Nadam typically uses slightly higher LR than Adam
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
	}
}

// NewNadamOptimizer creates a new GPU-resident Nadam optimizer
func NewNadamOptimizer(
	config NadamConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*NadamOptimizerState, error) {
	if memoryManager == nil {
		return nil, fmt.Errorf("memory manager cannot be nil")
	}

	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}

	if len(weightShapes) == 0 {
		return nil, fmt.Errorf("no weight shapes provided")
	}

	// Validate configuration
	if config.LearningRate <= 0 {
		return nil, fmt.Errorf("learning rate must be positive, got %f", config.LearningRate)
	}
	if config.Beta1 < 0 || config.Beta1 >= 1 {
		return nil, fmt.Errorf("beta1 must be in [0, 1), got %f", config.Beta1)
	}
	if config.Beta2 < 0 || config.Beta2 >= 1 {
		return nil, fmt.Errorf("beta2 must be in [0, 1), got %f", config.Beta2)
	}
	if config.Epsilon <= 0 {
		return nil, fmt.Errorf("epsilon must be positive, got %e", config.Epsilon)
	}
	if config.WeightDecay < 0 {
		return nil, fmt.Errorf("weight decay must be non-negative, got %f", config.WeightDecay)
	}

	numWeights := len(weightShapes)

	nadam := &NadamOptimizerState{
		config:          config,
		momentumBuffers: make([]unsafe.Pointer, numWeights),
		varianceBuffers: make([]unsafe.Pointer, numWeights),
		WeightBuffers:   make([]unsafe.Pointer, numWeights),
		currentStep:     0,
		memoryManager:   memoryManager,
		device:          device,
		bufferSizes:     make([]int, numWeights),
	}

	// Allocate GPU buffers for momentum and variance
	for i, shape := range weightShapes {
		// Calculate buffer size (assume float32)
		size := calculateTensorSize(shape) * 4 // 4 bytes per float32
		nadam.bufferSizes[i] = size

		// Allocate momentum buffer
		momentumBuffer := nadam.memoryManager.AllocateBuffer(size)
		if momentumBuffer == nil {
			nadam.cleanup(i) // Cleanup previously allocated buffers
			return nil, fmt.Errorf("failed to allocate momentum buffer for weight %d", i)
		}
		nadam.momentumBuffers[i] = momentumBuffer

		// Allocate variance buffer
		varianceBuffer := nadam.memoryManager.AllocateBuffer(size)
		if varianceBuffer == nil {
			nadam.memoryManager.ReleaseBuffer(momentumBuffer)
			nadam.cleanup(i) // Cleanup previously allocated buffers
			return nil, fmt.Errorf("failed to allocate variance buffer for weight %d", i)
		}
		nadam.varianceBuffers[i] = varianceBuffer

		// Initialize buffers to zero (momentum and variance start at 0)
		err := cgo_bridge.ZeroMetalBuffer(nadam.device, momentumBuffer, size)
		if err != nil {
			nadam.memoryManager.ReleaseBuffer(momentumBuffer)
			nadam.memoryManager.ReleaseBuffer(varianceBuffer)
			nadam.cleanup(i)
			return nil, fmt.Errorf("failed to zero momentum buffer for weight %d: %v", i, err)
		}

		err = cgo_bridge.ZeroMetalBuffer(nadam.device, varianceBuffer, size)
		if err != nil {
			nadam.memoryManager.ReleaseBuffer(momentumBuffer)
			nadam.memoryManager.ReleaseBuffer(varianceBuffer)
			nadam.cleanup(i)
			return nil, fmt.Errorf("failed to zero variance buffer for weight %d: %v", i, err)
		}
	}

	return nadam, nil
}

// cleanup releases previously allocated buffers in case of partial initialization failure
func (nadam *NadamOptimizerState) cleanup(upToIndex int) {
	for i := 0; i < upToIndex; i++ {
		if nadam.momentumBuffers[i] != nil {
			nadam.memoryManager.ReleaseBuffer(nadam.momentumBuffers[i])
			nadam.momentumBuffers[i] = nil
		}
		if nadam.varianceBuffers[i] != nil {
			nadam.memoryManager.ReleaseBuffer(nadam.varianceBuffers[i])
			nadam.varianceBuffers[i] = nil
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
// This should be called before each optimization step
func (nadam *NadamOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(nadam.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(nadam.WeightBuffers), len(weightBuffers))
	}

	copy(nadam.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single Nadam optimization step
// Nadam combines Adam's adaptive learning rates with Nesterov momentum
func (nadam *NadamOptimizerState) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(nadam.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(nadam.WeightBuffers))
	}

	nadam.currentStep++

	// Execute Nadam step using MPSGraph
	err := cgo_bridge.ExecuteNadamStepMPSGraph(
		nadam.device,
		nadam.WeightBuffers,
		gradientBuffers,
		nadam.momentumBuffers,
		nadam.varianceBuffers,
		nadam.bufferSizes,
		nadam.config.LearningRate,
		nadam.config.Beta1,
		nadam.config.Beta2,
		nadam.config.Epsilon,
		nadam.config.WeightDecay,
		int(nadam.currentStep),
	)

	if err != nil {
		return fmt.Errorf("Nadam step execution failed: %v", err)
	}

	return nil
}

// UpdateLearningRate updates the learning rate (useful for learning rate scheduling)
func (nadam *NadamOptimizerState) UpdateLearningRate(newLR float32) error {
	if newLR <= 0 {
		return fmt.Errorf("learning rate must be positive, got %f", newLR)
	}
	nadam.config.LearningRate = newLR
	return nil
}

// SetCommandPool enables command buffer pooling for Metal operations
func (nadam *NadamOptimizerState) SetCommandPool(commandPool unsafe.Pointer) {
	nadam.commandPool = commandPool
	nadam.usePooling = (commandPool != nil)
}

// GetStep returns the current step count
func (nadam *NadamOptimizerState) GetStep() uint64 {
	return nadam.currentStep
}

// GetStats returns optimizer statistics as a map for generic access
func (nadam *NadamOptimizerState) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"step":          nadam.currentStep,
		"learning_rate": nadam.config.LearningRate,
		"beta1":         nadam.config.Beta1,
		"beta2":         nadam.config.Beta2,
		"epsilon":       nadam.config.Epsilon,
		"weight_decay":  nadam.config.WeightDecay,
	}
}

// Cleanup releases all GPU buffers
func (nadam *NadamOptimizerState) Cleanup() {
	for i := range nadam.momentumBuffers {
		if nadam.momentumBuffers[i] != nil {
			nadam.memoryManager.ReleaseBuffer(nadam.momentumBuffers[i])
			nadam.momentumBuffers[i] = nil
		}
		if nadam.varianceBuffers[i] != nil {
			nadam.memoryManager.ReleaseBuffer(nadam.varianceBuffers[i])
			nadam.varianceBuffers[i] = nil
		}
	}

	// Clear slices
	nadam.momentumBuffers = nil
	nadam.varianceBuffers = nil
	nadam.WeightBuffers = nil
	nadam.bufferSizes = nil
}