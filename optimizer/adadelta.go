package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// AdaDeltaOptimizerState represents GPU-resident AdaDelta optimizer state
type AdaDeltaOptimizerState struct {
	// Configuration
	config AdaDeltaConfig
	
	// GPU-resident state buffers
	squaredGradAvgBuffers []unsafe.Pointer // E[g^2]_t - Accumulated squared gradient averages
	squaredUpdateAvgBuffers []unsafe.Pointer // E[Δx^2]_t - Accumulated squared update averages
	WeightBuffers         []unsafe.Pointer // Current weight tensors
	
	// Step tracking
	currentStep uint64
	
	// Buffer management
	memoryManager *memory.MemoryManager
	device        unsafe.Pointer
	bufferSizes   []int
	
	// Command buffer pooling
	commandPool unsafe.Pointer
	usePooling  bool
}

// AdaDeltaConfig holds configuration for AdaDelta optimizer
type AdaDeltaConfig struct {
	Rho         float32 // Decay rate for moving averages (typically 0.95)
	Epsilon     float32 // Small constant for numerical stability
	WeightDecay float32 // L2 regularization strength
}

// DefaultAdaDeltaConfig returns default AdaDelta optimizer configuration
func DefaultAdaDeltaConfig() AdaDeltaConfig {
	return AdaDeltaConfig{
		Rho:         0.95,
		Epsilon:     1e-6,
		WeightDecay: 0.0,
	}
}

// NewAdaDeltaOptimizer creates a new GPU-resident AdaDelta optimizer
func NewAdaDeltaOptimizer(
	config AdaDeltaConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*AdaDeltaOptimizerState, error) {
	if memoryManager == nil {
		return nil, fmt.Errorf("memory manager cannot be nil")
	}
	
	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}
	
	if len(weightShapes) == 0 {
		return nil, fmt.Errorf("no weight shapes provided")
	}
	
	if config.Rho <= 0 || config.Rho >= 1 {
		return nil, fmt.Errorf("rho must be in range (0, 1), got %f", config.Rho)
	}
	
	numWeights := len(weightShapes)
	
	adadelta := &AdaDeltaOptimizerState{
		config:                  config,
		squaredGradAvgBuffers:   make([]unsafe.Pointer, numWeights),
		squaredUpdateAvgBuffers: make([]unsafe.Pointer, numWeights),
		WeightBuffers:           make([]unsafe.Pointer, numWeights),
		currentStep:             0,
		memoryManager:           memoryManager,
		device:                  device,
		bufferSizes:             make([]int, numWeights),
	}
	
	// Calculate buffer sizes and allocate state buffers
	for i, shape := range weightShapes {
		adadelta.bufferSizes[i] = calculateTensorSize(shape) * 4 // 4 bytes per float32
		
		// Allocate squared gradient average buffer (E[g^2])
		squaredGradAvgBuffer := adadelta.memoryManager.AllocateBuffer(adadelta.bufferSizes[i])
		if squaredGradAvgBuffer == nil {
			adadelta.cleanup()
			return nil, fmt.Errorf("failed to allocate squared gradient average buffer for weight %d", i)
		}
		adadelta.squaredGradAvgBuffers[i] = squaredGradAvgBuffer
		
		// Allocate squared update average buffer (E[Δx^2])
		squaredUpdateAvgBuffer := adadelta.memoryManager.AllocateBuffer(adadelta.bufferSizes[i])
		if squaredUpdateAvgBuffer == nil {
			adadelta.cleanup()
			return nil, fmt.Errorf("failed to allocate squared update average buffer for weight %d", i)
		}
		adadelta.squaredUpdateAvgBuffers[i] = squaredUpdateAvgBuffer
		
		// Initialize both to zero
		if err := cgo_bridge.ZeroMetalBuffer(adadelta.device, squaredGradAvgBuffer, adadelta.bufferSizes[i]); err != nil {
			adadelta.cleanup()
			return nil, fmt.Errorf("failed to zero squared gradient average buffer: %v", err)
		}
		if err := cgo_bridge.ZeroMetalBuffer(adadelta.device, squaredUpdateAvgBuffer, adadelta.bufferSizes[i]); err != nil {
			adadelta.cleanup()
			return nil, fmt.Errorf("failed to zero squared update average buffer: %v", err)
		}
	}
	
	return adadelta, nil
}

// cleanup releases all allocated buffers
func (adadelta *AdaDeltaOptimizerState) cleanup() {
	for i := range adadelta.squaredGradAvgBuffers {
		if adadelta.squaredGradAvgBuffers[i] != nil {
			adadelta.memoryManager.ReleaseBuffer(adadelta.squaredGradAvgBuffers[i])
		}
		if adadelta.squaredUpdateAvgBuffers[i] != nil {
			adadelta.memoryManager.ReleaseBuffer(adadelta.squaredUpdateAvgBuffers[i])
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
func (adadelta *AdaDeltaOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(adadelta.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(adadelta.WeightBuffers), len(weightBuffers))
	}
	
	copy(adadelta.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single AdaDelta optimization step
func (adadelta *AdaDeltaOptimizerState) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(adadelta.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(adadelta.WeightBuffers))
	}
	
	adadelta.currentStep++
	
	// Execute AdaDelta step using CGO bridge
	var err error
	if adadelta.usePooling && adadelta.commandPool != nil {
		err = cgo_bridge.ExecuteAdaDeltaStepMPSGraphPooled(
			adadelta.device,
			adadelta.WeightBuffers,
			gradientBuffers,
			adadelta.squaredGradAvgBuffers,
			adadelta.squaredUpdateAvgBuffers,
			len(adadelta.WeightBuffers),
			adadelta.bufferSizes,
			adadelta.config.Rho,
			adadelta.config.Epsilon,
			adadelta.config.WeightDecay,
			adadelta.commandPool,
		)
	} else {
		err = cgo_bridge.ExecuteAdaDeltaStepMPSGraph(
			adadelta.device,
			adadelta.WeightBuffers,
			gradientBuffers,
			adadelta.squaredGradAvgBuffers,
			adadelta.squaredUpdateAvgBuffers,
			len(adadelta.WeightBuffers),
			adadelta.bufferSizes,
			adadelta.config.Rho,
			adadelta.config.Epsilon,
			adadelta.config.WeightDecay,
		)
	}
	
	if err != nil {
		return fmt.Errorf("AdaDelta step failed: %v", err)
	}
	
	// Log progress periodically
	if adadelta.currentStep%10 == 0 {
		fmt.Printf("AdaDelta step %d completed, rho=%.3f\n", adadelta.currentStep, adadelta.config.Rho)
	}
	
	return nil
}

// SetCommandPool sets the command buffer pool for Metal operations
func (adadelta *AdaDeltaOptimizerState) SetCommandPool(pool unsafe.Pointer) {
	adadelta.commandPool = pool
	adadelta.usePooling = (pool != nil)
}

// GetStep returns the current optimization step count
func (adadelta *AdaDeltaOptimizerState) GetStep() uint64 {
	return adadelta.currentStep
}

// GetStats returns optimizer statistics
func (adadelta *AdaDeltaOptimizerState) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"step":        adadelta.currentStep,
		"rho":         adadelta.config.Rho,
		"epsilon":     adadelta.config.Epsilon,
		"weight_decay": adadelta.config.WeightDecay,
	}
}

// Cleanup releases all GPU buffers
func (adadelta *AdaDeltaOptimizerState) Cleanup() {
	adadelta.cleanup()
}

// UpdateLearningRate is not used in AdaDelta (it adapts automatically)
func (adadelta *AdaDeltaOptimizerState) UpdateLearningRate(newLR float32) error {
	return fmt.Errorf("AdaDelta does not use a fixed learning rate; it adapts automatically based on parameter updates")
}