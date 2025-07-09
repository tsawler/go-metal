package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
)

// RMSPropOptimizerState represents GPU-resident RMSProp optimizer state
type RMSPropOptimizerState struct {
	// Hyperparameters
	LearningRate float32
	Alpha        float32 // Smoothing constant (typically 0.99)
	Epsilon      float32 // Small constant to prevent division by zero (typically 1e-8)
	WeightDecay  float32 // L2 regularization coefficient
	Momentum     float32 // Momentum coefficient (typically 0.9, 0.0 for no momentum)
	Centered     bool    // Whether to use centered RMSProp (subtract mean of gradients)

	// GPU-resident state buffers
	SquaredGradAvgBuffers []unsafe.Pointer // Running average of squared gradients for each weight tensor
	MomentumBuffers       []unsafe.Pointer // Momentum buffers for each weight tensor (if momentum > 0)
	GradientAvgBuffers    []unsafe.Pointer // Running average of gradients for each weight tensor (if centered)
	WeightBuffers         []unsafe.Pointer // Current weight tensors

	// Step tracking
	StepCount uint64

	// Buffer management
	memoryManager *memory.MemoryManager
	device        unsafe.Pointer

	// Buffer sizes for proper cleanup
	bufferSizes []int

	// Command buffer pooling
	commandPool unsafe.Pointer // Optional command buffer pool for Metal operations
	usePooling  bool           // Whether to use command buffer pooling
}

// RMSPropConfig holds configuration for RMSProp optimizer
type RMSPropConfig struct {
	LearningRate float32
	Alpha        float32
	Epsilon      float32
	WeightDecay  float32
	Momentum     float32
	Centered     bool
}

// DefaultRMSPropConfig returns default RMSProp optimizer configuration
func DefaultRMSPropConfig() RMSPropConfig {
	return RMSPropConfig{
		LearningRate: 0.01,
		Alpha:        0.99,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.0,
		Centered:     false,
	}
}

// NewRMSPropOptimizer creates a new GPU-resident RMSProp optimizer
func NewRMSPropOptimizer(
	config RMSPropConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*RMSPropOptimizerState, error) {
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

	rmsprop := &RMSPropOptimizerState{
		LearningRate:          config.LearningRate,
		Alpha:                 config.Alpha,
		Epsilon:               config.Epsilon,
		WeightDecay:           config.WeightDecay,
		Momentum:              config.Momentum,
		Centered:              config.Centered,
		SquaredGradAvgBuffers: make([]unsafe.Pointer, numWeights),
		MomentumBuffers:       make([]unsafe.Pointer, numWeights),
		GradientAvgBuffers:    make([]unsafe.Pointer, numWeights),
		WeightBuffers:         make([]unsafe.Pointer, numWeights),
		StepCount:             0,
		memoryManager:         memoryManager,
		device:                device,
		bufferSizes:           make([]int, numWeights),
	}

	// Allocate GPU buffers for squared gradient averages (always needed)
	for i, shape := range weightShapes {
		// Calculate buffer size (assume float32)
		size := calculateTensorSize(shape) * 4 // 4 bytes per float32
		rmsprop.bufferSizes[i] = size

		// Allocate squared gradient average buffer
		squaredGradAvgBuffer := rmsprop.memoryManager.AllocateBuffer(size)
		if squaredGradAvgBuffer == nil {
			rmsprop.cleanup(i) // Cleanup previously allocated buffers
			return nil, fmt.Errorf("failed to allocate squared gradient average buffer for weight %d", i)
		}
		rmsprop.SquaredGradAvgBuffers[i] = squaredGradAvgBuffer

		// Allocate momentum buffer if momentum > 0
		if config.Momentum > 0.0 {
			momentumBuffer := rmsprop.memoryManager.AllocateBuffer(size)
			if momentumBuffer == nil {
				rmsprop.memoryManager.ReleaseBuffer(squaredGradAvgBuffer)
				rmsprop.cleanup(i)
				return nil, fmt.Errorf("failed to allocate momentum buffer for weight %d", i)
			}
			rmsprop.MomentumBuffers[i] = momentumBuffer
		}

		// Allocate gradient average buffer if centered
		if config.Centered {
			gradientAvgBuffer := rmsprop.memoryManager.AllocateBuffer(size)
			if gradientAvgBuffer == nil {
				rmsprop.memoryManager.ReleaseBuffer(squaredGradAvgBuffer)
				if rmsprop.MomentumBuffers[i] != nil {
					rmsprop.memoryManager.ReleaseBuffer(rmsprop.MomentumBuffers[i])
				}
				rmsprop.cleanup(i)
				return nil, fmt.Errorf("failed to allocate gradient average buffer for weight %d", i)
			}
			rmsprop.GradientAvgBuffers[i] = gradientAvgBuffer
		}

		// Initialize buffers to zero
		err := cgo_bridge.ZeroMetalBuffer(rmsprop.device, squaredGradAvgBuffer, size)
		if err != nil {
			rmsprop.memoryManager.ReleaseBuffer(squaredGradAvgBuffer)
			if rmsprop.MomentumBuffers[i] != nil {
				rmsprop.memoryManager.ReleaseBuffer(rmsprop.MomentumBuffers[i])
			}
			if rmsprop.GradientAvgBuffers[i] != nil {
				rmsprop.memoryManager.ReleaseBuffer(rmsprop.GradientAvgBuffers[i])
			}
			rmsprop.cleanup(i)
			return nil, fmt.Errorf("failed to zero squared gradient average buffer for weight %d: %v", i, err)
		}

		if rmsprop.MomentumBuffers[i] != nil {
			err = cgo_bridge.ZeroMetalBuffer(rmsprop.device, rmsprop.MomentumBuffers[i], size)
			if err != nil {
				rmsprop.memoryManager.ReleaseBuffer(squaredGradAvgBuffer)
				rmsprop.memoryManager.ReleaseBuffer(rmsprop.MomentumBuffers[i])
				if rmsprop.GradientAvgBuffers[i] != nil {
					rmsprop.memoryManager.ReleaseBuffer(rmsprop.GradientAvgBuffers[i])
				}
				rmsprop.cleanup(i)
				return nil, fmt.Errorf("failed to zero momentum buffer for weight %d: %v", i, err)
			}
		}

		if rmsprop.GradientAvgBuffers[i] != nil {
			err = cgo_bridge.ZeroMetalBuffer(rmsprop.device, rmsprop.GradientAvgBuffers[i], size)
			if err != nil {
				rmsprop.memoryManager.ReleaseBuffer(squaredGradAvgBuffer)
				if rmsprop.MomentumBuffers[i] != nil {
					rmsprop.memoryManager.ReleaseBuffer(rmsprop.MomentumBuffers[i])
				}
				rmsprop.memoryManager.ReleaseBuffer(rmsprop.GradientAvgBuffers[i])
				rmsprop.cleanup(i)
				return nil, fmt.Errorf("failed to zero gradient average buffer for weight %d: %v", i, err)
			}
		}
	}

	return rmsprop, nil
}

// cleanup releases previously allocated buffers in case of partial initialization failure
func (rmsprop *RMSPropOptimizerState) cleanup(upToIndex int) {
	for i := 0; i < upToIndex; i++ {
		if rmsprop.SquaredGradAvgBuffers[i] != nil {
			rmsprop.memoryManager.ReleaseBuffer(rmsprop.SquaredGradAvgBuffers[i])
			rmsprop.SquaredGradAvgBuffers[i] = nil
		}
		if rmsprop.MomentumBuffers[i] != nil {
			rmsprop.memoryManager.ReleaseBuffer(rmsprop.MomentumBuffers[i])
			rmsprop.MomentumBuffers[i] = nil
		}
		if rmsprop.GradientAvgBuffers[i] != nil {
			rmsprop.memoryManager.ReleaseBuffer(rmsprop.GradientAvgBuffers[i])
			rmsprop.GradientAvgBuffers[i] = nil
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
// This should be called before each optimization step
func (rmsprop *RMSPropOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(rmsprop.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(rmsprop.WeightBuffers), len(weightBuffers))
	}

	copy(rmsprop.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single RMSProp optimization step
func (rmsprop *RMSPropOptimizerState) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(rmsprop.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(rmsprop.WeightBuffers))
	}

	rmsprop.StepCount++

	var err error
	err = cgo_bridge.ExecuteRMSPropStepMPSGraph(
		rmsprop.device,
		rmsprop.WeightBuffers,
		gradientBuffers,
		rmsprop.SquaredGradAvgBuffers,
		rmsprop.MomentumBuffers,
		rmsprop.GradientAvgBuffers,
		rmsprop.bufferSizes,
		rmsprop.LearningRate,
		rmsprop.Alpha,
		rmsprop.Epsilon,
		rmsprop.WeightDecay,
		rmsprop.Momentum,
		rmsprop.Centered,
		int(rmsprop.StepCount),
	)

	if err != nil {
		return fmt.Errorf("RMSProp step execution failed: %v", err)
	}

	return nil
}

// UpdateLearningRate updates the learning rate (useful for learning rate scheduling)
func (rmsprop *RMSPropOptimizerState) UpdateLearningRate(newLR float32) {
	rmsprop.LearningRate = newLR
}

// SetCommandPool enables command buffer pooling for Metal operations
func (rmsprop *RMSPropOptimizerState) SetCommandPool(commandPool unsafe.Pointer) {
	rmsprop.commandPool = commandPool
	rmsprop.usePooling = (commandPool != nil)
}

// GetStep returns the current step count
func (rmsprop *RMSPropOptimizerState) GetStep() uint64 {
	return rmsprop.StepCount
}

// GetStats returns optimizer statistics
func (rmsprop *RMSPropOptimizerState) GetStats() RMSPropStats {
	return RMSPropStats{
		StepCount:       rmsprop.StepCount,
		LearningRate:    rmsprop.LearningRate,
		Alpha:           rmsprop.Alpha,
		Epsilon:         rmsprop.Epsilon,
		WeightDecay:     rmsprop.WeightDecay,
		Momentum:        rmsprop.Momentum,
		Centered:        rmsprop.Centered,
		NumParameters:   len(rmsprop.WeightBuffers),
		TotalBufferSize: rmsprop.getTotalBufferSize(),
	}
}

// RMSPropStats provides statistics about the RMSProp optimizer
type RMSPropStats struct {
	StepCount       uint64
	LearningRate    float32
	Alpha           float32
	Epsilon         float32
	WeightDecay     float32
	Momentum        float32
	Centered        bool
	NumParameters   int
	TotalBufferSize int
}

// getTotalBufferSize calculates total memory used by optimizer state
func (rmsprop *RMSPropOptimizerState) getTotalBufferSize() int {
	total := 0
	for _, size := range rmsprop.bufferSizes {
		total += size // squared gradient average buffer (always present)
		if rmsprop.Momentum > 0.0 {
			total += size // momentum buffer
		}
		if rmsprop.Centered {
			total += size // gradient average buffer
		}
	}
	return total
}

// Cleanup releases all GPU buffers
func (rmsprop *RMSPropOptimizerState) Cleanup() {
	for i := range rmsprop.SquaredGradAvgBuffers {
		if rmsprop.SquaredGradAvgBuffers[i] != nil {
			rmsprop.memoryManager.ReleaseBuffer(rmsprop.SquaredGradAvgBuffers[i])
			rmsprop.SquaredGradAvgBuffers[i] = nil
		}
		if rmsprop.MomentumBuffers[i] != nil {
			rmsprop.memoryManager.ReleaseBuffer(rmsprop.MomentumBuffers[i])
			rmsprop.MomentumBuffers[i] = nil
		}
		if rmsprop.GradientAvgBuffers[i] != nil {
			rmsprop.memoryManager.ReleaseBuffer(rmsprop.GradientAvgBuffers[i])
			rmsprop.GradientAvgBuffers[i] = nil
		}
	}

	// Clear slices
	rmsprop.SquaredGradAvgBuffers = nil
	rmsprop.MomentumBuffers = nil
	rmsprop.GradientAvgBuffers = nil
	rmsprop.WeightBuffers = nil
	rmsprop.bufferSizes = nil
}