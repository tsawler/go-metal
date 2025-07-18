package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/memory"
)

// SGDOptimizerState represents GPU-resident SGD optimizer state
type SGDOptimizerState struct {
	// Hyperparameters
	LearningRate float32
	Momentum     float32 // Momentum coefficient (0 for vanilla SGD)
	WeightDecay  float32 // L2 regularization coefficient
	Nesterov     bool    // Whether to use Nesterov momentum

	// GPU-resident state buffers
	MomentumBuffers []unsafe.Pointer // Momentum buffers (only if momentum > 0)
	WeightBuffers   []unsafe.Pointer // Current weight tensors

	// Step tracking
	StepCount uint64

	// Buffer management
	memoryManager *memory.MemoryManager
	device        unsafe.Pointer

	// Buffer sizes for proper cleanup
	bufferSizes []int

	// Command buffer pooling
	commandPool unsafe.Pointer
	usePooling  bool
}

// SGDConfig holds configuration for SGD optimizer
type SGDConfig struct {
	LearningRate float32
	Momentum     float32
	WeightDecay  float32
	Nesterov     bool
}

// DefaultSGDConfig returns default SGD optimizer configuration
func DefaultSGDConfig() SGDConfig {
	return SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.0,
		WeightDecay:  0.0,
		Nesterov:     false,
	}
}

// NewSGDOptimizer creates a new GPU-resident SGD optimizer
func NewSGDOptimizer(
	config SGDConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*SGDOptimizerState, error) {
	if memoryManager == nil {
		return nil, fmt.Errorf("memory manager cannot be nil")
	}

	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}

	if len(weightShapes) == 0 {
		return nil, fmt.Errorf("no weight shapes provided")
	}

	// Validate configuration parameters
	if config.LearningRate < 0 {
		return nil, fmt.Errorf("learning rate cannot be negative: %f", config.LearningRate)
	}
	if config.Momentum < 0 {
		return nil, fmt.Errorf("momentum cannot be negative: %f", config.Momentum)
	}
	if config.Momentum > 1.0 {
		return nil, fmt.Errorf("momentum cannot be greater than 1.0: %f", config.Momentum)
	}
	if config.WeightDecay < 0 {
		return nil, fmt.Errorf("weight decay cannot be negative: %f", config.WeightDecay)
	}

	numWeights := len(weightShapes)

	sgd := &SGDOptimizerState{
		LearningRate:    config.LearningRate,
		Momentum:        config.Momentum,
		WeightDecay:     config.WeightDecay,
		Nesterov:        config.Nesterov,
		WeightBuffers:   make([]unsafe.Pointer, numWeights),
		StepCount:       0,
		memoryManager:   memoryManager,
		device:          device,
		bufferSizes:     make([]int, numWeights),
	}

	// Only allocate momentum buffers if momentum > 0
	if config.Momentum > 0 {
		sgd.MomentumBuffers = make([]unsafe.Pointer, numWeights)

		// Allocate GPU buffers for momentum
		for i, shape := range weightShapes {
			size := calculateTensorSize(shape) * 4 // 4 bytes per float32
			sgd.bufferSizes[i] = size

			momentumBuffer := sgd.memoryManager.AllocateBuffer(size)
			if momentumBuffer == nil {
				sgd.cleanup(i)
				return nil, fmt.Errorf("failed to allocate momentum buffer for weight %d", i)
			}
			sgd.MomentumBuffers[i] = momentumBuffer

			// Initialize to zero
			err := cgo_bridge.ZeroMetalBuffer(sgd.device, momentumBuffer, size)
			if err != nil {
				sgd.memoryManager.ReleaseBuffer(momentumBuffer)
				sgd.cleanup(i)
				return nil, fmt.Errorf("failed to zero momentum buffer for weight %d: %v", i, err)
			}
		}
	} else {
		// For vanilla SGD, just store buffer sizes
		for i, shape := range weightShapes {
			sgd.bufferSizes[i] = calculateTensorSize(shape) * 4
		}
	}

	return sgd, nil
}

// cleanup releases previously allocated buffers
func (sgd *SGDOptimizerState) cleanup(upToIndex int) {
	if sgd.MomentumBuffers != nil {
		for i := 0; i < upToIndex; i++ {
			if sgd.MomentumBuffers[i] != nil {
				sgd.memoryManager.ReleaseBuffer(sgd.MomentumBuffers[i])
				sgd.MomentumBuffers[i] = nil
			}
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
func (sgd *SGDOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(sgd.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(sgd.WeightBuffers), len(weightBuffers))
	}

	copy(sgd.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single SGD optimization step
func (sgd *SGDOptimizerState) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(sgd.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(sgd.WeightBuffers))
	}

	sgd.StepCount++

	// TODO: SGD CGO bridge implementation not yet available
	// For now, return an error indicating SGD is not fully implemented
	return fmt.Errorf("SGD optimizer execution not yet implemented in CGO bridge")
}

// UpdateLearningRate updates the learning rate
func (sgd *SGDOptimizerState) UpdateLearningRate(newLR float32) {
	sgd.LearningRate = newLR
}

// SetCommandPool enables command buffer pooling
func (sgd *SGDOptimizerState) SetCommandPool(commandPool unsafe.Pointer) {
	sgd.commandPool = commandPool
	sgd.usePooling = (commandPool != nil)
}

// GetStep returns the current optimization step count
func (sgd *SGDOptimizerState) GetStep() uint64 {
	return sgd.StepCount
}

// GetStepCount returns the current step count
func (sgd *SGDOptimizerState) GetStepCount() uint64 {
	return sgd.StepCount
}

// Cleanup releases all GPU buffers
func (sgd *SGDOptimizerState) Cleanup() {
	if sgd.MomentumBuffers != nil && sgd.memoryManager != nil {
		for i := range sgd.MomentumBuffers {
			if sgd.MomentumBuffers[i] != nil {
				sgd.memoryManager.ReleaseBuffer(sgd.MomentumBuffers[i])
				sgd.MomentumBuffers[i] = nil
			}
		}
	}

	// Clear slices
	sgd.MomentumBuffers = nil
	sgd.WeightBuffers = nil
	sgd.bufferSizes = nil
}

// GetState extracts optimizer state for checkpointing
func (sgd *SGDOptimizerState) GetState() (*OptimizerState, error) {
	stateData := make([]checkpoints.OptimizerTensor, 0)

	// Extract momentum buffers if momentum is used
	if sgd.Momentum > 0 && sgd.MomentumBuffers != nil {
		for i, buffer := range sgd.MomentumBuffers {
			tensor, err := extractBufferState(buffer, sgd.bufferSizes[i], 
				fmt.Sprintf("momentum_%d", i), "momentum")
			if err != nil {
				return nil, err
			}
			if tensor != nil {
				stateData = append(stateData, *tensor)
			}
		}
	}

	return &OptimizerState{
		Type: "SGD",
		Parameters: map[string]interface{}{
			"learning_rate": sgd.LearningRate,
			"momentum":      sgd.Momentum,
			"weight_decay":  sgd.WeightDecay,
			"nesterov":      sgd.Nesterov,
			"step_count":    sgd.StepCount,
		},
		StateData: stateData,
	}, nil
}

// LoadState restores optimizer state from checkpoint
func (sgd *SGDOptimizerState) LoadState(state *OptimizerState) error {
	// Validate state type
	if err := validateStateType("SGD", state); err != nil {
		return err
	}

	// Restore hyperparameters
	sgd.LearningRate = extractFloat32Param(state.Parameters, "learning_rate", sgd.LearningRate)
	sgd.Momentum = extractFloat32Param(state.Parameters, "momentum", sgd.Momentum)
	sgd.WeightDecay = extractFloat32Param(state.Parameters, "weight_decay", sgd.WeightDecay)
	sgd.Nesterov = extractBoolParam(state.Parameters, "nesterov", sgd.Nesterov)
	sgd.StepCount = extractUint64Param(state.Parameters, "step_count", sgd.StepCount)

	// Restore momentum buffers if present
	for _, tensor := range state.StateData {
		if tensor.StateType == "momentum" {
			idx := extractBufferIndex(tensor.Name)
			if idx < 0 || idx >= len(sgd.bufferSizes) {
				return fmt.Errorf("invalid buffer index in tensor name: %s", tensor.Name)
			}

			if sgd.MomentumBuffers == nil || sgd.MomentumBuffers[idx] == nil {
				return fmt.Errorf("momentum buffer %d not allocated", idx)
			}

			err := restoreBufferState(sgd.MomentumBuffers[idx], tensor.Data, 
				sgd.bufferSizes[idx], tensor.Name)
			if err != nil {
				return err
			}
		}
	}

	return nil
}