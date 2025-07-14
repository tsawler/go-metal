package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
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
func (adadelta *AdaDeltaOptimizerState) UpdateLearningRate(newLR float32) {
	// AdaDelta does not use a fixed learning rate; it adapts automatically based on parameter updates
	// This method is provided to satisfy the Optimizer interface but has no effect
}

// GetState extracts optimizer state for checkpointing
// Transfers GPU state to CPU in a single batched operation per buffer type
func (adadelta *AdaDeltaOptimizerState) GetState() (*OptimizerState, error) {
	stateData := make([]checkpoints.OptimizerTensor, 0, len(adadelta.squaredGradAvgBuffers)*2)
	
	// Extract squared gradient average buffers
	for i, buffer := range adadelta.squaredGradAvgBuffers {
		if buffer != nil {
			// Calculate number of elements (buffer size / 4 bytes per float32)
			numElements := adadelta.bufferSizes[i] / 4
			
			// Read GPU buffer to CPU
			data, err := cgo_bridge.CopyMetalBufferToFloat32Array(buffer, numElements)
			if err != nil {
				return nil, fmt.Errorf("failed to read squared gradient average buffer %d: %v", i, err)
			}
			
			stateData = append(stateData, checkpoints.OptimizerTensor{
				Name:      fmt.Sprintf("squared_grad_avg_%d", i),
				Shape:     []int{len(data)},
				Data:      data,
				StateType: "squared_grad_avg",
			})
		}
	}
	
	// Extract squared update average buffers
	for i, buffer := range adadelta.squaredUpdateAvgBuffers {
		if buffer != nil {
			// Calculate number of elements
			numElements := adadelta.bufferSizes[i] / 4
			
			// Read GPU buffer to CPU
			data, err := cgo_bridge.CopyMetalBufferToFloat32Array(buffer, numElements)
			if err != nil {
				return nil, fmt.Errorf("failed to read squared update average buffer %d: %v", i, err)
			}
			
			stateData = append(stateData, checkpoints.OptimizerTensor{
				Name:      fmt.Sprintf("squared_update_avg_%d", i),
				Shape:     []int{len(data)},
				Data:      data,
				StateType: "squared_update_avg",
			})
		}
	}
	
	return &OptimizerState{
		Type: "AdaDelta",
		Parameters: map[string]interface{}{
			"rho":         adadelta.config.Rho,
			"epsilon":     adadelta.config.Epsilon,
			"weight_decay": adadelta.config.WeightDecay,
			"step_count":  adadelta.currentStep,
		},
		StateData: stateData,
	}, nil
}

// LoadState restores optimizer state from checkpoint
// Transfers CPU state back to GPU in batched operations
func (adadelta *AdaDeltaOptimizerState) LoadState(state *OptimizerState) error {
	// Validate state type
	if err := validateStateType("AdaDelta", state); err != nil {
		return err
	}
	
	// Restore hyperparameters
	if rho, ok := state.Parameters["rho"].(float64); ok {
		adadelta.config.Rho = float32(rho)
	}
	if eps, ok := state.Parameters["epsilon"].(float64); ok {
		adadelta.config.Epsilon = float32(eps)
	}
	if wd, ok := state.Parameters["weight_decay"].(float64); ok {
		adadelta.config.WeightDecay = float32(wd)
	}
	if sc, ok := state.Parameters["step_count"].(float64); ok {
		adadelta.currentStep = uint64(sc)
	} else if sc, ok := state.Parameters["step_count"].(uint64); ok {
		adadelta.currentStep = sc
	}
	
	// Restore GPU buffers
	for _, tensor := range state.StateData {
		idx := extractBufferIndex(tensor.Name)
		if idx < 0 || idx >= len(adadelta.bufferSizes) {
			return fmt.Errorf("invalid buffer index in tensor name: %s", tensor.Name)
		}
		
		// Validate data size matches buffer size
		expectedElements := adadelta.bufferSizes[idx] / 4
		if len(tensor.Data) != expectedElements {
			return fmt.Errorf("data size mismatch for %s: expected %d elements, got %d",
				tensor.Name, expectedElements, len(tensor.Data))
		}
		
		// Write data back to GPU buffer
		switch tensor.StateType {
		case "squared_grad_avg":
			if adadelta.squaredGradAvgBuffers[idx] == nil {
				return fmt.Errorf("squared gradient average buffer %d is nil", idx)
			}
			if err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(adadelta.squaredGradAvgBuffers[idx], tensor.Data); err != nil {
				return fmt.Errorf("failed to restore squared gradient average buffer %d: %v", idx, err)
			}
		case "squared_update_avg":
			if adadelta.squaredUpdateAvgBuffers[idx] == nil {
				return fmt.Errorf("squared update average buffer %d is nil", idx)
			}
			if err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(adadelta.squaredUpdateAvgBuffers[idx], tensor.Data); err != nil {
				return fmt.Errorf("failed to restore squared update average buffer %d: %v", idx, err)
			}
		default:
			return fmt.Errorf("unknown state type: %s", tensor.StateType)
		}
	}
	
	return nil
}

// GetStepCount returns the current optimization step number
func (adadelta *AdaDeltaOptimizerState) GetStepCount() uint64 {
	return adadelta.currentStep
}