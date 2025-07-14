package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/memory"
)

// AdaGradOptimizerState represents GPU-resident AdaGrad optimizer state
type AdaGradOptimizerState struct {
	// Configuration
	config AdaGradConfig
	
	// GPU-resident state buffers
	squaredGradAvgBuffers []unsafe.Pointer // Accumulated squared gradient averages
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

// AdaGradConfig holds configuration for AdaGrad optimizer
type AdaGradConfig struct {
	LearningRate float32 // Learning rate
	Epsilon      float32 // Small constant for numerical stability
	WeightDecay  float32 // L2 regularization strength
}

// DefaultAdaGradConfig returns default AdaGrad optimizer configuration
func DefaultAdaGradConfig() AdaGradConfig {
	return AdaGradConfig{
		LearningRate: 0.01,
		Epsilon:      1e-10,
		WeightDecay:  0.0,
	}
}

// NewAdaGradOptimizer creates a new GPU-resident AdaGrad optimizer
func NewAdaGradOptimizer(
	config AdaGradConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*AdaGradOptimizerState, error) {
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
	
	adagrad := &AdaGradOptimizerState{
		config:                config,
		squaredGradAvgBuffers: make([]unsafe.Pointer, numWeights),
		WeightBuffers:         make([]unsafe.Pointer, numWeights),
		currentStep:           0,
		memoryManager:         memoryManager,
		device:                device,
		bufferSizes:           make([]int, numWeights),
	}
	
	// Calculate buffer sizes and allocate squared gradient average buffers
	for i, shape := range weightShapes {
		adagrad.bufferSizes[i] = calculateTensorSize(shape) * 4 // 4 bytes per float32
		
		// Allocate squared gradient average buffer
		squaredGradAvgBuffer := adagrad.memoryManager.AllocateBuffer(adagrad.bufferSizes[i])
		if squaredGradAvgBuffer == nil {
			adagrad.cleanup()
			return nil, fmt.Errorf("failed to allocate squared gradient average buffer for weight %d", i)
		}
		adagrad.squaredGradAvgBuffers[i] = squaredGradAvgBuffer
		
		// Initialize to zero
		if err := cgo_bridge.ZeroMetalBuffer(adagrad.device, squaredGradAvgBuffer, adagrad.bufferSizes[i]); err != nil {
			adagrad.cleanup()
			return nil, fmt.Errorf("failed to zero squared gradient average buffer: %v", err)
		}
	}
	
	return adagrad, nil
}

// cleanup releases all allocated buffers
func (adagrad *AdaGradOptimizerState) cleanup() {
	for i := range adagrad.squaredGradAvgBuffers {
		if adagrad.squaredGradAvgBuffers[i] != nil {
			adagrad.memoryManager.ReleaseBuffer(adagrad.squaredGradAvgBuffers[i])
		}
	}
}

// SetWeightBuffers sets the current weight buffer pointers
func (adagrad *AdaGradOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) != len(adagrad.WeightBuffers) {
		return fmt.Errorf("expected %d weight buffers, got %d", len(adagrad.WeightBuffers), len(weightBuffers))
	}
	
	copy(adagrad.WeightBuffers, weightBuffers)
	return nil
}

// Step performs a single AdaGrad optimization step
func (adagrad *AdaGradOptimizerState) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(adagrad.WeightBuffers) {
		return fmt.Errorf("gradient buffers length (%d) doesn't match weight buffers length (%d)",
			len(gradientBuffers), len(adagrad.WeightBuffers))
	}
	
	adagrad.currentStep++
	
	// Execute AdaGrad step using CGO bridge
	var err error
	if adagrad.usePooling && adagrad.commandPool != nil {
		err = cgo_bridge.ExecuteAdaGradStepMPSGraphPooled(
			adagrad.device,
			adagrad.WeightBuffers,
			gradientBuffers,
			adagrad.squaredGradAvgBuffers,
			len(adagrad.WeightBuffers),
			adagrad.bufferSizes,
			adagrad.config.LearningRate,
			adagrad.config.Epsilon,
			adagrad.config.WeightDecay,
			adagrad.commandPool,
		)
	} else {
		err = cgo_bridge.ExecuteAdaGradStepMPSGraph(
			adagrad.device,
			adagrad.WeightBuffers,
			gradientBuffers,
			adagrad.squaredGradAvgBuffers,
			len(adagrad.WeightBuffers),
			adagrad.bufferSizes,
			adagrad.config.LearningRate,
			adagrad.config.Epsilon,
			adagrad.config.WeightDecay,
		)
	}
	
	if err != nil {
		return fmt.Errorf("AdaGrad step failed: %v", err)
	}
	
	// Log progress periodically
	if adagrad.currentStep%10 == 0 {
		fmt.Printf("AdaGrad step %d completed, lr=%.6f\n", adagrad.currentStep, adagrad.config.LearningRate)
	}
	
	return nil
}

// SetCommandPool sets the command buffer pool for Metal operations
func (adagrad *AdaGradOptimizerState) SetCommandPool(pool unsafe.Pointer) {
	adagrad.commandPool = pool
	adagrad.usePooling = (pool != nil)
}

// GetStep returns the current optimization step count
func (adagrad *AdaGradOptimizerState) GetStep() uint64 {
	return adagrad.currentStep
}

// GetStats returns optimizer statistics
func (adagrad *AdaGradOptimizerState) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"step":          adagrad.currentStep,
		"learning_rate": adagrad.config.LearningRate,
		"epsilon":       adagrad.config.Epsilon,
		"weight_decay":  adagrad.config.WeightDecay,
	}
}

// Cleanup releases all GPU buffers
func (adagrad *AdaGradOptimizerState) Cleanup() {
	adagrad.cleanup()
}

// UpdateLearningRate updates the learning rate for the optimizer
func (adagrad *AdaGradOptimizerState) UpdateLearningRate(newLR float32) {
	adagrad.config.LearningRate = newLR
}

// GetState extracts optimizer state for checkpointing
// Transfers GPU state to CPU in a single batched operation per buffer type
func (adagrad *AdaGradOptimizerState) GetState() (*OptimizerState, error) {
	stateData := make([]checkpoints.OptimizerTensor, 0, len(adagrad.squaredGradAvgBuffers))
	
	// Extract squared gradient average buffers
	for i, buffer := range adagrad.squaredGradAvgBuffers {
		if buffer != nil {
			// Calculate number of elements (buffer size / 4 bytes per float32)
			numElements := adagrad.bufferSizes[i] / 4
			
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
	
	return &OptimizerState{
		Type: "AdaGrad",
		Parameters: map[string]interface{}{
			"learning_rate": adagrad.config.LearningRate,
			"epsilon":       adagrad.config.Epsilon,
			"weight_decay":  adagrad.config.WeightDecay,
			"step_count":    adagrad.currentStep,
		},
		StateData: stateData,
	}, nil
}

// LoadState restores optimizer state from checkpoint
// Transfers CPU state back to GPU in batched operations
func (adagrad *AdaGradOptimizerState) LoadState(state *OptimizerState) error {
	// Validate state type
	if err := validateStateType("AdaGrad", state); err != nil {
		return err
	}
	
	// Restore hyperparameters
	if lr, ok := state.Parameters["learning_rate"].(float64); ok {
		adagrad.config.LearningRate = float32(lr)
	}
	if eps, ok := state.Parameters["epsilon"].(float64); ok {
		adagrad.config.Epsilon = float32(eps)
	}
	if wd, ok := state.Parameters["weight_decay"].(float64); ok {
		adagrad.config.WeightDecay = float32(wd)
	}
	if sc, ok := state.Parameters["step_count"].(float64); ok {
		adagrad.currentStep = uint64(sc)
	} else if sc, ok := state.Parameters["step_count"].(uint64); ok {
		adagrad.currentStep = sc
	}
	
	// Restore GPU buffers
	for _, tensor := range state.StateData {
		idx := extractBufferIndex(tensor.Name)
		if idx < 0 || idx >= len(adagrad.bufferSizes) {
			return fmt.Errorf("invalid buffer index in tensor name: %s", tensor.Name)
		}
		
		// Validate data size matches buffer size
		expectedElements := adagrad.bufferSizes[idx] / 4
		if len(tensor.Data) != expectedElements {
			return fmt.Errorf("data size mismatch for %s: expected %d elements, got %d",
				tensor.Name, expectedElements, len(tensor.Data))
		}
		
		// Write data back to GPU buffer
		switch tensor.StateType {
		case "squared_grad_avg":
			if adagrad.squaredGradAvgBuffers[idx] == nil {
				return fmt.Errorf("squared gradient average buffer %d is nil", idx)
			}
			if err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(adagrad.squaredGradAvgBuffers[idx], tensor.Data); err != nil {
				return fmt.Errorf("failed to restore squared gradient average buffer %d: %v", idx, err)
			}
		default:
			return fmt.Errorf("unknown state type: %s", tensor.StateType)
		}
	}
	
	return nil
}

// GetStepCount returns the current optimization step number
func (adagrad *AdaGradOptimizerState) GetStepCount() uint64 {
	return adagrad.currentStep
}