package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
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

	// DEBUG: Add logging to verify Adam is being called
	// if adam.StepCount%10 == 1 {
	//	fmt.Printf("🔧 Adam step %d: lr=%.6f, %d weights, %d gradients\n", 
	//		adam.StepCount, adam.LearningRate, len(adam.WeightBuffers), len(gradientBuffers))
	// }

	// DEBUG: Temporarily disable pooled Adam to test if non-pooled works
	// Learning broke again - need to isolate issue
	var err error
	// if adam.usePooling && adam.commandPool != nil {
	// 	// Use pooled command buffers with proper bias correction
	// 	err = cgo_bridge.ExecuteAdamStepMPSGraphPooled(
	// 		adam.device,
	// 		adam.WeightBuffers,
	// 		gradientBuffers,
	// 		adam.MomentumBuffers,
	// 		adam.VarianceBuffers,
	// 		adam.bufferSizes,
	// 		adam.LearningRate,
	// 		adam.Beta1,
	// 		adam.Beta2,
	// 		adam.Epsilon,
	// 		adam.WeightDecay,
	// 		int(adam.StepCount),
	// 		adam.commandPool,
	// 	)
	// } else {
		// Force non-pooled version to test learning
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
	// }

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

// GetState extracts optimizer state for checkpointing
// Transfers GPU state to CPU in a single batched operation per buffer type
func (adam *AdamOptimizerState) GetState() (*OptimizerState, error) {
	stateData := make([]checkpoints.OptimizerTensor, 0, len(adam.MomentumBuffers)*2)
	
	// Extract momentum buffers
	for i, buffer := range adam.MomentumBuffers {
		if buffer != nil {
			// Calculate number of elements (buffer size / 4 bytes per float32)
			numElements := adam.bufferSizes[i] / 4
			
			// Read GPU buffer to CPU
			data, err := cgo_bridge.CopyMetalBufferToFloat32Array(buffer, numElements)
			if err != nil {
				return nil, fmt.Errorf("failed to read momentum buffer %d: %v", i, err)
			}
			
			stateData = append(stateData, checkpoints.OptimizerTensor{
				Name:      fmt.Sprintf("momentum_%d", i),
				Shape:     []int{len(data)},
				Data:      data,
				StateType: "momentum",
			})
		}
	}
	
	// Extract variance buffers  
	for i, buffer := range adam.VarianceBuffers {
		if buffer != nil {
			// Calculate number of elements
			numElements := adam.bufferSizes[i] / 4
			
			// Read GPU buffer to CPU
			data, err := cgo_bridge.CopyMetalBufferToFloat32Array(buffer, numElements)
			if err != nil {
				return nil, fmt.Errorf("failed to read variance buffer %d: %v", i, err)
			}
			
			stateData = append(stateData, checkpoints.OptimizerTensor{
				Name:      fmt.Sprintf("variance_%d", i),
				Shape:     []int{len(data)},
				Data:      data,
				StateType: "variance",
			})
		}
	}
	
	return &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": adam.LearningRate,
			"beta1":         adam.Beta1,
			"beta2":         adam.Beta2,
			"epsilon":       adam.Epsilon,
			"weight_decay":  adam.WeightDecay,
			"step_count":    adam.StepCount,
		},
		StateData: stateData,
	}, nil
}

// LoadState restores optimizer state from checkpoint
// Transfers CPU state back to GPU in batched operations
func (adam *AdamOptimizerState) LoadState(state *OptimizerState) error {
	// Validate state type
	if err := validateStateType("Adam", state); err != nil {
		return err
	}
	
	// Restore hyperparameters
	if lr, ok := state.Parameters["learning_rate"].(float64); ok {
		adam.LearningRate = float32(lr)
	}
	if b1, ok := state.Parameters["beta1"].(float64); ok {
		adam.Beta1 = float32(b1)
	}
	if b2, ok := state.Parameters["beta2"].(float64); ok {
		adam.Beta2 = float32(b2)
	}
	if eps, ok := state.Parameters["epsilon"].(float64); ok {
		adam.Epsilon = float32(eps)
	}
	if wd, ok := state.Parameters["weight_decay"].(float64); ok {
		adam.WeightDecay = float32(wd)
	}
	if sc, ok := state.Parameters["step_count"].(float64); ok {
		adam.StepCount = uint64(sc)
	}
	
	// Restore GPU buffers
	for _, tensor := range state.StateData {
		idx := extractBufferIndex(tensor.Name)
		if idx < 0 || idx >= len(adam.bufferSizes) {
			return fmt.Errorf("invalid buffer index in tensor name: %s", tensor.Name)
		}
		
		// Validate data size matches buffer size
		expectedElements := adam.bufferSizes[idx] / 4
		if len(tensor.Data) != expectedElements {
			return fmt.Errorf("data size mismatch for %s: expected %d elements, got %d",
				tensor.Name, expectedElements, len(tensor.Data))
		}
		
		// Write data back to GPU buffer
		switch tensor.StateType {
		case "momentum":
			if adam.MomentumBuffers[idx] == nil {
				return fmt.Errorf("momentum buffer %d is nil", idx)
			}
			if err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(adam.MomentumBuffers[idx], tensor.Data); err != nil {
				return fmt.Errorf("failed to restore momentum buffer %d: %v", idx, err)
			}
		case "variance":
			if adam.VarianceBuffers[idx] == nil {
				return fmt.Errorf("variance buffer %d is nil", idx) 
			}
			if err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(adam.VarianceBuffers[idx], tensor.Data); err != nil {
				return fmt.Errorf("failed to restore variance buffer %d: %v", idx, err)
			}
		default:
			return fmt.Errorf("unknown state type: %s", tensor.StateType)
		}
	}
	
	return nil
}