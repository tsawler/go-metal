package engine

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/optimizer"
)

// MPSTrainingEngine handles training execution using MPSGraph
type MPSTrainingEngine struct {
	device       unsafe.Pointer           // MTLDevice
	engine       unsafe.Pointer           // Native training engine
	config       cgo_bridge.TrainingConfig
	initialized  bool
	isDynamic    bool                     // Always true for dynamic engine
	adamOptimizer     *optimizer.AdamOptimizerState     // Optional Adam optimizer
	rmspropOptimizer  *optimizer.RMSPropOptimizerState  // Optional RMSProp optimizer
	sgdOptimizer      *optimizer.SGDOptimizerState      // Optional SGD optimizer
	lbfgsOptimizer    *optimizer.LBFGSOptimizerState    // Optional L-BFGS optimizer
	adagradOptimizer  *optimizer.AdaGradOptimizerState  // Optional AdaGrad optimizer
	adadeltaOptimizer *optimizer.AdaDeltaOptimizerState // Optional AdaDelta optimizer
	nadamOptimizer    *optimizer.NadamOptimizerState    // Optional Nadam optimizer
	
	// RESOURCE LEAK FIX: Command buffer pooling support
	commandQueue unsafe.Pointer           // MTLCommandQueue for command buffer creation
	useCommandPooling bool                // Flag to enable command buffer pooling
}

// NewMPSTrainingEngine creates a new training engine
// DEPRECATED: Use NewModelTrainingEngineDynamic with proper layer specifications instead.
// This function is kept for backward compatibility but should not be used in new code.
func NewMPSTrainingEngine(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error) {
	return nil, fmt.Errorf("NewMPSTrainingEngine is deprecated. Use NewModelTrainingEngineDynamic with proper layer specifications instead. The dynamic engine requires model architecture specification and cannot work with generic configurations.")
}

// NewMPSTrainingEngineConstantWeights creates a new training engine with constant weights
// DEPRECATED: Use NewModelTrainingEngineDynamic with proper layer specifications instead.
// This function is kept for backward compatibility but should not be used in new code.
func NewMPSTrainingEngineConstantWeights(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error) {
	return nil, fmt.Errorf("NewMPSTrainingEngineConstantWeights is deprecated. Use NewModelTrainingEngineDynamic with proper layer specifications instead. The dynamic engine requires model architecture specification and cannot work with generic configurations.")
}



// ExecuteStep executes a complete training step
func (e *MPSTrainingEngine) ExecuteStep(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (float32, error) {
	if !e.initialized {
		return 0, fmt.Errorf("training engine not initialized")
	}
	
	// Validate inputs
	if inputTensor == nil || labelTensor == nil {
		return 0, fmt.Errorf("input or label tensor is nil")
	}
	
	if len(weightTensors) == 0 {
		return 0, fmt.Errorf("no weight tensors provided")
	}
	
	// Extract Metal buffer pointers
	inputBuffer := inputTensor.MetalBuffer()
	labelBuffer := labelTensor.MetalBuffer()
	
	weightBuffers := make([]unsafe.Pointer, len(weightTensors))
	for i, tensor := range weightTensors {
		if tensor == nil {
			return 0, fmt.Errorf("weight tensor %d is nil", i)
		}
		weightBuffers[i] = tensor.MetalBuffer()
	}
	
	// Execute training step via CGO
	loss, err := cgo_bridge.ExecuteTrainingStep(
		e.engine,
		inputBuffer,
		labelBuffer,
		weightBuffers,
	)
	
	if err != nil {
		return 0, fmt.Errorf("training step execution failed: %v", err)
	}
	
	return loss, nil
}




// ExecuteStepWithPrecomputedGradients executes Adam optimization with pre-computed gradients
// This is for cases where forward/backward pass has already been done separately
func (e *MPSTrainingEngine) ExecuteStepWithPrecomputedGradients(
	weightTensors []*memory.Tensor,
	gradientTensors []*memory.Tensor, // Pre-computed gradients from backward pass
) error {
	if !e.initialized {
		return fmt.Errorf("training engine not initialized")
	}

	if e.adamOptimizer == nil {
		return fmt.Errorf("Adam optimizer not initialized")
	}

	if len(weightTensors) != len(gradientTensors) {
		return fmt.Errorf("weight tensors (%d) and gradient tensors (%d) count mismatch",
			len(weightTensors), len(gradientTensors))
	}

	// Set current weight buffers in Adam optimizer
	weightBuffers := make([]unsafe.Pointer, len(weightTensors))
	for i, tensor := range weightTensors {
		if tensor == nil {
			return fmt.Errorf("weight tensor %d is nil", i)
		}
		weightBuffers[i] = tensor.MetalBuffer()
	}

	err := e.adamOptimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		return fmt.Errorf("failed to set weight buffers: %v", err)
	}

	// Extract gradient buffer pointers
	gradientBuffers := make([]unsafe.Pointer, len(gradientTensors))
	for i, tensor := range gradientTensors {
		if tensor == nil {
			return fmt.Errorf("gradient tensor %d is nil", i)
		}
		gradientBuffers[i] = tensor.MetalBuffer()
	}

	// Execute Adam optimization step
	err = e.adamOptimizer.Step(gradientBuffers)
	if err != nil {
		return fmt.Errorf("Adam optimization step failed: %v", err)
	}

	return nil
}

// GetAdamStats returns Adam optimizer statistics
func (e *MPSTrainingEngine) GetAdamStats() *optimizer.AdamStats {
	if e.adamOptimizer == nil {
		return nil
	}

	stats := e.adamOptimizer.GetStats()
	return &stats
}

// GetLBFGSStats returns L-BFGS optimizer statistics
func (e *MPSTrainingEngine) GetLBFGSStats() map[string]interface{} {
	if e.lbfgsOptimizer == nil {
		return nil
	}

	return e.lbfgsOptimizer.GetStats()
}

// UpdateAdamLearningRate updates the Adam optimizer learning rate
func (e *MPSTrainingEngine) UpdateAdamLearningRate(newLR float32) error {
	if e.adamOptimizer == nil {
		return fmt.Errorf("Adam optimizer not initialized")
	}

	e.adamOptimizer.UpdateLearningRate(newLR)
	return nil
}

// GetDevice returns the Metal device
func (e *MPSTrainingEngine) GetDevice() unsafe.Pointer {
	return e.device
}

// GetConfig returns the training configuration
func (e *MPSTrainingEngine) GetConfig() cgo_bridge.TrainingConfig {
	return e.config
}

// Cleanup releases resources with enhanced robustness and deterministic ordering
func (e *MPSTrainingEngine) Cleanup() {
	// Clean up optimizers in sequence (order matters for memory management)
	if e.adamOptimizer != nil {
		e.adamOptimizer.Cleanup()
		e.adamOptimizer = nil
	}
	
	if e.rmspropOptimizer != nil {
		e.rmspropOptimizer.Cleanup()
		e.rmspropOptimizer = nil
	}
	
	if e.sgdOptimizer != nil {
		e.sgdOptimizer.Cleanup()
		e.sgdOptimizer = nil
	}
	
	if e.lbfgsOptimizer != nil {
		e.lbfgsOptimizer.Cleanup()
		e.lbfgsOptimizer = nil
	}
	
	if e.adagradOptimizer != nil {
		e.adagradOptimizer.Cleanup()
		e.adagradOptimizer = nil
	}
	
	if e.adadeltaOptimizer != nil {
		e.adadeltaOptimizer.Cleanup()
		e.adadeltaOptimizer = nil
	}
	
	if e.nadamOptimizer != nil {
		e.nadamOptimizer.Cleanup()
		e.nadamOptimizer = nil
	}
	
	// Clean up training engine using the improved C/Objective-C bridge
	// The enhanced destroy_training_engine function now handles all cleanup robustly
	if e.initialized && e.engine != nil {
		cgo_bridge.DestroyTrainingEngine(e.engine)
		e.engine = nil
		e.initialized = false
	}

	// Release command queue (order matters: before device cleanup)
	if e.commandQueue != nil {
		cgo_bridge.ReleaseCommandQueue(e.commandQueue)
		e.commandQueue = nil
	}

	// Release Metal device last
	if e.device != nil {
		cgo_bridge.DestroyMetalDevice(e.device)
		e.device = nil
	}
	
	// Disable command pooling flag
	e.useCommandPooling = false
}

// TrainingStep represents a single training step result
type TrainingStep struct {
	Loss        float32
	BatchSize   int
	StepTime    int64 // Nanoseconds
	Success     bool
}

// BatchTrainer provides a higher-level interface for batch training
type BatchTrainer struct {
	engine       *MPSTrainingEngine
	batchSize    int
	currentStep  int
}

// NewBatchTrainer creates a new batch trainer
// DEPRECATED: Use NewModelTrainingEngineDynamic with proper layer specifications instead.
// This function is kept for backward compatibility but should not be used in new code.
func NewBatchTrainer(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error) {
	return nil, fmt.Errorf("NewBatchTrainer is deprecated. Use NewModelTrainingEngineDynamic with proper layer specifications instead. The dynamic engine requires model architecture specification and cannot work with generic configurations.")
}

// NewBatchTrainerConstantWeights creates a new batch trainer with constant weights
// DEPRECATED: Use NewModelTrainingEngineDynamic with proper layer specifications instead.
// This function is kept for backward compatibility but should not be used in new code.
func NewBatchTrainerConstantWeights(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error) {
	return nil, fmt.Errorf("NewBatchTrainerConstantWeights is deprecated. Use NewModelTrainingEngineDynamic with proper layer specifications instead. The dynamic engine requires model architecture specification and cannot work with generic configurations.")
}


// TrainBatch trains on a single batch
func (bt *BatchTrainer) TrainBatch(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (*TrainingStep, error) {
	startTime := getCurrentTime()
	
	loss, err := bt.engine.ExecuteStep(inputTensor, labelTensor, weightTensors)
	
	endTime := getCurrentTime()
	stepTime := endTime - startTime
	
	bt.currentStep++
	
	result := &TrainingStep{
		Loss:      loss,
		BatchSize: bt.batchSize,
		StepTime:  stepTime,
		Success:   err == nil,
	}
	
	return result, err
}



// GetCurrentStep returns the current step number
func (bt *BatchTrainer) GetCurrentStep() int {
	return bt.currentStep
}

// Cleanup releases resources with enhanced robustness
func (bt *BatchTrainer) Cleanup() {
	if bt.engine != nil {
		bt.engine.Cleanup()
		bt.engine = nil
	}
}

// Helper function to get current time in nanoseconds
func getCurrentTime() int64 {
	return time.Now().UnixNano()
}

