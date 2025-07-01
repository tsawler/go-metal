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
	isDynamic    bool                     // True if using dynamic engine, false for hybrid
	adamOptimizer *optimizer.AdamOptimizerState // Optional Adam optimizer
}

// NewMPSTrainingEngine creates a new training engine
func NewMPSTrainingEngine(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error) {
	// Create Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal device: %v", err)
	}
	
	// Initialize global memory manager
	memory.InitializeGlobalMemoryManager(device)
	
	// Create training engine
	engine, err := cgo_bridge.CreateTrainingEngine(device, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create training engine: %v", err)
	}
	
	return &MPSTrainingEngine{
		device:        device,
		engine:        engine,
		config:        config,
		initialized:   true,
		isDynamic:     false, // Regular hybrid engine
		adamOptimizer: nil, // No Adam optimizer by default
	}, nil
}

// NewMPSTrainingEngineConstantWeights creates a new training engine with constant weights
// This avoids the MPSGraph isStaticMPSType assertion issue with convolution operations
func NewMPSTrainingEngineConstantWeights(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error) {
	// Create Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal device: %v", err)
	}
	
	// Initialize global memory manager
	memory.InitializeGlobalMemoryManager(device)
	
	// Create training engine with constant weights
	engine, err := cgo_bridge.CreateTrainingEngineConstantWeights(device, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create constant weights training engine: %v", err)
	}
	
	return &MPSTrainingEngine{
		device:        device,
		engine:        engine,
		config:        config,
		initialized:   true,
		isDynamic:     false, // Constant weights hybrid engine
		adamOptimizer: nil, // No Adam optimizer by default
	}, nil
}

// NewMPSTrainingEngineHybrid creates a new hybrid MPS/MPSGraph training engine
// This uses MPS for convolution and MPSGraph for other operations, avoiding the assertion issue
func NewMPSTrainingEngineHybrid(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error) {
	// Create Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal device: %v", err)
	}
	
	// Initialize global memory manager
	memory.InitializeGlobalMemoryManager(device)
	
	// Create hybrid training engine
	engine, err := cgo_bridge.CreateTrainingEngineHybrid(device, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create hybrid training engine: %v", err)
	}
	
	return &MPSTrainingEngine{
		device:        device,
		engine:        engine,
		config:        config,
		initialized:   true,
		isDynamic:     false, // Constant weights hybrid engine
		adamOptimizer: nil, // No Adam optimizer by default
	}, nil
}

// NewMPSTrainingEngineWithAdam creates a new hybrid training engine with Adam optimizer
func NewMPSTrainingEngineWithAdam(config cgo_bridge.TrainingConfig, adamConfig optimizer.AdamConfig, weightShapes [][]int) (*MPSTrainingEngine, error) {
	// Create base training engine
	engine, err := NewMPSTrainingEngineHybrid(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create base training engine: %v", err)
	}

	// Create Adam optimizer
	memoryManager := memory.GetGlobalMemoryManager()
	adamOpt, err := optimizer.NewAdamOptimizer(adamConfig, weightShapes, memoryManager, engine.device)
	if err != nil {
		engine.Cleanup()
		return nil, fmt.Errorf("failed to create Adam optimizer: %v", err)
	}

	engine.adamOptimizer = adamOpt
	return engine, nil
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

// ExecuteStepHybrid executes a complete training step using hybrid MPS/MPSGraph approach
func (e *MPSTrainingEngine) ExecuteStepHybrid(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor, // Only FC weights for hybrid approach
) (float32, error) {
	if !e.initialized {
		return 0, fmt.Errorf("training engine not initialized")
	}
	
	// Validate inputs
	if inputTensor == nil || labelTensor == nil {
		return 0, fmt.Errorf("input or label tensor is nil")
	}
	
	// Hybrid approach expects only FC weights (conv weights are built-in)
	if len(weightTensors) != 2 {
		return 0, fmt.Errorf("hybrid approach expects 2 weight tensors (FC weights + bias), got %d", len(weightTensors))
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
	
	// Execute hybrid training step via CGO
	loss, err := cgo_bridge.ExecuteTrainingStepHybrid(
		e.engine,
		inputBuffer,
		labelBuffer,
		weightBuffers,
	)
	
	if err != nil {
		return 0, fmt.Errorf("hybrid training step execution failed: %v", err)
	}
	
	return loss, nil
}

// ExecuteStepHybridFull executes a complete training step with backward pass using hybrid MPS/MPSGraph approach
func (e *MPSTrainingEngine) ExecuteStepHybridFull(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor, // Only FC weights for hybrid approach
	learningRate float32,
) (float32, error) {
	if !e.initialized {
		return 0, fmt.Errorf("training engine not initialized")
	}
	
	// Validate inputs
	if inputTensor == nil || labelTensor == nil {
		return 0, fmt.Errorf("input or label tensor is nil")
	}
	
	// Hybrid approach expects only FC weights (conv weights are built-in)
	if len(weightTensors) != 2 {
		return 0, fmt.Errorf("hybrid approach expects 2 weight tensors (FC weights + bias), got %d", len(weightTensors))
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
	
	// Execute hybrid full training step via CGO
	loss, err := cgo_bridge.ExecuteTrainingStepHybridFull(
		e.engine,
		inputBuffer,
		labelBuffer,
		weightBuffers,
		learningRate,
	)
	
	if err != nil {
		return 0, fmt.Errorf("hybrid full training step execution failed: %v", err)
	}
	
	return loss, nil
}

// ExecuteStepHybridFullWithAdam executes a complete training step with Adam optimizer
// This performs forward pass, backward pass, and Adam optimization in one call
func (e *MPSTrainingEngine) ExecuteStepHybridFullWithAdam(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (float32, error) {
	if !e.initialized {
		return 0, fmt.Errorf("training engine not initialized")
	}

	if e.adamOptimizer == nil {
		return 0, fmt.Errorf("Adam optimizer not initialized")
	}

	// Validate inputs
	if inputTensor == nil || labelTensor == nil {
		return 0, fmt.Errorf("input or label tensor is nil")
	}

	// Hybrid approach expects only FC weights (conv weights are built-in)
	if len(weightTensors) != 2 {
		return 0, fmt.Errorf("hybrid approach expects 2 weight tensors (FC weights + bias), got %d", len(weightTensors))
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

	// Create gradient tensors to receive the computed gradients
	gradientTensors := make([]*memory.Tensor, len(weightTensors))
	gradientBuffers := make([]unsafe.Pointer, len(weightTensors))
	
	for i, weightTensor := range weightTensors {
		// Create gradient tensor with same shape as weight tensor
		gradTensor, err := memory.NewTensor(weightTensor.Shape(), memory.Float32, memory.GPU)
		if err != nil {
			// Cleanup previously created gradient tensors
			for j := 0; j < i; j++ {
				gradientTensors[j].Release()
			}
			return 0, fmt.Errorf("failed to create gradient tensor %d: %v", i, err)
		}
		gradientTensors[i] = gradTensor
		gradientBuffers[i] = gradTensor.MetalBuffer()
	}
	
	// Ensure gradient tensors are cleaned up
	defer func() {
		for _, gradTensor := range gradientTensors {
			if gradTensor != nil {
				gradTensor.Release()
			}
		}
	}()

	// Execute forward + backward pass to get loss and gradients
	loss, err := cgo_bridge.ExecuteTrainingStepHybridWithGradients(
		e.engine,
		inputBuffer,
		labelBuffer,
		weightBuffers,
		gradientBuffers,
	)
	
	if err != nil {
		return 0, fmt.Errorf("forward+backward pass failed: %v", err)
	}

	// Set current weight buffers in Adam optimizer
	err = e.adamOptimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		return 0, fmt.Errorf("failed to set weight buffers: %v", err)
	}

	// Execute Adam optimization step with computed gradients
	err = e.adamOptimizer.Step(gradientBuffers)
	if err != nil {
		return 0, fmt.Errorf("Adam optimization step failed: %v", err)
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

// Cleanup releases resources
func (e *MPSTrainingEngine) Cleanup() {
	if e.adamOptimizer != nil {
		e.adamOptimizer.Cleanup()
		e.adamOptimizer = nil
	}
	
	if e.initialized && e.engine != nil {
		cgo_bridge.DestroyTrainingEngine(e.engine)
		e.engine = nil
		e.initialized = false
	}
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
func NewBatchTrainer(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error) {
	engine, err := NewMPSTrainingEngine(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create training engine: %v", err)
	}
	
	return &BatchTrainer{
		engine:      engine,
		batchSize:   batchSize,
		currentStep: 0,
	}, nil
}

// NewBatchTrainerConstantWeights creates a new batch trainer with constant weights
func NewBatchTrainerConstantWeights(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error) {
	engine, err := NewMPSTrainingEngineConstantWeights(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create constant weights training engine: %v", err)
	}
	
	return &BatchTrainer{
		engine:      engine,
		batchSize:   batchSize,
		currentStep: 0,
	}, nil
}

// NewBatchTrainerHybrid creates a new batch trainer with hybrid MPS/MPSGraph approach
func NewBatchTrainerHybrid(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error) {
	engine, err := NewMPSTrainingEngineHybrid(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create hybrid training engine: %v", err)
	}
	
	return &BatchTrainer{
		engine:      engine,
		batchSize:   batchSize,
		currentStep: 0,
	}, nil
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

// TrainBatchHybrid trains on a single batch using hybrid MPS/MPSGraph approach
func (bt *BatchTrainer) TrainBatchHybrid(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor, // Only FC weights for hybrid approach
) (*TrainingStep, error) {
	startTime := getCurrentTime()
	
	loss, err := bt.engine.ExecuteStepHybrid(inputTensor, labelTensor, weightTensors)
	
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

// TrainBatchHybridFull trains on a single batch using hybrid MPS/MPSGraph approach with full training loop
func (bt *BatchTrainer) TrainBatchHybridFull(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor, // Only FC weights for hybrid approach
	learningRate float32,
) (*TrainingStep, error) {
	startTime := getCurrentTime()
	
	loss, err := bt.engine.ExecuteStepHybridFull(inputTensor, labelTensor, weightTensors, learningRate)
	
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

// Cleanup releases resources
func (bt *BatchTrainer) Cleanup() {
	if bt.engine != nil {
		bt.engine.Cleanup()
	}
}

// Helper function to get current time in nanoseconds
func getCurrentTime() int64 {
	return time.Now().UnixNano()
}