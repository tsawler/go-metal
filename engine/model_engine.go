package engine

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// ModelTrainingEngine extends the existing MPSTrainingEngine with layer-based model support
// This maintains the proven single-CGO-call architecture while adding layer abstraction
type ModelTrainingEngine struct {
	*MPSTrainingEngine
	modelSpec       *layers.ModelSpec
	parameterTensors []*memory.Tensor
	compiledForModel bool
}

// NewModelTrainingEngine creates a model-based training engine
// This integrates with the existing high-performance TrainingEngine architecture
func NewModelTrainingEngine(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
) (*ModelTrainingEngine, error) {
	// Validate model compatibility with TrainingEngine
	if err := modelSpec.ValidateModelForTrainingEngine(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}
	
	// Create base training engine using proven hybrid approach
	baseEngine, err := NewMPSTrainingEngineHybrid(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create base training engine: %v", err)
	}
	
	// Create parameter tensors for the model
	paramTensors, err := modelSpec.CreateParameterTensors()
	if err != nil {
		baseEngine.Cleanup()
		return nil, fmt.Errorf("failed to create parameter tensors: %v", err)
	}
	
	modelEngine := &ModelTrainingEngine{
		MPSTrainingEngine: baseEngine,
		modelSpec:         modelSpec,
		parameterTensors:  paramTensors,
		compiledForModel:  false,
	}
	
	// Initialize parameters with proper values
	if err := modelEngine.initializeModelParameters(); err != nil {
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to initialize model parameters: %v", err)
	}
	
	// Compile model for execution (configure the TrainingEngine)
	if err := modelEngine.compileForExecution(); err != nil {
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to compile model for execution: %v", err)
	}
	
	return modelEngine, nil
}

// NewModelTrainingEngineWithAdam creates a model-based training engine with Adam optimizer
func NewModelTrainingEngineWithAdam(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
	adamConfig map[string]interface{},
) (*ModelTrainingEngine, error) {
	// Create base model engine
	modelEngine, err := NewModelTrainingEngine(modelSpec, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create base model engine: %v", err)
	}
	
	// Configure Adam optimizer with model parameter shapes
	// TODO: Integration with existing Adam optimizer
	// For now, the base engine handles this through the existing mechanism
	
	return modelEngine, nil
}

// initializeModelParameters initializes all model parameters with appropriate values
func (mte *ModelTrainingEngine) initializeModelParameters() error {
	paramIndex := 0
	
	for layerIndex, layerSpec := range mte.modelSpec.Layers {
		switch layerSpec.Type {
		case layers.Dense:
			err := mte.initializeDenseParameters(layerIndex, &layerSpec, &paramIndex)
			if err != nil {
				return fmt.Errorf("failed to initialize dense layer %d parameters: %v", layerIndex, err)
			}
			
		case layers.Conv2D:
			err := mte.initializeConv2DParameters(layerIndex, &layerSpec, &paramIndex)
			if err != nil {
				return fmt.Errorf("failed to initialize conv2d layer %d parameters: %v", layerIndex, err)
			}
			
		case layers.ReLU, layers.Softmax:
			// Activation layers have no parameters
			continue
			
		default:
			return fmt.Errorf("unsupported layer type for parameter initialization: %s", layerSpec.Type.String())
		}
	}
	
	return nil
}

// initializeDenseParameters initializes dense layer parameters with Xavier initialization
func (mte *ModelTrainingEngine) initializeDenseParameters(
	layerIndex int,
	layerSpec *layers.LayerSpec,
	paramIndex *int,
) error {
	inputSize, ok := layerSpec.Parameters["input_size"].(int)
	if !ok {
		return fmt.Errorf("missing input_size parameter")
	}
	
	outputSize, ok := layerSpec.Parameters["output_size"].(int)
	if !ok {
		return fmt.Errorf("missing output_size parameter")
	}
	
	useBias := true
	if bias, exists := layerSpec.Parameters["use_bias"].(bool); exists {
		useBias = bias
	}
	
	// Initialize weight tensor with Xavier initialization
	weightTensor := mte.parameterTensors[*paramIndex]
	err := mte.initializeXavier(weightTensor, inputSize, outputSize)
	if err != nil {
		return fmt.Errorf("failed to initialize weights: %v", err)
	}
	*paramIndex++
	
	// Initialize bias tensor (if present)
	if useBias {
		biasTensor := mte.parameterTensors[*paramIndex]
		err := biasTensor.Zero()
		if err != nil {
			return fmt.Errorf("failed to zero bias: %v", err)
		}
		*paramIndex++
	}
	
	return nil
}

// initializeConv2DParameters initializes Conv2D layer parameters with He initialization
func (mte *ModelTrainingEngine) initializeConv2DParameters(
	layerIndex int,
	layerSpec *layers.LayerSpec,
	paramIndex *int,
) error {
	inputChannels, ok := layerSpec.Parameters["input_channels"].(int)
	if !ok {
		return fmt.Errorf("missing input_channels parameter")
	}
	
	kernelSize, ok := layerSpec.Parameters["kernel_size"].(int)
	if !ok {
		return fmt.Errorf("missing kernel_size parameter")
	}
	
	useBias := true
	if bias, exists := layerSpec.Parameters["use_bias"].(bool); exists {
		useBias = bias
	}
	
	// Initialize weight tensor with He initialization
	weightTensor := mte.parameterTensors[*paramIndex]
	fanIn := inputChannels * kernelSize * kernelSize
	err := mte.initializeHe(weightTensor, fanIn)
	if err != nil {
		return fmt.Errorf("failed to initialize conv weights: %v", err)
	}
	*paramIndex++
	
	// Initialize bias tensor (if present)
	if useBias {
		biasTensor := mte.parameterTensors[*paramIndex]
		err := biasTensor.Zero()
		if err != nil {
			return fmt.Errorf("failed to zero conv bias: %v", err)
		}
		*paramIndex++
	}
	
	return nil
}

// initializeXavier initializes tensor with Xavier/Glorot uniform initialization
func (mte *ModelTrainingEngine) initializeXavier(tensor *memory.Tensor, fanIn, fanOut int) error {
	// Xavier initialization: uniform distribution in [-limit, limit]
	// where limit = sqrt(6 / (fan_in + fan_out))
	limit := 2.449490 / float32(fanIn + fanOut) // sqrt(6) ≈ 2.449
	
	return mte.initializeUniform(tensor, -limit, limit)
}

// initializeHe initializes tensor with He initialization for ReLU networks
func (mte *ModelTrainingEngine) initializeHe(tensor *memory.Tensor, fanIn int) error {
	// He initialization: normal distribution with std = sqrt(2 / fan_in)
	std := 1.4142136 / float32(fanIn) // sqrt(2) / fan_in
	
	return mte.initializeNormal(tensor, 0.0, std)
}

// initializeUniform initializes tensor with uniform distribution
func (mte *ModelTrainingEngine) initializeUniform(tensor *memory.Tensor, min, max float32) error {
	// Create uniform random data
	totalElements := 1
	for _, dim := range tensor.Shape() {
		totalElements *= dim
	}
	
	// Generate random data (simplified - would use proper random generation)
	data := make([]float32, totalElements)
	for i := range data {
		// Simple uniform distribution (should use proper random number generator)
		data[i] = min + (max-min)*0.5 // Placeholder - use actual random values
	}
	
	// Copy to GPU tensor
	return cgo_bridge.CopyFloat32ArrayToMetalBuffer(tensor.MetalBuffer(), data)
}

// initializeNormal initializes tensor with normal distribution
func (mte *ModelTrainingEngine) initializeNormal(tensor *memory.Tensor, mean, std float32) error {
	// Create normal random data
	totalElements := 1
	for _, dim := range tensor.Shape() {
		totalElements *= dim
	}
	
	// Generate random data (simplified - would use proper random generation)
	data := make([]float32, totalElements)
	for i := range data {
		// Simple normal distribution (should use proper random number generator)
		data[i] = mean // Placeholder - use actual normal random values
	}
	
	// Copy to GPU tensor
	return cgo_bridge.CopyFloat32ArrayToMetalBuffer(tensor.MetalBuffer(), data)
}

// compileForExecution configures the TrainingEngine for this specific model
// This is where we bridge the layer specification with the existing high-performance engine
func (mte *ModelTrainingEngine) compileForExecution() error {
	// Generate TrainingEngine configuration from model specification
	engineConfig, err := mte.modelSpec.GetTrainingEngineConfig()
	if err != nil {
		return fmt.Errorf("failed to get training engine config: %v", err)
	}
	
	// For now, we use the existing hybrid CNN architecture
	// TODO: Extend CGO bridge to accept generic layer configurations
	
	// Validate that our model matches the expected hybrid CNN structure
	// (This ensures compatibility with the existing 20k+ batch/s engine)
	if err := mte.validateHybridCNNCompatibility(); err != nil {
		return fmt.Errorf("model not compatible with hybrid CNN engine: %v", err)
	}
	
	mte.compiledForModel = true
	return nil
}

// validateHybridCNNCompatibility ensures model works with existing TrainingEngine
func (mte *ModelTrainingEngine) validateHybridCNNCompatibility() error {
	// The existing TrainingEngine expects:
	// 1. Conv2D layer (handled by MPS)
	// 2. ReLU activation
	// 3. Dense layer (handled by MPSGraph)
	// 4. Softmax/Loss (handled by MPSGraph)
	
	layers := mte.modelSpec.Layers
	if len(layers) < 3 {
		return fmt.Errorf("hybrid CNN requires at least 3 layers (Conv2D, activation, Dense)")
	}
	
	// Check for required layer types
	hasConv := false
	hasDense := false
	
	for _, layer := range layers {
		switch layer.Type {
		case layers.Conv2D:
			hasConv = true
		case layers.Dense:
			hasDense = true
		}
	}
	
	if !hasConv {
		return fmt.Errorf("hybrid CNN requires Conv2D layer")
	}
	
	if !hasDense {
		return fmt.Errorf("hybrid CNN requires Dense layer")
	}
	
	return nil
}

// ExecuteModelTrainingStep executes a complete model training step
// This maintains the single-CGO-call principle while supporting flexible layer models
func (mte *ModelTrainingEngine) ExecuteModelTrainingStep(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	learningRate float32,
) (float32, error) {
	if !mte.compiledForModel {
		return 0, fmt.Errorf("model not compiled for execution")
	}
	
	// Use the existing high-performance hybrid training step
	// The layer specification has been compiled into the appropriate parameter tensors
	
	// For hybrid CNN, we need the FC layer parameters (conv parameters are built-in)
	fcParameters := mte.getFCLayerParameters()
	if len(fcParameters) == 0 {
		return 0, fmt.Errorf("no FC layer parameters found")
	}
	
	// Execute using proven single-CGO-call architecture
	return mte.MPSTrainingEngine.ExecuteStepHybridFull(
		inputTensor,
		labelTensor,
		fcParameters,
		learningRate,
	)
}

// ExecuteModelTrainingStepWithAdam executes model training with Adam optimizer
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithAdam(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error) {
	if !mte.compiledForModel {
		return 0, fmt.Errorf("model not compiled for execution")
	}
	
	// Use the existing Adam training step
	fcParameters := mte.getFCLayerParameters()
	if len(fcParameters) == 0 {
		return 0, fmt.Errorf("no FC layer parameters found")
	}
	
	// Execute using proven Adam architecture
	return mte.MPSTrainingEngine.ExecuteStepHybridFullWithAdam(
		inputTensor,
		labelTensor,
		fcParameters,
	)
}

// getFCLayerParameters extracts the fully connected layer parameters for hybrid execution
func (mte *ModelTrainingEngine) getFCLayerParameters() []*memory.Tensor {
	var fcParams []*memory.Tensor
	paramIndex := 0
	
	// Find the first Dense layer and return its parameters
	for _, layerSpec := range mte.modelSpec.Layers {
		switch layerSpec.Type {
		case layers.Dense:
			// Dense layer has weight + optional bias
			fcParams = append(fcParams, mte.parameterTensors[paramIndex]) // weights
			paramIndex++
			
			if useBias, exists := layerSpec.Parameters["use_bias"].(bool); exists && useBias {
				fcParams = append(fcParams, mte.parameterTensors[paramIndex]) // bias
				paramIndex++
			}
			
			return fcParams // Return first Dense layer parameters
			
		case layers.Conv2D:
			// Skip Conv2D parameters (handled by MPS)
			paramIndex++ // weights
			if useBias, exists := layerSpec.Parameters["use_bias"].(bool); exists && useBias {
				paramIndex++ // bias
			}
			
		case layers.ReLU, layers.Softmax:
			// No parameters for activation layers
			continue
		}
	}
	
	return fcParams
}

// GetModelSpec returns the model specification
func (mte *ModelTrainingEngine) GetModelSpec() *layers.ModelSpec {
	return mte.modelSpec
}

// GetParameterTensors returns all model parameter tensors
func (mte *ModelTrainingEngine) GetParameterTensors() []*memory.Tensor {
	return mte.parameterTensors
}

// GetModelSummary returns a human-readable model summary
func (mte *ModelTrainingEngine) GetModelSummary() string {
	return mte.modelSpec.Summary()
}

// Cleanup releases all resources including model parameters
func (mte *ModelTrainingEngine) Cleanup() {
	// Release parameter tensors
	for _, tensor := range mte.parameterTensors {
		if tensor != nil {
			tensor.Release()
		}
	}
	mte.parameterTensors = nil
	
	// Cleanup base training engine
	if mte.MPSTrainingEngine != nil {
		mte.MPSTrainingEngine.Cleanup()
	}
	
	mte.compiledForModel = false
}