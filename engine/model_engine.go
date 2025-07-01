package engine

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/optimizer"
)

// ModelTrainingEngine extends the existing MPSTrainingEngine with layer-based model support
// This maintains the proven single-CGO-call architecture while adding layer abstraction
type ModelTrainingEngine struct {
	*MPSTrainingEngine
	modelSpec       *layers.ModelSpec
	parameterTensors []*memory.Tensor
	compiledForModel bool
	isDynamicEngine  bool // True if using dynamic graph engine, false if using hybrid fallback
}

// NewModelTrainingEngineDynamic creates a model-based training engine with dynamic graph support
// This supports any model architecture by building the MPSGraph dynamically from layer specs
func NewModelTrainingEngineDynamic(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
) (*ModelTrainingEngine, error) {
	// Validate model compatibility
	if err := modelSpec.ValidateModelForTrainingEngine(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}
	
	// Convert model to dynamic layer specifications
	dynamicSpecs, err := modelSpec.ConvertToDynamicLayerSpecs()
	if err != nil {
		return nil, fmt.Errorf("failed to convert model to dynamic specs: %v", err)
	}
	
	// Convert to CGO-compatible format
	cgoLayerSpecs := make([]cgo_bridge.LayerSpecC, len(dynamicSpecs))
	for i, spec := range dynamicSpecs {
		cgoLayerSpecs[i] = cgo_bridge.LayerSpecC{
			LayerType:       spec.LayerType,
			Name:            spec.NameBytes,
			InputShape:      spec.InputShape,
			InputShapeLen:   spec.InputShapeLen,
			OutputShape:     spec.OutputShape,
			OutputShapeLen:  spec.OutputShapeLen,
			ParamInt:        spec.ParamInt,
			ParamFloat:      spec.ParamFloat,
			ParamIntCount:   spec.ParamIntCount,
			ParamFloatCount: spec.ParamFloatCount,
		}
	}
	
	// Create Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal device: %v", err)
	}
	
	// Initialize global memory manager (required for parameter tensor creation)
	memory.InitializeGlobalMemoryManager(device)
	
	// Create TRUE dynamic training engine using the actual dynamic implementation
	dynamicEnginePtr, err := cgo_bridge.CreateTrainingEngineDynamic(
		device,
		config,
		cgoLayerSpecs,
		modelSpec.InputShape,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic training engine: %v", err)
	}
	
	// Create a wrapper MPSTrainingEngine to maintain compatibility
	baseEngine := &MPSTrainingEngine{
		device:        device,
		engine:        dynamicEnginePtr,
		config:        config,
		initialized:   true,
		isDynamic:     true, // This is a dynamic engine
		adamOptimizer: nil, // Dynamic engine handles optimization externally
	}
	
	// For Adam optimizer, initialize the external optimizer state
	if config.OptimizerType == cgo_bridge.Adam {
		adamConfig := optimizer.AdamConfig{
			LearningRate: config.LearningRate,
			Beta1:        config.Beta1,
			Beta2:        config.Beta2,
			Epsilon:      config.Epsilon,
			WeightDecay:  config.WeightDecay,
		}
		
		// Get all parameter shapes for Adam initialization (dynamic engine uses all parameters)
		paramShapes := modelSpec.ParameterShapes
		adamOptimizer, err := optimizer.NewAdamOptimizer(
			adamConfig,
			paramShapes,
			memory.GetGlobalMemoryManager(),
			device,
		)
		if err != nil {
			baseEngine.Cleanup()
			return nil, fmt.Errorf("failed to create Adam optimizer for dynamic engine: %v", err)
		}
		baseEngine.adamOptimizer = adamOptimizer
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
		compiledForModel:  true, // Dynamic engine compiles during creation
		isDynamicEngine:   true, // Flag to identify dynamic engines
	}
	
	return modelEngine, nil
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
		isDynamicEngine:   false, // Using hybrid fallback engine
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
	// Validate model first
	if err := modelSpec.ValidateModelForTrainingEngine(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}
	
	// Convert adamConfig to proper AdamConfig struct
	optAdamConfig := optimizer.AdamConfig{
		LearningRate: config.LearningRate,
		Beta1:        config.Beta1,
		Beta2:        config.Beta2,
		Epsilon:      config.Epsilon,
		WeightDecay:  config.WeightDecay,
	}
	
	// Get only FC layer parameter shapes for Adam (hybrid architecture)
	fcParameterShapes := getFCParameterShapes(modelSpec)
	
	// Create base training engine with Adam for FC parameters only
	baseEngine, err := NewMPSTrainingEngineWithAdam(config, optAdamConfig, fcParameterShapes)
	if err != nil {
		return nil, fmt.Errorf("failed to create Adam training engine: %v", err)
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
		isDynamicEngine:   false, // Using hybrid fallback engine
	}
	
	// Initialize parameters with proper values
	if err := modelEngine.initializeModelParameters(); err != nil {
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to initialize model parameters: %v", err)
	}
	
	// Compile model for execution
	if err := modelEngine.compileForExecution(); err != nil {
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to compile model for execution: %v", err)
	}
	
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
		err := cgo_bridge.ZeroMetalBuffer(mte.MPSTrainingEngine.GetDevice(), biasTensor.MetalBuffer(), biasTensor.Size())
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
		err := cgo_bridge.ZeroMetalBuffer(mte.MPSTrainingEngine.GetDevice(), biasTensor.MetalBuffer(), biasTensor.Size())
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
	
	// Generate proper random data
	data := make([]float32, totalElements)
	for i := range data {
		// Proper uniform distribution using rand.Float32()
		data[i] = min + (max-min)*rand.Float32()
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
	
	// Generate proper normal distribution using Box-Muller transform
	data := make([]float32, totalElements)
	for i := 0; i < totalElements; i += 2 {
		// Box-Muller transform for normal distribution
		u1 := rand.Float32()
		u2 := rand.Float32()
		
		// Avoid log(0)
		if u1 < 1e-8 {
			u1 = 1e-8
		}
		
		z0 := float32(math.Sqrt(-2.0*math.Log(float64(u1))) * math.Cos(2.0*math.Pi*float64(u2)))
		z1 := float32(math.Sqrt(-2.0*math.Log(float64(u1))) * math.Sin(2.0*math.Pi*float64(u2)))
		
		// Apply mean and standard deviation
		data[i] = mean + std*z0
		if i+1 < totalElements {
			data[i+1] = mean + std*z1
		}
	}
	
	// Copy to GPU tensor
	return cgo_bridge.CopyFloat32ArrayToMetalBuffer(tensor.MetalBuffer(), data)
}

// compileForExecution configures the TrainingEngine for this specific model
// This is where we bridge the layer specification with the existing high-performance engine
func (mte *ModelTrainingEngine) compileForExecution() error {
	// For now, we use the existing hybrid CNN architecture
	// TODO: Extend CGO bridge to accept generic layer configurations
	// TODO: Use model specification to configure the engine
	
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
	
	modelLayerSpecs := mte.modelSpec.Layers
	if len(modelLayerSpecs) < 3 {
		return fmt.Errorf("hybrid CNN requires at least 3 layers (Conv2D, activation, Dense)")
	}
	
	// Check for required layer types
	hasConv := false
	hasDense := false
	
	for _, layer := range modelLayerSpecs {
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
	
	if mte.isDynamicEngine {
		// Dynamic engine uses external Adam optimization with gradient extraction
		return mte.executeAdamStepDynamic(inputTensor, labelTensor)
	} else {
		// Hybrid fallback engine uses built-in Adam optimization
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
}

// executeAdamStepDynamic executes Adam optimization for dynamic engines using forward+backward+Adam pattern
func (mte *ModelTrainingEngine) executeAdamStepDynamic(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error) {
	// FIXED: Use proper forward+backward+Adam optimization pattern instead of incomplete C function
	
	// Get all parameter tensors for the dynamic engine (it uses all parameters, not just FC)
	allParameters := mte.parameterTensors
	if len(allParameters) == 0 {
		return 0, fmt.Errorf("no parameter tensors found for dynamic engine")
	}
	
	// Convert parameter tensors to weight buffers for CGO call
	weightBuffers := make([]unsafe.Pointer, len(allParameters))
	for i, tensor := range allParameters {
		if tensor == nil {
			return 0, fmt.Errorf("parameter tensor %d is nil", i)
		}
		weightBuffers[i] = tensor.MetalBuffer()
	}
	
	// Get batch size from input tensor shape
	inputShape := inputTensor.Shape()
	if len(inputShape) == 0 {
		return 0, fmt.Errorf("input tensor has no shape")
	}
	batchSize := inputShape[0]
	
	// FIXED: Use actual gradient computation instead of dummy gradients
	// This replaces the entire forward+dummy gradient approach with real gradient computation
	
	// Step 1: Set weight buffers in Adam optimizer
	if mte.MPSTrainingEngine.adamOptimizer == nil {
		return 0, fmt.Errorf("Adam optimizer not initialized for dynamic engine")
	}
	
	err := mte.MPSTrainingEngine.adamOptimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		return 0, fmt.Errorf("failed to set weight buffers in Adam optimizer: %v", err)
	}
	
	// Step 2: Create gradient tensors with same shapes as parameters
	gradientBuffers := make([]unsafe.Pointer, len(allParameters))
	gradientTensors := make([]*memory.Tensor, len(allParameters))
	
	for i, paramTensor := range allParameters {
		// Create gradient tensor with same shape as parameter
		gradTensor, err := memory.NewTensor(paramTensor.Shape(), memory.Float32, memory.GPU)
		if err != nil {
			// Cleanup previously created tensors
			for j := 0; j < i; j++ {
				gradientTensors[j].Release()
			}
			return 0, fmt.Errorf("failed to create gradient tensor %d: %v", i, err)
		}
		gradientTensors[i] = gradTensor
		gradientBuffers[i] = gradTensor.MetalBuffer()
	}
	
	// Cleanup gradient tensors after use
	defer func() {
		for _, gradTensor := range gradientTensors {
			if gradTensor != nil {
				gradTensor.Release()
			}
		}
	}()
	
	// Step 3: Compute ACTUAL loss and gradients using MPSGraph automatic differentiation
	actualLoss, err := cgo_bridge.ExecuteTrainingStepDynamicWithGradients(
		unsafe.Pointer(mte.MPSTrainingEngine.engine),
		inputTensor.MetalBuffer(),
		labelTensor.MetalBuffer(),
		weightBuffers,
		gradientBuffers,
		0.001, // learning rate (not used in gradient computation, only for legacy interface)
		batchSize,
	)
	if err != nil {
		return 0, fmt.Errorf("gradient computation failed: %v", err)
	}
	
	// Step 4: Apply Adam optimization with REAL gradients
	err = mte.MPSTrainingEngine.adamOptimizer.Step(gradientBuffers)
	if err != nil {
		return 0, fmt.Errorf("Adam optimization step failed: %v", err)
	}
	
	// Return actual computed loss
	return actualLoss, nil
}

// getAllParameterTensors returns all parameter tensors for dynamic inference
func (mte *ModelTrainingEngine) getAllParameterTensors() []*memory.Tensor {
	return mte.parameterTensors
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

// getFCParameterShapes extracts the parameter shapes for FC layers only
func getFCParameterShapes(modelSpec *layers.ModelSpec) [][]int {
	var fcShapes [][]int
	paramIndex := 0
	
	// Find the first Dense layer and return its parameter shapes
	for _, layerSpec := range modelSpec.Layers {
		switch layerSpec.Type {
		case layers.Dense:
			// Dense layer has weight + optional bias
			fcShapes = append(fcShapes, modelSpec.ParameterShapes[paramIndex]) // weights
			paramIndex++
			
			if useBias, exists := layerSpec.Parameters["use_bias"].(bool); exists && useBias {
				fcShapes = append(fcShapes, modelSpec.ParameterShapes[paramIndex]) // bias
				paramIndex++
			}
			
			return fcShapes // Return first Dense layer parameter shapes
			
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
	
	return fcShapes
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

// ExecuteInference performs forward-only pass returning model predictions
// Conforms to design requirements: single CGO call, GPU-resident, shared resources
func (mte *ModelTrainingEngine) ExecuteInference(
	inputTensor *memory.Tensor,
	batchSize int,
) (*cgo_bridge.InferenceResult, error) {
	// Validate model is compiled
	if !mte.compiledForModel {
		return nil, fmt.Errorf("model not compiled for execution")
	}
	
	// Validate input tensor
	if inputTensor == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}
	
	if batchSize <= 0 {
		return nil, fmt.Errorf("invalid batch size: %d", batchSize)
	}
	
	// Extract parameters based on engine type
	var weightBuffers []unsafe.Pointer
	if mte.MPSTrainingEngine.isDynamic {
		// Dynamic engines need ALL parameters in order
		allParameters := mte.getAllParameterTensors()
		if len(allParameters) == 0 {
			return nil, fmt.Errorf("no parameters found for dynamic inference")
		}
		
		weightBuffers = make([]unsafe.Pointer, len(allParameters))
		for i, tensor := range allParameters {
			if tensor == nil {
				return nil, fmt.Errorf("parameter tensor %d is nil", i)
			}
			weightBuffers[i] = tensor.MetalBuffer()
		}
	} else {
		// Hybrid engines only need FC layer parameters
		fcParameters := mte.getFCLayerParameters()
		if len(fcParameters) == 0 {
			return nil, fmt.Errorf("no FC layer parameters found for inference")
		}
		
		weightBuffers = make([]unsafe.Pointer, len(fcParameters))
		for i, tensor := range fcParameters {
			if tensor == nil {
				return nil, fmt.Errorf("FC parameter tensor %d is nil", i)
			}
			weightBuffers[i] = tensor.MetalBuffer()
		}
	}
	
	// Calculate output dimensions from model spec
	outputShape := mte.modelSpec.OutputShape
	if len(outputShape) < 2 {
		return nil, fmt.Errorf("invalid model output shape: %v", outputShape)
	}
	numClasses := outputShape[len(outputShape)-1] // Last dimension is number of classes
	
	// Single CGO call for complete inference (design compliant)
	result, err := cgo_bridge.ExecuteInference(
		mte.MPSTrainingEngine.engine,
		inputTensor.MetalBuffer(),
		weightBuffers,
		batchSize,
		numClasses,
		mte.MPSTrainingEngine.isDynamic, // Pass dynamic engine flag
	)
	
	if err != nil {
		return nil, fmt.Errorf("inference execution failed: %v", err)
	}
	
	return result, nil
}