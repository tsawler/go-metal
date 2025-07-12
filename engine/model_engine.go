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

// Helper function to convert bool to int32
func boolToInt32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// ModelTrainingEngine extends the existing MPSTrainingEngine with layer-based model support
// This maintains the proven single-CGO-call architecture while adding layer abstraction
type ModelTrainingEngine struct {
	*MPSTrainingEngine
	modelSpec       *layers.ModelSpec
	parameterTensors []*memory.Tensor
	gradientTensors  []*memory.Tensor // Pre-allocated gradient tensors for performance
	compiledForModel bool
	isDynamicEngine  bool // True if using dynamic graph engine, false if using hybrid fallback
}

// NewModelTrainingEngineDynamic creates a model-based training engine with dynamic graph support
// This supports any model architecture by building the MPSGraph dynamically from layer specs
func NewModelTrainingEngineDynamic(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
) (*ModelTrainingEngine, error) {
	// Validate model compatibility for Dynamic Engine (supports 2D, 4D, any architecture)
	if err := modelSpec.ValidateModelForDynamicEngine(); err != nil {
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
			// ARCHITECTURAL FIX: Copy running statistics from DynamicLayerSpec
			RunningMean:     spec.RunningMean,
			RunningVar:      spec.RunningVar,
			RunningStatsSize: int32(len(spec.RunningMean)),
			HasRunningStats:  boolToInt32(spec.HasRunningStats),
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
	
	// RESOURCE LEAK FIX: Initialize command queue for command buffer pooling
	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		cgo_bridge.DestroyTrainingEngine(dynamicEnginePtr)
		return nil, fmt.Errorf("failed to create command queue: %v", err)
	}

	// Create a wrapper MPSTrainingEngine to maintain compatibility
	baseEngine := &MPSTrainingEngine{
		device:        device,
		engine:        dynamicEnginePtr,
		config:        config,
		initialized:   true,
		isDynamic:     true, // This is a dynamic engine
		adamOptimizer: nil, // Dynamic engine handles optimization externally
		
		// RESOURCE LEAK FIX: Command buffer pooling support
		commandQueue:      commandQueue,
		useCommandPooling: true, // Enable command pooling by default
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
		
		// RESOURCE LEAK FIX: Enable command buffer pooling in Adam optimizer
		if baseEngine.useCommandPooling && baseEngine.commandQueue != nil {
			adamOptimizer.SetCommandPool(baseEngine.commandQueue)
		}
	}
	
	// For L-BFGS optimizer, initialize the external optimizer state
	if config.OptimizerType == cgo_bridge.LBFGS {
		lbfgsConfig := optimizer.LBFGSConfig{
			HistorySize:   10,    // Default history size
			LineSearchTol: 1e-4,  // Line search tolerance
			MaxLineSearch: 20,    // Maximum line search iterations
			C1:            1e-4,  // Armijo condition parameter
			C2:            0.9,   // Wolfe condition parameter
			InitialStep:   1.0,   // Initial step size
		}
		
		// Get all parameter shapes for L-BFGS initialization (dynamic engine uses all parameters)
		paramShapes := modelSpec.ParameterShapes
		lbfgsOptimizer, err := optimizer.NewLBFGSOptimizer(
			lbfgsConfig,
			paramShapes,
			memory.GetGlobalMemoryManager(),
			baseEngine.device,
		)
		if err != nil {
			baseEngine.Cleanup()
			return nil, fmt.Errorf("failed to create L-BFGS optimizer for dynamic engine: %v", err)
		}
		baseEngine.lbfgsOptimizer = lbfgsOptimizer
		
		// RESOURCE LEAK FIX: Enable command buffer pooling in L-BFGS optimizer
		if baseEngine.useCommandPooling && baseEngine.commandQueue != nil {
			lbfgsOptimizer.SetCommandPool(baseEngine.commandQueue)
		}
	}
	
	// Create parameter tensors for the model
	paramTensors, err := modelSpec.CreateParameterTensors()
	if err != nil {
		baseEngine.Cleanup()
		return nil, fmt.Errorf("failed to create parameter tensors: %v", err)
	}
	
	// Create gradient tensors with same shapes as parameter tensors for performance
	gradTensors := make([]*memory.Tensor, len(paramTensors))
	for i, paramTensor := range paramTensors {
		gradTensor, err := memory.NewTensor(paramTensor.Shape(), memory.Float32, memory.GPU)
		if err != nil {
			// Cleanup previously created gradient tensors
			for j := 0; j < i; j++ {
				if gradTensors[j] != nil {
					gradTensors[j].Release()
				}
			}
			// Cleanup parameter tensors
			for _, tensor := range paramTensors {
				if tensor != nil {
					tensor.Release()
				}
			}
			baseEngine.Cleanup()
			return nil, fmt.Errorf("failed to create gradient tensor %d: %v", i, err)
		}
		gradTensors[i] = gradTensor
	}
	
	modelEngine := &ModelTrainingEngine{
		MPSTrainingEngine: baseEngine,
		modelSpec:         modelSpec,
		parameterTensors:  paramTensors,
		gradientTensors:   gradTensors,
		compiledForModel:  true, // Dynamic engine compiles during creation
		isDynamicEngine:   true, // Flag to identify dynamic engines
	}
	
	// CRITICAL FIX: Initialize parameters with proper values for gradient flow
	// Dynamic engine was missing this step, causing gradient flow issues
	if err := modelEngine.initializeModelParameters(); err != nil {
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to initialize model parameters: %v", err)
	}
	
	return modelEngine, nil
}

// NewModelTrainingEngine creates a model-based training engine
// This integrates with the existing high-performance TrainingEngine architecture
func NewModelTrainingEngine(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
) (*ModelTrainingEngine, error) {
	// Validate model compatibility with Hybrid TrainingEngine
	if err := modelSpec.ValidateModelForHybridEngine(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}
	
	// Create model configuration from model specification
	modelConfig, err := createModelConfigFromSpec(modelSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to create model configuration: %v", err)
	}
	
	// DEBUG: Log the generated model configuration
	fmt.Printf("🔧 Generated Model Configuration:\n")
	fmt.Printf("  Input: %dx%dx%d (batch: %d)\n", modelConfig.InputChannels, modelConfig.InputHeight, modelConfig.InputWidth, modelConfig.BatchSize)
	fmt.Printf("  Conv1: %d filters, %dx%d → %dx%dx%d\n", modelConfig.Conv1OutChannels, modelConfig.Conv1KernelSize, modelConfig.Conv1KernelSize, 
		modelConfig.Conv1OutChannels, modelConfig.Conv1OutHeight, modelConfig.Conv1OutWidth)
	fmt.Printf("  Conv2: %d filters, %dx%d → %dx%dx%d\n", modelConfig.Conv2OutChannels, modelConfig.Conv2KernelSize, modelConfig.Conv2KernelSize,
		modelConfig.Conv2OutChannels, modelConfig.Conv2OutHeight, modelConfig.Conv2OutWidth)
	fmt.Printf("  Conv3: %d filters, %dx%d → %dx%dx%d\n", modelConfig.Conv3OutChannels, modelConfig.Conv3KernelSize, modelConfig.Conv3KernelSize,
		modelConfig.Conv3OutChannels, modelConfig.Conv3OutHeight, modelConfig.Conv3OutWidth)
	fmt.Printf("  FC1: %d → %d\n", modelConfig.FC1InputSize, modelConfig.FC1OutputSize)
	fmt.Printf("  FC2: %d → %d\n", modelConfig.FC1OutputSize, modelConfig.FC2OutputSize)
	
	// Create base training engine using proven hybrid approach
	baseEngine, err := NewMPSTrainingEngineHybrid(config, modelConfig)
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
	// Validate model compatibility with Hybrid TrainingEngine
	if err := modelSpec.ValidateModelForHybridEngine(); err != nil {
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
			
		case layers.BatchNorm:
			err := mte.initializeBatchNormParameters(layerIndex, &layerSpec, &paramIndex)
			if err != nil {
				return fmt.Errorf("failed to initialize batchnorm layer %d parameters: %v", layerIndex, err)
			}
			
		case layers.ReLU, layers.Softmax, layers.Dropout, layers.LeakyReLU, layers.ELU:
			// Activation layers and dropout have no parameters
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
	
	useBias := true
	if bias, exists := layerSpec.Parameters["use_bias"].(bool); exists {
		useBias = bias
	}
	
	// Initialize weight tensor with He initialization (better for ReLU networks)
	weightTensor := mte.parameterTensors[*paramIndex]
	err := mte.initializeHe(weightTensor, inputSize)
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

// initializeBatchNormParameters initializes BatchNorm layer parameters
// gamma (scale) initialized to 1.0, beta (shift) initialized to 0.0
func (mte *ModelTrainingEngine) initializeBatchNormParameters(
	layerIndex int,
	layerSpec *layers.LayerSpec,
	paramIndex *int,
) error {
	// Get BatchNorm parameters - just validate num_features exists
	_, ok := layerSpec.Parameters["num_features"].(int)
	if !ok {
		return fmt.Errorf("missing num_features parameter")
	}
	
	affine := true
	if af, exists := layerSpec.Parameters["affine"].(bool); exists {
		affine = af
	}
	
	// Only initialize parameters if affine=true (learnable scale and shift)
	if affine {
		// Initialize gamma (scale) parameter to 1.0
		gammaTensor := mte.parameterTensors[*paramIndex]
		err := mte.initializeUniform(gammaTensor, 1.0, 1.0) // Constant 1.0
		if err != nil {
			return fmt.Errorf("failed to initialize gamma (scale): %v", err)
		}
		*paramIndex++
		
		// Initialize beta (shift) parameter to 0.0  
		betaTensor := mte.parameterTensors[*paramIndex]
		err = cgo_bridge.ZeroMetalBuffer(mte.MPSTrainingEngine.GetDevice(), betaTensor.MetalBuffer(), betaTensor.Size())
		if err != nil {
			return fmt.Errorf("failed to initialize beta (shift): %v", err)
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
	std := float32(math.Sqrt(2.0 / float64(fanIn))) // FIXED: sqrt(2 / fan_in) not sqrt(2) / fan_in
	
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
// This method adapts to any model architecture using dynamic or hybrid engines
func (mte *ModelTrainingEngine) compileForExecution() error {
	// Use dynamic architecture-aware compilation based on engine type
	if mte.isDynamicEngine {
		// Dynamic Engine: Supports any model architecture
		// The engine is already configured with the model specification
		// through the dynamic layer specs in the constructor
		return mte.compileForDynamicEngine()
	} else {
		// Hybrid Engine: Optimized for CNN patterns but with validation
		// Validate compatibility and configure for hybrid execution
		return mte.compileForHybridEngine()
	}
}

// compileForDynamicEngine configures the engine for any model architecture
// Dynamic engines are pre-configured during creation and support arbitrary architectures
func (mte *ModelTrainingEngine) compileForDynamicEngine() error {
	// Dynamic engines are already configured with model specifications during creation
	// We just need to validate that the model parameters match the engine setup
	
	if len(mte.parameterTensors) == 0 {
		return fmt.Errorf("no parameter tensors available for dynamic engine")
	}
	
	// Validate parameter count matches model specification
	expectedParams := len(mte.modelSpec.ParameterShapes)
	actualParams := len(mte.parameterTensors)
	
	if expectedParams != actualParams {
		return fmt.Errorf("parameter count mismatch: expected %d, got %d", expectedParams, actualParams)
	}
	
	// Dynamic engine supports any architecture - no additional validation needed
	mte.compiledForModel = true
	return nil
}

// compileForHybridEngine configures the engine for optimized CNN architectures
// Hybrid engines require specific patterns for optimal performance
func (mte *ModelTrainingEngine) compileForHybridEngine() error {
	// Validate that the model is compatible with hybrid CNN optimizations
	if err := mte.validateHybridCNNCompatibility(); err != nil {
		return fmt.Errorf("model not compatible with hybrid CNN engine: %v", err)
	}
	
	// Additional hybrid engine setup could go here
	// For now, the validation is sufficient
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
	
	if mte.isDynamicEngine {
		// Use SGD implementation that matches Adam's resource management approach
		return mte.executeSGDStepDynamicWithGradients(inputTensor, labelTensor, learningRate, mte.gradientTensors)
	} else {
		// Use hybrid engine execution path for legacy compatibility
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
	return mte.executeAdamStepDynamicWithGradients(inputTensor, labelTensor, nil)
}

// executeAdamStepDynamicWithGradients executes Adam optimization with optional pre-allocated gradient tensors
// PERFORMANCE CRITICAL: This eliminates 128MB/step allocation when persistentGradientTensors is provided
func (mte *ModelTrainingEngine) executeAdamStepDynamicWithGradients(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	persistentGradientTensors []*memory.Tensor, // If nil, will allocate (fallback behavior)
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
	
	// Step 2: Use persistent gradient tensors if provided, otherwise allocate
	gradientBuffers := make([]unsafe.Pointer, len(allParameters))
	var gradientTensors []*memory.Tensor
	var allocatedGradients bool

	if persistentGradientTensors != nil && len(persistentGradientTensors) == len(allParameters) {
		// PERFORMANCE OPTIMIZATION: Use pre-allocated persistent gradient tensors
		gradientTensors = persistentGradientTensors
		allocatedGradients = false
		for i, gradTensor := range gradientTensors {
			gradientBuffers[i] = gradTensor.MetalBuffer()
		}
	} else {
		// FALLBACK: Allocate gradient tensors (original behavior for compatibility)
		gradientTensors = make([]*memory.Tensor, len(allParameters))
		allocatedGradients = true
		defer func() {
			if allocatedGradients {
				for _, gradTensor := range gradientTensors {
					if gradTensor != nil {
						gradTensor.Release()
					}
				}
			}
		}()

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
	}

	// Step 3: Compute ACTUAL loss and gradients using MPSGraph automatic differentiation
	// RESOURCE LEAK FIX: Use pooled version if command pooling is enabled
	var actualLoss float32
	
	if mte.MPSTrainingEngine.useCommandPooling && mte.MPSTrainingEngine.commandQueue != nil {
		// Use pooled version with command queue for resource leak prevention
		// fmt.Printf("🚀 Using POOLED execution (commandQueue: %v)\n", mte.MPSTrainingEngine.commandQueue != nil)
		actualLoss, err = cgo_bridge.ExecuteTrainingStepDynamicWithGradientsPooled(
			unsafe.Pointer(mte.MPSTrainingEngine.engine),
			inputTensor.MetalBuffer(),
			labelTensor.MetalBuffer(),
			weightBuffers,
			gradientBuffers,
			batchSize,
			mte.MPSTrainingEngine.commandQueue, // Pass command queue as pool
		)
	} else {
		// Fallback to original version
		fmt.Printf("❌ Using NON-POOLED execution (useCommandPooling: %v, commandQueue: %v)\n", 
			mte.MPSTrainingEngine.useCommandPooling, mte.MPSTrainingEngine.commandQueue != nil)
		actualLoss, err = cgo_bridge.ExecuteTrainingStepDynamicWithGradients(
			unsafe.Pointer(mte.MPSTrainingEngine.engine),
			inputTensor.MetalBuffer(),
			labelTensor.MetalBuffer(),
			weightBuffers,
			gradientBuffers,
			0.001, // learning rate (not used in gradient computation, only for legacy interface)
			batchSize,
		)
	}
	
	if err != nil {
		return 0, fmt.Errorf("gradient computation failed: %v", err)
	}

	// Step 4: Apply Adam optimization with REAL gradients
	// UNIFIED OPTIMIZER: Skip separate Adam step if optimizer is integrated into the graph
	// When using the pooled version with unified optimizer, the parameter updates
	// are already performed within the same MPSGraph execution
	if !mte.MPSTrainingEngine.useCommandPooling {
		// Only run separate Adam step if NOT using unified optimizer
		err = mte.MPSTrainingEngine.adamOptimizer.Step(gradientBuffers)
		if err != nil {
			return 0, fmt.Errorf("Adam optimization step failed: %v", err)
		}
	} else {
		// UNIFIED OPTIMIZER: Parameters already updated in single MPSGraph execution
		// This implements the design-doc.md requirement for single command buffer
		// No separate optimizer step needed
	}
	
	// DEBUG: Log loss to track learning progress
	// fmt.Printf("DEBUG: Loss after Adam step: %.6f\n", actualLoss)
	
	// Return actual computed loss
	return actualLoss, nil
}

// ExecuteModelTrainingStepWithLBFGS executes model training with L-BFGS optimizer
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithLBFGS(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error) {
	if !mte.compiledForModel {
		return 0, fmt.Errorf("model not compiled for execution")
	}
	
	if mte.isDynamicEngine {
		// Dynamic engine uses external L-BFGS optimization with gradient extraction
		return mte.executeLBFGSStepDynamic(inputTensor, labelTensor)
	} else {
		// Hybrid fallback engine - L-BFGS not yet supported for hybrid
		return 0, fmt.Errorf("L-BFGS not yet supported for hybrid engine")
	}
}

// executeLBFGSStepDynamic executes L-BFGS optimization for dynamic engines
func (mte *ModelTrainingEngine) executeLBFGSStepDynamic(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error) {
	return mte.executeLBFGSStepDynamicWithGradients(inputTensor, labelTensor, mte.gradientTensors)
}

// executeLBFGSStepDynamicWithGradients executes L-BFGS optimization with persistent gradient tensors
func (mte *ModelTrainingEngine) executeLBFGSStepDynamicWithGradients(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	persistentGradientTensors []*memory.Tensor,
) (float32, error) {
	// Step 1: Set weight buffers in L-BFGS optimizer
	if mte.MPSTrainingEngine.lbfgsOptimizer == nil {
		return 0, fmt.Errorf("L-BFGS optimizer not initialized for dynamic engine")
	}
	
	// Get all parameter tensors for dynamic execution
	allParameters := mte.getAllParameterTensors()
	weightBuffers := make([]unsafe.Pointer, len(allParameters))
	for i, param := range allParameters {
		weightBuffers[i] = param.MetalBuffer()
	}
	
	err := mte.MPSTrainingEngine.lbfgsOptimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		return 0, fmt.Errorf("failed to set weight buffers in L-BFGS optimizer: %v", err)
	}
	
	// Step 2: Use persistent gradient tensors if provided, otherwise allocate
	gradientBuffers := make([]unsafe.Pointer, len(allParameters))
	var gradientTensors []*memory.Tensor
	var allocatedGradients bool

	if persistentGradientTensors != nil && len(persistentGradientTensors) == len(allParameters) {
		// PERFORMANCE OPTIMIZATION: Use pre-allocated persistent gradient tensors
		gradientTensors = persistentGradientTensors
		allocatedGradients = false
		for i, gradTensor := range gradientTensors {
			gradientBuffers[i] = gradTensor.MetalBuffer()
		}
	} else {
		// Allocate new gradient tensors
		gradientTensors = make([]*memory.Tensor, len(allParameters))
		allocatedGradients = true
		for i, param := range allParameters {
			gradTensor, err := memory.NewTensor(param.Shape(), memory.Float32, memory.GPU)
			if err != nil {
				return 0, fmt.Errorf("failed to create gradient tensor %d: %v", i, err)
			}
			gradientTensors[i] = gradTensor
			gradientBuffers[i] = gradTensor.MetalBuffer()
		}
	}
	
	// Step 3: Execute forward and backward pass to get gradients
	var actualLoss float32
	
	if mte.MPSTrainingEngine.useCommandPooling && mte.MPSTrainingEngine.commandQueue != nil {
		// Use pooled version with command queue for resource leak prevention
		actualLoss, err = cgo_bridge.ExecuteTrainingStepDynamicWithGradientsPooled(
			unsafe.Pointer(mte.MPSTrainingEngine.engine),
			inputTensor.MetalBuffer(),
			labelTensor.MetalBuffer(),
			weightBuffers,
			gradientBuffers,
			inputTensor.Shape()[0], // batch size
			mte.MPSTrainingEngine.commandQueue,
		)
	} else {
		// Use non-pooled version 
		actualLoss, err = cgo_bridge.ExecuteTrainingStepDynamicWithGradients(
			unsafe.Pointer(mte.MPSTrainingEngine.engine),
			inputTensor.MetalBuffer(),
			labelTensor.MetalBuffer(),
			weightBuffers,
			gradientBuffers,
			1.0, // learning rate (not used for L-BFGS since it uses line search)
			inputTensor.Shape()[0], // batch size
		)
	}
	
	if err != nil {
		if allocatedGradients {
			for _, gradTensor := range gradientTensors {
				gradTensor.Release()
			}
		}
		return 0, fmt.Errorf("forward-backward pass failed: %v", err)
	}
	
	// Step 4: Execute L-BFGS optimization step
	// L-BFGS needs loss value for line search
	err = mte.MPSTrainingEngine.lbfgsOptimizer.Step(gradientBuffers, actualLoss)
	if err != nil {
		if allocatedGradients {
			for _, gradTensor := range gradientTensors {
				gradTensor.Release()
			}
		}
		return 0, fmt.Errorf("L-BFGS optimization step failed: %v", err)
	}
	
	// Step 5: Clean up allocated gradient tensors
	if allocatedGradients {
		for _, gradTensor := range gradientTensors {
			gradTensor.Release()
		}
	}
	
	// Return actual computed loss
	return actualLoss, nil
}

// executeSGDStepDynamicWithGradients executes SGD optimization with Adam's resource management approach
// PERFORMANCE CRITICAL: Matches Adam's persistent gradient tensor and pooled command buffer usage
func (mte *ModelTrainingEngine) executeSGDStepDynamicWithGradients(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	learningRate float32,
	persistentGradientTensors []*memory.Tensor, // If nil, will allocate (fallback behavior)
) (float32, error) {
	// Get all parameter tensors for the dynamic engine (matches Adam approach)
	allParameters := mte.getAllParameterTensors()
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
	
	// CRITICAL: Use persistent gradient tensors if provided, otherwise allocate (matches Adam)
	gradientBuffers := make([]unsafe.Pointer, len(allParameters))
	var gradientTensors []*memory.Tensor
	var allocatedGradients bool

	if persistentGradientTensors != nil && len(persistentGradientTensors) == len(allParameters) {
		// PERFORMANCE OPTIMIZATION: Use pre-allocated persistent gradient tensors (Adam approach)
		gradientTensors = persistentGradientTensors
		allocatedGradients = false
		for i, gradTensor := range gradientTensors {
			gradientBuffers[i] = gradTensor.MetalBuffer()
		}
	} else {
		// FALLBACK: Allocate gradient tensors (original behavior for compatibility)
		gradientTensors = make([]*memory.Tensor, len(allParameters))
		allocatedGradients = true
		defer func() {
			if allocatedGradients {
				for _, gradTensor := range gradientTensors {
					if gradTensor != nil {
						gradTensor.Release()
					}
				}
			}
		}()

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
	}

	// CRITICAL: Compute loss and gradients with optimal SGD implementation
	var actualLoss float32
	var err error
	
	// Enable SGD pooled execution for optimal performance
	// The pooled implementation should achieve 11-12 batch/s with proper resource management
	if mte.MPSTrainingEngine.useCommandPooling && mte.MPSTrainingEngine.commandQueue != nil {
		// Use SGD-specific pooled version for optimal performance (11-12 batch/s)
		actualLoss, err = cgo_bridge.ExecuteTrainingStepSGDPooled(
			unsafe.Pointer(mte.MPSTrainingEngine.engine),
			inputTensor.MetalBuffer(),
			labelTensor.MetalBuffer(),
			weightBuffers,
			gradientBuffers,
			learningRate,
			batchSize,
			mte.MPSTrainingEngine.commandQueue, // Pass command queue as pool
		)
	} else {
		// Use non-pooled dynamic execution for SGD (works without placeholder issues)
		actualLoss, err = cgo_bridge.ExecuteTrainingStepDynamicWithGradients(
			mte.MPSTrainingEngine.engine,
			inputTensor.MetalBuffer(),
			labelTensor.MetalBuffer(),
			weightBuffers,
			gradientBuffers,
			learningRate,
			batchSize,
		)
	}
	
	if err != nil {
		return 0, fmt.Errorf("SGD training step failed: %v", err)
	}
	
	// Return actual computed loss
	return actualLoss, nil
}

// getAllParameterTensors returns all parameter tensors for dynamic inference
func (mte *ModelTrainingEngine) getAllParameterTensors() []*memory.Tensor {
	return mte.parameterTensors
}

// getFCLayerParameters extracts ALL fully connected layer parameters for hybrid execution
func (mte *ModelTrainingEngine) getFCLayerParameters() []*memory.Tensor {
	var fcParams []*memory.Tensor
	paramIndex := 0
	
	// Extract ALL Dense layer parameters (FC1, FC2, etc.)
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
			
			// Continue to find more Dense layers (don't return early)
			
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

// getFCParameterShapes extracts the parameter shapes for ALL FC layers
func getFCParameterShapes(modelSpec *layers.ModelSpec) [][]int {
	var fcShapes [][]int
	paramIndex := 0
	
	// Extract ALL Dense layer parameter shapes (FC1, FC2, etc.)
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
			
			// Continue to find more Dense layers (don't return early)
			
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

// IsDynamicEngine returns whether this is a dynamic engine
func (mte *ModelTrainingEngine) IsDynamicEngine() bool {
	return mte.isDynamicEngine
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
	
	// Release gradient tensors
	for _, tensor := range mte.gradientTensors {
		if tensor != nil {
			tensor.Release()
		}
	}
	mte.gradientTensors = nil
	
	// Cleanup base training engine
	if mte.MPSTrainingEngine != nil {
		mte.MPSTrainingEngine.Cleanup()
	}
	
	mte.compiledForModel = false
}

// BatchedTrainingResult represents the result of an optimized batched training step
type BatchedTrainingResult struct {
	Loss     float32
	Accuracy float64 // Only valid if accuracy was calculated
}

// ExecuteModelTrainingStepBatched executes a complete training step with batched CGO operations
// This reduces CGO overhead by combining multiple operations into a single call
// Follows design principle: "Single CGO call per training step"
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepBatched(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	inputData []float32,
	labelData []float32,
	calculateAccuracy bool,
) (*BatchedTrainingResult, error) {
	if !mte.compiledForModel {
		return nil, fmt.Errorf("model not compiled for execution")
	}
	
	// Copy data to GPU tensors
	err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), labelData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy label data to GPU: %v", err)
	}
	
	// Execute training step based on engine type
	var loss float32
	if mte.isDynamicEngine {
		loss, err = mte.executeAdamStepDynamic(inputTensor, labelTensor)
	} else {
		// Check configured optimizer type for hybrid engine
		if mte.MPSTrainingEngine.config.OptimizerType == cgo_bridge.Adam {
			loss, err = mte.ExecuteModelTrainingStepWithAdam(inputTensor, labelTensor)
		} else if mte.MPSTrainingEngine.config.OptimizerType == cgo_bridge.LBFGS {
			loss, err = mte.ExecuteModelTrainingStepWithLBFGS(inputTensor, labelTensor)
		} else {
			// SGD optimizer - use regular training step with learning rate
			loss, err = mte.ExecuteModelTrainingStep(inputTensor, labelTensor, mte.MPSTrainingEngine.config.LearningRate)
		}
	}
	
	if err != nil {
		return nil, fmt.Errorf("training step failed: %v", err)
	}
	
	result := &BatchedTrainingResult{
		Loss:     loss,
		Accuracy: 0.0, // Will be calculated if requested
	}
	
	// Calculate accuracy if requested
	if calculateAccuracy {
		inferenceResult, err := mte.ExecuteInference(inputTensor, inputTensor.Shape()[0])
		if err != nil {
			// Non-fatal: return loss without accuracy
			return result, nil
		}
		
		// Extract labels from label tensor for accuracy calculation
		labelShape := labelTensor.Shape()
		batchSize := labelShape[0]
		numClasses := labelShape[1]
		
		// Convert one-hot back to class indices for accuracy calculation
		classLabels := make([]int32, batchSize)
		for i := 0; i < batchSize; i++ {
			maxIdx := 0
			for j := 1; j < numClasses; j++ {
				labelIdx := i*numClasses + j
				if labelIdx < len(labelData) && labelData[labelIdx] > labelData[i*numClasses+maxIdx] {
					maxIdx = j
				}
			}
			classLabels[i] = int32(maxIdx)
		}
		
		// Calculate accuracy using existing method
		accuracy := mte.calculateAccuracyFromPredictions(
			inferenceResult.Predictions,
			classLabels,
			batchSize,
			numClasses,
		)
		result.Accuracy = accuracy
	}
	
	return result, nil
}

// ExecuteModelTrainingStepBatchedPersistent executes a training step using persistent GPU buffers
// This provides maximum performance by eliminating per-step tensor allocations
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepBatchedPersistent(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	inputData []float32,
	labelData []float32,
	calculateAccuracy bool,
) (*BatchedTrainingResult, error) {
	return mte.ExecuteModelTrainingStepBatchedPersistentWithGradients(
		inputTensor, labelTensor, inputData, labelData, calculateAccuracy, nil)
}

// ExecuteModelTrainingStepBatchedPersistentWithGradients executes training with pre-allocated gradient tensors
// This eliminates the 128MB/step gradient allocation that caused 83% performance degradation
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepBatchedPersistentWithGradients(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	inputData []float32,
	labelData []float32,
	calculateAccuracy bool,
	persistentGradientTensors []*memory.Tensor, // Pre-allocated gradient tensors
) (*BatchedTrainingResult, error) {
	if !mte.compiledForModel {
		return nil, fmt.Errorf("model not compiled for execution")
	}
	
	// Copy data to persistent GPU tensors (maximum performance)
	err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to persistent GPU buffer: %v", err)
	}
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), labelData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy label data to persistent GPU buffer: %v", err)
	}
	
	// Execute training step based on engine type and optimizer
	var loss float32
	if mte.isDynamicEngine {
		// Use persistent gradients if provided, otherwise allocate (fallback)
		loss, err = mte.executeAdamStepDynamicWithGradients(inputTensor, labelTensor, persistentGradientTensors)
	} else {
		// Check configured optimizer type for hybrid engine
		if mte.MPSTrainingEngine.config.OptimizerType == cgo_bridge.Adam {
			loss, err = mte.ExecuteModelTrainingStepWithAdam(inputTensor, labelTensor)
		} else if mte.MPSTrainingEngine.config.OptimizerType == cgo_bridge.LBFGS {
			loss, err = mte.ExecuteModelTrainingStepWithLBFGS(inputTensor, labelTensor)
		} else {
			// SGD optimizer - use regular training step with learning rate
			loss, err = mte.ExecuteModelTrainingStep(inputTensor, labelTensor, mte.MPSTrainingEngine.config.LearningRate)
		}
	}
	
	if err != nil {
		return nil, fmt.Errorf("persistent training step failed: %v", err)
	}
	
	result := &BatchedTrainingResult{
		Loss:     loss,
		Accuracy: 0.0, // Will be calculated if requested
	}
	
	// Calculate accuracy if requested (using persistent buffers)
	if calculateAccuracy {
		inferenceResult, err := mte.ExecuteInference(inputTensor, inputTensor.Shape()[0])
		if err != nil {
			// Non-fatal: return loss without accuracy
			return result, nil
		}
		
		// Extract labels from label tensor for accuracy calculation
		labelShape := labelTensor.Shape()
		batchSize := labelShape[0]
		numClasses := labelShape[1]
		
		// Convert one-hot back to class indices for accuracy calculation
		classLabels := make([]int32, batchSize)
		for i := 0; i < batchSize; i++ {
			maxIdx := 0
			for j := 1; j < numClasses; j++ {
				labelIdx := i*numClasses + j
				if labelIdx < len(labelData) && labelData[labelIdx] > labelData[i*numClasses+maxIdx] {
					maxIdx = j
				}
			}
			classLabels[i] = int32(maxIdx)
		}
		
		// Calculate accuracy using existing method
		accuracy := mte.calculateAccuracyFromPredictions(
			inferenceResult.Predictions,
			classLabels,
			batchSize,
			numClasses,
		)
		result.Accuracy = accuracy
	}
	
	return result, nil
}

// calculateAccuracyFromPredictions computes accuracy from inference results and true labels
func (mte *ModelTrainingEngine) calculateAccuracyFromPredictions(
	predictions []float32,
	trueLabels []int32,
	batchSize int,
	numClasses int,
) float64 {
	if len(predictions) != batchSize*numClasses {
		return 0.0 // Invalid predictions array
	}
	
	if len(trueLabels) != batchSize {
		return 0.0 // Invalid labels array
	}
	
	correctPredictions := 0
	
	for i := 0; i < batchSize; i++ {
		// Find predicted class (argmax)
		maxIdx := 0
		maxVal := predictions[i*numClasses]
		
		for j := 1; j < numClasses; j++ {
			if predictions[i*numClasses+j] > maxVal {
				maxVal = predictions[i*numClasses+j]
				maxIdx = j
			}
		}
		
		// Check if prediction matches true label
		if int32(maxIdx) == trueLabels[i] {
			correctPredictions++
		}
	}
	
	return float64(correctPredictions) / float64(batchSize)
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
	if mte.isDynamicEngine {
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
		mte.isDynamicEngine, // Pass correct dynamic engine flag
	)
	
	if err != nil {
		return nil, fmt.Errorf("inference execution failed: %v", err)
	}
	
	return result, nil
}

// createModelConfigFromSpec creates a ModelConfig from a ModelSpec by analyzing layer architectures
func createModelConfigFromSpec(modelSpec *layers.ModelSpec) (cgo_bridge.ModelConfig, error) {
	inputShape := modelSpec.InputShape
	if len(inputShape) != 4 {
		return cgo_bridge.ModelConfig{}, fmt.Errorf("expected 4D input shape [batch, channels, height, width], got %v", inputShape)
	}
	
	// Extract input dimensions
	batchSize := inputShape[0]
	inputChannels := inputShape[1]
	inputHeight := inputShape[2]
	inputWidth := inputShape[3]
	
	// Analyze the actual model layers to extract architecture
	convLayers := []interface{}{}
	fcLayers := []interface{}{}
	
	// Parse layers to find conv and dense layers
	for _, layer := range modelSpec.Layers {
		switch layer.Type {
		case layers.Conv2D:
			convLayers = append(convLayers, layer)
		case layers.Dense:
			fcLayers = append(fcLayers, layer)
		}
	}
	
	// Hybrid Engine optimizations are designed for 3 conv + 2 FC pattern
	// For other architectures, recommend Dynamic Engine instead
	if len(convLayers) != 3 || len(fcLayers) != 2 {
		return cgo_bridge.ModelConfig{}, fmt.Errorf("Hybrid Engine is optimized for 3 Conv2D + 2 Dense layers (got %d conv, %d dense). Consider using Dynamic Engine for flexible architectures", len(convLayers), len(fcLayers))
	}
	
	// Extract actual layer parameters from the model spec using the correct API
	conv1Idx := findLayerIndex(modelSpec, layers.Conv2D, 0)
	conv2Idx := findLayerIndex(modelSpec, layers.Conv2D, 1)
	conv3Idx := findLayerIndex(modelSpec, layers.Conv2D, 2)
	fc1Idx := findLayerIndex(modelSpec, layers.Dense, 0)
	fc2Idx := findLayerIndex(modelSpec, layers.Dense, 1)
	
	if conv1Idx == -1 || conv2Idx == -1 || conv3Idx == -1 || fc1Idx == -1 || fc2Idx == -1 {
		return cgo_bridge.ModelConfig{}, fmt.Errorf("could not find required layers in model spec")
	}
	
	conv1 := modelSpec.Layers[conv1Idx]
	conv2 := modelSpec.Layers[conv2Idx]
	conv3 := modelSpec.Layers[conv3Idx]
	fc1 := modelSpec.Layers[fc1Idx]
	fc2 := modelSpec.Layers[fc2Idx]
	
	// Extract Conv2D parameters using the Parameters map
	conv1OutChannels := conv1.Parameters["output_channels"].(int)
	conv1KernelSize := conv1.Parameters["kernel_size"].(int)
	conv1Stride := conv1.Parameters["stride"].(int)
	conv1Padding := conv1.Parameters["padding"].(int)
	conv1OutHeight := (inputHeight + 2*conv1Padding - conv1KernelSize) / conv1Stride + 1
	conv1OutWidth := (inputWidth + 2*conv1Padding - conv1KernelSize) / conv1Stride + 1
	
	conv2OutChannels := conv2.Parameters["output_channels"].(int)
	conv2KernelSize := conv2.Parameters["kernel_size"].(int)
	conv2Stride := conv2.Parameters["stride"].(int)
	conv2Padding := conv2.Parameters["padding"].(int)
	conv2OutHeight := (conv1OutHeight + 2*conv2Padding - conv2KernelSize) / conv2Stride + 1
	conv2OutWidth := (conv1OutWidth + 2*conv2Padding - conv2KernelSize) / conv2Stride + 1
	
	conv3OutChannels := conv3.Parameters["output_channels"].(int)
	conv3KernelSize := conv3.Parameters["kernel_size"].(int)
	conv3Stride := conv3.Parameters["stride"].(int)
	conv3Padding := conv3.Parameters["padding"].(int)
	conv3OutHeight := (conv2OutHeight + 2*conv3Padding - conv3KernelSize) / conv3Stride + 1
	conv3OutWidth := (conv2OutWidth + 2*conv3Padding - conv3KernelSize) / conv3Stride + 1
	
	// Extract Dense layer parameters
	fc1InputSize := conv3OutChannels * conv3OutHeight * conv3OutWidth
	fc1OutputSize := fc1.Parameters["output_size"].(int)
	fc2OutputSize := fc2.Parameters["output_size"].(int)
	
	modelConfig := cgo_bridge.ModelConfig{
		BatchSize:     batchSize,
		InputChannels: inputChannels,
		InputHeight:   inputHeight,
		InputWidth:    inputWidth,
		
		Conv1OutChannels: conv1OutChannels,
		Conv1OutHeight:   conv1OutHeight,
		Conv1OutWidth:    conv1OutWidth,
		Conv1KernelSize:  conv1KernelSize,
		Conv1Stride:      conv1Stride,
		
		Conv2OutChannels: conv2OutChannels,
		Conv2OutHeight:   conv2OutHeight,
		Conv2OutWidth:    conv2OutWidth,
		Conv2KernelSize:  conv2KernelSize,
		Conv2Stride:      conv2Stride,
		
		Conv3OutChannels: conv3OutChannels,
		Conv3OutHeight:   conv3OutHeight,
		Conv3OutWidth:    conv3OutWidth,
		Conv3KernelSize:  conv3KernelSize,
		Conv3Stride:      conv3Stride,
		
		FC1InputSize:  fc1InputSize,
		FC1OutputSize: fc1OutputSize,
		FC2OutputSize: fc2OutputSize,
	}
	
	return modelConfig, nil
}

// Helper function to find the nth layer of a specific type
func findLayerIndex(modelSpec *layers.ModelSpec, layerType layers.LayerType, occurrence int) int {
	count := 0
	for i, layer := range modelSpec.Layers {
		if layer.Type == layerType {
			if count == occurrence {
				return i
			}
			count++
		}
	}
	return -1 // Not found
}

// GetOptimizerState extracts the current optimizer state for checkpointing
// This method bridges between the CGO-level optimizer and the Go optimizer interface
func (mte *ModelTrainingEngine) GetOptimizerState() (*optimizer.OptimizerState, error) {
	// Check if we have an optimizer
	if mte.MPSTrainingEngine == nil {
		return nil, fmt.Errorf("training engine not initialized")
	}
	
	// Currently, the optimizer is managed at the CGO level
	// We need to extract the state based on the optimizer type
	config := mte.MPSTrainingEngine.GetConfig()
	
	switch config.OptimizerType {
	case cgo_bridge.Adam:
		if mte.MPSTrainingEngine.adamOptimizer != nil {
			return mte.MPSTrainingEngine.adamOptimizer.GetState()
		}
		return nil, fmt.Errorf("Adam optimizer not initialized")
		
	case cgo_bridge.RMSProp:
		if mte.MPSTrainingEngine.rmspropOptimizer != nil {
			return mte.MPSTrainingEngine.rmspropOptimizer.GetState()
		}
		return nil, fmt.Errorf("RMSProp optimizer not initialized")
		
	case cgo_bridge.SGD:
		if mte.MPSTrainingEngine.sgdOptimizer != nil {
			return mte.MPSTrainingEngine.sgdOptimizer.GetState()
		}
		return nil, fmt.Errorf("SGD optimizer not initialized")
		
	default:
		return nil, fmt.Errorf("unsupported optimizer type: %v", config.OptimizerType)
	}
}

// SetOptimizerState restores optimizer state from a checkpoint
// This method bridges between the Go optimizer interface and the CGO-level optimizer
func (mte *ModelTrainingEngine) SetOptimizerState(state *optimizer.OptimizerState) error {
	if state == nil {
		return fmt.Errorf("optimizer state is nil")
	}
	
	// Check if we have an optimizer
	if mte.MPSTrainingEngine == nil {
		return fmt.Errorf("training engine not initialized")
	}
	
	// Restore state based on the optimizer type
	config := mte.MPSTrainingEngine.GetConfig()
	
	// Validate state type matches current optimizer
	expectedType := ""
	switch config.OptimizerType {
	case cgo_bridge.Adam:
		expectedType = "Adam"
	case cgo_bridge.RMSProp:
		expectedType = "RMSProp"
	case cgo_bridge.SGD:
		expectedType = "SGD"
	default:
		return fmt.Errorf("unsupported optimizer type: %v", config.OptimizerType)
	}
	
	if state.Type != expectedType {
		return fmt.Errorf("optimizer type mismatch: expected %s, got %s", expectedType, state.Type)
	}
	
	// Restore state to the appropriate optimizer
	switch config.OptimizerType {
	case cgo_bridge.Adam:
		if mte.MPSTrainingEngine.adamOptimizer != nil {
			return mte.MPSTrainingEngine.adamOptimizer.LoadState(state)
		}
		return fmt.Errorf("Adam optimizer not initialized")
		
	case cgo_bridge.RMSProp:
		if mte.MPSTrainingEngine.rmspropOptimizer != nil {
			return mte.MPSTrainingEngine.rmspropOptimizer.LoadState(state)
		}
		return fmt.Errorf("RMSProp optimizer not initialized")
		
	case cgo_bridge.SGD:
		if mte.MPSTrainingEngine.sgdOptimizer != nil {
			return mte.MPSTrainingEngine.sgdOptimizer.LoadState(state)
		}
		return fmt.Errorf("SGD optimizer not initialized")
		
	default:
		return fmt.Errorf("unsupported optimizer type: %v", config.OptimizerType)
	}
}

// UpdateLearningRate updates the learning rate for the current optimizer
// This method bridges between the model trainer and the optimizer implementations
func (mte *ModelTrainingEngine) UpdateLearningRate(newLR float32) error {
	if mte.MPSTrainingEngine == nil {
		return fmt.Errorf("training engine not initialized")
	}
	
	// Update the config first
	mte.MPSTrainingEngine.config.LearningRate = newLR
	
	// Get the optimizer type and update the appropriate optimizer
	config := mte.MPSTrainingEngine.GetConfig()
	
	switch config.OptimizerType {
	case cgo_bridge.Adam:
		if mte.MPSTrainingEngine.adamOptimizer != nil {
			mte.MPSTrainingEngine.adamOptimizer.UpdateLearningRate(newLR)
			return nil
		}
		return fmt.Errorf("Adam optimizer not initialized")
		
	case cgo_bridge.RMSProp:
		if mte.MPSTrainingEngine.rmspropOptimizer != nil {
			mte.MPSTrainingEngine.rmspropOptimizer.UpdateLearningRate(newLR)
			return nil
		}
		return fmt.Errorf("RMSProp optimizer not initialized")
		
	case cgo_bridge.SGD:
		if mte.MPSTrainingEngine.sgdOptimizer != nil {
			mte.MPSTrainingEngine.sgdOptimizer.UpdateLearningRate(newLR)
			return nil
		}
		return fmt.Errorf("SGD optimizer not initialized")
		
	case cgo_bridge.LBFGS:
		if mte.MPSTrainingEngine.lbfgsOptimizer != nil {
			mte.MPSTrainingEngine.lbfgsOptimizer.UpdateLearningRate(newLR)
			return nil
		}
		return fmt.Errorf("L-BFGS optimizer not initialized")
		
	default:
		return fmt.Errorf("unsupported optimizer type: %v", config.OptimizerType)
	}
}