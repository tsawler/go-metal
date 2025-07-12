package engine

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// MPSInferenceEngine handles inference execution using MPSGraph
// Optimized for forward-pass only without loss computation or gradients
type MPSInferenceEngine struct {
	device       unsafe.Pointer           // MTLDevice
	engine       unsafe.Pointer           // Native inference engine
	config       cgo_bridge.InferenceConfig
	initialized  bool
	isDynamic    bool                     // True if using dynamic engine
	
	// Resource management following design principles
	commandQueue unsafe.Pointer           // MTLCommandQueue for command buffer creation
	useCommandPooling bool                // Flag to enable command buffer pooling
}

// NewMPSInferenceEngine creates a new inference-only engine
func NewMPSInferenceEngine(config cgo_bridge.InferenceConfig) (*MPSInferenceEngine, error) {
	// Create Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal device: %v", err)
	}
	
	// Initialize global memory manager (GPU-resident everything principle)
	memory.InitializeGlobalMemoryManager(device)
	
	// Create inference engine (dedicated for forward-pass only)
	engine, err := cgo_bridge.CreateInferenceEngine(device, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create inference engine: %v", err)
	}
	
	// Initialize command queue for resource management
	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		cgo_bridge.DestroyInferenceEngine(engine)
		return nil, fmt.Errorf("failed to create command queue: %v", err)
	}
	
	return &MPSInferenceEngine{
		device:            device,
		engine:            engine,
		config:            config,
		initialized:       true,
		isDynamic:         config.UseDynamicEngine,
		commandQueue:      commandQueue,
		useCommandPooling: true, // Enable by default for performance
	}, nil
}

// Cleanup performs deterministic resource cleanup (reference counting principle)
func (ie *MPSInferenceEngine) Cleanup() {
	if ie.engine != nil {
		cgo_bridge.DestroyInferenceEngine(ie.engine)
		ie.engine = nil
	}
	
	if ie.commandQueue != nil {
		cgo_bridge.DestroyCommandQueue(ie.commandQueue)
		ie.commandQueue = nil
	}
	
	ie.initialized = false
}

// ModelInferenceEngine extends MPSInferenceEngine with layer-based model support
// Optimized for inference without training overhead
type ModelInferenceEngine struct {
	*MPSInferenceEngine
	modelSpec       *layers.ModelSpec
	parameterTensors []*memory.Tensor
	compiledForModel bool
	batchNormInferenceMode bool // Handle batch normalization in inference mode
}

// NewModelInferenceEngine creates a model-based inference engine
func NewModelInferenceEngine(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.InferenceConfig,
) (*ModelInferenceEngine, error) {
	// Validate model for inference (less strict than training validation)
	if err := modelSpec.ValidateModelForInference(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}
	
	// Convert model to inference-optimized layer specifications
	inferenceSpecs, err := modelSpec.ConvertToInferenceLayerSpecs()
	if err != nil {
		return nil, fmt.Errorf("failed to convert model to inference specs: %v", err)
	}
	
	// Convert to CGO-compatible format
	cgoLayerSpecs := make([]cgo_bridge.LayerSpecC, len(inferenceSpecs))
	for i, spec := range inferenceSpecs {
		cgoLayerSpecs[i] = cgo_bridge.LayerSpecC{
			LayerType:       spec.LayerType,
			Name:            spec.NameBytes,
			InputShape:      spec.InputShape,
			InputShapeLen:   spec.InputShapeLen,
			OutputShape:     spec.OutputShape,
			OutputShapeLen:  spec.OutputShapeLen,
			ParamFloat:      spec.ParamFloat,
			ParamFloatCount: spec.ParamFloatCount,
			ParamInt:        spec.ParamInt,
			ParamIntCount:   spec.ParamIntCount,
			// ARCHITECTURAL FIX: Copy running statistics for flexible normalization
			RunningMean:     spec.RunningMean,
			RunningVar:      spec.RunningVar,
			RunningStatsSize: int32(len(spec.RunningMean)),
			HasRunningStats:  func() int32 { if spec.HasRunningStats { return 1 }; return 0 }(),
		}
	}
	
	// Update config with layer specifications
	config.LayerSpecs = cgoLayerSpecs
	config.LayerSpecsLen = int32(len(cgoLayerSpecs))
	
	// Convert input shape from []int to []int32
	inputShape := make([]int32, len(modelSpec.InputShape))
	for i, dim := range modelSpec.InputShape {
		inputShape[i] = int32(dim)
	}
	config.InputShape = inputShape
	config.InputShapeLen = int32(len(modelSpec.InputShape))
	
	// Create base inference engine
	baseEngine, err := NewMPSInferenceEngine(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create base inference engine: %v", err)
	}
	
	mie := &ModelInferenceEngine{
		MPSInferenceEngine:     baseEngine,
		modelSpec:              modelSpec,
		batchNormInferenceMode: true, // Always use inference mode for batch norm
	}
	
	// Initialize parameter tensors (GPU-resident)
	if err := mie.initializeParameterTensors(); err != nil {
		mie.Cleanup()
		return nil, fmt.Errorf("failed to initialize parameter tensors: %v", err)
	}
	
	// Compile model for inference execution
	if err := mie.compileModelForInference(); err != nil {
		mie.Cleanup()
		return nil, fmt.Errorf("failed to compile model for inference: %v", err)
	}
	
	return mie, nil
}

// Cleanup performs complete resource cleanup
func (mie *ModelInferenceEngine) Cleanup() {
	// Release parameter tensors
	for _, tensor := range mie.parameterTensors {
		if tensor != nil {
			tensor.Release()
		}
	}
	mie.parameterTensors = nil
	
	// Cleanup base engine
	if mie.MPSInferenceEngine != nil {
		mie.MPSInferenceEngine.Cleanup()
	}
}

// LoadWeights loads pre-trained weights into the inference engine
func (mie *ModelInferenceEngine) LoadWeights(weights []checkpoints.WeightTensor) error {
	if !mie.compiledForModel {
		return fmt.Errorf("model not compiled for inference")
	}
	
	// UNIFIED SOLUTION: Filter out running statistics - only load learnable parameters
	// Running statistics are handled separately by the inference execution engine
	learnableWeights := make([]checkpoints.WeightTensor, 0, len(weights))
	runningStatsWeights := make([]checkpoints.WeightTensor, 0)
	
	for _, weight := range weights {
		if weight.Type == "running_mean" || weight.Type == "running_var" {
			runningStatsWeights = append(runningStatsWeights, weight)
		} else {
			// This is a learnable parameter (weight, bias, etc.)
			learnableWeights = append(learnableWeights, weight)
		}
	}
	
	fmt.Printf("Total weights: %d, Learnable: %d, Running stats: %d, Expected parameters: %d\n", 
		len(weights), len(learnableWeights), len(runningStatsWeights), len(mie.parameterTensors))
	
	if len(learnableWeights) != len(mie.parameterTensors) {
		return fmt.Errorf("weight count mismatch: %d weights, %d tensors", 
			len(learnableWeights), len(mie.parameterTensors))
	}
	
	// Load only learnable parameters into GPU tensors (GPU-resident principle)
	for i, weight := range learnableWeights {
		if i >= len(mie.parameterTensors) {
			break
		}
		
		tensor := mie.parameterTensors[i]
		if tensor == nil {
			continue
		}
		
		// Copy weights to GPU tensor (minimal CPU-GPU transfers)
		err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(
			tensor.MetalBuffer(),
			weight.Data,
		)
		if err != nil {
			return fmt.Errorf("failed to load weight %d: %v", i, err)
		}
	}
	
	// FLEXIBLE NORMALIZATION: Handle running statistics as separate from learnable parameters
	// Running statistics (mean/var) are stored in layer specifications for inference
	// This allows ANY model to specify custom normalization values rather than hardcoded defaults
	if len(runningStatsWeights) > 0 {
		fmt.Printf("Loading %d running statistics into layer specifications...\n", len(runningStatsWeights))
		if err := mie.loadRunningStatistics(runningStatsWeights); err != nil {
			return fmt.Errorf("failed to load running statistics: %v", err)
		}
	}
	
	return nil
}

// Predict performs single forward pass for inference
// Optimized for single-image or small batch inference
func (mie *ModelInferenceEngine) Predict(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error) {
	// Validate inputs
	if len(inputData) == 0 {
		return nil, fmt.Errorf("input data is empty")
	}
	
	if len(inputShape) < 2 {
		return nil, fmt.Errorf("input shape must have at least 2 dimensions, got %v", inputShape)
	}
	
	batchSize := inputShape[0]
	if batchSize <= 0 {
		return nil, fmt.Errorf("invalid batch size: %d", batchSize)
	}
	
	// Create input tensor and copy data to GPU (GPU-resident everything principle)
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Release()
	
	// Copy input data to GPU (minimal CPU-GPU transfers)
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(
		inputTensor.MetalBuffer(), 
		inputData,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	// Execute inference using single CGO call (design compliant)
	return mie.executeInference(inputTensor, batchSize)
}

// executeInference performs the actual inference execution
func (mie *ModelInferenceEngine) executeInference(
	inputTensor *memory.Tensor,
	batchSize int,
) (*cgo_bridge.InferenceResult, error) {
	// DEBUG: Log inference preparation
	fmt.Printf("\n=== ModelInferenceEngine.executeInference Debug ===\n")
	fmt.Printf("Model compiled: %v\n", mie.compiledForModel)
	fmt.Printf("Input tensor shape: %v\n", inputTensor.Shape())
	fmt.Printf("Input tensor buffer: %p\n", inputTensor.MetalBuffer())
	fmt.Printf("Batch size: %d\n", batchSize)
	fmt.Printf("Is dynamic engine: %v\n", mie.MPSInferenceEngine.isDynamic)
	fmt.Printf("Batch norm inference mode: %v\n", mie.batchNormInferenceMode)
	fmt.Printf("Number of parameter tensors: %d\n", len(mie.parameterTensors))
	
	// Validate model is compiled
	if !mie.compiledForModel {
		return nil, fmt.Errorf("model not compiled for inference")
	}
	
	// Validate input tensor
	if inputTensor == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}
	
	if batchSize <= 0 {
		return nil, fmt.Errorf("invalid batch size: %d", batchSize)
	}
	
	// Extract parameter buffers (coarse-grained interface principle)
	weightBuffers := make([]unsafe.Pointer, len(mie.parameterTensors))
	for i, tensor := range mie.parameterTensors {
		if tensor != nil {
			weightBuffers[i] = tensor.MetalBuffer()
			fmt.Printf("Parameter tensor[%d]: shape=%v, buffer=%p\n", i, tensor.Shape(), tensor.MetalBuffer())
		} else {
			fmt.Printf("Parameter tensor[%d]: nil\n", i)
		}
	}
	
	// Get output shape for predictions
	outputShape := mie.modelSpec.OutputShape
	if len(outputShape) < 2 {
		return nil, fmt.Errorf("invalid model output shape: %v", outputShape)
	}
	numClasses := outputShape[len(outputShape)-1] // Last dimension is number of classes
	
	fmt.Printf("Model output shape: %v\n", outputShape)
	fmt.Printf("Number of classes: %d\n", numClasses)
	fmt.Printf("About to call ExecuteInferenceOnly...\n")
	
	// Single CGO call for complete inference (design compliant)
	result, err := cgo_bridge.ExecuteInferenceOnly(
		mie.MPSInferenceEngine.engine,
		inputTensor.MetalBuffer(),
		weightBuffers,
		batchSize,
		numClasses,
		mie.MPSInferenceEngine.isDynamic,
		mie.batchNormInferenceMode, // Pass batch norm inference flag
	)
	
	if err != nil {
		fmt.Printf("ERROR: ExecuteInferenceOnly failed: %v\n", err)
		return nil, fmt.Errorf("inference execution failed: %v", err)
	}
	
	fmt.Printf("ExecuteInferenceOnly succeeded, result: %+v\n", result)
	fmt.Printf("=== End ModelInferenceEngine.executeInference Debug ===\n\n")
	
	return result, nil
}

// initializeParameterTensors creates GPU-resident parameter tensors
func (mie *ModelInferenceEngine) initializeParameterTensors() error {
	// Count total parameters needed
	totalParams := 0
	for _, layer := range mie.modelSpec.Layers {
		totalParams += len(layer.ParameterShapes)
	}
	
	// Pre-allocate parameter tensors (arena allocation principle)
	mie.parameterTensors = make([]*memory.Tensor, totalParams)
	
	paramIdx := 0
	for _, layer := range mie.modelSpec.Layers {
		for _, paramShape := range layer.ParameterShapes {
			// Create GPU-resident parameter tensor
			tensor, err := memory.NewTensor(paramShape, memory.Float32, memory.GPU)
			if err != nil {
				return fmt.Errorf("failed to create parameter tensor %d: %v", paramIdx, err)
			}
			
			mie.parameterTensors[paramIdx] = tensor
			paramIdx++
		}
	}
	
	return nil
}

// compileModelForInference builds the inference graph
func (mie *ModelInferenceEngine) compileModelForInference() error {
	// Build inference-optimized MPSGraph
	err := cgo_bridge.BuildInferenceGraph(
		mie.MPSInferenceEngine.engine,
		mie.modelSpec.InputShape,
		int32(len(mie.modelSpec.InputShape)),
		mie.batchNormInferenceMode,
	)
	if err != nil {
		return fmt.Errorf("failed to build inference graph: %v", err)
	}
	
	mie.compiledForModel = true
	return nil
}

// GetParameterTensors returns the GPU-resident parameter tensors
func (mie *ModelInferenceEngine) GetParameterTensors() []*memory.Tensor {
	return mie.parameterTensors
}

// GetModelSpec returns the model specification
func (mie *ModelInferenceEngine) GetModelSpec() *layers.ModelSpec {
	return mie.modelSpec
}

// loadRunningStatistics loads BatchNorm running statistics for flexible normalization
// This resolves the hardcoded mean=0, var=1 limitation
func (mie *ModelInferenceEngine) loadRunningStatistics(runningStatsWeights []checkpoints.WeightTensor) error {
	// Create a map of layer name to running statistics for fast lookup
	runningStatsMap := make(map[string]map[string][]float32)
	
	for _, weight := range runningStatsWeights {
		layerName := weight.Layer
		statType := weight.Type // "running_mean" or "running_var"
		
		if runningStatsMap[layerName] == nil {
			runningStatsMap[layerName] = make(map[string][]float32)
		}
		runningStatsMap[layerName][statType] = weight.Data
		
		fmt.Printf("Loaded %s for layer %s: %d values\n", statType, layerName, len(weight.Data))
	}
	
	// Update model spec layers with running statistics
	// This ensures the values are available for subsequent conversion to inference specs
	for i := range mie.modelSpec.Layers {
		layer := &mie.modelSpec.Layers[i]
		
		// Only process BatchNorm layers
		if layer.Type == layers.BatchNorm {
			layerStats, exists := runningStatsMap[layer.Name]
			if exists {
				// Initialize RunningStatistics map if needed
				if layer.RunningStatistics == nil {
					layer.RunningStatistics = make(map[string][]float32)
				}
				
				// Copy running mean if available
				if runningMean, hasMean := layerStats["running_mean"]; hasMean {
					layer.RunningStatistics["running_mean"] = make([]float32, len(runningMean))
					copy(layer.RunningStatistics["running_mean"], runningMean)
					fmt.Printf("Updated layer %s with running_mean: %v\n", layer.Name, runningMean[:min(5, len(runningMean))])
				}
				
				// Copy running variance if available
				if runningVar, hasVar := layerStats["running_var"]; hasVar {
					layer.RunningStatistics["running_var"] = make([]float32, len(runningVar))
					copy(layer.RunningStatistics["running_var"], runningVar)
					fmt.Printf("Updated layer %s with running_var: %v\n", layer.Name, runningVar[:min(5, len(runningVar))])
				}
			} else {
				fmt.Printf("Warning: No running statistics found for BatchNorm layer %s, using defaults\n", layer.Name)
			}
		}
	}
	
	// IMPORTANT: Recompile the model for inference with updated running statistics
	// This ensures the new normalization values are used
	return mie.recompileForInference()
}

// recompileForInference rebuilds the inference graph with updated running statistics
func (mie *ModelInferenceEngine) recompileForInference() error {
	// Convert model to inference specs with updated running statistics
	inferenceSpecs, err := mie.modelSpec.ConvertToInferenceLayerSpecs()
	if err != nil {
		return fmt.Errorf("failed to convert model to inference specs: %v", err)
	}
	
	// Convert to CGO-compatible format with running statistics
	cgoLayerSpecs := make([]cgo_bridge.LayerSpecC, len(inferenceSpecs))
	for i, spec := range inferenceSpecs {
		cgoLayerSpecs[i] = cgo_bridge.LayerSpecC{
			LayerType:       spec.LayerType,
			Name:            spec.NameBytes,
			InputShape:      spec.InputShape,
			InputShapeLen:   spec.InputShapeLen,
			OutputShape:     spec.OutputShape,
			OutputShapeLen:  spec.OutputShapeLen,
			ParamFloat:      spec.ParamFloat,
			ParamFloatCount: spec.ParamFloatCount,
			ParamInt:        spec.ParamInt,
			ParamIntCount:   spec.ParamIntCount,
			// Include updated running statistics
			RunningMean:     spec.RunningMean,
			RunningVar:      spec.RunningVar,
			RunningStatsSize: int32(len(spec.RunningMean)),
			HasRunningStats:  func() int32 { if spec.HasRunningStats { return 1 }; return 0 }(),
		}
		
		// Log running statistics for debugging
		if spec.HasRunningStats {
			fmt.Printf("Layer %d (%s): RunningMean=%v, RunningVar=%v\n", 
				i, string(spec.NameBytes[:]), 
				spec.RunningMean[:min(3, len(spec.RunningMean))], 
				spec.RunningVar[:min(3, len(spec.RunningVar))])
		}
	}
	
	// Update inference configuration with new layer specs
	config := mie.MPSInferenceEngine.config
	config.LayerSpecs = cgoLayerSpecs
	config.LayerSpecsLen = int32(len(cgoLayerSpecs))
	
	// Rebuild inference graph with updated specifications
	err = cgo_bridge.BuildInferenceGraph(
		mie.MPSInferenceEngine.engine,
		mie.modelSpec.InputShape,
		int32(len(mie.modelSpec.InputShape)),
		mie.batchNormInferenceMode,
	)
	if err != nil {
		return fmt.Errorf("failed to rebuild inference graph: %v", err)
	}
	
	fmt.Println("Successfully recompiled inference engine with custom normalization values")
	return nil
}

// SetCustomNormalization allows setting custom normalization values for any model
// This enables the library to work with ANY model architecture and normalization scheme
// layerName: name of the BatchNorm layer to update
// mean, variance: custom normalization values (must match layer's num_features)
func (mie *ModelInferenceEngine) SetCustomNormalization(layerName string, mean, variance []float32) error {
	if len(mean) != len(variance) {
		return fmt.Errorf("mean and variance must have the same length")
	}
	
	// Find the specified layer
	var targetLayer *layers.LayerSpec
	for i := range mie.modelSpec.Layers {
		layer := &mie.modelSpec.Layers[i]
		if layer.Name == layerName && layer.Type == layers.BatchNorm {
			targetLayer = layer
			break
		}
	}
	
	if targetLayer == nil {
		return fmt.Errorf("BatchNorm layer '%s' not found", layerName)
	}
	
	// Validate the length matches num_features
	numFeatures, ok := targetLayer.Parameters["num_features"].(int)
	if !ok {
		return fmt.Errorf("invalid num_features parameter for layer %s", layerName)
	}
	
	if len(mean) != numFeatures {
		return fmt.Errorf("mean length %d doesn't match num_features %d for layer %s", 
			len(mean), numFeatures, layerName)
	}
	
	// Initialize RunningStatistics if needed
	if targetLayer.RunningStatistics == nil {
		targetLayer.RunningStatistics = make(map[string][]float32)
	}
	
	// Set custom normalization values
	targetLayer.RunningStatistics["running_mean"] = make([]float32, len(mean))
	copy(targetLayer.RunningStatistics["running_mean"], mean)
	
	targetLayer.RunningStatistics["running_var"] = make([]float32, len(variance))
	copy(targetLayer.RunningStatistics["running_var"], variance)
	
	fmt.Printf("Set custom normalization for layer %s: mean=%v, var=%v\n", 
		layerName, mean[:min(3, len(mean))], variance[:min(3, len(variance))])
	
	// Recompile inference engine with new normalization values
	return mie.recompileForInference()
}

// SetStandardNormalization sets standard normalization (mean=0, var=1) for a BatchNorm layer
// This is equivalent to the old hardcoded behavior but now configurable
func (mie *ModelInferenceEngine) SetStandardNormalization(layerName string) error {
	// Find the layer to get num_features
	for i := range mie.modelSpec.Layers {
		layer := &mie.modelSpec.Layers[i]
		if layer.Name == layerName && layer.Type == layers.BatchNorm {
			numFeatures, ok := layer.Parameters["num_features"].(int)
			if !ok {
				return fmt.Errorf("invalid num_features parameter for layer %s", layerName)
			}
			
			// Create standard normalization: mean=0, var=1
			mean := make([]float32, numFeatures)
			variance := make([]float32, numFeatures)
			for i := 0; i < numFeatures; i++ {
				mean[i] = 0.0
				variance[i] = 1.0
			}
			
			return mie.SetCustomNormalization(layerName, mean, variance)
		}
	}
	
	return fmt.Errorf("BatchNorm layer '%s' not found", layerName)
}

// ListBatchNormLayers returns the names of all BatchNorm layers in the model
// This helps users identify which layers can have custom normalization
func (mie *ModelInferenceEngine) ListBatchNormLayers() []string {
	var batchNormLayers []string
	
	for _, layer := range mie.modelSpec.Layers {
		if layer.Type == layers.BatchNorm {
			batchNormLayers = append(batchNormLayers, layer.Name)
		}
	}
	
	return batchNormLayers
}

// min helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}