package layers

import (
	"fmt"

	"github.com/tsawler/go-metal/memory"
)

// LayerType represents the type of neural network layer
type LayerType int

const (
	Dense LayerType = iota
	Conv2D
	ReLU
	Softmax
	MaxPool2D
	Dropout
	BatchNorm
	LeakyReLU
	ELU
)

func (lt LayerType) String() string {
	switch lt {
	case Dense:
		return "Dense"
	case Conv2D:
		return "Conv2D"
	case ReLU:
		return "ReLU"
	case Softmax:
		return "Softmax"
	case MaxPool2D:
		return "MaxPool2D"
	case Dropout:
		return "Dropout"
	case BatchNorm:
		return "BatchNorm"
	case LeakyReLU:
		return "LeakyReLU"
	case ELU:
		return "ELU"
	default:
		return "Unknown"
	}
}

// LayerSpec defines layer configuration for the TrainingEngine
// This is pure configuration - no execution logic
type LayerSpec struct {
	Type       LayerType              `json:"type"`
	Name       string                 `json:"name"`
	Parameters map[string]interface{} `json:"parameters"`

	// Shape information (computed during model compilation)
	InputShape  []int `json:"input_shape,omitempty"`
	OutputShape []int `json:"output_shape,omitempty"`

	// Parameter metadata (computed during model compilation)
	ParameterShapes [][]int `json:"parameter_shapes,omitempty"`
	ParameterCount  int64   `json:"parameter_count,omitempty"`

	// Non-learnable parameters (e.g., BatchNorm running statistics)
	// These are used during graph construction but not counted as learnable parameters
	RunningStatistics map[string][]float32 `json:"running_statistics,omitempty"`
}

// ModelSpec defines a complete neural network model as layer configuration
// This replaces individual layer execution with unified model specification
type ModelSpec struct {
	Layers []LayerSpec `json:"layers"`

	// Compiled model information
	TotalParameters int64   `json:"total_parameters"`
	ParameterShapes [][]int `json:"parameter_shapes"`
	InputShape      []int   `json:"input_shape"`
	OutputShape     []int   `json:"output_shape"`
	Compiled        bool    `json:"compiled"`
}

// LayerFactory creates layer specifications (configuration only)
type LayerFactory struct{}

// NewFactory creates a new layer factory
func NewFactory() *LayerFactory {
	return &LayerFactory{}
}

// CreateDenseSpec creates a dense layer specification
func (lf *LayerFactory) CreateDenseSpec(inputSize, outputSize int, useBias bool, name string) LayerSpec {
	return LayerSpec{
		Type: Dense,
		Name: name,
		Parameters: map[string]interface{}{
			"input_size":  inputSize,
			"output_size": outputSize,
			"use_bias":    useBias,
		},
	}
}

// CreateConv2DSpec creates a Conv2D layer specification
func (lf *LayerFactory) CreateConv2DSpec(
	inputChannels, outputChannels, kernelSize, stride, padding int,
	useBias bool, name string,
) LayerSpec {
	return LayerSpec{
		Type: Conv2D,
		Name: name,
		Parameters: map[string]interface{}{
			"input_channels":  inputChannels,
			"output_channels": outputChannels,
			"kernel_size":     kernelSize,
			"stride":          stride,
			"padding":         padding,
			"use_bias":        useBias,
		},
	}
}

// CreateReLUSpec creates a ReLU activation specification
func (lf *LayerFactory) CreateReLUSpec(name string) LayerSpec {
	return LayerSpec{
		Type:       ReLU,
		Name:       name,
		Parameters: map[string]interface{}{},
	}
}

// CreateSoftmaxSpec creates a Softmax activation specification
func (lf *LayerFactory) CreateSoftmaxSpec(axis int, name string) LayerSpec {
	return LayerSpec{
		Type: Softmax,
		Name: name,
		Parameters: map[string]interface{}{
			"axis": axis,
		},
	}
}

// CreateLeakyReLUSpec creates a Leaky ReLU activation specification
func (lf *LayerFactory) CreateLeakyReLUSpec(negativeSlope float32, name string) LayerSpec {
	return LayerSpec{
		Type: LeakyReLU,
		Name: name,
		Parameters: map[string]interface{}{
			"negative_slope": negativeSlope,
		},
	}
}

// CreateELUSpec creates an ELU activation specification
func (lf *LayerFactory) CreateELUSpec(alpha float32, name string) LayerSpec {
	return LayerSpec{
		Type: ELU,
		Name: name,
		Parameters: map[string]interface{}{
			"alpha": alpha,
		},
	}
}

// CreateDropoutSpec creates a Dropout layer specification
func (lf *LayerFactory) CreateDropoutSpec(rate float32, name string) LayerSpec {
	return LayerSpec{
		Type: Dropout,
		Name: name,
		Parameters: map[string]interface{}{
			"rate":     rate,
			"training": true, // Default to training mode
		},
	}
}

// CreateBatchNormSpec creates a Batch Normalization layer specification
func (lf *LayerFactory) CreateBatchNormSpec(numFeatures int, eps float32, momentum float32, affine bool, name string) LayerSpec {
	return LayerSpec{
		Type: BatchNorm,
		Name: name,
		Parameters: map[string]interface{}{
			"num_features":        numFeatures,
			"eps":                eps,
			"momentum":           momentum,
			"affine":             affine,
			"track_running_stats": true, // Always track for training
			"training":           true,  // Default to training mode
		},
	}
}

// ModelBuilder helps construct neural network models
type ModelBuilder struct {
	layers     []LayerSpec
	inputShape []int
	compiled   bool
}

// NewModelBuilder creates a new model builder
func NewModelBuilder(inputShape []int) *ModelBuilder {
	return &ModelBuilder{
		layers:     make([]LayerSpec, 0),
		inputShape: inputShape,
		compiled:   false,
	}
}

// AddLayer adds a layer to the model
func (mb *ModelBuilder) AddLayer(layer LayerSpec) *ModelBuilder {
	mb.layers = append(mb.layers, layer)
	mb.compiled = false // Invalidate compilation
	return mb
}

// AddDense adds a dense layer to the model
func (mb *ModelBuilder) AddDense(outputSize int, useBias bool, name string) *ModelBuilder {
	// Input size will be computed during compilation
	layer := LayerSpec{
		Type: Dense,
		Name: name,
		Parameters: map[string]interface{}{
			"output_size": outputSize,
			"use_bias":    useBias,
		},
	}
	return mb.AddLayer(layer)
}

// AddConv2D adds a Conv2D layer to the model
func (mb *ModelBuilder) AddConv2D(
	outputChannels, kernelSize, stride, padding int,
	useBias bool, name string,
) *ModelBuilder {
	layer := LayerSpec{
		Type: Conv2D,
		Name: name,
		Parameters: map[string]interface{}{
			"output_channels": outputChannels,
			"kernel_size":     kernelSize,
			"stride":          stride,
			"padding":         padding,
			"use_bias":        useBias,
		},
	}
	return mb.AddLayer(layer)
}

// AddReLU adds a ReLU activation to the model
func (mb *ModelBuilder) AddReLU(name string) *ModelBuilder {
	layer := LayerSpec{
		Type:       ReLU,
		Name:       name,
		Parameters: map[string]interface{}{},
	}
	return mb.AddLayer(layer)
}

// AddSoftmax adds a Softmax activation to the model
func (mb *ModelBuilder) AddSoftmax(axis int, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: Softmax,
		Name: name,
		Parameters: map[string]interface{}{
			"axis": axis,
		},
	}
	return mb.AddLayer(layer)
}

// AddDropout adds a Dropout layer to the model
// rate: dropout probability (0.0 = no dropout, 1.0 = drop all)
// training: whether the layer is in training mode (affects dropout behavior)
func (mb *ModelBuilder) AddDropout(rate float32, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: Dropout,
		Name: name,
		Parameters: map[string]interface{}{
			"rate":     rate,
			"training": true, // Default to training mode, will be controlled by trainer
		},
	}
	return mb.AddLayer(layer)
}

// AddBatchNorm adds a Batch Normalization layer to the model
// num_features: number of input features (channels for Conv layers, neurons for Dense layers)
// eps: small value added for numerical stability (default: 1e-5)
// momentum: momentum for running statistics update (default: 0.1)
// affine: whether to use learnable scale and shift parameters (default: true)
// track_running_stats: whether to track running statistics during training (default: true)
func (mb *ModelBuilder) AddBatchNorm(numFeatures int, eps float32, momentum float32, affine bool, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: BatchNorm,
		Name: name,
		Parameters: map[string]interface{}{
			"num_features":        numFeatures,
			"eps":                eps,
			"momentum":           momentum,
			"affine":             affine,
			"track_running_stats": true, // Always track for training, controlled by trainer mode
			"training":           true,  // Default to training mode, controlled by trainer
		},
	}
	return mb.AddLayer(layer)
}

// AddLeakyReLU adds a Leaky ReLU activation to the model
// negativeSlope: slope for negative input values (default: 0.01)
func (mb *ModelBuilder) AddLeakyReLU(negativeSlope float32, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: LeakyReLU,
		Name: name,
		Parameters: map[string]interface{}{
			"negative_slope": negativeSlope,
		},
	}
	return mb.AddLayer(layer)
}

// AddELU adds an ELU activation to the model
// alpha: controls saturation level for negative inputs (default: 1.0)
func (mb *ModelBuilder) AddELU(alpha float32, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: ELU,
		Name: name,
		Parameters: map[string]interface{}{
			"alpha": alpha,
		},
	}
	return mb.AddLayer(layer)
}

// Compile compiles the model and computes shapes and parameter counts
func (mb *ModelBuilder) Compile() (*ModelSpec, error) {
	if len(mb.layers) == 0 {
		return nil, fmt.Errorf("cannot compile empty model")
	}

	model := &ModelSpec{
		Layers:     make([]LayerSpec, len(mb.layers)),
		InputShape: mb.inputShape,
		Compiled:   false,
	}

	copy(model.Layers, mb.layers)

	// Compute shapes and parameter information
	currentShape := mb.inputShape
	var allParameterShapes [][]int
	totalParams := int64(0)

	for i := range model.Layers {
		layer := &model.Layers[i]

		// Set input shape for this layer
		layer.InputShape = make([]int, len(currentShape))
		copy(layer.InputShape, currentShape)

		// Compute output shape and parameters based on layer type
		outputShape, paramShapes, paramCount, err := mb.computeLayerInfo(layer, currentShape)
		if err != nil {
			return nil, fmt.Errorf("failed to compute layer %d (%s) info: %v", i, layer.Name, err)
		}

		layer.OutputShape = outputShape
		layer.ParameterShapes = paramShapes
		layer.ParameterCount = paramCount

		// Add to global parameter information
		allParameterShapes = append(allParameterShapes, paramShapes...)
		totalParams += paramCount

		// Update current shape for next layer
		currentShape = outputShape
	}

	model.OutputShape = currentShape
	model.ParameterShapes = allParameterShapes
	model.TotalParameters = totalParams
	model.Compiled = true
	mb.compiled = true

	return model, nil
}

// computeLayerInfo computes output shape and parameter information for a layer
func (mb *ModelBuilder) computeLayerInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	switch layer.Type {
	case Dense:
		return mb.computeDenseInfo(layer, inputShape)
	case Conv2D:
		return mb.computeConv2DInfo(layer, inputShape)
	case BatchNorm:
		return mb.computeBatchNormInfo(layer, inputShape)
	case ReLU, Softmax, Dropout, LeakyReLU, ELU:
		return mb.computeActivationInfo(layer, inputShape)
	default:
		return nil, nil, 0, fmt.Errorf("unsupported layer type: %s", layer.Type.String())
	}
}

// computeDenseInfo computes dense layer information
func (mb *ModelBuilder) computeDenseInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	if len(inputShape) < 2 {
		return nil, nil, 0, fmt.Errorf("dense layer requires at least 2D input")
	}

	// Get parameters
	outputSize, ok := layer.Parameters["output_size"].(int)
	if !ok {
		return nil, nil, 0, fmt.Errorf("missing output_size parameter")
	}

	useBias := true
	if bias, exists := layer.Parameters["use_bias"].(bool); exists {
		useBias = bias
	}

	// Compute input size by flattening all dimensions except batch
	// For 2D input [batch, features]: input_size = features
	// For 4D input [batch, channels, height, width]: input_size = channels * height * width
	inputSize := 1
	for i := 1; i < len(inputShape); i++ {
		inputSize *= inputShape[i]
	}

	// Update layer parameters with computed input size
	layer.Parameters["input_size"] = inputSize

	// Output shape: Dense layer always outputs 2D [batch, outputSize]
	// regardless of input dimensionality (handles automatic flattening)
	batchSize := inputShape[0]
	outputShape := []int{batchSize, outputSize}

	// Parameter shapes: weights + optional bias
	var paramShapes [][]int
	paramCount := int64(0)

	// Weight matrix: [inputSize, outputSize]
	weightShape := []int{inputSize, outputSize}
	paramShapes = append(paramShapes, weightShape)
	paramCount += int64(inputSize * outputSize)

	// Bias vector: [outputSize] (if enabled)
	if useBias {
		biasShape := []int{outputSize}
		paramShapes = append(paramShapes, biasShape)
		paramCount += int64(outputSize)
	}

	return outputShape, paramShapes, paramCount, nil
}

// computeConv2DInfo computes Conv2D layer information
func (mb *ModelBuilder) computeConv2DInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	if len(inputShape) != 4 {
		return nil, nil, 0, fmt.Errorf("Conv2D layer requires 4D input [batch, channels, height, width]")
	}

	// Get parameters
	outputChannels, ok := layer.Parameters["output_channels"].(int)
	if !ok {
		return nil, nil, 0, fmt.Errorf("missing output_channels parameter")
	}

	kernelSize, ok := layer.Parameters["kernel_size"].(int)
	if !ok {
		return nil, nil, 0, fmt.Errorf("missing kernel_size parameter")
	}

	stride, ok := layer.Parameters["stride"].(int)
	if !ok {
		stride = 1
	}

	padding, ok := layer.Parameters["padding"].(int)
	if !ok {
		padding = 0
	}

	useBias := true
	if bias, exists := layer.Parameters["use_bias"].(bool); exists {
		useBias = bias
	}

	// Extract input dimensions
	batchSize := inputShape[0]
	inputChannels := inputShape[1]
	inputHeight := inputShape[2]
	inputWidth := inputShape[3]

	// Update layer parameters with computed input channels
	layer.Parameters["input_channels"] = inputChannels

	// Compute output dimensions
	outputHeight := (inputHeight+2*padding-kernelSize)/stride + 1
	outputWidth := (inputWidth+2*padding-kernelSize)/stride + 1

	outputShape := []int{batchSize, outputChannels, outputHeight, outputWidth}

	// Parameter shapes: weights + optional bias
	var paramShapes [][]int
	paramCount := int64(0)

	// Weight tensor: [outputChannels, inputChannels, kernelSize, kernelSize]
	weightShape := []int{outputChannels, inputChannels, kernelSize, kernelSize}
	paramShapes = append(paramShapes, weightShape)
	paramCount += int64(outputChannels * inputChannels * kernelSize * kernelSize)

	// Bias vector: [outputChannels] (if enabled)
	if useBias {
		biasShape := []int{outputChannels}
		paramShapes = append(paramShapes, biasShape)
		paramCount += int64(outputChannels)
	}

	return outputShape, paramShapes, paramCount, nil
}

// computeActivationInfo computes activation layer information (no parameters)
// computeBatchNormInfo computes batch normalization layer information
func (mb *ModelBuilder) computeBatchNormInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	if len(inputShape) < 2 {
		return nil, nil, 0, fmt.Errorf("batch norm layer requires at least 2D input")
	}

	// Get parameters
	numFeatures, ok := layer.Parameters["num_features"].(int)
	if !ok {
		return nil, nil, 0, fmt.Errorf("missing num_features parameter")
	}

	affine := true
	if af, exists := layer.Parameters["affine"].(bool); exists {
		affine = af
	}

	// BatchNorm doesn't change the input shape - it normalizes along the feature dimension
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)

	// Validate num_features matches the appropriate dimension
	// For 2D input [batch, features]: features dimension is index 1
	// For 4D input [batch, channels, height, width]: channels dimension is index 1
	expectedFeatures := inputShape[1]
	if numFeatures != expectedFeatures {
		return nil, nil, 0, fmt.Errorf("num_features (%d) doesn't match input feature dimension (%d)", numFeatures, expectedFeatures)
	}

	var paramShapes [][]int
	var paramCount int64

	if affine {
		// Learnable scale (gamma) and shift (beta) parameters
		// Both have shape [num_features]
		paramShapes = append(paramShapes, []int{numFeatures}) // gamma (scale)
		paramShapes = append(paramShapes, []int{numFeatures}) // beta (shift)
		paramCount = int64(numFeatures * 2) // gamma + beta
	}

	// Note: running_mean and running_var are not trainable parameters
	// They are buffers managed by the training engine and don't count as parameters

	return outputShape, paramShapes, paramCount, nil
}

func (mb *ModelBuilder) computeActivationInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	// Activation layers don't change shape and have no parameters
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)

	return outputShape, [][]int{}, 0, nil
}

// GetCompiledModel returns the compiled model (must call Compile first)
func (mb *ModelBuilder) GetCompiledModel() (*ModelSpec, error) {
	if !mb.compiled {
		return nil, fmt.Errorf("model not compiled - call Compile() first")
	}

	return mb.Compile() // Re-compile to get fresh copy
}

// CreateParameterTensors creates GPU tensors for all model parameters
func (ms *ModelSpec) CreateParameterTensors() ([]*memory.Tensor, error) {
	if !ms.Compiled {
		return nil, fmt.Errorf("model not compiled")
	}

	var tensors []*memory.Tensor

	for _, shape := range ms.ParameterShapes {
		tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
		if err != nil {
			// Cleanup on error
			for _, t := range tensors {
				t.Release()
			}
			return nil, fmt.Errorf("failed to create parameter tensor: %v", err)
		}
		tensors = append(tensors, tensor)
	}

	return tensors, nil
}

// Summary returns a human-readable model summary
func (ms *ModelSpec) Summary() string {
	if !ms.Compiled {
		return "Model not compiled"
	}

	summary := fmt.Sprintf("Model Summary:\n")
	summary += fmt.Sprintf("Input Shape: %v\n", ms.InputShape)
	summary += fmt.Sprintf("Output Shape: %v\n", ms.OutputShape)
	summary += fmt.Sprintf("Total Parameters: %d\n", ms.TotalParameters)
	summary += fmt.Sprintf("Layers: %d\n\n", len(ms.Layers))

	for i, layer := range ms.Layers {
		summary += fmt.Sprintf("Layer %d: %s (%s)\n", i+1, layer.Name, layer.Type.String())
		summary += fmt.Sprintf("  Input:  %v\n", layer.InputShape)
		summary += fmt.Sprintf("  Output: %v\n", layer.OutputShape)
		summary += fmt.Sprintf("  Params: %d\n", layer.ParameterCount)

		if len(layer.Parameters) > 0 {
			summary += fmt.Sprintf("  Config: %v\n", layer.Parameters)
		}
		summary += "\n"
	}

	return summary
}

// ValidateModelForTrainingEngine checks if model is compatible with existing TrainingEngine
// DEPRECATED: Use ValidateModelForHybridEngine or ValidateModelForDynamicEngine instead
func (ms *ModelSpec) ValidateModelForTrainingEngine() error {
	// For backward compatibility, default to hybrid engine validation
	return ms.ValidateModelForHybridEngine()
}

// ValidateModelForHybridEngine checks if model is compatible with Hybrid TrainingEngine
// Optimized for CNN architectures with 4D input and Conv+Dense layer combinations
func (ms *ModelSpec) ValidateModelForHybridEngine() error {
	if !ms.Compiled {
		return fmt.Errorf("model not compiled")
	}

	if len(ms.Layers) == 0 {
		return fmt.Errorf("empty model")
	}

	// Hybrid engine is optimized for 4D CNN inputs
	if len(ms.InputShape) != 4 {
		return fmt.Errorf("Hybrid TrainingEngine requires 4D input [batch, channels, height, width]")
	}

	// Check that we have both Conv2D and Dense layers (CNN pattern)
	hasConv := false
	hasDense := false

	for _, layer := range ms.Layers {
		switch layer.Type {
		case Conv2D:
			hasConv = true
		case Dense:
			hasDense = true
		}
	}

	if !hasConv {
		return fmt.Errorf("Hybrid TrainingEngine requires at least one Conv2D layer")
	}

	if !hasDense {
		return fmt.Errorf("Hybrid TrainingEngine requires at least one Dense layer")
	}

	return nil
}

// ValidateModelForDynamicEngine checks if model is compatible with Dynamic TrainingEngine
// Supports any architecture with flexible input dimensions (2D, 4D, etc.)
func (ms *ModelSpec) ValidateModelForDynamicEngine() error {
	// Remove compiled requirement - allow validation before compilation
	if len(ms.Layers) == 0 {
		return fmt.Errorf("empty model")
	}

	// Dynamic engine supports flexible input dimensions - even 1D is allowed
	if len(ms.InputShape) == 0 {
		return fmt.Errorf("model must specify input shape")
	}

	// Validate each layer has required parameters for dynamic processing
	for i, layer := range ms.Layers {
		if err := ms.validateLayerForDynamicEngine(layer, i); err != nil {
			return fmt.Errorf("layer %d (%s): %v", i, layer.Name, err)
		}
	}

	// Check that we have at least one trainable layer
	hasTrainableLayer := false
	for _, layer := range ms.Layers {
		switch layer.Type {
		case Dense, Conv2D:
			hasTrainableLayer = true
			break
		}
	}

	if !hasTrainableLayer {
		return fmt.Errorf("Dynamic TrainingEngine requires at least one trainable layer (Dense or Conv2D)")
	}

	return nil
}

// validateLayerForDynamicEngine validates individual layers for dynamic engine compatibility
func (ms *ModelSpec) validateLayerForDynamicEngine(layer LayerSpec, index int) error {
	switch layer.Type {
	case Dense:
		// Dense layers need input_size and output_size
		if _, ok := layer.Parameters["input_size"]; !ok {
			return fmt.Errorf("Dense layer missing input_size parameter")
		}
		if _, ok := layer.Parameters["output_size"]; !ok {
			return fmt.Errorf("Dense layer missing output_size parameter")
		}
		
	case Conv2D:
		// Conv2D layers need kernel_size, input_channels, output_channels
		requiredParams := []string{"kernel_size", "input_channels", "output_channels"}
		for _, param := range requiredParams {
			if _, ok := layer.Parameters[param]; !ok {
				return fmt.Errorf("Conv2D layer missing %s parameter", param)
			}
		}
		
	case ReLU, Softmax, LeakyReLU, ELU:
		// Activation layers don't require specific parameters
		
	case MaxPool2D:
		// MaxPool2D needs pool_size
		if _, ok := layer.Parameters["pool_size"]; !ok {
			return fmt.Errorf("MaxPool2D layer missing pool_size parameter")
		}
		
	case BatchNorm:
		// BatchNorm needs num_features
		if _, ok := layer.Parameters["num_features"]; !ok {
			return fmt.Errorf("BatchNorm layer missing num_features parameter")
		}
		
	case Dropout:
		// Dropout needs rate parameter
		if _, ok := layer.Parameters["rate"]; !ok {
			return fmt.Errorf("Dropout layer missing rate parameter")
		}
		
	default:
		return fmt.Errorf("unsupported layer type: %v", layer.Type)
	}
	
	return nil
}

// ValidateModelForInference checks if model is compatible with InferenceEngine
// More lenient than training validation - supports Dense-only models
func (ms *ModelSpec) ValidateModelForInference() error {
	if !ms.Compiled {
		return fmt.Errorf("model not compiled")
	}

	if len(ms.Layers) == 0 {
		return fmt.Errorf("empty model")
	}

	// Check input shape compatibility (support both 2D and 4D)
	if len(ms.InputShape) < 2 {
		return fmt.Errorf("InferenceEngine requires at least 2D input [batch, features]")
	}

	// Must have at least one Dense layer for classification output
	hasDense := false
	for _, layer := range ms.Layers {
		if layer.Type == Dense {
			hasDense = true
			break
		}
	}

	if !hasDense {
		return fmt.Errorf("InferenceEngine requires at least one Dense layer for output")
	}

	return nil
}

// ConvertToInferenceLayerSpecs converts ModelSpec to inference-optimized layer specifications
func (ms *ModelSpec) ConvertToInferenceLayerSpecs() ([]DynamicLayerSpec, error) {
	if err := ms.ValidateModelForInference(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}

	// Convert layers to inference-optimized specifications
	specs := make([]DynamicLayerSpec, len(ms.Layers))
	
	for i, layer := range ms.Layers {
		spec := DynamicLayerSpec{
			LayerType:       int32(layer.Type),
			InputShape:      convertIntSliceToInt32Array(layer.InputShape),
			InputShapeLen:   int32(len(layer.InputShape)),
			OutputShape:     convertIntSliceToInt32Array(layer.OutputShape),
			OutputShapeLen:  int32(len(layer.OutputShape)),
		}
		
		// Copy layer name
		nameBytes := [64]byte{}
		copy(nameBytes[:], []byte(layer.Name))
		spec.NameBytes = nameBytes
		
		// Copy parameters for inference
		if layer.Parameters != nil {
			// Convert layer-specific parameters
			switch layer.Type {
			case Dense:
				if inputSize, ok := layer.Parameters["input_size"].(int); ok {
					spec.ParamInt[0] = int32(inputSize)
					spec.ParamIntCount++
				}
				if outputSize, ok := layer.Parameters["output_size"].(int); ok {
					spec.ParamInt[1] = int32(outputSize)
					spec.ParamIntCount++
				}
				if useBias, ok := layer.Parameters["use_bias"].(bool); ok {
					if useBias {
						spec.ParamInt[2] = 1
					} else {
						spec.ParamInt[2] = 0
					}
					spec.ParamIntCount++
				}
				
			case Conv2D:
				// JSON numbers are often decoded as float64, so try both int and float64
				// Check for input_channels/in_channels
				if inChannels, ok := layer.Parameters["input_channels"].(int); ok {
					spec.ParamInt[0] = int32(inChannels)
					spec.ParamIntCount++
				} else if inChannels, ok := layer.Parameters["in_channels"].(int); ok {
					spec.ParamInt[0] = int32(inChannels)
					spec.ParamIntCount++
				} else if inChannelsFloat, ok := layer.Parameters["input_channels"].(float64); ok {
					spec.ParamInt[0] = int32(inChannelsFloat)
					spec.ParamIntCount++
				} else if inChannelsFloat, ok := layer.Parameters["in_channels"].(float64); ok {
					spec.ParamInt[0] = int32(inChannelsFloat)
					spec.ParamIntCount++
				}
				
				// Check for output_channels/out_channels
				if outChannels, ok := layer.Parameters["output_channels"].(int); ok {
					spec.ParamInt[1] = int32(outChannels)
					spec.ParamIntCount++
				} else if outChannels, ok := layer.Parameters["out_channels"].(int); ok {
					spec.ParamInt[1] = int32(outChannels)
					spec.ParamIntCount++
				} else if outChannelsFloat, ok := layer.Parameters["output_channels"].(float64); ok {
					spec.ParamInt[1] = int32(outChannelsFloat)
					spec.ParamIntCount++
				} else if outChannelsFloat, ok := layer.Parameters["out_channels"].(float64); ok {
					spec.ParamInt[1] = int32(outChannelsFloat)
					spec.ParamIntCount++
				}
				
				if kernelSize, ok := layer.Parameters["kernel_size"].(int); ok {
					spec.ParamInt[2] = int32(kernelSize)
					spec.ParamIntCount++
				} else if kernelSizeFloat, ok := layer.Parameters["kernel_size"].(float64); ok {
					spec.ParamInt[2] = int32(kernelSizeFloat)
					spec.ParamIntCount++
				}
				
				if stride, ok := layer.Parameters["stride"].(int); ok {
					spec.ParamInt[3] = int32(stride)
					spec.ParamIntCount++
				} else if strideFloat, ok := layer.Parameters["stride"].(float64); ok {
					spec.ParamInt[3] = int32(strideFloat)
					spec.ParamIntCount++
				}
				
				if padding, ok := layer.Parameters["padding"].(int); ok {
					spec.ParamInt[4] = int32(padding)
					spec.ParamIntCount++
				} else if paddingFloat, ok := layer.Parameters["padding"].(float64); ok {
					spec.ParamInt[4] = int32(paddingFloat)
					spec.ParamIntCount++
				}
				
				// Handle use_bias
				if useBias, ok := layer.Parameters["use_bias"].(bool); ok {
					if useBias {
						spec.ParamInt[5] = 1
					} else {
						spec.ParamInt[5] = 0
					}
					spec.ParamIntCount++
				}
			
			case BatchNorm:
				// Handle num_features
				if numFeatures, ok := layer.Parameters["num_features"].(int); ok {
					spec.ParamInt[0] = int32(numFeatures)
					spec.ParamIntCount++
				} else if numFeaturesFloat, ok := layer.Parameters["num_features"].(float64); ok {
					spec.ParamInt[0] = int32(numFeaturesFloat)
					spec.ParamIntCount++
				}
				
				// Handle eps (epsilon) - check both float32 and float64
				if eps, ok := layer.Parameters["eps"].(float32); ok {
					spec.ParamFloat[0] = eps
					spec.ParamFloatCount++
				} else if eps, ok := layer.Parameters["epsilon"].(float32); ok {
					spec.ParamFloat[0] = eps
					spec.ParamFloatCount++
				} else if eps, ok := layer.Parameters["eps"].(float64); ok {
					spec.ParamFloat[0] = float32(eps)
					spec.ParamFloatCount++
				} else if eps, ok := layer.Parameters["epsilon"].(float64); ok {
					spec.ParamFloat[0] = float32(eps)
					spec.ParamFloatCount++
				}
				
				// Handle momentum - check both float32 and float64
				if momentum, ok := layer.Parameters["momentum"].(float32); ok {
					spec.ParamFloat[1] = momentum
					spec.ParamFloatCount++
				} else if momentum, ok := layer.Parameters["momentum"].(float64); ok {
					spec.ParamFloat[1] = float32(momentum)
					spec.ParamFloatCount++
				}
				
				// Handle affine flag
				if affine, ok := layer.Parameters["affine"].(bool); ok {
					if affine {
						spec.ParamInt[1] = 1
					} else {
						spec.ParamInt[1] = 0
					}
					spec.ParamIntCount++
				}
				
				// Handle track_running_stats flag
				if trackRunningStats, ok := layer.Parameters["track_running_stats"].(bool); ok {
					if trackRunningStats {
						spec.ParamInt[2] = 1
					} else {
						spec.ParamInt[2] = 0
					}
					spec.ParamIntCount++
				}
				
				// For inference, set training=false regardless of stored value
				spec.ParamInt[3] = 0 // training = false for inference
				spec.ParamIntCount++
				
				// ARCHITECTURAL FIX: Copy running statistics for inference (mean=0, var=1 → actual values)
				// This resolves the hardcoded normalization limitation
				if layer.RunningStatistics != nil {
					// Extract running statistics if available from trained model
					if runningMean, exists := layer.RunningStatistics["running_mean"]; exists {
						spec.RunningMean = make([]float32, len(runningMean))
						copy(spec.RunningMean, runningMean)
						spec.HasRunningStats = boolToInt32(true)
					}
					if runningVar, exists := layer.RunningStatistics["running_var"]; exists {
						spec.RunningVar = make([]float32, len(runningVar))
						copy(spec.RunningVar, runningVar)
						spec.HasRunningStats = boolToInt32(true)
					}
				}
				
				// If no running statistics are available, initialize with proper defaults
				// Use num_features to determine size
				if spec.HasRunningStats == 0 && spec.ParamIntCount > 0 {
					numFeatures := int(spec.ParamInt[0]) // num_features is ParamInt[0]
					if numFeatures > 0 {
						// Initialize with standard defaults: mean=0, var=1
						spec.RunningMean = make([]float32, numFeatures)
						spec.RunningVar = make([]float32, numFeatures)
						for i := 0; i < numFeatures; i++ {
							spec.RunningMean[i] = 0.0 // Default mean
							spec.RunningVar[i] = 1.0  // Default variance
						}
						spec.HasRunningStats = boolToInt32(true)
					}
				}
				
			case LeakyReLU:
				if negativeSlope, ok := layer.Parameters["negative_slope"].(float64); ok {
					spec.ParamFloat[0] = float32(negativeSlope)
					spec.ParamFloatCount++
				} else {
					// Default negative slope
					spec.ParamFloat[0] = 0.01
					spec.ParamFloatCount++
				}
				
			case ELU:
				if alpha, ok := layer.Parameters["alpha"].(float64); ok {
					spec.ParamFloat[0] = float32(alpha)
					spec.ParamFloatCount++
				} else {
					// Default alpha
					spec.ParamFloat[0] = 1.0
					spec.ParamFloatCount++
				}
				
			case Dropout:
				if dropRate, ok := layer.Parameters["drop_rate"].(float64); ok {
					spec.ParamFloat[0] = float32(dropRate)
					spec.ParamFloatCount++
				}
				// For inference, dropout is disabled (rate = 0.0)
				spec.ParamFloat[0] = 0.0
				spec.ParamFloatCount = 1
			}
		}
		
		specs[i] = spec
	}
	
	return specs, nil
}

// GetTrainingEngineConfig generates configuration for the existing TrainingEngine
// This bridges the new layer system with the proven high-performance engine
func (ms *ModelSpec) GetTrainingEngineConfig() (map[string]interface{}, error) {
	if err := ms.ValidateModelForTrainingEngine(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}

	config := map[string]interface{}{
		"model_type":       "cnn",
		"input_shape":      ms.InputShape,
		"output_shape":     ms.OutputShape,
		"total_parameters": ms.TotalParameters,
		"parameter_shapes": ms.ParameterShapes,
		"layer_count":      len(ms.Layers),
		"layers":           ms.Layers,
	}

	return config, nil
}

// SerializeForCGO converts ModelSpec to CGO-compatible format
func (ms *ModelSpec) SerializeForCGO() (*ModelSpecC, error) {
	if !ms.Compiled {
		return nil, fmt.Errorf("model must be compiled before serialization")
	}
	
	// Convert input and output shapes
	inputShape := make([]int32, len(ms.InputShape))
	for i, dim := range ms.InputShape {
		inputShape[i] = int32(dim)
	}
	
	outputShape := make([]int32, len(ms.OutputShape))
	for i, dim := range ms.OutputShape {
		outputShape[i] = int32(dim)
	}
	
	// Convert layers
	layers := make([]LayerSpecC, len(ms.Layers))
	for i, layer := range ms.Layers {
		cLayer := LayerSpecC{
			LayerType: int32(layer.Type),
			Name:      layer.Name,
		}
		
		// Convert input shape
		if len(layer.InputShape) > 0 {
			cLayer.InputShape = make([]int32, len(layer.InputShape))
			for j, dim := range layer.InputShape {
				cLayer.InputShape[j] = int32(dim)
			}
		}
		
		// Convert output shape
		if len(layer.OutputShape) > 0 {
			cLayer.OutputShape = make([]int32, len(layer.OutputShape))
			for j, dim := range layer.OutputShape {
				cLayer.OutputShape[j] = int32(dim)
			}
		}
		
		// Convert layer-specific parameters based on type
		switch layer.Type {
		case Conv2D:
			// Conv2D parameters: input_channels, output_channels, kernel_size, stride, padding, use_bias
			cLayer.ParamInt = []int32{
				int32(getIntParam(layer.Parameters, "input_channels", 0)),
				int32(getIntParam(layer.Parameters, "output_channels", 0)),
				int32(getIntParam(layer.Parameters, "kernel_size", 0)),
				int32(getIntParam(layer.Parameters, "stride", 1)),
				int32(getIntParam(layer.Parameters, "padding", 0)),
			}
			if getBoolParam(layer.Parameters, "use_bias", false) {
				cLayer.ParamInt = append(cLayer.ParamInt, 1)
			} else {
				cLayer.ParamInt = append(cLayer.ParamInt, 0)
			}
			
		case Dense:
			// Dense parameters: input_size, output_size, use_bias
			cLayer.ParamInt = []int32{
				int32(getIntParam(layer.Parameters, "input_size", 0)),
				int32(getIntParam(layer.Parameters, "output_size", 0)),
			}
			if getBoolParam(layer.Parameters, "use_bias", false) {
				cLayer.ParamInt = append(cLayer.ParamInt, 1)
			} else {
				cLayer.ParamInt = append(cLayer.ParamInt, 0)
			}
			
		case Softmax:
			// Softmax parameters: axis
			cLayer.ParamInt = []int32{
				int32(getIntParam(layer.Parameters, "axis", -1)),
			}
			
		case ReLU:
			// ReLU has no parameters
			break
			
		case LeakyReLU:
			// Leaky ReLU parameters: negative_slope
			negativeSlope := getFloatParam(layer.Parameters, "negative_slope", 0.01)
			cLayer.ParamFloat = []float32{negativeSlope}
			
		case ELU:
			// ELU parameters: alpha
			alpha := getFloatParam(layer.Parameters, "alpha", 1.0)
			cLayer.ParamFloat = []float32{alpha}
			
		case Dropout:
			// Dropout parameters: rate, training
			rate := getFloatParam(layer.Parameters, "rate", 0.5)
			training := getBoolParam(layer.Parameters, "training", true)
			
			cLayer.ParamFloat = []float32{rate}
			cLayer.ParamInt = []int32{boolToInt32(training)}
			
		case BatchNorm:
			// BatchNorm parameters: [eps, momentum] in floats, [num_features, affine, track_running_stats, training] in ints
			numFeatures := getIntParam(layer.Parameters, "num_features", 0)
			eps := getFloatParam(layer.Parameters, "eps", 1e-5)
			momentum := getFloatParam(layer.Parameters, "momentum", 0.1)
			affine := getBoolParam(layer.Parameters, "affine", true)
			trackRunningStats := getBoolParam(layer.Parameters, "track_running_stats", true)
			training := getBoolParam(layer.Parameters, "training", true)
			
			cLayer.ParamFloat = []float32{eps, momentum}
			cLayer.ParamInt = []int32{int32(numFeatures), boolToInt32(affine), boolToInt32(trackRunningStats), boolToInt32(training)}
			
		default:
			return nil, fmt.Errorf("unsupported layer type for serialization: %s", layer.Type.String())
		}
		
		layers[i] = cLayer
	}
	
	return &ModelSpecC{
		Layers:      layers,
		InputShape:  inputShape,
		OutputShape: outputShape,
	}, nil
}

// Helper functions for parameter extraction
func getIntParam(params map[string]interface{}, key string, defaultValue int) int {
	if val, exists := params[key]; exists {
		if intVal, ok := val.(int); ok {
			return intVal
		}
	}
	return defaultValue
}

func getBoolParam(params map[string]interface{}, key string, defaultValue bool) bool {
	if val, exists := params[key]; exists {
		if boolVal, ok := val.(bool); ok {
			return boolVal
		}
	}
	return defaultValue
}

func getFloatParam(params map[string]interface{}, key string, defaultValue float32) float32 {
	if val, exists := params[key]; exists {
		if floatVal, ok := val.(float32); ok {
			return floatVal
		}
		// Handle float64 conversion
		if floatVal, ok := val.(float64); ok {
			return float32(floatVal)
		}
	}
	return defaultValue
}

// ModelSpecC represents a CGO-compatible model specification
// This needs to be defined here to avoid circular imports
type ModelSpecC struct {
	Layers      []LayerSpecC
	InputShape  []int32
	OutputShape []int32
}

type LayerSpecC struct {
	LayerType   int32
	Name        string
	InputShape  []int32
	OutputShape []int32
	ParamInt    []int32
	ParamFloat  []float32
}

// ConvertToDynamicLayerSpecs converts ModelSpec to dynamic engine layer specifications
// This is used by the true dynamic engine implementation for any architecture support
func (ms *ModelSpec) ConvertToDynamicLayerSpecs() ([]DynamicLayerSpec, error) {
	if !ms.Compiled {
		return nil, fmt.Errorf("model must be compiled before conversion to dynamic specs")
	}

	specs := make([]DynamicLayerSpec, len(ms.Layers))
	currentShape := ms.InputShape

	for i, layer := range ms.Layers {
		spec := DynamicLayerSpec{
			Name: layer.Name,
		}

		// Copy name as bytes for C compatibility
		nameBytes := []byte(layer.Name)
		copy(spec.NameBytes[:], nameBytes)

		// Set input shape
		spec.InputShape, spec.InputShapeLen = copyShapeToArray(currentShape)

		switch layer.Type {
		case Dense:
			spec.LayerType = 0 // Dense = 0 in C
			
			inputSize := getIntParam(layer.Parameters, "input_size", 0)
			outputSize := getIntParam(layer.Parameters, "output_size", 0)
			useBias := getBoolParam(layer.Parameters, "use_bias", true)
			
			spec.ParamInt[0] = int32(inputSize)
			spec.ParamInt[1] = int32(outputSize)
			spec.ParamInt[2] = boolToInt32(useBias)
			spec.ParamIntCount = 3
			
			// Update current shape for next layer
			currentShape = []int{currentShape[0], outputSize}

		case Conv2D:
			spec.LayerType = 1 // Conv2D = 1 in C
			
			inputChannels := getIntParam(layer.Parameters, "input_channels", 0)
			outputChannels := getIntParam(layer.Parameters, "output_channels", 0)
			kernelSize := getIntParam(layer.Parameters, "kernel_size", 3)
			stride := getIntParam(layer.Parameters, "stride", 1)
			padding := getIntParam(layer.Parameters, "padding", 0)
			useBias := getBoolParam(layer.Parameters, "use_bias", true)
			
			spec.ParamInt[0] = int32(inputChannels)
			spec.ParamInt[1] = int32(outputChannels)
			spec.ParamInt[2] = int32(kernelSize)
			spec.ParamInt[3] = int32(stride)
			spec.ParamInt[4] = int32(padding)
			spec.ParamInt[5] = boolToInt32(useBias)
			spec.ParamIntCount = 6
			
			// Calculate output spatial dimensions
			if len(currentShape) >= 4 {
				inputH := currentShape[2]
				inputW := currentShape[3]
				outputH := (inputH + 2*padding - kernelSize) / stride + 1
				outputW := (inputW + 2*padding - kernelSize) / stride + 1
				currentShape = []int{currentShape[0], outputChannels, outputH, outputW}
			}

		case ReLU:
			spec.LayerType = 2 // ReLU = 2 in C
			spec.ParamIntCount = 0
			spec.ParamFloatCount = 0
			// Shape unchanged for ReLU

		case LeakyReLU:
			spec.LayerType = 7 // LeakyReLU = 7 in Go enum (next available after BatchNorm=6)
			
			negativeSlope := getFloatParam(layer.Parameters, "negative_slope", 0.01)
			spec.ParamFloat[0] = negativeSlope
			spec.ParamFloatCount = 1
			spec.ParamIntCount = 0
			// Shape unchanged for LeakyReLU

		case ELU:
			spec.LayerType = 8 // ELU = 8 in Go enum (next available after LeakyReLU=7)
			
			alpha := getFloatParam(layer.Parameters, "alpha", 1.0)
			spec.ParamFloat[0] = alpha
			spec.ParamFloatCount = 1
			spec.ParamIntCount = 0
			// Shape unchanged for ELU

		case Softmax:
			spec.LayerType = 3 // Softmax = 3 in C
			
			axis := getIntParam(layer.Parameters, "axis", -1)
			spec.ParamInt[0] = int32(axis)
			spec.ParamIntCount = 1
			// Shape unchanged for Softmax

		case Dropout:
			spec.LayerType = 5 // Dropout = 5 in Go enum
			
			rate := getFloatParam(layer.Parameters, "rate", 0.5)
			training := getBoolParam(layer.Parameters, "training", true)
			
			spec.ParamFloat[0] = rate
			spec.ParamInt[0] = boolToInt32(training)
			spec.ParamFloatCount = 1
			spec.ParamIntCount = 1
			// Shape unchanged for Dropout

		case BatchNorm:
			spec.LayerType = 6 // BatchNorm = 6 in Go enum (next available after Dropout=5)
			
			numFeatures := getIntParam(layer.Parameters, "num_features", 0)
			eps := getFloatParam(layer.Parameters, "eps", 1e-5)
			momentum := getFloatParam(layer.Parameters, "momentum", 0.1)
			affine := getBoolParam(layer.Parameters, "affine", true)
			trackRunningStats := getBoolParam(layer.Parameters, "track_running_stats", true)
			training := getBoolParam(layer.Parameters, "training", true)
			
			// Pack parameters: [eps, momentum] in floats, [num_features, affine, track_running_stats, training] in ints
			spec.ParamFloat[0] = eps
			spec.ParamFloat[1] = momentum
			spec.ParamInt[0] = int32(numFeatures)
			spec.ParamInt[1] = boolToInt32(affine)
			spec.ParamInt[2] = boolToInt32(trackRunningStats)
			spec.ParamInt[3] = boolToInt32(training)
			spec.ParamFloatCount = 2
			spec.ParamIntCount = 4
			
			// ARCHITECTURAL FIX: Initialize running statistics for BatchNorm layers
			if trackRunningStats {
				// Initialize running statistics (will be properly set by training engine)
				spec.RunningMean = make([]float32, numFeatures)
				spec.RunningVar = make([]float32, numFeatures)
				// Initialize running_var to 1.0 (standard initialization)
				for i := range spec.RunningVar {
					spec.RunningVar[i] = 1.0
				}
				spec.HasRunningStats = true
				
				// Extract from existing running statistics if available
				if layer.RunningStatistics != nil {
					if runningMean, exists := layer.RunningStatistics["running_mean"]; exists {
						copy(spec.RunningMean, runningMean)
					}
					if runningVar, exists := layer.RunningStatistics["running_var"]; exists {
						copy(spec.RunningVar, runningVar)
					}
				}
			}
			// Shape unchanged for BatchNorm

		default:
			return nil, fmt.Errorf("unsupported layer type for dynamic conversion: %s", layer.Type.String())
		}

		// Set output shape
		spec.OutputShape, spec.OutputShapeLen = copyShapeToArray(currentShape)
		specs[i] = spec
	}

	return specs, nil
}

// DynamicLayerSpec represents a layer specification compatible with the dynamic engine
type DynamicLayerSpec struct {
	LayerType       int32
	Name            string
	NameBytes       [64]byte  // C-compatible name storage
	InputShape      [4]int32
	InputShapeLen   int32
	OutputShape     [4]int32
	OutputShapeLen  int32
	ParamInt        [8]int32
	ParamFloat      [8]float32
	ParamIntCount   int32
	ParamFloatCount int32
	// Running statistics for layers like BatchNorm (non-learnable parameters)
	RunningMean     []float32
	RunningVar      []float32
	HasRunningStats bool
}

// Helper functions
func copyShapeToArray(shape []int) ([4]int32, int32) {
	var arr [4]int32
	count := int32(len(shape))
	if count > 4 {
		count = 4
	}
	for i := 0; i < int(count); i++ {
		arr[i] = int32(shape[i])
	}
	return arr, count
}

func boolToInt32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// ConvertToCGOLayerSpecs converts DynamicLayerSpec array to CGO-compatible format
// convertIntSliceToInt32Array converts []int to [4]int32 for CGO compatibility
func convertIntSliceToInt32Array(slice []int) [4]int32 {
	var arr [4]int32
	for i := 0; i < len(slice) && i < 4; i++ {
		arr[i] = int32(slice[i])
	}
	return arr
}

func ConvertToCGOLayerSpecs(dynamicSpecs []DynamicLayerSpec) []interface{} {
	// We need to import the cgo_bridge package to use LayerSpecC
	// For now, return interface{} and handle conversion in the engine
	specs := make([]interface{}, len(dynamicSpecs))
	for i, spec := range dynamicSpecs {
		specs[i] = map[string]interface{}{
			"layer_type":        spec.LayerType,
			"name_bytes":        spec.NameBytes,
			"input_shape":       spec.InputShape,
			"input_shape_len":   spec.InputShapeLen,
			"output_shape":      spec.OutputShape,
			"output_shape_len":  spec.OutputShapeLen,
			"param_int":         spec.ParamInt,
			"param_float":       spec.ParamFloat,
			"param_int_count":   spec.ParamIntCount,
			"param_float_count": spec.ParamFloatCount,
		}
	}
	return specs
}
