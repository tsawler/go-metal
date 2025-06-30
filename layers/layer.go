package layers

import (
	"fmt"
	"unsafe"

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
}

// ModelSpec defines a complete neural network model as layer configuration
// This replaces individual layer execution with unified model specification
type ModelSpec struct {
	Layers []LayerSpec `json:"layers"`
	
	// Compiled model information
	TotalParameters int64     `json:"total_parameters"`
	ParameterShapes [][]int   `json:"parameter_shapes"`
	InputShape      []int     `json:"input_shape"`
	OutputShape     []int     `json:"output_shape"`
	Compiled        bool      `json:"compiled"`
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

// ModelBuilder helps construct neural network models
type ModelBuilder struct {
	layers      []LayerSpec
	inputShape  []int
	compiled    bool
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

// Compile compiles the model and computes shapes and parameter counts
func (mb *ModelBuilder) Compile() (*ModelSpec, error) {
	if len(mb.layers) == 0 {
		return nil, fmt.Errorf("cannot compile empty model")
	}
	
	model := &ModelSpec{
		Layers:      make([]LayerSpec, len(mb.layers)),
		InputShape:  mb.inputShape,
		Compiled:    false,
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
	case ReLU, Softmax:
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
	
	// Compute input size from last dimension
	inputSize := inputShape[len(inputShape)-1]
	
	// Update layer parameters with computed input size
	layer.Parameters["input_size"] = inputSize
	
	// Output shape: same as input but last dimension becomes outputSize
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	outputShape[len(outputShape)-1] = outputSize
	
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
	outputHeight := (inputHeight + 2*padding - kernelSize) / stride + 1
	outputWidth := (inputWidth + 2*padding - kernelSize) / stride + 1
	
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
func (ms *ModelSpec) ValidateModelForTrainingEngine() error {
	if !ms.Compiled {
		return fmt.Errorf("model not compiled")
	}
	
	// Current TrainingEngine supports specific architectures
	// For now, validate against the existing hybrid CNN architecture
	
	if len(ms.Layers) == 0 {
		return fmt.Errorf("empty model")
	}
	
	// Check input shape compatibility (must be 4D for CNN)
	if len(ms.InputShape) != 4 {
		return fmt.Errorf("TrainingEngine currently requires 4D input [batch, channels, height, width]")
	}
	
	// Check that we have at least one Conv2D and one Dense layer
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
		return fmt.Errorf("TrainingEngine currently requires at least one Conv2D layer")
	}
	
	if !hasDense {
		return fmt.Errorf("TrainingEngine currently requires at least one Dense layer")
	}
	
	return nil
}

// GetTrainingEngineConfig generates configuration for the existing TrainingEngine
// This bridges the new layer system with the proven high-performance engine
func (ms *ModelSpec) GetTrainingEngineConfig() (map[string]interface{}, error) {
	if err := ms.ValidateModelForTrainingEngine(); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}
	
	config := map[string]interface{}{
		"model_type":        "cnn",
		"input_shape":       ms.InputShape,
		"output_shape":      ms.OutputShape,
		"total_parameters":  ms.TotalParameters,
		"parameter_shapes":  ms.ParameterShapes,
		"layer_count":       len(ms.Layers),
		"layers":            ms.Layers,
	}
	
	return config, nil
}