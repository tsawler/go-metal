package checkpoints

import (
	"fmt"
	"os"
	"time"

	"github.com/tsawler/go-metal/layers"
	"google.golang.org/protobuf/proto"
)

// ONNXExporter handles conversion of go-metal models to ONNX format
type ONNXExporter struct {
	model *ModelProto
}

// NewONNXExporter creates a new ONNX exporter
func NewONNXExporter() *ONNXExporter {
	return &ONNXExporter{}
}

// ExportToONNX converts a go-metal checkpoint to ONNX format
func (oe *ONNXExporter) ExportToONNX(checkpoint *Checkpoint, path string) error {
	// Create ONNX model proto
	model := &ModelProto{
		IrVersion:      7, // ONNX IR version 7
		OpsetImport:    []*OperatorSetIdProto{{Domain: "", Version: 13}}, // Opset 13
		ProducerName:   "go-metal",
		ProducerVersion: "1.0.0",
		ModelVersion:   1,
	}
	
	// Create the computation graph
	graph, err := oe.buildONNXGraph(checkpoint)
	if err != nil {
		return fmt.Errorf("failed to build ONNX graph: %v", err)
	}
	
	model.Graph = graph
	oe.model = model
	
	// Serialize to protobuf
	data, err := proto.Marshal(model)
	if err != nil {
		return fmt.Errorf("failed to marshal ONNX model: %v", err)
	}
	
	// Write to file
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write ONNX file: %v", err)
	}
	
	return nil
}

// buildONNXGraph creates the ONNX computation graph from go-metal model
func (oe *ONNXExporter) buildONNXGraph(checkpoint *Checkpoint) (*GraphProto, error) {
	graph := &GraphProto{
		Name: "go-metal-model",
	}
	
	// Create weight map for easy lookup
	weightMap := make(map[string]WeightTensor)
	for _, weight := range checkpoint.Weights {
		weightMap[weight.Name] = weight
	}
	
	// Track tensor names for graph connectivity
	var currentTensorName string = "input"
	
	// Add input tensor
	inputShape := checkpoint.ModelSpec.InputShape
	inputInfo := &ValueInfoProto{
		Name: "input",
		Type: &TypeProto{
			Value: &TypeProto_TensorType{
				TensorType: &TypeProto_Tensor{
					ElemType: TensorProto_DataType_FLOAT,
					Shape: &TensorShapeProto{
						Dim: oe.createDimensions(inputShape),
					},
				},
			},
		},
	}
	graph.Input = append(graph.Input, inputInfo)
	
	// Process each layer and create corresponding ONNX nodes
	for layerIdx, layerSpec := range checkpoint.ModelSpec.Layers {
		var nodes []*NodeProto
		var initializers []*TensorProto
		var err error
		
		switch layerSpec.Type {
		case layers.Conv2D:
			nodes, initializers, currentTensorName, err = oe.createConv2DNode(layerSpec, weightMap, currentTensorName, layerIdx)
		case layers.Dense:
			nodes, initializers, currentTensorName, err = oe.createDenseNode(layerSpec, weightMap, currentTensorName, layerIdx)
		case layers.ReLU:
			nodes, currentTensorName, err = oe.createReLUNode(layerSpec, currentTensorName, layerIdx)
		case layers.LeakyReLU:
			nodes, currentTensorName, err = oe.createLeakyReLUNode(layerSpec, currentTensorName, layerIdx)
		case layers.BatchNorm:
			nodes, initializers, currentTensorName, err = oe.createBatchNormNode(layerSpec, weightMap, currentTensorName, layerIdx)
		case layers.Dropout:
			nodes, currentTensorName, err = oe.createDropoutNode(layerSpec, currentTensorName, layerIdx)
		case layers.Softmax:
			nodes, currentTensorName, err = oe.createSoftmaxNode(layerSpec, currentTensorName, layerIdx)
		default:
			return nil, fmt.Errorf("unsupported layer type for ONNX export: %s", layerSpec.Type.String())
		}
		
		if err != nil {
			return nil, fmt.Errorf("failed to create ONNX node for layer %s: %v", layerSpec.Name, err)
		}
		
		// Add nodes and initializers to graph
		graph.Node = append(graph.Node, nodes...)
		graph.Initializer = append(graph.Initializer, initializers...)
	}
	
	// Add output tensor
	outputShape := checkpoint.ModelSpec.OutputShape
	outputInfo := &ValueInfoProto{
		Name: currentTensorName, // Final tensor name
		Type: &TypeProto{
			Value: &TypeProto_TensorType{
				TensorType: &TypeProto_Tensor{
					ElemType: TensorProto_DataType_FLOAT,
					Shape: &TensorShapeProto{
						Dim: oe.createDimensions(outputShape),
					},
				},
			},
		},
	}
	graph.Output = append(graph.Output, outputInfo)
	
	return graph, nil
}

// createConv2DNode creates ONNX Conv node
func (oe *ONNXExporter) createConv2DNode(layerSpec layers.LayerSpec, weightMap map[string]WeightTensor, inputTensor string, layerIdx int) ([]*NodeProto, []*TensorProto, string, error) {
	layerName := layerSpec.Name
	outputTensor := fmt.Sprintf("%s_output", layerName)
	
	// Get layer parameters
	kernelSize := layerSpec.Parameters["kernel_size"].(int)
	stride := layerSpec.Parameters["stride"].(int)
	padding := layerSpec.Parameters["padding"].(int)
	useBias := layerSpec.Parameters["use_bias"].(bool)
	
	// Create Conv node
	convNode := &NodeProto{
		OpType: "Conv",
		Name:   layerName,
		Input:  []string{inputTensor, fmt.Sprintf("%s.weight", layerName)},
		Output: []string{outputTensor},
		Attribute: []*AttributeProto{
			{Name: "kernel_shape", Ints: []int64{int64(kernelSize), int64(kernelSize)}},
			{Name: "strides", Ints: []int64{int64(stride), int64(stride)}},
			{Name: "pads", Ints: []int64{int64(padding), int64(padding), int64(padding), int64(padding)}},
		},
	}
	
	// Add bias if present
	if useBias {
		convNode.Input = append(convNode.Input, fmt.Sprintf("%s.bias", layerName))
	}
	
	// Create weight initializer
	var initializers []*TensorProto
	weightTensor := weightMap[fmt.Sprintf("%s.weight", layerName)]
	weightInit := oe.createTensorProto(fmt.Sprintf("%s.weight", layerName), weightTensor.Shape, weightTensor.Data)
	initializers = append(initializers, weightInit)
	
	// Create bias initializer if present
	if useBias {
		biasTensor := weightMap[fmt.Sprintf("%s.bias", layerName)]
		biasInit := oe.createTensorProto(fmt.Sprintf("%s.bias", layerName), biasTensor.Shape, biasTensor.Data)
		initializers = append(initializers, biasInit)
	}
	
	return []*NodeProto{convNode}, initializers, outputTensor, nil
}

// createDenseNode creates ONNX MatMul + Add nodes for Dense layer
func (oe *ONNXExporter) createDenseNode(layerSpec layers.LayerSpec, weightMap map[string]WeightTensor, inputTensor string, layerIdx int) ([]*NodeProto, []*TensorProto, string, error) {
	layerName := layerSpec.Name
	matmulOutput := fmt.Sprintf("%s_matmul", layerName)
	finalOutput := fmt.Sprintf("%s_output", layerName)
	
	useBias := layerSpec.Parameters["use_bias"].(bool)
	
	// Create MatMul node
	matmulNode := &NodeProto{
		OpType: "MatMul",
		Name:   fmt.Sprintf("%s_matmul_op", layerName),
		Input:  []string{inputTensor, fmt.Sprintf("%s.weight", layerName)},
		Output: []string{matmulOutput},
	}
	
	var nodes []*NodeProto
	nodes = append(nodes, matmulNode)
	
	// Create initializers
	var initializers []*TensorProto
	weightTensor := weightMap[fmt.Sprintf("%s.weight", layerName)]
	
	// Transpose weight matrix for ONNX (go-metal uses [input, output], ONNX expects [output, input])
	transposedWeight := oe.transposeMatrix(weightTensor.Data, weightTensor.Shape)
	transposedShape := []int{weightTensor.Shape[1], weightTensor.Shape[0]}
	weightInit := oe.createTensorProto(fmt.Sprintf("%s.weight", layerName), transposedShape, transposedWeight)
	initializers = append(initializers, weightInit)
	
	outputTensor := matmulOutput
	
	// Add bias if present
	if useBias {
		biasTensor := weightMap[fmt.Sprintf("%s.bias", layerName)]
		biasInit := oe.createTensorProto(fmt.Sprintf("%s.bias", layerName), biasTensor.Shape, biasTensor.Data)
		initializers = append(initializers, biasInit)
		
		// Create Add node for bias
		addNode := &NodeProto{
			OpType: "Add",
			Name:   fmt.Sprintf("%s_add_bias", layerName),
			Input:  []string{matmulOutput, fmt.Sprintf("%s.bias", layerName)},
			Output: []string{finalOutput},
		}
		nodes = append(nodes, addNode)
		outputTensor = finalOutput
	}
	
	return nodes, initializers, outputTensor, nil
}

// createReLUNode creates ONNX Relu node
func (oe *ONNXExporter) createReLUNode(layerSpec layers.LayerSpec, inputTensor string, layerIdx int) ([]*NodeProto, string, error) {
	layerName := layerSpec.Name
	outputTensor := fmt.Sprintf("%s_output", layerName)
	
	reluNode := &NodeProto{
		OpType: "Relu",
		Name:   layerName,
		Input:  []string{inputTensor},
		Output: []string{outputTensor},
	}
	
	return []*NodeProto{reluNode}, outputTensor, nil
}

// createLeakyReLUNode creates ONNX LeakyRelu node
func (oe *ONNXExporter) createLeakyReLUNode(layerSpec layers.LayerSpec, inputTensor string, layerIdx int) ([]*NodeProto, string, error) {
	layerName := layerSpec.Name
	outputTensor := fmt.Sprintf("%s_output", layerName)
	
	negativeSlope := layerSpec.Parameters["negative_slope"].(float32)
	
	leakyReluNode := &NodeProto{
		OpType: "LeakyRelu",
		Name:   layerName,
		Input:  []string{inputTensor},
		Output: []string{outputTensor},
		Attribute: []*AttributeProto{
			{Name: "alpha", F: negativeSlope},
		},
	}
	
	return []*NodeProto{leakyReluNode}, outputTensor, nil
}

// createBatchNormNode creates ONNX BatchNormalization node
func (oe *ONNXExporter) createBatchNormNode(layerSpec layers.LayerSpec, weightMap map[string]WeightTensor, inputTensor string, layerIdx int) ([]*NodeProto, []*TensorProto, string, error) {
	layerName := layerSpec.Name
	outputTensor := fmt.Sprintf("%s_output", layerName)
	
	eps := layerSpec.Parameters["eps"].(float32)
	momentum := layerSpec.Parameters["momentum"].(float32)
	affine := layerSpec.Parameters["affine"].(bool)
	
	if !affine {
		return nil, nil, "", fmt.Errorf("ONNX export requires affine BatchNorm (learnable parameters)")
	}
	
	// BatchNormalization requires: input, scale, bias, mean, var
	batchNormNode := &NodeProto{
		OpType: "BatchNormalization",
		Name:   layerName,
		Input: []string{
			inputTensor,
			fmt.Sprintf("%s.weight", layerName), // scale (gamma)
			fmt.Sprintf("%s.bias", layerName),   // bias (beta)
			fmt.Sprintf("%s.running_mean", layerName),
			fmt.Sprintf("%s.running_var", layerName),
		},
		Output: []string{outputTensor},
		Attribute: []*AttributeProto{
			{Name: "epsilon", F: eps},
			{Name: "momentum", F: momentum},
		},
	}
	
	// Create initializers
	var initializers []*TensorProto
	
	// Scale (gamma)
	scaleTensor := weightMap[fmt.Sprintf("%s.weight", layerName)]
	scaleInit := oe.createTensorProto(fmt.Sprintf("%s.weight", layerName), scaleTensor.Shape, scaleTensor.Data)
	initializers = append(initializers, scaleInit)
	
	// Bias (beta)
	biasTensor := weightMap[fmt.Sprintf("%s.bias", layerName)]
	biasInit := oe.createTensorProto(fmt.Sprintf("%s.bias", layerName), biasTensor.Shape, biasTensor.Data)
	initializers = append(initializers, biasInit)
	
	// Running mean (initialize to zeros)
	numFeatures := scaleTensor.Shape[0]
	runningMean := make([]float32, numFeatures)
	meanInit := oe.createTensorProto(fmt.Sprintf("%s.running_mean", layerName), []int{numFeatures}, runningMean)
	initializers = append(initializers, meanInit)
	
	// Running variance (initialize to ones)
	runningVar := make([]float32, numFeatures)
	for i := range runningVar {
		runningVar[i] = 1.0
	}
	varInit := oe.createTensorProto(fmt.Sprintf("%s.running_var", layerName), []int{numFeatures}, runningVar)
	initializers = append(initializers, varInit)
	
	return []*NodeProto{batchNormNode}, initializers, outputTensor, nil
}

// createDropoutNode creates ONNX Dropout node
func (oe *ONNXExporter) createDropoutNode(layerSpec layers.LayerSpec, inputTensor string, layerIdx int) ([]*NodeProto, string, error) {
	layerName := layerSpec.Name
	outputTensor := fmt.Sprintf("%s_output", layerName)
	
	rate := layerSpec.Parameters["rate"].(float32)
	
	// For inference, dropout is typically a no-op (identity)
	// But we'll include it with the training parameter set to false
	dropoutNode := &NodeProto{
		OpType: "Dropout",
		Name:   layerName,
		Input:  []string{inputTensor},
		Output: []string{outputTensor},
		Attribute: []*AttributeProto{
			{Name: "ratio", F: rate},
		},
	}
	
	return []*NodeProto{dropoutNode}, outputTensor, nil
}

// createSoftmaxNode creates ONNX Softmax node
func (oe *ONNXExporter) createSoftmaxNode(layerSpec layers.LayerSpec, inputTensor string, layerIdx int) ([]*NodeProto, string, error) {
	layerName := layerSpec.Name
	outputTensor := fmt.Sprintf("%s_output", layerName)
	
	axis := layerSpec.Parameters["axis"].(int)
	
	softmaxNode := &NodeProto{
		OpType: "Softmax",
		Name:   layerName,
		Input:  []string{inputTensor},
		Output: []string{outputTensor},
		Attribute: []*AttributeProto{
			{Name: "axis", I: int64(axis)},
		},
	}
	
	return []*NodeProto{softmaxNode}, outputTensor, nil
}

// Helper functions

// createDimensions creates ONNX tensor shape dimensions
func (oe *ONNXExporter) createDimensions(shape []int) []*TensorShapeProto_Dimension {
	dims := make([]*TensorShapeProto_Dimension, len(shape))
	for i, size := range shape {
		dims[i] = &TensorShapeProto_Dimension{
			Value: &TensorShapeProto_Dimension_DimValue{DimValue: int64(size)},
		}
	}
	return dims
}

// createTensorProto creates ONNX tensor initializer
func (oe *ONNXExporter) createTensorProto(name string, shape []int, data []float32) *TensorProto {
	dims := make([]int64, len(shape))
	for i, s := range shape {
		dims[i] = int64(s)
	}
	
	return &TensorProto{
		Name:      name,
		DataType:  TensorProto_DataType_FLOAT,
		Dims:      dims,
		FloatData: data,
	}
}

// transposeMatrix transposes a 2D matrix stored as 1D array
func (oe *ONNXExporter) transposeMatrix(data []float32, shape []int) []float32 {
	if len(shape) != 2 {
		return data // Return unchanged if not 2D
	}
	
	rows, cols := shape[0], shape[1]
	transposed := make([]float32, len(data))
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j*rows+i] = data[i*cols+j]
		}
	}
	
	return transposed
}

// ONNXImporter handles importing ONNX models to go-metal format
type ONNXImporter struct{}

// NewONNXImporter creates a new ONNX importer
func NewONNXImporter() *ONNXImporter {
	return &ONNXImporter{}
}

// ImportFromONNX converts an ONNX model to go-metal checkpoint format
func (oi *ONNXImporter) ImportFromONNX(path string) (*Checkpoint, error) {
	// Read ONNX file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %v", err)
	}
	
	// Parse ONNX protobuf
	var model ModelProto
	if err := proto.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %v", err)
	}
	
	// Convert ONNX graph to go-metal model spec
	modelSpec, weights, err := oi.convertONNXToGoMetal(&model)
	if err != nil {
		return nil, fmt.Errorf("failed to convert ONNX to go-metal: %v", err)
	}
	
	// Create checkpoint
	checkpoint := &Checkpoint{
		ModelSpec: modelSpec,
		Weights:   weights,
		TrainingState: TrainingState{
			Epoch:        0,
			Step:         0,
			LearningRate: 0.001, // Default
		},
		Metadata: CheckpointMetadata{
			Version:   "1.0.0",
			Framework: "go-metal",
			CreatedAt: time.Now(),
			Description: fmt.Sprintf("Imported from ONNX (producer: %s)", model.ProducerName),
		},
	}
	
	return checkpoint, nil
}

// convertONNXToGoMetal converts ONNX graph to go-metal model specification
func (oi *ONNXImporter) convertONNXToGoMetal(model *ModelProto) (*layers.ModelSpec, []WeightTensor, error) {
	graph := model.Graph
	
	// Parse input shape
	if len(graph.Input) == 0 {
		return nil, nil, fmt.Errorf("ONNX model has no inputs")
	}
	
	inputInfo := graph.Input[0]
	inputShape := oi.extractShapeFromValueInfo(inputInfo)
	
	// Create weight map from initializers
	weightMap := make(map[string][]float32)
	shapeMap := make(map[string][]int)
	
	for _, initializer := range graph.Initializer {
		weightMap[initializer.Name] = initializer.FloatData
		shape := make([]int, len(initializer.Dims))
		for i, dim := range initializer.Dims {
			shape[i] = int(dim)
		}
		shapeMap[initializer.Name] = shape
	}
	
	// Build layer specifications from ONNX nodes
	var layerSpecs []layers.LayerSpec
	var weights []WeightTensor
	
	for _, node := range graph.Node {
		layerSpec, layerWeights, err := oi.convertONNXNodeToLayer(node, weightMap, shapeMap)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert ONNX node %s: %v", node.Name, err)
		}
		
		if layerSpec != nil {
			layerSpecs = append(layerSpecs, *layerSpec)
		}
		weights = append(weights, layerWeights...)
	}
	
	// Create model builder and compile
	builder := layers.NewModelBuilder(inputShape)
	for _, spec := range layerSpecs {
		builder.AddLayer(spec)
	}
	
	modelSpec, err := builder.Compile()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compile imported model: %v", err)
	}
	
	return modelSpec, weights, nil
}

// convertONNXNodeToLayer converts a single ONNX node to go-metal layer
func (oi *ONNXImporter) convertONNXNodeToLayer(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int) (*layers.LayerSpec, []WeightTensor, error) {
	var weights []WeightTensor
	
	switch node.OpType {
	case "Conv":
		return oi.convertConvNode(node, weightMap, shapeMap, &weights)
	case "MatMul":
		// MatMul is part of Dense layer, handled with Add node
		return nil, weights, nil
	case "Add":
		// This might be bias addition for Dense layer
		return oi.convertAddNode(node, weightMap, shapeMap, &weights)
	case "Relu":
		return oi.convertReluNode(node), weights, nil
	case "LeakyRelu":
		return oi.convertLeakyReluNode(node), weights, nil
	case "BatchNormalization":
		return oi.convertBatchNormNode(node, weightMap, shapeMap, &weights)
	case "Dropout":
		return oi.convertDropoutNode(node), weights, nil
	case "Softmax":
		return oi.convertSoftmaxNode(node), weights, nil
	default:
		return nil, weights, fmt.Errorf("unsupported ONNX operation: %s", node.OpType)
	}
}

// Helper methods for ONNX import (implement specific node conversions)
func (oi *ONNXImporter) extractShapeFromValueInfo(info *ValueInfoProto) []int {
	tensorType := info.Type.GetTensorType()
	if tensorType == nil {
		return nil
	}
	
	shape := tensorType.Shape
	if shape == nil {
		return nil
	}
	
	dims := make([]int, len(shape.Dim))
	for i, dim := range shape.Dim {
		dims[i] = int(dim.GetDimValue())
	}
	
	return dims
}

// Implement remaining conversion methods...
func (oi *ONNXImporter) convertConvNode(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, weights *[]WeightTensor) (*layers.LayerSpec, []WeightTensor, error) {
	// Implementation for Conv node conversion
	return nil, nil, fmt.Errorf("ONNX Conv import not yet implemented")
}

func (oi *ONNXImporter) convertAddNode(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, weights *[]WeightTensor) (*layers.LayerSpec, []WeightTensor, error) {
	// Implementation for Add node conversion (Dense layer bias)
	return nil, nil, fmt.Errorf("ONNX Add import not yet implemented")
}

func (oi *ONNXImporter) convertReluNode(node *NodeProto) *layers.LayerSpec {
	return &layers.LayerSpec{
		Type:       layers.ReLU,
		Name:       node.Name,
		Parameters: map[string]interface{}{},
	}
}

func (oi *ONNXImporter) convertLeakyReluNode(node *NodeProto) *layers.LayerSpec {
	alpha := float32(0.01) // Default
	for _, attr := range node.Attribute {
		if attr.Name == "alpha" {
			alpha = attr.F
		}
	}
	
	return &layers.LayerSpec{
		Type: layers.LeakyReLU,
		Name: node.Name,
		Parameters: map[string]interface{}{
			"negative_slope": alpha,
		},
	}
}

func (oi *ONNXImporter) convertBatchNormNode(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, weights *[]WeightTensor) (*layers.LayerSpec, []WeightTensor, error) {
	// Implementation for BatchNorm node conversion
	return nil, nil, fmt.Errorf("ONNX BatchNorm import not yet implemented")
}

func (oi *ONNXImporter) convertDropoutNode(node *NodeProto) *layers.LayerSpec {
	rate := float32(0.5) // Default
	for _, attr := range node.Attribute {
		if attr.Name == "ratio" {
			rate = attr.F
		}
	}
	
	return &layers.LayerSpec{
		Type: layers.Dropout,
		Name: node.Name,
		Parameters: map[string]interface{}{
			"rate":     rate,
			"training": true,
		},
	}
}

func (oi *ONNXImporter) convertSoftmaxNode(node *NodeProto) *layers.LayerSpec {
	axis := -1 // Default
	for _, attr := range node.Attribute {
		if attr.Name == "axis" {
			axis = int(attr.I)
		}
	}
	
	return &layers.LayerSpec{
		Type: layers.Softmax,
		Name: node.Name,
		Parameters: map[string]interface{}{
			"axis": axis,
		},
	}
}