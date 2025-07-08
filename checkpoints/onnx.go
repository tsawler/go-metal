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
	
	// First pass: identify MatMul+Add pairs for bias absorption
	matmulAddPairs := oi.identifyMatMulAddPairs(graph.Node)
	
	for i, node := range graph.Node {
		// Skip Add nodes that are part of MatMul+Add pairs
		if oi.isAddNodeInPair(node, matmulAddPairs) {
			continue
		}
		
		layerSpec, layerWeights, err := oi.convertONNXNodeToLayer(node, weightMap, shapeMap)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert ONNX node %s: %v", node.Name, err)
		}
		
		// For MatMul nodes that are part of a pair, add the bias weight
		if node.OpType == "MatMul" {
			if addNode, hasBias := matmulAddPairs[i]; hasBias {
				// Add bias tensor for this MatMul
				biasWeights, err := oi.extractBiasFromAddNode(addNode, weightMap, shapeMap, node.Name)
				if err != nil {
					return nil, nil, fmt.Errorf("failed to extract bias from Add node %s: %v", addNode.Name, err)
				}
				layerWeights = append(layerWeights, biasWeights...)
				
				// Update layer spec to use bias
				if layerSpec != nil {
					layerSpec.Parameters["use_bias"] = true
				}
			}
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
	
	// Debug: log the operation type for troubleshooting
	// fmt.Printf("Converting ONNX node: %s (type: %s)\n", node.Name, node.OpType)
	
	switch node.OpType {
	case "Conv":
		return oi.convertConvNode(node, weightMap, shapeMap, &weights)
	case "MatMul":
		// MatMul represents Dense layer in ONNX - convert directly
		return oi.convertMatMulNode(node, weightMap, shapeMap, &weights)
	case "Add":
		// Add nodes are typically bias additions, handled as part of Dense layers
		// or standalone element-wise additions - analyze context
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
	case "Flatten", "flatten":
		return oi.convertFlattenNode(node), weights, nil
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

// convertMatMulNode converts ONNX MatMul to go-metal Dense layer
func (oi *ONNXImporter) convertMatMulNode(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, weights *[]WeightTensor) (*layers.LayerSpec, []WeightTensor, error) {
	// ONNX MatMul inputs: [X, W] where X=input, W=weight matrix
	if len(node.Input) < 2 {
		return nil, nil, fmt.Errorf("MatMul node %s: expected 2 inputs [X, W], got %d", node.Name, len(node.Input))
	}
	
	weightName := node.Input[1]
	weightData, exists := weightMap[weightName]
	if !exists {
		return nil, nil, fmt.Errorf("MatMul node %s: weight tensor %s not found", node.Name, weightName)
	}
	
	weightShape, exists := shapeMap[weightName]
	if !exists {
		return nil, nil, fmt.Errorf("MatMul node %s: weight shape for %s not found", node.Name, weightName)
	}
	
	// Validate weight shape 
	if len(weightShape) != 2 {
		return nil, nil, fmt.Errorf("MatMul node %s: weight tensor must be 2D, got shape %v", node.Name, weightShape)
	}
	
	// ONNX MatMul weight format analysis:
	// Based on the shapes we see: [32, 32768] and [2, 32]
	// This suggests ONNX stores weights as [output_size, input_size]
	// But go-metal expects [input_size, output_size]
	// We need to transpose the interpretation
	outputSize := weightShape[0]  
	inputSize := weightShape[1]   
	
	// Debug: log weight shape for troubleshooting  
	// fmt.Printf("MatMul node %s: weight shape %v -> input_size=%d, output_size=%d (transposed interpretation)\n", 
	//	node.Name, weightShape, inputSize, outputSize)
	
	var layerWeights []WeightTensor
	
	// Create weight tensor (GPU-resident principle)
	// Note: ONNX uses [output, input] but go-metal expects [input, output]
	// We need to transpose both the shape and the data
	transposedShape := []int{inputSize, outputSize}
	transposedData := transposeMatrix2D(weightData, weightShape[0], weightShape[1])
	
	weightTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.weight", node.Name),
		Shape: transposedShape,
		Data:  transposedData,
		Layer: node.Name,
		Type:  "weight",
	}
	layerWeights = append(layerWeights, weightTensor)
	
	// Create go-metal Dense layer specification
	// Note: useBias=false for pure MatMul, bias handled by subsequent Add node
	layerSpec := &layers.LayerSpec{
		Type: layers.Dense,
		Name: node.Name,
		Parameters: map[string]interface{}{
			"input_size":  inputSize,
			"output_size": outputSize,
			"use_bias":    false, // Pure MatMul, bias added separately via Add node
		},
	}
	
	return layerSpec, layerWeights, nil
}

func (oi *ONNXImporter) convertConvNode(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, weights *[]WeightTensor) (*layers.LayerSpec, []WeightTensor, error) {
	// Extract Conv attributes from ONNX node
	var kernelShape []int64
	var strides []int64
	var pads []int64
	
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "kernel_shape":
			kernelShape = attr.Ints
		case "strides":
			strides = attr.Ints
		case "pads":
			pads = attr.Ints
		}
	}
	
	// Validate Conv parameters
	if len(kernelShape) != 2 {
		return nil, nil, fmt.Errorf("Conv node %s: only 2D convolutions supported, got kernel_shape %v", node.Name, kernelShape)
	}
	if len(strides) != 2 {
		return nil, nil, fmt.Errorf("Conv node %s: strides must be 2D, got %v", node.Name, strides)
	}
	if len(pads) != 4 {
		return nil, nil, fmt.Errorf("Conv node %s: pads must be [top, left, bottom, right], got %v", node.Name, pads)
	}
	
	// Validate uniform padding (go-metal supports uniform padding only)
	if pads[0] != pads[1] || pads[1] != pads[2] || pads[2] != pads[3] {
		return nil, nil, fmt.Errorf("Conv node %s: only uniform padding supported, got %v", node.Name, pads)
	}
	
	// Validate square kernels and uniform strides
	if kernelShape[0] != kernelShape[1] {
		return nil, nil, fmt.Errorf("Conv node %s: only square kernels supported, got %v", node.Name, kernelShape)
	}
	if strides[0] != strides[1] {
		return nil, nil, fmt.Errorf("Conv node %s: only uniform strides supported, got %v", node.Name, strides)
	}
	
	// Extract weight tensor (ONNX format: [output_channels, input_channels, kernel_h, kernel_w])
	if len(node.Input) < 2 {
		return nil, nil, fmt.Errorf("Conv node %s: expected at least 2 inputs (input, weight), got %d", node.Name, len(node.Input))
	}
	
	weightName := node.Input[1]
	weightData, exists := weightMap[weightName]
	if !exists {
		return nil, nil, fmt.Errorf("Conv node %s: weight tensor %s not found", node.Name, weightName)
	}
	
	weightShape, exists := shapeMap[weightName]
	if !exists {
		return nil, nil, fmt.Errorf("Conv node %s: weight shape for %s not found", node.Name, weightName)
	}
	
	// Validate weight shape [out_channels, in_channels, kernel_h, kernel_w]
	if len(weightShape) != 4 {
		return nil, nil, fmt.Errorf("Conv node %s: weight tensor must be 4D, got shape %v", node.Name, weightShape)
	}
	
	outputChannels := weightShape[0]
	inputChannels := weightShape[1]
	kernelH := weightShape[2]
	kernelW := weightShape[3]
	
	// Validate kernel dimensions match attributes
	if int64(kernelH) != kernelShape[0] || int64(kernelW) != kernelShape[1] {
		return nil, nil, fmt.Errorf("Conv node %s: kernel shape mismatch: weight %dx%d vs attribute %v", 
			node.Name, kernelH, kernelW, kernelShape)
	}
	
	// Check for bias (optional third input)
	useBias := len(node.Input) >= 3
	var layerWeights []WeightTensor
	
	// Create weight tensor (GPU-resident principle - data stays as-is, will be copied to GPU)
	weightTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.weight", node.Name),
		Shape: weightShape,
		Data:  weightData,
		Layer: node.Name,
		Type:  "weight",
	}
	layerWeights = append(layerWeights, weightTensor)
	
	// Handle bias if present
	if useBias {
		biasName := node.Input[2]
		biasData, exists := weightMap[biasName]
		if !exists {
			return nil, nil, fmt.Errorf("Conv node %s: bias tensor %s not found", node.Name, biasName)
		}
		
		biasShape, exists := shapeMap[biasName]
		if !exists {
			return nil, nil, fmt.Errorf("Conv node %s: bias shape for %s not found", node.Name, biasName)
		}
		
		// Validate bias shape [output_channels]
		if len(biasShape) != 1 || biasShape[0] != outputChannels {
			return nil, nil, fmt.Errorf("Conv node %s: bias shape must be [%d], got %v", 
				node.Name, outputChannels, biasShape)
		}
		
		biasTensor := WeightTensor{
			Name:  fmt.Sprintf("%s.bias", node.Name),
			Shape: biasShape,
			Data:  biasData,
			Layer: node.Name,
			Type:  "bias",
		}
		layerWeights = append(layerWeights, biasTensor)
	}
	
	// Create go-metal Conv2D layer specification (adhering to layer factory design)
	layerSpec := &layers.LayerSpec{
		Type: layers.Conv2D,
		Name: node.Name,
		Parameters: map[string]interface{}{
			"input_channels":  inputChannels,
			"output_channels": outputChannels,
			"kernel_size":     int(kernelShape[0]),
			"stride":          int(strides[0]),
			"padding":         int(pads[0]),
			"use_bias":        useBias,
		},
	}
	
	return layerSpec, layerWeights, nil
}

func (oi *ONNXImporter) convertAddNode(node *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, weights *[]WeightTensor) (*layers.LayerSpec, []WeightTensor, error) {
	// Add nodes in ONNX are typically used for bias addition in Dense layers
	// In go-metal, bias is handled as part of the Dense layer itself
	// So we don't create a separate layer for Add - it's absorbed into Dense layer
	
	// This means Add nodes are processed during Dense layer construction
	// and don't generate standalone layers in go-metal architecture
	
	// Return nil to indicate this node doesn't create a layer
	// (GPU-resident principle: minimize separate operations, fuse into Dense layer)
	return nil, nil, nil
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
	// ONNX BatchNormalization inputs: [X, scale, B, input_mean, input_var]
	// where X=input, scale=gamma, B=beta, input_mean=running_mean, input_var=running_var
	
	if len(node.Input) < 5 {
		return nil, nil, fmt.Errorf("BatchNorm node %s: expected 5 inputs [X, scale, B, mean, var], got %d", 
			node.Name, len(node.Input))
	}
	
	// Extract BatchNorm attributes
	epsilon := float32(1e-5) // Default epsilon
	momentum := float32(0.9) // Default momentum
	
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "epsilon":
			epsilon = attr.F
		case "momentum":
			momentum = attr.F
		}
	}
	
	// Extract weight tensors
	scaleName := node.Input[1]    // gamma (scale parameter)
	biasName := node.Input[2]     // beta (bias parameter)
	meanName := node.Input[3]     // running mean
	varName := node.Input[4]      // running variance
	
	var layerWeights []WeightTensor
	
	// Validate and extract scale (gamma) tensor
	scaleData, exists := weightMap[scaleName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: scale tensor %s not found", node.Name, scaleName)
	}
	scaleShape, exists := shapeMap[scaleName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: scale shape for %s not found", node.Name, scaleName)
	}
	
	// Validate scale shape [num_features]
	if len(scaleShape) != 1 {
		return nil, nil, fmt.Errorf("BatchNorm node %s: scale tensor must be 1D, got shape %v", node.Name, scaleShape)
	}
	numFeatures := scaleShape[0]
	
	// Create gamma (scale) tensor - use .weight naming for compatibility
	gammaTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.weight", node.Name),
		Shape: scaleShape,
		Data:  scaleData,
		Layer: node.Name,
		Type:  "weight",
	}
	layerWeights = append(layerWeights, gammaTensor)
	
	// Extract and validate bias (beta) tensor
	biasData, exists := weightMap[biasName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: bias tensor %s not found", node.Name, biasName)
	}
	biasShape, exists := shapeMap[biasName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: bias shape for %s not found", node.Name, biasName)
	}
	
	// Validate bias shape matches scale
	if len(biasShape) != 1 || biasShape[0] != numFeatures {
		return nil, nil, fmt.Errorf("BatchNorm node %s: bias shape must be [%d], got %v", 
			node.Name, numFeatures, biasShape)
	}
	
	// Create beta (bias) tensor - use .bias naming for compatibility
	betaTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.bias", node.Name),
		Shape: biasShape,
		Data:  biasData,
		Layer: node.Name,
		Type:  "bias",
	}
	layerWeights = append(layerWeights, betaTensor)
	
	// Validate that running mean tensor exists and has correct shape
	_, exists = weightMap[meanName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: mean tensor %s not found", node.Name, meanName)
	}
	meanShape, exists := shapeMap[meanName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: mean shape for %s not found", node.Name, meanName)
	}
	
	// Validate mean shape
	if len(meanShape) != 1 || meanShape[0] != numFeatures {
		return nil, nil, fmt.Errorf("BatchNorm node %s: mean shape must be [%d], got %v", 
			node.Name, numFeatures, meanShape)
	}
	
	// Note: For inference mode, running_mean and running_var are typically
	// not loaded as separate weight tensors but are used to pre-compute
	// the normalization parameters and fold them into the weight and bias.
	// However, for compatibility with the go-metal engine, we include them
	// as separate tensors but don't count them toward the parameter count.
	
	// UNIFIED SOLUTION: Extract running mean and variance data for inference mode
	// These will be fed to BatchNorm placeholders during inference execution
	runningMeanData, exists := weightMap[meanName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: mean tensor %s not found", node.Name, meanName)
	}
	runningVarData, exists := weightMap[varName]
	if !exists {
		return nil, nil, fmt.Errorf("BatchNorm node %s: variance tensor %s not found", node.Name, varName)
	}
	
	// Create running statistics tensors - these are NOT counted as learnable parameters
	// but are needed to feed the BatchNorm placeholders during inference
	runningMeanTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.running_mean", node.Name),
		Shape: meanShape,
		Data:  runningMeanData,
		Layer: node.Name,
		Type:  "running_mean",
	}
	runningVarTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.running_var", node.Name),
		Shape: meanShape, // Same shape as mean
		Data:  runningVarData,
		Layer: node.Name,
		Type:  "running_var",
	}
	
	// Add running statistics to layer weights (they will be handled separately from learnable parameters)
	layerWeights = append(layerWeights, runningMeanTensor)
	layerWeights = append(layerWeights, runningVarTensor)
	
	// Create go-metal BatchNorm layer specification 
	// (MPSGraph-centric: will use MPSGraph batch normalization operations)
	layerSpec := &layers.LayerSpec{
		Type: layers.BatchNorm,
		Name: node.Name,
		Parameters: map[string]interface{}{
			"num_features": numFeatures,
			"epsilon":      epsilon,
			"momentum":     momentum,
			"affine":       true,  // Always true since we have gamma and beta
			"track_running_stats": true, // Always true since we have running stats
			"training":     false, // INFERENCE MODE: Use pre-trained running stats, not batch stats
		},
		// ARCHITECTURAL FIX: Embed running statistics in LayerSpec for graph construction
		RunningStatistics: map[string][]float32{
			"running_mean": runningMeanData,
			"running_var":  runningVarData,
		},
	}
	
	return layerSpec, layerWeights, nil
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
			"training": false, // INFERENCE MODE: Disable dropout during inference
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

// convertFlattenNode handles ONNX Flatten operations
func (oi *ONNXImporter) convertFlattenNode(node *NodeProto) *layers.LayerSpec {
	// In go-metal, flattening is handled automatically by Dense layers
	// when they receive higher-dimensional input (like 4D from Conv2D)
	// So Flatten operations don't need to create actual layers
	// 
	// GPU-resident principle: minimize operations by fusing Flatten into Dense layer
	// MPSGraph-centric: let MPSGraph handle tensor reshaping automatically
	
	// Return nil to indicate this operation doesn't generate a standalone layer
	return nil
}

// transposeMatrix2D transposes a 2D matrix stored as 1D array
func transposeMatrix2D(data []float32, rows, cols int) []float32 {
	transposed := make([]float32, len(data))
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Original: data[i*cols + j] 
			// Transposed: transposed[j*rows + i]
			transposed[j*rows+i] = data[i*cols+j]
		}
	}
	
	return transposed
}

// identifyMatMulAddPairs identifies MatMul+Add pairs for bias absorption
func (oi *ONNXImporter) identifyMatMulAddPairs(nodes []*NodeProto) map[int]*NodeProto {
	pairs := make(map[int]*NodeProto)
	
	for i, node := range nodes {
		if node.OpType == "MatMul" && i+1 < len(nodes) {
			nextNode := nodes[i+1]
			if nextNode.OpType == "Add" {
				// Check if the Add node takes the MatMul output as input
				if len(nextNode.Input) >= 2 && len(node.Output) >= 1 {
					if nextNode.Input[0] == node.Output[0] {
						pairs[i] = nextNode
					}
				}
			}
		}
	}
	
	return pairs
}

// isAddNodeInPair checks if an Add node is part of a MatMul+Add pair
func (oi *ONNXImporter) isAddNodeInPair(node *NodeProto, pairs map[int]*NodeProto) bool {
	if node.OpType != "Add" {
		return false
	}
	
	for _, addNode := range pairs {
		if addNode.Name == node.Name {
			return true
		}
	}
	
	return false
}

// extractBiasFromAddNode extracts bias weights from an Add node
func (oi *ONNXImporter) extractBiasFromAddNode(addNode *NodeProto, weightMap map[string][]float32, shapeMap map[string][]int, matmulName string) ([]WeightTensor, error) {
	var biasWeights []WeightTensor
	
	// Add node should have 2 inputs: [matmul_output, bias_tensor]
	if len(addNode.Input) < 2 {
		return nil, fmt.Errorf("Add node %s: expected 2 inputs, got %d", addNode.Name, len(addNode.Input))
	}
	
	// The bias tensor is the second input
	biasName := addNode.Input[1]
	biasData, exists := weightMap[biasName]
	if !exists {
		return nil, fmt.Errorf("Add node %s: bias tensor %s not found", addNode.Name, biasName)
	}
	
	biasShape, exists := shapeMap[biasName]
	if !exists {
		return nil, fmt.Errorf("Add node %s: bias shape for %s not found", addNode.Name, biasName)
	}
	
	// Create bias tensor with MatMul layer name
	biasTensor := WeightTensor{
		Name:  fmt.Sprintf("%s.bias", matmulName),
		Shape: biasShape,
		Data:  biasData,
		Layer: matmulName,
		Type:  "bias",
	}
	biasWeights = append(biasWeights, biasTensor)
	
	return biasWeights, nil
}