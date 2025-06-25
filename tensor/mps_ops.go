package tensor

import (
	"fmt"
	"github.com/tsawler/go-metal/metal_bridge"
)

// MPSGraphEngine provides high-level ML operations using MPSGraph
type MPSGraphEngine struct {
	device      *metal_bridge.Device
	graphDevice *metal_bridge.GraphDevice
	commandQueue *metal_bridge.CommandQueue
}

var mpsGraphEngine *MPSGraphEngine

// GetMPSGraphEngine returns the singleton MPSGraph engine
func GetMPSGraphEngine() (*MPSGraphEngine, error) {
	if mpsGraphEngine == nil {
		device := metal_bridge.CreateSystemDefaultDevice()
		if device == nil {
			return nil, fmt.Errorf("failed to create Metal device")
		}
		
		graphDevice := metal_bridge.NewGraphDevice(device)
		if graphDevice == nil {
			return nil, fmt.Errorf("failed to create MPSGraph device")
		}
		
		commandQueue := device.NewCommandQueue()
		if commandQueue == nil {
			return nil, fmt.Errorf("failed to create command queue")
		}
		
		mpsGraphEngine = &MPSGraphEngine{
			device:      device,
			graphDevice: graphDevice,
			commandQueue: commandQueue,
		}
	}
	return mpsGraphEngine, nil
}

// isCompatibleForOp checks if two tensors are compatible for element-wise operations
func isCompatibleForOp(a, b *Tensor) bool {
	// Check data types
	if a.DType != b.DType {
		return false
	}
	
	// Check shapes
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	
	for i, dim := range a.Shape {
		if dim != b.Shape[i] {
			return false
		}
	}
	
	return true
}

// convertDTypeToMPS converts Go tensor DType to MPSGraph data type
func convertDTypeToMPS(dtype DType) int {
	switch dtype {
	case Float32:
		return int(metal_bridge.MPSDataTypeFloat32)
	case Float16:
		return int(metal_bridge.MPSDataTypeFloat16)
	case Int32:
		return int(metal_bridge.MPSDataTypeInt32)
	default:
		return int(metal_bridge.MPSDataTypeFloat32) // Default fallback
	}
}

// AddMPS performs tensor addition using MPSGraph
func AddMPS(a, b *Tensor) (*Tensor, error) {
	if !isCompatibleForOp(a, b) {
		return nil, fmt.Errorf("tensors are not compatible for addition")
	}
	
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholders for inputs
	dataType := convertDTypeToMPS(a.DType)
	placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
	placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
	
	if placeholderA == nil || placeholderB == nil {
		return nil, fmt.Errorf("failed to create placeholder tensors")
	}
	
	// Create addition operation
	resultTensor := graph.Addition(placeholderA, placeholderB)
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create addition operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Create output tensor
	result := &Tensor{
		Shape:    a.Shape,
		Strides:  a.Strides,
		DType:    a.DType,
		Device:   GPU,
		NumElems: a.NumElems,
	}
	
	// Create Metal buffers directly from tensor data
	aBuffer, err := engine.device.CreateBufferWithBytes(a.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	
	bBuffer, err := engine.device.CreateBufferWithBytes(b.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		[]*metal_bridge.GraphTensor{placeholderA, placeholderB},
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}

// MatMulMPS performs matrix multiplication using MPSGraph
func MatMulMPS(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		return nil, fmt.Errorf("incompatible dimensions for matrix multiplication: (%d,%d) x (%d,%d)", 
			a.Shape[0], a.Shape[1], b.Shape[0], b.Shape[1])
	}
	if a.DType != b.DType {
		return nil, fmt.Errorf("tensors must have the same data type")
	}
	
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholders for inputs
	dataType := convertDTypeToMPS(a.DType)
	placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
	placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
	
	if placeholderA == nil || placeholderB == nil {
		return nil, fmt.Errorf("failed to create placeholder tensors")
	}
	
	// Create matrix multiplication operation
	resultTensor := graph.MatrixMultiplication(placeholderA, placeholderB)
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create matrix multiplication operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Create output tensor with correct shape
	outputShape := []int{a.Shape[0], b.Shape[1]}
	result := &Tensor{
		Shape:    outputShape,
		Strides:  calculateStrides(outputShape),
		DType:    a.DType,
		Device:   GPU,
		NumElems: outputShape[0] * outputShape[1],
	}
	
	// Create Metal buffers directly from tensor data
	aBuffer, err := engine.device.CreateBufferWithBytes(a.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	
	bBuffer, err := engine.device.CreateBufferWithBytes(b.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		[]*metal_bridge.GraphTensor{placeholderA, placeholderB},
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}

// ReLUMPS performs ReLU activation using MPSGraph
func ReLUMPS(a *Tensor) (*Tensor, error) {
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholder for input
	dataType := convertDTypeToMPS(a.DType)
	placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
	if placeholderA == nil {
		return nil, fmt.Errorf("failed to create placeholder tensor")
	}
	
	// Create ReLU operation
	resultTensor := graph.ReLU(placeholderA)
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create ReLU operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Create output tensor
	result := &Tensor{
		Shape:    a.Shape,
		Strides:  a.Strides,
		DType:    a.DType,
		Device:   GPU,
		NumElems: a.NumElems,
	}
	
	// Create Metal buffer directly from tensor data
	aBuffer, err := engine.device.CreateBufferWithBytes(a.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		[]*metal_bridge.GraphTensor{placeholderA},
		[]*metal_bridge.Buffer{aBuffer},
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}

// SigmoidMPS performs Sigmoid activation using MPSGraph
func SigmoidMPS(a *Tensor) (*Tensor, error) {
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholder for input
	dataType := convertDTypeToMPS(a.DType)
	placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
	if placeholderA == nil {
		return nil, fmt.Errorf("failed to create placeholder tensor")
	}
	
	// Create Sigmoid operation
	resultTensor := graph.Sigmoid(placeholderA)
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create Sigmoid operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Create output tensor
	result := &Tensor{
		Shape:    a.Shape,
		Strides:  a.Strides,
		DType:    a.DType,
		Device:   GPU,
		NumElems: a.NumElems,
	}
	
	// Create Metal buffer directly from tensor data
	aBuffer, err := engine.device.CreateBufferWithBytes(a.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		[]*metal_bridge.GraphTensor{placeholderA},
		[]*metal_bridge.Buffer{aBuffer},
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}

// calculateTensorSize calculates the size in bytes for a tensor
func calculateTensorSize(shape []int, dtype DType) uintptr {
	numElems := 1
	for _, dim := range shape {
		numElems *= dim
	}
	
	var elemSize int
	switch dtype {
	case Float32:
		elemSize = 4
	case Float16:
		elemSize = 2
	case Int32:
		elemSize = 4
	default:
		elemSize = 4 // Default to Float32
	}
	
	return uintptr(numElems * elemSize)
}

// Conv2DMPS performs 2D convolution using MPSGraph
func Conv2DMPS(input, weights, bias *Tensor, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) (*Tensor, error) {
	// Validate input tensors
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (NCHW), got %d dimensions", len(input.Shape))
	}
	if len(weights.Shape) != 4 {
		return nil, fmt.Errorf("weights tensor must be 4D (OIHW), got %d dimensions", len(weights.Shape))
	}
	if input.DType != weights.DType {
		return nil, fmt.Errorf("input and weights must have the same data type")
	}
	if bias != nil && (bias.DType != input.DType || len(bias.Shape) != 1) {
		return nil, fmt.Errorf("bias must be 1D and have the same data type as input")
	}
	
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholders
	dataType := convertDTypeToMPS(input.DType)
	placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
	placeholderWeights := graph.PlaceholderTensor(weights.Shape, dataType)
	
	var placeholderBias *metal_bridge.GraphTensor
	var inputTensors []*metal_bridge.GraphTensor
	if bias != nil {
		placeholderBias = graph.PlaceholderTensor(bias.Shape, dataType)
		inputTensors = []*metal_bridge.GraphTensor{placeholderInput, placeholderWeights, placeholderBias}
	} else {
		inputTensors = []*metal_bridge.GraphTensor{placeholderInput, placeholderWeights}
	}
	
	if placeholderInput == nil || placeholderWeights == nil {
		return nil, fmt.Errorf("failed to create placeholder tensors")
	}
	
	// Create convolution operation
	resultTensor := graph.Conv2D(placeholderInput, placeholderWeights, placeholderBias, 
		1, 1, // strideX, strideY (using 1,1 for simplicity, can be parameterized later)
		1, 1, // dilationX, dilationY
		paddingLeft, paddingRight, paddingTop, paddingBottom,
		1) // groups
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create convolution operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, inputTensors, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Calculate output shape
	N := input.Shape[0]     // Batch size
	C_out := weights.Shape[0] // Output channels
	H := input.Shape[2]     // Input height
	W := input.Shape[3]     // Input width
	kernelH := weights.Shape[2]
	kernelW := weights.Shape[3]
	
	outputH := (H-kernelH+paddingTop+paddingBottom)/strideY + 1
	outputW := (W-kernelW+paddingLeft+paddingRight)/strideX + 1
	outputShape := []int{N, C_out, outputH, outputW}
	
	// Create output tensor
	result := &Tensor{
		Shape:    outputShape,
		Strides:  calculateStrides(outputShape),
		DType:    input.DType,
		Device:   GPU,
		NumElems: N * C_out * outputH * outputW,
	}
	
	// Create Metal buffers
	inputBuffer, err := engine.device.CreateBufferWithBytes(input.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	
	weightsBuffer, err := engine.device.CreateBufferWithBytes(weights.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for weights: %v", err)
	}
	
	var inputBuffers []*metal_bridge.Buffer
	if bias != nil {
		biasBuffer, err := engine.device.CreateBufferWithBytes(bias.Data, metal_bridge.ResourceStorageModeShared)
		if err != nil {
			return nil, fmt.Errorf("failed to create buffer for bias: %v", err)
		}
		inputBuffers = []*metal_bridge.Buffer{inputBuffer, weightsBuffer, biasBuffer}
	} else {
		inputBuffers = []*metal_bridge.Buffer{inputBuffer, weightsBuffer}
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		inputTensors,
		inputBuffers,
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}

// MaxPool2DMPS performs 2D max pooling using MPSGraph
func MaxPool2DMPS(input *Tensor, kernelSize, stride, padding int) (*Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (NCHW), got %d dimensions", len(input.Shape))
	}
	
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholder
	dataType := convertDTypeToMPS(input.DType)
	placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
	if placeholderInput == nil {
		return nil, fmt.Errorf("failed to create placeholder tensor")
	}
	
	// Create max pooling operation
	resultTensor := graph.MaxPool2D(placeholderInput, 
		kernelSize, kernelSize, // kernelWidth, kernelHeight
		stride, stride,         // strideX, strideY
		padding, padding, padding, padding) // paddingLeft, paddingRight, paddingTop, paddingBottom
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create max pooling operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Calculate output shape
	N := input.Shape[0] // Batch size
	C := input.Shape[1] // Channels
	H := input.Shape[2] // Input height
	W := input.Shape[3] // Input width
	
	outputH := (H-kernelSize+2*padding)/stride + 1
	outputW := (W-kernelSize+2*padding)/stride + 1
	outputShape := []int{N, C, outputH, outputW}
	
	// Create output tensor
	result := &Tensor{
		Shape:    outputShape,
		Strides:  calculateStrides(outputShape),
		DType:    input.DType,
		Device:   GPU,
		NumElems: N * C * outputH * outputW,
	}
	
	// Create Metal buffer
	inputBuffer, err := engine.device.CreateBufferWithBytes(input.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		[]*metal_bridge.GraphTensor{placeholderInput},
		[]*metal_bridge.Buffer{inputBuffer},
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}

// AvgPool2DMPS performs 2D average pooling using MPSGraph
func AvgPool2DMPS(input *Tensor, kernelSize, stride, padding int) (*Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (NCHW), got %d dimensions", len(input.Shape))
	}
	
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create graph
	graph := metal_bridge.NewGraph()
	if graph == nil {
		return nil, fmt.Errorf("failed to create MPSGraph")
	}
	
	// Create placeholder
	dataType := convertDTypeToMPS(input.DType)
	placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
	if placeholderInput == nil {
		return nil, fmt.Errorf("failed to create placeholder tensor")
	}
	
	// Create average pooling operation
	resultTensor := graph.AvgPool2D(placeholderInput, 
		kernelSize, kernelSize, // kernelWidth, kernelHeight
		stride, stride,         // strideX, strideY
		padding, padding, padding, padding) // paddingLeft, paddingRight, paddingTop, paddingBottom
	if resultTensor == nil {
		return nil, fmt.Errorf("failed to create average pooling operation")
	}
	
	// Compile graph
	compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
	if compilationDescriptor == nil {
		return nil, fmt.Errorf("failed to create compilation descriptor")
	}
	
	executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
	if executable == nil {
		return nil, fmt.Errorf("failed to compile graph")
	}
	
	// Calculate output shape (same as MaxPool2D)
	N := input.Shape[0] // Batch size
	C := input.Shape[1] // Channels
	H := input.Shape[2] // Input height
	W := input.Shape[3] // Input width
	
	outputH := (H-kernelSize+2*padding)/stride + 1
	outputW := (W-kernelSize+2*padding)/stride + 1
	outputShape := []int{N, C, outputH, outputW}
	
	// Create output tensor
	result := &Tensor{
		Shape:    outputShape,
		Strides:  calculateStrides(outputShape),
		DType:    input.DType,
		Device:   GPU,
		NumElems: N * C * outputH * outputW,
	}
	
	// Create Metal buffer
	inputBuffer, err := engine.device.CreateBufferWithBytes(input.Data, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	
	// Allocate result buffer
	resultSize := calculateTensorSize(result.Shape, result.DType)
	resultBuffer, err := engine.device.CreateBufferWithLength(resultSize, metal_bridge.ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}
	
	// Execute graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	executable.Execute(engine.commandQueue,
		[]*metal_bridge.GraphTensor{placeholderInput},
		[]*metal_bridge.Buffer{inputBuffer},
		[]*metal_bridge.GraphTensor{resultTensor},
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)
	
	// Copy result data from Metal buffer to CPU slice
	switch result.DType {
	case Float32:
		resultData := make([]float32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsFloat32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	case Int32:
		resultData := make([]int32, result.NumElems)
		bufferContents := resultBuffer.ContentsAsInt32()
		copy(resultData, bufferContents[:result.NumElems])
		result.Data = resultData
	default:
		return nil, fmt.Errorf("unsupported data type for result copying: %v", result.DType)
	}
	
	return result, nil
}