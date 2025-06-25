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