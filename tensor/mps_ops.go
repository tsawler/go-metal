package tensor

import (
	"fmt"
	"sync"
	"unsafe"
	"github.com/tsawler/go-metal/metal_bridge"
)

// cachedGraph holds all the necessary components for a reusable, compiled graph.
// This prevents re-creation and re-compilation for identical operations.
type cachedGraph struct {
	executable    *metal_bridge.GraphExecutable
	inputTensors  []*metal_bridge.GraphTensor
	resultTensors []*metal_bridge.GraphTensor
}

// MPSGraphEngine provides high-level ML operations using MPSGraph.
// It now includes a thread-safe cache for compiled graph executables.
type MPSGraphEngine struct {
	device       *metal_bridge.Device
	graphDevice  *metal_bridge.GraphDevice
	commandQueue *metal_bridge.CommandQueue

	// Cache for compiled graph executables and their tensors.
	cacheMutex sync.Mutex
	graphCache map[string]*cachedGraph
}

var mpsGraphEngine *MPSGraphEngine
var once sync.Once

// GetMPSGraphEngine returns the singleton MPSGraph engine, initialized safely.
func GetMPSGraphEngine() (*MPSGraphEngine, error) {
	var err error
	once.Do(func() {
		device := metal_bridge.CreateSystemDefaultDevice()
		if device == nil {
			err = fmt.Errorf("failed to create Metal device")
			return
		}

		graphDevice := metal_bridge.NewGraphDevice(device)
		if graphDevice == nil {
			err = fmt.Errorf("failed to create MPSGraph device")
			return
		}

		commandQueue := device.NewCommandQueue()
		if commandQueue == nil {
			err = fmt.Errorf("failed to create command queue")
			return
		}

		mpsGraphEngine = &MPSGraphEngine{
			device:       device,
			graphDevice:  graphDevice,
			commandQueue: commandQueue,
			cacheMutex:   sync.Mutex{},
			graphCache:   make(map[string]*cachedGraph),
		}
	})
	return mpsGraphEngine, err
}

// generateCacheKey creates a unique key for an operation based on its name and tensor properties.
func generateCacheKey(opName string, tensors ...*Tensor) string {
	key := opName
	for _, t := range tensors {
		if t == nil {
			key += ":nil"
		} else {
			key += fmt.Sprintf(":%s:%v", t.DType, t.Shape)
		}
	}
	return key
}

// getOrCreateGraph handles the caching logic for MPS graphs.
func (engine *MPSGraphEngine) getOrCreateGraph(key string, createGraphFunc func() *cachedGraph) (*cachedGraph, error) {
	engine.cacheMutex.Lock()
	if graph, found := engine.graphCache[key]; found {
		engine.cacheMutex.Unlock()
		return graph, nil
	}
	engine.cacheMutex.Unlock()

	// If not found, create and compile the graph.
	newGraph := createGraphFunc()
	if newGraph == nil || newGraph.executable == nil {
		return nil, fmt.Errorf("failed to create and compile graph for key: %s", key)
	}

	// Store the entire cachedGraph object.
	engine.cacheMutex.Lock()
	engine.graphCache[key] = newGraph
	engine.cacheMutex.Unlock()

	return newGraph, nil
}

// executeGraph executes a cached graph with the given input tensors
func (engine *MPSGraphEngine) executeGraph(cachedGraphObj *cachedGraph, inputs []*Tensor) ([]*Tensor, error) {
	if len(inputs) != len(cachedGraphObj.inputTensors) {
		return nil, fmt.Errorf("input tensor count mismatch: expected %d, got %d", len(cachedGraphObj.inputTensors), len(inputs))
	}
	
	// Create input buffers for the tensors
	inputBuffers := make([]*metal_bridge.Buffer, len(inputs))
	for i, tensor := range inputs {
		buffer, err := engine.createBufferFromTensor(tensor)
		if err != nil {
			return nil, fmt.Errorf("failed to create input buffer %d: %v", i, err)
		}
		inputBuffers[i] = buffer
	}
	
	// Create output buffers based on result tensor shapes
	resultBuffers := make([]*metal_bridge.Buffer, len(cachedGraphObj.resultTensors))
	resultTensors := make([]*Tensor, len(cachedGraphObj.resultTensors))
	
	for i, graphTensor := range cachedGraphObj.resultTensors {
		// Calculate buffer size
		numElems := 1
		for _, dim := range graphTensor.Shape() {
			numElems *= dim
		}
		bufferSize := numElems * 4 // Assume float32
		
		buffer := engine.device.NewBufferWithLength(bufferSize, int(metal_bridge.ResourceStorageModeShared))
		if buffer == nil {
			return nil, fmt.Errorf("failed to create result buffer %d", i)
		}
		resultBuffers[i] = buffer
		
		// Create result tensor
		shape := graphTensor.Shape()
		resultTensor := &Tensor{
			Shape:    shape,
			Strides:  calculateStrides(shape),
			DType:    Float32,
			Device:   determineResultDevice(inputs...),
			NumElems: numElems, // Use the numElems calculated above
		}
		resultTensors[i] = resultTensor
	}
	
	// Execute the graph
	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}
	
	cachedGraphObj.executable.Execute(engine.commandQueue, cachedGraphObj.inputTensors, inputBuffers, 
		cachedGraphObj.resultTensors, resultBuffers, executionDescriptor)
	
	// Handle results - copy data from GPU buffers to CPU tensors
	for i, resultTensor := range resultTensors {
		err := handleMPSResult(resultTensor, resultBuffers[i])
		if err != nil {
			return nil, fmt.Errorf("failed to handle result %d: %v", i, err)
		}
	}
	
	return resultTensors, nil
}

// createBufferFromTensor creates a Metal buffer from a tensor's data
func (engine *MPSGraphEngine) createBufferFromTensor(tensor *Tensor) (*metal_bridge.Buffer, error) {
	// For persistent GPU tensors, use existing GPU buffer if available
	if tensor.Device == PersistentGPU && tensor.GetGPUBuffer() != nil {
		if buffer, ok := tensor.GetGPUBuffer().(*metal_bridge.Buffer); ok {
			return buffer, nil
		}
	}
	
	// Get tensor data as bytes
	var data []byte
	
	switch tensor.DType {
	case Float32:
		var float32Data []float32
		
		if tensor.Data == nil {
			// For GPU tensors without CPU data, copy from GPU buffer
			if tensor.GetGPUBuffer() != nil {
				cpuTensor, err := tensor.ToCPU()
				if err != nil {
					return nil, fmt.Errorf("failed to copy tensor to CPU: %v", err)
				}
				float32Data = cpuTensor.Data.([]float32)
			} else {
				return nil, fmt.Errorf("tensor data is nil and no GPU buffer available")
			}
		} else {
			// Use existing CPU data
			if tensor.Device == CPU {
				float32Data = tensor.Data.([]float32)
			} else {
				// For GPU tensors, we need to copy data to CPU first
				cpuTensor, err := tensor.ToCPU()
				if err != nil {
					return nil, fmt.Errorf("failed to copy tensor to CPU: %v", err)
				}
				float32Data = cpuTensor.Data.([]float32)
			}
		}
		
		// Convert to bytes
		data = make([]byte, len(float32Data)*4)
		for i, val := range float32Data {
			*(*float32)(unsafe.Pointer(&data[i*4])) = val
		}
	default:
		return nil, fmt.Errorf("unsupported data type: %v", tensor.DType)
	}
	
	// Create Metal buffer
	buffer := engine.device.NewBufferWithBytes(data, int(metal_bridge.ResourceStorageModeShared))
	if buffer == nil {
		return nil, fmt.Errorf("failed to create Metal buffer")
	}
	
	return buffer, nil
}

// determineResultDevice determines the device type for the result based on input tensors
func determineResultDevice(tensors ...*Tensor) DeviceType {
	// If any tensor is PersistentGPU, result should be PersistentGPU
	for _, t := range tensors {
		if t.Device == PersistentGPU {
			return PersistentGPU
		}
	}
	
	// For MPS operations, result should be GPU (temporary) since computation runs on GPU
	// This applies even if input tensors are CPU (they get copied to GPU for computation)
	return GPU
}

// shouldKeepOnGPU returns true if the result tensor should stay on GPU
func shouldKeepOnGPU(device DeviceType) bool {
	return device == PersistentGPU
}

// handleMPSResult handles the result of an MPS operation, either keeping it on GPU or copying to CPU
func handleMPSResult(result *Tensor, resultBuffer interface{}) error {
	if shouldKeepOnGPU(result.Device) {
		// Store the GPU buffer in the tensor for persistent GPU tensors
		result.SetGPUBuffer(resultBuffer)
		result.Data = nil // No CPU data for persistent GPU tensors
		// Buffer is now owned by the tensor - don't release it here
	} else {
		// Copy result data from Metal buffer to CPU slice for temporary GPU tensors
		resultData, err := copyDataFromGPUBuffer(resultBuffer, result.DType, result.NumElems)
		if err != nil {
			return fmt.Errorf("failed to copy result from GPU buffer: %v", err)
		}
		result.Data = resultData
		// GPU buffer will be garbage collected
	}
	return nil
}

// isCompatibleForOp checks if two tensors are compatible for element-wise operations
func isCompatibleForOp(a, b *Tensor) bool {
	// Check data types
	if a.DType != b.DType {
		return false
	}
	
	// Check if shapes are broadcastable (let MPS handle the actual broadcasting)
	_, err := BroadcastShapes(a.Shape, b.Shape)
	return err == nil
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

// createMPSBufferFromTensor creates a GPU buffer and copies tensor data
func createMPSBufferFromTensor(tensor *Tensor, device *metal_bridge.Device) (*metal_bridge.Buffer, error) {
	size := calculateTensorSize(tensor.Shape, tensor.DType)
	buffer := device.NewBufferWithLength(int(size), int(metal_bridge.ResourceStorageModeShared))
	if buffer == nil {
		return nil, fmt.Errorf("failed to allocate buffer")
	}
	
	// Handle different tensor device types
	if tensor.Device == PersistentGPU && tensor.gpuBuffer != nil {
		// For PersistentGPU tensors, copy from existing GPU buffer
		err := copyGPUBufferToGPUBuffer(tensor.gpuBuffer, buffer, tensor.DType, tensor.NumElems)
		if err != nil {
			// Clean up on error - no Release method available, buffer will be garbage collected
			return nil, fmt.Errorf("failed to copy GPU buffer to buffer: %v", err)
		}
	} else {
		// For CPU tensors, views, or temporary GPU tensors, materialize the data and copy
		data := tensor.materializeView()
		if data == nil {
			return nil, fmt.Errorf("failed to materialize tensor data")
		}
		err := copyDataToGPUBuffer(data, buffer, tensor.DType)
		if err != nil {
			return nil, fmt.Errorf("failed to copy data to buffer: %v", err)
		}
	}
	
	return buffer, nil
}

// copyGPUBufferToGPUBuffer copies data from one GPU buffer to another
func copyGPUBufferToGPUBuffer(srcBuffer, dstBuffer interface{}, dtype DType, numElems int) error {
	// Type assert both buffers
	srcMtlBuffer, ok := srcBuffer.(*metal_bridge.Buffer)
	if !ok {
		return fmt.Errorf("invalid source buffer type for GPU-to-GPU copy")
	}
	dstMtlBuffer, ok := dstBuffer.(*metal_bridge.Buffer)
	if !ok {
		return fmt.Errorf("invalid destination buffer type for GPU-to-GPU copy")
	}
	
	// Get buffer contents pointers
	srcPtr := srcMtlBuffer.Contents()
	dstPtr := dstMtlBuffer.Contents()
	
	switch dtype {
	case Float32:
		// Copy float32 data
		srcData := (*[1<<30]float32)(srcPtr)[:numElems]
		dstData := (*[1<<30]float32)(dstPtr)[:numElems]
		copy(dstData, srcData)
		
	case Int32:
		// Copy int32 data
		srcData := (*[1<<30]int32)(srcPtr)[:numElems]
		dstData := (*[1<<30]int32)(dstPtr)[:numElems]
		copy(dstData, srcData)
		
	default:
		return fmt.Errorf("unsupported data type for GPU-to-GPU copy: %v", dtype)
	}
	
	return nil
}

// createMPSResultBuffer creates a result buffer
func createMPSResultBuffer(shape []int, dtype DType, device *metal_bridge.Device) (*metal_bridge.Buffer, error) {
	size := calculateTensorSize(shape, dtype)
	buffer := device.NewBufferWithLength(int(size), int(metal_bridge.ResourceStorageModeShared))
	if buffer == nil {
		return nil, fmt.Errorf("failed to allocate result buffer")
	}
	return buffer, nil
}

// AddMPS performs tensor addition using a cached MPSGraph
func AddMPS(a, b *Tensor) (*Tensor, error) {
	if !isCompatibleForOp(a, b) {
		return nil, fmt.Errorf("tensors are not compatible for addition")
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("add", a, b)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
		resultTensor := graph.Addition(placeholderA, placeholderB)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Determine broadcast result shape
	broadcastShape, err := BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast tensors for addition: %v", err)
	}
	
	resultDevice := determineResultDevice(a, b)
	result := &Tensor{
		Shape:    broadcastShape,
		Strides:  calculateStrides(broadcastShape),
		DType:    a.DType,
		Device:   resultDevice,
		NumElems: calculateNumElements(broadcastShape),
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	bBuffer, err := createMPSBufferFromTensor(b, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	// bBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// SubMPS performs tensor subtraction using a cached MPSGraph
func SubMPS(a, b *Tensor) (*Tensor, error) {
	if !isCompatibleForOp(a, b) {
		return nil, fmt.Errorf("tensors are not compatible for subtraction")
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("sub", a, b)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
		resultTensor := graph.Subtraction(placeholderA, placeholderB)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Determine broadcast result shape
	broadcastShape, err := BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast tensors for subtraction: %v", err)
	}
	
	resultDevice := determineResultDevice(a, b)
	result := &Tensor{
		Shape:    broadcastShape,
		Strides:  calculateStrides(broadcastShape),
		DType:    a.DType,
		Device:   resultDevice,
		NumElems: calculateNumElements(broadcastShape),
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	bBuffer, err := createMPSBufferFromTensor(b, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	// bBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// MulMPS performs tensor multiplication using a cached MPSGraph
func MulMPS(a, b *Tensor) (*Tensor, error) {
	if !isCompatibleForOp(a, b) {
		return nil, fmt.Errorf("tensors are not compatible for multiplication")
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("mul", a, b)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
		resultTensor := graph.Multiplication(placeholderA, placeholderB)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Determine broadcast result shape
	broadcastShape, err := BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast tensors for multiplication: %v", err)
	}
	
	resultDevice := determineResultDevice(a, b)
	result := &Tensor{
		Shape:    broadcastShape,
		Strides:  calculateStrides(broadcastShape),
		DType:    a.DType,
		Device:   resultDevice,
		NumElems: calculateNumElements(broadcastShape),
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	bBuffer, err := createMPSBufferFromTensor(b, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	// bBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// DivMPS performs tensor division using a cached MPSGraph
func DivMPS(a, b *Tensor) (*Tensor, error) {
	if !isCompatibleForOp(a, b) {
		return nil, fmt.Errorf("tensors are not compatible for division")
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("div", a, b)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
		resultTensor := graph.Division(placeholderA, placeholderB)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Determine broadcast result shape
	broadcastShape, err := BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast tensors for division: %v", err)
	}
	
	resultDevice := determineResultDevice(a, b)
	result := &Tensor{
		Shape:    broadcastShape,
		Strides:  calculateStrides(broadcastShape),
		DType:    a.DType,
		Device:   resultDevice,
		NumElems: calculateNumElements(broadcastShape),
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	bBuffer, err := createMPSBufferFromTensor(b, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	// bBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// MatMulMPS performs matrix multiplication using a cached MPSGraph
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

	cacheKey := generateCacheKey("matmul", a, b)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
		resultTensor := graph.MatrixMultiplication(placeholderA, placeholderB)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	outputShape := []int{a.Shape[0], b.Shape[1]}
	result := &Tensor{
		Shape:    outputShape,
		Strides:  calculateStrides(outputShape),
		DType:    a.DType,
		Device:   determineResultDevice(a, b),
		NumElems: outputShape[0] * outputShape[1],
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	bBuffer, err := createMPSBufferFromTensor(b, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor B: %v", err)
	}
	// bBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer, bBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// ReLUMPS performs ReLU activation using a cached MPSGraph
func ReLUMPS(a *Tensor) (*Tensor, error) {
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("relu", a)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		resultTensor := graph.ReLU(placeholderA)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	result := &Tensor{
		Shape:    a.Shape,
		Strides:  a.Strides,
		DType:    a.DType,
		Device:   determineResultDevice(a),
		NumElems: a.NumElems,
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// SigmoidMPS performs Sigmoid activation using a cached MPSGraph
func SigmoidMPS(a *Tensor) (*Tensor, error) {
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("sigmoid", a)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(a.DType)
		placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
		resultTensor := graph.Sigmoid(placeholderA)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	result := &Tensor{
		Shape:    a.Shape,
		Strides:  a.Strides,
		DType:    a.DType,
		Device:   determineResultDevice(a),
		NumElems: a.NumElems,
	}

	aBuffer, err := createMPSBufferFromTensor(a, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
	}
	// aBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{aBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
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

// Conv2DMPS performs 2D convolution using a cached MPSGraph
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

	cacheKey := generateCacheKey(fmt.Sprintf("conv2d_s%dx%d_p%d%d%d%d", strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom), input, weights, bias)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(input.DType)
		placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
		placeholderWeights := graph.PlaceholderTensor(weights.Shape, dataType)
		var inputTensors []*metal_bridge.GraphTensor
		var resultTensor *metal_bridge.GraphTensor
		
		if bias != nil {
			placeholderBias := graph.PlaceholderTensor(bias.Shape, dataType)
			inputTensors = []*metal_bridge.GraphTensor{placeholderInput, placeholderWeights, placeholderBias}
			
			// Perform convolution without bias first
			convResult := graph.Conv2D(placeholderInput, placeholderWeights, nil,
				strideX, strideY, // strideX, strideY
				1, 1, // dilationX, dilationY
				paddingLeft, paddingRight, paddingTop, paddingBottom,
				1) // groups
			
			// Add bias with proper broadcasting - bias needs to be reshaped to [1, C, 1, 1] for broadcasting
			biasShape := []int{1, bias.Shape[0], 1, 1}
			reshapedBias := graph.Reshape(placeholderBias, biasShape)
			resultTensor = graph.Addition(convResult, reshapedBias)
		} else {
			inputTensors = []*metal_bridge.GraphTensor{placeholderInput, placeholderWeights}
			resultTensor = graph.Conv2D(placeholderInput, placeholderWeights, nil,
				strideX, strideY, // strideX, strideY
				1, 1, // dilationX, dilationY
				paddingLeft, paddingRight, paddingTop, paddingBottom,
				1) // groups
		}
		
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, inputTensors, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, inputTensors, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Calculate output shape
	N := input.Shape[0]         // Batch size
	C_out := weights.Shape[0]   // Output channels
	H := input.Shape[2]         // Input height
	W := input.Shape[3]         // Input width
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
		Device:   determineResultDevice(input, weights),
		NumElems: N * C_out * outputH * outputW,
	}


	inputBuffer, err := createMPSBufferFromTensor(input, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	// inputBuffer will be garbage collected

	weightsBuffer, err := createMPSBufferFromTensor(weights, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for weights: %v", err)
	}
	// weightsBuffer will be garbage collected

	var inputBuffers []*metal_bridge.Buffer
	if bias != nil {
		biasBuffer, err := createMPSBufferFromTensor(bias, engine.device)
		if err != nil {
			return nil, fmt.Errorf("failed to create buffer for bias: %v", err)
		}
		// biasBuffer will be garbage collected
		inputBuffers = []*metal_bridge.Buffer{inputBuffer, weightsBuffer, biasBuffer}
	} else {
		inputBuffers = []*metal_bridge.Buffer{inputBuffer, weightsBuffer}
	}

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		inputBuffers,
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// MaxPool2DMPS performs 2D max pooling using a cached MPSGraph
func MaxPool2DMPS(input *Tensor, kernelSize, stride, padding int) (*Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (NCHW), got %d dimensions", len(input.Shape))
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("maxpool2d", input)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(input.DType)
		placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
		resultTensor := graph.MaxPool2D(placeholderInput,
			kernelSize, kernelSize, // kernelWidth, kernelHeight
			stride, stride,         // strideX, strideY
			padding, padding, padding, padding) // paddingLeft, paddingRight, paddingTop, paddingBottom
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
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
		Device:   determineResultDevice(input),
		NumElems: N * C * outputH * outputW,
	}


	inputBuffer, err := createMPSBufferFromTensor(input, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	// inputBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{inputBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// AvgPool2DMPS performs 2D average pooling using a cached MPSGraph
func AvgPool2DMPS(input *Tensor, kernelSize, stride, padding int) (*Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("input tensor must be 4D (NCHW), got %d dimensions", len(input.Shape))
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := generateCacheKey("avgpool2d", input)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(input.DType)
		placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
		resultTensor := graph.AvgPool2D(placeholderInput,
			kernelSize, kernelSize, // kernelWidth, kernelHeight
			stride, stride,         // strideX, strideY
			padding, padding, padding, padding) // paddingLeft, paddingRight, paddingTop, paddingBottom
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
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
		Device:   determineResultDevice(input),
		NumElems: N * C * outputH * outputW,
	}


	inputBuffer, err := createMPSBufferFromTensor(input, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	// inputBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{inputBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// ReshapeMPS performs tensor reshape using a cached MPSGraph
func ReshapeMPS(input *Tensor, newShape []int) (*Tensor, error) {
	// Validate new shape
	newNumElems := 1
	for _, dim := range newShape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid dimension in new shape: %d", dim)
		}
		newNumElems *= dim
	}
	
	if newNumElems != input.NumElems {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v (size %d)", input.NumElems, newShape, newNumElems)
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	// Create cache key that includes both input shape and target shape
	cacheKey := fmt.Sprintf("reshape_%v_%v_%v", input.Shape, newShape, input.DType)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(input.DType)
		placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
		
		resultTensor := graph.Reshape(placeholderInput, newShape)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Create output tensor
	result := &Tensor{
		Shape:    make([]int, len(newShape)),
		Strides:  calculateStrides(newShape),
		DType:    input.DType,
		Device:   determineResultDevice(input),
		NumElems: newNumElems,
	}
	copy(result.Shape, newShape)


	inputBuffer, err := createMPSBufferFromTensor(input, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	// inputBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{inputBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// SumMPS performs tensor reduction sum along specified axis using MPSGraph
func SumMPS(input *Tensor, axis int, keepdim bool) (*Tensor, error) {
	if len(input.Shape) == 0 {
		return nil, fmt.Errorf("cannot sum empty tensor")
	}
	if axis < 0 || axis >= len(input.Shape) {
		return nil, fmt.Errorf("axis %d out of range for tensor with %d dimensions", axis, len(input.Shape))
	}

	engine, err := GetMPSGraphEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}

	cacheKey := fmt.Sprintf("sum_axis%d_keepdim%t_%v_%v", axis, keepdim, input.Shape, input.DType)
	cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
		graph := metal_bridge.NewGraph()
		dataType := convertDTypeToMPS(input.DType)
		placeholderInput := graph.PlaceholderTensor(input.Shape, dataType)
		
		// Use the reduction sum operation
		resultTensor := graph.ReductionSum(placeholderInput, axis, keepdim)
		compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
		executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
		return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderInput}, []*metal_bridge.GraphTensor{resultTensor}}
	})

	if err != nil {
		return nil, err
	}

	// Calculate output shape
	outputShape := make([]int, 0, len(input.Shape))
	for i, dim := range input.Shape {
		if i == axis {
			if keepdim {
				outputShape = append(outputShape, 1)
			}
			// Skip this dimension if not keeping it
		} else {
			outputShape = append(outputShape, dim)
		}
	}

	// Create output tensor
	result := &Tensor{
		Shape:    outputShape,
		Strides:  calculateStrides(outputShape),
		DType:    input.DType,
		Device:   determineResultDevice(input),
		NumElems: calculateNumElements(outputShape),
	}


	inputBuffer, err := createMPSBufferFromTensor(input, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer for input: %v", err)
	}
	// inputBuffer will be garbage collected

	resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
	if executionDescriptor == nil {
		return nil, fmt.Errorf("failed to create execution descriptor")
	}

	cached.executable.Execute(engine.commandQueue,
		cached.inputTensors,
		[]*metal_bridge.Buffer{inputBuffer},
		cached.resultTensors,
		[]*metal_bridge.Buffer{resultBuffer},
		executionDescriptor)

	if err := handleMPSResult(result, resultBuffer); err != nil {
		return nil, err
	}

	return result, nil
}

// IsGPUAvailable checks if Metal GPU compute is available
func IsGPUAvailable() bool {
	_, err := GetMPSGraphEngine()
	return err == nil
}

// AddGPUAsync performs tensor addition on GPU asynchronously
func AddGPUAsync(t1, t2 *Tensor, completion func(*Tensor, error)) error {
	// Validate inputs before starting async operation
	if !isCompatibleForOp(t1, t2) {
		return fmt.Errorf("tensors are not compatible for addition")
	}
	
	// Perform operation asynchronously
	go func() {
		result, err := AddMPS(t1, t2)
		completion(result, err)
	}()
	return nil
}

// MatMulGPUAsync performs matrix multiplication on GPU asynchronously  
func MatMulGPUAsync(t1, t2 *Tensor, completion func(*Tensor, error)) error {
	// Validate matrix dimensions before starting async operation
	if len(t1.Shape) != 2 || len(t2.Shape) != 2 {
		return fmt.Errorf("matrix multiplication requires 2D tensors")
	}
	if t1.Shape[1] != t2.Shape[0] {
		return fmt.Errorf("inner dimensions must match for matrix multiplication: %d vs %d", t1.Shape[1], t2.Shape[0])
	}
	
	// Perform operation asynchronously
	go func() {
		result, err := MatMulMPS(t1, t2)
		completion(result, err)
	}()
	return nil
}

// AddGPU performs tensor addition on GPU synchronously
func AddGPU(t1, t2 *Tensor) (*Tensor, error) {
	return AddMPS(t1, t2)
}

// MatMulGPU performs matrix multiplication on GPU synchronously
func MatMulGPU(t1, t2 *Tensor) (*Tensor, error) {
	return MatMulMPS(t1, t2)
}

// ReLUGPU performs ReLU activation on GPU synchronously
func ReLUGPU(t *Tensor) (*Tensor, error) {
	return ReLUMPS(t)
}

// GPUInfo returns information about the GPU device
func GPUInfo() (string, error) {
	if !IsGPUAvailable() {
		return "", fmt.Errorf("GPU not available")
	}
	return "Metal Performance Shaders GPU", nil
}

// createGPUBuffer creates a GPU buffer for a tensor and copies CPU data to it
func (t *Tensor) createGPUBuffer() error {
	if t.Data == nil {
		return fmt.Errorf("cannot create GPU buffer: tensor data is nil")
	}
	
	engine, err := GetMPSGraphEngine()
	if err != nil {
		return fmt.Errorf("failed to get MPSGraph engine: %v", err)
	}
	
	// Create buffer based on tensor size
	size := calculateTensorSize(t.Shape, t.DType)
	buffer := engine.device.NewBufferWithLength(int(size), int(metal_bridge.ResourceStorageModeShared))
	if buffer == nil {
		return fmt.Errorf("failed to allocate GPU buffer")
	}
	
	// Copy CPU data to GPU buffer
	err = copyDataToGPUBuffer(t.Data, buffer, t.DType)
	if err != nil {
		return fmt.Errorf("failed to copy data to GPU buffer: %v", err)
	}
	
	// Set GPU buffer and initialize reference counting
	t.SetGPUBuffer(buffer)
	
	return nil
}
