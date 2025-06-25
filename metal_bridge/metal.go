package metal_bridge

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -framework CoreFoundation
#include "metal_bridge.h"
// Define a Go function callable from C for completion handlers
extern void goCommandBufferCompleted(void* userData, long statusCode);
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Go equivalent of MTLSize
type MTLSize struct {
	Width, Height, Depth uint
}

// Resource storage mode constants
var (
	ResourceStorageModeShared  = C.MTLResourceStorageModeShared_Const
	ResourceStorageModeManaged = C.MTLResourceStorageModeManaged_Const
	ResourceStorageModePrivate = C.MTLResourceStorageModePrivate_Const
)

// MPSGraph data type constants
var (
	MPSDataTypeFloat32 = C.MPSDataTypeFloat32_Const
	MPSDataTypeFloat16 = C.MPSDataTypeFloat16_Const
	MPSDataTypeInt32   = C.MPSDataTypeInt32_Const
)

// Wrapper struct for MTLDevice
type Device struct {
	c_device C.MTLDeviceRef
}

func CreateSystemDefaultDevice() *Device {
	c_dev := C.CreateSystemDefaultDevice()
	if c_dev == nil {
		return nil
	}
	dev := &Device{c_device: c_dev}
	runtime.SetFinalizer(dev, func(d *Device) {
		C.ReleaseMetalObject(unsafe.Pointer(d.c_device))
	})
	return dev
}

// Wrapper struct for MTLCommandQueue
type CommandQueue struct {
	c_queue C.MTLCommandQueueRef
}

func (d *Device) NewCommandQueue() *CommandQueue {
	c_q := C.CreateCommandQueue(d.c_device)
	if c_q == nil {
		return nil
	}
	q := &CommandQueue{c_queue: c_q}
	runtime.SetFinalizer(q, func(cq *CommandQueue) {
		C.ReleaseMetalObject(unsafe.Pointer(cq.c_queue))
	})
	return q
}

// Wrapper struct for MTLBuffer
type Buffer struct {
	c_buffer C.MTLBufferRef
	length   uintptr // Length in bytes
}

func (d *Device) CreateBufferWithBytes(data interface{}, options C.size_t) (*Buffer, error) {
	switch v := data.(type) {
	case []float32:
		if len(v) == 0 {
			return nil, fmt.Errorf("data slice cannot be empty")
		}
		byteLength := C.size_t(len(v) * int(unsafe.Sizeof(v[0])))
		// When passing Go slice data to C, it's crucial that the Go slice
		// is not garbage collected until the C function (and potentially GPU) is done with it.
		// For MTLResourceStorageModeShared, the buffer refers directly to Go's memory.
		// We'll rely on the Go tensor holding a reference to the underlying Go slice
		// until GPU computation using the buffer is complete.
		c_buf := C.CreateBufferWithBytes(d.c_device, unsafe.Pointer(&v[0]), byteLength, options)
		if c_buf == nil {
			return nil, fmt.Errorf("failed to create Metal buffer")
		}
		// Use CFRetain on the Objective-C object when passing it from Objective-C to Go
		// to explicitly manage its lifetime in Go. This makes Go the owner.
		// The `ReleaseMetalObject` in the finalizer will then call `CFRelease`.
		C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_buf))) // Explicitly retain to ensure Go owns it
		buf := &Buffer{c_buffer: c_buf, length: uintptr(byteLength)}
		runtime.SetFinalizer(buf, func(b *Buffer) {
			C.ReleaseMetalObject(unsafe.Pointer(b.c_buffer))
		})
		return buf, nil
	case []int32:
		if len(v) == 0 {
			return nil, fmt.Errorf("data slice cannot be empty")
		}
		byteLength := C.size_t(len(v) * int(unsafe.Sizeof(v[0])))
		c_buf := C.CreateBufferWithBytes(d.c_device, unsafe.Pointer(&v[0]), byteLength, options)
		if c_buf == nil {
			return nil, fmt.Errorf("failed to create Metal buffer")
		}
		C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_buf)))
		buf := &Buffer{c_buffer: c_buf, length: uintptr(byteLength)}
		runtime.SetFinalizer(buf, func(b *Buffer) {
			C.ReleaseMetalObject(unsafe.Pointer(b.c_buffer))
		})
		return buf, nil
	case []uint32:
		if len(v) == 0 {
			return nil, fmt.Errorf("data slice cannot be empty")
		}
		byteLength := C.size_t(len(v) * int(unsafe.Sizeof(v[0])))
		c_buf := C.CreateBufferWithBytes(d.c_device, unsafe.Pointer(&v[0]), byteLength, options)
		if c_buf == nil {
			return nil, fmt.Errorf("failed to create Metal buffer")
		}
		C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_buf)))
		buf := &Buffer{c_buffer: c_buf, length: uintptr(byteLength)}
		runtime.SetFinalizer(buf, func(b *Buffer) {
			C.ReleaseMetalObject(unsafe.Pointer(b.c_buffer))
		})
		return buf, nil
	default:
		return nil, fmt.Errorf("unsupported data type for buffer creation")
	}
}

func (d *Device) CreateBufferWithLength(length uintptr, options C.size_t) (*Buffer, error) {
	c_buf := C.CreateBufferWithLength(d.c_device, C.size_t(length), options)
	if c_buf == nil {
		return nil, fmt.Errorf("failed to create Metal buffer")
	}
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_buf)))
	buf := &Buffer{c_buffer: c_buf, length: length}
	runtime.SetFinalizer(buf, func(b *Buffer) {
		C.ReleaseMetalObject(unsafe.Pointer(b.c_buffer))
	})
	return buf, nil
}

func (b *Buffer) Contents() unsafe.Pointer {
	return C.GetBufferContents(b.c_buffer)
}

func (b *Buffer) Length() uintptr {
	return uintptr(C.GetBufferLength(b.c_buffer))
}

func (b *Buffer) ContentsAsFloat32() []float32 {
	// This is unsafe, assume float32 for example
	return (*[1 << 30]float32)(C.GetBufferContents(b.c_buffer))[:b.length/unsafe.Sizeof(float32(0))]
}

func (b *Buffer) ContentsAsInt32() []int32 {
	// This is unsafe, assume int32 for example
	return (*[1 << 30]int32)(C.GetBufferContents(b.c_buffer))[:b.length/unsafe.Sizeof(int32(0))]
}

// Wrapper struct for MTLLibrary
type Library struct {
	c_library C.MTLLibraryRef
}

func (d *Device) CreateLibraryWithSource(source string) (*Library, error) {
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	
	c_lib := C.CreateLibraryWithSource(d.c_device, cSource)
	if c_lib == nil {
		return nil, fmt.Errorf("failed to create Metal library")
	}
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_lib)))
	lib := &Library{c_library: c_lib}
	runtime.SetFinalizer(lib, func(l *Library) {
		C.ReleaseMetalObject(unsafe.Pointer(l.c_library))
	})
	return lib, nil
}

// Wrapper struct for MTLFunction
type Function struct {
	c_function C.MTLFunctionRef
}

func (l *Library) GetFunction(functionName string) (*Function, error) {
	cFunctionName := C.CString(functionName)
	defer C.free(unsafe.Pointer(cFunctionName))
	
	c_func := C.GetFunction(l.c_library, cFunctionName)
	if c_func == nil {
		return nil, fmt.Errorf("failed to get function '%s' from library", functionName)
	}
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_func)))
	function := &Function{c_function: c_func}
	runtime.SetFinalizer(function, func(f *Function) {
		C.ReleaseMetalObject(unsafe.Pointer(f.c_function))
	})
	return function, nil
}

// Wrapper for MTLComputePipelineState
type ComputePipelineState struct {
	c_pipelineState C.MTLComputePipelineStateRef
}

func (d *Device) NewComputePipelineStateWithFunction(function *Function) (*ComputePipelineState, error) {
	c_ps := C.CreateComputePipelineStateWithFunction(d.c_device, function.c_function)
	if c_ps == nil {
		return nil, fmt.Errorf("failed to create compute pipeline state")
	}
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_ps)))
	ps := &ComputePipelineState{c_pipelineState: c_ps}
	runtime.SetFinalizer(ps, func(p *ComputePipelineState) {
		C.ReleaseMetalObject(unsafe.Pointer(p.c_pipelineState))
	})
	return ps, nil
}

// Wrapper for MTLCommandBuffer
type CommandBuffer struct {
	c_commandBuffer C.MTLCommandBufferRef
	// Keep a reference to resources that must stay alive until completion,
	// e.g., source Go slices for buffers, completion handler context.
	retainedResources []interface{}
}

func (q *CommandQueue) CommandBuffer() *CommandBuffer {
	c_cb := C.CreateCommandBuffer(q.c_queue)
	if c_cb == nil {
		return nil
	}
	// Retain the command buffer on the Go side for its lifetime management
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_cb)))
	cb := &CommandBuffer{c_commandBuffer: c_cb}
	runtime.SetFinalizer(cb, func(b *CommandBuffer) {
		C.ReleaseMetalObject(unsafe.Pointer(b.c_commandBuffer))
	})
	return cb
}

// Wrapper for MTLComputeCommandEncoder
type ComputeCommandEncoder struct {
	c_encoder C.MTLComputeCommandEncoderRef
}

func (cb *CommandBuffer) ComputeCommandEncoder() *ComputeCommandEncoder {
	c_encoder := C.CreateComputeCommandEncoder(cb.c_commandBuffer)
	if c_encoder == nil {
		return nil
	}
	// Retain the encoder to ensure it's not released prematurely.
	// It's typically transient, but safer to manage explicitly if it's held by Go.
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_encoder)))
	encoder := &ComputeCommandEncoder{c_encoder: c_encoder}
	runtime.SetFinalizer(encoder, func(e *ComputeCommandEncoder) {
		C.ReleaseMetalObject(unsafe.Pointer(e.c_encoder))
	})
	return encoder
}

func (e *ComputeCommandEncoder) SetComputePipelineState(pipelineState *ComputePipelineState) {
	C.SetComputePipelineState(e.c_encoder, pipelineState.c_pipelineState)
}

func (e *ComputeCommandEncoder) SetBuffer(buffer *Buffer, offset, index uint) {
	C.SetBuffer(e.c_encoder, buffer.c_buffer, C.size_t(offset), C.size_t(index))
}

func (e *ComputeCommandEncoder) DispatchThreads(gridSize, threadgroupSize MTLSize) {
	C.DispatchThreads(e.c_encoder, 
		C.size_t(gridSize.Width), C.size_t(gridSize.Height), C.size_t(gridSize.Depth),
		C.size_t(threadgroupSize.Width), C.size_t(threadgroupSize.Height), C.size_t(threadgroupSize.Depth))
}

func (e *ComputeCommandEncoder) EndEncoding() {
	C.EndEncoding(e.c_encoder)
	// After EndEncoding, the encoder is no longer needed. Release it immediately.
	// This assumes the encoder itself doesn't hold strong references to command buffer.
	C.ReleaseMetalObject(unsafe.Pointer(e.c_encoder))
	e.c_encoder = nil // Clear the C pointer
}

func (cb *CommandBuffer) Commit() {
	C.CommitCommandBuffer(cb.c_commandBuffer)
}

func (cb *CommandBuffer) WaitUntilCompleted() {
	C.WaitUntilCommandBufferCompleted(cb.c_commandBuffer)
}

// The Go function called by Objective-C completion handler
//export goCommandBufferCompleted
func goCommandBufferCompleted(userData unsafe.Pointer, statusCode C.long) {
	// Recover the context or resources from userData if needed
	// For example, if userData points to a struct containing Go slices to be released.
	// This is where you'd signal to the Go allocator that resources can be reclaimed.
	fmt.Printf("Command buffer completed with status: %d\n", statusCode)
	// IMPORTANT: If 'userData' was a Go pointer created with C.CBytes or similar,
	// or contained references to Go objects that needed to be kept alive,
	// this is the place to signal their release or decref their reference count.
	// For now, it just prints.
	// Example: If a map of `id` to Go channels was passed via userData
	// you could signal completion here.
}

// AddCompletedHandler allows registering a Go callback for command buffer completion.
// It passes a userData pointer which can be used to pass context to the Go function.
func (cb *CommandBuffer) AddCompletedHandler(handler func(status int)) {
	// Create a goroutine-safe way to manage handler context if necessary.
	// For a simple callback, we can pass a unique ID or a channel.
	// For more complex resource management, userData might point to a Go struct
	// that gets cleaned up when this handler fires.
	cb.retainedResources = append(cb.retainedResources, handler) // Retain handler to prevent GC
	C.AddCommandBufferCompletedHandler(cb.c_commandBuffer, nil, (C.CompletionHandlerFunc)(C.goCommandBufferCompleted))
}

// MPSGraph wrapper structs and functions

// Wrapper struct for MPSGraph  
type Graph struct {
	c_graph C.MPSGraphRef
}

// Wrapper struct for MPSGraphDevice
type GraphDevice struct {
	c_device C.MPSGraphDeviceRef
}

func NewGraph() *Graph {
	c_graph := C.CreateMPSGraph()
	if c_graph == nil {
		return nil
	}
	graph := &Graph{c_graph: c_graph}
	runtime.SetFinalizer(graph, func(g *Graph) {
		C.ReleaseMetalObject(unsafe.Pointer(g.c_graph))
	})
	return graph
}

func NewGraphDevice(device *Device) *GraphDevice {
	c_device := C.CreateMPSGraphDevice(device.c_device)
	if c_device == nil {
		return nil
	}
	graphDevice := &GraphDevice{c_device: c_device}
	runtime.SetFinalizer(graphDevice, func(gd *GraphDevice) {
		C.ReleaseMetalObject(unsafe.Pointer(gd.c_device))
	})
	return graphDevice
}

// Wrapper struct for MPSGraphTensor
type GraphTensor struct {
	c_tensor C.MPSGraphTensorRef
	shape    []int
	dataType int
}

func (g *Graph) PlaceholderTensor(shape []int, dataType int) *GraphTensor {
	cShape := make([]C.int, len(shape))
	for i, dim := range shape {
		cShape[i] = C.int(dim)
	}
	
	c_tensor := C.MPSGraphPlaceholderTensor(g.c_graph, &cShape[0], C.size_t(len(shape)), C.int(dataType))
	if c_tensor == nil {
		return nil
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    shape,
		dataType: dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

func (g *Graph) ConstantTensor(value float64, shape []int, dataType int) *GraphTensor {
	cShape := make([]C.int, len(shape))
	for i, dim := range shape {
		cShape[i] = C.int(dim)
	}
	
	c_tensor := C.MPSGraphConstantTensor(g.c_graph, C.double(value), &cShape[0], C.size_t(len(shape)), C.int(dataType))
	if c_tensor == nil {
		return nil
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    shape,
		dataType: dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

// MPSGraph operations
func (g *Graph) Addition(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphAddition(g.c_graph, primaryTensor.c_tensor, secondaryTensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    primaryTensor.shape, // Assuming same shape for element-wise ops
		dataType: primaryTensor.dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

func (g *Graph) Subtraction(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphSubtraction(g.c_graph, primaryTensor.c_tensor, secondaryTensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    primaryTensor.shape,
		dataType: primaryTensor.dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

func (g *Graph) Multiplication(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphMultiplication(g.c_graph, primaryTensor.c_tensor, secondaryTensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    primaryTensor.shape,
		dataType: primaryTensor.dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

func (g *Graph) Division(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphDivision(g.c_graph, primaryTensor.c_tensor, secondaryTensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    primaryTensor.shape,
		dataType: primaryTensor.dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

func (g *Graph) MatrixMultiplication(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphMatrixMultiplication(g.c_graph, primaryTensor.c_tensor, secondaryTensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	// For matrix multiplication, compute output shape
	var outputShape []int
	if len(primaryTensor.shape) == 2 && len(secondaryTensor.shape) == 2 {
		outputShape = []int{primaryTensor.shape[0], secondaryTensor.shape[1]}
	} else {
		outputShape = primaryTensor.shape // Fallback
	}
	
	tensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    outputShape,
		dataType: primaryTensor.dataType,
	}
	runtime.SetFinalizer(tensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return tensor
}

func (g *Graph) ReLU(tensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphReLU(g.c_graph, tensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    tensor.shape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

func (g *Graph) Sigmoid(tensor *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphSigmoid(g.c_graph, tensor.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    tensor.shape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

func (g *Graph) Softmax(tensor *GraphTensor, axis int) *GraphTensor {
	c_tensor := C.MPSGraphSoftmax(g.c_graph, tensor.c_tensor, C.size_t(axis))
	if c_tensor == nil {
		return nil
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    tensor.shape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

func (g *Graph) Transpose(tensor *GraphTensor, dimension, dimensionTwo int) *GraphTensor {
	c_tensor := C.MPSGraphTranspose(g.c_graph, tensor.c_tensor, C.size_t(dimension), C.size_t(dimensionTwo))
	if c_tensor == nil {
		return nil
	}
	
	// Compute transposed shape
	transposedShape := make([]int, len(tensor.shape))
	copy(transposedShape, tensor.shape)
	if dimension < len(transposedShape) && dimensionTwo < len(transposedShape) {
		transposedShape[dimension], transposedShape[dimensionTwo] = transposedShape[dimensionTwo], transposedShape[dimension]
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    transposedShape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

func (g *Graph) Reshape(tensor *GraphTensor, shape []int) *GraphTensor {
	cShape := make([]C.int, len(shape))
	for i, dim := range shape {
		cShape[i] = C.int(dim)
	}
	
	c_tensor := C.MPSGraphReshape(g.c_graph, tensor.c_tensor, &cShape[0], C.size_t(len(shape)))
	if c_tensor == nil {
		return nil
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    shape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

// Conv2D performs 2D convolution operation
func (g *Graph) Conv2D(source, weights, bias *GraphTensor, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor {
	var cBias C.MPSGraphTensorRef
	if bias != nil {
		cBias = bias.c_tensor
	}
	
	c_tensor := C.MPSGraphConvolution2D(g.c_graph, source.c_tensor, weights.c_tensor, cBias, 
		C.int(strideX), C.int(strideY), C.int(dilationX), C.int(dilationY),
		C.int(paddingLeft), C.int(paddingRight), C.int(paddingTop), C.int(paddingBottom), C.int(groups))
	if c_tensor == nil {
		return nil
	}
	
	// Calculate output shape for NCHW format
	// Output = (Input - Kernel + PaddingLeft + PaddingRight) / Stride + 1
	var outputShape []int
	if len(source.shape) == 4 && len(weights.shape) == 4 {
		N := source.shape[0]  // Batch size
		C := weights.shape[0] // Output channels
		H := source.shape[2]  // Input height
		W := source.shape[3]  // Input width
		
		kernelH := weights.shape[2]
		kernelW := weights.shape[3]
		
		outputH := (H-kernelH+paddingTop+paddingBottom)/strideY + 1
		outputW := (W-kernelW+paddingLeft+paddingRight)/strideX + 1
		
		outputShape = []int{N, C, outputH, outputW}
	} else {
		outputShape = source.shape // Fallback
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    outputShape,
		dataType: source.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

// MaxPool2D performs 2D max pooling operation
func (g *Graph) MaxPool2D(source *GraphTensor, kernelWidth, kernelHeight, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) *GraphTensor {
	c_tensor := C.MPSGraphMaxPooling2D(g.c_graph, source.c_tensor,
		C.int(kernelWidth), C.int(kernelHeight),
		C.int(strideX), C.int(strideY),
		C.int(paddingLeft), C.int(paddingRight), C.int(paddingTop), C.int(paddingBottom))
	if c_tensor == nil {
		return nil
	}
	
	// Calculate output shape for pooling
	var outputShape []int
	if len(source.shape) == 4 {
		N := source.shape[0] // Batch size
		C := source.shape[1] // Channels
		H := source.shape[2] // Input height
		W := source.shape[3] // Input width
		
		outputH := (H-kernelHeight+paddingTop+paddingBottom)/strideY + 1
		outputW := (W-kernelWidth+paddingLeft+paddingRight)/strideX + 1
		
		outputShape = []int{N, C, outputH, outputW}
	} else {
		outputShape = source.shape // Fallback
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    outputShape,
		dataType: source.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

// AvgPool2D performs 2D average pooling operation
func (g *Graph) AvgPool2D(source *GraphTensor, kernelWidth, kernelHeight, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) *GraphTensor {
	c_tensor := C.MPSGraphAvgPooling2D(g.c_graph, source.c_tensor,
		C.int(kernelWidth), C.int(kernelHeight),
		C.int(strideX), C.int(strideY),
		C.int(paddingLeft), C.int(paddingRight), C.int(paddingTop), C.int(paddingBottom))
	if c_tensor == nil {
		return nil
	}
	
	// Calculate output shape for pooling (same as MaxPool2D)
	var outputShape []int
	if len(source.shape) == 4 {
		N := source.shape[0] // Batch size
		C := source.shape[1] // Channels
		H := source.shape[2] // Input height
		W := source.shape[3] // Input width
		
		outputH := (H-kernelHeight+paddingTop+paddingBottom)/strideY + 1
		outputW := (W-kernelWidth+paddingLeft+paddingRight)/strideX + 1
		
		outputShape = []int{N, C, outputH, outputW}
	} else {
		outputShape = source.shape // Fallback
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    outputShape,
		dataType: source.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		C.ReleaseMetalObject(unsafe.Pointer(t.c_tensor))
	})
	return resultTensor
}

// Wrapper structs for MPSGraph execution
type GraphExecutable struct {
	c_executable C.MPSGraphExecutableRef
}

type GraphExecutionDescriptor struct {
	c_descriptor C.MPSGraphExecutionDescriptorRef
}

type GraphCompilationDescriptor struct {
	c_descriptor C.MPSGraphCompilationDescriptorRef
}

func NewGraphExecutionDescriptor() *GraphExecutionDescriptor {
	c_descriptor := C.CreateMPSGraphExecutionDescriptor()
	if c_descriptor == nil {
		return nil
	}
	descriptor := &GraphExecutionDescriptor{c_descriptor: c_descriptor}
	runtime.SetFinalizer(descriptor, func(d *GraphExecutionDescriptor) {
		C.ReleaseMetalObject(unsafe.Pointer(d.c_descriptor))
	})
	return descriptor
}

func NewGraphCompilationDescriptor() *GraphCompilationDescriptor {
	c_descriptor := C.CreateMPSGraphCompilationDescriptor()
	if c_descriptor == nil {
		return nil
	}
	descriptor := &GraphCompilationDescriptor{c_descriptor: c_descriptor}
	runtime.SetFinalizer(descriptor, func(d *GraphCompilationDescriptor) {
		C.ReleaseMetalObject(unsafe.Pointer(d.c_descriptor))
	})
	return descriptor
}

func (g *Graph) Compile(device *GraphDevice, inputTensors []*GraphTensor, targetTensors []*GraphTensor, compilationDescriptor *GraphCompilationDescriptor) *GraphExecutable {
	// Convert Go slices to C arrays
	var cInputTensors *C.MPSGraphTensorRef
	var cTargetTensors *C.MPSGraphTensorRef
	inputCount := len(inputTensors)
	targetCount := len(targetTensors)
	
	if len(inputTensors) > 0 {
		inputArray := make([]C.MPSGraphTensorRef, len(inputTensors))
		for i, input := range inputTensors {
			inputArray[i] = input.c_tensor
		}
		cInputTensors = (*C.MPSGraphTensorRef)(unsafe.Pointer(&inputArray[0]))
	}
	
	if len(targetTensors) > 0 {
		targetArray := make([]C.MPSGraphTensorRef, len(targetTensors))
		for i, target := range targetTensors {
			targetArray[i] = target.c_tensor
		}
		cTargetTensors = (*C.MPSGraphTensorRef)(unsafe.Pointer(&targetArray[0]))
	}
	
	c_executable := C.MPSGraphCompile(g.c_graph, device.c_device, cInputTensors, C.size_t(inputCount), cTargetTensors, C.size_t(targetCount), compilationDescriptor.c_descriptor)
	if c_executable == nil {
		return nil
	}
	
	executable := &GraphExecutable{c_executable: c_executable}
	runtime.SetFinalizer(executable, func(e *GraphExecutable) {
		C.ReleaseMetalObject(unsafe.Pointer(e.c_executable))
	})
	return executable
}

func (e *GraphExecutable) Execute(commandQueue *CommandQueue, inputTensors []*GraphTensor, inputBuffers []*Buffer, resultTensors []*GraphTensor, resultBuffers []*Buffer, executionDescriptor *GraphExecutionDescriptor) {
	inputCount := len(inputTensors)
	resultCount := len(resultTensors)
	
	// Convert Go slices to C arrays
	var inputTensorArray []C.MPSGraphTensorRef
	var inputBufferArray []C.MTLBufferRef
	var resultTensorArray []C.MPSGraphTensorRef
	var resultBufferArray []C.MTLBufferRef
	
	if inputCount > 0 {
		inputTensorArray = make([]C.MPSGraphTensorRef, inputCount)
		inputBufferArray = make([]C.MTLBufferRef, inputCount)
		for i := 0; i < inputCount; i++ {
			inputTensorArray[i] = inputTensors[i].c_tensor
			inputBufferArray[i] = inputBuffers[i].c_buffer
		}
	}
	
	if resultCount > 0 {
		resultTensorArray = make([]C.MPSGraphTensorRef, resultCount)
		resultBufferArray = make([]C.MTLBufferRef, resultCount)
		for i := 0; i < resultCount; i++ {
			resultTensorArray[i] = resultTensors[i].c_tensor
			resultBufferArray[i] = resultBuffers[i].c_buffer
		}
	}
	
	C.MPSGraphExecuteExecutable(e.c_executable, commandQueue.c_queue,
		(*C.MPSGraphTensorRef)(unsafe.Pointer(&inputTensorArray[0])),
		(*C.MTLBufferRef)(unsafe.Pointer(&inputBufferArray[0])),
		C.size_t(inputCount),
		(*C.MPSGraphTensorRef)(unsafe.Pointer(&resultTensorArray[0])),
		(*C.MTLBufferRef)(unsafe.Pointer(&resultBufferArray[0])),
		C.size_t(resultCount),
		executionDescriptor.c_descriptor)
}