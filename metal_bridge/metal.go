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
	"sync"
	"sync/atomic"
	"unsafe"
)

// Global map to store completion handlers
var (
	completionHandlers = make(map[int]func(status int))
	handlerMutex       sync.Mutex
	nextHandlerID      int = 1 // Start from 1 to avoid nil pointer issue
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
	released int32 // atomic flag to prevent double-release
}

func CreateSystemDefaultDevice() *Device {
	c_dev := C.CreateSystemDefaultDevice()
	if c_dev == nil {
		return nil
	}
	dev := &Device{c_device: c_dev}
	runtime.SetFinalizer(dev, func(d *Device) {
		d.safeRelease()
	})
	return dev
}

// safeRelease safely releases the Device using atomic operations
func (d *Device) safeRelease() {
	if atomic.CompareAndSwapInt32(&d.released, 0, 1) {
		if d.c_device != nil {
			C.ReleaseMetalObject(unsafe.Pointer(d.c_device))
			d.c_device = nil
		}
	}
}

// Wrapper struct for MTLCommandQueue
type CommandQueue struct {
	c_queue C.MTLCommandQueueRef
	released int32 // atomic flag to prevent double-release
}

func (d *Device) NewCommandQueue() *CommandQueue {
	c_q := C.CreateCommandQueue(d.c_device)
	if c_q == nil {
		return nil
	}
	q := &CommandQueue{c_queue: c_q}
	runtime.SetFinalizer(q, func(cq *CommandQueue) {
		cq.safeRelease()
	})
	return q
}

// safeRelease safely releases the CommandQueue using atomic operations
func (cq *CommandQueue) safeRelease() {
	if atomic.CompareAndSwapInt32(&cq.released, 0, 1) {
		if cq.c_queue != nil {
			C.ReleaseMetalObject(unsafe.Pointer(cq.c_queue))
			cq.c_queue = nil
		}
	}
}

// Wrapper struct for MTLBuffer
type Buffer struct {
	c_buffer  C.MTLBufferRef
	length    uintptr // Length in bytes
	inUse     bool    // Whether buffer is currently in use
	refCount  int32   // Reference count for lifetime management
	allocator *BufferAllocator // Reference to allocator for release
	released  int32   // atomic flag to prevent double-release
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
		buf := &Buffer{
			c_buffer:  c_buf, 
			length:    uintptr(byteLength),
			inUse:     true,
			refCount:  1,
			allocator: nil, // Will be set by allocator if created through it
		}
		runtime.SetFinalizer(buf, (*Buffer).finalize)
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
		buf := &Buffer{
			c_buffer:  c_buf, 
			length:    uintptr(byteLength),
			inUse:     true,
			refCount:  1,
			allocator: nil,
		}
		runtime.SetFinalizer(buf, (*Buffer).finalize)
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
		buf := &Buffer{
			c_buffer:  c_buf, 
			length:    uintptr(byteLength),
			inUse:     true,
			refCount:  1,
			allocator: nil,
		}
		runtime.SetFinalizer(buf, (*Buffer).finalize)
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
	buf := &Buffer{
		c_buffer:  c_buf, 
		length:    length,
		inUse:     true,
		refCount:  1,
		allocator: nil,
	}
	runtime.SetFinalizer(buf, (*Buffer).finalize)
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

// Retain increments the reference count for this buffer
func (b *Buffer) Retain() {
	atomic.AddInt32(&b.refCount, 1)
}

// Release decrements the reference count and returns the buffer to allocator when it reaches zero
func (b *Buffer) Release() {
	newCount := atomic.AddInt32(&b.refCount, -1)
	if newCount == 0 && b.allocator != nil {
		b.allocator.Release(b)
	} else if newCount < 0 {
		// Safety check - prevent double release
		atomic.StoreInt32(&b.refCount, 0)
	}
}

// RefCount returns the current reference count
func (b *Buffer) RefCount() int32 {
	return atomic.LoadInt32(&b.refCount)
}

// releaseNow immediately releases the Metal buffer without going through allocator
func (b *Buffer) releaseNow() {
	// Use atomic CAS to ensure we only release once
	if atomic.CompareAndSwapInt32(&b.released, 0, 1) {
		if b.c_buffer != nil {
			C.ReleaseMetalObject(unsafe.Pointer(b.c_buffer))
			b.c_buffer = nil
		}
		b.inUse = false
		atomic.StoreInt32(&b.refCount, 0)
	}
}

// finalize is called by Go's finalizer when buffer is garbage collected
func (b *Buffer) finalize() {
	if b.inUse && b.allocator != nil {
		// Buffer is still in use but being finalized - return to allocator
		b.allocator.Release(b)
	} else if b.c_buffer != nil {
		// Buffer was not managed by allocator, release directly
		b.releaseNow()
	}
}

// Wrapper struct for MTLLibrary
type Library struct {
	c_library C.MTLLibraryRef
	released  int32 // atomic flag to prevent double-release
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
		l.safeRelease()
	})
	return lib, nil
}

// safeRelease safely releases the Library using atomic operations
func (l *Library) safeRelease() {
	if atomic.CompareAndSwapInt32(&l.released, 0, 1) {
		if l.c_library != nil {
			C.ReleaseMetalObject(unsafe.Pointer(l.c_library))
			l.c_library = nil
		}
	}
}

// Wrapper struct for MTLFunction
type Function struct {
	c_function C.MTLFunctionRef
	released   int32 // atomic flag to prevent double-release
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
		f.safeRelease()
	})
	return function, nil
}

// safeRelease safely releases the Function using atomic operations
func (f *Function) safeRelease() {
	if atomic.CompareAndSwapInt32(&f.released, 0, 1) {
		if f.c_function != nil {
			C.ReleaseMetalObject(unsafe.Pointer(f.c_function))
			f.c_function = nil
		}
	}
}

// Wrapper for MTLComputePipelineState
type ComputePipelineState struct {
	c_pipelineState C.MTLComputePipelineStateRef
	released        int32 // atomic flag to prevent double-release
}

func (d *Device) NewComputePipelineStateWithFunction(function *Function) (*ComputePipelineState, error) {
	c_ps := C.CreateComputePipelineStateWithFunction(d.c_device, function.c_function)
	if c_ps == nil {
		return nil, fmt.Errorf("failed to create compute pipeline state")
	}
	C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_ps)))
	ps := &ComputePipelineState{c_pipelineState: c_ps}
	runtime.SetFinalizer(ps, func(p *ComputePipelineState) {
		p.safeRelease()
	})
	return ps, nil
}

// safeRelease safely releases the ComputePipelineState using atomic operations
func (ps *ComputePipelineState) safeRelease() {
	if atomic.CompareAndSwapInt32(&ps.released, 0, 1) {
		if ps.c_pipelineState != nil {
			C.ReleaseMetalObject(unsafe.Pointer(ps.c_pipelineState))
			ps.c_pipelineState = nil
		}
	}
}

// Wrapper for MTLCommandBuffer
type CommandBuffer struct {
	c_commandBuffer C.MTLCommandBufferRef
	// Keep a reference to resources that must stay alive until completion,
	// e.g., source Go slices for buffers, completion handler context.
	retainedResources []interface{}
	released          int32 // atomic flag to prevent double-release
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
		b.safeRelease()
	})
	return cb
}

// safeRelease safely releases the CommandBuffer using atomic operations
func (cb *CommandBuffer) safeRelease() {
	if atomic.CompareAndSwapInt32(&cb.released, 0, 1) {
		if cb.c_commandBuffer != nil {
			C.ReleaseMetalObject(unsafe.Pointer(cb.c_commandBuffer))
			cb.c_commandBuffer = nil
		}
	}
}

// Wrapper for MTLComputeCommandEncoder
type ComputeCommandEncoder struct {
	c_encoder C.MTLComputeCommandEncoderRef
	released  int32 // atomic flag to prevent double-release
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
		e.safeRelease()
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
	// Don't manually release here to avoid race with finalizer
	// The encoder will be cleaned up when Go GC runs the finalizer
}

// safeRelease safely releases the encoder using atomic operations
func (e *ComputeCommandEncoder) safeRelease() {
	// Use atomic CAS to ensure we only release once
	if atomic.CompareAndSwapInt32(&e.released, 0, 1) {
		if e.c_encoder != nil {
			C.ReleaseMetalObject(unsafe.Pointer(e.c_encoder))
			e.c_encoder = nil
		}
	}
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
	// If userData contains a handler ID, look up and execute the handler
	if userData != nil {
		// Convert userData back to handler ID
		handlerID := int(uintptr(userData))
		
		handlerMutex.Lock()
		handler, exists := completionHandlers[handlerID]
		if exists {
			delete(completionHandlers, handlerID) // Clean up after execution
		}
		handlerMutex.Unlock()
		
		if exists {
			// Execute the handler in a goroutine to avoid blocking the Metal callback
			go handler(int(statusCode))
		}
	}
}

// AddCompletedHandler allows registering a Go callback for command buffer completion.
// It passes a userData pointer which can be used to pass context to the Go function.
func (cb *CommandBuffer) AddCompletedHandler(handler func(status int)) {
	// Generate unique handler ID and store the handler
	handlerMutex.Lock()
	handlerID := nextHandlerID
	nextHandlerID++
	completionHandlers[handlerID] = handler
	handlerMutex.Unlock()
	
	// Convert handler ID to userData pointer
	userData := unsafe.Pointer(uintptr(handlerID))
	
	// Register the completion handler with the command buffer
	C.AddCommandBufferCompletedHandler(cb.c_commandBuffer, userData, (C.CompletionHandlerFunc)(C.goCommandBufferCompleted))
}

// MPSGraph wrapper structs and functions

// Wrapper struct for MPSGraph  
type Graph struct {
	c_graph  C.MPSGraphRef
	released int32 // atomic flag to prevent double-release
}

// Wrapper struct for MPSGraphDevice
type GraphDevice struct {
	c_device C.MPSGraphDeviceRef
	released int32 // atomic flag to prevent double-release
}

func NewGraph() *Graph {
	c_graph := C.CreateMPSGraph()
	if c_graph == nil {
		return nil
	}
	graph := &Graph{c_graph: c_graph}
	runtime.SetFinalizer(graph, func(g *Graph) {
		g.safeRelease()
	})
	return graph
}

// safeRelease safely releases the Graph using atomic operations
func (g *Graph) safeRelease() {
	if atomic.CompareAndSwapInt32(&g.released, 0, 1) {
		if g.c_graph != nil {
			C.ReleaseMetalObject(unsafe.Pointer(g.c_graph))
			g.c_graph = nil
		}
	}
}

func NewGraphDevice(device *Device) *GraphDevice {
	c_device := C.CreateMPSGraphDevice(device.c_device)
	if c_device == nil {
		return nil
	}
	graphDevice := &GraphDevice{c_device: c_device}
	runtime.SetFinalizer(graphDevice, func(gd *GraphDevice) {
		gd.safeRelease()
	})
	return graphDevice
}

// safeRelease safely releases the GraphDevice using atomic operations
func (gd *GraphDevice) safeRelease() {
	if atomic.CompareAndSwapInt32(&gd.released, 0, 1) {
		if gd.c_device != nil {
			C.ReleaseMetalObject(unsafe.Pointer(gd.c_device))
			gd.c_device = nil
		}
	}
}

// Wrapper struct for MPSGraphTensor
type GraphTensor struct {
	c_tensor C.MPSGraphTensorRef
	shape    []int
	dataType int
	released int32 // atomic flag to prevent double-release
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
		t.safeRelease()
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
		t.safeRelease()
	})
	return tensor
}

// safeRelease safely releases the GraphTensor using atomic operations
func (gt *GraphTensor) safeRelease() {
	if atomic.CompareAndSwapInt32(&gt.released, 0, 1) {
		if gt.c_tensor != nil {
			C.ReleaseMetalObject(unsafe.Pointer(gt.c_tensor))
			gt.c_tensor = nil
		}
	}
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
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
		t.safeRelease()
	})
	return resultTensor
}

// Wrapper structs for MPSGraph execution
type GraphExecutable struct {
	c_executable C.MPSGraphExecutableRef
	released     int32 // atomic flag to prevent double-release
}

type GraphExecutionDescriptor struct {
	c_descriptor C.MPSGraphExecutionDescriptorRef
	released     int32 // atomic flag to prevent double-release
}

type GraphCompilationDescriptor struct {
	c_descriptor C.MPSGraphCompilationDescriptorRef
	released     int32 // atomic flag to prevent double-release
}

func NewGraphExecutionDescriptor() *GraphExecutionDescriptor {
	c_descriptor := C.CreateMPSGraphExecutionDescriptor()
	if c_descriptor == nil {
		return nil
	}
	descriptor := &GraphExecutionDescriptor{c_descriptor: c_descriptor}
	runtime.SetFinalizer(descriptor, func(d *GraphExecutionDescriptor) {
		d.safeRelease()
	})
	return descriptor
}

// safeRelease safely releases the GraphExecutionDescriptor using atomic operations
func (gd *GraphExecutionDescriptor) safeRelease() {
	if atomic.CompareAndSwapInt32(&gd.released, 0, 1) {
		if gd.c_descriptor != nil {
			C.ReleaseMetalObject(unsafe.Pointer(gd.c_descriptor))
			gd.c_descriptor = nil
		}
	}
}

func NewGraphCompilationDescriptor() *GraphCompilationDescriptor {
	c_descriptor := C.CreateMPSGraphCompilationDescriptor()
	if c_descriptor == nil {
		return nil
	}
	descriptor := &GraphCompilationDescriptor{c_descriptor: c_descriptor}
	runtime.SetFinalizer(descriptor, func(d *GraphCompilationDescriptor) {
		d.safeRelease()
	})
	return descriptor
}

// safeRelease safely releases the GraphCompilationDescriptor using atomic operations
func (gcd *GraphCompilationDescriptor) safeRelease() {
	if atomic.CompareAndSwapInt32(&gcd.released, 0, 1) {
		if gcd.c_descriptor != nil {
			C.ReleaseMetalObject(unsafe.Pointer(gcd.c_descriptor))
			gcd.c_descriptor = nil
		}
	}
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
		e.safeRelease()
	})
	return executable
}

// safeRelease safely releases the GraphExecutable using atomic operations
func (ge *GraphExecutable) safeRelease() {
	if atomic.CompareAndSwapInt32(&ge.released, 0, 1) {
		if ge.c_executable != nil {
			C.ReleaseMetalObject(unsafe.Pointer(ge.c_executable))
			ge.c_executable = nil
		}
	}
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