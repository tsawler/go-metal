package metal_bridge

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -framework CoreFoundation
#include "metal_bridge.h"
// Define a Go function callable from C for completion handlers
extern void goCommandBufferCompleted(void* userData, long statusCode);
*/
import "C"
import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// goHandlerData is a small struct to hold the handler ID, allowing us to pass a pointer
// to allocated memory to C, instead of converting an integer to a pointer directly.
type goHandlerData struct {
	id int
}

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
	c_buffer C.MTLBufferRef
	released int32 // atomic flag to prevent double-release
}

func (d *Device) NewBufferWithBytes(data []byte, resourceOptions int) *Buffer {
	if len(data) == 0 {
		return nil
	}
	c_buf := C.CreateBufferWithBytes(d.c_device, unsafe.Pointer(&data[0]), C.size_t(len(data)), C.size_t(resourceOptions))
	if c_buf == nil {
		return nil
	}
	buf := &Buffer{c_buffer: c_buf}
	runtime.SetFinalizer(buf, func(b *Buffer) {
		b.safeRelease()
	})
	return buf
}

func (d *Device) NewBufferWithLength(length int, resourceOptions int) *Buffer {
	c_buf := C.CreateBufferWithLength(d.c_device, C.size_t(length), C.size_t(resourceOptions))
	if c_buf == nil {
		return nil
	}
	buf := &Buffer{c_buffer: c_buf}
	runtime.SetFinalizer(buf, func(b *Buffer) {
		b.safeRelease()
	})
	return buf
}

// safeRelease safely releases the Buffer using atomic operations
func (b *Buffer) safeRelease() {
	if atomic.CompareAndSwapInt32(&b.released, 0, 1) {
		if b.c_buffer != nil {
			C.ReleaseMetalObject(unsafe.Pointer(b.c_buffer))
			b.c_buffer = nil
		}
	}
}

func (b *Buffer) Contents() unsafe.Pointer {
	return C.GetBufferContents(b.c_buffer)
}

func (b *Buffer) Length() int {
	return int(C.GetBufferLength(b.c_buffer))
}

// Wrapper struct for MTLLibrary
type Library struct {
	c_library C.MTLLibraryRef
	released int32 // atomic flag to prevent double-release
}

func (d *Device) NewLibraryWithSource(source string) *Library {
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	
	c_lib := C.CreateLibraryWithSource(d.c_device, cSource)
	if c_lib == nil {
		return nil
	}
	lib := &Library{c_library: c_lib}
	runtime.SetFinalizer(lib, func(l *Library) {
		l.safeRelease()
	})
	return lib
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
	released int32 // atomic flag to prevent double-release
}

func (l *Library) NewFunctionWithName(name string) *Function {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	
	c_func := C.GetFunction(l.c_library, cName)
	if c_func == nil {
		return nil
	}
	function := &Function{c_function: c_func}
	runtime.SetFinalizer(function, func(f *Function) {
		f.safeRelease()
	})
	return function
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

// Wrapper struct for MTLComputePipelineState
type ComputePipelineState struct {
	c_pipelineState C.MTLComputePipelineStateRef
	released int32 // atomic flag to prevent double-release
}

func (d *Device) NewComputePipelineStateWithFunction(function *Function) *ComputePipelineState {
	c_ps := C.CreateComputePipelineStateWithFunction(d.c_device, function.c_function)
	if c_ps == nil {
		return nil
	}
	ps := &ComputePipelineState{c_pipelineState: c_ps}
	runtime.SetFinalizer(ps, func(cps *ComputePipelineState) {
		cps.safeRelease()
	})
	return ps
}

// safeRelease safely releases the ComputePipelineState using atomic operations
func (cps *ComputePipelineState) safeRelease() {
	if atomic.CompareAndSwapInt32(&cps.released, 0, 1) {
		if cps.c_pipelineState != nil {
			C.ReleaseMetalObject(unsafe.Pointer(cps.c_pipelineState))
			cps.c_pipelineState = nil
		}
	}
}

// Wrapper struct for MTLCommandBuffer
type CommandBuffer struct {
	c_commandBuffer C.MTLCommandBufferRef
	released int32 // atomic flag to prevent double-release
}

func (cq *CommandQueue) NewCommandBuffer() *CommandBuffer {
	c_cb := C.CreateCommandBuffer(cq.c_queue)
	if c_cb == nil {
		return nil
	}
	cb := &CommandBuffer{c_commandBuffer: c_cb}
	runtime.SetFinalizer(cb, func(commandBuffer *CommandBuffer) {
		commandBuffer.safeRelease()
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

func (cb *CommandBuffer) Commit() {
	C.CommitCommandBuffer(cb.c_commandBuffer)
}

func (cb *CommandBuffer) WaitUntilCompleted() {
	C.WaitUntilCommandBufferCompleted(cb.c_commandBuffer)
}

func (cb *CommandBuffer) AddCompletedHandler(handler func(status int)) {
	handlerMutex.Lock()
	id := nextHandlerID
	nextHandlerID++
	completionHandlers[id] = handler
	handlerMutex.Unlock()
	
	// Create handler data and pass to C
	handlerData := &goHandlerData{id: id}
	C.AddCommandBufferCompletedHandler(cb.c_commandBuffer, unsafe.Pointer(handlerData), C.CompletionHandlerFunc(C.goCommandBufferCompleted))
}

// Export this function for the C code to call
//export goCommandBufferCompleted
func goCommandBufferCompleted(userData unsafe.Pointer, statusCode C.long) {
	if userData == nil {
		return
	}
	
	handlerData := (*goHandlerData)(userData)
	
	handlerMutex.Lock()
	handler, exists := completionHandlers[handlerData.id]
	if exists {
		delete(completionHandlers, handlerData.id)
	}
	handlerMutex.Unlock()
	
	if exists {
		handler(int(statusCode))
	}
}

// Wrapper struct for MTLComputeCommandEncoder
type ComputeCommandEncoder struct {
	c_encoder C.MTLComputeCommandEncoderRef
	released int32 // atomic flag to prevent double-release
}

func (cb *CommandBuffer) NewComputeCommandEncoder() *ComputeCommandEncoder {
	c_enc := C.CreateComputeCommandEncoder(cb.c_commandBuffer)
	if c_enc == nil {
		return nil
	}
	enc := &ComputeCommandEncoder{c_encoder: c_enc}
	runtime.SetFinalizer(enc, func(encoder *ComputeCommandEncoder) {
		encoder.safeRelease()
	})
	return enc
}

// safeRelease safely releases the ComputeCommandEncoder using atomic operations
func (cce *ComputeCommandEncoder) safeRelease() {
	if atomic.CompareAndSwapInt32(&cce.released, 0, 1) {
		if cce.c_encoder != nil {
			C.ReleaseMetalObject(unsafe.Pointer(cce.c_encoder))
			cce.c_encoder = nil
		}
	}
}

func (cce *ComputeCommandEncoder) SetComputePipelineState(pipelineState *ComputePipelineState) {
	C.SetComputePipelineState(cce.c_encoder, pipelineState.c_pipelineState)
}

func (cce *ComputeCommandEncoder) SetBuffer(buffer *Buffer, offset, index int) {
	C.SetBuffer(cce.c_encoder, buffer.c_buffer, C.size_t(offset), C.size_t(index))
}

func (cce *ComputeCommandEncoder) DispatchThreads(gridSize, threadgroupSize MTLSize) {
	C.DispatchThreads(cce.c_encoder, 
		C.size_t(gridSize.Width), C.size_t(gridSize.Height), C.size_t(gridSize.Depth),
		C.size_t(threadgroupSize.Width), C.size_t(threadgroupSize.Height), C.size_t(threadgroupSize.Depth))
}

func (cce *ComputeCommandEncoder) EndEncoding() {
	C.EndEncoding(cce.c_encoder)
}

// MPSGraph wrapper types
type Graph struct {
	c_graph C.MPSGraphRef
	released int32 // atomic flag to prevent double-release
}

func NewGraph() *Graph {
	c_g := C.CreateMPSGraph()
	if c_g == nil {
		return nil
	}
	g := &Graph{c_graph: c_g}
	runtime.SetFinalizer(g, func(graph *Graph) {
		graph.safeRelease()
	})
	return g
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

// Wrapper struct for MPSGraphDevice
type GraphDevice struct {
	c_device C.MPSGraphDeviceRef
	released int32 // atomic flag to prevent double-release
}

func NewGraphDevice(device *Device) *GraphDevice {
	c_gd := C.CreateMPSGraphDevice(device.c_device)
	if c_gd == nil {
		return nil
	}
	gd := &GraphDevice{c_device: c_gd}
	runtime.SetFinalizer(gd, func(graphDevice *GraphDevice) {
		graphDevice.safeRelease()
	})
	return gd
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

// safeRelease safely releases the GraphTensor using atomic operations
func (gt *GraphTensor) safeRelease() {
	if atomic.CompareAndSwapInt32(&gt.released, 0, 1) {
		if gt.c_tensor != nil {
			C.ReleaseMetalObject(unsafe.Pointer(gt.c_tensor))
			gt.c_tensor = nil
		}
	}
}

func (gt *GraphTensor) Shape() []int {
	return gt.shape
}

func (gt *GraphTensor) DataType() int {
	return gt.dataType
}

// Create a placeholder tensor
func (g *Graph) PlaceholderTensor(shape []int, dataType int) *GraphTensor {
	// Convert shape to C array
	cShape := make([]C.int, len(shape))
	for i, dim := range shape {
		cShape[i] = C.int(dim)
	}
	
	var cShapePtr *C.int
	if len(cShape) > 0 {
		cShapePtr = &cShape[0]
	}
	
	c_tensor := C.MPSGraphPlaceholderTensor(g.c_graph, cShapePtr, C.size_t(len(shape)), C.int(dataType))
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

// Create a constant tensor
func (g *Graph) ConstantTensor(value float64, shape []int, dataType int) *GraphTensor {
	// Convert shape to C array
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

// Addition operation
func (g *Graph) Addition(a, b *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphAddition(g.c_graph, a.c_tensor, b.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	// Use broadcasting shape calculation
	resultShape := broadcastShapes(a.shape, b.shape)
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    resultShape,
		dataType: a.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Subtraction operation
func (g *Graph) Subtraction(a, b *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphSubtraction(g.c_graph, a.c_tensor, b.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	// Use broadcasting shape calculation
	resultShape := broadcastShapes(a.shape, b.shape)
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    resultShape,
		dataType: a.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Multiplication operation
func (g *Graph) Multiplication(a, b *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphMultiplication(g.c_graph, a.c_tensor, b.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	// Use broadcasting shape calculation
	resultShape := broadcastShapes(a.shape, b.shape)
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    resultShape,
		dataType: a.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Division operation
func (g *Graph) Division(a, b *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphDivision(g.c_graph, a.c_tensor, b.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	// Use broadcasting shape calculation
	resultShape := broadcastShapes(a.shape, b.shape)
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    resultShape,
		dataType: a.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Matrix multiplication operation
func (g *Graph) MatrixMultiplication(a, b *GraphTensor) *GraphTensor {
	c_tensor := C.MPSGraphMatrixMultiplication(g.c_graph, a.c_tensor, b.c_tensor)
	if c_tensor == nil {
		return nil
	}
	
	// Calculate the result shape for matrix multiplication
	var resultShape []int
	if len(a.shape) >= 2 && len(b.shape) >= 2 {
		// Basic matrix multiplication: [..., M, K] x [..., K, N] -> [..., M, N]
		aRows := a.shape[len(a.shape)-2]
		bCols := b.shape[len(b.shape)-1]
		
		// Handle batch dimensions
		if len(a.shape) > 2 || len(b.shape) > 2 {
			// Use broadcasting for batch dimensions
			aBatchDims := a.shape[:len(a.shape)-2]
			bBatchDims := b.shape[:len(b.shape)-2]
			batchShape := broadcastShapes(aBatchDims, bBatchDims)
			resultShape = append(batchShape, aRows, bCols)
		} else {
			resultShape = []int{aRows, bCols}
		}
	} else {
		// Fallback
		resultShape = a.shape
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    resultShape,
		dataType: a.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// ReLU activation function
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

// Sigmoid activation function
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

// Softmax activation function
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

// Transpose operation
func (g *Graph) Transpose(tensor *GraphTensor, dimension1, dimension2 int) *GraphTensor {
	c_tensor := C.MPSGraphTranspose(g.c_graph, tensor.c_tensor, C.size_t(dimension1), C.size_t(dimension2))
	if c_tensor == nil {
		return nil
	}
	
	// Calculate the result shape after transposition
	resultShape := make([]int, len(tensor.shape))
	copy(resultShape, tensor.shape)
	resultShape[dimension1], resultShape[dimension2] = resultShape[dimension2], resultShape[dimension1]
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    resultShape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Reshape operation
func (g *Graph) Reshape(tensor *GraphTensor, shape []int) *GraphTensor {
	// Convert shape to C array
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

// Wrapper structs for MPSGraph compilation and execution
type GraphExecutionDescriptor struct {
	c_descriptor C.MPSGraphExecutionDescriptorRef
	released int32 // atomic flag to prevent double-release
}

func NewGraphExecutionDescriptor() *GraphExecutionDescriptor {
	c_desc := C.CreateMPSGraphExecutionDescriptor()
	if c_desc == nil {
		return nil
	}
	desc := &GraphExecutionDescriptor{c_descriptor: c_desc}
	runtime.SetFinalizer(desc, func(ged *GraphExecutionDescriptor) {
		ged.safeRelease()
	})
	return desc
}

// safeRelease safely releases the GraphExecutionDescriptor using atomic operations
func (ged *GraphExecutionDescriptor) safeRelease() {
	if atomic.CompareAndSwapInt32(&ged.released, 0, 1) {
		if ged.c_descriptor != nil {
			C.ReleaseMetalObject(unsafe.Pointer(ged.c_descriptor))
			ged.c_descriptor = nil
		}
	}
}

type GraphCompilationDescriptor struct {
	c_descriptor C.MPSGraphCompilationDescriptorRef
	released int32 // atomic flag to prevent double-release
}

func NewGraphCompilationDescriptor() *GraphCompilationDescriptor {
	c_desc := C.CreateMPSGraphCompilationDescriptor()
	if c_desc == nil {
		return nil
	}
	desc := &GraphCompilationDescriptor{c_descriptor: c_desc}
	runtime.SetFinalizer(desc, func(gcd *GraphCompilationDescriptor) {
		gcd.safeRelease()
	})
	return desc
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

type GraphExecutable struct {
	c_executable C.MPSGraphExecutableRef
	released int32 // atomic flag to prevent double-release
}

func (g *Graph) Compile(device *GraphDevice, inputTensors, targetTensors []*GraphTensor, compilationDescriptor *GraphCompilationDescriptor) *GraphExecutable {
	inputCount := len(inputTensors)
	targetCount := len(targetTensors)
	
	// Convert Go slices to C arrays
	var inputArray []C.MPSGraphTensorRef
	var targetArray []C.MPSGraphTensorRef
	
	if inputCount > 0 {
		inputArray = make([]C.MPSGraphTensorRef, inputCount)
		for i := 0; i < inputCount; i++ {
			inputArray[i] = inputTensors[i].c_tensor
		}
	}
	
	if targetCount > 0 {
		targetArray = make([]C.MPSGraphTensorRef, targetCount)
		for i := 0; i < targetCount; i++ {
			targetArray[i] = targetTensors[i].c_tensor
		}
	}
	
	c_executable := C.MPSGraphCompile(g.c_graph, device.c_device,
		(*C.MPSGraphTensorRef)(unsafe.Pointer(&inputArray[0])), C.size_t(inputCount),
		(*C.MPSGraphTensorRef)(unsafe.Pointer(&targetArray[0])), C.size_t(targetCount),
		compilationDescriptor.c_descriptor)
	
	if c_executable == nil {
		return nil
	}
	
	executable := &GraphExecutable{c_executable: c_executable}
	runtime.SetFinalizer(executable, func(ge *GraphExecutable) {
		ge.safeRelease()
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
// ReductionSum performs tensor reduction sum along specified axis
func (g *Graph) ReductionSum(tensor *GraphTensor, axis int, keepdim bool) *GraphTensor {
	keepdimInt := 0
	if keepdim {
		keepdimInt = 1
	}
	
	c_tensor := C.MPSGraphReductionSum(g.c_graph, tensor.c_tensor, C.int(axis), C.int(keepdimInt))
	if c_tensor == nil {
		return nil
	}
	
	// Calculate output shape
	outputShape := make([]int, 0, len(tensor.shape))
	for i, dim := range tensor.shape {
		if i == axis {
			if keepdim {
				outputShape = append(outputShape, 1)
			}
			// Skip this dimension if not keeping it
		} else {
			outputShape = append(outputShape, dim)
		}
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    outputShape,
		dataType: tensor.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// ConvolutionTranspose2D performs 2D transposed convolution operation
func (g *Graph) ConvolutionTranspose2D(source, weights *GraphTensor, outputShape []int, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor {
	// Convert Go slice to C array
	cOutputShape := make([]C.int, len(outputShape))
	for i, dim := range outputShape {
		cOutputShape[i] = C.int(dim)
	}
	
	c_tensor := C.MPSGraphConvolutionTranspose2D(g.c_graph, source.c_tensor, weights.c_tensor, 
		&cOutputShape[0], C.size_t(len(outputShape)),
		C.int(strideX), C.int(strideY), C.int(dilationX), C.int(dilationY),
		C.int(paddingLeft), C.int(paddingRight), C.int(paddingTop), C.int(paddingBottom), C.int(groups))
	if c_tensor == nil {
		return nil
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

// Convolution2DDataGradient computes the gradient with respect to input data
func (g *Graph) Convolution2DDataGradient(incomingGradient, weights *GraphTensor, inputShape []int, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor {
	// Convert Go slice to C array
	cInputShape := make([]C.int, len(inputShape))
	for i, dim := range inputShape {
		cInputShape[i] = C.int(dim)
	}
	
	c_tensor := C.MPSGraphConvolution2DDataGradient(g.c_graph, incomingGradient.c_tensor, weights.c_tensor, 
		&cInputShape[0], C.size_t(len(inputShape)),
		C.int(strideX), C.int(strideY), C.int(dilationX), C.int(dilationY),
		C.int(paddingLeft), C.int(paddingRight), C.int(paddingTop), C.int(paddingBottom), C.int(groups))
	if c_tensor == nil {
		return nil
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    inputShape,
		dataType: incomingGradient.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Convolution2DWeightsGradient computes the gradient with respect to weights
func (g *Graph) Convolution2DWeightsGradient(incomingGradient, source *GraphTensor, weightsShape []int, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor {
	// Convert Go slice to C array
	cWeightsShape := make([]C.int, len(weightsShape))
	for i, dim := range weightsShape {
		cWeightsShape[i] = C.int(dim)
	}
	
	c_tensor := C.MPSGraphConvolution2DWeightsGradient(g.c_graph, incomingGradient.c_tensor, source.c_tensor, 
		&cWeightsShape[0], C.size_t(len(weightsShape)),
		C.int(strideX), C.int(strideY), C.int(dilationX), C.int(dilationY),
		C.int(paddingLeft), C.int(paddingRight), C.int(paddingTop), C.int(paddingBottom), C.int(groups))
	if c_tensor == nil {
		return nil
	}
	
	resultTensor := &GraphTensor{
		c_tensor: c_tensor,
		shape:    weightsShape,
		dataType: incomingGradient.dataType,
	}
	runtime.SetFinalizer(resultTensor, func(t *GraphTensor) {
		t.safeRelease()
	})
	return resultTensor
}

// Helper function to calculate broadcasting shapes
func broadcastShapes(shape1, shape2 []int) []int {
	// Determine the maximum number of dimensions
	maxDims := len(shape1)
	if len(shape2) > maxDims {
		maxDims = len(shape2)
	}
	
	// Pad shapes with leading 1s
	paddedShape1 := make([]int, maxDims)
	paddedShape2 := make([]int, maxDims)
	
	for i := 0; i < maxDims; i++ {
		if i < len(shape1) {
			paddedShape1[maxDims-1-i] = shape1[len(shape1)-1-i]
		} else {
			paddedShape1[maxDims-1-i] = 1
		}
		
		if i < len(shape2) {
			paddedShape2[maxDims-1-i] = shape2[len(shape2)-1-i]
		} else {
			paddedShape2[maxDims-1-i] = 1
		}
	}
	
	// Calculate broadcasted shape
	result := make([]int, maxDims)
	for i := 0; i < maxDims; i++ {
		if paddedShape1[i] == paddedShape2[i] {
			result[i] = paddedShape1[i]
		} else if paddedShape1[i] == 1 {
			result[i] = paddedShape2[i]
		} else if paddedShape2[i] == 1 {
			result[i] = paddedShape1[i]
		} else {
			// Incompatible shapes - this should be an error in practice
			// For now, use the first shape's dimension
			result[i] = paddedShape1[i]
		}
	}
	
	return result
}