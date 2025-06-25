package metal_bridge

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation
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