package memory

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

// DataType represents the data type of tensor elements
type DataType int

const (
	Float32 DataType = iota
	Int32
	Float16
	Int8
)

// DeviceType represents where the tensor data resides
type DeviceType int

const (
	CPU DeviceType = iota
	GPU
	PersistentGPU
)

// Tensor represents a GPU-resident tensor with reference counting
type Tensor struct {
	metalBuffer unsafe.Pointer // MTLBuffer ID (C pointer)
	shape       []int
	dtype       DataType
	device      DeviceType
	refCount    *int32 // Atomic reference count
	pooled      bool   // Can be returned to pool
	generation  uint64 // For debugging use-after-free
	size        int    // Total size in bytes
}

// NewTensor creates a new GPU-resident tensor
func NewTensor(shape []int, dtype DataType, device DeviceType) (*Tensor, error) {
	size := calculateSize(shape, dtype)
	
	// Get buffer from memory manager
	buffer, err := GetGlobalMemoryManager().GetBuffer(size, device)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate buffer: %v", err)
	}
	
	refCount := int32(1) // Start with 1 reference
	
	tensor := &Tensor{
		metalBuffer: buffer,
		shape:       make([]int, len(shape)),
		dtype:       dtype,
		device:      device,
		refCount:    &refCount,
		pooled:      true,
		generation:  atomic.AddUint64(&globalGeneration, 1),
		size:        size,
	}
	copy(tensor.shape, shape) // Copy to avoid external modification
	return tensor, nil
}

// Retain increments the reference count and returns the same tensor
func (t *Tensor) Retain() *Tensor {
	if t.refCount == nil {
		panic("tensor already released")
	}
	atomic.AddInt32(t.refCount, 1)
	return t
}

// Release decrements the reference count and returns buffer to pool when it reaches 0
func (t *Tensor) Release() {
	if t.refCount == nil {
		return // Already released
	}
	
	if atomic.AddInt32(t.refCount, -1) == 0 {
		if t.pooled && t.metalBuffer != nil {
			GetGlobalMemoryManager().ReturnBuffer(t.metalBuffer, t.size, t.device)
		}
		
		// Clear fields to prevent use-after-free
		t.metalBuffer = nil
		t.refCount = nil
		t.shape = nil
	}
}

// Clone returns the same tensor with incremented reference count
func (t *Tensor) Clone() *Tensor {
	return t.Retain()
}

// Shape returns the tensor shape (defensive copy)
func (t *Tensor) Shape() []int {
	result := make([]int, len(t.shape))
	copy(result, t.shape)
	return result
}

// DType returns the data type
func (t *Tensor) DType() DataType {
	return t.dtype
}

// Device returns the device type
func (t *Tensor) Device() DeviceType {
	return t.device
}

// MetalBuffer returns the underlying Metal buffer pointer
func (t *Tensor) MetalBuffer() unsafe.Pointer {
	if t.metalBuffer == nil {
		panic("tensor buffer is nil - tensor may have been released")
	}
	return t.metalBuffer
}

// Size returns the total size in bytes
func (t *Tensor) Size() int {
	return t.size
}

// RefCount returns the current reference count (for debugging)
func (t *Tensor) RefCount() int32 {
	if t.refCount == nil {
		return 0
	}
	return atomic.LoadInt32(t.refCount)
}

// calculateSize computes the total size in bytes for the given shape and dtype
func calculateSize(shape []int, dtype DataType) int {
	if len(shape) == 0 {
		return 0
	}
	
	elements := 1
	for _, dim := range shape {
		elements *= dim
	}
	
	var elementSize int
	switch dtype {
	case Float32, Int32:
		elementSize = 4
	case Float16:
		elementSize = 2
	case Int8:
		elementSize = 1
	default:
		panic(fmt.Sprintf("unsupported data type: %d", dtype))
	}
	
	return elements * elementSize
}

// Global generation counter for debugging
var globalGeneration uint64

// CopyFloat32Data copies float32 data to the tensor's Metal buffer
func (t *Tensor) CopyFloat32Data(data []float32) error {
	if t.dtype != Float32 {
		return fmt.Errorf("tensor data type is %d, expected Float32 (%d)", t.dtype, Float32)
	}
	
	expectedElements := 1
	for _, dim := range t.shape {
		expectedElements *= dim
	}
	
	if len(data) != expectedElements {
		return fmt.Errorf("data length %d doesn't match tensor shape %v (expected %d elements)", 
			len(data), t.shape, expectedElements)
	}
	
	// Import the bridge package for data transfer
	return t.copyDataToBuffer(data)
}

// CopyInt32Data copies int32 data to the tensor's Metal buffer
func (t *Tensor) CopyInt32Data(data []int32) error {
	if t.dtype != Int32 {
		return fmt.Errorf("tensor data type is %d, expected Int32 (%d)", t.dtype, Int32)
	}
	
	expectedElements := 1
	for _, dim := range t.shape {
		expectedElements *= dim
	}
	
	if len(data) != expectedElements {
		return fmt.Errorf("data length %d doesn't match tensor shape %v (expected %d elements)", 
			len(data), t.shape, expectedElements)
	}
	
	// Import the bridge package for data transfer
	return t.copyDataToBuffer(data)
}

// copyDataToBuffer is a helper that calls the appropriate CGO bridge function
func (t *Tensor) copyDataToBuffer(data interface{}) error {
	switch d := data.(type) {
	case []float32:
		if CopyFloat32DataFunc == nil {
			return fmt.Errorf("CopyFloat32Data bridge function not initialized - import cgo_bridge package")
		}
		return CopyFloat32DataFunc(t.metalBuffer, d)
	case []int32:
		if CopyInt32DataFunc == nil {
			return fmt.Errorf("CopyInt32Data bridge function not initialized - import cgo_bridge package")
		}
		return CopyInt32DataFunc(t.metalBuffer, d)
	default:
		return fmt.Errorf("unsupported data type for copy: %T", data)
	}
}

// Bridge functions for data transfer - set up during cgo_bridge initialization
var ToFloat32SliceFunc func(buffer unsafe.Pointer, numElements int) ([]float32, error)
var CopyFloat32DataFunc func(buffer unsafe.Pointer, data []float32) error
var CopyInt32DataFunc func(buffer unsafe.Pointer, data []int32) error

// SetupBridge allows external packages to set up bridge functions
func SetupBridge(
	toFloat32SliceFunc func(unsafe.Pointer, int) ([]float32, error),
	copyFloat32DataFunc func(unsafe.Pointer, []float32) error,
	copyInt32DataFunc func(unsafe.Pointer, []int32) error,
) {
	ToFloat32SliceFunc = toFloat32SliceFunc
	CopyFloat32DataFunc = copyFloat32DataFunc
	CopyInt32DataFunc = copyInt32DataFunc
}

func (t *Tensor) ToFloat32Slice() ([]float32, error) {
	if t.dtype != Float32 {
		return nil, fmt.Errorf("tensor data type is %d, expected Float32 (%d)", t.dtype, Float32)
	}
	
	expectedElements := 1
	for _, dim := range t.shape {
		expectedElements *= dim
	}
	
	if expectedElements <= 0 {
		return nil, fmt.Errorf("tensor has invalid shape: %v", t.shape)
	}
	
	if ToFloat32SliceFunc == nil {
		return nil, fmt.Errorf("ToFloat32Slice bridge function not initialized - import cgo_bridge package")
	}
	
	return ToFloat32SliceFunc(t.metalBuffer, expectedElements)
}

// String returns a string representation for debugging
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor{shape=%v, dtype=%d, device=%d, refs=%d, gen=%d}", 
		t.shape, t.dtype, t.device, t.RefCount(), t.generation)
}