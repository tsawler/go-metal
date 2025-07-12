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

// ConvertTo creates a new tensor with the specified data type, performing type conversion on GPU
func (t *Tensor) ConvertTo(dtype DataType) (*Tensor, error) {
	if t.dtype == dtype {
		// Already the correct type, just retain and return
		return t.Retain(), nil
	}
	
	// Create new tensor with target dtype
	newTensor, err := NewTensor(t.shape, dtype, t.device)
	if err != nil {
		return nil, fmt.Errorf("failed to create converted tensor: %v", err)
	}
	
	// Perform GPU-side conversion
	err = convertTensorType(t.metalBuffer, newTensor.metalBuffer, t.shape, t.dtype, dtype)
	if err != nil {
		newTensor.Release()
		return nil, fmt.Errorf("failed to convert tensor type: %v", err)
	}
	
	return newTensor, nil
}

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

// CopyFrom copies data from another tensor using GPU-resident buffer operations
// This performs a direct GPU-to-GPU memory transfer without CPU involvement
func (t *Tensor) CopyFrom(src *Tensor) error {
	// Validate tensors are compatible
	if src == nil {
		return fmt.Errorf("source tensor is nil")
	}
	
	if t.metalBuffer == nil {
		return fmt.Errorf("destination tensor has nil metal buffer")
	}
	
	if src.metalBuffer == nil {
		return fmt.Errorf("source tensor has nil metal buffer")
	}
	
	// Check that shapes are compatible
	if len(t.shape) != len(src.shape) {
		return fmt.Errorf("tensor shape dimensions don't match: dst %v vs src %v", t.shape, src.shape)
	}
	
	for i, dim := range t.shape {
		if dim != src.shape[i] {
			return fmt.Errorf("tensor shapes don't match: dst %v vs src %v", t.shape, src.shape)
		}
	}
	
	// Check that data types are compatible
	if t.dtype != src.dtype {
		return fmt.Errorf("tensor data types don't match: dst %v vs src %v", t.dtype, src.dtype)
	}
	
	// Calculate copy size
	copySize := t.size
	if src.size != copySize {
		return fmt.Errorf("tensor sizes don't match: dst %d bytes vs src %d bytes", copySize, src.size)
	}
	
	// Perform GPU-resident buffer copy using bridge function
	if CopyTensorFunc == nil {
		return fmt.Errorf("CopyTensor bridge function not initialized - import cgo_bridge package")
	}
	
	// Direct GPU-to-GPU memory transfer (GPU-resident everything principle)
	return CopyTensorFunc(src.metalBuffer, t.metalBuffer, copySize)
}

// Bridge functions for data transfer - set up during cgo_bridge initialization
var ToFloat32SliceFunc func(buffer unsafe.Pointer, numElements int) ([]float32, error)
var CopyFloat32DataFunc func(buffer unsafe.Pointer, data []float32) error
var CopyInt32DataFunc func(buffer unsafe.Pointer, data []int32) error
var ConvertTensorTypeFunc func(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType DataType) error
var CopyTensorFunc func(srcBuffer, dstBuffer unsafe.Pointer, size int) error

// convertTensorType performs GPU-side type conversion
func convertTensorType(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType DataType) error {
	if ConvertTensorTypeFunc == nil {
		return fmt.Errorf("ConvertTensorType bridge function not initialized - import cgo_bridge package")
	}
	return ConvertTensorTypeFunc(srcBuffer, dstBuffer, shape, srcType, dstType)
}

// GetDevice returns the device pointer from the global memory manager
func GetDevice() unsafe.Pointer {
	return GetGlobalMemoryManager().device
}

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

// SetupBridgeWithConvert allows external packages to set up bridge functions including type conversion
func SetupBridgeWithConvert(
	toFloat32SliceFunc func(unsafe.Pointer, int) ([]float32, error),
	copyFloat32DataFunc func(unsafe.Pointer, []float32) error,
	copyInt32DataFunc func(unsafe.Pointer, []int32) error,
	convertTensorTypeFunc func(unsafe.Pointer, unsafe.Pointer, []int, int, int) error,
	copyTensorFunc func(unsafe.Pointer, unsafe.Pointer, int) error,
) {
	ToFloat32SliceFunc = toFloat32SliceFunc
	CopyFloat32DataFunc = copyFloat32DataFunc
	CopyInt32DataFunc = copyInt32DataFunc
	CopyTensorFunc = copyTensorFunc
	
	// Create wrapper that converts int to DataType
	ConvertTensorTypeFunc = func(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType DataType) error {
		return convertTensorTypeFunc(srcBuffer, dstBuffer, shape, int(srcType), int(dstType))
	}
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