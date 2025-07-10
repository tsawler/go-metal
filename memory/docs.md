# memory
--
    import "."


## Usage

```go
var ConvertTensorTypeFunc func(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType DataType) error
```

```go
var CopyFloat32DataFunc func(buffer unsafe.Pointer, data []float32) error
```

```go
var CopyInt32DataFunc func(buffer unsafe.Pointer, data []int32) error
```

```go
var ToFloat32SliceFunc func(buffer unsafe.Pointer, numElements int) ([]float32, error)
```
Bridge functions for data transfer - set up during cgo_bridge initialization

#### func  CreateMockCommandQueue

```go
func CreateMockCommandQueue() unsafe.Pointer
```
CreateMockCommandQueue creates a mock Metal command queue for testing

#### func  CreateMockDevice

```go
func CreateMockDevice() unsafe.Pointer
```
CreateMockDevice creates a mock Metal device for testing

#### func  GetDevice

```go
func GetDevice() unsafe.Pointer
```
GetDevice returns the device pointer from the global memory manager

#### func  InitializeGlobalMemoryManager

```go
func InitializeGlobalMemoryManager(device unsafe.Pointer)
```
InitializeGlobalMemoryManager initializes the global memory manager

#### func  SetupBridge

```go
func SetupBridge(
	toFloat32SliceFunc func(unsafe.Pointer, int) ([]float32, error),
	copyFloat32DataFunc func(unsafe.Pointer, []float32) error,
	copyInt32DataFunc func(unsafe.Pointer, []int32) error,
)
```
SetupBridge allows external packages to set up bridge functions

#### func  SetupBridgeWithConvert

```go
func SetupBridgeWithConvert(
	toFloat32SliceFunc func(unsafe.Pointer, int) ([]float32, error),
	copyFloat32DataFunc func(unsafe.Pointer, []float32) error,
	copyInt32DataFunc func(unsafe.Pointer, []int32) error,
	convertTensorTypeFunc func(unsafe.Pointer, unsafe.Pointer, []int, int, int) error,
)
```
SetupBridgeWithConvert allows external packages to set up bridge functions
including type conversion

#### type BufferPool

```go
type BufferPool struct {
}
```

BufferPool manages a pool of Metal buffers of a specific size

#### func  NewBufferPool

```go
func NewBufferPool(bufferSize int, maxSize int, device DeviceType) *BufferPool
```
NewBufferPool creates a new buffer pool

#### func (*BufferPool) Get

```go
func (bp *BufferPool) Get() (unsafe.Pointer, error)
```
Get retrieves a buffer from the pool or allocates a new one

#### func (*BufferPool) Return

```go
func (bp *BufferPool) Return(buffer unsafe.Pointer)
```
Return puts a buffer back into the pool

#### func (*BufferPool) Stats

```go
func (bp *BufferPool) Stats() (available int, allocated int, maxSize int)
```
Stats returns pool statistics

#### type DataType

```go
type DataType int
```

DataType represents the data type of tensor elements

```go
const (
	Float32 DataType = iota
	Int32
	Float16
	Int8
)
```

#### type DeviceType

```go
type DeviceType int
```

DeviceType represents where the tensor data resides

```go
const (
	CPU DeviceType = iota
	GPU
	PersistentGPU
)
```

#### type MemoryManager

```go
type MemoryManager struct {
}
```

MemoryManager manages GPU buffer lifecycle and pooling

#### func  GetGlobalMemoryManager

```go
func GetGlobalMemoryManager() *MemoryManager
```
GetGlobalMemoryManager returns the global memory manager instance

#### func  NewMemoryManager

```go
func NewMemoryManager(device unsafe.Pointer) *MemoryManager
```
NewMemoryManager creates a new memory manager

#### func (*MemoryManager) AllocateBuffer

```go
func (mm *MemoryManager) AllocateBuffer(size int) unsafe.Pointer
```
AllocateBuffer is a simple interface for external packages

#### func (*MemoryManager) GetBuffer

```go
func (mm *MemoryManager) GetBuffer(size int, device DeviceType) (unsafe.Pointer, error)
```
GetBuffer gets a buffer of at least the specified size

#### func (*MemoryManager) ReleaseBuffer

```go
func (mm *MemoryManager) ReleaseBuffer(buffer unsafe.Pointer)
```
ReleaseBuffer is a simple interface for external packages

#### func (*MemoryManager) ReturnBuffer

```go
func (mm *MemoryManager) ReturnBuffer(buffer unsafe.Pointer, size int, device DeviceType)
```
ReturnBuffer returns a buffer to the appropriate pool

#### func (*MemoryManager) Stats

```go
func (mm *MemoryManager) Stats() map[PoolKey]string
```
Stats returns memory manager statistics

#### type PoolKey

```go
type PoolKey struct {
	Size   int
	Device DeviceType
}
```

PoolKey represents a key for the buffer pool map

#### type Tensor

```go
type Tensor struct {
}
```

Tensor represents a GPU-resident tensor with reference counting

#### func  NewTensor

```go
func NewTensor(shape []int, dtype DataType, device DeviceType) (*Tensor, error)
```
NewTensor creates a new GPU-resident tensor

#### func (*Tensor) Clone

```go
func (t *Tensor) Clone() *Tensor
```
Clone returns the same tensor with incremented reference count

#### func (*Tensor) ConvertTo

```go
func (t *Tensor) ConvertTo(dtype DataType) (*Tensor, error)
```
ConvertTo creates a new tensor with the specified data type, performing type
conversion on GPU

#### func (*Tensor) CopyFloat32Data

```go
func (t *Tensor) CopyFloat32Data(data []float32) error
```
CopyFloat32Data copies float32 data to the tensor's Metal buffer

#### func (*Tensor) CopyInt32Data

```go
func (t *Tensor) CopyInt32Data(data []int32) error
```
CopyInt32Data copies int32 data to the tensor's Metal buffer

#### func (*Tensor) DType

```go
func (t *Tensor) DType() DataType
```
DType returns the data type

#### func (*Tensor) Device

```go
func (t *Tensor) Device() DeviceType
```
Device returns the device type

#### func (*Tensor) MetalBuffer

```go
func (t *Tensor) MetalBuffer() unsafe.Pointer
```
MetalBuffer returns the underlying Metal buffer pointer

#### func (*Tensor) RefCount

```go
func (t *Tensor) RefCount() int32
```
RefCount returns the current reference count (for debugging)

#### func (*Tensor) Release

```go
func (t *Tensor) Release()
```
Release decrements the reference count and returns buffer to pool when it
reaches 0

#### func (*Tensor) Retain

```go
func (t *Tensor) Retain() *Tensor
```
Retain increments the reference count and returns the same tensor

#### func (*Tensor) Shape

```go
func (t *Tensor) Shape() []int
```
Shape returns the tensor shape (defensive copy)

#### func (*Tensor) Size

```go
func (t *Tensor) Size() int
```
Size returns the total size in bytes

#### func (*Tensor) String

```go
func (t *Tensor) String() string
```
String returns a string representation for debugging

#### func (*Tensor) ToFloat32Slice

```go
func (t *Tensor) ToFloat32Slice() ([]float32, error)
```
