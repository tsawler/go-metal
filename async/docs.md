# async
--
    import "."


## Usage

#### type AsyncDataLoader

```go
type AsyncDataLoader struct {
}
```

AsyncDataLoader manages background data loading with GPU transfer pipeline

#### func  NewAsyncDataLoader

```go
func NewAsyncDataLoader(dataSource DataSource, config AsyncDataLoaderConfig) (*AsyncDataLoader, error)
```
NewAsyncDataLoader creates a new asynchronous data loader

#### func (*AsyncDataLoader) GetBatch

```go
func (adl *AsyncDataLoader) GetBatch() (*GPUBatch, error)
```
GetBatch returns the next ready batch (blocks until available)

#### func (*AsyncDataLoader) Start

```go
func (adl *AsyncDataLoader) Start() error
```
Start begins the async data loading pipeline

#### func (*AsyncDataLoader) Stats

```go
func (adl *AsyncDataLoader) Stats() AsyncDataLoaderStats
```
Stats returns statistics about the data loader

#### func (*AsyncDataLoader) Stop

```go
func (adl *AsyncDataLoader) Stop() error
```
Stop stops the async data loading pipeline

#### func (*AsyncDataLoader) TryGetBatch

```go
func (adl *AsyncDataLoader) TryGetBatch() (*GPUBatch, error)
```
TryGetBatch returns the next ready batch (non-blocking)

#### type AsyncDataLoaderConfig

```go
type AsyncDataLoaderConfig struct {
	BatchSize     int // Size of each batch
	PrefetchDepth int // Number of batches to prefetch (default: 3)
	Workers       int // Number of background workers (default: 2)
	MemoryManager *memory.MemoryManager
}
```

AsyncDataLoaderConfig holds configuration for the data loader

#### type AsyncDataLoaderStats

```go
type AsyncDataLoaderStats struct {
	IsRunning       bool
	BatchesProduced uint64
	QueuedBatches   int
	QueueCapacity   int
	Workers         int
	Generation      uint64
}
```

AsyncDataLoaderStats provides statistics about the data loader

#### type BatchOperation

```go
type BatchOperation struct {
	Type       string // "training_step", "data_transfer", etc.
	Data       interface{}
	Completion func(error)
}
```

Batch operation support for multiple operations in single command buffer

#### type CommandBuffer

```go
type CommandBuffer struct {
}
```

CommandBuffer represents a Metal command buffer wrapper

#### type CommandBufferPool

```go
type CommandBufferPool struct {
}
```

CommandBufferPool manages a pool of Metal command buffers for reuse

#### func  NewCommandBufferPool

```go
func NewCommandBufferPool(commandQueue unsafe.Pointer, maxBuffers int) (*CommandBufferPool, error)
```
NewCommandBufferPool creates a new command buffer pool

#### func (*CommandBufferPool) Cleanup

```go
func (cbp *CommandBufferPool) Cleanup()
```
Cleanup releases all command buffers

#### func (*CommandBufferPool) ExecuteAsync

```go
func (cbp *CommandBufferPool) ExecuteAsync(buffer *CommandBuffer, completion func(error)) error
```
ExecuteAsync submits a command buffer for async execution

#### func (*CommandBufferPool) ExecuteBatch

```go
func (cbp *CommandBufferPool) ExecuteBatch(operations []BatchOperation) error
```
ExecuteBatch executes multiple operations in a single command buffer for
efficiency

#### func (*CommandBufferPool) GetBuffer

```go
func (cbp *CommandBufferPool) GetBuffer() (*CommandBuffer, error)
```
GetBuffer gets an available command buffer (creates new one if needed and under
limit)

#### func (*CommandBufferPool) ReturnBuffer

```go
func (cbp *CommandBufferPool) ReturnBuffer(buffer *CommandBuffer)
```
ReturnBuffer returns a command buffer to the pool after completion

#### func (*CommandBufferPool) Stats

```go
func (cbp *CommandBufferPool) Stats() CommandPoolStats
```
Stats returns statistics about the command buffer pool

#### type CommandPoolStats

```go
type CommandPoolStats struct {
	TotalBuffers     int
	AvailableBuffers int
	InUseBuffers     int
	MaxBuffers       int
}
```

CommandPoolStats provides statistics about the command buffer pool

#### type DataSource

```go
type DataSource interface {
	// GetBatch returns the next batch of data
	// inputData: raw input data, inputShape: tensor shape
	// labelData: raw label data, labelShape: tensor shape
	GetBatch(batchSize int) (inputData []float32, inputShape []int, labelData []float32, labelShape []int, err error)

	// Size returns the total number of samples available
	Size() int

	// Reset resets the data source to the beginning
	Reset() error
}
```

DataSource represents a source of training data

#### type GPUBatch

```go
type GPUBatch struct {
	InputTensor   *memory.Tensor   // GPU-resident input data
	LabelTensor   *memory.Tensor   // GPU-resident label data
	WeightTensors []*memory.Tensor // GPU-resident weight tensors
	BatchID       uint64           // Unique identifier for this batch
	Generation    uint64           // For cleanup tracking
}
```

GPUBatch represents a batch of data ready for GPU training

#### func (*GPUBatch) Release

```go
func (gb *GPUBatch) Release()
```
Release releases all tensors in the batch

#### type StagingBuffer

```go
type StagingBuffer struct {
}
```

StagingBuffer represents a CPU-accessible buffer for GPU transfers

#### type StagingBufferPool

```go
type StagingBufferPool struct {
}
```

StagingBufferPool manages a pool of staging buffers for async CPU→GPU transfers

#### func  NewStagingBufferPool

```go
func NewStagingBufferPool(memoryManager *memory.MemoryManager, maxBuffers int) (*StagingBufferPool, error)
```
NewStagingBufferPool creates a new staging buffer pool

#### func (*StagingBufferPool) Cleanup

```go
func (sbp *StagingBufferPool) Cleanup()
```
Cleanup releases all staging buffers and command queue

#### func (*StagingBufferPool) GetBuffer

```go
func (sbp *StagingBufferPool) GetBuffer() (*StagingBuffer, error)
```
GetBuffer gets an available staging buffer (creates new one if needed and under
limit)

#### func (*StagingBufferPool) ReturnBuffer

```go
func (sbp *StagingBufferPool) ReturnBuffer(buffer *StagingBuffer)
```
ReturnBuffer returns a staging buffer to the pool

#### func (*StagingBufferPool) Stats

```go
func (sbp *StagingBufferPool) Stats() StagingPoolStats
```
Stats returns statistics about the staging buffer pool

#### func (*StagingBufferPool) TransferToGPU

```go
func (sbp *StagingBufferPool) TransferToGPU(data interface{}, gpuTensor *memory.Tensor) error
```
TransferToGPU transfers data from CPU slice to GPU tensor using staging buffers

#### func (*StagingBufferPool) TransferToGPUSync

```go
func (sbp *StagingBufferPool) TransferToGPUSync(data interface{}, gpuTensor *memory.Tensor) error
```
TransferToGPUSync transfers data from CPU slice to GPU tensor and waits for
completion

#### func (*StagingBufferPool) WaitForTransferCompletion

```go
func (sbp *StagingBufferPool) WaitForTransferCompletion() error
```
WaitForTransferCompletion waits for all pending transfers to complete

#### type StagingPoolStats

```go
type StagingPoolStats struct {
	TotalBuffers     int
	AvailableBuffers int
	InUseBuffers     int
	BufferSize       int
	MaxBuffers       int
}
```

StagingPoolStats provides statistics about the staging buffer pool
