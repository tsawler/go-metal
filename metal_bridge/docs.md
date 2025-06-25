# metal_bridge
--
    import "."


## Usage

```go
const MetalKernelSource = `
#include <metal_stdlib>
using namespace metal;

// Simple element-wise addition kernel for float32 tensors
kernel void add_arrays_float32(device const float* inputA [[buffer(0)]],
                              device const float* inputB [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] + inputB[index];
}

// Simple element-wise addition kernel for int32 tensors
kernel void add_arrays_int32(device const int* inputA [[buffer(0)]],
                            device const int* inputB [[buffer(1)]],
                            device int* result [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] + inputB[index];
}

// Element-wise multiplication kernel for float32 tensors
kernel void mul_arrays_float32(device const float* inputA [[buffer(0)]],
                              device const float* inputB [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] * inputB[index];
}

// Element-wise multiplication kernel for int32 tensors
kernel void mul_arrays_int32(device const int* inputA [[buffer(0)]],
                            device const int* inputB [[buffer(1)]],
                            device int* result [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] * inputB[index];
}

// ReLU activation kernel for float32 tensors
kernel void relu_float32(device const float* input [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
    result[index] = max(0.0f, input[index]);
}

// ReLU activation kernel for int32 tensors
kernel void relu_int32(device const int* input [[buffer(0)]],
                      device int* result [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = max(0, input[index]);
}

// Sigmoid activation kernel for float32 tensors
kernel void sigmoid_float32(device const float* input [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
    result[index] = 1.0f / (1.0f + exp(-input[index]));
}

// Matrix multiplication kernel for float32 tensors
// Simple version for 2D matrices
kernel void matmul_float32(device const float* matrixA [[buffer(0)]],
                          device const float* matrixB [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          constant uint& M [[buffer(3)]],     // rows of A
                          constant uint& N [[buffer(4)]],     // cols of A / rows of B
                          constant uint& P [[buffer(5)]],     // cols of B
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= P) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; k++) {
        sum += matrixA[row * N + k] * matrixB[k * P + col];
    }
    result[row * P + col] = sum;
}
`
```
MSL source code for Metal compute kernels

```go
var (
	ResourceStorageModeShared  = C.MTLResourceStorageModeShared_Const
	ResourceStorageModeManaged = C.MTLResourceStorageModeManaged_Const
	ResourceStorageModePrivate = C.MTLResourceStorageModePrivate_Const
)
```
Resource storage mode constants

```go
var (
	MPSDataTypeFloat32 = C.MPSDataTypeFloat32_Const
	MPSDataTypeFloat16 = C.MPSDataTypeFloat16_Const
	MPSDataTypeInt32   = C.MPSDataTypeInt32_Const
)
```
MPSGraph data type constants

#### type Buffer

```go
type Buffer struct {
}
```

Wrapper struct for MTLBuffer

#### func (*Buffer) Contents

```go
func (b *Buffer) Contents() unsafe.Pointer
```

#### func (*Buffer) ContentsAsFloat32

```go
func (b *Buffer) ContentsAsFloat32() []float32
```

#### func (*Buffer) ContentsAsInt32

```go
func (b *Buffer) ContentsAsInt32() []int32
```

#### func (*Buffer) Length

```go
func (b *Buffer) Length() uintptr
```

#### type CommandBuffer

```go
type CommandBuffer struct {
}
```

Wrapper for MTLCommandBuffer

#### func (*CommandBuffer) AddCompletedHandler

```go
func (cb *CommandBuffer) AddCompletedHandler(handler func(status int))
```
AddCompletedHandler allows registering a Go callback for command buffer
completion. It passes a userData pointer which can be used to pass context to
the Go function.

#### func (*CommandBuffer) Commit

```go
func (cb *CommandBuffer) Commit()
```

#### func (*CommandBuffer) ComputeCommandEncoder

```go
func (cb *CommandBuffer) ComputeCommandEncoder() *ComputeCommandEncoder
```

#### func (*CommandBuffer) WaitUntilCompleted

```go
func (cb *CommandBuffer) WaitUntilCompleted()
```

#### type CommandQueue

```go
type CommandQueue struct {
}
```

Wrapper struct for MTLCommandQueue

#### func (*CommandQueue) CommandBuffer

```go
func (q *CommandQueue) CommandBuffer() *CommandBuffer
```

#### type ComputeCommandEncoder

```go
type ComputeCommandEncoder struct {
}
```

Wrapper for MTLComputeCommandEncoder

#### func (*ComputeCommandEncoder) DispatchThreads

```go
func (e *ComputeCommandEncoder) DispatchThreads(gridSize, threadgroupSize MTLSize)
```

#### func (*ComputeCommandEncoder) EndEncoding

```go
func (e *ComputeCommandEncoder) EndEncoding()
```

#### func (*ComputeCommandEncoder) SetBuffer

```go
func (e *ComputeCommandEncoder) SetBuffer(buffer *Buffer, offset, index uint)
```

#### func (*ComputeCommandEncoder) SetComputePipelineState

```go
func (e *ComputeCommandEncoder) SetComputePipelineState(pipelineState *ComputePipelineState)
```

#### type ComputeEngine

```go
type ComputeEngine struct {
}
```

ComputeEngine manages Metal compute operations

#### func  NewComputeEngine

```go
func NewComputeEngine() (*ComputeEngine, error)
```
NewComputeEngine creates a new Metal compute engine

#### func (*ComputeEngine) AddArraysFloat32

```go
func (e *ComputeEngine) AddArraysFloat32(inputA, inputB []float32) ([]float32, error)
```
AddArraysFloat32 performs element-wise addition of two float32 arrays on GPU

#### func (*ComputeEngine) AddArraysInt32

```go
func (e *ComputeEngine) AddArraysInt32(inputA, inputB []int32) ([]int32, error)
```
AddArraysInt32 performs element-wise addition of two int32 arrays on GPU

#### func (*ComputeEngine) LoadKernel

```go
func (e *ComputeEngine) LoadKernel(kernelName string) error
```
LoadKernel loads a specific kernel function and creates its pipeline state

#### func (*ComputeEngine) MatMulFloat32

```go
func (e *ComputeEngine) MatMulFloat32(matrixA, matrixB []float32, M, N, P uint) ([]float32, error)
```
MatMulFloat32 performs matrix multiplication on GPU

#### func (*ComputeEngine) ReLUFloat32

```go
func (e *ComputeEngine) ReLUFloat32(input []float32) ([]float32, error)
```
ReLUFloat32 applies ReLU activation to float32 array on GPU

#### type ComputePipelineState

```go
type ComputePipelineState struct {
}
```

Wrapper for MTLComputePipelineState

#### type Device

```go
type Device struct {
}
```

Wrapper struct for MTLDevice

#### func  CreateSystemDefaultDevice

```go
func CreateSystemDefaultDevice() *Device
```

#### func (*Device) CreateBufferWithBytes

```go
func (d *Device) CreateBufferWithBytes(data interface{}, options C.size_t) (*Buffer, error)
```

#### func (*Device) CreateBufferWithLength

```go
func (d *Device) CreateBufferWithLength(length uintptr, options C.size_t) (*Buffer, error)
```

#### func (*Device) CreateLibraryWithSource

```go
func (d *Device) CreateLibraryWithSource(source string) (*Library, error)
```

#### func (*Device) NewCommandQueue

```go
func (d *Device) NewCommandQueue() *CommandQueue
```

#### func (*Device) NewComputePipelineStateWithFunction

```go
func (d *Device) NewComputePipelineStateWithFunction(function *Function) (*ComputePipelineState, error)
```

#### type Function

```go
type Function struct {
}
```

Wrapper struct for MTLFunction

#### type Graph

```go
type Graph struct {
}
```

Wrapper struct for MPSGraph

#### func  NewGraph

```go
func NewGraph() *Graph
```

#### func (*Graph) Addition

```go
func (g *Graph) Addition(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor
```
MPSGraph operations

#### func (*Graph) AvgPool2D

```go
func (g *Graph) AvgPool2D(source *GraphTensor, kernelWidth, kernelHeight, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) *GraphTensor
```
AvgPool2D performs 2D average pooling operation

#### func (*Graph) Compile

```go
func (g *Graph) Compile(device *GraphDevice, inputTensors []*GraphTensor, targetTensors []*GraphTensor, compilationDescriptor *GraphCompilationDescriptor) *GraphExecutable
```

#### func (*Graph) ConstantTensor

```go
func (g *Graph) ConstantTensor(value float64, shape []int, dataType int) *GraphTensor
```

#### func (*Graph) Conv2D

```go
func (g *Graph) Conv2D(source, weights, bias *GraphTensor, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor
```
Conv2D performs 2D convolution operation

#### func (*Graph) Division

```go
func (g *Graph) Division(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor
```

#### func (*Graph) MatrixMultiplication

```go
func (g *Graph) MatrixMultiplication(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor
```

#### func (*Graph) MaxPool2D

```go
func (g *Graph) MaxPool2D(source *GraphTensor, kernelWidth, kernelHeight, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) *GraphTensor
```
MaxPool2D performs 2D max pooling operation

#### func (*Graph) Multiplication

```go
func (g *Graph) Multiplication(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor
```

#### func (*Graph) PlaceholderTensor

```go
func (g *Graph) PlaceholderTensor(shape []int, dataType int) *GraphTensor
```

#### func (*Graph) ReLU

```go
func (g *Graph) ReLU(tensor *GraphTensor) *GraphTensor
```

#### func (*Graph) Reshape

```go
func (g *Graph) Reshape(tensor *GraphTensor, shape []int) *GraphTensor
```

#### func (*Graph) Sigmoid

```go
func (g *Graph) Sigmoid(tensor *GraphTensor) *GraphTensor
```

#### func (*Graph) Softmax

```go
func (g *Graph) Softmax(tensor *GraphTensor, axis int) *GraphTensor
```

#### func (*Graph) Subtraction

```go
func (g *Graph) Subtraction(primaryTensor, secondaryTensor *GraphTensor) *GraphTensor
```

#### func (*Graph) Transpose

```go
func (g *Graph) Transpose(tensor *GraphTensor, dimension, dimensionTwo int) *GraphTensor
```

#### type GraphCompilationDescriptor

```go
type GraphCompilationDescriptor struct {
}
```


#### func  NewGraphCompilationDescriptor

```go
func NewGraphCompilationDescriptor() *GraphCompilationDescriptor
```

#### type GraphDevice

```go
type GraphDevice struct {
}
```

Wrapper struct for MPSGraphDevice

#### func  NewGraphDevice

```go
func NewGraphDevice(device *Device) *GraphDevice
```

#### type GraphExecutable

```go
type GraphExecutable struct {
}
```

Wrapper structs for MPSGraph execution

#### func (*GraphExecutable) Execute

```go
func (e *GraphExecutable) Execute(commandQueue *CommandQueue, inputTensors []*GraphTensor, inputBuffers []*Buffer, resultTensors []*GraphTensor, resultBuffers []*Buffer, executionDescriptor *GraphExecutionDescriptor)
```

#### type GraphExecutionDescriptor

```go
type GraphExecutionDescriptor struct {
}
```


#### func  NewGraphExecutionDescriptor

```go
func NewGraphExecutionDescriptor() *GraphExecutionDescriptor
```

#### type GraphTensor

```go
type GraphTensor struct {
}
```

Wrapper struct for MPSGraphTensor

#### type Library

```go
type Library struct {
}
```

Wrapper struct for MTLLibrary

#### func (*Library) GetFunction

```go
func (l *Library) GetFunction(functionName string) (*Function, error)
```

#### type MTLSize

```go
type MTLSize struct {
	Width, Height, Depth uint
}
```

Go equivalent of MTLSize
