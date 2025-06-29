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

// ===== FUSED OPERATION KERNELS =====
// These kernels combine multiple operations to reduce GPU kernel launch overhead

// Fused Linear layer kernel: MatMul + Bias addition in one GPU call
kernel void linear_forward_float32(device const float* input [[buffer(0)]],      // [batch_size, input_features]
                                  device const float* weight [[buffer(1)]],     // [output_features, input_features]
                                  device const float* bias [[buffer(2)]],       // [output_features]
                                  device float* output [[buffer(3)]],            // [batch_size, output_features]
                                  constant uint& batch_size [[buffer(4)]],
                                  constant uint& input_features [[buffer(5)]],
                                  constant uint& output_features [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    if (batch_idx >= batch_size || out_idx >= output_features) return;
    
    // Compute matrix multiplication for this output element
    float sum = 0.0f;
    for (uint in_idx = 0; in_idx < input_features; in_idx++) {
        sum += input[batch_idx * input_features + in_idx] * weight[in_idx * output_features + out_idx];
    }
    
    // Add bias
    sum += bias[out_idx];
    
    // Store result
    output[batch_idx * output_features + out_idx] = sum;
}

// Fused Linear + ReLU kernel: MatMul + Bias + ReLU activation in one GPU call
kernel void linear_relu_float32(device const float* input [[buffer(0)]],
                               device const float* weight [[buffer(1)]],
                               device const float* bias [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& batch_size [[buffer(4)]],
                               constant uint& input_features [[buffer(5)]],
                               constant uint& output_features [[buffer(6)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    if (batch_idx >= batch_size || out_idx >= output_features) return;
    
    // Compute matrix multiplication
    float sum = 0.0f;
    for (uint in_idx = 0; in_idx < input_features; in_idx++) {
        sum += input[batch_idx * input_features + in_idx] * weight[in_idx * output_features + out_idx];
    }
    
    // Add bias and apply ReLU activation
    sum = max(0.0f, sum + bias[out_idx]);
    
    // Store result
    output[batch_idx * output_features + out_idx] = sum;
}

// Fused Linear + Sigmoid kernel: MatMul + Bias + Sigmoid activation in one GPU call
kernel void linear_sigmoid_float32(device const float* input [[buffer(0)]],
                                  device const float* weight [[buffer(1)]],
                                  device const float* bias [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& batch_size [[buffer(4)]],
                                  constant uint& input_features [[buffer(5)]],
                                  constant uint& output_features [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    if (batch_idx >= batch_size || out_idx >= output_features) return;
    
    // Compute matrix multiplication
    float sum = 0.0f;
    for (uint in_idx = 0; in_idx < input_features; in_idx++) {
        sum += input[batch_idx * input_features + in_idx] * weight[in_idx * output_features + out_idx];
    }
    
    // Add bias and apply Sigmoid activation
    sum = 1.0f / (1.0f + exp(-(sum + bias[out_idx])));
    
    // Store result
    output[batch_idx * output_features + out_idx] = sum;
}

// Fused Batch MatMul kernel for processing multiple matrix multiplications in one call
kernel void batch_matmul_float32(device const float* batchA [[buffer(0)]],
                                device const float* batchB [[buffer(1)]],
                                device float* batchResult [[buffer(2)]],
                                constant uint& batch_size [[buffer(3)]],
                                constant uint& M [[buffer(4)]],
                                constant uint& N [[buffer(5)]],
                                constant uint& P [[buffer(6)]],
                                uint3 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    if (batch_idx >= batch_size || row >= M || col >= P) return;
    
    uint batch_offset_a = batch_idx * M * N;
    uint batch_offset_b = batch_idx * N * P;
    uint batch_offset_result = batch_idx * M * P;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; k++) {
        sum += batchA[batch_offset_a + row * N + k] * batchB[batch_offset_b + k * P + col];
    }
    
    batchResult[batch_offset_result + row * P + col] = sum;
}

// Fused gradient accumulation kernel for optimizer updates
kernel void adam_update_float32(device float* params [[buffer(0)]],
                               device const float* gradients [[buffer(1)]],
                               device float* m [[buffer(2)]],           // First moment estimate
                               device float* v [[buffer(3)]],           // Second moment estimate
                               constant float& lr [[buffer(4)]],        // Learning rate
                               constant float& beta1 [[buffer(5)]],     // Beta1
                               constant float& beta2 [[buffer(6)]],     // Beta2
                               constant float& eps [[buffer(7)]],       // Epsilon
                               constant uint& t [[buffer(8)]],          // Time step
                               uint index [[thread_position_in_grid]]) {
    // Update biased first moment estimate
    m[index] = beta1 * m[index] + (1.0f - beta1) * gradients[index];
    
    // Update biased second moment estimate
    v[index] = beta2 * v[index] + (1.0f - beta2) * gradients[index] * gradients[index];
    
    // Compute bias correction
    float m_hat = m[index] / (1.0f - pow(beta1, float(t)));
    float v_hat = v[index] / (1.0f - pow(beta2, float(t)));
    
    // Update parameters
    params[index] -= lr * m_hat / (sqrt(v_hat) + eps);
}

// Fused SGD with momentum update kernel
kernel void sgd_momentum_update_float32(device float* params [[buffer(0)]],
                                       device const float* gradients [[buffer(1)]],
                                       device float* velocity [[buffer(2)]],
                                       constant float& lr [[buffer(3)]],
                                       constant float& momentum [[buffer(4)]],
                                       constant float& weight_decay [[buffer(5)]],
                                       uint index [[thread_position_in_grid]]) {
    // Apply weight decay
    float grad = gradients[index] + weight_decay * params[index];
    
    // Update velocity with momentum
    velocity[index] = momentum * velocity[index] + grad;
    
    // Update parameters
    params[index] -= lr * velocity[index];
}

// Fused layer norm forward kernel
kernel void layer_norm_float32(device const float* input [[buffer(0)]],
                              device const float* gamma [[buffer(1)]],
                              device const float* beta [[buffer(2)]],
                              device float* output [[buffer(3)]],
                              device float* mean_out [[buffer(4)]],
                              device float* var_out [[buffer(5)]],
                              constant uint& batch_size [[buffer(6)]],
                              constant uint& features [[buffer(7)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    
    if (batch_idx >= batch_size) return;
    
    uint offset = batch_idx * features;
    
    // Calculate mean
    float mean = 0.0f;
    for (uint i = 0; i < features; i++) {
        mean += input[offset + i];
    }
    mean /= float(features);
    mean_out[batch_idx] = mean;
    
    // Calculate variance
    float var = 0.0f;
    for (uint i = 0; i < features; i++) {
        float diff = input[offset + i] - mean;
        var += diff * diff;
    }
    var /= float(features);
    var_out[batch_idx] = var;
    
    // Normalize and scale
    float std_inv = rsqrt(var + 1e-5f);
    
    uint feature_idx = gid.x;
    if (feature_idx < features) {
        float normalized = (input[offset + feature_idx] - mean) * std_inv;
        output[offset + feature_idx] = gamma[feature_idx] * normalized + beta[feature_idx];
    }
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

#### func (*Buffer) Length

```go
func (b *Buffer) Length() int
```

#### type BufferAllocator

```go
type BufferAllocator struct {
}
```

BufferAllocator is a stub for compatibility

#### func  GetGlobalAllocator

```go
func GetGlobalAllocator() *BufferAllocator
```
GetGlobalAllocator returns a stub allocator for compatibility

#### func (*BufferAllocator) GetMemoryStats

```go
func (ba *BufferAllocator) GetMemoryStats() MemoryStats
```
GetMemoryStats returns zero stats for compatibility

#### type CommandBuffer

```go
type CommandBuffer struct {
}
```

Wrapper struct for MTLCommandBuffer

#### func (*CommandBuffer) AddCompletedHandler

```go
func (cb *CommandBuffer) AddCompletedHandler(handler func(status int))
```

#### func (*CommandBuffer) Commit

```go
func (cb *CommandBuffer) Commit()
```

#### func (*CommandBuffer) NewComputeCommandEncoder

```go
func (cb *CommandBuffer) NewComputeCommandEncoder() *ComputeCommandEncoder
```

#### func (*CommandBuffer) WaitUntilCompleted

```go
func (cb *CommandBuffer) WaitUntilCompleted()
```

#### type CommandBufferManager

```go
type CommandBufferManager struct {
}
```

CommandBufferManager manages command buffer queuing and dependency tracking

#### func  NewCommandBufferManager

```go
func NewCommandBufferManager(device *Device, commandQueue *CommandQueue) *CommandBufferManager
```
NewCommandBufferManager creates a new command buffer manager

#### func (*CommandBufferManager) GenerateOperationID

```go
func (m *CommandBufferManager) GenerateOperationID() OperationID
```
GenerateOperationID creates a unique operation ID

#### func (*CommandBufferManager) GetStats

```go
func (m *CommandBufferManager) GetStats() (queued, executed int64, pending int)
```
GetStats returns operation statistics for monitoring

#### func (*CommandBufferManager) QueueOperation

```go
func (m *CommandBufferManager) QueueOperation(op *PendingOperation) error
```
QueueOperation adds an operation to the queue with dependency tracking

#### func (*CommandBufferManager) Shutdown

```go
func (m *CommandBufferManager) Shutdown()
```
Shutdown gracefully shuts down the command buffer manager

#### func (*CommandBufferManager) WaitForOperation

```go
func (m *CommandBufferManager) WaitForOperation(opID OperationID) error
```
WaitForOperation blocks until a specific operation completes

#### type CommandQueue

```go
type CommandQueue struct {
}
```

Wrapper struct for MTLCommandQueue

#### func (*CommandQueue) NewCommandBuffer

```go
func (cq *CommandQueue) NewCommandBuffer() *CommandBuffer
```

#### type ComputeCommandEncoder

```go
type ComputeCommandEncoder struct {
}
```

Wrapper struct for MTLComputeCommandEncoder

#### func (*ComputeCommandEncoder) DispatchThreads

```go
func (cce *ComputeCommandEncoder) DispatchThreads(gridSize, threadgroupSize MTLSize)
```

#### func (*ComputeCommandEncoder) EndEncoding

```go
func (cce *ComputeCommandEncoder) EndEncoding()
```

#### func (*ComputeCommandEncoder) SetBuffer

```go
func (cce *ComputeCommandEncoder) SetBuffer(buffer *Buffer, offset, index int)
```

#### func (*ComputeCommandEncoder) SetComputePipelineState

```go
func (cce *ComputeCommandEncoder) SetComputePipelineState(pipelineState *ComputePipelineState)
```

#### type ComputePipelineState

```go
type ComputePipelineState struct {
}
```

Wrapper struct for MTLComputePipelineState

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

#### func (*Device) NewBufferWithBytes

```go
func (d *Device) NewBufferWithBytes(data []byte, resourceOptions int) *Buffer
```

#### func (*Device) NewBufferWithLength

```go
func (d *Device) NewBufferWithLength(length int, resourceOptions int) *Buffer
```

#### func (*Device) NewCommandQueue

```go
func (d *Device) NewCommandQueue() *CommandQueue
```

#### func (*Device) NewComputePipelineStateWithFunction

```go
func (d *Device) NewComputePipelineStateWithFunction(function *Function) *ComputePipelineState
```

#### func (*Device) NewLibraryWithSource

```go
func (d *Device) NewLibraryWithSource(source string) *Library
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

MPSGraph wrapper types

#### func  NewGraph

```go
func NewGraph() *Graph
```

#### func (*Graph) Addition

```go
func (g *Graph) Addition(a, b *GraphTensor) *GraphTensor
```
Addition operation

#### func (*Graph) AvgPool2D

```go
func (g *Graph) AvgPool2D(source *GraphTensor, kernelWidth, kernelHeight, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) *GraphTensor
```
AvgPool2D performs 2D average pooling operation

#### func (*Graph) Compile

```go
func (g *Graph) Compile(device *GraphDevice, inputTensors, targetTensors []*GraphTensor, compilationDescriptor *GraphCompilationDescriptor) *GraphExecutable
```

#### func (*Graph) ConstantTensor

```go
func (g *Graph) ConstantTensor(value float64, shape []int, dataType int) *GraphTensor
```
Create a constant tensor

#### func (*Graph) Conv2D

```go
func (g *Graph) Conv2D(source, weights, bias *GraphTensor, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor
```
Conv2D performs 2D convolution operation

#### func (*Graph) Convolution2DDataGradient

```go
func (g *Graph) Convolution2DDataGradient(incomingGradient, weights *GraphTensor, inputShape []int, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor
```
Convolution2DDataGradient computes the gradient with respect to input data

#### func (*Graph) Convolution2DWeightsGradient

```go
func (g *Graph) Convolution2DWeightsGradient(incomingGradient, source *GraphTensor, weightsShape []int, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor
```
Convolution2DWeightsGradient computes the gradient with respect to weights

#### func (*Graph) ConvolutionTranspose2D

```go
func (g *Graph) ConvolutionTranspose2D(source, weights *GraphTensor, outputShape []int, strideX, strideY, dilationX, dilationY, paddingLeft, paddingRight, paddingTop, paddingBottom, groups int) *GraphTensor
```
ConvolutionTranspose2D performs 2D transposed convolution operation

#### func (*Graph) Division

```go
func (g *Graph) Division(a, b *GraphTensor) *GraphTensor
```
Division operation

#### func (*Graph) MatrixMultiplication

```go
func (g *Graph) MatrixMultiplication(a, b *GraphTensor) *GraphTensor
```
Matrix multiplication operation

#### func (*Graph) MaxPool2D

```go
func (g *Graph) MaxPool2D(source *GraphTensor, kernelWidth, kernelHeight, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) *GraphTensor
```
MaxPool2D performs 2D max pooling operation

#### func (*Graph) Multiplication

```go
func (g *Graph) Multiplication(a, b *GraphTensor) *GraphTensor
```
Multiplication operation

#### func (*Graph) PlaceholderTensor

```go
func (g *Graph) PlaceholderTensor(shape []int, dataType int) *GraphTensor
```
Create a placeholder tensor

#### func (*Graph) ReLU

```go
func (g *Graph) ReLU(tensor *GraphTensor) *GraphTensor
```
ReLU activation function

#### func (*Graph) ReductionSum

```go
func (g *Graph) ReductionSum(tensor *GraphTensor, axis int, keepdim bool) *GraphTensor
```
ReductionSum performs tensor reduction sum along specified axis

#### func (*Graph) Reshape

```go
func (g *Graph) Reshape(tensor *GraphTensor, shape []int) *GraphTensor
```
Reshape operation

#### func (*Graph) Sigmoid

```go
func (g *Graph) Sigmoid(tensor *GraphTensor) *GraphTensor
```
Sigmoid activation function

#### func (*Graph) Softmax

```go
func (g *Graph) Softmax(tensor *GraphTensor, axis int) *GraphTensor
```
Softmax activation function

#### func (*Graph) Subtraction

```go
func (g *Graph) Subtraction(a, b *GraphTensor) *GraphTensor
```
Subtraction operation

#### func (*Graph) Transpose

```go
func (g *Graph) Transpose(tensor *GraphTensor, dimension1, dimension2 int) *GraphTensor
```
Transpose operation

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


#### func (*GraphExecutable) Execute

```go
func (e *GraphExecutable) Execute(commandQueue *CommandQueue, inputTensors []*GraphTensor, inputBuffers []*Buffer, resultTensors []*GraphTensor, resultBuffers []*Buffer, executionDescriptor *GraphExecutionDescriptor)
```

#### type GraphExecutionDescriptor

```go
type GraphExecutionDescriptor struct {
}
```

Wrapper structs for MPSGraph compilation and execution

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

#### func (*GraphTensor) DataType

```go
func (gt *GraphTensor) DataType() int
```

#### func (*GraphTensor) Shape

```go
func (gt *GraphTensor) Shape() []int
```

#### type Library

```go
type Library struct {
}
```

Wrapper struct for MTLLibrary

#### func (*Library) NewFunctionWithName

```go
func (l *Library) NewFunctionWithName(name string) *Function
```

#### type MTLSize

```go
type MTLSize struct {
	Width, Height, Depth uint
}
```

Go equivalent of MTLSize

#### type MemoryStats

```go
type MemoryStats struct {
	NumAllocations   int
	NumDeallocations int
	NumPoolHits      int
	NumPoolMisses    int
	TotalMemory      int
	UsedMemory       int
	TotalFree        int
}
```

MemoryStats represents memory allocation statistics

#### type OperationID

```go
type OperationID uint64
```

OperationID represents a unique identifier for GPU operations

#### type PendingOperation

```go
type PendingOperation struct {
	ID           OperationID
	Dependencies []OperationID
	Execute      func() error
	Cleanup      func() error
	OnComplete   func(error)

	// Resource tracking for memory safety
	InputBuffers  []*Buffer
	OutputBuffers []*Buffer
	TempBuffers   []*Buffer
}
```

PendingOperation represents an operation waiting to be executed
