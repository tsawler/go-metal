# tensor
--
    import "."


## Usage

#### func  AddGPUAsync

```go
func AddGPUAsync(t1, t2 *Tensor, completion func(*Tensor, error)) error
```
AddGPUAsync performs tensor addition on GPU asynchronously

#### func  AreBroadcastable

```go
func AreBroadcastable(shape1, shape2 []int) bool
```
AreBroadcastable checks if two shapes can be broadcast together

#### func  BroadcastShapes

```go
func BroadcastShapes(shape1, shape2 []int) ([]int, error)
```
BroadcastShapes determines if two shapes are broadcastable and returns the
resulting shape Follows NumPy/PyTorch broadcasting rules: 1. Start from trailing
dimensions and work backwards 2. Dimensions are compatible if they are equal, or
one of them is 1, or one is missing 3. Result shape is the maximum of each
dimension

#### func  BroadcastTensorsForOperation

```go
func BroadcastTensorsForOperation(a, b *Tensor) (*Tensor, *Tensor, error)
```
BroadcastTensorsForOperation broadcasts two tensors to a common shape for
element-wise operations

#### func  GPUInfo

```go
func GPUInfo() (string, error)
```
GPUInfo returns information about the GPU device

#### func  IsFusedOperation

```go
func IsFusedOperation(opType string) bool
```
IsFusedOperation checks if an operation type is a fused operation

#### func  IsGPUAvailable

```go
func IsGPUAvailable() bool
```
IsGPUAvailable checks if Metal GPU compute is available

#### func  MatMulGPUAsync

```go
func MatMulGPUAsync(t1, t2 *Tensor, completion func(*Tensor, error)) error
```
MatMulGPU performs matrix multiplication on GPU

#### func  ZeroGrad

```go
func ZeroGrad(tensors []*Tensor)
```

#### type AddOp

```go
type AddOp struct {
}
```

AddOp implements the Operation interface for tensor addition

#### func (*AddOp) Backward

```go
func (op *AddOp) Backward(gradOut *Tensor) []*Tensor
```

#### func (*AddOp) Forward

```go
func (op *AddOp) Forward(inputs ...*Tensor) *Tensor
```

#### type DType

```go
type DType int
```


```go
const (
	Float32 DType = iota
	Float16
	Int32
)
```

#### func (DType) String

```go
func (d DType) String() string
```

#### type DeviceType

```go
type DeviceType int
```


```go
const (
	CPU           DeviceType = iota
	GPU                      // Temporary GPU tensors - copy results back to CPU
	PersistentGPU            // Persistent GPU tensors - keep results on GPU across operations
)
```

#### func (DeviceType) String

```go
func (d DeviceType) String() string
```

#### type FusedOperationDetector

```go
type FusedOperationDetector struct {
}
```

FusedOperationDetector analyzes a sequence of operations to detect fusion
opportunities

#### func  NewFusedOperationDetector

```go
func NewFusedOperationDetector() *FusedOperationDetector
```
NewFusedOperationDetector creates a new operation fusion detector

#### func (*FusedOperationDetector) AddOperation

```go
func (fod *FusedOperationDetector) AddOperation(opType string, inputs []*Tensor, params map[string]interface{})
```
AddOperation adds an operation to the sequence for analysis

#### func (*FusedOperationDetector) DetectFusions

```go
func (fod *FusedOperationDetector) DetectFusions() ([]OperationDesc, error)
```
DetectFusions analyzes the operation sequence and returns optimized fused
operations

#### type GPUComputationGraph

```go
type GPUComputationGraph struct {
}
```

GPUComputationGraph manages a graph of GPU operations with dependency tracking

#### func  NewGPUComputationGraph

```go
func NewGPUComputationGraph() (*GPUComputationGraph, error)
```
NewGPUComputationGraph creates a new GPU computation graph

#### func (*GPUComputationGraph) AddOperation

```go
func (g *GPUComputationGraph) AddOperation(opType string, inputs []*Tensor, dependencies []metal_bridge.OperationID, params map[string]interface{}) (metal_bridge.OperationID, error)
```
AddOperation adds an operation to the computation graph

#### func (*GPUComputationGraph) ExecuteSequence

```go
func (g *GPUComputationGraph) ExecuteSequence(operations []OperationDesc) (*Tensor, error)
```
ExecuteSequence executes a sequence of operations and returns the final result

#### func (*GPUComputationGraph) GetStats

```go
func (g *GPUComputationGraph) GetStats() (queued, executed int64, pending int)
```
GetStats returns statistics about the computation graph

#### func (*GPUComputationGraph) Shutdown

```go
func (g *GPUComputationGraph) Shutdown()
```
Shutdown gracefully shuts down the computation graph

#### func (*GPUComputationGraph) WaitForOperation

```go
func (g *GPUComputationGraph) WaitForOperation(opID metal_bridge.OperationID) (*Tensor, error)
```
WaitForOperation waits for a specific operation to complete

#### type GPUTrainingContext

```go
type GPUTrainingContext struct {
}
```

GPUTrainingContext manages GPU operations for neural network training

#### func  GetGlobalGPUTrainingContext

```go
func GetGlobalGPUTrainingContext() (*GPUTrainingContext, error)
```
GetGlobalGPUTrainingContext returns the global GPU training context

#### func  NewGPUTrainingContext

```go
func NewGPUTrainingContext() (*GPUTrainingContext, error)
```
NewGPUTrainingContext creates a new GPU training context

#### func (*GPUTrainingContext) BatchOperationsAsync

```go
func (ctx *GPUTrainingContext) BatchOperationsAsync(ops []OperationDesc) ([]*Tensor, error)
```
BatchOperationsAsync batches multiple operations for efficient GPU execution
This version includes automatic operation fusion optimization

#### func (*GPUTrainingContext) ConvolutionForwardAsync

```go
func (ctx *GPUTrainingContext) ConvolutionForwardAsync(input, weights, bias *Tensor, stride, padding int) (*Tensor, error)
```
ConvolutionForwardAsync performs a convolution forward pass with dependency
tracking

#### func (*GPUTrainingContext) FlushBatch

```go
func (ctx *GPUTrainingContext) FlushBatch() error
```
FlushBatch executes any remaining operations in the batch

#### func (*GPUTrainingContext) GetGPUStats

```go
func (ctx *GPUTrainingContext) GetGPUStats() (queued, executed int64, pending int, batchEfficiency float64)
```
GetGPUStats returns GPU operation statistics

#### func (*GPUTrainingContext) LinearLayerForwardAsync

```go
func (ctx *GPUTrainingContext) LinearLayerForwardAsync(input, weight, bias *Tensor, activation string) (*Tensor, error)
```
LinearLayerForwardAsync performs a linear layer forward pass asynchronously This
combines MatMul + Bias addition + optional activation in a dependency chain

#### func (*GPUTrainingContext) OptimizedMatMulChain

```go
func (ctx *GPUTrainingContext) OptimizedMatMulChain(tensors []*Tensor) (*Tensor, error)
```
OptimizedMatMulChain performs a chain of matrix multiplications with minimal
memory transfers

#### func (*GPUTrainingContext) QueueOperation

```go
func (ctx *GPUTrainingContext) QueueOperation(op OperationDesc)
```
QueueOperation adds an operation to the batch queue

#### func (*GPUTrainingContext) SetBatchSize

```go
func (ctx *GPUTrainingContext) SetBatchSize(size int)
```
SetBatchSize sets the operation batch size for GPU operations

#### func (*GPUTrainingContext) Shutdown

```go
func (ctx *GPUTrainingContext) Shutdown()
```
Shutdown gracefully shuts down the training context

#### func (*GPUTrainingContext) TrainingStepAsync

```go
func (ctx *GPUTrainingContext) TrainingStepAsync(forward, backward []OperationDesc) error
```
TrainingStepAsync performs a complete training step with batched operations

#### type GraphOperation

```go
type GraphOperation struct {
	ID           metal_bridge.OperationID
	Type         string
	InputTensors []*Tensor
	OutputTensor *Tensor
	Dependencies []metal_bridge.OperationID

	// Operation-specific data
	Params map[string]interface{}
}
```

GraphOperation represents a single operation in the computation graph

#### type MPSGraphEngine

```go
type MPSGraphEngine struct {
}
```

MPSGraphEngine provides high-level ML operations using MPSGraph

#### func  GetMPSGraphEngine

```go
func GetMPSGraphEngine() (*MPSGraphEngine, error)
```
GetMPSGraphEngine returns the singleton MPSGraph engine

#### type MatMulOp

```go
type MatMulOp struct {
}
```

MatMulOp implements the Operation interface for matrix multiplication

#### func (*MatMulOp) Backward

```go
func (op *MatMulOp) Backward(gradOut *Tensor) []*Tensor
```

#### func (*MatMulOp) Forward

```go
func (op *MatMulOp) Forward(inputs ...*Tensor) *Tensor
```

#### type MulOp

```go
type MulOp struct {
}
```

MulOp implements the Operation interface for tensor multiplication

#### func (*MulOp) Backward

```go
func (op *MulOp) Backward(gradOut *Tensor) []*Tensor
```

#### func (*MulOp) Forward

```go
func (op *MulOp) Forward(inputs ...*Tensor) *Tensor
```

#### type Operation

```go
type Operation interface {
	Forward(...*Tensor) *Tensor
	Backward(gradOut *Tensor) []*Tensor
}
```


#### type OperationDesc

```go
type OperationDesc struct {
	Type   string
	Inputs []*Tensor
	Params map[string]interface{}
}
```

OperationDesc describes an operation to be added to the graph

#### func  NewAddOp

```go
func NewAddOp(a, b *Tensor) OperationDesc
```

#### func  NewBatchMatMulOp

```go
func NewBatchMatMulOp(a, b *Tensor) OperationDesc
```

#### func  NewLinearForwardOp

```go
func NewLinearForwardOp(input, weight, bias *Tensor) OperationDesc
```
Helper functions for creating fused operation descriptors

#### func  NewLinearReLUOp

```go
func NewLinearReLUOp(input, weight, bias *Tensor) OperationDesc
```

#### func  NewLinearSigmoidOp

```go
func NewLinearSigmoidOp(input, weight, bias *Tensor) OperationDesc
```

#### func  NewMatMulOp

```go
func NewMatMulOp(a, b *Tensor) OperationDesc
```
Helper function for creating operation descriptors

#### func  NewReLUOp

```go
func NewReLUOp(input *Tensor) OperationDesc
```

#### func  OptimizeOperationSequence

```go
func OptimizeOperationSequence(operations []OperationDesc) ([]OperationDesc, error)
```
OptimizeOperationSequence takes a sequence of operations and returns an
optimized version with fusions

#### type ReLUOp

```go
type ReLUOp struct {
}
```

ReLUOp implements the Operation interface for ReLU activation

#### func (*ReLUOp) Backward

```go
func (op *ReLUOp) Backward(gradOut *Tensor) []*Tensor
```

#### func (*ReLUOp) Forward

```go
func (op *ReLUOp) Forward(inputs ...*Tensor) *Tensor
```

#### type SigmoidOp

```go
type SigmoidOp struct {
}
```

SigmoidOp implements the Operation interface for Sigmoid activation

#### func (*SigmoidOp) Backward

```go
func (op *SigmoidOp) Backward(gradOut *Tensor) []*Tensor
```

#### func (*SigmoidOp) Forward

```go
func (op *SigmoidOp) Forward(inputs ...*Tensor) *Tensor
```

#### type SubOp

```go
type SubOp struct {
}
```

SubOp implements the Operation interface for tensor subtraction

#### func (*SubOp) Backward

```go
func (op *SubOp) Backward(gradOut *Tensor) []*Tensor
```

#### func (*SubOp) Forward

```go
func (op *SubOp) Forward(inputs ...*Tensor) *Tensor
```

#### type Tensor

```go
type Tensor struct {
	Shape    []int
	Strides  []int
	DType    DType
	Device   DeviceType
	Data     interface{}
	NumElems int
}
```


#### func  Add

```go
func Add(t1, t2 *Tensor) (*Tensor, error)
```

#### func  AddAutograd

```go
func AddAutograd(a, b *Tensor) *Tensor
```
AddAutograd performs addition with automatic differentiation

#### func  AddGPU

```go
func AddGPU(t1, t2 *Tensor) (*Tensor, error)
```
AddGPU performs tensor addition on GPU

#### func  AddMPS

```go
func AddMPS(a, b *Tensor) (*Tensor, error)
```
AddMPS performs tensor addition using MPSGraph

#### func  AvgPool2DMPS

```go
func AvgPool2DMPS(input *Tensor, kernelSize, stride, padding int) (*Tensor, error)
```
AvgPool2DMPS performs 2D average pooling using MPSGraph

#### func  BatchMatMul

```go
func BatchMatMul(tensorA, tensorB *Tensor) (*Tensor, error)
```
BatchMatMul performs batched matrix multiplication Input tensors should be 3D:
[batch_size, M, N] x [batch_size, N, P] -> [batch_size, M, P]

#### func  BroadcastTensor

```go
func BroadcastTensor(t *Tensor, targetShape []int) (*Tensor, error)
```
BroadcastTensor expands a tensor to a target shape using broadcasting rules

#### func  Conv2DMPS

```go
func Conv2DMPS(input, weights, bias *Tensor, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom int) (*Tensor, error)
```
Conv2DMPS performs 2D convolution using MPSGraph

#### func  Div

```go
func Div(t1, t2 *Tensor) (*Tensor, error)
```

#### func  ExecuteFusedOperation

```go
func ExecuteFusedOperation(op OperationDesc) (*Tensor, error)
```
ExecuteFusedOperation executes a fused operation based on its type

#### func  Exp

```go
func Exp(t *Tensor) (*Tensor, error)
```

#### func  Flatten

```go
func Flatten(t *Tensor) (*Tensor, error)
```

#### func  FromScalar

```go
func FromScalar(value float64, dtype DType, device DeviceType) *Tensor
```
FromScalar creates a scalar tensor from a float64 value

#### func  Full

```go
func Full(shape []int, value interface{}, dtype DType, device DeviceType) (*Tensor, error)
```

#### func  LinearForward

```go
func LinearForward(input, weight, bias *Tensor) (*Tensor, error)
```
LinearForward performs fused matrix multiplication + bias addition Equivalent
to: MatMul(input, weight) + bias, but in a single GPU kernel

#### func  LinearReLU

```go
func LinearReLU(input, weight, bias *Tensor) (*Tensor, error)
```
LinearReLU performs fused matrix multiplication + bias addition + ReLU
activation Equivalent to: ReLU(MatMul(input, weight) + bias), but in a single
GPU kernel

#### func  LinearSigmoid

```go
func LinearSigmoid(input, weight, bias *Tensor) (*Tensor, error)
```
LinearSigmoid performs fused matrix multiplication + bias addition + Sigmoid
activation Equivalent to: Sigmoid(MatMul(input, weight) + bias), but in a single
GPU kernel

#### func  Log

```go
func Log(t *Tensor) (*Tensor, error)
```

#### func  MatMul

```go
func MatMul(t1, t2 *Tensor) (*Tensor, error)
```

#### func  MatMulAutograd

```go
func MatMulAutograd(a, b *Tensor) *Tensor
```
MatMulAutograd performs matrix multiplication with automatic differentiation

#### func  MatMulGPU

```go
func MatMulGPU(t1, t2 *Tensor) (*Tensor, error)
```

#### func  MatMulMPS

```go
func MatMulMPS(a, b *Tensor) (*Tensor, error)
```
MatMulMPS performs matrix multiplication using MPSGraph

#### func  MaxPool2DMPS

```go
func MaxPool2DMPS(input *Tensor, kernelSize, stride, padding int) (*Tensor, error)
```
MaxPool2DMPS performs 2D max pooling using MPSGraph

#### func  Mul

```go
func Mul(t1, t2 *Tensor) (*Tensor, error)
```

#### func  MulAutograd

```go
func MulAutograd(a, b *Tensor) *Tensor
```
MulAutograd performs multiplication with automatic differentiation

#### func  NewTensor

```go
func NewTensor(shape []int, dtype DType, device DeviceType, data interface{}) (*Tensor, error)
```

#### func  Ones

```go
func Ones(shape []int, dtype DType, device DeviceType) (*Tensor, error)
```

#### func  Random

```go
func Random(shape []int, dtype DType, device DeviceType) (*Tensor, error)
```

#### func  RandomNormal

```go
func RandomNormal(shape []int, mean, std float32, dtype DType, device DeviceType) (*Tensor, error)
```

#### func  ReLU

```go
func ReLU(t *Tensor) (*Tensor, error)
```

#### func  ReLUAutograd

```go
func ReLUAutograd(a *Tensor) *Tensor
```
ReLUAutograd performs ReLU activation with automatic differentiation

#### func  ReLUGPU

```go
func ReLUGPU(t *Tensor) (*Tensor, error)
```
ReLUGPU performs ReLU activation on GPU

#### func  ReLUMPS

```go
func ReLUMPS(a *Tensor) (*Tensor, error)
```
ReLUMPS performs ReLU activation using MPSGraph

#### func  Reshape

```go
func Reshape(t *Tensor, newShape []int) (*Tensor, error)
```

#### func  Sigmoid

```go
func Sigmoid(t *Tensor) (*Tensor, error)
```

#### func  SigmoidAutograd

```go
func SigmoidAutograd(a *Tensor) *Tensor
```
SigmoidAutograd performs Sigmoid activation with automatic differentiation

#### func  SigmoidMPS

```go
func SigmoidMPS(a *Tensor) (*Tensor, error)
```
SigmoidMPS performs Sigmoid activation using MPSGraph

#### func  Sqrt

```go
func Sqrt(t *Tensor) (*Tensor, error)
```
Sqrt computes the square root of a tensor element-wise

#### func  Squeeze

```go
func Squeeze(t *Tensor, dim int) (*Tensor, error)
```

#### func  Sub

```go
func Sub(t1, t2 *Tensor) (*Tensor, error)
```

#### func  SubAutograd

```go
func SubAutograd(a, b *Tensor) *Tensor
```
SubAutograd performs subtraction with automatic differentiation

#### func  Sum

```go
func Sum(t *Tensor, dim int, keepDim bool) (*Tensor, error)
```

#### func  Tanh

```go
func Tanh(t *Tensor) (*Tensor, error)
```

#### func  Transpose

```go
func Transpose(t *Tensor, dim0, dim1 int) (*Tensor, error)
```

#### func  Unsqueeze

```go
func Unsqueeze(t *Tensor, dim int) (*Tensor, error)
```

#### func  Zeros

```go
func Zeros(shape []int, dtype DType, device DeviceType) (*Tensor, error)
```

#### func (*Tensor) At

```go
func (t *Tensor) At(indices ...int) (interface{}, error)
```

#### func (*Tensor) Backward

```go
func (t *Tensor) Backward() error
```
Backward performs backpropagation from this tensor

#### func (*Tensor) Cleanup

```go
func (t *Tensor) Cleanup()
```

#### func (*Tensor) Clone

```go
func (t *Tensor) Clone() (*Tensor, error)
```

#### func (*Tensor) Dim

```go
func (t *Tensor) Dim() int
```

#### func (*Tensor) Equal

```go
func (t *Tensor) Equal(other *Tensor) (bool, error)
```

#### func (*Tensor) GetFloat32Data

```go
func (t *Tensor) GetFloat32Data() ([]float32, error)
```

#### func (*Tensor) GetGPUBuffer

```go
func (t *Tensor) GetGPUBuffer() interface{}
```
GetGPUBuffer returns the GPU buffer for this tensor

#### func (*Tensor) GetInt32Data

```go
func (t *Tensor) GetInt32Data() ([]int32, error)
```

#### func (*Tensor) Grad

```go
func (t *Tensor) Grad() *Tensor
```

#### func (*Tensor) IsOnGPU

```go
func (t *Tensor) IsOnGPU() bool
```
IsOnGPU returns true if tensor is on any GPU device

#### func (*Tensor) IsPersistent

```go
func (t *Tensor) IsPersistent() bool
```
IsPersistent returns true if tensor stays on GPU across operations

#### func (*Tensor) Item

```go
func (t *Tensor) Item() (interface{}, error)
```

#### func (*Tensor) Numel

```go
func (t *Tensor) Numel() int
```

#### func (*Tensor) PrintData

```go
func (t *Tensor) PrintData(maxElements int) string
```

#### func (*Tensor) RefCount

```go
func (t *Tensor) RefCount() int32
```
RefCount returns the current reference count for GPU tensors

#### func (*Tensor) Release

```go
func (t *Tensor) Release()
```
Release decrements the reference count and releases GPU buffer when count
reaches zero

#### func (*Tensor) RequiresGrad

```go
func (t *Tensor) RequiresGrad() bool
```

#### func (*Tensor) Reshape

```go
func (t *Tensor) Reshape(newShape []int) (*Tensor, error)
```
Reshape returns a new tensor with the same data but different shape The new
shape must have the same total number of elements

#### func (*Tensor) Retain

```go
func (t *Tensor) Retain()
```
Retain increments the reference count for GPU tensors

#### func (*Tensor) SetAt

```go
func (t *Tensor) SetAt(value interface{}, indices ...int) error
```

#### func (*Tensor) SetData

```go
func (t *Tensor) SetData(data interface{}) error
```
SetData sets the data for this tensor (public version of setData)

#### func (*Tensor) SetGPUBuffer

```go
func (t *Tensor) SetGPUBuffer(buffer interface{})
```
SetGPUBuffer sets the GPU buffer for this tensor and initializes reference
counting

#### func (*Tensor) SetGrad

```go
func (t *Tensor) SetGrad(grad *Tensor)
```
SetGrad sets the gradient for this tensor

#### func (*Tensor) SetRequiresGrad

```go
func (t *Tensor) SetRequiresGrad(requires bool)
```

#### func (*Tensor) Size

```go
func (t *Tensor) Size() []int
```

#### func (*Tensor) String

```go
func (t *Tensor) String() string
```

#### func (*Tensor) ToCPU

```go
func (t *Tensor) ToCPU() (*Tensor, error)
```
ToCPU moves a tensor to CPU device (creates a copy with CPU device type)

#### func (*Tensor) ToDevice

```go
func (t *Tensor) ToDevice(device DeviceType) (*Tensor, error)
```

#### func (*Tensor) ToGPU

```go
func (t *Tensor) ToGPU() (*Tensor, error)
```
ToGPU moves a tensor to GPU device using the BufferAllocator

#### func (*Tensor) ToPersistentGPU

```go
func (t *Tensor) ToPersistentGPU() (*Tensor, error)
```
ToPersistentGPU converts a CPU tensor to persistent GPU tensor that stays on GPU

#### func (*Tensor) Transpose

```go
func (t *Tensor) Transpose(dim0, dim1 int) (*Tensor, error)
```
Transpose returns a transposed tensor

#### func (*Tensor) ZeroGrad

```go
func (t *Tensor) ZeroGrad()
```
ZeroGrad clears the gradient for this tensor
