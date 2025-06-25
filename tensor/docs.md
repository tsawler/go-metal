# tensor
--
    import "."


## Usage

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

#### func  IsGPUAvailable

```go
func IsGPUAvailable() bool
```
IsGPUAvailable checks if Metal GPU compute is available

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
	CPU DeviceType = iota
	GPU
)
```

#### func (DeviceType) String

```go
func (d DeviceType) String() string
```

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

#### func  Exp

```go
func Exp(t *Tensor) (*Tensor, error)
```

#### func  Flatten

```go
func Flatten(t *Tensor) (*Tensor, error)
```

#### func  Full

```go
func Full(shape []int, value interface{}, dtype DType, device DeviceType) (*Tensor, error)
```

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
MatMulGPU performs matrix multiplication on GPU

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

#### func (*Tensor) GetInt32Data

```go
func (t *Tensor) GetInt32Data() ([]int32, error)
```

#### func (*Tensor) Grad

```go
func (t *Tensor) Grad() *Tensor
```

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

#### func (*Tensor) Release

```go
func (t *Tensor) Release()
```

#### func (*Tensor) RequiresGrad

```go
func (t *Tensor) RequiresGrad() bool
```

#### func (*Tensor) SetAt

```go
func (t *Tensor) SetAt(value interface{}, indices ...int) error
```

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
ToGPU moves a tensor to GPU device (creates a copy with GPU device type)

#### func (*Tensor) ZeroGrad

```go
func (t *Tensor) ZeroGrad()
```
ZeroGrad clears the gradient for this tensor
