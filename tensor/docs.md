# tensor
--
    import "."


## Usage

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

#### type Operation

```go
type Operation interface {
	Forward(...*Tensor) *Tensor
	Backward(gradOut *Tensor) []*Tensor
}
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

#### func  AddGPU

```go
func AddGPU(t1, t2 *Tensor) (*Tensor, error)
```
AddGPU performs tensor addition on GPU

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

#### func  MatMulGPU

```go
func MatMulGPU(t1, t2 *Tensor) (*Tensor, error)
```
MatMulGPU performs matrix multiplication on GPU

#### func  Mul

```go
func Mul(t1, t2 *Tensor) (*Tensor, error)
```

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

#### func  ReLUGPU

```go
func ReLUGPU(t *Tensor) (*Tensor, error)
```
ReLUGPU performs ReLU activation on GPU

#### func  Reshape

```go
func Reshape(t *Tensor, newShape []int) (*Tensor, error)
```

#### func  Sigmoid

```go
func Sigmoid(t *Tensor) (*Tensor, error)
```

#### func  Squeeze

```go
func Squeeze(t *Tensor, dim int) (*Tensor, error)
```

#### func  Sub

```go
func Sub(t1, t2 *Tensor) (*Tensor, error)
```

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
