# layers
--
    import "."


## Usage

#### func  ConvertToCGOLayerSpecs

```go
func ConvertToCGOLayerSpecs(dynamicSpecs []DynamicLayerSpec) []interface{}
```

#### type DynamicLayerSpec

```go
type DynamicLayerSpec struct {
	LayerType       int32
	Name            string
	NameBytes       [64]byte // C-compatible name storage
	InputShape      [4]int32
	InputShapeLen   int32
	OutputShape     [4]int32
	OutputShapeLen  int32
	ParamInt        [8]int32
	ParamFloat      [8]float32
	ParamIntCount   int32
	ParamFloatCount int32
	// Running statistics for layers like BatchNorm (non-learnable parameters)
	RunningMean     []float32
	RunningVar      []float32
	HasRunningStats bool
}
```

DynamicLayerSpec represents a layer specification compatible with the dynamic
engine

#### type LayerFactory

```go
type LayerFactory struct{}
```

LayerFactory creates layer specifications (configuration only)

#### func  NewFactory

```go
func NewFactory() *LayerFactory
```
NewFactory creates a new layer factory

#### func (*LayerFactory) CreateBatchNormSpec

```go
func (lf *LayerFactory) CreateBatchNormSpec(numFeatures int, eps float32, momentum float32, affine bool, name string) LayerSpec
```
CreateBatchNormSpec creates a Batch Normalization layer specification

#### func (*LayerFactory) CreateConv2DSpec

```go
func (lf *LayerFactory) CreateConv2DSpec(
	inputChannels, outputChannels, kernelSize, stride, padding int,
	useBias bool, name string,
) LayerSpec
```
CreateConv2DSpec creates a Conv2D layer specification

#### func (*LayerFactory) CreateDenseSpec

```go
func (lf *LayerFactory) CreateDenseSpec(inputSize, outputSize int, useBias bool, name string) LayerSpec
```
CreateDenseSpec creates a dense layer specification

#### func (*LayerFactory) CreateDropoutSpec

```go
func (lf *LayerFactory) CreateDropoutSpec(rate float32, name string) LayerSpec
```
CreateDropoutSpec creates a Dropout layer specification

#### func (*LayerFactory) CreateELUSpec

```go
func (lf *LayerFactory) CreateELUSpec(alpha float32, name string) LayerSpec
```
CreateELUSpec creates an ELU activation specification

#### func (*LayerFactory) CreateLeakyReLUSpec

```go
func (lf *LayerFactory) CreateLeakyReLUSpec(negativeSlope float32, name string) LayerSpec
```
CreateLeakyReLUSpec creates a Leaky ReLU activation specification

#### func (*LayerFactory) CreateReLUSpec

```go
func (lf *LayerFactory) CreateReLUSpec(name string) LayerSpec
```
CreateReLUSpec creates a ReLU activation specification

#### func (*LayerFactory) CreateSigmoidSpec

```go
func (lf *LayerFactory) CreateSigmoidSpec(name string) LayerSpec
```
CreateSigmoidSpec creates a Sigmoid activation specification

#### func (*LayerFactory) CreateSoftmaxSpec

```go
func (lf *LayerFactory) CreateSoftmaxSpec(axis int, name string) LayerSpec
```
CreateSoftmaxSpec creates a Softmax activation specification

#### func (*LayerFactory) CreateSwishSpec

```go
func (lf *LayerFactory) CreateSwishSpec(name string) LayerSpec
```
CreateSwishSpec creates a Swish activation specification

#### func (*LayerFactory) CreateTanhSpec

```go
func (lf *LayerFactory) CreateTanhSpec(name string) LayerSpec
```
CreateTanhSpec creates a Tanh activation specification

#### type LayerSpec

```go
type LayerSpec struct {
	Type       LayerType              `json:"type"`
	Name       string                 `json:"name"`
	Parameters map[string]interface{} `json:"parameters"`

	// Shape information (computed during model compilation)
	InputShape  []int `json:"input_shape,omitempty"`
	OutputShape []int `json:"output_shape,omitempty"`

	// Parameter metadata (computed during model compilation)
	ParameterShapes [][]int `json:"parameter_shapes,omitempty"`
	ParameterCount  int64   `json:"parameter_count,omitempty"`

	// Non-learnable parameters (e.g., BatchNorm running statistics)
	// These are used during graph construction but not counted as learnable parameters
	RunningStatistics map[string][]float32 `json:"running_statistics,omitempty"`
}
```

LayerSpec defines layer configuration for the TrainingEngine This is pure
configuration - no execution logic

#### type LayerSpecC

```go
type LayerSpecC struct {
	LayerType   int32
	Name        string
	InputShape  []int32
	OutputShape []int32
	ParamInt    []int32
	ParamFloat  []float32
}
```


#### type LayerType

```go
type LayerType int
```

LayerType represents the type of neural network layer

```go
const (
	Dense LayerType = iota
	Conv2D
	ReLU
	Softmax
	MaxPool2D
	Dropout
	BatchNorm
	LeakyReLU
	ELU
	Sigmoid
	Tanh
	Swish
)
```

#### func (LayerType) String

```go
func (lt LayerType) String() string
```

#### type ModelBuilder

```go
type ModelBuilder struct {
}
```

ModelBuilder helps construct neural network models

#### func  NewModelBuilder

```go
func NewModelBuilder(inputShape []int) *ModelBuilder
```
NewModelBuilder creates a new model builder

#### func (*ModelBuilder) AddBatchNorm

```go
func (mb *ModelBuilder) AddBatchNorm(numFeatures int, eps float32, momentum float32, affine bool, name string) *ModelBuilder
```
AddBatchNorm adds a Batch Normalization layer to the model num_features: number
of input features (channels for Conv layers, neurons for Dense layers) eps:
small value added for numerical stability (default: 1e-5) momentum: momentum for
running statistics update (default: 0.1) affine: whether to use learnable scale
and shift parameters (default: true) track_running_stats: whether to track
running statistics during training (default: true)

#### func (*ModelBuilder) AddConv2D

```go
func (mb *ModelBuilder) AddConv2D(
	outputChannels, kernelSize, stride, padding int,
	useBias bool, name string,
) *ModelBuilder
```
AddConv2D adds a Conv2D layer to the model

#### func (*ModelBuilder) AddDense

```go
func (mb *ModelBuilder) AddDense(outputSize int, useBias bool, name string) *ModelBuilder
```
AddDense adds a dense layer to the model

#### func (*ModelBuilder) AddDropout

```go
func (mb *ModelBuilder) AddDropout(rate float32, name string) *ModelBuilder
```
AddDropout adds a Dropout layer to the model rate: dropout probability (0.0 = no
dropout, 1.0 = drop all) training: whether the layer is in training mode
(affects dropout behavior)

#### func (*ModelBuilder) AddELU

```go
func (mb *ModelBuilder) AddELU(alpha float32, name string) *ModelBuilder
```
AddELU adds an ELU activation to the model alpha: controls saturation level for
negative inputs (default: 1.0)

#### func (*ModelBuilder) AddLayer

```go
func (mb *ModelBuilder) AddLayer(layer LayerSpec) *ModelBuilder
```
AddLayer adds a layer to the model

#### func (*ModelBuilder) AddLeakyReLU

```go
func (mb *ModelBuilder) AddLeakyReLU(negativeSlope float32, name string) *ModelBuilder
```
AddLeakyReLU adds a Leaky ReLU activation to the model negativeSlope: slope for
negative input values (default: 0.01)

#### func (*ModelBuilder) AddReLU

```go
func (mb *ModelBuilder) AddReLU(name string) *ModelBuilder
```
AddReLU adds a ReLU activation to the model

#### func (*ModelBuilder) AddSigmoid

```go
func (mb *ModelBuilder) AddSigmoid(name string) *ModelBuilder
```
AddSigmoid adds a Sigmoid activation to the model Sigmoid(x) = 1/(1+e^(-x)) -
outputs values between 0 and 1

#### func (*ModelBuilder) AddSoftmax

```go
func (mb *ModelBuilder) AddSoftmax(axis int, name string) *ModelBuilder
```
AddSoftmax adds a Softmax activation to the model

#### func (*ModelBuilder) AddSwish

```go
func (mb *ModelBuilder) AddSwish(name string) *ModelBuilder
```
AddSwish adds a Swish activation to the model Swish(x) = x * Sigmoid(x) - smooth
activation with improved gradient flow

#### func (*ModelBuilder) AddTanh

```go
func (mb *ModelBuilder) AddTanh(name string) *ModelBuilder
```
AddTanh adds a Tanh activation to the model Tanh(x) = (e^x - e^(-x))/(e^x +
e^(-x)) - outputs values between -1 and 1

#### func (*ModelBuilder) Compile

```go
func (mb *ModelBuilder) Compile() (*ModelSpec, error)
```
Compile compiles the model and computes shapes and parameter counts

#### func (*ModelBuilder) GetCompiledModel

```go
func (mb *ModelBuilder) GetCompiledModel() (*ModelSpec, error)
```
GetCompiledModel returns the compiled model (must call Compile first)

#### type ModelSpec

```go
type ModelSpec struct {
	Layers []LayerSpec `json:"layers"`

	// Compiled model information
	TotalParameters int64   `json:"total_parameters"`
	ParameterShapes [][]int `json:"parameter_shapes"`
	InputShape      []int   `json:"input_shape"`
	OutputShape     []int   `json:"output_shape"`
	Compiled        bool    `json:"compiled"`
}
```

ModelSpec defines a complete neural network model as layer configuration This
replaces individual layer execution with unified model specification

#### func (*ModelSpec) ConvertToDynamicLayerSpecs

```go
func (ms *ModelSpec) ConvertToDynamicLayerSpecs() ([]DynamicLayerSpec, error)
```
ConvertToDynamicLayerSpecs converts ModelSpec to dynamic engine layer
specifications This is used by the true dynamic engine implementation for any
architecture support

#### func (*ModelSpec) ConvertToInferenceLayerSpecs

```go
func (ms *ModelSpec) ConvertToInferenceLayerSpecs() ([]DynamicLayerSpec, error)
```
ConvertToInferenceLayerSpecs converts ModelSpec to inference-optimized layer
specifications

#### func (*ModelSpec) CreateParameterTensors

```go
func (ms *ModelSpec) CreateParameterTensors() ([]*memory.Tensor, error)
```
CreateParameterTensors creates GPU tensors for all model parameters

#### func (*ModelSpec) GetTrainingEngineConfig

```go
func (ms *ModelSpec) GetTrainingEngineConfig() (map[string]interface{}, error)
```
GetTrainingEngineConfig generates configuration for the existing TrainingEngine
This bridges the new layer system with the proven high-performance engine

#### func (*ModelSpec) SerializeForCGO

```go
func (ms *ModelSpec) SerializeForCGO() (*ModelSpecC, error)
```
SerializeForCGO converts ModelSpec to CGO-compatible format

#### func (*ModelSpec) Summary

```go
func (ms *ModelSpec) Summary() string
```
Summary returns a human-readable model summary

#### func (*ModelSpec) ValidateModelForDynamicEngine

```go
func (ms *ModelSpec) ValidateModelForDynamicEngine() error
```
ValidateModelForDynamicEngine checks if model is compatible with Dynamic
TrainingEngine Supports any architecture with flexible input dimensions (2D, 4D,
etc.)

#### func (*ModelSpec) ValidateModelForHybridEngine

```go
func (ms *ModelSpec) ValidateModelForHybridEngine() error
```
ValidateModelForHybridEngine checks if model is compatible with training engine
DEPRECATED: The hybrid engine has been removed. Use
ValidateModelForDynamicEngine instead.

#### func (*ModelSpec) ValidateModelForInference

```go
func (ms *ModelSpec) ValidateModelForInference() error
```
ValidateModelForInference checks if model is compatible with InferenceEngine
Uses dynamic engine validation to support any model architecture

#### func (*ModelSpec) ValidateModelForTrainingEngine

```go
func (ms *ModelSpec) ValidateModelForTrainingEngine() error
```
ValidateModelForTrainingEngine checks if model is compatible with training
engine DEPRECATED: The hybrid engine has been removed. Use
ValidateModelForDynamicEngine instead.

#### type ModelSpecC

```go
type ModelSpecC struct {
	Layers      []LayerSpecC
	InputShape  []int32
	OutputShape []int32
}
```

ModelSpecC represents a CGO-compatible model specification This needs to be
defined here to avoid circular imports
