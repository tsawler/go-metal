# cgo_bridge
--
    import "."


## Usage

#### func  AllocateMetalBuffer

```go
func AllocateMetalBuffer(device unsafe.Pointer, size int, deviceType DeviceType) (unsafe.Pointer, error)
```
AllocateMetalBuffer allocates a Metal buffer

#### func  CopyDataToMetalBuffer

```go
func CopyDataToMetalBuffer(buffer unsafe.Pointer, data []byte) error
```
CopyDataToMetalBuffer copies raw byte data to a Metal buffer

#### func  CopyFloat32ArrayToMetalBuffer

```go
func CopyFloat32ArrayToMetalBuffer(buffer unsafe.Pointer, data []float32) error
```
CopyFloat32ArrayToMetalBuffer copies float32 array data to a Metal buffer

#### func  CopyInt32ArrayToMetalBuffer

```go
func CopyInt32ArrayToMetalBuffer(buffer unsafe.Pointer, data []int32) error
```
CopyInt32ArrayToMetalBuffer copies int32 array data to a Metal buffer

#### func  CreateMetalDevice

```go
func CreateMetalDevice() (unsafe.Pointer, error)
```
CreateMetalDevice creates a Metal device

#### func  CreateTrainingEngine

```go
func CreateTrainingEngine(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error)
```
CreateTrainingEngine creates a training engine

#### func  CreateTrainingEngineConstantWeights

```go
func CreateTrainingEngineConstantWeights(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error)
```
CreateTrainingEngineConstantWeights creates a training engine with constant
weights to avoid MPSGraph assertion

#### func  CreateTrainingEngineHybrid

```go
func CreateTrainingEngineHybrid(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error)
```
CreateTrainingEngineHybrid creates a hybrid MPS/MPSGraph training engine

#### func  DeallocateMetalBuffer

```go
func DeallocateMetalBuffer(buffer unsafe.Pointer)
```
DeallocateMetalBuffer deallocates a Metal buffer

#### func  DestroyTrainingEngine

```go
func DestroyTrainingEngine(engine unsafe.Pointer)
```
DestroyTrainingEngine destroys a training engine

#### func  ExecuteAdamStep

```go
func ExecuteAdamStep(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	varianceBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	beta1 float32,
	beta2 float32,
	epsilon float32,
	weightDecay float32,
	stepCount int,
) error
```
ExecuteAdamStep executes a single Adam optimization step on GPU

#### func  ExecuteTrainingStep

```go
func ExecuteTrainingStep(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStep executes a complete training step

#### func  ExecuteTrainingStepHybrid

```go
func ExecuteTrainingStepHybrid(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStepHybrid executes a training step using hybrid MPS/MPSGraph
approach

#### func  ExecuteTrainingStepHybridFull

```go
func ExecuteTrainingStepHybridFull(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	learningRate float32,
) (float32, error)
```
ExecuteTrainingStepHybridFull executes a complete training step with backward
pass using hybrid MPS/MPSGraph approach

#### func  ExecuteTrainingStepHybridWithGradients

```go
func ExecuteTrainingStepHybridWithGradients(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStepHybridWithGradients executes forward+backward pass and
returns gradients

#### func  ZeroMetalBuffer

```go
func ZeroMetalBuffer(device unsafe.Pointer, buffer unsafe.Pointer, size int) error
```
ZeroMetalBuffer zeros a Metal buffer

#### type DeviceType

```go
type DeviceType int
```

DeviceType maps to our memory package

```go
const (
	CPU DeviceType = iota
	GPU
	PersistentGPU
)
```

#### type OptimizerType

```go
type OptimizerType int
```

OptimizerType represents the type of optimizer

```go
const (
	SGD OptimizerType = iota
	Adam
)
```

#### type TrainingConfig

```go
type TrainingConfig struct {
	LearningRate  float32
	Beta1         float32
	Beta2         float32
	WeightDecay   float32
	Epsilon       float32
	OptimizerType OptimizerType
}
```

TrainingConfig holds training configuration
