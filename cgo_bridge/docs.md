# cgo_bridge
--
    import "."


## Usage

#### func  AllocateMetalBuffer

```go
func AllocateMetalBuffer(device unsafe.Pointer, size int, deviceType DeviceType) (unsafe.Pointer, error)
```
AllocateMetalBuffer allocates a Metal buffer

#### func  BuildInferenceGraph

```go
func BuildInferenceGraph(
	engine unsafe.Pointer,
	inputShape []int,
	inputShapeLen int32,
	batchNormInferenceMode bool,
) error
```
BuildInferenceGraph builds an optimized inference graph

#### func  CommitCommandBuffer

```go
func CommitCommandBuffer(commandBuffer unsafe.Pointer) error
```
CommitCommandBuffer commits a command buffer for execution

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

#### func  CopyMetalBufferToFloat32Array

```go
func CopyMetalBufferToFloat32Array(buffer unsafe.Pointer, numElements int) ([]float32, error)
```
CopyMetalBufferToFloat32Array copies data from a Metal buffer to a float32 array

#### func  CopyMetalBufferToInt32Array

```go
func CopyMetalBufferToInt32Array(buffer unsafe.Pointer, numElements int) ([]int32, error)
```
CopyMetalBufferToInt32Array copies data from a Metal buffer to an int32 array

#### func  CreateCommandBuffer

```go
func CreateCommandBuffer(commandQueue unsafe.Pointer) (unsafe.Pointer, error)
```
CreateCommandBuffer creates a Metal command buffer from the given command queue

#### func  CreateCommandQueue

```go
func CreateCommandQueue(device unsafe.Pointer) (unsafe.Pointer, error)
```
CreateCommandQueue creates a Metal command queue for the given device

#### func  CreateInferenceEngine

```go
func CreateInferenceEngine(device unsafe.Pointer, config InferenceConfig) (unsafe.Pointer, error)
```
CreateInferenceEngine creates an inference-only engine optimized for forward
pass

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

#### func  CreateTrainingEngineDynamic

```go
func CreateTrainingEngineDynamic(
	device unsafe.Pointer,
	config TrainingConfig,
	layerSpecs []LayerSpecC,
	inputShape []int,
) (unsafe.Pointer, error)
```
CreateTrainingEngineDynamic creates a training engine with dynamic graph from
model specification

#### func  CreateTrainingEngineHybrid

```go
func CreateTrainingEngineHybrid(device unsafe.Pointer, config TrainingConfig, modelConfig ModelConfig) (unsafe.Pointer, error)
```
CreateTrainingEngineHybrid creates a hybrid MPS/MPSGraph training engine

#### func  DeallocateMetalBuffer

```go
func DeallocateMetalBuffer(buffer unsafe.Pointer)
```
DeallocateMetalBuffer deallocates a Metal buffer

#### func  DestroyCommandQueue

```go
func DestroyCommandQueue(commandQueue unsafe.Pointer)
```
DestroyCommandQueue is an alias for ReleaseCommandQueue for consistency

#### func  DestroyInferenceEngine

```go
func DestroyInferenceEngine(engine unsafe.Pointer)
```
DestroyInferenceEngine destroys an inference engine

#### func  DestroyMetalDevice

```go
func DestroyMetalDevice(device unsafe.Pointer)
```
DestroyMetalDevice destroys a Metal device

#### func  DestroyTrainingEngine

```go
func DestroyTrainingEngine(engine unsafe.Pointer)
```
DestroyTrainingEngine destroys a training engine

#### func  DrainAutoreleasePool

```go
func DrainAutoreleasePool()
```
DrainAutoreleasePool drains the autorelease pool to release Metal resources

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
ExecuteAdamStep executes a single Adam optimization step on GPU (legacy
CPU-based implementation)

#### func  ExecuteAdamStepMPSGraph

```go
func ExecuteAdamStepMPSGraph(
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
ExecuteAdamStepMPSGraph executes a single Adam optimization step using MPSGraph
for optimal GPU performance

#### func  ExecuteAdamStepMPSGraphPooled

```go
func ExecuteAdamStepMPSGraphPooled(
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
	commandPool unsafe.Pointer,
) error
```
ExecuteAdamStepMPSGraphPooled performs Adam optimization with pooled command
buffers RESOURCE LEAK FIX: Uses command buffer pooling to prevent Metal resource
accumulation

#### func  ExecuteRMSPropStepMPSGraph

```go
func ExecuteRMSPropStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	gradientAvgBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	alpha float32,
	epsilon float32,
	weightDecay float32,
	momentum float32,
	centered bool,
	stepCount int,
) error
```
ExecuteRMSPropStepMPSGraph executes a single RMSProp optimization step using
MPSGraph for optimal GPU performance

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

#### func  ExecuteTrainingStepDynamic

```go
func ExecuteTrainingStepDynamic(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	learningRate float32,
	batchSize int,
) (float32, error)
```
ExecuteTrainingStepDynamic executes a training step using dynamic engine with
real loss computation

#### func  ExecuteTrainingStepDynamicWithGradients

```go
func ExecuteTrainingStepDynamicWithGradients(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	learningRate float32,
	batchSize int,
) (float32, error)
```
ExecuteTrainingStepDynamicWithGradients executes a dynamic training step with
real gradient computation

#### func  ExecuteTrainingStepDynamicWithGradientsPooled

```go
func ExecuteTrainingStepDynamicWithGradientsPooled(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	batchSize int,
	commandPool unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStepDynamicWithGradientsPooled executes a dynamic training step
with pooled command buffers

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

#### func  ExecuteTrainingStepHybridFullPooled

```go
func ExecuteTrainingStepHybridFullPooled(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	learningRate float32,
	commandPool unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStepHybridFullPooled executes a hybrid training step using
command buffer pooling

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

#### func  ExecuteTrainingStepHybridWithGradientsPooled

```go
func ExecuteTrainingStepHybridWithGradientsPooled(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	commandPool unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStepHybridWithGradientsPooled executes forward+backward pass with
pooled command buffers RESOURCE LEAK FIX: Uses command buffer pooling to prevent
Metal resource accumulation

#### func  ExecuteTrainingStepSGDPooled

```go
func ExecuteTrainingStepSGDPooled(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	learningRate float32,
	batchSize int,
	commandPool unsafe.Pointer,
) (float32, error)
```
ExecuteTrainingStepSGDPooled executes SGD training step with pooled command
buffers for optimal performance

#### func  GetCommandBufferFromPool

```go
func GetCommandBufferFromPool(commandPool unsafe.Pointer) (unsafe.Pointer, error)
```
GetCommandBufferFromPool gets a command buffer from the pool (Metal level
interface)

#### func  ReleaseCommandBuffer

```go
func ReleaseCommandBuffer(commandBuffer unsafe.Pointer)
```
ReleaseCommandBuffer releases a Metal command buffer

#### func  ReleaseCommandQueue

```go
func ReleaseCommandQueue(commandQueue unsafe.Pointer)
```
ReleaseCommandQueue releases a Metal command queue

#### func  ReturnCommandBufferToPool

```go
func ReturnCommandBufferToPool(commandPool unsafe.Pointer, commandBuffer unsafe.Pointer)
```
ReturnCommandBufferToPool returns a command buffer to the pool (Metal level
interface)

#### func  SetupAutoreleasePool

```go
func SetupAutoreleasePool()
```
SetupAutoreleasePool sets up an autorelease pool for Metal resource management

#### func  SetupMemoryBridge

```go
func SetupMemoryBridge(setupFunc func(
	func(unsafe.Pointer, int) ([]float32, error),
	func(unsafe.Pointer, []float32) error,
	func(unsafe.Pointer, []int32) error,
))
```
SetupMemoryBridge sets up bridge functions for memory package to avoid import
cycles Call this from packages that need both cgo_bridge and memory
functionality

#### func  WaitCommandBufferCompletion

```go
func WaitCommandBufferCompletion(commandBuffer unsafe.Pointer) error
```
WaitCommandBufferCompletion waits for a command buffer to complete execution

#### func  ZeroMetalBuffer

```go
func ZeroMetalBuffer(device unsafe.Pointer, buffer unsafe.Pointer, size int) error
```
ZeroMetalBuffer zeros a Metal buffer (uses CPU for accessible buffers, MPSGraph
for GPU-only)

#### func  ZeroMetalBufferMPSGraph

```go
func ZeroMetalBufferMPSGraph(device unsafe.Pointer, buffer unsafe.Pointer, size int) error
```
ZeroMetalBufferMPSGraph zeros a Metal buffer using MPSGraph (works for all
buffer types)

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

#### type InferenceConfig

```go
type InferenceConfig struct {
	// Model configuration
	UseDynamicEngine       bool // Use dynamic graph engine
	BatchNormInferenceMode bool // Use batch norm in inference mode

	// Input configuration
	InputShape    []int32 // Input tensor shape
	InputShapeLen int32   // Length of input shape array

	// Layer specifications for dynamic models
	LayerSpecs    []LayerSpecC // Layer specifications
	LayerSpecsLen int32        // Number of layer specs

	// Performance settings
	UseCommandPooling      bool // Enable command buffer pooling
	OptimizeForSingleBatch bool // Optimize for batch size 1
}
```

InferenceConfig holds configuration for inference-only engines

#### type InferenceResult

```go
type InferenceResult struct {
	Predictions []float32 // Model output logits/probabilities [batch_size * num_classes]
	BatchSize   int       // Actual batch size processed
	OutputShape []int     // Shape of prediction tensor [batch_size, num_classes]
	Success     bool      // Inference execution status
}
```

InferenceResult contains model predictions and metadata

#### func  ExecuteInference

```go
func ExecuteInference(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	batchSize int,
	numClasses int,
	isDynamic bool,
) (*InferenceResult, error)
```
ExecuteInference performs forward-only pass and returns predictions Conforms to
design requirements: single CGO call, GPU-resident, shared resources

#### func  ExecuteInferenceOnly

```go
func ExecuteInferenceOnly(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	batchSize int,
	numClasses int,
	isDynamic bool,
	batchNormInferenceMode bool,
) (*InferenceResult, error)
```
ExecuteInferenceOnly performs forward-only inference without loss computation

#### type LayerSpecC

```go
type LayerSpecC struct {
	LayerType       int32
	Name            [64]byte // Fixed-size array for C compatibility
	InputShape      [4]int32
	InputShapeLen   int32
	OutputShape     [4]int32
	OutputShapeLen  int32
	ParamInt        [8]int32
	ParamFloat      [8]float32
	ParamIntCount   int32
	ParamFloatCount int32
	// Running statistics for layers like BatchNorm (non-learnable parameters)
	RunningMean      []float32
	RunningVar       []float32
	RunningStatsSize int32
	HasRunningStats  int32 // Boolean flag (0 or 1)
}
```

LayerSpecC represents a C-compatible layer specification

#### type ModelConfig

```go
type ModelConfig struct {
	// Input configuration
	BatchSize     int
	InputChannels int
	InputHeight   int
	InputWidth    int

	// Convolution layer outputs
	Conv1OutChannels int
	Conv1OutHeight   int
	Conv1OutWidth    int

	Conv2OutChannels int
	Conv2OutHeight   int
	Conv2OutWidth    int

	Conv3OutChannels int
	Conv3OutHeight   int
	Conv3OutWidth    int

	// Fully connected layer dimensions
	FC1InputSize  int // Flattened conv output size
	FC1OutputSize int // Hidden layer size
	FC2OutputSize int // Number of classes

	// Convolution parameters
	Conv1KernelSize int
	Conv1Stride     int
	Conv2KernelSize int
	Conv2Stride     int
	Conv3KernelSize int
	Conv3Stride     int
}
```

ModelConfig holds model architecture configuration for dynamic dimensions

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
