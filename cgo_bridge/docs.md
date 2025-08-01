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

#### func  ConvertTensorType

```go
func ConvertTensorType(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType int, device unsafe.Pointer) error
```
ConvertTensorType converts a tensor from one data type to another on GPU

#### func  CopyBufferToBufferAsync

```go
func CopyBufferToBufferAsync(srcBuffer, dstBuffer unsafe.Pointer,
	srcOffset, dstOffset, size int,
	commandQueue unsafe.Pointer) error
```
CopyBufferToBufferAsync performs asynchronous buffer-to-buffer copy using Metal
blit encoder

#### func  CopyDataToMetalBuffer

```go
func CopyDataToMetalBuffer(buffer unsafe.Pointer, data []byte) error
```
CopyDataToMetalBuffer copies raw byte data to a Metal buffer

#### func  CopyDataToStagingBuffer

```go
func CopyDataToStagingBuffer(stagingBuffer unsafe.Pointer, data []byte) error
```
CopyDataToStagingBuffer copies data from CPU memory to staging buffer

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

#### func  CopyStagingToGPUBufferAsync

```go
func CopyStagingToGPUBufferAsync(stagingBuffer, gpuBuffer unsafe.Pointer,
	stagingOffset, gpuOffset, size int,
	commandQueue unsafe.Pointer) error
```
CopyStagingToGPUBufferAsync copies from staging buffer to GPU buffer
asynchronously

#### func  CopyTensorBufferSync

```go
func CopyTensorBufferSync(srcBuffer, dstBuffer unsafe.Pointer, size int) error
```
CopyTensorBufferSync copies data from one tensor buffer to another synchronously
This implements GPU-resident tensor copying with minimal CGO calls using the
optimized buffer copy

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
DestroyTrainingEngine destroys a training engine with comprehensive cleanup This
function properly handles all optimizer states, cached resources, and
pre-compiled operations with robust resource management

#### func  DrainAutoreleasePool

```go
func DrainAutoreleasePool()
```
DrainAutoreleasePool drains the autorelease pool to release Metal resources

#### func  ExecuteAdaDeltaStepMPSGraph

```go
func ExecuteAdaDeltaStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	squaredUpdateAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	rho float32,
	epsilon float32,
	weightDecay float32,
) error
```
ExecuteAdaDeltaStepMPSGraph executes a single AdaDelta optimization step using
MPSGraph for optimal GPU performance

#### func  ExecuteAdaDeltaStepMPSGraphPooled

```go
func ExecuteAdaDeltaStepMPSGraphPooled(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	squaredUpdateAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	rho float32,
	epsilon float32,
	weightDecay float32,
	commandPool unsafe.Pointer,
) error
```
ExecuteAdaDeltaStepMPSGraphPooled executes a single AdaDelta optimization step
using MPSGraph with command buffer pooling

#### func  ExecuteAdaGradStepMPSGraph

```go
func ExecuteAdaGradStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	learningRate float32,
	epsilon float32,
	weightDecay float32,
) error
```
ExecuteAdaGradStepMPSGraph executes a single AdaGrad optimization step using
MPSGraph for optimal GPU performance

#### func  ExecuteAdaGradStepMPSGraphPooled

```go
func ExecuteAdaGradStepMPSGraphPooled(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	learningRate float32,
	epsilon float32,
	weightDecay float32,
	commandPool unsafe.Pointer,
) error
```
ExecuteAdaGradStepMPSGraphPooled executes a single AdaGrad optimization step
using MPSGraph with command buffer pooling

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

#### func  ExecuteLBFGSStepMPSGraph

```go
func ExecuteLBFGSStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	oldGradientBuffers []unsafe.Pointer,
	searchDirBuffers []unsafe.Pointer,
	sVectors [][]unsafe.Pointer,
	yVectors [][]unsafe.Pointer,
	rhoBuffers []unsafe.Pointer,
	alphaBuffer unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	historySize int,
	historyCount int,
	historyIndex int,
	initialStep float32,
	c1 float32,
	c2 float32,
	maxLineSearch int,
	currentLoss float32,
	prevLoss float32,
	commandPool unsafe.Pointer,
	usePooling bool,
) (float32, error)
```
ExecuteLBFGSStepMPSGraph executes a single L-BFGS optimization step using
MPSGraph for optimal GPU performance

#### func  ExecuteNadamStepMPSGraph

```go
func ExecuteNadamStepMPSGraph(
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
ExecuteNadamStepMPSGraph executes a single Nadam optimization step using
MPSGraph for optimal GPU performance Nadam combines Adam's adaptive learning
rates with Nesterov momentum

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

#### func  ExecuteSGDStepMPSGraph

```go
func ExecuteSGDStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	momentum float32,
	weightDecay float32,
	nesterov bool,
	stepCount int,
) error
```
ExecuteSGDStepMPSGraph executes a single SGD optimization step using MPSGraph
for optimal GPU performance

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
func ReturnCommandBufferToPool(commandBuffer unsafe.Pointer)
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

#### func  SetupMemoryBridgeWithConvert

```go
func SetupMemoryBridgeWithConvert(setupFunc func(
	func(unsafe.Pointer, int) ([]float32, error),
	func(unsafe.Pointer, []float32) error,
	func(unsafe.Pointer, []int32) error,
	func(unsafe.Pointer, unsafe.Pointer, []int, int, int) error,
	func(unsafe.Pointer, unsafe.Pointer, int) error,
), getDeviceFunc func() unsafe.Pointer)
```
SetupMemoryBridgeWithConvert sets up bridge functions including type conversion

#### func  WaitCommandBufferCompletion

```go
func WaitCommandBufferCompletion(commandBuffer unsafe.Pointer) error
```
WaitCommandBufferCompletion waits for a command buffer to complete execution

#### func  WaitForBufferCopyCompletion

```go
func WaitForBufferCopyCompletion(commandQueue unsafe.Pointer) error
```
WaitForBufferCopyCompletion waits for all pending buffer copy operations to
complete

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

#### type DedicatedInferenceConfig

```go
type DedicatedInferenceConfig struct {
	PrecisionThreshold  float32           // Float16 conversion threshold
	MaxBatchSize        int               // Maximum supported batch size
	OptimizationLevel   OptimizationLevel // Optimization aggressiveness
	MemoryStrategy      MemoryStrategy    // Memory management approach
	EnableTelemetry     bool              // Enable performance monitoring
	CacheCompiledGraphs bool              // Cache compiled MPSGraph executables
}
```

DedicatedInferenceConfig holds configuration for the dedicated inference engine

#### type DedicatedInferenceEngine

```go
type DedicatedInferenceEngine struct {
}
```

DedicatedInferenceEngine represents a GPU-resident inference engine optimized
for forward pass only

#### func  NewDedicatedInferenceEngine

```go
func NewDedicatedInferenceEngine(
	device unsafe.Pointer,
	config DedicatedInferenceConfig,
	layers []LayerSpecC,
	parameters [][]float32,
) (*DedicatedInferenceEngine, error)
```
NewDedicatedInferenceEngine creates a new dedicated inference engine optimized
for forward pass only

#### func (*DedicatedInferenceEngine) Destroy

```go
func (e *DedicatedInferenceEngine) Destroy() error
```
Destroy properly cleans up the inference engine resources

#### func (*DedicatedInferenceEngine) GetTelemetry

```go
func (e *DedicatedInferenceEngine) GetTelemetry() (*InferenceTelemetry, error)
```
GetTelemetry returns performance telemetry data

#### func (*DedicatedInferenceEngine) InferBatch

```go
func (e *DedicatedInferenceEngine) InferBatch(
	inputData []float32,
	inputShape []int,
	batchSize int,
) (*DedicatedInferenceResult, error)
```
InferBatch performs batch inference with optimized GPU execution

#### func (*DedicatedInferenceEngine) InferSingle

```go
func (e *DedicatedInferenceEngine) InferSingle(
	inputData []float32,
	inputShape []int,
) (*DedicatedInferenceResult, error)
```
InferSingle performs single sample inference

#### func (*DedicatedInferenceEngine) PreallocateBuffers

```go
func (e *DedicatedInferenceEngine) PreallocateBuffers(maxBatchSize int) error
```
PreallocateBuffers pre-allocates GPU buffers for optimal performance

#### func (*DedicatedInferenceEngine) ResetTelemetry

```go
func (e *DedicatedInferenceEngine) ResetTelemetry() error
```
ResetTelemetry clears all telemetry counters

#### type DedicatedInferenceResult

```go
type DedicatedInferenceResult struct {
	Predictions     []float32 // Output predictions (GPU -> CPU copied)
	OutputShape     []int     // Shape of output tensor
	ConfidenceScore float32   // Maximum confidence/probability
	PredictedClass  int       // Predicted class index (for classification)
	InferenceTimeMs float64   // Time taken for this inference
	MemoryUsedBytes uint64    // GPU memory used for this inference
}
```

DedicatedInferenceResult contains comprehensive inference results and metadata

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

	// Problem type and loss function (CRITICAL FIX for regression inference)
	ProblemType  int // 0 = Classification, 1 = Regression
	LossFunction int // 0 = CrossEntropy, 1 = SparseCrossEntropy, 2 = MSE, 3 = MAE, 4 = Huber

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

#### type InferenceTelemetry

```go
type InferenceTelemetry struct {
	TotalInferences uint64  // Total inference calls
	TotalTimeMs     float64 // Total inference time in milliseconds
	AvgLatencyMs    float64 // Average inference latency
	PeakThroughput  float64 // Peak throughput (inferences/second)
	PeakMemoryUsage uint64  // Peak GPU memory usage
	CacheHits       uint64  // Graph compilation cache hits
	CacheMisses     uint64  // Graph compilation cache misses
}
```

InferenceTelemetry provides performance metrics for the inference engine

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

#### type MemoryStrategy

```go
type MemoryStrategy int
```

MemoryStrategy controls how the inference engine manages GPU memory

```go
const (
	Minimal      MemoryStrategy = iota // Minimal memory usage
	BalancedMem                        // Balanced memory vs performance
	PreAllocated                       // Pre-allocate buffers for maximum performance
)
```

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

#### type OptimizationLevel

```go
type OptimizationLevel int
```

OptimizationLevel controls the level of optimizations applied to the inference
engine

```go
const (
	Conservative OptimizationLevel = iota // Safe optimizations only
	Balanced                              // Standard inference optimizations
	Aggressive                            // Maximum performance optimizations
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
	RMSProp
	LBFGS
	AdaGrad
	AdaDelta
	Nadam
)
```

#### type TrainingConfig

```go
type TrainingConfig struct {
	LearningRate  float32
	Beta1         float32 // Adam momentum decay (or RMSProp momentum if > 0)
	Beta2         float32 // Adam variance decay (unused for RMSProp)
	WeightDecay   float32
	Epsilon       float32
	Alpha         float32 // RMSProp smoothing constant (typically 0.99)
	Momentum      float32 // RMSProp momentum (typically 0.0 or 0.9)
	Centered      bool    // RMSProp centered variant
	OptimizerType OptimizerType
	ProblemType   int // 0 = Classification, 1 = Regression
	LossFunction  int // 0 = CrossEntropy, 1 = SparseCrossEntropy, 2 = MSE, 3 = MAE, 4 = Huber
}
```

TrainingConfig holds training configuration
