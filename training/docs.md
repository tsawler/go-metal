# training
--
    import "."


## Usage

#### func  CreateDummyWeights

```go
func CreateDummyWeights() ([]*memory.Tensor, error)
```
CreateDummyWeights creates dummy weight tensors for testing (hybrid approach)

#### func  RunProgressBarDemo

```go
func RunProgressBarDemo()
```
RunProgressBarDemo runs a demonstration of the progress bar

#### func  TestPhase1

```go
func TestPhase1() error
```
TestPhase1 runs a basic test of Phase 1 implementation

#### func  TrainingExample

```go
func TrainingExample() error
```
TrainingExample demonstrates the PyTorch-style progress bar in action

#### type ModelArchitecturePrinter

```go
type ModelArchitecturePrinter struct {
}
```

ModelArchitecturePrinter prints PyTorch-style model architecture

#### func  NewModelArchitecturePrinter

```go
func NewModelArchitecturePrinter(modelName string) *ModelArchitecturePrinter
```
NewModelArchitecturePrinter creates a new model architecture printer

#### func (*ModelArchitecturePrinter) PrintArchitecture

```go
func (p *ModelArchitecturePrinter) PrintArchitecture(modelSpec *layers.ModelSpec)
```
PrintArchitecture prints the model architecture in PyTorch style

#### type ModelTrainer

```go
type ModelTrainer struct {
}
```

ModelTrainer provides layer-based training while maintaining the proven
single-CGO-call architecture This is the compliant implementation that
integrates with the existing high-performance TrainingEngine

#### func  NewModelTrainer

```go
func NewModelTrainer(
	modelSpec *layers.ModelSpec,
	config TrainerConfig,
) (*ModelTrainer, error)
```
NewModelTrainer creates a new model-based trainer using the existing
TrainingEngine architecture

#### func (*ModelTrainer) CalculateAccuracy

```go
func (mt *ModelTrainer) CalculateAccuracy(
	predictions []float32,
	trueLabels []int32,
	batchSize int,
	numClasses int,
) float64
```
CalculateAccuracy computes accuracy from inference results and true labels Uses
CPU-based argmax for final scalar metric (design compliant)

#### func (*ModelTrainer) Cleanup

```go
func (mt *ModelTrainer) Cleanup()
```
Cleanup releases all resources

#### func (*ModelTrainer) CreateTrainingSession

```go
func (mt *ModelTrainer) CreateTrainingSession(
	modelName string,
	epochs int,
	stepsPerEpoch int,
	validationSteps int,
) *TrainingSession
```
CreateTrainingSession creates a training session with progress visualization

#### func (*ModelTrainer) GetModelSpec

```go
func (mt *ModelTrainer) GetModelSpec() *layers.ModelSpec
```
GetModelSpec returns the model specification

#### func (*ModelTrainer) GetModelSummary

```go
func (mt *ModelTrainer) GetModelSummary() string
```
GetModelSummary returns a human-readable model summary

#### func (*ModelTrainer) GetStats

```go
func (mt *ModelTrainer) GetStats() *ModelTrainingStats
```
GetStats returns comprehensive training statistics

#### func (*ModelTrainer) InferBatch

```go
func (mt *ModelTrainer) InferBatch(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error)
```
InferBatch performs inference on a batch of data Conforms to design
requirements: single CGO call, GPU-resident, shared resources

#### func (*ModelTrainer) PrintModelArchitecture

```go
func (mt *ModelTrainer) PrintModelArchitecture(modelName string)
```
PrintModelArchitecture prints the model architecture in PyTorch style

#### func (*ModelTrainer) TrainBatch

```go
func (mt *ModelTrainer) TrainBatch(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResult, error)
```
TrainBatch executes a single training step on a batch of data This maintains the
single-CGO-call principle while supporting flexible layer models

#### type ModelTrainerFactory

```go
type ModelTrainerFactory struct{}
```

ModelTrainerFactory provides methods to create model trainers with different
configurations

#### func  NewModelFactory

```go
func NewModelFactory() *ModelTrainerFactory
```
NewModelFactory creates a new model trainer factory

#### func (*ModelTrainerFactory) CreateCNNTrainer

```go
func (mtf *ModelTrainerFactory) CreateCNNTrainer(
	inputShape []int,
	numClasses int,
	config TrainerConfig,
) (*ModelTrainer, error)
```
CreateCNNTrainer creates a CNN trainer with typical architecture

#### func (*ModelTrainerFactory) CreateMLPTrainer

```go
func (mtf *ModelTrainerFactory) CreateMLPTrainer(
	inputSize int,
	hiddenSizes []int,
	outputSize int,
	config TrainerConfig,
) (*ModelTrainer, error)
```
CreateMLPTrainer creates a multi-layer perceptron trainer

#### func (*ModelTrainerFactory) CreateModelTrainer

```go
func (mtf *ModelTrainerFactory) CreateModelTrainer(
	modelSpec *layers.ModelSpec,
	config TrainerConfig,
) (*ModelTrainer, error)
```
CreateModelTrainer creates a model trainer with full configuration control

#### type ModelTrainingStats

```go
type ModelTrainingStats struct {
	CurrentStep     int
	TotalSteps      int64
	BatchSize       int
	OptimizerType   cgo_bridge.OptimizerType
	LearningRate    float32
	AverageLoss     float32
	LastStepTime    time.Duration
	ModelSummary    string
	MemoryPoolStats map[memory.PoolKey]string
	ModelParameters int64
	LayerCount      int64
}
```

ModelTrainingStats provides comprehensive statistics for model-based training

#### type OptimizerConfig

```go
type OptimizerConfig struct {
	Type         cgo_bridge.OptimizerType
	LearningRate float32
	Beta1        float32 // Adam only
	Beta2        float32 // Adam only
	Epsilon      float32 // Adam only
	WeightDecay  float32
}
```

OptimizerConfig provides optimizer-specific configurations

#### type ProgressBar

```go
type ProgressBar struct {
}
```

ProgressBar provides PyTorch-style training progress visualization

#### func  NewProgressBar

```go
func NewProgressBar(description string, total int) *ProgressBar
```
NewProgressBar creates a new progress bar

#### func (*ProgressBar) Finish

```go
func (pb *ProgressBar) Finish()
```
Finish completes the progress bar

#### func (*ProgressBar) Update

```go
func (pb *ProgressBar) Update(step int, metrics map[string]float64)
```
Update advances the progress bar

#### func (*ProgressBar) UpdateMetrics

```go
func (pb *ProgressBar) UpdateMetrics(metrics map[string]float64)
```
UpdateMetrics updates metrics without advancing progress

#### type SimpleTrainer

```go
type SimpleTrainer struct {
}
```

SimpleTrainer provides a basic training interface for testing Phase 1

#### func  NewAdamTrainer

```go
func NewAdamTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
NewAdamTrainer creates an Adam trainer with defaults (convenience function)

#### func  NewSGDTrainer

```go
func NewSGDTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
NewSGDTrainer creates an SGD trainer (convenience function)

#### func  NewSimpleTrainer

```go
func NewSimpleTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
NewSimpleTrainer creates a new simple trainer (legacy function - use factory for
production) DEPRECATED: Use NewSGDTrainer, NewAdamTrainer, or the factory system
for production code

#### func  NewTrainerWithConfig

```go
func NewTrainerWithConfig(config TrainerConfig) (*SimpleTrainer, error)
```
NewTrainerWithConfig creates a trainer with full configuration (convenience
function)

#### func (*SimpleTrainer) Cleanup

```go
func (st *SimpleTrainer) Cleanup()
```
Cleanup releases resources

#### func (*SimpleTrainer) GetStats

```go
func (st *SimpleTrainer) GetStats() *TrainingStats
```
GetStats returns training statistics

#### func (*SimpleTrainer) TrainBatch

```go
func (st *SimpleTrainer) TrainBatch(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
	weights []*memory.Tensor,
) (*TrainingResult, error)
```
TrainBatch trains on a single batch with timing (full training loop)

#### type TrainerConfig

```go
type TrainerConfig struct {
	// Training parameters
	BatchSize    int     `json:"batch_size"`
	LearningRate float32 `json:"learning_rate"`

	// Optimizer configuration
	OptimizerType cgo_bridge.OptimizerType `json:"optimizer_type"`

	// Adam-specific parameters (ignored for SGD)
	Beta1       float32 `json:"beta1"`        // Adam momentum decay (default: 0.9)
	Beta2       float32 `json:"beta2"`        // Adam RMSprop decay (default: 0.999)
	Epsilon     float32 `json:"epsilon"`      // Adam numerical stability (default: 1e-8)
	WeightDecay float32 `json:"weight_decay"` // L2 regularization (default: 0.0)

	// Training behavior
	UseHybridEngine  bool `json:"use_hybrid_engine"`  // Use hybrid MPS/MPSGraph (recommended: true)
	UseDynamicEngine bool `json:"use_dynamic_engine"` // Use dynamic graph creation for any architecture (recommended: true)
}
```

TrainerConfig provides comprehensive configuration for training

#### type TrainerFactory

```go
type TrainerFactory struct{}
```

TrainerFactory provides methods to create different types of trainers

#### func  NewFactory

```go
func NewFactory() *TrainerFactory
```
NewFactory creates a new trainer factory

#### func (*TrainerFactory) CreateAdamTrainer

```go
func (tf *TrainerFactory) CreateAdamTrainer(batchSize int, learningRate float32, beta1, beta2, epsilon, weightDecay float32) (*SimpleTrainer, error)
```
CreateAdamTrainer creates an Adam trainer with specified parameters

#### func (*TrainerFactory) CreateAdamTrainerWithDefaults

```go
func (tf *TrainerFactory) CreateAdamTrainerWithDefaults(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
CreateAdamTrainerWithDefaults creates an Adam trainer with sensible defaults

#### func (*TrainerFactory) CreateProductionTrainer

```go
func (tf *TrainerFactory) CreateProductionTrainer(batchSize int, optimizerConfig OptimizerConfig) (*SimpleTrainer, error)
```
CreateProductionTrainer creates a trainer optimized for production use

#### func (*TrainerFactory) CreateSGDTrainer

```go
func (tf *TrainerFactory) CreateSGDTrainer(batchSize int, learningRate float32, weightDecay float32) (*SimpleTrainer, error)
```
CreateSGDTrainer creates an SGD trainer with specified parameters

#### func (*TrainerFactory) CreateTrainer

```go
func (tf *TrainerFactory) CreateTrainer(config TrainerConfig) (*SimpleTrainer, error)
```
CreateTrainer creates a trainer with full configuration control

#### func (*TrainerFactory) GetDefaultAdamConfig

```go
func (tf *TrainerFactory) GetDefaultAdamConfig(learningRate float32) OptimizerConfig
```
GetDefaultAdamConfig returns default Adam configuration

#### func (*TrainerFactory) GetDefaultSGDConfig

```go
func (tf *TrainerFactory) GetDefaultSGDConfig(learningRate float32) OptimizerConfig
```
GetDefaultSGDConfig returns default SGD configuration

#### type TrainingResult

```go
type TrainingResult struct {
	Loss      float32
	BatchSize int
	StepTime  time.Duration
	Success   bool
	BatchRate float64 // batches per second
}
```

TrainingResult represents the result of a training step

#### type TrainingSession

```go
type TrainingSession struct {
}
```

TrainingSession manages a complete training session with progress visualization

#### func  NewTrainingSession

```go
func NewTrainingSession(
	trainer *ModelTrainer,
	modelName string,
	epochs int,
	stepsPerEpoch int,
	validationSteps int,
) *TrainingSession
```
NewTrainingSession creates a new training session with progress visualization

#### func (*TrainingSession) FinishTrainingEpoch

```go
func (ts *TrainingSession) FinishTrainingEpoch()
```
FinishTrainingEpoch completes the training phase of an epoch

#### func (*TrainingSession) FinishValidationEpoch

```go
func (ts *TrainingSession) FinishValidationEpoch()
```
FinishValidationEpoch completes the validation phase of an epoch

#### func (*TrainingSession) PrintEpochSummary

```go
func (ts *TrainingSession) PrintEpochSummary()
```
PrintEpochSummary prints a summary of the completed epoch

#### func (*TrainingSession) StartEpoch

```go
func (ts *TrainingSession) StartEpoch(epoch int)
```
StartEpoch begins a new epoch

#### func (*TrainingSession) StartTraining

```go
func (ts *TrainingSession) StartTraining()
```
StartTraining begins the training session with model architecture display

#### func (*TrainingSession) StartValidation

```go
func (ts *TrainingSession) StartValidation()
```
StartValidation begins the validation phase

#### func (*TrainingSession) UpdateTrainingProgress

```go
func (ts *TrainingSession) UpdateTrainingProgress(step int, loss float64, accuracy float64)
```
UpdateTrainingProgress updates training progress

#### func (*TrainingSession) UpdateValidationProgress

```go
func (ts *TrainingSession) UpdateValidationProgress(step int, loss float64, accuracy float64)
```
UpdateValidationProgress updates validation progress

#### type TrainingStats

```go
type TrainingStats struct {
	CurrentStep     int
	BatchSize       int
	OptimizerType   cgo_bridge.OptimizerType
	LearningRate    float32
	MemoryPoolStats map[memory.PoolKey]string
}
```

TrainingStats provides training statistics
