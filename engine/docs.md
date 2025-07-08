# engine
--
    import "."


## Usage

#### type BatchTrainer

```go
type BatchTrainer struct {
}
```

BatchTrainer provides a higher-level interface for batch training

#### func  NewBatchTrainer

```go
func NewBatchTrainer(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error)
```
NewBatchTrainer creates a new batch trainer

#### func  NewBatchTrainerConstantWeights

```go
func NewBatchTrainerConstantWeights(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error)
```
NewBatchTrainerConstantWeights creates a new batch trainer with constant weights

#### func  NewBatchTrainerHybrid

```go
func NewBatchTrainerHybrid(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error)
```
NewBatchTrainerHybrid creates a new batch trainer with hybrid MPS/MPSGraph
approach

#### func (*BatchTrainer) Cleanup

```go
func (bt *BatchTrainer) Cleanup()
```
Cleanup releases resources

#### func (*BatchTrainer) GetCurrentStep

```go
func (bt *BatchTrainer) GetCurrentStep() int
```
GetCurrentStep returns the current step number

#### func (*BatchTrainer) TrainBatch

```go
func (bt *BatchTrainer) TrainBatch(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (*TrainingStep, error)
```
TrainBatch trains on a single batch

#### func (*BatchTrainer) TrainBatchHybrid

```go
func (bt *BatchTrainer) TrainBatchHybrid(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (*TrainingStep, error)
```
TrainBatchHybrid trains on a single batch using hybrid MPS/MPSGraph approach

#### func (*BatchTrainer) TrainBatchHybridFull

```go
func (bt *BatchTrainer) TrainBatchHybridFull(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
	learningRate float32,
) (*TrainingStep, error)
```
TrainBatchHybridFull trains on a single batch using hybrid MPS/MPSGraph approach
with full training loop

#### type BatchedTrainingResult

```go
type BatchedTrainingResult struct {
	Loss     float32
	Accuracy float64 // Only valid if accuracy was calculated
}
```

BatchedTrainingResult represents the result of an optimized batched training
step

#### type MPSInferenceEngine

```go
type MPSInferenceEngine struct {
}
```

MPSInferenceEngine handles inference execution using MPSGraph Optimized for
forward-pass only without loss computation or gradients

#### func  NewMPSInferenceEngine

```go
func NewMPSInferenceEngine(config cgo_bridge.InferenceConfig) (*MPSInferenceEngine, error)
```
NewMPSInferenceEngine creates a new inference-only engine

#### func (*MPSInferenceEngine) Cleanup

```go
func (ie *MPSInferenceEngine) Cleanup()
```
Cleanup performs deterministic resource cleanup (reference counting principle)

#### type MPSTrainingEngine

```go
type MPSTrainingEngine struct {
}
```

MPSTrainingEngine handles training execution using MPSGraph

#### func  NewMPSTrainingEngine

```go
func NewMPSTrainingEngine(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error)
```
NewMPSTrainingEngine creates a new training engine

#### func  NewMPSTrainingEngineConstantWeights

```go
func NewMPSTrainingEngineConstantWeights(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error)
```
NewMPSTrainingEngineConstantWeights creates a new training engine with constant
weights This avoids the MPSGraph isStaticMPSType assertion issue with
convolution operations

#### func  NewMPSTrainingEngineHybrid

```go
func NewMPSTrainingEngineHybrid(config cgo_bridge.TrainingConfig, modelConfig cgo_bridge.ModelConfig) (*MPSTrainingEngine, error)
```
NewMPSTrainingEngineHybrid creates a new hybrid MPS/MPSGraph training engine
This uses MPS for convolution and MPSGraph for other operations, avoiding the
assertion issue

#### func  NewMPSTrainingEngineWithAdam

```go
func NewMPSTrainingEngineWithAdam(config cgo_bridge.TrainingConfig, adamConfig optimizer.AdamConfig, weightShapes [][]int) (*MPSTrainingEngine, error)
```
NewMPSTrainingEngineWithAdam creates a new hybrid training engine with Adam
optimizer

#### func (*MPSTrainingEngine) Cleanup

```go
func (e *MPSTrainingEngine) Cleanup()
```
Cleanup releases resources

#### func (*MPSTrainingEngine) ExecuteStep

```go
func (e *MPSTrainingEngine) ExecuteStep(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (float32, error)
```
ExecuteStep executes a complete training step

#### func (*MPSTrainingEngine) ExecuteStepHybrid

```go
func (e *MPSTrainingEngine) ExecuteStepHybrid(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (float32, error)
```
ExecuteStepHybrid executes a complete training step using hybrid MPS/MPSGraph
approach

#### func (*MPSTrainingEngine) ExecuteStepHybridFull

```go
func (e *MPSTrainingEngine) ExecuteStepHybridFull(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
	learningRate float32,
) (float32, error)
```
ExecuteStepHybridFull executes a complete training step with backward pass using
hybrid MPS/MPSGraph approach

#### func (*MPSTrainingEngine) ExecuteStepHybridFullWithAdam

```go
func (e *MPSTrainingEngine) ExecuteStepHybridFullWithAdam(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	weightTensors []*memory.Tensor,
) (float32, error)
```
ExecuteStepHybridFullWithAdam executes a complete training step with Adam
optimizer This performs forward pass, backward pass, and Adam optimization in
one call

#### func (*MPSTrainingEngine) ExecuteStepWithPrecomputedGradients

```go
func (e *MPSTrainingEngine) ExecuteStepWithPrecomputedGradients(
	weightTensors []*memory.Tensor,
	gradientTensors []*memory.Tensor,
) error
```
ExecuteStepWithPrecomputedGradients executes Adam optimization with pre-computed
gradients This is for cases where forward/backward pass has already been done
separately

#### func (*MPSTrainingEngine) GetAdamStats

```go
func (e *MPSTrainingEngine) GetAdamStats() *optimizer.AdamStats
```
GetAdamStats returns Adam optimizer statistics

#### func (*MPSTrainingEngine) GetConfig

```go
func (e *MPSTrainingEngine) GetConfig() cgo_bridge.TrainingConfig
```
GetConfig returns the training configuration

#### func (*MPSTrainingEngine) GetDevice

```go
func (e *MPSTrainingEngine) GetDevice() unsafe.Pointer
```
GetDevice returns the Metal device

#### func (*MPSTrainingEngine) UpdateAdamLearningRate

```go
func (e *MPSTrainingEngine) UpdateAdamLearningRate(newLR float32) error
```
UpdateAdamLearningRate updates the Adam optimizer learning rate

#### type ModelInferenceEngine

```go
type ModelInferenceEngine struct {
	*MPSInferenceEngine
}
```

ModelInferenceEngine extends MPSInferenceEngine with layer-based model support
Optimized for inference without training overhead

#### func  NewModelInferenceEngine

```go
func NewModelInferenceEngine(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.InferenceConfig,
) (*ModelInferenceEngine, error)
```
NewModelInferenceEngine creates a model-based inference engine

#### func (*ModelInferenceEngine) Cleanup

```go
func (mie *ModelInferenceEngine) Cleanup()
```
Cleanup performs complete resource cleanup

#### func (*ModelInferenceEngine) GetModelSpec

```go
func (mie *ModelInferenceEngine) GetModelSpec() *layers.ModelSpec
```
GetModelSpec returns the model specification

#### func (*ModelInferenceEngine) GetParameterTensors

```go
func (mie *ModelInferenceEngine) GetParameterTensors() []*memory.Tensor
```
GetParameterTensors returns the GPU-resident parameter tensors

#### func (*ModelInferenceEngine) LoadWeights

```go
func (mie *ModelInferenceEngine) LoadWeights(weights []checkpoints.WeightTensor) error
```
LoadWeights loads pre-trained weights into the inference engine

#### func (*ModelInferenceEngine) Predict

```go
func (mie *ModelInferenceEngine) Predict(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error)
```
Predict performs single forward pass for inference Optimized for single-image or
small batch inference

#### type ModelTrainingEngine

```go
type ModelTrainingEngine struct {
	*MPSTrainingEngine
}
```

ModelTrainingEngine extends the existing MPSTrainingEngine with layer-based
model support This maintains the proven single-CGO-call architecture while
adding layer abstraction

#### func  NewModelTrainingEngine

```go
func NewModelTrainingEngine(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
) (*ModelTrainingEngine, error)
```
NewModelTrainingEngine creates a model-based training engine This integrates
with the existing high-performance TrainingEngine architecture

#### func  NewModelTrainingEngineDynamic

```go
func NewModelTrainingEngineDynamic(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
) (*ModelTrainingEngine, error)
```
NewModelTrainingEngineDynamic creates a model-based training engine with dynamic
graph support This supports any model architecture by building the MPSGraph
dynamically from layer specs

#### func  NewModelTrainingEngineWithAdam

```go
func NewModelTrainingEngineWithAdam(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.TrainingConfig,
	adamConfig map[string]interface{},
) (*ModelTrainingEngine, error)
```
NewModelTrainingEngineWithAdam creates a model-based training engine with Adam
optimizer

#### func (*ModelTrainingEngine) Cleanup

```go
func (mte *ModelTrainingEngine) Cleanup()
```
Cleanup releases all resources including model parameters

#### func (*ModelTrainingEngine) ExecuteInference

```go
func (mte *ModelTrainingEngine) ExecuteInference(
	inputTensor *memory.Tensor,
	batchSize int,
) (*cgo_bridge.InferenceResult, error)
```
ExecuteInference performs forward-only pass returning model predictions Conforms
to design requirements: single CGO call, GPU-resident, shared resources

#### func (*ModelTrainingEngine) ExecuteModelTrainingStep

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStep(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	learningRate float32,
) (float32, error)
```
ExecuteModelTrainingStep executes a complete model training step This maintains
the single-CGO-call principle while supporting flexible layer models

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepBatched

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepBatched(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	inputData []float32,
	labelData []float32,
	calculateAccuracy bool,
) (*BatchedTrainingResult, error)
```
ExecuteModelTrainingStepBatched executes a complete training step with batched
CGO operations This reduces CGO overhead by combining multiple operations into a
single call Follows design principle: "Single CGO call per training step"

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepBatchedPersistent

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepBatchedPersistent(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	inputData []float32,
	labelData []float32,
	calculateAccuracy bool,
) (*BatchedTrainingResult, error)
```
ExecuteModelTrainingStepBatchedPersistent executes a training step using
persistent GPU buffers This provides maximum performance by eliminating per-step
tensor allocations

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepBatchedPersistentWithGradients

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepBatchedPersistentWithGradients(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
	inputData []float32,
	labelData []float32,
	calculateAccuracy bool,
	persistentGradientTensors []*memory.Tensor,
) (*BatchedTrainingResult, error)
```
ExecuteModelTrainingStepBatchedPersistentWithGradients executes training with
pre-allocated gradient tensors This eliminates the 128MB/step gradient
allocation that caused 83% performance degradation

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepWithAdam

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithAdam(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error)
```
ExecuteModelTrainingStepWithAdam executes model training with Adam optimizer

#### func (*ModelTrainingEngine) GetModelSpec

```go
func (mte *ModelTrainingEngine) GetModelSpec() *layers.ModelSpec
```
GetModelSpec returns the model specification

#### func (*ModelTrainingEngine) GetModelSummary

```go
func (mte *ModelTrainingEngine) GetModelSummary() string
```
GetModelSummary returns a human-readable model summary

#### func (*ModelTrainingEngine) GetParameterTensors

```go
func (mte *ModelTrainingEngine) GetParameterTensors() []*memory.Tensor
```
GetParameterTensors returns all model parameter tensors

#### func (*ModelTrainingEngine) IsDynamicEngine

```go
func (mte *ModelTrainingEngine) IsDynamicEngine() bool
```
IsDynamicEngine returns whether this is a dynamic engine

#### type TrainingStep

```go
type TrainingStep struct {
	Loss      float32
	BatchSize int
	StepTime  int64 // Nanoseconds
	Success   bool
}
```

TrainingStep represents a single training step result
