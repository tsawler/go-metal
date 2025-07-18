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
NewBatchTrainer creates a new batch trainer DEPRECATED: Use
NewModelTrainingEngineDynamic with proper layer specifications instead. This
function is kept for backward compatibility but should not be used in new code.

#### func  NewBatchTrainerConstantWeights

```go
func NewBatchTrainerConstantWeights(config cgo_bridge.TrainingConfig, batchSize int) (*BatchTrainer, error)
```
NewBatchTrainerConstantWeights creates a new batch trainer with constant weights
DEPRECATED: Use NewModelTrainingEngineDynamic with proper layer specifications
instead. This function is kept for backward compatibility but should not be used
in new code.

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
forward-pass only without loss computation or gradients Works with any model
architecture supported by the dynamic training engine

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
NewMPSTrainingEngine creates a new training engine DEPRECATED: Use
NewModelTrainingEngineDynamic with proper layer specifications instead. This
function is kept for backward compatibility but should not be used in new code.

#### func  NewMPSTrainingEngineConstantWeights

```go
func NewMPSTrainingEngineConstantWeights(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error)
```
NewMPSTrainingEngineConstantWeights creates a new training engine with constant
weights DEPRECATED: Use NewModelTrainingEngineDynamic with proper layer
specifications instead. This function is kept for backward compatibility but
should not be used in new code.

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

#### func (*MPSTrainingEngine) GetLBFGSStats

```go
func (e *MPSTrainingEngine) GetLBFGSStats() map[string]interface{}
```
GetLBFGSStats returns L-BFGS optimizer statistics

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
NewModelInferenceEngine creates a model-based inference engine Works with any
model architecture supported by the dynamic training engine

#### func  NewModelInferenceEngineFromDynamicTraining

```go
func NewModelInferenceEngineFromDynamicTraining(
	modelSpec *layers.ModelSpec,
	config cgo_bridge.InferenceConfig,
) (*ModelInferenceEngine, error)
```
NewModelInferenceEngineFromDynamicTraining creates an inference engine from a
model trained with dynamic training engine This ensures full compatibility with
any architecture supported by the dynamic training engine

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

#### func (*ModelInferenceEngine) ListBatchNormLayers

```go
func (mie *ModelInferenceEngine) ListBatchNormLayers() []string
```
ListBatchNormLayers returns the names of all BatchNorm layers in the model This
helps users identify which layers can have custom normalization

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
Predict performs single forward pass for inference Supports any model
architecture and input dimensionality

#### func (*ModelInferenceEngine) SetCustomNormalization

```go
func (mie *ModelInferenceEngine) SetCustomNormalization(layerName string, mean, variance []float32) error
```
SetCustomNormalization allows setting custom normalization values for any model
This enables the library to work with ANY model architecture and normalization
scheme layerName: name of the BatchNorm layer to update mean, variance: custom
normalization values (must match layer's num_features)

#### func (*ModelInferenceEngine) SetStandardNormalization

```go
func (mie *ModelInferenceEngine) SetStandardNormalization(layerName string) error
```
SetStandardNormalization sets standard normalization (mean=0, var=1) for a
BatchNorm layer This is equivalent to the old hardcoded behavior but now
configurable

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
NewModelTrainingEngine creates a model-based training engine using dynamic
engine by default This integrates with the high-performance dynamic
TrainingEngine architecture

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

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepWithAdaDelta

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithAdaDelta(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error)
```
ExecuteModelTrainingStepWithAdaDelta executes model training with AdaDelta
optimizer

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepWithAdaGrad

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithAdaGrad(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error)
```
ExecuteModelTrainingStepWithAdaGrad executes model training with AdaGrad
optimizer

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepWithAdam

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithAdam(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error)
```
ExecuteModelTrainingStepWithAdam executes model training with Adam optimizer

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepWithLBFGS

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithLBFGS(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error)
```
ExecuteModelTrainingStepWithLBFGS executes model training with L-BFGS optimizer

#### func (*ModelTrainingEngine) ExecuteModelTrainingStepWithNadam

```go
func (mte *ModelTrainingEngine) ExecuteModelTrainingStepWithNadam(
	inputTensor *memory.Tensor,
	labelTensor *memory.Tensor,
) (float32, error)
```
ExecuteModelTrainingStepWithNadam executes model training with Nadam optimizer

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

#### func (*ModelTrainingEngine) GetOptimizerState

```go
func (mte *ModelTrainingEngine) GetOptimizerState() (*optimizer.OptimizerState, error)
```
GetOptimizerState extracts the current optimizer state for checkpointing This
method bridges between the CGO-level optimizer and the Go optimizer interface

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

#### func (*ModelTrainingEngine) SetOptimizerState

```go
func (mte *ModelTrainingEngine) SetOptimizerState(state *optimizer.OptimizerState) error
```
SetOptimizerState restores optimizer state from a checkpoint This method bridges
between the Go optimizer interface and the CGO-level optimizer

#### func (*ModelTrainingEngine) UpdateLearningRate

```go
func (mte *ModelTrainingEngine) UpdateLearningRate(newLR float32) error
```
UpdateLearningRate updates the learning rate for the current optimizer This
method bridges between the model trainer and the optimizer implementations

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
