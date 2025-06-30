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
func NewMPSTrainingEngineHybrid(config cgo_bridge.TrainingConfig) (*MPSTrainingEngine, error)
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
