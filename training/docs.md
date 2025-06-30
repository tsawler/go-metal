# training
--
    import "."


## Usage

#### func  CreateDummyWeights

```go
func CreateDummyWeights() ([]*memory.Tensor, error)
```
CreateDummyWeights creates dummy weight tensors for testing (hybrid approach)

#### func  TestPhase1

```go
func TestPhase1() error
```
TestPhase1 runs a basic test of Phase 1 implementation

#### type SimpleTrainer

```go
type SimpleTrainer struct {
}
```

SimpleTrainer provides a basic training interface for testing Phase 1

#### func  NewSimpleTrainer

```go
func NewSimpleTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
NewSimpleTrainer creates a new simple trainer

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
