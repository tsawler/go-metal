# optimizer
--
    import "."


## Usage

#### type AdamConfig

```go
type AdamConfig struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	WeightDecay  float32
}
```

AdamConfig holds configuration for Adam optimizer

#### func  DefaultAdamConfig

```go
func DefaultAdamConfig() AdamConfig
```
DefaultAdamConfig returns default Adam optimizer configuration

#### type AdamOptimizerState

```go
type AdamOptimizerState struct {
	// Hyperparameters
	LearningRate float32
	Beta1        float32 // Momentum decay (typically 0.9)
	Beta2        float32 // Variance decay (typically 0.999)
	Epsilon      float32 // Small constant to prevent division by zero (typically 1e-8)
	WeightDecay  float32 // L2 regularization coefficient

	// GPU-resident state buffers
	MomentumBuffers []unsafe.Pointer // First moment (momentum) for each weight tensor
	VarianceBuffers []unsafe.Pointer // Second moment (variance) for each weight tensor
	WeightBuffers   []unsafe.Pointer // Current weight tensors

	// Step tracking for bias correction
	StepCount uint64
}
```

AdamOptimizerState represents GPU-resident Adam optimizer state

#### func  NewAdamOptimizer

```go
func NewAdamOptimizer(
	config AdamConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*AdamOptimizerState, error)
```
NewAdamOptimizer creates a new GPU-resident Adam optimizer

#### func (*AdamOptimizerState) Cleanup

```go
func (adam *AdamOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*AdamOptimizerState) GetStats

```go
func (adam *AdamOptimizerState) GetStats() AdamStats
```
GetStats returns optimizer statistics

#### func (*AdamOptimizerState) GetStep

```go
func (adam *AdamOptimizerState) GetStep() uint64
```
GetStep returns the current step count

#### func (*AdamOptimizerState) SetWeightBuffers

```go
func (adam *AdamOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers This should be called
before each optimization step

#### func (*AdamOptimizerState) Step

```go
func (adam *AdamOptimizerState) Step(gradientBuffers []unsafe.Pointer) error
```
Step performs a single Adam optimization step This will be implemented as a
Metal compute kernel for maximum performance

#### func (*AdamOptimizerState) UpdateLearningRate

```go
func (adam *AdamOptimizerState) UpdateLearningRate(newLR float32)
```
UpdateLearningRate updates the learning rate (useful for learning rate
scheduling)

#### type AdamStats

```go
type AdamStats struct {
	StepCount       uint64
	LearningRate    float32
	Beta1           float32
	Beta2           float32
	Epsilon         float32
	WeightDecay     float32
	NumParameters   int
	TotalBufferSize int
}
```

AdamStats provides statistics about the Adam optimizer
