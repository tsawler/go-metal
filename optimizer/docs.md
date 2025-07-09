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

#### func (*AdamOptimizerState) SetCommandPool

```go
func (adam *AdamOptimizerState) SetCommandPool(commandPool unsafe.Pointer)
```
SetCommandPool enables command buffer pooling for Metal operations RESOURCE LEAK
FIX: Allows Adam optimizer to use pooled command buffers

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

#### type RMSPropConfig

```go
type RMSPropConfig struct {
	LearningRate float32
	Alpha        float32
	Epsilon      float32
	WeightDecay  float32
	Momentum     float32
	Centered     bool
}
```

RMSPropConfig holds configuration for RMSProp optimizer

#### func  DefaultRMSPropConfig

```go
func DefaultRMSPropConfig() RMSPropConfig
```
DefaultRMSPropConfig returns default RMSProp optimizer configuration

#### type RMSPropOptimizerState

```go
type RMSPropOptimizerState struct {
	// Hyperparameters
	LearningRate float32
	Alpha        float32 // Smoothing constant (typically 0.99)
	Epsilon      float32 // Small constant to prevent division by zero (typically 1e-8)
	WeightDecay  float32 // L2 regularization coefficient
	Momentum     float32 // Momentum coefficient (typically 0.9, 0.0 for no momentum)
	Centered     bool    // Whether to use centered RMSProp (subtract mean of gradients)

	// GPU-resident state buffers
	SquaredGradAvgBuffers []unsafe.Pointer // Running average of squared gradients for each weight tensor
	MomentumBuffers       []unsafe.Pointer // Momentum buffers for each weight tensor (if momentum > 0)
	GradientAvgBuffers    []unsafe.Pointer // Running average of gradients for each weight tensor (if centered)
	WeightBuffers         []unsafe.Pointer // Current weight tensors

	// Step tracking
	StepCount uint64
}
```

RMSPropOptimizerState represents GPU-resident RMSProp optimizer state

#### func  NewRMSPropOptimizer

```go
func NewRMSPropOptimizer(
	config RMSPropConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*RMSPropOptimizerState, error)
```
NewRMSPropOptimizer creates a new GPU-resident RMSProp optimizer

#### func (*RMSPropOptimizerState) Cleanup

```go
func (rmsprop *RMSPropOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*RMSPropOptimizerState) GetStats

```go
func (rmsprop *RMSPropOptimizerState) GetStats() RMSPropStats
```
GetStats returns optimizer statistics

#### func (*RMSPropOptimizerState) GetStep

```go
func (rmsprop *RMSPropOptimizerState) GetStep() uint64
```
GetStep returns the current step count

#### func (*RMSPropOptimizerState) SetCommandPool

```go
func (rmsprop *RMSPropOptimizerState) SetCommandPool(commandPool unsafe.Pointer)
```
SetCommandPool enables command buffer pooling for Metal operations

#### func (*RMSPropOptimizerState) SetWeightBuffers

```go
func (rmsprop *RMSPropOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers This should be called
before each optimization step

#### func (*RMSPropOptimizerState) Step

```go
func (rmsprop *RMSPropOptimizerState) Step(gradientBuffers []unsafe.Pointer) error
```
Step performs a single RMSProp optimization step

#### func (*RMSPropOptimizerState) UpdateLearningRate

```go
func (rmsprop *RMSPropOptimizerState) UpdateLearningRate(newLR float32)
```
UpdateLearningRate updates the learning rate (useful for learning rate
scheduling)

#### type RMSPropStats

```go
type RMSPropStats struct {
	StepCount       uint64
	LearningRate    float32
	Alpha           float32
	Epsilon         float32
	WeightDecay     float32
	Momentum        float32
	Centered        bool
	NumParameters   int
	TotalBufferSize int
}
```

RMSPropStats provides statistics about the RMSProp optimizer
