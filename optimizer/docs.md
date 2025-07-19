# optimizer
--
    import "."


## Usage

#### type AdaDeltaConfig

```go
type AdaDeltaConfig struct {
	Rho         float32 // Decay rate for moving averages (typically 0.95)
	Epsilon     float32 // Small constant for numerical stability
	WeightDecay float32 // L2 regularization strength
}
```

AdaDeltaConfig holds configuration for AdaDelta optimizer

#### func  DefaultAdaDeltaConfig

```go
func DefaultAdaDeltaConfig() AdaDeltaConfig
```
DefaultAdaDeltaConfig returns default AdaDelta optimizer configuration

#### type AdaDeltaOptimizerState

```go
type AdaDeltaOptimizerState struct {
	WeightBuffers []unsafe.Pointer // Current weight tensors
}
```

AdaDeltaOptimizerState represents GPU-resident AdaDelta optimizer state

#### func  NewAdaDeltaOptimizer

```go
func NewAdaDeltaOptimizer(
	config AdaDeltaConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*AdaDeltaOptimizerState, error)
```
NewAdaDeltaOptimizer creates a new GPU-resident AdaDelta optimizer

#### func (*AdaDeltaOptimizerState) Cleanup

```go
func (adadelta *AdaDeltaOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*AdaDeltaOptimizerState) GetState

```go
func (adadelta *AdaDeltaOptimizerState) GetState() (*OptimizerState, error)
```
GetState extracts optimizer state for checkpointing Transfers GPU state to CPU
in a single batched operation per buffer type

#### func (*AdaDeltaOptimizerState) GetStats

```go
func (adadelta *AdaDeltaOptimizerState) GetStats() map[string]interface{}
```
GetStats returns optimizer statistics

#### func (*AdaDeltaOptimizerState) GetStep

```go
func (adadelta *AdaDeltaOptimizerState) GetStep() uint64
```
GetStep returns the current optimization step count

#### func (*AdaDeltaOptimizerState) GetStepCount

```go
func (adadelta *AdaDeltaOptimizerState) GetStepCount() uint64
```
GetStepCount returns the current optimization step number

#### func (*AdaDeltaOptimizerState) LoadState

```go
func (adadelta *AdaDeltaOptimizerState) LoadState(state *OptimizerState) error
```
LoadState restores optimizer state from checkpoint Transfers CPU state back to
GPU in batched operations

#### func (*AdaDeltaOptimizerState) SetCommandPool

```go
func (adadelta *AdaDeltaOptimizerState) SetCommandPool(pool unsafe.Pointer)
```
SetCommandPool sets the command buffer pool for Metal operations

#### func (*AdaDeltaOptimizerState) SetWeightBuffers

```go
func (adadelta *AdaDeltaOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers

#### func (*AdaDeltaOptimizerState) Step

```go
func (adadelta *AdaDeltaOptimizerState) Step(gradientBuffers []unsafe.Pointer) error
```
Step performs a single AdaDelta optimization step

#### func (*AdaDeltaOptimizerState) UpdateLearningRate

```go
func (adadelta *AdaDeltaOptimizerState) UpdateLearningRate(newLR float32)
```
UpdateLearningRate is not used in AdaDelta (it adapts automatically)

#### type AdaGradConfig

```go
type AdaGradConfig struct {
	LearningRate float32 // Learning rate
	Epsilon      float32 // Small constant for numerical stability
	WeightDecay  float32 // L2 regularization strength
}
```

AdaGradConfig holds configuration for AdaGrad optimizer

#### func  DefaultAdaGradConfig

```go
func DefaultAdaGradConfig() AdaGradConfig
```
DefaultAdaGradConfig returns default AdaGrad optimizer configuration

#### type AdaGradOptimizerState

```go
type AdaGradOptimizerState struct {
	WeightBuffers []unsafe.Pointer // Current weight tensors
}
```

AdaGradOptimizerState represents GPU-resident AdaGrad optimizer state

#### func  NewAdaGradOptimizer

```go
func NewAdaGradOptimizer(
	config AdaGradConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*AdaGradOptimizerState, error)
```
NewAdaGradOptimizer creates a new GPU-resident AdaGrad optimizer

#### func (*AdaGradOptimizerState) Cleanup

```go
func (adagrad *AdaGradOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*AdaGradOptimizerState) GetState

```go
func (adagrad *AdaGradOptimizerState) GetState() (*OptimizerState, error)
```
GetState extracts optimizer state for checkpointing Transfers GPU state to CPU
in a single batched operation per buffer type

#### func (*AdaGradOptimizerState) GetStats

```go
func (adagrad *AdaGradOptimizerState) GetStats() map[string]interface{}
```
GetStats returns optimizer statistics

#### func (*AdaGradOptimizerState) GetStep

```go
func (adagrad *AdaGradOptimizerState) GetStep() uint64
```
GetStep returns the current optimization step count

#### func (*AdaGradOptimizerState) GetStepCount

```go
func (adagrad *AdaGradOptimizerState) GetStepCount() uint64
```
GetStepCount returns the current optimization step number

#### func (*AdaGradOptimizerState) LoadState

```go
func (adagrad *AdaGradOptimizerState) LoadState(state *OptimizerState) error
```
LoadState restores optimizer state from checkpoint Transfers CPU state back to
GPU in batched operations

#### func (*AdaGradOptimizerState) SetCommandPool

```go
func (adagrad *AdaGradOptimizerState) SetCommandPool(pool unsafe.Pointer)
```
SetCommandPool sets the command buffer pool for Metal operations

#### func (*AdaGradOptimizerState) SetWeightBuffers

```go
func (adagrad *AdaGradOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers

#### func (*AdaGradOptimizerState) Step

```go
func (adagrad *AdaGradOptimizerState) Step(gradientBuffers []unsafe.Pointer) error
```
Step performs a single AdaGrad optimization step

#### func (*AdaGradOptimizerState) UpdateLearningRate

```go
func (adagrad *AdaGradOptimizerState) UpdateLearningRate(newLR float32)
```
UpdateLearningRate updates the learning rate for the optimizer

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

#### func (*AdamOptimizerState) GetState

```go
func (adam *AdamOptimizerState) GetState() (*OptimizerState, error)
```
GetState extracts optimizer state for checkpointing Transfers GPU state to CPU
in a single batched operation per buffer type

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

#### func (*AdamOptimizerState) LoadState

```go
func (adam *AdamOptimizerState) LoadState(state *OptimizerState) error
```
LoadState restores optimizer state from checkpoint Transfers CPU state back to
GPU in batched operations

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

#### type LBFGSConfig

```go
type LBFGSConfig struct {
	HistorySize   int     // m parameter (number of corrections to store)
	LineSearchTol float32 // Tolerance for line search
	MaxLineSearch int     // Maximum line search iterations
	C1            float32 // Armijo condition parameter
	C2            float32 // Wolfe condition parameter
	InitialStep   float32 // Initial step size for line search
}
```

LBFGSConfig holds configuration for L-BFGS optimizer

#### func  DefaultLBFGSConfig

```go
func DefaultLBFGSConfig() LBFGSConfig
```
DefaultLBFGSConfig returns default L-BFGS optimizer configuration

#### type LBFGSOptimizerState

```go
type LBFGSOptimizerState struct {
	WeightBuffers []unsafe.Pointer // Current weight tensors
}
```

LBFGSOptimizerState represents GPU-resident L-BFGS optimizer state

#### func  NewLBFGSOptimizer

```go
func NewLBFGSOptimizer(
	config LBFGSConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*LBFGSOptimizerState, error)
```
NewLBFGSOptimizer creates a new GPU-resident L-BFGS optimizer

#### func (*LBFGSOptimizerState) Cleanup

```go
func (lbfgs *LBFGSOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*LBFGSOptimizerState) GetStats

```go
func (lbfgs *LBFGSOptimizerState) GetStats() map[string]interface{}
```
GetStats returns optimizer statistics

#### func (*LBFGSOptimizerState) GetStep

```go
func (lbfgs *LBFGSOptimizerState) GetStep() uint64
```
GetStep returns the current optimization step count

#### func (*LBFGSOptimizerState) SetCommandPool

```go
func (lbfgs *LBFGSOptimizerState) SetCommandPool(pool unsafe.Pointer)
```
SetCommandPool sets the command buffer pool for Metal operations

#### func (*LBFGSOptimizerState) SetWeightBuffers

```go
func (lbfgs *LBFGSOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers

#### func (*LBFGSOptimizerState) Step

```go
func (lbfgs *LBFGSOptimizerState) Step(gradientBuffers []unsafe.Pointer, currentLoss float32) error
```
Step performs a single L-BFGS optimization step

#### func (*LBFGSOptimizerState) UpdateLearningRate

```go
func (lbfgs *LBFGSOptimizerState) UpdateLearningRate(newLR float32) error
```
UpdateLearningRate is not used in L-BFGS (uses line search instead)

#### type NadamConfig

```go
type NadamConfig struct {
	LearningRate float32 // Base learning rate (typically 0.002)
	Beta1        float32 // Exponential decay rate for first moment estimates (typically 0.9)
	Beta2        float32 // Exponential decay rate for second moment estimates (typically 0.999)
	Epsilon      float32 // Small constant for numerical stability (typically 1e-8)
	WeightDecay  float32 // L2 regularization coefficient (typically 0.0)
}
```

NadamConfig holds configuration for Nadam optimizer

#### func  DefaultNadamConfig

```go
func DefaultNadamConfig() NadamConfig
```
DefaultNadamConfig returns default Nadam optimizer configuration

#### type NadamOptimizerState

```go
type NadamOptimizerState struct {
	WeightBuffers []unsafe.Pointer // Current weight tensors
}
```

NadamOptimizerState represents GPU-resident Nadam optimizer state Nadam combines
Adam's adaptive learning rates with Nesterov momentum

#### func  NewNadamOptimizer

```go
func NewNadamOptimizer(
	config NadamConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*NadamOptimizerState, error)
```
NewNadamOptimizer creates a new GPU-resident Nadam optimizer

#### func (*NadamOptimizerState) Cleanup

```go
func (nadam *NadamOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*NadamOptimizerState) GetState

```go
func (nadam *NadamOptimizerState) GetState() (*OptimizerState, error)
```
GetState extracts optimizer state for checkpointing Transfers GPU state to CPU
in a single batched operation per buffer type

#### func (*NadamOptimizerState) GetStats

```go
func (nadam *NadamOptimizerState) GetStats() map[string]interface{}
```
GetStats returns optimizer statistics as a map for generic access

#### func (*NadamOptimizerState) GetStep

```go
func (nadam *NadamOptimizerState) GetStep() uint64
```
GetStep returns the current step count

#### func (*NadamOptimizerState) GetStepCount

```go
func (nadam *NadamOptimizerState) GetStepCount() uint64
```
GetStepCount returns the current optimization step number

#### func (*NadamOptimizerState) LoadState

```go
func (nadam *NadamOptimizerState) LoadState(state *OptimizerState) error
```
LoadState restores optimizer state from checkpoint Transfers CPU state back to
GPU in batched operations

#### func (*NadamOptimizerState) SetCommandPool

```go
func (nadam *NadamOptimizerState) SetCommandPool(commandPool unsafe.Pointer)
```
SetCommandPool enables command buffer pooling for Metal operations

#### func (*NadamOptimizerState) SetWeightBuffers

```go
func (nadam *NadamOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers This should be called
before each optimization step

#### func (*NadamOptimizerState) Step

```go
func (nadam *NadamOptimizerState) Step(gradientBuffers []unsafe.Pointer) error
```
Step performs a single Nadam optimization step Nadam combines Adam's adaptive
learning rates with Nesterov momentum

#### func (*NadamOptimizerState) UpdateLearningRate

```go
func (nadam *NadamOptimizerState) UpdateLearningRate(newLR float32)
```
UpdateLearningRate updates the learning rate (useful for learning rate
scheduling)

#### type Optimizer

```go
type Optimizer interface {
	// Step performs a single optimization step
	// gradientBuffers must match the number of weight buffers
	Step(gradientBuffers []unsafe.Pointer) error

	// GetState extracts optimizer state for checkpointing
	// This transfers GPU state to CPU only when needed for saving
	// The implementation should batch reads to minimize CGO calls
	GetState() (*OptimizerState, error)

	// LoadState restores optimizer state from checkpoint
	// This transfers CPU state back to GPU buffers
	// The implementation should batch writes to minimize CGO calls
	LoadState(state *OptimizerState) error

	// GetStepCount returns the current optimization step number
	GetStepCount() uint64

	// UpdateLearningRate updates the learning rate
	UpdateLearningRate(lr float32)

	// SetWeightBuffers sets the current weight buffer pointers
	// Must be called before each optimization step
	SetWeightBuffers(weightBuffers []unsafe.Pointer) error

	// SetCommandPool enables command buffer pooling for Metal operations
	SetCommandPool(commandPool unsafe.Pointer)

	// Cleanup releases all GPU resources
	Cleanup()
}
```

Optimizer defines the common interface for all optimizers This interface enables
state save/restore for checkpoint functionality while maintaining GPU-resident
state and minimizing CGO calls

#### type OptimizerState

```go
type OptimizerState struct {
	Type       string                        `json:"type"`       // "Adam", "SGD", etc.
	Parameters map[string]interface{}        `json:"parameters"` // Hyperparameters
	StateData  []checkpoints.OptimizerTensor `json:"state_data"` // GPU state tensors
}
```

OptimizerState represents the complete state of an optimizer Compatible with
checkpoints.OptimizerState for serialization

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

#### func (*RMSPropOptimizerState) GetState

```go
func (rmsprop *RMSPropOptimizerState) GetState() (*OptimizerState, error)
```
GetState extracts optimizer state for checkpointing Transfers GPU state to CPU
in batched operations

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

#### func (*RMSPropOptimizerState) LoadState

```go
func (rmsprop *RMSPropOptimizerState) LoadState(state *OptimizerState) error
```
LoadState restores optimizer state from checkpoint

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

#### type SGDConfig

```go
type SGDConfig struct {
	LearningRate float32
	Momentum     float32
	WeightDecay  float32
	Nesterov     bool
}
```

SGDConfig holds configuration for SGD optimizer

#### func  DefaultSGDConfig

```go
func DefaultSGDConfig() SGDConfig
```
DefaultSGDConfig returns default SGD optimizer configuration

#### type SGDOptimizerState

```go
type SGDOptimizerState struct {
	// Hyperparameters
	LearningRate float32
	Momentum     float32 // Momentum coefficient (0 for vanilla SGD)
	WeightDecay  float32 // L2 regularization coefficient
	Nesterov     bool    // Whether to use Nesterov momentum

	// GPU-resident state buffers
	MomentumBuffers []unsafe.Pointer // Momentum buffers (only if momentum > 0)
	WeightBuffers   []unsafe.Pointer // Current weight tensors

	// Step tracking
	StepCount uint64
}
```

SGDOptimizerState represents GPU-resident SGD optimizer state

#### func  NewSGDOptimizer

```go
func NewSGDOptimizer(
	config SGDConfig,
	weightShapes [][]int,
	memoryManager *memory.MemoryManager,
	device unsafe.Pointer,
) (*SGDOptimizerState, error)
```
NewSGDOptimizer creates a new GPU-resident SGD optimizer

#### func (*SGDOptimizerState) Cleanup

```go
func (sgd *SGDOptimizerState) Cleanup()
```
Cleanup releases all GPU buffers

#### func (*SGDOptimizerState) GetState

```go
func (sgd *SGDOptimizerState) GetState() (*OptimizerState, error)
```
GetState extracts optimizer state for checkpointing

#### func (*SGDOptimizerState) GetStep

```go
func (sgd *SGDOptimizerState) GetStep() uint64
```
GetStep returns the current optimization step count

#### func (*SGDOptimizerState) GetStepCount

```go
func (sgd *SGDOptimizerState) GetStepCount() uint64
```
GetStepCount returns the current step count

#### func (*SGDOptimizerState) LoadState

```go
func (sgd *SGDOptimizerState) LoadState(state *OptimizerState) error
```
LoadState restores optimizer state from checkpoint

#### func (*SGDOptimizerState) SetCommandPool

```go
func (sgd *SGDOptimizerState) SetCommandPool(commandPool unsafe.Pointer)
```
SetCommandPool enables command buffer pooling

#### func (*SGDOptimizerState) SetWeightBuffers

```go
func (sgd *SGDOptimizerState) SetWeightBuffers(weightBuffers []unsafe.Pointer) error
```
SetWeightBuffers sets the current weight buffer pointers

#### func (*SGDOptimizerState) Step

```go
func (sgd *SGDOptimizerState) Step(gradientBuffers []unsafe.Pointer) error
```
Step performs a single SGD optimization step

#### func (*SGDOptimizerState) UpdateLearningRate

```go
func (sgd *SGDOptimizerState) UpdateLearningRate(newLR float32)
```
UpdateLearningRate updates the learning rate
