# training
--
    import "."


## Usage

#### func  SetRandomSeed

```go
func SetRandomSeed(seed int64)
```
SetRandomSeed sets the global random seed for deterministic weight
initialization

#### type Adam

```go
type Adam struct {
}
```

Adam implements the Adam optimizer

#### func  NewAdam

```go
func NewAdam(parameters []*tensor.Tensor, lr, beta1, beta2, eps, weightDecay float64) *Adam
```
NewAdam creates a new Adam optimizer

#### func (*Adam) GetLR

```go
func (adam *Adam) GetLR() float64
```
GetLR returns the current learning rate

#### func (*Adam) SetLR

```go
func (adam *Adam) SetLR(lr float64)
```
SetLR sets the learning rate

#### func (*Adam) Step

```go
func (adam *Adam) Step() error
```
Step performs a single optimization step

#### func (*Adam) ZeroGrad

```go
func (adam *Adam) ZeroGrad()
```
ZeroGrad resets gradients to zero for all parameters

#### type Batch

```go
type Batch struct {
	Data   *tensor.Tensor
	Labels *tensor.Tensor
}
```

Batch represents a batch of data and labels

#### type BatchNorm

```go
type BatchNorm struct {
}
```

BatchNorm implements Batch Normalization layer

#### func  NewBatchNorm

```go
func NewBatchNorm(numFeatures int, eps, momentum float64, device tensor.DeviceType) (*BatchNorm, error)
```
NewBatchNorm creates a new Batch Normalization layer

#### func (*BatchNorm) Eval

```go
func (bn *BatchNorm) Eval()
```
Eval sets the module to evaluation mode

#### func (*BatchNorm) Forward

```go
func (bn *BatchNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs batch normalization

#### func (*BatchNorm) IsTraining

```go
func (bn *BatchNorm) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*BatchNorm) Parameters

```go
func (bn *BatchNorm) Parameters() []*tensor.Tensor
```
Parameters returns the trainable parameters

#### func (*BatchNorm) Train

```go
func (bn *BatchNorm) Train()
```
Train sets the module to training mode

#### type Conv2D

```go
type Conv2D struct {
}
```

Conv2D implements a 2D convolution layer

#### func  NewConv2D

```go
func NewConv2D(inputChannels, outputChannels, kernelSize, stride, padding int, bias bool, device tensor.DeviceType) (*Conv2D, error)
```
NewConv2D creates a new Conv2D layer

#### func (*Conv2D) Eval

```go
func (c *Conv2D) Eval()
```
Eval sets the module to evaluation mode

#### func (*Conv2D) Forward

```go
func (c *Conv2D) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs 2D convolution

#### func (*Conv2D) IsTraining

```go
func (c *Conv2D) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*Conv2D) Parameters

```go
func (c *Conv2D) Parameters() []*tensor.Tensor
```
Parameters returns the trainable parameters

#### func (*Conv2D) Train

```go
func (c *Conv2D) Train()
```
Train sets the module to training mode

#### type CrossEntropyLoss

```go
type CrossEntropyLoss struct {
}
```

CrossEntropyLoss implements Cross Entropy loss function for classification

#### func  NewCrossEntropyLoss

```go
func NewCrossEntropyLoss(reduction string) *CrossEntropyLoss
```
NewCrossEntropyLoss creates a new Cross Entropy loss function

#### func (*CrossEntropyLoss) Backward

```go
func (ce *CrossEntropyLoss) Backward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
```
Backward computes the gradient of Cross Entropy loss

#### func (*CrossEntropyLoss) Forward

```go
func (ce *CrossEntropyLoss) Forward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
```
Forward computes the Cross Entropy loss using autograd operations predicted:
[batch_size, num_classes] logits target: [batch_size] class indices

#### type DataLoader

```go
type DataLoader struct {
}
```

DataLoader provides batching, shuffling, and efficient data loading

#### func  NewDataLoader

```go
func NewDataLoader(dataset Dataset, batchSize int, shuffle bool, numWorkers int, device tensor.DeviceType) *DataLoader
```
NewDataLoader creates a new DataLoader

#### func (*DataLoader) HasNext

```go
func (dl *DataLoader) HasNext() bool
```
HasNext returns true if there are more batches in the current epoch

#### func (*DataLoader) Iterator

```go
func (dl *DataLoader) Iterator() <-chan *Batch
```
Iterator returns a channel-based iterator for easy use in training loops

#### func (*DataLoader) Len

```go
func (dl *DataLoader) Len() int
```
Len returns the number of batches in an epoch

#### func (*DataLoader) Next

```go
func (dl *DataLoader) Next() (*Batch, error)
```
Next returns the next batch or nil if epoch is complete

#### func (*DataLoader) Reset

```go
func (dl *DataLoader) Reset()
```
Reset resets the data loader for a new epoch

#### type Dataset

```go
type Dataset interface {
	Len() int                                                           // Total number of samples
	Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error) // Returns a single sample (CPU Tensor initially)
}
```

Dataset interface defines methods that all datasets must implement

#### type Flatten

```go
type Flatten struct {
}
```

Flatten reshapes input tensor to [batch_size, -1]

#### func  NewFlatten

```go
func NewFlatten() *Flatten
```
NewFlatten creates a new Flatten layer

#### func (*Flatten) Eval

```go
func (f *Flatten) Eval()
```
Eval sets the module to evaluation mode

#### func (*Flatten) Forward

```go
func (f *Flatten) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward flattens the input tensor to [batch_size, -1]

#### func (*Flatten) IsTraining

```go
func (f *Flatten) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*Flatten) Parameters

```go
func (f *Flatten) Parameters() []*tensor.Tensor
```
Parameters returns empty slice (Flatten has no parameters)

#### func (*Flatten) Train

```go
func (f *Flatten) Train()
```
Train sets the module to training mode

#### type Linear

```go
type Linear struct {
}
```

Linear implements a fully connected (dense) layer: y = xW^T + b

#### func  NewLinear

```go
func NewLinear(inputSize, outputSize int, bias bool, device tensor.DeviceType) (*Linear, error)
```
NewLinear creates a new Linear layer

#### func (*Linear) Eval

```go
func (l *Linear) Eval()
```
Eval sets the module to evaluation mode

#### func (*Linear) Forward

```go
func (l *Linear) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs the forward pass: y = xW^T + b

#### func (*Linear) IsTraining

```go
func (l *Linear) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*Linear) Parameters

```go
func (l *Linear) Parameters() []*tensor.Tensor
```
Parameters returns the trainable parameters

#### func (*Linear) Train

```go
func (l *Linear) Train()
```
Train sets the module to training mode

#### type Loss

```go
type Loss interface {
	Forward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
	Backward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
}
```

Loss interface defines methods that all loss functions must implement

#### type MSELoss

```go
type MSELoss struct {
}
```

MSELoss implements Mean Squared Error loss function

#### func  NewMSELoss

```go
func NewMSELoss(reduction string) *MSELoss
```
NewMSELoss creates a new Mean Squared Error loss function

#### func (*MSELoss) Backward

```go
func (mse *MSELoss) Backward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
```
Backward computes the gradient of MSE loss

#### func (*MSELoss) Forward

```go
func (mse *MSELoss) Forward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
```
Forward computes the MSE loss: L = (1/N) * sum((y_pred - y_true)^2)

#### type MaxPool2D

```go
type MaxPool2D struct {
}
```

MaxPool2D implements a 2D max pooling layer

#### func  NewMaxPool2D

```go
func NewMaxPool2D(kernelSize, stride, padding int) *MaxPool2D
```
NewMaxPool2D creates a new MaxPool2D layer

#### func (*MaxPool2D) Eval

```go
func (m *MaxPool2D) Eval()
```
Eval sets the module to evaluation mode

#### func (*MaxPool2D) Forward

```go
func (m *MaxPool2D) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs 2D max pooling

#### func (*MaxPool2D) IsTraining

```go
func (m *MaxPool2D) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*MaxPool2D) Parameters

```go
func (m *MaxPool2D) Parameters() []*tensor.Tensor
```
Parameters returns empty slice (MaxPool2D has no parameters)

#### func (*MaxPool2D) Train

```go
func (m *MaxPool2D) Train()
```
Train sets the module to training mode

#### type Module

```go
type Module interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*tensor.Tensor // Returns trainable parameters (tensors with requiresGrad=true)
	Train()                       // Sets module to training mode
	Eval()                        // Sets module to evaluation mode
	IsTraining() bool             // Returns true if in training mode
}
```

Module interface defines methods that all neural network layers must implement

#### type Optimizer

```go
type Optimizer interface {
	Step() error      // Updates model parameters based on gradients
	ZeroGrad()        // Resets gradients to zero for all parameters
	GetLR() float64   // Gets current learning rate
	SetLR(lr float64) // Sets learning rate
}
```

Optimizer interface defines the methods that all optimizers must implement

#### type RandomDataset

```go
type RandomDataset struct {
}
```

RandomDataset generates random data for testing purposes

#### func  NewRandomDataset

```go
func NewRandomDataset(size int, dataShape []int, labelShape []int, dataType, labelType tensor.DType, numClasses int) *RandomDataset
```
NewRandomDataset creates a new RandomDataset

#### func (*RandomDataset) Get

```go
func (rd *RandomDataset) Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error)
```
Get generates a random sample

#### func (*RandomDataset) Len

```go
func (rd *RandomDataset) Len() int
```
Len returns the size of the dataset

#### type ReLU

```go
type ReLU struct {
}
```

ReLU implements ReLU activation function module

#### func  NewReLU

```go
func NewReLU() *ReLU
```
NewReLU creates a new ReLU activation module

#### func (*ReLU) Eval

```go
func (r *ReLU) Eval()
```
Eval sets the module to evaluation mode

#### func (*ReLU) Forward

```go
func (r *ReLU) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward performs ReLU activation

#### func (*ReLU) IsTraining

```go
func (r *ReLU) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*ReLU) Parameters

```go
func (r *ReLU) Parameters() []*tensor.Tensor
```
Parameters returns empty slice (ReLU has no parameters)

#### func (*ReLU) Train

```go
func (r *ReLU) Train()
```
Train sets the module to training mode

#### type SGD

```go
type SGD struct {
}
```

SGD implements Stochastic Gradient Descent optimizer

#### func  NewSGD

```go
func NewSGD(parameters []*tensor.Tensor, lr float64, momentum float64, weightDecay float64, dampening float64, nesterov bool) *SGD
```
NewSGD creates a new SGD optimizer

#### func (*SGD) GetLR

```go
func (sgd *SGD) GetLR() float64
```
GetLR returns the current learning rate

#### func (*SGD) SetLR

```go
func (sgd *SGD) SetLR(lr float64)
```
SetLR sets the learning rate

#### func (*SGD) Step

```go
func (sgd *SGD) Step() error
```
Step performs a single optimization step

#### func (*SGD) ZeroGrad

```go
func (sgd *SGD) ZeroGrad()
```
ZeroGrad resets gradients to zero for all parameters

#### type Sequential

```go
type Sequential struct {
}
```

Sequential allows chaining multiple modules together

#### func  NewSequential

```go
func NewSequential(modules ...Module) *Sequential
```
NewSequential creates a new Sequential container

#### func (*Sequential) Add

```go
func (s *Sequential) Add(module Module)
```
Add appends a module to the sequential container

#### func (*Sequential) Eval

```go
func (s *Sequential) Eval()
```
Eval sets all modules to evaluation mode

#### func (*Sequential) Forward

```go
func (s *Sequential) Forward(input *tensor.Tensor) (*tensor.Tensor, error)
```
Forward passes input through all modules in sequence

#### func (*Sequential) IsTraining

```go
func (s *Sequential) IsTraining() bool
```
IsTraining returns true if in training mode

#### func (*Sequential) Parameters

```go
func (s *Sequential) Parameters() []*tensor.Tensor
```
Parameters returns all trainable parameters from all modules

#### func (*Sequential) Train

```go
func (s *Sequential) Train()
```
Train sets all modules to training mode

#### type SimpleDataset

```go
type SimpleDataset struct {
}
```

SimpleDataset provides a basic implementation of Dataset for testing and simple
use cases

#### func  NewSimpleDataset

```go
func NewSimpleDataset(data, labels []*tensor.Tensor) (*SimpleDataset, error)
```
NewSimpleDataset creates a new SimpleDataset

#### func (*SimpleDataset) Get

```go
func (ds *SimpleDataset) Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error)
```
Get returns a sample at the given index

#### func (*SimpleDataset) Len

```go
func (ds *SimpleDataset) Len() int
```
Len returns the number of samples in the dataset

#### type SubsetDataset

```go
type SubsetDataset struct {
}
```

SubsetDataset allows training on a limited number of samples from an underlying
dataset.

#### func  NewSubsetDataset

```go
func NewSubsetDataset(original Dataset, limit int) (*SubsetDataset, error)
```
NewSubsetDataset creates a new SubsetDataset that wraps an existing dataset and
limits the number of samples it exposes.

#### func (*SubsetDataset) Get

```go
func (sd *SubsetDataset) Get(idx int) (data *tensor.Tensor, label *tensor.Tensor, err error)
```
Get returns a sample at the given index from the original dataset. It assumes
the index is within the bounds of the subset.

#### func (*SubsetDataset) Len

```go
func (sd *SubsetDataset) Len() int
```
Len returns the number of samples in the subset, which is the minimum of the
original dataset's length and the specified limit.

#### type Trainer

```go
type Trainer struct {
}
```

Trainer manages the training process

#### func  NewTrainer

```go
func NewTrainer(model Module, optimizer Optimizer, criterion Loss, config TrainingConfig) *Trainer
```
NewTrainer creates a new Trainer

#### func (*Trainer) Evaluate

```go
func (t *Trainer) Evaluate(dataLoader *DataLoader) (float64, float64, error)
```
Evaluate runs the model on a dataset and returns overall metrics

#### func (*Trainer) GetMetrics

```go
func (t *Trainer) GetMetrics() []TrainingMetrics
```
GetMetrics returns all training metrics

#### func (*Trainer) GetModelDevice

```go
func (t *Trainer) GetModelDevice() tensor.DeviceType
```
GetModelDevice returns the device type of the first model parameter

#### func (*Trainer) LoadCheckpoint

```go
func (t *Trainer) LoadCheckpoint(filepath string) error
```
LoadCheckpoint loads model parameters (simplified version)

#### func (*Trainer) Predict

```go
func (t *Trainer) Predict(input *tensor.Tensor) (*tensor.Tensor, error)
```
Predict runs inference on a single batch

#### func (*Trainer) SaveCheckpoint

```go
func (t *Trainer) SaveCheckpoint(filepath string) error
```
SaveCheckpoint saves model parameters (simplified version)

#### func (*Trainer) Train

```go
func (t *Trainer) Train(trainLoader, validLoader *DataLoader) error
```
Train runs the complete training loop

#### type TrainingConfig

```go
type TrainingConfig struct {
	Epochs        int
	Device        tensor.DeviceType
	PrintEvery    int  // Print training stats every N batches
	ValidateEvery int  // Run validation every N epochs (0 = no validation)
	EarlyStopping bool // Enable early stopping based on validation loss
	Patience      int  // Number of epochs to wait for improvement before stopping
}
```

TrainingConfig holds configuration for training

#### type TrainingMetrics

```go
type TrainingMetrics struct {
	Epoch         int
	TrainLoss     float64
	TrainAccuracy float64
	ValidLoss     float64
	ValidAccuracy float64
	EpochDuration time.Duration
	BatchCount    int
}
```

TrainingMetrics holds metrics for a single epoch
