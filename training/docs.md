# training
--
    import "."


## Usage

#### func  CalculateAUCROC

```go
func CalculateAUCROC(
	predictions []float32,
	trueLabels []int32,
	batchSize int,
) float64
```
CalculateAUCROC calculates Area Under ROC Curve for binary classification
GPU-resident architecture: operates on GPU tensor data, returns CPU scalar

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

#### type ActivationStats

```go
type ActivationStats struct {
	LayerName      string    `json:"layer_name"`
	ActivationType string    `json:"activation_type"`
	Mean           float64   `json:"mean"`
	Std            float64   `json:"std"`
	SparsityRatio  float64   `json:"sparsity_ratio"`
	Histogram      []float64 `json:"histogram"`
	Bins           []float64 `json:"bins"`
}
```

ActivationStats represents activation pattern statistics

#### type BatchPlotResult

```go
type BatchPlotResult struct {
	Success   bool   `json:"success"`
	PlotID    string `json:"plot_id,omitempty"`
	PlotURL   string `json:"plot_url,omitempty"`
	ViewURL   string `json:"view_url,omitempty"`
	PlotType  string `json:"plot_type,omitempty"`
	Message   string `json:"message,omitempty"`
	ErrorCode string `json:"error_code,omitempty"`
}
```

BatchPlotResult represents a single plot result within a batch response

#### type BatchPlottingResponse

```go
type BatchPlottingResponse struct {
	Success      bool              `json:"success"`
	Message      string            `json:"message"`
	BatchID      string            `json:"batch_id,omitempty"`
	Results      []BatchPlotResult `json:"results,omitempty"`
	DashboardURL string            `json:"dashboard_url,omitempty"`
	Summary      BatchSummary      `json:"summary,omitempty"`
}
```

BatchPlottingResponse represents the response from the batch plotting endpoint

#### type BatchSummary

```go
type BatchSummary struct {
	TotalPlots int `json:"total_plots"`
	Successful int `json:"successful"`
	Failed     int `json:"failed"`
}
```

BatchSummary represents the summary of a batch operation

#### type BatchedLabels

```go
type BatchedLabels struct {
}
```

BatchedLabels represents a collection of label batches Useful for multi-GPU
training or pipeline parallelism

#### func  NewBatchedLabels

```go
func NewBatchedLabels(batches []LabelData) (*BatchedLabels, error)
```
NewBatchedLabels creates a collection of label batches

#### func (*BatchedLabels) GetBatch

```go
func (bl *BatchedLabels) GetBatch(index int) (LabelData, error)
```
GetBatch returns the label batch at the specified index

#### func (*BatchedLabels) NumBatches

```go
func (bl *BatchedLabels) NumBatches() int
```
NumBatches returns the number of label batches

#### type CheckpointCapable

```go
type CheckpointCapable interface {
	GetModelSpec() *layers.ModelSpec
	GetParameterTensors() []*memory.Tensor
	GetCurrentLearningRate() float32
	GetLRScheduler() interface{} // Returns the scheduler if available
	SetLearningRate(lr float32)
	GetOptimizerState() *OptimizerStateData
	SetOptimizerState(state *checkpoints.OptimizerState) error
}
```

Add checkpoint methods to ModelTrainer interface

#### type CheckpointConfig

```go
type CheckpointConfig struct {
	SaveDirectory   string                       // Directory to save checkpoints
	SaveFrequency   int                          // Save every N epochs (0 = disabled)
	SaveBest        bool                         // Save checkpoint when validation improves
	MaxCheckpoints  int                          // Maximum number of checkpoints to keep (0 = unlimited)
	Format          checkpoints.CheckpointFormat // JSON or ONNX
	FilenamePattern string                       // Pattern for checkpoint filenames
}
```

CheckpointConfig configures checkpoint saving behavior

#### func  DefaultCheckpointConfig

```go
func DefaultCheckpointConfig() CheckpointConfig
```
DefaultCheckpointConfig returns a sensible default configuration

#### type CheckpointManager

```go
type CheckpointManager struct {
}
```

CheckpointManager handles checkpoint saving and loading for ModelTrainer

#### func  NewCheckpointManager

```go
func NewCheckpointManager(trainer *ModelTrainer, config CheckpointConfig) *CheckpointManager
```
NewCheckpointManager creates a new checkpoint manager

#### func (*CheckpointManager) LoadCheckpoint

```go
func (cm *CheckpointManager) LoadCheckpoint(filepath string) error
```
LoadCheckpoint loads a checkpoint and restores trainer state

#### func (*CheckpointManager) SaveBestCheckpoint

```go
func (cm *CheckpointManager) SaveBestCheckpoint(epoch int, step int, loss float32, accuracy float32) (bool, error)
```
SaveBestCheckpoint saves a checkpoint if it's better than previous best

#### func (*CheckpointManager) SaveCheckpoint

```go
func (cm *CheckpointManager) SaveCheckpoint(epoch int, step int, loss float32, accuracy float32, description string) error
```
SaveCheckpoint saves the current model state

#### func (*CheckpointManager) SavePeriodicCheckpoint

```go
func (cm *CheckpointManager) SavePeriodicCheckpoint(epoch int, step int, loss float32, accuracy float32) (bool, error)
```
SavePeriodicCheckpoint saves a checkpoint if it's time based on frequency

#### type ConfusionMatrix

```go
type ConfusionMatrix struct {
	NumClasses   int
	Matrix       [][]int // [true_class][predicted_class]
	TotalSamples int
}
```

ConfusionMatrix represents a confusion matrix for classification tasks Adheres
to GPU-resident architecture: all tensors on GPU, only scalar results on CPU

#### func  NewConfusionMatrix

```go
func NewConfusionMatrix(numClasses int) *ConfusionMatrix
```
NewConfusionMatrix creates a new confusion matrix

#### func (*ConfusionMatrix) GetAccuracy

```go
func (cm *ConfusionMatrix) GetAccuracy() float64
```
GetAccuracy returns overall classification accuracy

#### func (*ConfusionMatrix) GetMetric

```go
func (cm *ConfusionMatrix) GetMetric(metric MetricType) float64
```
GetMetric calculates and caches evaluation metrics Only CPU access for final
scalar metrics (GPU-resident architecture compliance)

#### func (*ConfusionMatrix) Reset

```go
func (cm *ConfusionMatrix) Reset()
```
Reset clears the confusion matrix

#### func (*ConfusionMatrix) UpdateFromPredictions

```go
func (cm *ConfusionMatrix) UpdateFromPredictions(
	predictions []float32,
	trueLabels []int32,
	batchSize int,
	numClasses int,
) error
```
UpdateFromPredictions updates confusion matrix from GPU-resident predictions
Maintains design compliance: predictions/labels come from GPU, only scalar
updates on CPU

#### type CosineAnnealingLRScheduler

```go
type CosineAnnealingLRScheduler struct {
	TMax   int     // Maximum number of epochs
	EtaMin float64 // Minimum learning rate
}
```

CosineAnnealingLRScheduler implements cosine annealing schedule

#### func  NewCosineAnnealingLRScheduler

```go
func NewCosineAnnealingLRScheduler(tMax int, etaMin float64) *CosineAnnealingLRScheduler
```
NewCosineAnnealingLRScheduler creates a cosine annealing scheduler

#### func (*CosineAnnealingLRScheduler) GetLR

```go
func (s *CosineAnnealingLRScheduler) GetLR(epoch int, step int, baseLR float64) float64
```

#### func (*CosineAnnealingLRScheduler) GetName

```go
func (s *CosineAnnealingLRScheduler) GetName() string
```

#### type DataPoint

```go
type DataPoint struct {
	X     interface{} `json:"x"`
	Y     interface{} `json:"y"`
	Z     interface{} `json:"z,omitempty"`     // For heatmaps, 3D plots
	Label string      `json:"label,omitempty"` // For categorical data
	Color string      `json:"color,omitempty"` // For custom coloring
}
```

DataPoint represents a single data point - flexible for different plot types

#### type EngineType

```go
type EngineType int
```

EngineType represents the training engine selection strategy Maintains
GPU-resident architecture compliance across all engine types

```go
const (
	// Auto automatically selects the optimal engine based on model architecture
	// - 4D input + Conv layers → Hybrid Engine (20k+ batches/sec performance)
	// - 2D input + Dense-only → Dynamic Engine (flexibility for MLPs)
	// - Complex architectures → Dynamic Engine (any architecture support)
	Auto EngineType = iota

	// Hybrid uses MPS for convolutions + MPSGraph for other operations
	// - Optimized for CNN architectures (3 conv + 2 FC pattern)
	// - Hardcoded optimizations for maximum performance
	// - Requires 4D input and Conv+Dense layer combination
	Hybrid

	// Dynamic builds MPSGraph dynamically for any architecture
	// - Supports any input dimensionality (2D, 4D, etc.)
	// - Supports any layer combination
	// - More flexible but slightly slower than Hybrid
	Dynamic
)
```

#### func  SelectOptimalEngine

```go
func SelectOptimalEngine(modelSpec *layers.ModelSpec, config TrainerConfig) EngineType
```
SelectOptimalEngine selects the best engine based on architecture analysis
Maintains GPU-resident architecture compliance for all engine types

#### func (EngineType) String

```go
func (et EngineType) String() string
```

#### type ExponentialLRScheduler

```go
type ExponentialLRScheduler struct {
	Gamma float64 // Multiplicative factor of LR decay per epoch
}
```

ExponentialLRScheduler decays learning rate exponentially

#### func  NewExponentialLRScheduler

```go
func NewExponentialLRScheduler(gamma float64) *ExponentialLRScheduler
```
NewExponentialLRScheduler creates an exponential learning rate scheduler

#### func (*ExponentialLRScheduler) GetLR

```go
func (s *ExponentialLRScheduler) GetLR(epoch int, step int, baseLR float64) float64
```

#### func (*ExponentialLRScheduler) GetName

```go
func (s *ExponentialLRScheduler) GetName() string
```

#### type Float32Labels

```go
type Float32Labels struct {
}
```

Float32Labels wraps []float32 for regression tasks Implements LabelData
interface with zero overhead

#### func  NewFloat32Labels

```go
func NewFloat32Labels(data []float32, shape []int) (*Float32Labels, error)
```
NewFloat32Labels creates regression labels with shape validation

#### func (*Float32Labels) DataType

```go
func (l *Float32Labels) DataType() LabelDataType
```
DataType returns LabelTypeFloat32 for regression

#### func (*Float32Labels) Shape

```go
func (l *Float32Labels) Shape() []int
```
Shape returns a copy of the label tensor shape

#### func (*Float32Labels) Size

```go
func (l *Float32Labels) Size() int
```
Size returns the total number of labels

#### func (*Float32Labels) ToFloat32Slice

```go
func (l *Float32Labels) ToFloat32Slice() []float32
```
ToFloat32Slice returns the underlying float32 slice directly ZERO-COST: No
allocation, no copying

#### func (*Float32Labels) UnsafePointer

```go
func (l *Float32Labels) UnsafePointer() unsafe.Pointer
```
UnsafePointer returns pointer to the underlying float32 data This enables
zero-copy transfer to GPU for regression

#### type GradientStats

```go
type GradientStats struct {
	LayerName    string    `json:"layer_name"`
	ParamType    string    `json:"param_type"`
	GradientNorm float64   `json:"gradient_norm"`
	Mean         float64   `json:"mean"`
	Std          float64   `json:"std"`
	Histogram    []float64 `json:"histogram"`
	Bins         []float64 `json:"bins"`
}
```

GradientStats represents gradient statistics

#### type InferencerConfig

```go
type InferencerConfig struct {
	// Performance settings
	BatchSize              int  `json:"batch_size"`            // Target batch size for inference
	UseDynamicEngine       bool `json:"use_dynamic_engine"`    // Use dynamic graph (recommended: true)
	OptimizeForSingleBatch bool `json:"optimize_single_batch"` // Optimize for batch size 1
	UseCommandPooling      bool `json:"use_command_pooling"`   // Enable command buffer pooling

	// Batch normalization mode
	BatchNormInferenceMode bool `json:"batchnorm_inference_mode"` // Use running stats for batch norm
}
```

InferencerConfig holds configuration for inference-only operations

#### func  DefaultInferencerConfig

```go
func DefaultInferencerConfig() InferencerConfig
```
DefaultInferencerConfig returns a sensible default configuration for inference

#### type Int32Labels

```go
type Int32Labels struct {
}
```

Int32Labels wraps []int32 for classification tasks Implements LabelData
interface with minimal overhead

#### func  NewInt32Labels

```go
func NewInt32Labels(data []int32, shape []int) (*Int32Labels, error)
```
NewInt32Labels creates classification labels with shape validation

#### func (*Int32Labels) DataType

```go
func (l *Int32Labels) DataType() LabelDataType
```
DataType returns LabelTypeInt32 for classification

#### func (*Int32Labels) Shape

```go
func (l *Int32Labels) Shape() []int
```
Shape returns a copy of the label tensor shape

#### func (*Int32Labels) Size

```go
func (l *Int32Labels) Size() int
```
Size returns the total number of labels

#### func (*Int32Labels) ToFloat32Slice

```go
func (l *Int32Labels) ToFloat32Slice() []float32
```
ToFloat32Slice converts int32 labels to float32 Uses cached conversion to
minimize allocations

#### func (*Int32Labels) UnsafePointer

```go
func (l *Int32Labels) UnsafePointer() unsafe.Pointer
```
UnsafePointer returns pointer to the underlying int32 data This enables
zero-copy transfer to GPU for classification

#### type LRScheduler

```go
type LRScheduler interface {
	// GetLR returns the learning rate for the current epoch/step
	// This is a pure function - no state modifications
	GetLR(epoch int, step int, baseLR float64) float64

	// GetName returns the scheduler name for logging
	GetName() string
}
```

LRScheduler defines the interface for learning rate scheduling strategies All
schedulers must be stateless and pure functions to maintain GPU-resident
principles

#### type LabelData

```go
type LabelData interface {
	// ToFloat32Slice returns the underlying data as []float32 for GPU consumption
	// For Float32Labels: returns slice directly (zero-cost)
	// For Int32Labels: converts to float32 (one-time cost)
	// PERFORMANCE: This method is designed to minimize allocations
	ToFloat32Slice() []float32

	// DataType returns the semantic type of labels for loss function selection
	DataType() LabelDataType

	// Size returns the number of label elements
	Size() int

	// Shape returns the tensor shape of labels [batch_size, num_classes/dims]
	Shape() []int

	// UnsafePointer returns a pointer to the underlying data for CGO
	// This enables zero-copy transfer to GPU
	UnsafePointer() unsafe.Pointer
}
```

LabelData provides a flexible interface for different label types This enables
zero-cost abstractions for both classification and regression while maintaining
GPU-residency and minimizing CGO calls

#### type LabelDataType

```go
type LabelDataType int
```

LabelDataType represents the semantic type of labels

```go
const (
	LabelTypeInt32   LabelDataType = iota // Classification labels
	LabelTypeFloat32                      // Regression targets
)
```

#### func (LabelDataType) String

```go
func (ldt LabelDataType) String() string
```
String returns human-readable label type name

#### type LossFunction

```go
type LossFunction int
```

LossFunction represents the loss function for training

```go
const (
	// Classification losses
	CrossEntropy            LossFunction = iota // Softmax cross-entropy for multi-class
	SparseCrossEntropy                          // Sparse categorical cross-entropy
	BinaryCrossEntropy                          // Binary cross-entropy for binary classification
	BCEWithLogits                               // Binary cross-entropy with logits (more numerically stable)
	CategoricalCrossEntropy                     // Categorical cross-entropy without softmax

	// Regression losses
	MeanSquaredError  // 5 - MSE for regression
	MeanAbsoluteError // 6 - MAE for regression
	Huber             // 7 - Huber loss for robust regression
)
```

#### func (LossFunction) String

```go
func (lf LossFunction) String() string
```

#### type MetricType

```go
type MetricType int
```

MetricType represents different evaluation metrics

```go
const (
	// Binary Classification Metrics
	Precision MetricType = iota
	Recall
	F1Score
	Specificity
	NPV // Negative Predictive Value

	// Multi-class Metrics
	MacroPrecision
	MacroRecall
	MacroF1
	MicroPrecision
	MicroRecall
	MicroF1

	// Ranking Metrics
	AUCROC
	AUCPR // Area Under Precision-Recall Curve

	// Regression Metrics
	MAE  // Mean Absolute Error
	MSE  // Mean Squared Error
	RMSE // Root Mean Squared Error
	R2   // R-squared
	NMAE // Normalized Mean Absolute Error
)
```

#### func (MetricType) String

```go
func (mt MetricType) String() string
```

#### type ModelArchitectureInfo

```go
type ModelArchitectureInfo struct {
	InputDimensions int    // Number of input dimensions (2D, 4D, etc.)
	HasConvLayers   bool   // Whether model contains Conv2D layers
	HasDenseLayers  bool   // Whether model contains Dense layers
	IsMLPOnly       bool   // Whether model is MLP-only (Dense + activations)
	IsCNNPattern    bool   // Whether model follows CNN pattern (Conv + Dense)
	LayerCount      int    // Total number of layers
	ParameterCount  int64  // Total trainable parameters
	Complexity      string // "Simple", "Standard", "Complex"
}
```

ModelArchitectureInfo provides architecture analysis for engine selection

#### func  AnalyzeModelArchitecture

```go
func AnalyzeModelArchitecture(modelSpec *layers.ModelSpec) *ModelArchitectureInfo
```
AnalyzeModelArchitecture analyzes model architecture for optimal engine
selection Maintains GPU-resident principles by analyzing layer specifications
only

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

#### type ModelInferencer

```go
type ModelInferencer struct {
}
```

ModelInferencer provides inference-only functionality using dedicated
InferenceEngine Optimized for forward-pass only without training overhead

#### func  NewModelInferencer

```go
func NewModelInferencer(
	modelSpec *layers.ModelSpec,
	config InferencerConfig,
) (*ModelInferencer, error)
```
NewModelInferencer creates a new inference-only engine

#### func (*ModelInferencer) Cleanup

```go
func (mi *ModelInferencer) Cleanup()
```
Cleanup performs deterministic resource cleanup

#### func (*ModelInferencer) GetModelSpec

```go
func (mi *ModelInferencer) GetModelSpec() *layers.ModelSpec
```
GetModelSpec returns the model specification

#### func (*ModelInferencer) GetParameterTensors

```go
func (mi *ModelInferencer) GetParameterTensors() []*memory.Tensor
```
GetParameterTensors returns the GPU-resident parameter tensors

#### func (*ModelInferencer) LoadWeights

```go
func (mi *ModelInferencer) LoadWeights(weights []checkpoints.WeightTensor) error
```
LoadWeights loads pre-trained weights into the inference engine

#### func (*ModelInferencer) LoadWeightsFromCheckpoint

```go
func (mi *ModelInferencer) LoadWeightsFromCheckpoint(weights []checkpoints.WeightTensor) error
```
LoadWeightsFromCheckpoint loads weights from a checkpoint

#### func (*ModelInferencer) Predict

```go
func (mi *ModelInferencer) Predict(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error)
```
Predict performs single forward pass for inference This is the lightweight
method optimized for single-image or small batch inference

#### func (*ModelInferencer) PredictBatch

```go
func (mi *ModelInferencer) PredictBatch(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error)
```
PredictBatch performs inference on a batch of data Optimized for larger batches
with efficient GPU utilization

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

#### func (*ModelTrainer) CalculateAccuracyUnified

```go
func (mt *ModelTrainer) CalculateAccuracyUnified(
	predictions []float32,
	trueLabels []float32,
	batchSize int,
	outputSize int,
	labelType LabelDataType,
) float64
```
CalculateAccuracyUnified calculates accuracy for both classification and
regression For classification: returns percentage of correct predictions For
regression: returns 1 - normalized mean absolute error

#### func (*ModelTrainer) CalculateRegressionMetric

```go
func (mt *ModelTrainer) CalculateRegressionMetric(
	predictions []float32,
	trueLabels []float32,
	batchSize int,
) float64
```
CalculateRegressionMetric calculates a metric for regression Returns 1 -
normalized mean absolute error (closer to 1 is better)

#### func (*ModelTrainer) CheckPlottingServiceHealth

```go
func (mt *ModelTrainer) CheckPlottingServiceHealth() error
```
CheckPlottingServiceHealth checks if the plotting service is available

#### func (*ModelTrainer) Cleanup

```go
func (mt *ModelTrainer) Cleanup()
```
Cleanup releases all resources

#### func (*ModelTrainer) ClearVisualizationData

```go
func (mt *ModelTrainer) ClearVisualizationData()
```
ClearVisualizationData clears all collected visualization data

#### func (*ModelTrainer) ConfigurePlottingService

```go
func (mt *ModelTrainer) ConfigurePlottingService(config PlottingServiceConfig)
```
ConfigurePlottingService configures the plotting service with custom settings

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

#### func (*ModelTrainer) DisableEvaluationMetrics

```go
func (mt *ModelTrainer) DisableEvaluationMetrics()
```
DisableEvaluationMetrics disables evaluation metrics for performance

#### func (*ModelTrainer) DisablePlottingService

```go
func (mt *ModelTrainer) DisablePlottingService()
```
DisablePlottingService disables the plotting service

#### func (*ModelTrainer) DisableVisualization

```go
func (mt *ModelTrainer) DisableVisualization()
```
DisableVisualization disables visualization data collection

#### func (*ModelTrainer) EnableEvaluationMetrics

```go
func (mt *ModelTrainer) EnableEvaluationMetrics()
```
EnableEvaluationMetrics enables comprehensive evaluation metrics collection
Metrics are calculated from GPU-resident tensors with CPU-only scalar results

#### func (*ModelTrainer) EnablePersistentBuffers

```go
func (mt *ModelTrainer) EnablePersistentBuffers(inputShape []int) error
```
EnablePersistentBuffers pre-allocates GPU tensors for reuse across training
steps This reduces allocation overhead and improves performance

#### func (*ModelTrainer) EnablePlottingService

```go
func (mt *ModelTrainer) EnablePlottingService()
```
EnablePlottingService enables the plotting service for sidecar communication

#### func (*ModelTrainer) EnableSidecarWithAutoStart

```go
func (mt *ModelTrainer) EnableSidecarWithAutoStart(config ...SidecarConfig) error
```
EnableSidecarWithAutoStart enables plotting service with automatic sidecar
management

#### func (*ModelTrainer) EnableVisualization

```go
func (mt *ModelTrainer) EnableVisualization()
```
EnableVisualization enables visualization data collection

#### func (*ModelTrainer) GenerateAllPlots

```go
func (mt *ModelTrainer) GenerateAllPlots() map[PlotType]PlotData
```
GenerateAllPlots generates all available plots and returns them

#### func (*ModelTrainer) GenerateAndOpenAllPlots

```go
func (mt *ModelTrainer) GenerateAndOpenAllPlots() error
```
GenerateAndOpenAllPlots generates all plots and opens them in a dashboard

#### func (*ModelTrainer) GenerateAndOpenPlot

```go
func (mt *ModelTrainer) GenerateAndOpenPlot(plotType PlotType) error
```
GenerateAndOpenPlot generates a plot and opens it in the browser

#### func (*ModelTrainer) GenerateConfusionMatrixPlot

```go
func (mt *ModelTrainer) GenerateConfusionMatrixPlot() PlotData
```
GenerateConfusionMatrixPlot generates and returns confusion matrix plot data

#### func (*ModelTrainer) GenerateLearningRateSchedulePlot

```go
func (mt *ModelTrainer) GenerateLearningRateSchedulePlot() PlotData
```
GenerateLearningRateSchedulePlot generates and returns learning rate schedule
plot data

#### func (*ModelTrainer) GeneratePrecisionRecallPlot

```go
func (mt *ModelTrainer) GeneratePrecisionRecallPlot() PlotData
```
GeneratePrecisionRecallPlot generates and returns Precision-Recall curve plot
data

#### func (*ModelTrainer) GenerateROCCurvePlot

```go
func (mt *ModelTrainer) GenerateROCCurvePlot() PlotData
```
GenerateROCCurvePlot generates and returns ROC curve plot data

#### func (*ModelTrainer) GenerateTrainingCurvesPlot

```go
func (mt *ModelTrainer) GenerateTrainingCurvesPlot() PlotData
```
GenerateTrainingCurvesPlot generates and returns training curves plot data

#### func (*ModelTrainer) GetClassificationMetrics

```go
func (mt *ModelTrainer) GetClassificationMetrics() map[string]float64
```
GetClassificationMetrics returns all classification metrics for the current
confusion matrix

#### func (*ModelTrainer) GetConfusionMatrix

```go
func (mt *ModelTrainer) GetConfusionMatrix() [][]int
```
GetConfusionMatrix returns a copy of the current confusion matrix

#### func (*ModelTrainer) GetCurrentLearningRate

```go
func (mt *ModelTrainer) GetCurrentLearningRate() float32
```
GetCurrentLearningRate returns the current learning rate based on scheduler This
is a pure computation - no GPU operations

#### func (*ModelTrainer) GetLRScheduler

```go
func (mt *ModelTrainer) GetLRScheduler() interface{}
```
GetLRScheduler returns the learning rate scheduler if available

#### func (*ModelTrainer) GetMetric

```go
func (mt *ModelTrainer) GetMetric(metric MetricType) float64
```
GetMetric returns the current value of a specific metric CPU-only scalar result
(GPU-resident architecture compliant)

#### func (*ModelTrainer) GetMetricHistory

```go
func (mt *ModelTrainer) GetMetricHistory(metric MetricType) []float64
```
GetMetricHistory returns the history of a specific metric for plotting

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

#### func (*ModelTrainer) GetOptimizerState

```go
func (mt *ModelTrainer) GetOptimizerState() *OptimizerStateData
```
GetOptimizerState returns the optimizer state for checkpoint saving

#### func (*ModelTrainer) GetParameterTensors

```go
func (mt *ModelTrainer) GetParameterTensors() []*memory.Tensor
```
GetParameterTensors returns the parameter tensors for weight extraction

#### func (*ModelTrainer) GetRegressionMetrics

```go
func (mt *ModelTrainer) GetRegressionMetrics() map[string]float64
```
GetRegressionMetrics returns all regression metrics

#### func (*ModelTrainer) GetSchedulerInfo

```go
func (mt *ModelTrainer) GetSchedulerInfo() string
```
GetSchedulerInfo returns current scheduler information for logging

#### func (*ModelTrainer) GetStats

```go
func (mt *ModelTrainer) GetStats() *ModelTrainingStats
```
GetStats returns comprehensive training statistics

#### func (*ModelTrainer) GetVisualizationCollector

```go
func (mt *ModelTrainer) GetVisualizationCollector() *VisualizationCollector
```
GetVisualizationCollector returns the visualization collector for advanced usage

#### func (*ModelTrainer) InferBatch

```go
func (mt *ModelTrainer) InferBatch(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error)
```
InferBatch performs inference on a batch of data Conforms to design
requirements: single CGO call, GPU-resident, shared resources

#### func (*ModelTrainer) IsEvaluationMetricsEnabled

```go
func (mt *ModelTrainer) IsEvaluationMetricsEnabled() bool
```
IsEvaluationMetricsEnabled returns whether comprehensive metrics are enabled

#### func (*ModelTrainer) IsVisualizationEnabled

```go
func (mt *ModelTrainer) IsVisualizationEnabled() bool
```
IsVisualizationEnabled returns whether visualization is enabled

#### func (*ModelTrainer) Predict

```go
func (mt *ModelTrainer) Predict(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error)
```
Predict provides a lightweight inference method for backward compatibility For
optimal inference performance, use ModelInferencer instead

#### func (*ModelTrainer) PrintModelArchitecture

```go
func (mt *ModelTrainer) PrintModelArchitecture(modelName string)
```
PrintModelArchitecture prints the model architecture in PyTorch style

#### func (*ModelTrainer) RecordEpochMetrics

```go
func (mt *ModelTrainer) RecordEpochMetrics(epoch int, trainLoss, trainAcc, valLoss, valAcc float64)
```
RecordEpochMetrics records epoch-level metrics for visualization

#### func (*ModelTrainer) RecordMetricsForVisualization

```go
func (mt *ModelTrainer) RecordMetricsForVisualization()
```
RecordMetricsForVisualization records comprehensive metrics for visualization
This method integrates with the evaluation metrics system

#### func (*ModelTrainer) ResetMetrics

```go
func (mt *ModelTrainer) ResetMetrics()
```
ResetMetrics clears all accumulated metrics and history

#### func (*ModelTrainer) SendAllPlotsToSidecar

```go
func (mt *ModelTrainer) SendAllPlotsToSidecar() map[PlotType]*PlottingResponse
```
SendAllPlotsToSidecar sends all available plots to the sidecar plotting service

#### func (*ModelTrainer) SendPlotToSidecar

```go
func (mt *ModelTrainer) SendPlotToSidecar(plotType PlotType) (*PlottingResponse, error)
```
SendPlotToSidecar sends a specific plot to the sidecar plotting service

#### func (*ModelTrainer) SetAccuracyCheckInterval

```go
func (mt *ModelTrainer) SetAccuracyCheckInterval(interval int)
```
SetAccuracyCheckInterval configures how often accuracy is calculated interval=0:
every step (default, maximum accuracy but higher CGO overhead) interval=10:
every 10 steps (reduces CGO calls by ~40%, slight accuracy tracking lag)
interval=50: every 50 steps (reduces CGO calls by ~80%, minimal accuracy
tracking)

#### func (*ModelTrainer) SetEpoch

```go
func (mt *ModelTrainer) SetEpoch(epoch int)
```
SetEpoch updates the current epoch for learning rate scheduling Call this at the
start of each epoch

#### func (*ModelTrainer) SetLRScheduler

```go
func (mt *ModelTrainer) SetLRScheduler(scheduler LRScheduler)
```
SetLRScheduler sets a learning rate scheduler for the trainer This maintains
GPU-resident principles by only updating LR between epochs

#### func (*ModelTrainer) SetLearningRate

```go
func (mt *ModelTrainer) SetLearningRate(lr float32)
```
SetLearningRate sets the learning rate

#### func (*ModelTrainer) SetOptimizerState

```go
func (mt *ModelTrainer) SetOptimizerState(state interface{}) error
```
SetOptimizerState restores optimizer state from checkpoint

#### func (*ModelTrainer) StartValidationPhase

```go
func (mt *ModelTrainer) StartValidationPhase()
```
StartValidationPhase prepares for validation by resetting probability collection

#### func (*ModelTrainer) StepSchedulerWithMetric

```go
func (mt *ModelTrainer) StepSchedulerWithMetric(metric float64)
```
StepSchedulerWithMetric updates schedulers that depend on validation metrics For
ReduceLROnPlateauScheduler - call this after validation

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

#### func (*ModelTrainer) TrainBatchOptimized

```go
func (mt *ModelTrainer) TrainBatchOptimized(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error)
```
TrainBatchOptimized executes a training step with batched CGO operations This
reduces CGO overhead by combining multiple operations into a single call Follows
design principle: "Single CGO call per training step"

#### func (*ModelTrainer) TrainBatchPersistent

```go
func (mt *ModelTrainer) TrainBatchPersistent(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error)
```
TrainBatchPersistent executes a training step using persistent GPU buffers This
provides maximum performance by eliminating per-step tensor allocations

#### func (*ModelTrainer) TrainBatchPersistentWithCommandPool

```go
func (mt *ModelTrainer) TrainBatchPersistentWithCommandPool(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error)
```
TrainBatchPersistentWithCommandPool executes a training step using both
persistent tensors and pooled command buffers for maximum performance and
resource efficiency DEPRECATED: Use TrainBatchUnified for new code

#### func (*ModelTrainer) TrainBatchUnified

```go
func (mt *ModelTrainer) TrainBatchUnified(
	inputData []float32,
	inputShape []int,
	labelData LabelData,
) (*TrainingResultOptimized, error)
```
TrainBatchUnified executes a training step with flexible label types This is the
recommended API for new code as it supports both classification and regression
while maintaining GPU-residency and minimizing CGO calls

#### func (*ModelTrainer) TrainBatchWithCommandPool

```go
func (mt *ModelTrainer) TrainBatchWithCommandPool(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error)
```
TrainBatchWithCommandPool executes a training step using pooled command buffers
This method implements the complete command buffer pooling strategy to prevent
resource leaks

#### func (*ModelTrainer) UpdateMetricsFromInference

```go
func (mt *ModelTrainer) UpdateMetricsFromInference(
	predictions []float32,
	trueLabels interface{},
	batchSize int,
) error
```
UpdateMetricsFromInference updates evaluation metrics from inference results
GPU-resident architecture: operates on GPU tensor data, stores CPU scalars only

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

#### type NoOpScheduler

```go
type NoOpScheduler struct{}
```

NoOpScheduler maintains constant learning rate (default behavior)

#### func (*NoOpScheduler) GetLR

```go
func (s *NoOpScheduler) GetLR(epoch int, step int, baseLR float64) float64
```

#### func (*NoOpScheduler) GetName

```go
func (s *NoOpScheduler) GetName() string
```

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

#### type OptimizerStateData

```go
type OptimizerStateData struct {
	Type       string
	Parameters map[string]interface{}
	StateData  []checkpoints.OptimizerTensor
}
```

OptimizerStateData represents internal optimizer state

#### type PRPoint

```go
type PRPoint struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	Threshold float64 `json:"threshold"`
}
```

PRPoint represents a point on the Precision-Recall curve

#### type ParameterStats

```go
type ParameterStats struct {
	LayerName string    `json:"layer_name"`
	ParamType string    `json:"param_type"` // "weight", "bias"
	Mean      float64   `json:"mean"`
	Std       float64   `json:"std"`
	Min       float64   `json:"min"`
	Max       float64   `json:"max"`
	Histogram []float64 `json:"histogram"`
	Bins      []float64 `json:"bins"`
}
```

ParameterStats represents parameter distribution statistics

#### type PlotConfig

```go
type PlotConfig struct {
	XAxisLabel    string                 `json:"x_axis_label"`
	YAxisLabel    string                 `json:"y_axis_label"`
	ZAxisLabel    string                 `json:"z_axis_label,omitempty"`
	XAxisScale    string                 `json:"x_axis_scale"` // "linear", "log"
	YAxisScale    string                 `json:"y_axis_scale"` // "linear", "log"
	ShowLegend    bool                   `json:"show_legend"`
	ShowGrid      bool                   `json:"show_grid"`
	Width         int                    `json:"width"`
	Height        int                    `json:"height"`
	Interactive   bool                   `json:"interactive"`
	CustomOptions map[string]interface{} `json:"custom_options,omitempty"`
}
```

PlotConfig contains plot-specific configuration

#### type PlotData

```go
type PlotData struct {
	// Metadata
	PlotType  PlotType  `json:"plot_type"`
	Title     string    `json:"title"`
	Timestamp time.Time `json:"timestamp"`
	ModelName string    `json:"model_name"`

	// Data series - flexible structure for different plot types
	Series []SeriesData `json:"series"`

	// Plot configuration
	Config PlotConfig `json:"config"`

	// Metrics metadata
	Metrics map[string]interface{} `json:"metrics,omitempty"`
}
```

PlotData represents the universal JSON format for the sidecar plotting service

#### func (PlotData) ToJSON

```go
func (pd PlotData) ToJSON() (string, error)
```
ToJSON converts plot data to JSON string

#### type PlotType

```go
type PlotType string
```

PlotType represents different types of plots that can be generated

```go
const (
	// Training plots
	TrainingCurves       PlotType = "training_curves"
	LearningRateSchedule PlotType = "learning_rate_schedule"

	// Evaluation plots
	ROCCurve            PlotType = "roc_curve"
	PrecisionRecall     PlotType = "precision_recall"
	ConfusionMatrixPlot PlotType = "confusion_matrix"

	// Model analysis plots
	ParameterDistribution PlotType = "parameter_distribution"
	GradientHistogram     PlotType = "gradient_histogram"
	ActivationPattern     PlotType = "activation_pattern"

	// Regression plots
	RegressionScatter      PlotType = "regression_scatter"
	ResidualPlot           PlotType = "residual_plot"
	QQPlot                 PlotType = "qq_plot"
	FeatureImportancePlot  PlotType = "feature_importance"
	LearningCurvePlot      PlotType = "learning_curve"
	ValidationCurvePlot    PlotType = "validation_curve"
	PredictionIntervalPlot PlotType = "prediction_interval"
	FeatureCorrelationPlot PlotType = "feature_correlation"
	PartialDependencePlot  PlotType = "partial_dependence"
)
```

#### type PlottingResponse

```go
type PlottingResponse struct {
	Success      bool   `json:"success"`
	Message      string `json:"message"`
	PlotURL      string `json:"plot_url,omitempty"`
	ViewURL      string `json:"view_url,omitempty"`
	PlotID       string `json:"plot_id,omitempty"`
	BatchID      string `json:"batch_id,omitempty"`
	DashboardURL string `json:"dashboard_url,omitempty"`
	ErrorCode    string `json:"error_code,omitempty"`
}
```

PlottingResponse represents the response from the plotting service

#### type PlottingService

```go
type PlottingService struct {
}
```

PlottingService handles communication with the sidecar plotting application

#### func  NewPlottingService

```go
func NewPlottingService(config PlottingServiceConfig) *PlottingService
```
NewPlottingService creates a new plotting service client

#### func (*PlottingService) BatchSendPlots

```go
func (ps *PlottingService) BatchSendPlots(plotDataList []PlotData) (*BatchPlottingResponse, error)
```
BatchSendPlots sends multiple plots in a single request

#### func (*PlottingService) CheckHealth

```go
func (ps *PlottingService) CheckHealth() error
```
CheckHealth checks if the plotting service is available

#### func (*PlottingService) Disable

```go
func (ps *PlottingService) Disable()
```
Disable disables the plotting service

#### func (*PlottingService) Enable

```go
func (ps *PlottingService) Enable()
```
Enable enables the plotting service

#### func (*PlottingService) GenerateAndSendAllPlots

```go
func (ps *PlottingService) GenerateAndSendAllPlots(collector *VisualizationCollector) map[PlotType]*PlottingResponse
```
GenerateAndSendAllPlots generates all available plots and sends them to the
sidecar service

#### func (*PlottingService) GenerateAndSendAllPlotsWithBrowser

```go
func (ps *PlottingService) GenerateAndSendAllPlotsWithBrowser(collector *VisualizationCollector) map[PlotType]*PlottingResponse
```
GenerateAndSendAllPlotsWithBrowser generates all plots using batch endpoint and
opens dashboard

#### func (*PlottingService) GenerateAndSendPlot

```go
func (ps *PlottingService) GenerateAndSendPlot(collector *VisualizationCollector, plotType PlotType) (*PlottingResponse, error)
```
GenerateAndSendPlot generates a plot and sends it to the sidecar service

#### func (*PlottingService) IsEnabled

```go
func (ps *PlottingService) IsEnabled() bool
```
IsEnabled returns whether the plotting service is enabled

#### func (*PlottingService) OpenInBrowser

```go
func (ps *PlottingService) OpenInBrowser(url string) error
```
OpenInBrowser opens the given URL in the default web browser It automatically
detects the operating system and uses the appropriate command

#### func (*PlottingService) SendPlotData

```go
func (ps *PlottingService) SendPlotData(plotData PlotData) (*PlottingResponse, error)
```
SendPlotData sends plot data to the sidecar plotting service

#### func (*PlottingService) SendPlotDataAndOpen

```go
func (ps *PlottingService) SendPlotDataAndOpen(plotData PlotData) (*PlottingResponse, error)
```
SendPlotDataAndOpen sends plot data and automatically opens the result in
browser

#### func (*PlottingService) SendPlotDataWithRetry

```go
func (ps *PlottingService) SendPlotDataWithRetry(plotData PlotData, config PlottingServiceConfig) (*PlottingResponse, error)
```
SendPlotDataWithRetry sends plot data with retry logic

#### type PlottingServiceConfig

```go
type PlottingServiceConfig struct {
	BaseURL       string        `json:"base_url"`
	Timeout       time.Duration `json:"timeout"`
	RetryAttempts int           `json:"retry_attempts"`
	RetryDelay    time.Duration `json:"retry_delay"`
}
```

PlottingServiceConfig contains configuration for the plotting service

#### func  DefaultPlottingServiceConfig

```go
func DefaultPlottingServiceConfig() PlottingServiceConfig
```
DefaultPlottingServiceConfig returns default configuration for the plotting
service

#### type ProblemType

```go
type ProblemType int
```

ProblemType represents the type of machine learning problem

```go
const (
	// Classification for discrete class prediction
	Classification ProblemType = iota
	// Regression for continuous value prediction
	Regression
)
```

#### func (ProblemType) String

```go
func (pt ProblemType) String() string
```

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

#### type ROCPoint

```go
type ROCPoint struct {
	Threshold float32
	TPR       float64 // True Positive Rate (Recall)
	FPR       float64 // False Positive Rate (1 - Specificity)
}
```

ROCPoint represents a point on the ROC curve

#### type ROCPointViz

```go
type ROCPointViz struct {
	FPR       float64 `json:"fpr"`
	TPR       float64 `json:"tpr"`
	Threshold float64 `json:"threshold"`
}
```

ROCPointViz represents a point on the ROC curve for visualization

#### type ReduceLROnPlateauScheduler

```go
type ReduceLROnPlateauScheduler struct {
	Factor    float64 // Factor by which the learning rate will be reduced
	Patience  int     // Number of epochs with no improvement after which LR will be reduced
	Threshold float64 // Threshold for measuring the new optimum
	Mode      string  // One of "min" or "max"
}
```

ReduceLROnPlateauScheduler reduces LR when a metric has stopped improving This
scheduler requires state tracking, so it's handled differently

#### func  NewReduceLROnPlateauScheduler

```go
func NewReduceLROnPlateauScheduler(factor float64, patience int, threshold float64, mode string) *ReduceLROnPlateauScheduler
```
NewReduceLROnPlateauScheduler creates a plateau-based scheduler

#### func (*ReduceLROnPlateauScheduler) GetLR

```go
func (s *ReduceLROnPlateauScheduler) GetLR(epoch int, step int, baseLR float64) float64
```

#### func (*ReduceLROnPlateauScheduler) GetName

```go
func (s *ReduceLROnPlateauScheduler) GetName() string
```

#### func (*ReduceLROnPlateauScheduler) Step

```go
func (s *ReduceLROnPlateauScheduler) Step(metric float64, currentLR float64) float64
```
Step checks if LR should be reduced based on metric This is called once per
epoch with the validation metric

#### type RegressionMetrics

```go
type RegressionMetrics struct {
	MAE  float64 // Mean Absolute Error
	MSE  float64 // Mean Squared Error
	RMSE float64 // Root Mean Squared Error
	R2   float64 // R-squared
	NMAE float64 // Normalized Mean Absolute Error
}
```

RegressionMetrics holds comprehensive regression evaluation metrics

#### func  CalculateRegressionMetrics

```go
func CalculateRegressionMetrics(
	predictions []float32,
	trueValues []float32,
	batchSize int,
) *RegressionMetrics
```
CalculateRegressionMetrics computes comprehensive regression metrics
GPU-resident architecture: operates on GPU tensor data, returns CPU scalars

#### type SeriesData

```go
type SeriesData struct {
	Name  string                 `json:"name"`
	Type  string                 `json:"type"` // "line", "scatter", "histogram", "heatmap", "bar"
	Data  []DataPoint            `json:"data"`
	Style map[string]interface{} `json:"style,omitempty"`
}
```

SeriesData represents a single data series in a plot

#### type SidecarConfig

```go
type SidecarConfig struct {
	Port       int    `json:"port"`
	AutoStart  bool   `json:"auto_start"`
	DockerMode bool   `json:"docker_mode"`
	SidecarDir string `json:"sidecar_dir"`
}
```

SidecarConfig contains configuration for the sidecar service

#### func  DefaultSidecarConfig

```go
func DefaultSidecarConfig() SidecarConfig
```
DefaultSidecarConfig returns default configuration for the sidecar

#### type SidecarManager

```go
type SidecarManager struct {
}
```

SidecarManager handles automatic sidecar service management

#### func  NewSidecarManager

```go
func NewSidecarManager(config SidecarConfig) (*SidecarManager, error)
```
NewSidecarManager creates a new sidecar manager

#### func (*SidecarManager) EnsureRunning

```go
func (sm *SidecarManager) EnsureRunning() error
```
EnsureRunning ensures the sidecar service is running

#### func (*SidecarManager) GetBaseURL

```go
func (sm *SidecarManager) GetBaseURL() string
```
GetBaseURL returns the base URL for the sidecar service

#### func (*SidecarManager) IsRunning

```go
func (sm *SidecarManager) IsRunning() bool
```
IsRunning checks if the sidecar service is running

#### func (*SidecarManager) Start

```go
func (sm *SidecarManager) Start() error
```
Start starts the sidecar service

#### func (*SidecarManager) Stop

```go
func (sm *SidecarManager) Stop() error
```
Stop stops the sidecar service

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

#### func  NewRMSPropTrainer

```go
func NewRMSPropTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
NewRMSPropTrainer creates an RMSProp trainer with defaults (convenience
function)

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

#### type StepLRScheduler

```go
type StepLRScheduler struct {
	StepSize int     // Epochs between LR reductions
	Gamma    float64 // Multiplicative factor of LR decay
}
```

StepLRScheduler reduces learning rate by a factor every stepSize epochs

#### func  NewStepLRScheduler

```go
func NewStepLRScheduler(stepSize int, gamma float64) *StepLRScheduler
```
NewStepLRScheduler creates a step learning rate scheduler

#### func (*StepLRScheduler) GetLR

```go
func (s *StepLRScheduler) GetLR(epoch int, step int, baseLR float64) float64
```

#### func (*StepLRScheduler) GetName

```go
func (s *StepLRScheduler) GetName() string
```

#### type TrainerConfig

```go
type TrainerConfig struct {
	// Training parameters
	BatchSize    int     `json:"batch_size"`
	LearningRate float32 `json:"learning_rate"`

	// Optimizer configuration
	OptimizerType cgo_bridge.OptimizerType `json:"optimizer_type"`

	// Optimizer-specific parameters
	Beta1       float32 `json:"beta1"`        // Adam momentum decay (default: 0.9) / RMSProp momentum (default: 0.0)
	Beta2       float32 `json:"beta2"`        // Adam variance decay (default: 0.999) - unused for RMSProp
	Epsilon     float32 `json:"epsilon"`      // Numerical stability (default: 1e-8)
	WeightDecay float32 `json:"weight_decay"` // L2 regularization (default: 0.0)

	// RMSProp-specific parameters
	Alpha    float32 `json:"alpha"`    // RMSProp smoothing constant (default: 0.99)
	Momentum float32 `json:"momentum"` // RMSProp momentum (default: 0.0)
	Centered bool    `json:"centered"` // RMSProp centered variant (default: false)

	// Engine selection (GPU-resident architecture compliance)
	EngineType EngineType `json:"engine_type"` // Engine selection: Auto, Hybrid, Dynamic (default: Auto)

	// Problem type and loss function configuration
	ProblemType      ProblemType  `json:"problem_type"`       // Classification or Regression (default: Classification)
	LossFunction     LossFunction `json:"loss_function"`      // Loss function for the problem type (default: CrossEntropy)
	UseHybridEngine  bool         `json:"use_hybrid_engine"`  // DEPRECATED: Use EngineType instead
	UseDynamicEngine bool         `json:"use_dynamic_engine"` // DEPRECATED: Use EngineType instead
	InferenceOnly    bool         `json:"inference_only"`     // Skip training setup, optimize for inference (forward-pass only)
}
```

TrainerConfig provides comprehensive configuration for training

#### func (*TrainerConfig) Validate

```go
func (tc *TrainerConfig) Validate() error
```
Validate ensures the problem type and loss function are compatible

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

#### func (*TrainerFactory) CreateRMSPropTrainer

```go
func (tf *TrainerFactory) CreateRMSPropTrainer(batchSize int, learningRate, alpha, epsilon, weightDecay, momentum float32, centered bool) (*SimpleTrainer, error)
```
CreateRMSPropTrainer creates an RMSProp trainer with specified parameters

#### func (*TrainerFactory) CreateRMSPropTrainerWithDefaults

```go
func (tf *TrainerFactory) CreateRMSPropTrainerWithDefaults(batchSize int, learningRate float32) (*SimpleTrainer, error)
```
CreateRMSPropTrainerWithDefaults creates an RMSProp trainer with sensible
defaults

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

#### func (*TrainerFactory) GetDefaultRMSPropConfig

```go
func (tf *TrainerFactory) GetDefaultRMSPropConfig(learningRate float32) OptimizerConfig
```
GetDefaultRMSPropConfig returns default RMSProp configuration

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

#### type TrainingResultOptimized

```go
type TrainingResultOptimized struct {
	Loss        float32
	Accuracy    float64 // Only valid if HasAccuracy is true
	HasAccuracy bool    // Whether accuracy was calculated this step
	BatchSize   int
	StepTime    time.Duration
	Success     bool
	BatchRate   float64 // Batches per second
}
```

TrainingResultOptimized represents the result of an optimized training step
Includes optional accuracy calculation to reduce CGO overhead

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

#### type VisualizationCollector

```go
type VisualizationCollector struct {
}
```

VisualizationCollector handles data collection for plotting

#### func  NewVisualizationCollector

```go
func NewVisualizationCollector(modelName string) *VisualizationCollector
```
NewVisualizationCollector creates a new visualization collector

#### func (*VisualizationCollector) Clear

```go
func (vc *VisualizationCollector) Clear()
```
Clear resets all collected data

#### func (*VisualizationCollector) Disable

```go
func (vc *VisualizationCollector) Disable()
```
Disable disables visualization data collection

#### func (*VisualizationCollector) Enable

```go
func (vc *VisualizationCollector) Enable()
```
Enable enables visualization data collection

#### func (*VisualizationCollector) GenerateConfusionMatrixPlot

```go
func (vc *VisualizationCollector) GenerateConfusionMatrixPlot() PlotData
```
GenerateConfusionMatrixPlot generates confusion matrix plot data

#### func (*VisualizationCollector) GenerateFeatureCorrelationPlot

```go
func (vc *VisualizationCollector) GenerateFeatureCorrelationPlot() PlotData
```
GenerateFeatureCorrelationPlot generates feature correlation heatmap for
multicollinearity analysis

#### func (*VisualizationCollector) GenerateFeatureImportancePlot

```go
func (vc *VisualizationCollector) GenerateFeatureImportancePlot() PlotData
```
GenerateFeatureImportancePlot generates feature importance plot data

#### func (*VisualizationCollector) GenerateLearningCurvePlot

```go
func (vc *VisualizationCollector) GenerateLearningCurvePlot() PlotData
```
GenerateLearningCurvePlot generates learning curve plot data showing performance
vs training set size

#### func (*VisualizationCollector) GenerateLearningRateSchedulePlot

```go
func (vc *VisualizationCollector) GenerateLearningRateSchedulePlot() PlotData
```
GenerateLearningRateSchedulePlot generates learning rate schedule plot data

#### func (*VisualizationCollector) GeneratePartialDependencePlot

```go
func (vc *VisualizationCollector) GeneratePartialDependencePlot() PlotData
```
GeneratePartialDependencePlot generates partial dependence plots for individual
feature effect analysis

#### func (*VisualizationCollector) GeneratePrecisionRecallPlot

```go
func (vc *VisualizationCollector) GeneratePrecisionRecallPlot() PlotData
```
GeneratePrecisionRecallPlot generates Precision-Recall curve plot data

#### func (*VisualizationCollector) GeneratePredictionIntervalPlot

```go
func (vc *VisualizationCollector) GeneratePredictionIntervalPlot() PlotData
```
GeneratePredictionIntervalPlot generates prediction interval plot data showing
prediction uncertainty

#### func (*VisualizationCollector) GenerateQQPlot

```go
func (vc *VisualizationCollector) GenerateQQPlot() PlotData
```
GenerateQQPlot generates Q-Q plot data for validating normal distribution of
residuals

#### func (*VisualizationCollector) GenerateROCCurvePlot

```go
func (vc *VisualizationCollector) GenerateROCCurvePlot() PlotData
```
GenerateROCCurvePlot generates ROC curve plot data

#### func (*VisualizationCollector) GenerateRegressionScatterPlot

```go
func (vc *VisualizationCollector) GenerateRegressionScatterPlot() PlotData
```
GenerateRegressionScatterPlot generates regression scatter plot data

#### func (*VisualizationCollector) GenerateResidualPlot

```go
func (vc *VisualizationCollector) GenerateResidualPlot() PlotData
```
GenerateResidualPlot generates residual plot data

#### func (*VisualizationCollector) GenerateTrainingCurvesPlot

```go
func (vc *VisualizationCollector) GenerateTrainingCurvesPlot() PlotData
```
GenerateTrainingCurvesPlot generates training curves plot data

#### func (*VisualizationCollector) GenerateValidationCurvePlot

```go
func (vc *VisualizationCollector) GenerateValidationCurvePlot() PlotData
```
GenerateValidationCurvePlot generates validation curve plot data showing
performance vs hyperparameter values

#### func (*VisualizationCollector) IsEnabled

```go
func (vc *VisualizationCollector) IsEnabled() bool
```
IsEnabled returns whether visualization is enabled

#### func (*VisualizationCollector) RecordActivationStats

```go
func (vc *VisualizationCollector) RecordActivationStats(layerName, activationType string, stats ActivationStats)
```
RecordActivationStats records activation pattern statistics

#### func (*VisualizationCollector) RecordConfusionMatrix

```go
func (vc *VisualizationCollector) RecordConfusionMatrix(matrix [][]int, classNames []string)
```
RecordConfusionMatrix records confusion matrix data

#### func (*VisualizationCollector) RecordEpoch

```go
func (vc *VisualizationCollector) RecordEpoch(epoch int, trainLoss, trainAcc, valLoss, valAcc float64)
```
RecordEpoch records epoch-level metrics

#### func (*VisualizationCollector) RecordFeatureCorrelation

```go
func (vc *VisualizationCollector) RecordFeatureCorrelation(correlationMatrix [][]float64, featureNames []string)
```
RecordFeatureCorrelation records feature correlation matrix for
multicollinearity analysis

#### func (*VisualizationCollector) RecordFeatureImportance

```go
func (vc *VisualizationCollector) RecordFeatureImportance(featureNames []string, coefficients []float64, stdErrors []float64)
```
RecordFeatureImportance records feature names and their coefficients for
regression models

#### func (*VisualizationCollector) RecordGradientStats

```go
func (vc *VisualizationCollector) RecordGradientStats(layerName, paramType string, stats GradientStats)
```
RecordGradientStats records gradient statistics

#### func (*VisualizationCollector) RecordLearningCurve

```go
func (vc *VisualizationCollector) RecordLearningCurve(trainingSizes []int, trainingScores, validationScores []float64, trainingStdErrors, validationStdErrors []float64)
```
RecordLearningCurve records learning curve data showing performance vs training
set size

#### func (*VisualizationCollector) RecordPRData

```go
func (vc *VisualizationCollector) RecordPRData(prPoints []PRPoint)
```
RecordPRData records Precision-Recall curve data points

#### func (*VisualizationCollector) RecordParameterStats

```go
func (vc *VisualizationCollector) RecordParameterStats(layerName, paramType string, stats ParameterStats)
```
RecordParameterStats records parameter distribution statistics

#### func (*VisualizationCollector) RecordPartialDependence

```go
func (vc *VisualizationCollector) RecordPartialDependence(featureNames []string, featureValues [][]float64, partialEffects [][]float64)
```
RecordPartialDependence records partial dependence data for individual feature
effect analysis

#### func (*VisualizationCollector) RecordPredictionInterval

```go
func (vc *VisualizationCollector) RecordPredictionInterval(x, y []float64, confidenceLower, confidenceUpper, predictionLower, predictionUpper, standardErrors []float64)
```
RecordPredictionInterval records prediction interval data for regression
uncertainty visualization

#### func (*VisualizationCollector) RecordROCData

```go
func (vc *VisualizationCollector) RecordROCData(rocPoints []ROCPointViz)
```
RecordROCData records ROC curve data points

#### func (*VisualizationCollector) RecordRegressionData

```go
func (vc *VisualizationCollector) RecordRegressionData(predictions, trueValues []float64)
```
RecordRegressionData records regression predictions and true values

#### func (*VisualizationCollector) RecordTrainingStep

```go
func (vc *VisualizationCollector) RecordTrainingStep(step int, loss, accuracy, learningRate float64)
```
RecordTrainingStep records training metrics for a single step

#### func (*VisualizationCollector) RecordValidationCurve

```go
func (vc *VisualizationCollector) RecordValidationCurve(parameterName string, parameterValues []float64, trainingScores, validationScores []float64, trainingStdErrors, validationStdErrors []float64)
```
RecordValidationCurve records validation curve data showing performance vs
hyperparameter values

#### func (*VisualizationCollector) RecordValidationStep

```go
func (vc *VisualizationCollector) RecordValidationStep(step int, loss, accuracy float64)
```
RecordValidationStep records validation metrics for a single step
