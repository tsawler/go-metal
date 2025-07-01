package training

import (
	"fmt"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/engine"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// ModelTrainer provides layer-based training while maintaining the proven single-CGO-call architecture
// This is the compliant implementation that integrates with the existing high-performance TrainingEngine
type ModelTrainer struct {
	modelEngine *engine.ModelTrainingEngine
	modelSpec   *layers.ModelSpec
	batchSize   int
	config      cgo_bridge.TrainingConfig
	currentStep int
	
	// Performance tracking
	lastStepTime    time.Duration
	totalSteps      int64
	totalLoss       float64
	averageLoss     float32
}

// NewModelTrainer creates a new model-based trainer using the existing TrainingEngine architecture
func NewModelTrainer(
	modelSpec *layers.ModelSpec,
	config TrainerConfig,
) (*ModelTrainer, error) {
	// Validate configuration
	if err := validateTrainerConfig(config); err != nil {
		return nil, fmt.Errorf("invalid trainer configuration: %v", err)
	}
	
	// Convert to CGO bridge config
	bridgeConfig := cgo_bridge.TrainingConfig{
		LearningRate:  config.LearningRate,
		Beta1:         config.Beta1,
		Beta2:         config.Beta2,
		WeightDecay:   config.WeightDecay,
		Epsilon:       config.Epsilon,
		OptimizerType: config.OptimizerType,
	}
	
	// Create model training engine
	var modelEngine *engine.ModelTrainingEngine
	var err error
	
	if config.OptimizerType == cgo_bridge.Adam {
		// Create with Adam optimizer
		adamConfig := map[string]interface{}{
			"learning_rate": config.LearningRate,
			"beta1":         config.Beta1,
			"beta2":         config.Beta2,
			"epsilon":       config.Epsilon,
			"weight_decay":  config.WeightDecay,
		}
		
		modelEngine, err = engine.NewModelTrainingEngineWithAdam(modelSpec, bridgeConfig, adamConfig)
	} else {
		// Create with SGD optimizer
		modelEngine, err = engine.NewModelTrainingEngine(modelSpec, bridgeConfig)
	}
	
	if err != nil {
		return nil, fmt.Errorf("failed to create model training engine: %v", err)
	}
	
	return &ModelTrainer{
		modelEngine: modelEngine,
		modelSpec:   modelSpec,
		batchSize:   config.BatchSize,
		config:      bridgeConfig,
		currentStep: 0,
	}, nil
}

// TrainBatch executes a single training step on a batch of data
// This maintains the single-CGO-call principle while supporting flexible layer models
func (mt *ModelTrainer) TrainBatch(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResult, error) {
	start := time.Now()
	defer func() {
		mt.lastStepTime = time.Since(start)
	}()
	
	// Create input tensor and copy data to GPU
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Release()
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	// Create label tensor (one-hot encoded for loss computation)
	oneHotShape := []int{labelShape[0], mt.getOutputSize()}
	labelTensor, err := memory.NewTensor(oneHotShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create label tensor: %v", err)
	}
	defer labelTensor.Release()
	
	// Convert labels to one-hot format
	oneHotData := mt.labelsToOneHot(labelData, oneHotShape)
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), oneHotData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy label data to GPU: %v", err)
	}
	
	// Execute training step using the appropriate optimizer
	var loss float32
	if mt.config.OptimizerType == cgo_bridge.Adam {
		loss, err = mt.modelEngine.ExecuteModelTrainingStepWithAdam(inputTensor, labelTensor)
	} else {
		loss, err = mt.modelEngine.ExecuteModelTrainingStep(inputTensor, labelTensor, mt.config.LearningRate)
	}
	
	if err != nil {
		return nil, fmt.Errorf("model training step failed: %v", err)
	}
	
	// Update statistics
	mt.currentStep++
	mt.totalSteps++
	mt.totalLoss += float64(loss)
	mt.averageLoss = float32(mt.totalLoss / float64(mt.totalSteps))
	
	return &TrainingResult{
		Loss:      loss,
		BatchSize: mt.batchSize,
		StepTime:  mt.lastStepTime,
		Success:   true,
		BatchRate: float64(mt.batchSize) / mt.lastStepTime.Seconds(),
	}, nil
}

// labelsToOneHot converts integer labels to one-hot encoded format
func (mt *ModelTrainer) labelsToOneHot(labels []int32, oneHotShape []int) []float32 {
	batchSize := oneHotShape[0]
	numClasses := oneHotShape[1]
	
	oneHot := make([]float32, batchSize*numClasses)
	
	for i, label := range labels {
		if i >= batchSize {
			break
		}
		
		// Zero out the row
		baseIdx := i * numClasses
		for j := 0; j < numClasses; j++ {
			oneHot[baseIdx+j] = 0.0
		}
		
		// Set the correct class to 1.0
		if int(label) < numClasses {
			oneHot[baseIdx+int(label)] = 1.0
		}
	}
	
	return oneHot
}

// getOutputSize gets the number of output classes from the model
func (mt *ModelTrainer) getOutputSize() int {
	// Find the last Dense layer to get output size
	for i := len(mt.modelSpec.Layers) - 1; i >= 0; i-- {
		layer := mt.modelSpec.Layers[i]
		if layer.Type == layers.Dense {
			if outputSize, ok := layer.Parameters["output_size"].(int); ok {
				return outputSize
			}
		}
	}
	
	// Default to 2 classes if not found
	return 2
}

// GetStats returns comprehensive training statistics
func (mt *ModelTrainer) GetStats() *ModelTrainingStats {
	memStats := memory.GetGlobalMemoryManager().Stats()
	
	return &ModelTrainingStats{
		CurrentStep:      mt.currentStep,
		TotalSteps:       mt.totalSteps,
		BatchSize:        mt.batchSize,
		OptimizerType:    mt.config.OptimizerType,
		LearningRate:     mt.config.LearningRate,
		AverageLoss:      mt.averageLoss,
		LastStepTime:     mt.lastStepTime,
		ModelSummary:     mt.modelEngine.GetModelSummary(),
		MemoryPoolStats:  memStats,
		ModelParameters:  mt.modelSpec.TotalParameters,
		LayerCount:       int64(len(mt.modelSpec.Layers)),
	}
}

// GetModelSpec returns the model specification
func (mt *ModelTrainer) GetModelSpec() *layers.ModelSpec {
	return mt.modelSpec
}

// GetModelSummary returns a human-readable model summary
func (mt *ModelTrainer) GetModelSummary() string {
	return mt.modelEngine.GetModelSummary()
}

// CreateTrainingSession creates a training session with progress visualization
func (mt *ModelTrainer) CreateTrainingSession(
	modelName string,
	epochs int,
	stepsPerEpoch int,
	validationSteps int,
) *TrainingSession {
	return NewTrainingSession(mt, modelName, epochs, stepsPerEpoch, validationSteps)
}

// PrintModelArchitecture prints the model architecture in PyTorch style
func (mt *ModelTrainer) PrintModelArchitecture(modelName string) {
	printer := NewModelArchitecturePrinter(modelName)
	printer.PrintArchitecture(mt.modelSpec)
}

// Cleanup releases all resources
func (mt *ModelTrainer) Cleanup() {
	if mt.modelEngine != nil {
		mt.modelEngine.Cleanup()
		mt.modelEngine = nil
	}
}

// ModelTrainingStats provides comprehensive statistics for model-based training
type ModelTrainingStats struct {
	CurrentStep      int
	TotalSteps       int64
	BatchSize        int
	OptimizerType    cgo_bridge.OptimizerType
	LearningRate     float32
	AverageLoss      float32
	LastStepTime     time.Duration
	ModelSummary     string
	MemoryPoolStats  map[memory.PoolKey]string
	ModelParameters  int64
	LayerCount       int64
}

// ModelTrainerFactory provides methods to create model trainers with different configurations
type ModelTrainerFactory struct{}

// NewModelFactory creates a new model trainer factory
func NewModelFactory() *ModelTrainerFactory {
	return &ModelTrainerFactory{}
}

// CreateModelTrainer creates a model trainer with full configuration control
func (mtf *ModelTrainerFactory) CreateModelTrainer(
	modelSpec *layers.ModelSpec,
	config TrainerConfig,
) (*ModelTrainer, error) {
	return NewModelTrainer(modelSpec, config)
}

// CreateCNNTrainer creates a CNN trainer with typical architecture
func (mtf *ModelTrainerFactory) CreateCNNTrainer(
	inputShape []int,
	numClasses int,
	config TrainerConfig,
) (*ModelTrainer, error) {
	// Build a typical CNN model
	builder := layers.NewModelBuilder(inputShape)
	
	// Add layers
	model, err := builder.
		AddConv2D(8, 3, 1, 1, true, "conv1").    // 8 filters, 3x3 kernel, stride=1, padding=1
		AddReLU("relu1").
		AddDense(numClasses, true, "fc1").       // Fully connected to output classes
		AddSoftmax(-1, "softmax").               // Softmax on last dimension
		Compile()
	
	if err != nil {
		return nil, fmt.Errorf("failed to compile CNN model: %v", err)
	}
	
	return mtf.CreateModelTrainer(model, config)
}

// CreateMLPTrainer creates a multi-layer perceptron trainer
func (mtf *ModelTrainerFactory) CreateMLPTrainer(
	inputSize int,
	hiddenSizes []int,
	outputSize int,
	config TrainerConfig,
) (*ModelTrainer, error) {
	// Build MLP model
	inputShape := []int{config.BatchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)
	
	// Add hidden layers
	for i, hiddenSize := range hiddenSizes {
		layerName := fmt.Sprintf("hidden_%d", i+1)
		builder.AddDense(hiddenSize, true, layerName)
		
		reluName := fmt.Sprintf("relu_%d", i+1)
		builder.AddReLU(reluName)
	}
	
	// Add output layer
	builder.AddDense(outputSize, true, "output")
	builder.AddSoftmax(-1, "softmax")
	
	model, err := builder.Compile()
	if err != nil {
		return nil, fmt.Errorf("failed to compile MLP model: %v", err)
	}
	
	return mtf.CreateModelTrainer(model, config)
}

// validateTrainerConfig validates the trainer configuration (reusing existing validation)
func validateTrainerConfig(config TrainerConfig) error {
	if config.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive, got %d", config.BatchSize)
	}
	
	if config.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive, got %f", config.LearningRate)
	}
	
	if config.OptimizerType == cgo_bridge.Adam {
		if config.Beta1 <= 0 || config.Beta1 >= 1 {
			return fmt.Errorf("Adam beta1 must be in (0, 1), got %f", config.Beta1)
		}
		if config.Beta2 <= 0 || config.Beta2 >= 1 {
			return fmt.Errorf("Adam beta2 must be in (0, 1), got %f", config.Beta2)
		}
		if config.Epsilon <= 0 {
			return fmt.Errorf("Adam epsilon must be positive, got %f", config.Epsilon)
		}
	}
	
	if config.WeightDecay < 0 {
		return fmt.Errorf("weight decay must be non-negative, got %f", config.WeightDecay)
	}
	
	return nil
}