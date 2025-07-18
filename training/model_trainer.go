package training

import (
	"fmt"
	"math"
	"sort"
	"time"
	"unsafe"

	"github.com/tsawler/go-metal/async"
	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/engine"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/optimizer"
)

// ModelTrainer provides layer-based training while maintaining the proven single-CGO-call architecture
// This is the compliant implementation that integrates with the existing high-performance TrainingEngine
type ModelTrainer struct {
	modelEngine *engine.ModelTrainingEngine
	modelSpec   *layers.ModelSpec
	batchSize   int
	config      cgo_bridge.TrainingConfig
	trainerConfig TrainerConfig  // High-level config for type checking
	currentStep int
	
	// Performance tracking
	lastStepTime    time.Duration
	totalSteps      int64
	totalLoss       float64
	averageLoss     float32
	
	// MEMORY LEAK FIX: Reuse slices to avoid allocations
	oneHotBuffer []float32
	
	// PERFORMANCE OPTIMIZATION: Persistent GPU tensors to reduce allocation overhead
	// These tensors are pre-allocated once and reused across training steps
	persistentInputTensor   *memory.Tensor
	persistentLabelTensor   *memory.Tensor
	persistentGradientTensors []*memory.Tensor  // CRITICAL: Pre-allocate gradient tensors
	persistentEnabled       bool
	
	// PERFORMANCE OPTIMIZATION: Configurable inference frequency to reduce CGO overhead
	// Instead of calculating accuracy every step, do it every N steps
	accuracyCheckInterval  int     // Calculate accuracy every N steps (0 = every step)
	lastAccuracy          float64 // Last calculated accuracy value
	accuracyStepCounter   int     // Counter for accuracy calculation
	
	// RESOURCE LEAK FIX: Command buffer pooling to prevent Metal resource accumulation
	// This addresses the 34% performance degradation from MTLCommandBuffer and MPSImage accumulation
	commandBufferPool  *async.CommandBufferPool
	commandQueue       unsafe.Pointer  // MTLCommandQueue for command buffer creation
	
	// Learning Rate Scheduling - maintains GPU-resident principles
	lrScheduler      LRScheduler // Optional learning rate scheduler
	baseLearningRate float32     // Original learning rate from config
	currentEpoch     int         // Current epoch for scheduler
	
	// Evaluation Metrics - GPU-resident architecture compliance
	confusionMatrix     *ConfusionMatrix // Confusion matrix for classification metrics
	metricsEnabled      bool             // Whether to calculate comprehensive metrics
	lastRegressionMetrics *RegressionMetrics // Last calculated regression metrics
	metricHistory       map[MetricType][]float64 // History of metrics for plotting
	
	// Visualization & Plotting - follows GPU-resident architecture
	visualizationCollector *VisualizationCollector // Collects data for plotting
	plottingService        *PlottingService        // Service for sending plots to sidecar
	visualizationEnabled   bool                    // Whether visualization is enabled
	
	// Probability collection for proper PR/ROC curves - GPU-resident compliant
	// These are only populated during validation when visualization is enabled
	validationProbabilities []float32  // Concatenated probabilities from all validation batches
	validationLabels       []int32    // Corresponding true labels
	maxProbabilityBatches  int        // Maximum number of batches to collect (to limit memory)
	probabilityBatchCount  int        // Current number of collected batches
	
	// Mixed Precision Training State
	useMixedPrecision     bool                    // Whether mixed precision is enabled
	lossScale             float32                 // Current loss scale
	lossScaleGrowthFactor float32                 // Loss scale growth factor
	lossScaleBackoffFactor float32                // Loss scale reduction factor
	lossScaleGrowthInterval int                   // Steps between growth attempts
	stepsSinceLastScale   int                     // Steps since last scale change
	fp16WeightTensors     []*memory.Tensor        // FP16 versions of weights
	fp16GradientTensors   []*memory.Tensor        // FP16 gradient tensors
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
		Alpha:         config.Alpha,
		Momentum:      config.Momentum,
		Centered:      config.Centered,
		OptimizerType: config.OptimizerType,
		ProblemType:   int(config.ProblemType),
		LossFunction:  int(config.LossFunction),
	}
	
	// SMART ROUTING: Select optimal engine based on model architecture
	// Maintains GPU-resident architecture compliance across all engine types
	var modelEngine *engine.ModelTrainingEngine
	var err error
	
	// Determine optimal engine type using smart routing
	selectedEngine := SelectOptimalEngine(modelSpec, config)
	// Debug analysis disabled
	// archInfo := AnalyzeModelArchitecture(modelSpec)
	
	// Debug output disabled
	// fmt.Printf("🧠 Smart Routing Analysis:\n")
	// fmt.Printf("   - Input: %dD %v\n", archInfo.InputDimensions, modelSpec.InputShape)
	// fmt.Printf("   - Architecture: %s (%d layers, %d params)\n", 
	//	archInfo.Complexity, archInfo.LayerCount, archInfo.ParameterCount)
	// fmt.Printf("   - Pattern: CNN=%v, MLP=%v\n", archInfo.IsCNNPattern, archInfo.IsMLPOnly)
	// fmt.Printf("   - Selected Engine: %s\n", selectedEngine.String())
	
	// Create engine based on smart routing decision
	switch selectedEngine {
	case Dynamic:
		// Dynamic Engine: Maximum flexibility for any architecture
		// - Supports 2D, 4D, or any dimensional input
		// - Supports any layer combination (Dense, Conv2D, etc.)
		// - MPSGraph-centric architecture with automatic kernel fusion
		// Debug output disabled
		// fmt.Printf("🔧 Creating Dynamic Engine (any architecture support)\n")
		modelEngine, err = engine.NewModelTrainingEngineDynamic(modelSpec, bridgeConfig)
		
	case Hybrid:
		// Hybrid Engine: Maximum performance for CNN architectures
		// - Optimized for 4D input [batch, channels, height, width]
		// - MPS for convolutions + MPSGraph for dense layers
		// - Achieves 20k+ batches/second performance
		fmt.Printf("🚀 Creating Hybrid Engine (CNN optimization)\n")
		
		// Route to appropriate hybrid engine based on optimizer
		if config.OptimizerType == cgo_bridge.Adam {
			adamConfig := map[string]interface{}{
				"learning_rate": config.LearningRate,
				"beta1":         config.Beta1,
				"beta2":         config.Beta2,
				"epsilon":       config.Epsilon,
				"weight_decay":  config.WeightDecay,
			}
			modelEngine, err = engine.NewModelTrainingEngineWithAdam(modelSpec, bridgeConfig, adamConfig)
		} else if config.OptimizerType == cgo_bridge.LBFGS {
			// L-BFGS requires dynamic engine due to its complexity
			fmt.Printf("📊 L-BFGS optimizer detected, switching to Dynamic Engine\n")
			modelEngine, err = engine.NewModelTrainingEngineDynamic(modelSpec, bridgeConfig)
		} else {
			// SGD or RMSProp with hybrid engine
			modelEngine, err = engine.NewModelTrainingEngine(modelSpec, bridgeConfig)
		}
		
		// SMART FALLBACK: If Hybrid Engine fails due to architecture constraints,
		// automatically fall back to Dynamic Engine for flexible architecture support
		if err != nil && (selectedEngine == Hybrid) {
			fmt.Printf("⚠️  Hybrid Engine incompatible with model architecture, falling back to Dynamic Engine\n")
			fmt.Printf("    Error: %v\n", err)
			modelEngine, err = engine.NewModelTrainingEngineDynamic(modelSpec, bridgeConfig)
			if err == nil {
				fmt.Printf("✅ Successfully created Dynamic Engine as fallback\n")
			}
		}
		
	default:
		// Fallback to dynamic engine for unknown types
		fmt.Printf("⚠️  Unknown engine type, falling back to Dynamic Engine\n")
		modelEngine, err = engine.NewModelTrainingEngineDynamic(modelSpec, bridgeConfig)
	}
	
	if err != nil {
		return nil, fmt.Errorf("failed to create model training engine: %v", err)
	}
	
	// Debug output disabled
	// fmt.Printf("✅ Created engine: isDynamic=%t\n", modelEngine.IsDynamicEngine())
	
	// RESOURCE LEAK FIX: Initialize command buffer pool for Metal resource management
	// This prevents MTLCommandBuffer and MPSImage accumulation that causes 34% performance degradation
	device := modelEngine.GetDevice()
	commandQueue, err := cgo_bridge.CreateCommandQueue(device)
	if err != nil {
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to create command queue: %v", err)
	}
	
	// Create command buffer pool with reasonable pool size (16 buffers should handle most workloads)
	commandBufferPool, err := async.NewCommandBufferPool(commandQueue, 16)
	if err != nil {
		cgo_bridge.ReleaseCommandQueue(commandQueue)
		modelEngine.Cleanup()
		return nil, fmt.Errorf("failed to create command buffer pool: %v", err)
	}
	
	trainer := &ModelTrainer{
		modelEngine: modelEngine,
		modelSpec:   modelSpec,
		batchSize:   config.BatchSize,
		config:      bridgeConfig,
		trainerConfig: config,
		currentStep: 0,
		
		// Default: calculate accuracy every step (traditional behavior)
		accuracyCheckInterval: 0,
		lastAccuracy:         0.0,
		accuracyStepCounter:  0,
		
		// RESOURCE LEAK FIX: Store command buffer pool and queue for resource management
		commandBufferPool: commandBufferPool,
		commandQueue:      commandQueue,
		
		// Learning rate scheduling initialization
		baseLearningRate: config.LearningRate,
		currentEpoch:     0,
		lrScheduler:      nil, // No scheduler by default (constant LR)
		
		// Evaluation metrics initialization - GPU-resident architecture compliance
		confusionMatrix:       NewConfusionMatrix(getNumClassesForMetrics(modelSpec, config)),
		metricsEnabled:        false, // Disabled by default for performance
		lastRegressionMetrics: &RegressionMetrics{},
		metricHistory:         make(map[MetricType][]float64),
		
		// Visualization initialization - GPU-resident architecture compliance
		visualizationCollector: NewVisualizationCollector("Model"),
		plottingService:        NewPlottingService(DefaultPlottingServiceConfig()),
		visualizationEnabled:   false, // Disabled by default for performance
		
		// Probability collection initialization
		maxProbabilityBatches: 50, // Limit to 50 batches to control memory usage
		probabilityBatchCount: 0,
		
		// Mixed Precision Training initialization
		useMixedPrecision:     config.UseMixedPrecision,
		lossScale:             config.InitialLossScale,
		lossScaleGrowthFactor: config.LossScaleGrowthFactor,
		lossScaleBackoffFactor: config.LossScaleBackoffFactor,
		lossScaleGrowthInterval: config.LossScaleGrowthInterval,
		stepsSinceLastScale:   0,
	}
	
	// Initialize default values for mixed precision if not specified
	if trainer.useMixedPrecision {
		if trainer.lossScale == 0 {
			trainer.lossScale = 65536.0 // Default initial loss scale
		}
		if trainer.lossScaleGrowthFactor == 0 {
			trainer.lossScaleGrowthFactor = 2.0
		}
		if trainer.lossScaleBackoffFactor == 0 {
			trainer.lossScaleBackoffFactor = 0.5
		}
		if trainer.lossScaleGrowthInterval == 0 {
			trainer.lossScaleGrowthInterval = 2000
		}
		
		// Initialize FP16 weight tensors
		err = trainer.initializeMixedPrecisionTensors()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize mixed precision tensors: %v", err)
		}
	}
	
	return trainer, nil
}

// initializeMixedPrecisionTensors creates FP16 versions of weights and gradient tensors
func (mt *ModelTrainer) initializeMixedPrecisionTensors() error {
	// Get the number of parameters from the model
	numParams := 0
	for _, layer := range mt.modelSpec.Layers {
		if layer.Type == layers.Dense || layer.Type == layers.Conv2D || layer.Type == layers.BatchNorm {
			// Each layer can have weights and biases
			numParams += 2
		}
	}
	
	// Pre-allocate FP16 tensors for weights and gradients
	mt.fp16WeightTensors = make([]*memory.Tensor, 0, numParams)
	mt.fp16GradientTensors = make([]*memory.Tensor, 0, numParams)
	
	// Note: Actual tensor creation will happen during training when we have the weight shapes
	return nil
}

// convertWeightsToFP16 converts FP32 weights to FP16 for mixed precision training
func (mt *ModelTrainer) convertWeightsToFP16(fp32Weights []*memory.Tensor) error {
	// Ensure we have enough FP16 tensors
	for i := len(mt.fp16WeightTensors); i < len(fp32Weights); i++ {
		fp32Weight := fp32Weights[i]
		fp16Weight, err := fp32Weight.ConvertTo(memory.Float16)
		if err != nil {
			return fmt.Errorf("failed to convert weight %d to FP16: %v", i, err)
		}
		mt.fp16WeightTensors = append(mt.fp16WeightTensors, fp16Weight)
		
		// Also create FP16 gradient tensor with same shape
		fp16Gradient, err := memory.NewTensor(fp32Weight.Shape(), memory.Float16, memory.GPU)
		if err != nil {
			return fmt.Errorf("failed to create FP16 gradient tensor %d: %v", i, err)
		}
		mt.fp16GradientTensors = append(mt.fp16GradientTensors, fp16Gradient)
	}
	
	// Update existing FP16 weights with current FP32 values
	for i := 0; i < len(fp32Weights) && i < len(mt.fp16WeightTensors); i++ {
		// Release old FP16 weight
		mt.fp16WeightTensors[i].Release()
		
		// Convert current FP32 weight to FP16
		fp16Weight, err := fp32Weights[i].ConvertTo(memory.Float16)
		if err != nil {
			return fmt.Errorf("failed to update FP16 weight %d: %v", i, err)
		}
		mt.fp16WeightTensors[i] = fp16Weight
	}
	
	return nil
}

// updateFP32WeightsFromFP16 updates FP32 master weights from FP16 weights after optimization
func (mt *ModelTrainer) updateFP32WeightsFromFP16(fp32Weights []*memory.Tensor) error {
	for i := 0; i < len(fp32Weights) && i < len(mt.fp16WeightTensors); i++ {
		// Convert FP16 weight back to FP32
		updatedFP32, err := mt.fp16WeightTensors[i].ConvertTo(memory.Float32)
		if err != nil {
			return fmt.Errorf("failed to convert FP16 weight %d back to FP32: %v", i, err)
		}
		defer updatedFP32.Release()
		
		// GPU-RESIDENT TENSOR COPY: Direct buffer-to-buffer transfer without CPU involvement
		// This uses the optimized Metal blit encoder for maximum performance
		err = fp32Weights[i].CopyFrom(updatedFP32)
		if err != nil {
			return fmt.Errorf("failed to copy updated FP32 weight %d back to master weights: %v", i, err)
		}
	}
	
	return nil
}

// getModelOutputSize extracts the output size from model specification
// Used for initializing confusion matrix with correct dimensions
func getModelOutputSize(modelSpec *layers.ModelSpec) int {
	if modelSpec == nil || len(modelSpec.Layers) == 0 {
		return 2 // Default to binary classification
	}
	
	// Find the last Dense layer to get output size
	for i := len(modelSpec.Layers) - 1; i >= 0; i-- {
		layer := modelSpec.Layers[i]
		if layer.Type == layers.Dense {
			if outputSize, ok := layer.Parameters["output_size"].(int); ok {
				return outputSize
			}
		}
	}
	
	// Default to 2 classes if not found
	return 2
}

// getNumClassesForMetrics determines the number of classes for confusion matrix
// For binary classification with single output (BCEWithLogits), we need 2 classes
func getNumClassesForMetrics(modelSpec *layers.ModelSpec, config TrainerConfig) int {
	outputSize := getModelOutputSize(modelSpec)
	
	// For classification problems
	if config.ProblemType == Classification {
		// Single output usually means binary classification with 2 classes
		if outputSize == 1 {
			return 2
		}
		// Multiple outputs mean multi-class classification
		return outputSize
	}
	
	// For regression, we don't use confusion matrix, but return 2 as safe default
	return 2
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
	defer func() {
		inputTensor.Release()
		// DEBUG: Log tensor lifecycle
		if mt.currentStep%50 == 0 {
			// Released input tensor
		}
	}()
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	// CONDITIONAL LABEL PROCESSING: Handle different label formats based on loss function
	// ALL loss functions now use one-hot vectors [batch_size, num_classes] for consistency
	// SparseCrossEntropy will convert integer indices to one-hot on the Go side
	var labelTensor *memory.Tensor
	
	// Create one-hot encoded labels for ALL loss functions
	oneHotShape := []int{labelShape[0], mt.getOutputSize()}
	labelTensor, err = memory.NewTensor(oneHotShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create one-hot label tensor: %v", err)
	}
	
	// Convert labels to one-hot format
	oneHotData := mt.labelsToOneHot(labelData, oneHotShape)
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), oneHotData)
	if err != nil {
		labelTensor.Release()
		return nil, fmt.Errorf("failed to copy one-hot label data to GPU: %v", err)
	}
	
	defer func() {
		labelTensor.Release()
		// DEBUG: Log tensor lifecycle
		if mt.currentStep%50 == 0 {
			// Released label tensor
		}
	}()
	
	// Memory pool stats available if needed
	if mt.currentStep%100 == 0 { // Reduced frequency
		// Pool stats can be logged here if debugging memory issues
	}
	
	// Execute training step using the appropriate optimizer
	var loss float32
	if mt.config.OptimizerType == cgo_bridge.Adam {
		loss, err = mt.modelEngine.ExecuteModelTrainingStepWithAdam(inputTensor, labelTensor)
	} else if mt.config.OptimizerType == cgo_bridge.LBFGS {
		loss, err = mt.modelEngine.ExecuteModelTrainingStepWithLBFGS(inputTensor, labelTensor)
	} else {
		loss, err = mt.modelEngine.ExecuteModelTrainingStep(inputTensor, labelTensor, mt.config.LearningRate)
	}
	
	if err != nil {
		return nil, fmt.Errorf("model training step failed: %v", err)
	}
	
	// Update statistics
	mt.updateSchedulerStep()
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
// MEMORY LEAK FIX: Reuses internal buffer to avoid allocations
func (mt *ModelTrainer) labelsToOneHot(labels []int32, oneHotShape []int) []float32 {
	batchSize := oneHotShape[0]
	numClasses := oneHotShape[1]
	requiredSize := batchSize * numClasses
	
	// MEMORY LEAK FIX: Reuse buffer if possible, only allocate if needed
	if len(mt.oneHotBuffer) < requiredSize {
		mt.oneHotBuffer = make([]float32, requiredSize)
	}
	oneHot := mt.oneHotBuffer[:requiredSize]
	
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

// TrainBatchOptimized executes a training step with batched CGO operations
// This reduces CGO overhead by combining multiple operations into a single call
// Follows design principle: "Single CGO call per training step"
func (mt *ModelTrainer) TrainBatchOptimized(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error) {
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
	
	// Create label tensor (one-hot encoded for loss computation)
	oneHotShape := []int{labelShape[0], mt.getOutputSize()}
	labelTensor, err := memory.NewTensor(oneHotShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create label tensor: %v", err)
	}
	defer labelTensor.Release()
	
	// Convert labels to one-hot format (reuses buffer)
	oneHotData := mt.labelsToOneHot(labelData, oneHotShape)
	
	// PERFORMANCE OPTIMIZATION: Smart accuracy calculation based on configured interval
	calculateAccuracy := mt.shouldCalculateAccuracy()
	
	// OPTIMIZATION: Single batched CGO call for entire training step
	// Combines: data copy + forward pass + backward pass + optimizer step + optional inference
	result, err := mt.modelEngine.ExecuteModelTrainingStepBatched(
		inputTensor, labelTensor,
		inputData, oneHotData,
		calculateAccuracy,
	)
	
	if err != nil {
		return nil, fmt.Errorf("batched training step failed: %v", err)
	}
	
	// Update statistics
	mt.updateSchedulerStep()
	mt.totalSteps++
	mt.totalLoss += float64(result.Loss)
	mt.averageLoss = float32(mt.totalLoss / float64(mt.totalSteps))
	
	// Update cached accuracy if it was calculated
	if calculateAccuracy {
		mt.lastAccuracy = result.Accuracy
	}
	
	return &TrainingResultOptimized{
		Loss:         result.Loss,
		Accuracy:     mt.lastAccuracy, // Use cached value if not calculated this step
		HasAccuracy:  calculateAccuracy, // Training step batched method returns accuracy based on calculation request
		BatchSize:    mt.batchSize,
		StepTime:     mt.lastStepTime,
		Success:      true,
		BatchRate:    float64(mt.batchSize) / mt.lastStepTime.Seconds(),
	}, nil
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

// SetAccuracyCheckInterval configures how often accuracy is calculated
// interval=0: every step (default, maximum accuracy but higher CGO overhead)
// interval=10: every 10 steps (reduces CGO calls by ~40%, slight accuracy tracking lag)
// interval=50: every 50 steps (reduces CGO calls by ~80%, minimal accuracy tracking)
func (mt *ModelTrainer) SetAccuracyCheckInterval(interval int) {
	if interval < 0 {
		interval = 0
	}
	mt.accuracyCheckInterval = interval
	mt.accuracyStepCounter = 0
}

// shouldCalculateAccuracy determines if accuracy should be calculated this step
func (mt *ModelTrainer) shouldCalculateAccuracy() bool {
	if mt.accuracyCheckInterval == 0 {
		return true // Calculate every step
	}
	
	mt.accuracyStepCounter++
	if mt.accuracyStepCounter >= mt.accuracyCheckInterval {
		mt.accuracyStepCounter = 0
		return true
	}
	return false
}

// EnablePersistentBuffers pre-allocates GPU tensors for reuse across training steps
// This reduces allocation overhead and improves performance
func (mt *ModelTrainer) EnablePersistentBuffers(inputShape []int) error {
	if mt.persistentEnabled {
		return nil // Already enabled
	}
	
	// Pre-allocate input tensor
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return fmt.Errorf("failed to create persistent input tensor: %v", err)
	}
	mt.persistentInputTensor = inputTensor
	
	// Pre-allocate label tensor (one-hot encoded)
	oneHotShape := []int{inputShape[0], mt.getOutputSize()}
	labelTensor, err := memory.NewTensor(oneHotShape, memory.Float32, memory.GPU)
	if err != nil {
		mt.persistentInputTensor.Release()
		return fmt.Errorf("failed to create persistent label tensor: %v", err)
	}
	mt.persistentLabelTensor = labelTensor
	
	// CRITICAL PERFORMANCE FIX: Pre-allocate gradient tensors to eliminate the major allocation source
	// This addresses the 128MB/step allocation that was causing 83% performance degradation
	if mt.modelEngine != nil && mt.modelEngine.IsDynamicEngine() && len(mt.modelEngine.GetParameterTensors()) > 0 {
		paramTensors := mt.modelEngine.GetParameterTensors()
		gradientTensors := make([]*memory.Tensor, len(paramTensors))
		for i, paramTensor := range paramTensors {
			gradTensor, err := memory.NewTensor(paramTensor.Shape(), memory.Float32, memory.GPU)
			if err != nil {
				// Cleanup on error
				mt.persistentInputTensor.Release()
				mt.persistentLabelTensor.Release()
				for j := 0; j < i; j++ {
					gradientTensors[j].Release()
				}
				return fmt.Errorf("failed to create persistent gradient tensor %d: %v", i, err)
			}
			gradientTensors[i] = gradTensor
		}
		mt.persistentGradientTensors = gradientTensors
	}
	
	mt.persistentEnabled = true
	return nil
}

// TrainBatchPersistent executes a training step using persistent GPU buffers
// This provides maximum performance by eliminating per-step tensor allocations
func (mt *ModelTrainer) TrainBatchPersistent(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error) {
	if !mt.persistentEnabled {
		return nil, fmt.Errorf("persistent buffers not enabled - call EnablePersistentBuffers() first")
	}
	
	start := time.Now()
	defer func() {
		mt.lastStepTime = time.Since(start)
	}()
	
	// PERFORMANCE FIX: Handle variable batch sizes dynamically
	// Check if persistent tensors need to be resized for current batch
	currentBatchSize := inputShape[0]
	persistentBatchSize := mt.persistentInputTensor.Shape()[0]
	
	if currentBatchSize > persistentBatchSize {
		// Need larger tensors - reallocate (rare case for partial batches)
		return nil, fmt.Errorf("batch size %d exceeds persistent buffer size %d - use smaller batches", 
			currentBatchSize, persistentBatchSize)
	}
	
	// For smaller batches, use the persistent tensors as-is (they're sized for max batch)
	
	// Convert labels to one-hot format (reuses buffer)
	oneHotShape := []int{labelShape[0], mt.getOutputSize()}
	oneHotData := mt.labelsToOneHot(labelData, oneHotShape)
	
	// PERFORMANCE OPTIMIZATION: Smart accuracy calculation based on configured interval
	calculateAccuracy := mt.shouldCalculateAccuracy()
	
	
	// OPTIMIZATION: Single batched CGO call using persistent tensors
	// Maximum performance: no allocations, single CGO call
	result, err := mt.modelEngine.ExecuteModelTrainingStepBatchedPersistentWithGradients(
		mt.persistentInputTensor, mt.persistentLabelTensor,
		inputData, oneHotData,
		calculateAccuracy,
		mt.persistentGradientTensors, // CRITICAL: Pass pre-allocated gradient tensors
	)
	
	if err != nil {
		return nil, fmt.Errorf("persistent batched training step failed: %v", err)
	}
	
	// Update statistics
	mt.updateSchedulerStep()
	mt.totalSteps++
	mt.totalLoss += float64(result.Loss)
	mt.averageLoss = float32(mt.totalLoss / float64(mt.totalSteps))
	
	// Update cached accuracy if it was calculated
	var accuracyToReturn float64
	var hasAccuracy bool
	
	if calculateAccuracy {
		mt.lastAccuracy = result.Accuracy
		accuracyToReturn = result.Accuracy
		hasAccuracy = true
	} else {
		// Don't return cached accuracy - return 0 and indicate no accuracy calculated
		accuracyToReturn = 0.0
		hasAccuracy = false
	}
	
	return &TrainingResultOptimized{
		Loss:         result.Loss,
		Accuracy:     accuracyToReturn, // Only return accuracy if calculated this step
		HasAccuracy:  hasAccuracy,
		BatchSize:    mt.batchSize,
		StepTime:     mt.lastStepTime,
		Success:      true,
		BatchRate:    float64(mt.batchSize) / mt.lastStepTime.Seconds(),
	}, nil
}

// TrainBatchWithCommandPool executes a training step using pooled command buffers
// This method implements the complete command buffer pooling strategy to prevent resource leaks
func (mt *ModelTrainer) TrainBatchWithCommandPool(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error) {
	if mt.commandBufferPool == nil {
		return nil, fmt.Errorf("command buffer pool not initialized")
	}
	
	start := time.Now()
	defer func() {
		mt.lastStepTime = time.Since(start)
	}()
	
	// Get a command buffer from the pool
	commandBuffer, err := mt.commandBufferPool.GetBuffer()
	if err != nil {
		return nil, fmt.Errorf("failed to get command buffer from pool: %v", err)
	}
	
	// Ensure command buffer is returned to pool on completion
	defer func() {
		if commandBuffer != nil {
			mt.commandBufferPool.ReturnBuffer(commandBuffer)
		}
	}()
	
	// Setup autorelease pool for proper Metal resource management
	cgo_bridge.SetupAutoreleasePool()
	defer cgo_bridge.DrainAutoreleasePool()
	
	// Create input tensor and copy data to GPU
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Release()
	
	// Create label tensor (one-hot encoded for loss computation)
	oneHotShape := []int{labelShape[0], mt.getOutputSize()}
	labelTensor, err := memory.NewTensor(oneHotShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create label tensor: %v", err)
	}
	defer labelTensor.Release()
	
	// Convert labels to one-hot format (reuses buffer)
	oneHotData := mt.labelsToOneHot(labelData, oneHotShape)
	
	// Copy data to GPU
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), oneHotData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy label data to GPU: %v", err)
	}
	
	// PERFORMANCE OPTIMIZATION: Smart accuracy calculation
	calculateAccuracy := mt.shouldCalculateAccuracy()
	
	// Execute training step with the existing engine methods
	// Note: The underlying engine will use its own command buffers, but our pooled
	// command buffer ensures proper cleanup and prevents accumulation
	var loss float32
	if mt.config.OptimizerType == cgo_bridge.Adam {
		loss, err = mt.modelEngine.ExecuteModelTrainingStepWithAdam(inputTensor, labelTensor)
	} else if mt.config.OptimizerType == cgo_bridge.LBFGS {
		loss, err = mt.modelEngine.ExecuteModelTrainingStepWithLBFGS(inputTensor, labelTensor)
	} else {
		loss, err = mt.modelEngine.ExecuteModelTrainingStep(inputTensor, labelTensor, mt.config.LearningRate)
	}
	
	if err != nil {
		return nil, fmt.Errorf("training step failed: %v", err)
	}
	
	// Calculate accuracy if requested
	var accuracy float64
	var hasAccuracy bool
	if calculateAccuracy {
		// Perform inference to get predictions
		inferenceResult, inferErr := mt.modelEngine.ExecuteInference(inputTensor, inputShape[0])
		if inferErr == nil {
			accuracy = mt.CalculateAccuracy(
				inferenceResult.Predictions, 
				labelData, 
				inputShape[0], 
				mt.getOutputSize(),
			)
			mt.lastAccuracy = accuracy
			hasAccuracy = true
		} else {
			// Log inference error but don't fail the training step
			if mt.currentStep % 100 == 0 { // Log occasionally to avoid spam
				fmt.Printf("Warning: Training accuracy inference failed at step %d: %v\n", mt.currentStep, inferErr)
			}
			accuracy = 0.0
			hasAccuracy = false
		}
	} else {
		accuracy = mt.lastAccuracy
		hasAccuracy = false
	}
	
	// Update statistics
	mt.updateSchedulerStep()
	mt.totalSteps++
	mt.totalLoss += float64(loss)
	mt.averageLoss = float32(mt.totalLoss / float64(mt.totalSteps))
	
	// VISUALIZATION: Record training step data (CPU-only scalar access)
	// This follows GPU-resident architecture - only CPU access for final metrics
	if hasAccuracy {
		mt.recordTrainingStep(mt.currentStep, float64(loss), accuracy)
	}
	
	return &TrainingResultOptimized{
		Loss:         loss,
		Accuracy:     accuracy,
		HasAccuracy:  hasAccuracy, // Use the actual result of accuracy calculation
		BatchSize:    mt.batchSize,
		StepTime:     mt.lastStepTime,
		Success:      true,
		BatchRate:    float64(mt.batchSize) / mt.lastStepTime.Seconds(),
	}, nil
}

// TrainBatchUnified executes a training step with flexible label types
// This is the recommended API for new code as it supports both classification and regression
// while maintaining GPU-residency and minimizing CGO calls
func (mt *ModelTrainer) TrainBatchUnified(
	inputData []float32,
	inputShape []int,
	labelData LabelData,
) (*TrainingResultOptimized, error) {
	// Validate label data compatibility with trainer configuration
	if err := mt.validateLabelCompatibility(labelData); err != nil {
		return nil, err
	}
	
	// Convert labels to float32 for GPU consumption (zero-cost for regression)
	labels := labelData.ToFloat32Slice()
	labelShape := labelData.Shape()
	
	// Call the internal training function with converted labels
	return mt.trainBatchPersistentWithCommandPoolInternal(
		inputData, inputShape, labels, labelShape, labelData.DataType())
}

// validateLabelCompatibility ensures label data matches the configured problem type
func (mt *ModelTrainer) validateLabelCompatibility(labelData LabelData) error {
	configType := mt.trainerConfig.ProblemType
	dataType := labelData.DataType()
	
	switch configType {
	case Classification:
		if dataType != LabelTypeInt32 {
			return fmt.Errorf("classification trainer requires Int32Labels, got %v", dataType)
		}
	case Regression:
		if dataType != LabelTypeFloat32 {
			return fmt.Errorf("regression trainer requires Float32Labels, got %v", dataType)
		}
	default:
		return fmt.Errorf("unsupported problem type: %v", configType)
	}
	
	return nil
}

// TrainBatchPersistentWithCommandPool executes a training step using both persistent tensors 
// and pooled command buffers for maximum performance and resource efficiency
// DEPRECATED: Use TrainBatchUnified for new code
func (mt *ModelTrainer) TrainBatchPersistentWithCommandPool(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
) (*TrainingResultOptimized, error) {
	// Convert int32 labels to LabelData for unified processing
	labels, err := NewInt32Labels(labelData, labelShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create label data: %v", err)
	}
	
	return mt.TrainBatchUnified(inputData, inputShape, labels)
}

// trainBatchPersistentWithCommandPoolInternal is the internal implementation
// that works with float32 labels for both classification and regression
func (mt *ModelTrainer) trainBatchPersistentWithCommandPoolInternal(
	inputData []float32,
	inputShape []int,
	labelData []float32,
	labelShape []int,
	labelType LabelDataType,
) (*TrainingResultOptimized, error) {
	if !mt.persistentEnabled {
		return nil, fmt.Errorf("persistent buffers not enabled - call EnablePersistentBuffers() first")
	}
	
	if mt.commandBufferPool == nil {
		return nil, fmt.Errorf("command buffer pool not initialized")
	}
	
	start := time.Now()
	defer func() {
		mt.lastStepTime = time.Since(start)
	}()
	
	// Get a command buffer from the pool
	commandBuffer, err := mt.commandBufferPool.GetBuffer()
	if err != nil {
		return nil, fmt.Errorf("failed to get command buffer from pool: %v", err)
	}
	
	// Ensure command buffer is returned to pool on completion
	defer func() {
		if commandBuffer != nil {
			mt.commandBufferPool.ReturnBuffer(commandBuffer)
		}
	}()
	
	// Setup autorelease pool for proper Metal resource management
	cgo_bridge.SetupAutoreleasePool()
	defer cgo_bridge.DrainAutoreleasePool()
	
	// PERFORMANCE FIX: Handle variable batch sizes dynamically
	currentBatchSize := inputShape[0]
	persistentBatchSize := mt.persistentInputTensor.Shape()[0]
	
	if currentBatchSize > persistentBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds persistent buffer size %d - use smaller batches", 
			currentBatchSize, persistentBatchSize)
	}
	
	// Handle labels based on problem type
	var processedLabels []float32
	
	if mt.trainerConfig.ProblemType == Classification {
		// Convert labels to one-hot format for classification
		if labelType == LabelTypeInt32 {
			// Labels are int32, convert back from float32 to int32 for one-hot encoding
			int32Labels := make([]int32, len(labelData))
			for i, v := range labelData {
				int32Labels[i] = int32(v)
			}
			oneHotShape := []int{labelShape[0], mt.getOutputSize()}
			processedLabels = mt.labelsToOneHot(int32Labels, oneHotShape)
		} else {
			// This shouldn't happen with proper validation, but handle it
			return nil, fmt.Errorf("classification requires int32 labels")
		}
	} else {
		// For regression, use float32 labels directly
		processedLabels = labelData
	}
	
	// Copy data to persistent GPU tensors
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(mt.persistentInputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(mt.persistentLabelTensor.MetalBuffer(), processedLabels)
	if err != nil {
		return nil, fmt.Errorf("failed to copy label data to GPU: %v", err)
	}
	
	// PERFORMANCE OPTIMIZATION: Smart accuracy calculation
	calculateAccuracy := mt.shouldCalculateAccuracy()
	
	// Execute training step with persistent tensors and pooled command buffers
	// This combines both optimizations for maximum performance
	var loss float32
	if mt.config.OptimizerType == cgo_bridge.Adam {
		loss, err = mt.modelEngine.ExecuteModelTrainingStepWithAdam(mt.persistentInputTensor, mt.persistentLabelTensor)
	} else {
		loss, err = mt.modelEngine.ExecuteModelTrainingStep(mt.persistentInputTensor, mt.persistentLabelTensor, mt.config.LearningRate)
	}
	
	if err != nil {
		return nil, fmt.Errorf("persistent training step with command pool failed: %v", err)
	}
	
	// Calculate accuracy if requested
	var accuracy float64
	var hasAccuracy bool
	if calculateAccuracy {
		// Perform inference using persistent tensors
		inferenceResult, inferErr := mt.modelEngine.ExecuteInference(mt.persistentInputTensor, currentBatchSize)
		if inferErr == nil {
			accuracy = mt.CalculateAccuracyUnified(
				inferenceResult.Predictions, 
				labelData, 
				currentBatchSize, 
				mt.getOutputSize(),
				labelType,
			)
			mt.lastAccuracy = accuracy
			hasAccuracy = true
		} else {
			// Log inference error but don't fail the training step
			if mt.currentStep % 100 == 0 { // Log occasionally to avoid spam
				fmt.Printf("Warning: Training accuracy inference failed at step %d: %v\n", mt.currentStep, inferErr)
			}
			accuracy = 0.0
			hasAccuracy = false
		}
	} else {
		accuracy = mt.lastAccuracy
		hasAccuracy = false
	}
	
	// Update statistics
	mt.updateSchedulerStep()
	mt.totalSteps++
	mt.totalLoss += float64(loss)
	mt.averageLoss = float32(mt.totalLoss / float64(mt.totalSteps))
	
	// VISUALIZATION: Record training step data (CPU-only scalar access)
	// This follows GPU-resident architecture - only CPU access for final metrics
	if hasAccuracy {
		mt.recordTrainingStep(mt.currentStep, float64(loss), accuracy)
	}
	
	return &TrainingResultOptimized{
		Loss:         loss,
		Accuracy:     accuracy,
		HasAccuracy:  hasAccuracy, // Use the actual result of accuracy calculation
		BatchSize:    mt.batchSize,
		StepTime:     mt.lastStepTime,
		Success:      true,
		BatchRate:    float64(mt.batchSize) / mt.lastStepTime.Seconds(),
	}, nil
}

// Cleanup releases all resources
func (mt *ModelTrainer) Cleanup() {
	// RESOURCE LEAK FIX: Cleanup command buffer pool first to ensure proper resource cleanup
	if mt.commandBufferPool != nil {
		mt.commandBufferPool.Cleanup()
		mt.commandBufferPool = nil
	}
	
	// Release command queue
	if mt.commandQueue != nil {
		cgo_bridge.ReleaseCommandQueue(mt.commandQueue)
		mt.commandQueue = nil
	}
	
	// Release persistent tensors if allocated
	if mt.persistentInputTensor != nil {
		mt.persistentInputTensor.Release()
		mt.persistentInputTensor = nil
	}
	if mt.persistentLabelTensor != nil {
		mt.persistentLabelTensor.Release()
		mt.persistentLabelTensor = nil
	}
	// CRITICAL: Release pre-allocated gradient tensors
	for _, gradTensor := range mt.persistentGradientTensors {
		if gradTensor != nil {
			gradTensor.Release()
		}
	}
	mt.persistentGradientTensors = nil
	mt.persistentEnabled = false
	
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
	
	if config.OptimizerType == cgo_bridge.LBFGS {
		if config.LearningRate <= 0 {
			return fmt.Errorf("L-BFGS learning rate must be positive, got %f", config.LearningRate)
		}
		// L-BFGS is suitable for full-batch or large-batch training
		// For very small batches, recommend Adam or SGD instead
		if config.BatchSize < 32 {
			fmt.Printf("⚠️  Warning: L-BFGS with small batch size (%d) may be inefficient. Consider Adam or SGD for mini-batch training.\n", config.BatchSize)
		}
	}
	
	if config.WeightDecay < 0 {
		return fmt.Errorf("weight decay must be non-negative, got %f", config.WeightDecay)
	}
	
	return nil
}

// InferBatch performs inference on a batch of data
// Conforms to design requirements: single CGO call, GPU-resident, shared resources
func (mt *ModelTrainer) InferBatch(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error) {
	// Validate inputs
	if len(inputData) == 0 {
		return nil, fmt.Errorf("input data is empty")
	}
	
	if len(inputShape) < 2 {
		return nil, fmt.Errorf("input shape must have at least 2 dimensions, got %v", inputShape)
	}
	
	batchSize := inputShape[0]
	if batchSize <= 0 {
		return nil, fmt.Errorf("invalid batch size: %d", batchSize)
	}
	
	// Create input tensor and copy data to GPU (GPU-resident everything principle)
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Release()
	
	// Copy input data to GPU (minimal CPU-GPU transfers)
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(
		inputTensor.MetalBuffer(), 
		inputData,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	// Execute inference using single CGO call (design compliant)
	return mt.modelEngine.ExecuteInference(inputTensor, batchSize)
}

// CalculateAccuracyUnified calculates accuracy for both classification and regression
// For classification: returns percentage of correct predictions
// For regression: returns 1 - normalized mean absolute error
func (mt *ModelTrainer) CalculateAccuracyUnified(
	predictions []float32,
	trueLabels []float32,
	batchSize int,
	outputSize int,
	labelType LabelDataType,
) float64 {
	if mt.trainerConfig.ProblemType == Classification {
		// Convert float32 labels back to int32 for classification
		int32Labels := make([]int32, len(trueLabels))
		for i, v := range trueLabels {
			int32Labels[i] = int32(v)
		}
		return mt.CalculateAccuracy(predictions, int32Labels, batchSize, outputSize)
	} else {
		// For regression, calculate R² or 1-NMAE
		return mt.CalculateRegressionMetric(predictions, trueLabels, batchSize)
	}
}

// CalculateRegressionMetric calculates a metric for regression
// Returns 1 - normalized mean absolute error (closer to 1 is better)
func (mt *ModelTrainer) CalculateRegressionMetric(
	predictions []float32,
	trueLabels []float32,
	batchSize int,
) float64 {
	if len(predictions) < batchSize || len(trueLabels) < batchSize {
		return 0.0
	}
	
	var sumAbsError float64
	var sumTrue float64
	
	for i := 0; i < batchSize; i++ {
		pred := float64(predictions[i])
		true := float64(trueLabels[i])
		sumAbsError += math.Abs(pred - true)
		sumTrue += math.Abs(true)
	}
	
	if sumTrue == 0 {
		return 0.0
	}
	
	// Normalized mean absolute error
	nmae := sumAbsError / sumTrue
	
	// Return 1 - NMAE so higher is better (like accuracy)
	return math.Max(0, 1.0 - nmae)
}

// CalculateAccuracy computes accuracy from inference results and true labels
// Uses CPU-based argmax for final scalar metric (design compliant)
func (mt *ModelTrainer) CalculateAccuracy(
	predictions []float32,
	trueLabels []int32,
	batchSize int,
	numClasses int,
) float64 {
	if len(predictions) != batchSize*numClasses {
		return 0.0 // Invalid predictions array
	}
	
	if len(trueLabels) != batchSize {
		return 0.0 // Invalid labels array
	}
	
	correctPredictions := 0
	
	for i := 0; i < batchSize; i++ {
		// Find predicted class (argmax) - CPU computation for scalar result
		maxIdx := 0
		maxVal := predictions[i*numClasses]
		
		for j := 1; j < numClasses; j++ {
			if predictions[i*numClasses+j] > maxVal {
				maxVal = predictions[i*numClasses+j]
				maxIdx = j
			}
		}
		
		// Check if prediction matches true label
		if int32(maxIdx) == trueLabels[i] {
			correctPredictions++
		}
	}
	
	return float64(correctPredictions) / float64(batchSize)
}

// TrainingResultOptimized represents the result of an optimized training step
// Includes optional accuracy calculation to reduce CGO overhead
type TrainingResultOptimized struct {
	Loss        float32
	Accuracy    float64  // Only valid if HasAccuracy is true
	HasAccuracy bool     // Whether accuracy was calculated this step
	BatchSize   int
	StepTime    time.Duration
	Success     bool
	BatchRate   float64  // Batches per second
}

// shapesEqual compares two tensor shapes for equality
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

// SetLRScheduler sets a learning rate scheduler for the trainer
// This maintains GPU-resident principles by only updating LR between epochs
func (mt *ModelTrainer) SetLRScheduler(scheduler LRScheduler) {
	mt.lrScheduler = scheduler
}

// GetCurrentLearningRate returns the current learning rate based on scheduler
// This is a pure computation - no GPU operations
func (mt *ModelTrainer) GetCurrentLearningRate() float32 {
	if mt.lrScheduler == nil {
		return mt.config.LearningRate
	}
	
	// Get scheduled learning rate
	scheduledLR := mt.lrScheduler.GetLR(mt.currentEpoch, mt.currentStep, float64(mt.baseLearningRate))
	return float32(scheduledLR)
}

// SetEpoch updates the current epoch for learning rate scheduling
// Call this at the start of each epoch
func (mt *ModelTrainer) SetEpoch(epoch int) {
	mt.currentEpoch = epoch
	
	// Update the learning rate based on scheduler
	newLR := mt.GetCurrentLearningRate()
	mt.setLearningRateInternal(newLR, false) // Don't update base LR for scheduler changes
}

// StepSchedulerWithMetric updates schedulers that depend on validation metrics
// For ReduceLROnPlateauScheduler - call this after validation
func (mt *ModelTrainer) StepSchedulerWithMetric(metric float64) {
	if scheduler, ok := mt.lrScheduler.(*ReduceLROnPlateauScheduler); ok {
		// Update the scheduler's internal state
		newLR := scheduler.Step(metric, float64(mt.config.LearningRate))
		mt.setLearningRateInternal(float32(newLR), false) // Don't update base LR for scheduler changes
	}
}

// updateSchedulerStep increments the training step and updates learning rate if needed
// This method ensures step-based schedulers update the optimizer properly
func (mt *ModelTrainer) updateSchedulerStep() {
	mt.currentStep++ // Direct increment to avoid recursion
	
	// For step-based schedulers, check if LR should be updated
	if mt.lrScheduler != nil {
		// Check if this is a step-based scheduler (not plateau-based)
		if _, isPlateauScheduler := mt.lrScheduler.(*ReduceLROnPlateauScheduler); !isPlateauScheduler {
			// Update learning rate based on current epoch and step
			newLR := mt.GetCurrentLearningRate()
			if newLR != mt.config.LearningRate {
				mt.setLearningRateInternal(newLR, false) // Don't update base LR for scheduler changes
			}
		}
	}
}

// GetSchedulerInfo returns current scheduler information for logging
func (mt *ModelTrainer) GetSchedulerInfo() string {
	if mt.lrScheduler == nil {
		return "No scheduler (constant LR)"
	}
	
	currentLR := mt.GetCurrentLearningRate()
	return fmt.Sprintf("%s scheduler: LR=%.6f (epoch=%d, step=%d)", 
		mt.lrScheduler.GetName(), 
		currentLR, 
		mt.currentEpoch,
		mt.currentStep,
	)
}

// GetParameterTensors returns the parameter tensors for weight extraction
func (mt *ModelTrainer) GetParameterTensors() []*memory.Tensor {
	if mt.modelEngine == nil {
		return nil
	}
	return mt.modelEngine.GetParameterTensors()
}

// GetLRScheduler returns the learning rate scheduler if available
func (mt *ModelTrainer) GetLRScheduler() interface{} {
	return mt.lrScheduler
}

// SetLearningRate sets the learning rate manually (user-initiated)
func (mt *ModelTrainer) SetLearningRate(lr float32) {
	mt.setLearningRateInternal(lr, true)
}

// setLearningRateInternal sets the learning rate with optional base LR update
func (mt *ModelTrainer) setLearningRateInternal(lr float32, updateBaseLR bool) {
	mt.config.LearningRate = lr
	
	// Update the actual optimizer learning rate (GPU-resident)
	if mt.modelEngine != nil {
		err := mt.modelEngine.UpdateLearningRate(lr)
		if err != nil {
			// Log error but continue - this maintains backward compatibility
			fmt.Printf("Warning: Failed to update optimizer learning rate: %v\n", err)
		}
	}
	
	// Update base learning rate only for manual changes, not scheduler-driven changes
	if updateBaseLR {
		mt.baseLearningRate = lr
	}
}

// GetOptimizerState returns the optimizer state for checkpoint saving
func (mt *ModelTrainer) GetOptimizerState() *OptimizerStateData {
	// Get optimizer state from the engine
	state, err := mt.modelEngine.GetOptimizerState()
	if err != nil {
		// Log error but return nil to maintain backward compatibility
		fmt.Printf("Warning: Failed to get optimizer state: %v\n", err)
		return nil
	}
	
	// Convert from optimizer.OptimizerState to OptimizerStateData
	if state == nil {
		return nil
	}
	
	return &OptimizerStateData{
		Type:       state.Type,
		Parameters: state.Parameters,
		StateData:  state.StateData,
	}
}

// SetOptimizerState restores optimizer state from checkpoint
func (mt *ModelTrainer) SetOptimizerState(state interface{}) error {
	if state == nil {
		return fmt.Errorf("optimizer state is nil")
	}
	
	// Convert from interface{} to OptimizerStateData
	stateData, ok := state.(*OptimizerStateData)
	if !ok {
		// Try to convert from checkpoint format
		checkpointState, ok := state.(*checkpoints.OptimizerState)
		if !ok {
			return fmt.Errorf("invalid optimizer state type: expected *OptimizerStateData or *checkpoints.OptimizerState")
		}
		// Convert checkpoint format to internal format
		stateData = &OptimizerStateData{
			Type:       checkpointState.Type,
			Parameters: checkpointState.Parameters,
			StateData:  checkpointState.StateData,
		}
	}
	
	// Convert to optimizer.OptimizerState
	optimizerState := &optimizer.OptimizerState{
		Type:       stateData.Type,
		Parameters: stateData.Parameters,
		StateData:  stateData.StateData,
	}
	
	// Pass to engine for restoration
	return mt.modelEngine.SetOptimizerState(optimizerState)
}

// Predict provides a lightweight inference method for backward compatibility
// For optimal inference performance, use ModelInferencer instead
func (mt *ModelTrainer) Predict(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error) {
	// Use the existing InferBatch method for compatibility
	// This maintains the proven single-CGO-call architecture
	return mt.InferBatch(inputData, inputShape)
}

// ================================================================
// EVALUATION METRICS SYSTEM - GPU-RESIDENT ARCHITECTURE COMPLIANT
// ================================================================

// EnableEvaluationMetrics enables comprehensive evaluation metrics collection
// Metrics are calculated from GPU-resident tensors with CPU-only scalar results
func (mt *ModelTrainer) EnableEvaluationMetrics() {
	mt.metricsEnabled = true
	mt.confusionMatrix.Reset()
	mt.metricHistory = make(map[MetricType][]float64)
}

// DisableEvaluationMetrics disables evaluation metrics for performance
func (mt *ModelTrainer) DisableEvaluationMetrics() {
	mt.metricsEnabled = false
}

// IsEvaluationMetricsEnabled returns whether comprehensive metrics are enabled
func (mt *ModelTrainer) IsEvaluationMetricsEnabled() bool {
	return mt.metricsEnabled
}

// UpdateMetricsFromInference updates evaluation metrics from inference results
// GPU-resident architecture: operates on GPU tensor data, stores CPU scalars only
func (mt *ModelTrainer) UpdateMetricsFromInference(
	predictions []float32,
	trueLabels interface{}, // []int32 for classification, []float32 for regression
	batchSize int,
) error {
	if !mt.metricsEnabled {
		return nil // Metrics disabled, skip computation
	}
	
	outputSize := mt.getOutputSize()
	
	if mt.trainerConfig.ProblemType == Classification {
		// Handle classification metrics
		var int32Labels []int32
		switch labels := trueLabels.(type) {
		case []int32:
			int32Labels = labels
		case []float32:
			// Convert float32 labels to int32
			int32Labels = make([]int32, len(labels))
			for i, v := range labels {
				int32Labels[i] = int32(v)
			}
		default:
			return fmt.Errorf("invalid label type for classification: %T", trueLabels)
		}
		
		// Update confusion matrix (use number of classes, not output size)
		numClasses := mt.confusionMatrix.NumClasses
		err := mt.confusionMatrix.UpdateFromPredictions(predictions, int32Labels, batchSize, numClasses)
		if err != nil {
			return fmt.Errorf("failed to update confusion matrix: %v", err)
		}
		
		// Collect probabilities for PR/ROC curves if visualization is enabled
		mt.collectValidationProbabilities(predictions, int32Labels, batchSize, numClasses)
		
		// For binary classification, also calculate AUC-ROC if we have raw scores
		if outputSize == 2 || outputSize == 1 {
			var scores []float32
			if outputSize == 1 {
				// Single output (BCEWithLogits or sigmoid output)
				scores = predictions[:batchSize]
			} else {
				// Two outputs, use positive class probability
				scores = make([]float32, batchSize)
				for i := 0; i < batchSize; i++ {
					scores[i] = predictions[i*2+1] // Positive class score
				}
			}
			
			auc := CalculateAUCROC(scores, int32Labels, batchSize)
			mt.addToHistory(AUCROC, auc)
		}
		
	} else {
		// Handle regression metrics
		var float32Labels []float32
		switch labels := trueLabels.(type) {
		case []float32:
			float32Labels = labels
		case []int32:
			// Convert int32 to float32 (unusual but handle it)
			float32Labels = make([]float32, len(labels))
			for i, v := range labels {
				float32Labels[i] = float32(v)
			}
		default:
			return fmt.Errorf("invalid label type for regression: %T", trueLabels)
		}
		
		// Calculate comprehensive regression metrics
		mt.lastRegressionMetrics = CalculateRegressionMetrics(predictions, float32Labels, batchSize)
		
		// Add to history for plotting
		mt.addToHistory(MAE, mt.lastRegressionMetrics.MAE)
		mt.addToHistory(MSE, mt.lastRegressionMetrics.MSE)
		mt.addToHistory(RMSE, mt.lastRegressionMetrics.RMSE)
		mt.addToHistory(R2, mt.lastRegressionMetrics.R2)
		mt.addToHistory(NMAE, mt.lastRegressionMetrics.NMAE)
	}
	
	return nil
}

// GetMetric returns the current value of a specific metric
// CPU-only scalar result (GPU-resident architecture compliant)
func (mt *ModelTrainer) GetMetric(metric MetricType) float64 {
	if !mt.metricsEnabled {
		return 0.0
	}
	
	if mt.trainerConfig.ProblemType == Classification {
		return mt.confusionMatrix.GetMetric(metric)
	} else {
		// Regression metrics
		switch metric {
		case MAE:
			return mt.lastRegressionMetrics.MAE
		case MSE:
			return mt.lastRegressionMetrics.MSE
		case RMSE:
			return mt.lastRegressionMetrics.RMSE
		case R2:
			return mt.lastRegressionMetrics.R2
		case NMAE:
			return mt.lastRegressionMetrics.NMAE
		default:
			return 0.0
		}
	}
}

// GetClassificationMetrics returns all classification metrics for the current confusion matrix
func (mt *ModelTrainer) GetClassificationMetrics() map[string]float64 {
	if !mt.metricsEnabled || mt.trainerConfig.ProblemType != Classification {
		return make(map[string]float64)
	}
	
	metrics := make(map[string]float64)
	metrics["accuracy"] = mt.confusionMatrix.GetAccuracy()
	
	if mt.confusionMatrix.NumClasses == 2 {
		// Binary classification metrics
		metrics["precision"] = mt.confusionMatrix.GetMetric(Precision)
		metrics["recall"] = mt.confusionMatrix.GetMetric(Recall)
		metrics["f1_score"] = mt.confusionMatrix.GetMetric(F1Score)
		metrics["specificity"] = mt.confusionMatrix.GetMetric(Specificity)
		metrics["npv"] = mt.confusionMatrix.GetMetric(NPV)
		
		// Add AUC if available
		if history, exists := mt.metricHistory[AUCROC]; exists && len(history) > 0 {
			metrics["auc_roc"] = history[len(history)-1]
		}
	} else {
		// Multi-class metrics
		metrics["macro_precision"] = mt.confusionMatrix.GetMetric(MacroPrecision)
		metrics["macro_recall"] = mt.confusionMatrix.GetMetric(MacroRecall)
		metrics["macro_f1"] = mt.confusionMatrix.GetMetric(MacroF1)
		metrics["micro_precision"] = mt.confusionMatrix.GetMetric(MicroPrecision)
		metrics["micro_recall"] = mt.confusionMatrix.GetMetric(MicroRecall)
		metrics["micro_f1"] = mt.confusionMatrix.GetMetric(MicroF1)
	}
	
	return metrics
}

// GetRegressionMetrics returns all regression metrics
func (mt *ModelTrainer) GetRegressionMetrics() map[string]float64 {
	if !mt.metricsEnabled || mt.trainerConfig.ProblemType != Regression {
		return make(map[string]float64)
	}
	
	metrics := make(map[string]float64)
	metrics["mae"] = mt.lastRegressionMetrics.MAE
	metrics["mse"] = mt.lastRegressionMetrics.MSE
	metrics["rmse"] = mt.lastRegressionMetrics.RMSE
	metrics["r2"] = mt.lastRegressionMetrics.R2
	metrics["nmae"] = mt.lastRegressionMetrics.NMAE
	
	return metrics
}

// GetConfusionMatrix returns a copy of the current confusion matrix
func (mt *ModelTrainer) GetConfusionMatrix() [][]int {
	if !mt.metricsEnabled || mt.trainerConfig.ProblemType != Classification {
		return nil
	}
	
	// Return a copy to prevent external modification
	matrix := make([][]int, mt.confusionMatrix.NumClasses)
	for i := range matrix {
		matrix[i] = make([]int, mt.confusionMatrix.NumClasses)
		copy(matrix[i], mt.confusionMatrix.Matrix[i])
	}
	
	return matrix
}

// GetMetricHistory returns the history of a specific metric for plotting
func (mt *ModelTrainer) GetMetricHistory(metric MetricType) []float64 {
	if !mt.metricsEnabled {
		return nil
	}
	
	if history, exists := mt.metricHistory[metric]; exists {
		// Return a copy to prevent external modification
		result := make([]float64, len(history))
		copy(result, history)
		return result
	}
	
	return nil
}

// ResetMetrics clears all accumulated metrics and history
func (mt *ModelTrainer) ResetMetrics() {
	if mt.confusionMatrix != nil {
		mt.confusionMatrix.Reset()
	}
	mt.lastRegressionMetrics = &RegressionMetrics{}
	mt.metricHistory = make(map[MetricType][]float64)
}

// addToHistory adds a metric value to the history for plotting
func (mt *ModelTrainer) addToHistory(metric MetricType, value float64) {
	if mt.metricHistory == nil {
		mt.metricHistory = make(map[MetricType][]float64)
	}
	
	mt.metricHistory[metric] = append(mt.metricHistory[metric], value)
}

// VISUALIZATION METHODS - GPU-resident architecture compliance
// All visualization methods follow the four core requirements:
// 1. Only CPU access for final scalar metrics collection
// 2. Minimal CGO overhead by batching data collection
// 3. GPU-resident tensors throughout training
// 4. Proper memory management with buffer reuse

// EnableVisualization enables visualization data collection
func (mt *ModelTrainer) EnableVisualization() {
	mt.visualizationEnabled = true
	mt.visualizationCollector.Enable()
	// Reset probability collection when enabling visualization
	mt.resetProbabilityCollection()
}

// DisableVisualization disables visualization data collection
func (mt *ModelTrainer) DisableVisualization() {
	mt.visualizationEnabled = false
	mt.visualizationCollector.Disable()
	// Clear probability buffers to free memory
	mt.validationProbabilities = nil
	mt.validationLabels = nil
	mt.probabilityBatchCount = 0
}

// IsVisualizationEnabled returns whether visualization is enabled
func (mt *ModelTrainer) IsVisualizationEnabled() bool {
	return mt.visualizationEnabled
}

// EnablePlottingService enables the plotting service for sidecar communication
func (mt *ModelTrainer) EnablePlottingService() {
	mt.plottingService.Enable()
}

// DisablePlottingService disables the plotting service
func (mt *ModelTrainer) DisablePlottingService() {
	mt.plottingService.Disable()
}

// ConfigurePlottingService configures the plotting service with custom settings
func (mt *ModelTrainer) ConfigurePlottingService(config PlottingServiceConfig) {
	mt.plottingService = NewPlottingService(config)
}

// CheckPlottingServiceHealth checks if the plotting service is available
func (mt *ModelTrainer) CheckPlottingServiceHealth() error {
	return mt.plottingService.CheckHealth()
}

// RecordTrainingStep records training metrics for visualization
// This method is called internally during training and follows GPU-resident principles
func (mt *ModelTrainer) recordTrainingStep(step int, loss, accuracy float64) {
	if !mt.visualizationEnabled {
		return
	}
	
	// Get current learning rate for plotting
	currentLR := float64(mt.getCurrentLearningRate())
	
	// Record step data (CPU-only scalar values)
	mt.visualizationCollector.RecordTrainingStep(step, loss, accuracy, currentLR)
}

// RecordValidationStep records validation metrics for visualization
// This method is called internally during validation and follows GPU-resident principles
func (mt *ModelTrainer) recordValidationStep(step int, loss, accuracy float64) {
	if !mt.visualizationEnabled {
		return
	}
	
	// Record validation data (CPU-only scalar values)
	mt.visualizationCollector.RecordValidationStep(step, loss, accuracy)
}

// RecordEpochMetrics records epoch-level metrics for visualization
func (mt *ModelTrainer) RecordEpochMetrics(epoch int, trainLoss, trainAcc, valLoss, valAcc float64) {
	if !mt.visualizationEnabled {
		return
	}
	
	mt.visualizationCollector.RecordEpoch(epoch, trainLoss, trainAcc, valLoss, valAcc)
}

// StartValidationPhase prepares for validation by resetting probability collection
func (mt *ModelTrainer) StartValidationPhase() {
	if mt.visualizationEnabled {
		mt.resetProbabilityCollection()
	}
}

// RecordMetricsForVisualization records comprehensive metrics for visualization
// This method integrates with the evaluation metrics system
func (mt *ModelTrainer) RecordMetricsForVisualization() {
	if !mt.visualizationEnabled || !mt.metricsEnabled {
		return
	}
	
	// Record classification metrics
	if mt.trainerConfig.ProblemType == Classification {
		// Record confusion matrix
		confMatrix := mt.GetConfusionMatrix()
		if confMatrix != nil {
			classNames := mt.getClassNames()
			mt.visualizationCollector.RecordConfusionMatrix(confMatrix, classNames)
		}
		
		// Record ROC data if available
		if mt.confusionMatrix.NumClasses == 2 {
			rocPoints := mt.generateROCPoints()
			if len(rocPoints) > 0 {
				mt.visualizationCollector.RecordROCData(rocPoints)
			}
			
			// Record Precision-Recall data
			prPoints := mt.generatePRPoints()
			if len(prPoints) > 0 {
				mt.visualizationCollector.RecordPRData(prPoints)
			}
		}
	}
	
	// Record regression metrics
	if mt.trainerConfig.ProblemType == Regression {
		// Note: Regression predictions would need to be stored during training
		// This is a placeholder for when regression visualization is needed
		// mt.visualizationCollector.RecordRegressionData(predictions, trueValues)
	}
}

// GenerateTrainingCurvesPlot generates and returns training curves plot data
func (mt *ModelTrainer) GenerateTrainingCurvesPlot() PlotData {
	return mt.visualizationCollector.GenerateTrainingCurvesPlot()
}

// GenerateLearningRateSchedulePlot generates and returns learning rate schedule plot data
func (mt *ModelTrainer) GenerateLearningRateSchedulePlot() PlotData {
	return mt.visualizationCollector.GenerateLearningRateSchedulePlot()
}

// GenerateROCCurvePlot generates and returns ROC curve plot data
func (mt *ModelTrainer) GenerateROCCurvePlot() PlotData {
	return mt.visualizationCollector.GenerateROCCurvePlot()
}

// GeneratePrecisionRecallPlot generates and returns Precision-Recall curve plot data
func (mt *ModelTrainer) GeneratePrecisionRecallPlot() PlotData {
	return mt.visualizationCollector.GeneratePrecisionRecallPlot()
}

// GenerateConfusionMatrixPlot generates and returns confusion matrix plot data
func (mt *ModelTrainer) GenerateConfusionMatrixPlot() PlotData {
	return mt.visualizationCollector.GenerateConfusionMatrixPlot()
}

// GenerateAllPlots generates all available plots and returns them
func (mt *ModelTrainer) GenerateAllPlots() map[PlotType]PlotData {
	plots := make(map[PlotType]PlotData)
	
	if !mt.visualizationEnabled {
		return plots
	}
	
	// Generate all available plots
	plots[TrainingCurves] = mt.visualizationCollector.GenerateTrainingCurvesPlot()
	plots[LearningRateSchedule] = mt.visualizationCollector.GenerateLearningRateSchedulePlot()
	
	if mt.trainerConfig.ProblemType == Classification {
		plots[ROCCurve] = mt.visualizationCollector.GenerateROCCurvePlot()
		plots[PrecisionRecall] = mt.visualizationCollector.GeneratePrecisionRecallPlot()
		plots[ConfusionMatrixPlot] = mt.visualizationCollector.GenerateConfusionMatrixPlot()
	}
	
	if mt.trainerConfig.ProblemType == Regression {
		plots[RegressionScatter] = mt.visualizationCollector.GenerateRegressionScatterPlot()
		plots[ResidualPlot] = mt.visualizationCollector.GenerateResidualPlot()
	}
	
	return plots
}

// SendPlotToSidecar sends a specific plot to the sidecar plotting service
func (mt *ModelTrainer) SendPlotToSidecar(plotType PlotType) (*PlottingResponse, error) {
	if !mt.visualizationEnabled {
		return &PlottingResponse{
			Success: false,
			Message: "Visualization is disabled",
		}, nil
	}
	
	return mt.plottingService.GenerateAndSendPlot(mt.visualizationCollector, plotType)
}

// SendAllPlotsToSidecar sends all available plots to the sidecar plotting service
func (mt *ModelTrainer) SendAllPlotsToSidecar() map[PlotType]*PlottingResponse {
	if !mt.visualizationEnabled {
		return make(map[PlotType]*PlottingResponse)
	}
	
	return mt.plottingService.GenerateAndSendAllPlots(mt.visualizationCollector)
}

// ClearVisualizationData clears all collected visualization data
func (mt *ModelTrainer) ClearVisualizationData() {
	if mt.visualizationCollector != nil {
		mt.visualizationCollector.Clear()
	}
}

// GetVisualizationCollector returns the visualization collector for advanced usage
func (mt *ModelTrainer) GetVisualizationCollector() *VisualizationCollector {
	return mt.visualizationCollector
}

// resetProbabilityCollection resets the probability collection buffers
func (mt *ModelTrainer) resetProbabilityCollection() {
	mt.validationProbabilities = nil
	mt.validationLabels = nil
	mt.probabilityBatchCount = 0
}

// collectValidationProbabilities collects probabilities and labels for PR/ROC curves
// This is GPU-resident compliant: only copies final results from GPU once
func (mt *ModelTrainer) collectValidationProbabilities(predictions []float32, labels []int32, batchSize int, numClasses int) {
	// Only collect if visualization is enabled and we haven't exceeded max batches
	if !mt.visualizationEnabled || mt.probabilityBatchCount >= mt.maxProbabilityBatches {
		return
	}
	
	// For binary classification, we need probabilities for the positive class
	// For multi-class, we store all probabilities
	if numClasses == 2 {
		// Extract positive class probabilities (class 1)
		positiveProbabilities := make([]float32, batchSize)
		for i := 0; i < batchSize; i++ {
			positiveProbabilities[i] = predictions[i*numClasses + 1]
		}
		mt.validationProbabilities = append(mt.validationProbabilities, positiveProbabilities...)
	} else {
		// Store all probabilities for multi-class
		mt.validationProbabilities = append(mt.validationProbabilities, predictions[:batchSize*numClasses]...)
	}
	
	// Store corresponding labels
	mt.validationLabels = append(mt.validationLabels, labels[:batchSize]...)
	mt.probabilityBatchCount++
}

// Helper methods for visualization integration

// getClassNames returns class names for visualization
func (mt *ModelTrainer) getClassNames() []string {
	// For binary classification, use standard names
	if mt.confusionMatrix.NumClasses == 2 {
		return []string{"Class 0", "Class 1"}
	}
	
	// For multi-class, generate class names
	classNames := make([]string, mt.confusionMatrix.NumClasses)
	for i := range classNames {
		classNames[i] = fmt.Sprintf("Class %d", i)
	}
	
	return classNames
}

// generateROCPoints generates ROC curve points from collected probabilities
func (mt *ModelTrainer) generateROCPoints() []ROCPointViz {
	if mt.confusionMatrix.NumClasses != 2 || len(mt.validationProbabilities) == 0 {
		return nil
	}
	
	// Create a slice of probability-label pairs for sorting
	type probLabel struct {
		prob  float32
		label int32
	}
	
	pairs := make([]probLabel, len(mt.validationLabels))
	for i := range pairs {
		pairs[i] = probLabel{
			prob:  mt.validationProbabilities[i],
			label: mt.validationLabels[i],
		}
	}
	
	// Sort by probability in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})
	
	// Count total positives and negatives
	totalPositives := 0
	totalNegatives := 0
	for _, pair := range pairs {
		if pair.label == 1 {
			totalPositives++
		} else {
			totalNegatives++
		}
	}
	
	if totalPositives == 0 || totalNegatives == 0 {
		// All samples are of one class
		return nil
	}
	
	// Generate ROC curve points
	var rocPoints []ROCPointViz
	
	// Add the starting point (threshold = 1.0+epsilon)
	rocPoints = append(rocPoints, ROCPointViz{
		FPR:       0.0,
		TPR:       0.0,
		Threshold: 1.0,
	})
	
	truePositives := 0
	falsePositives := 0
	
	// Calculate points at different thresholds
	for i, pair := range pairs {
		if pair.label == 1 {
			truePositives++
		} else {
			falsePositives++
		}
		
		// Calculate TPR and FPR at this threshold
		tpr := float64(truePositives) / float64(totalPositives)
		fpr := float64(falsePositives) / float64(totalNegatives)
		
		// Add point at regular intervals or when there's a significant change
		if i%10 == 0 || i == len(pairs)-1 || 
		   (i > 0 && math.Abs(float64(pair.prob - pairs[i-1].prob)) > 0.05) ||
		   (len(rocPoints) > 0 && (math.Abs(tpr - rocPoints[len(rocPoints)-1].TPR) > 0.01 || 
		                          math.Abs(fpr - rocPoints[len(rocPoints)-1].FPR) > 0.01)) {
			rocPoints = append(rocPoints, ROCPointViz{
				FPR:       fpr,
				TPR:       tpr,
				Threshold: float64(pair.prob),
			})
		}
	}
	
	// Ensure we have the end point (threshold = 0.0)
	if rocPoints[len(rocPoints)-1].FPR < 1.0 || rocPoints[len(rocPoints)-1].TPR < 1.0 {
		rocPoints = append(rocPoints, ROCPointViz{
			FPR:       1.0,
			TPR:       1.0,
			Threshold: 0.0,
		})
	}
	
	return rocPoints
}

// generatePRPoints generates Precision-Recall curve points from collected probabilities
func (mt *ModelTrainer) generatePRPoints() []PRPoint {
	if mt.confusionMatrix.NumClasses != 2 || len(mt.validationProbabilities) == 0 {
		return nil
	}
	
	// Create a slice of probability-label pairs for sorting
	type probLabel struct {
		prob  float32
		label int32
	}
	
	pairs := make([]probLabel, len(mt.validationLabels))
	for i := range pairs {
		pairs[i] = probLabel{
			prob:  mt.validationProbabilities[i],
			label: mt.validationLabels[i],
		}
	}
	
	// Sort by probability in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})
	
	// Generate PR curve points at different thresholds
	var prPoints []PRPoint
	totalPositives := 0
	for _, pair := range pairs {
		if pair.label == 1 {
			totalPositives++
		}
	}
	
	if totalPositives == 0 || totalPositives == len(pairs) {
		// All samples are of one class
		return nil
	}
	
	// Add the starting point (threshold = 1.0)
	prPoints = append(prPoints, PRPoint{
		Precision: 1.0,
		Recall:    0.0,
		Threshold: 1.0,
	})
	
	// Generate points at regular intervals and significant changes
	truePositives := 0
	falsePositives := 0
	
	// Calculate points at different thresholds
	for i, pair := range pairs {
		if pair.label == 1 {
			truePositives++
		} else {
			falsePositives++
		}
		
		// Calculate precision and recall at this threshold
		predictedPositives := i + 1
		precision := float64(truePositives) / float64(predictedPositives)
		recall := float64(truePositives) / float64(totalPositives)
		
		// Add point at regular intervals or when there's a significant change
		if i%10 == 0 || i == len(pairs)-1 || 
		   (i > 0 && math.Abs(float64(pair.prob - pairs[i-1].prob)) > 0.05) {
			prPoints = append(prPoints, PRPoint{
				Precision: precision,
				Recall:    recall,
				Threshold: float64(pair.prob),
			})
		}
	}
	
	// Ensure we have the end point
	if prPoints[len(prPoints)-1].Recall < 1.0 {
		prPoints = append(prPoints, PRPoint{
			Precision: float64(totalPositives) / float64(len(pairs)),
			Recall:    1.0,
			Threshold: 0.0,
		})
	}
	
	return prPoints
}

// getCurrentLearningRate returns the current learning rate considering scheduling
func (mt *ModelTrainer) getCurrentLearningRate() float32 {
	if mt.lrScheduler != nil {
		return float32(mt.lrScheduler.GetLR(mt.currentEpoch, mt.currentStep, float64(mt.baseLearningRate)))
	}
	return mt.baseLearningRate
}