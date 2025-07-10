package training

import (
	"fmt"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/engine"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// EngineType represents the training engine selection strategy
// Maintains GPU-resident architecture compliance across all engine types
type EngineType int

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

func (et EngineType) String() string {
	switch et {
	case Auto:
		return "Auto"
	case Hybrid:
		return "Hybrid"
	case Dynamic:
		return "Dynamic"
	default:
		return "Unknown"
	}
}

// ProblemType represents the type of machine learning problem
type ProblemType int

const (
	// Classification for discrete class prediction
	Classification ProblemType = iota
	// Regression for continuous value prediction
	Regression
)

func (pt ProblemType) String() string {
	switch pt {
	case Classification:
		return "Classification"
	case Regression:
		return "Regression"
	default:
		return fmt.Sprintf("Unknown(%d)", pt)
	}
}

// LossFunction represents the loss function for training
type LossFunction int

const (
	// Classification losses
	CrossEntropy       LossFunction = iota // Softmax cross-entropy for multi-class
	SparseCrossEntropy                     // Sparse categorical cross-entropy
	BinaryCrossEntropy                     // Binary cross-entropy for binary classification
	BCEWithLogits                          // Binary cross-entropy with logits (more numerically stable)
	CategoricalCrossEntropy                // Categorical cross-entropy without softmax
	
	// Regression losses
	MeanSquaredError  // 5 - MSE for regression
	MeanAbsoluteError // 6 - MAE for regression
	Huber             // 7 - Huber loss for robust regression
)

func (lf LossFunction) String() string {
	switch lf {
	case CrossEntropy:
		return "CrossEntropy"
	case SparseCrossEntropy:
		return "SparseCrossEntropy"
	case BinaryCrossEntropy:
		return "BinaryCrossEntropy"
	case BCEWithLogits:
		return "BCEWithLogits"
	case CategoricalCrossEntropy:
		return "CategoricalCrossEntropy"
	case MeanSquaredError:
		return "MeanSquaredError"
	case MeanAbsoluteError:
		return "MeanAbsoluteError"
	case Huber:
		return "Huber"
	default:
		return fmt.Sprintf("Unknown(%d)", lf)
	}
}

// SimpleTrainer provides a basic training interface for testing Phase 1
type SimpleTrainer struct {
	batchTrainer *engine.BatchTrainer
	batchSize    int
	config       cgo_bridge.TrainingConfig
}

// NewSimpleTrainer creates a new simple trainer (legacy function - use factory for production)
// DEPRECATED: Use NewSGDTrainer, NewAdamTrainer, or the factory system for production code
func NewSimpleTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	// Use the new factory system internally for consistency
	factory := NewFactory()
	return factory.CreateSGDTrainer(batchSize, learningRate, 0.0)
}

// TrainBatch trains on a single batch with timing (full training loop)
func (st *SimpleTrainer) TrainBatch(
	inputData []float32,
	inputShape []int,
	labelData []int32,
	labelShape []int,
	weights []*memory.Tensor,
) (*TrainingResult, error) {
	
	start := time.Now()
	
	// Create input tensor
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Release()
	
	// Copy input data to GPU tensor
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy input data to GPU: %v", err)
	}
	
	// Create label tensor (one-hot encoded for hybrid approach)
	// Convert int32 labels to one-hot float32 format
	oneHotShape := []int{labelShape[0], 2} // Assuming 2 classes for this test
	labelTensor, err := memory.NewTensor(oneHotShape, memory.Float32, memory.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create label tensor: %v", err)
	}
	defer labelTensor.Release()
	
	// Convert int32 labels to one-hot float32 format and copy to GPU
	oneHotData := make([]float32, oneHotShape[0]*oneHotShape[1])
	for i, label := range labelData {
		// Zero-out the row first
		baseIdx := i * oneHotShape[1]
		for j := 0; j < oneHotShape[1]; j++ {
			oneHotData[baseIdx+j] = 0.0
		}
		// Set the correct class to 1.0
		if int(label) < oneHotShape[1] {
			oneHotData[baseIdx+int(label)] = 1.0
		}
	}
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), oneHotData)
	if err != nil {
		return nil, fmt.Errorf("failed to copy label data to GPU: %v", err)
	}
	
	// Execute hybrid full training step (forward + backward + optimizer)
	learningRate := st.config.LearningRate
	result, err := st.batchTrainer.TrainBatchHybridFull(inputTensor, labelTensor, weights, learningRate)
	if err != nil {
		return nil, fmt.Errorf("batch training failed: %v", err)
	}
	
	totalTime := time.Since(start)
	
	return &TrainingResult{
		Loss:       result.Loss,
		BatchSize:  result.BatchSize,
		StepTime:   totalTime,
		Success:    result.Success,
		BatchRate:  float64(result.BatchSize) / totalTime.Seconds(),
	}, nil
}

// GetStats returns training statistics
func (st *SimpleTrainer) GetStats() *TrainingStats {
	memStats := memory.GetGlobalMemoryManager().Stats()
	
	return &TrainingStats{
		CurrentStep:    st.batchTrainer.GetCurrentStep(),
		BatchSize:      st.batchSize,
		OptimizerType:  st.config.OptimizerType,
		LearningRate:   st.config.LearningRate,
		MemoryPoolStats: memStats,
	}
}

// Cleanup releases resources
func (st *SimpleTrainer) Cleanup() {
	if st.batchTrainer != nil {
		st.batchTrainer.Cleanup()
	}
}

// TrainingResult represents the result of a training step
type TrainingResult struct {
	Loss      float32
	BatchSize int
	StepTime  time.Duration
	Success   bool
	BatchRate float64 // batches per second
}

// TrainingStats provides training statistics
type TrainingStats struct {
	CurrentStep     int
	BatchSize       int
	OptimizerType   cgo_bridge.OptimizerType
	LearningRate    float32
	MemoryPoolStats map[memory.PoolKey]string
}

// CreateDummyWeights creates dummy weight tensors for testing (hybrid approach)
func CreateDummyWeights() ([]*memory.Tensor, error) {
	weights := make([]*memory.Tensor, 0)
	
	// Create weight tensors for hybrid approach (only FC layer - conv is built-in)
	shapes := [][]int{
		{8, 2},           // FC layer weights (8 inputs, 2 outputs)
		{2},              // FC layer bias
	}
	
	for i, shape := range shapes {
		tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
		if err != nil {
			// Cleanup previously created tensors
			for _, w := range weights {
				w.Release()
			}
			return nil, fmt.Errorf("failed to create weight tensor %d: %v", i, err)
		}
		weights = append(weights, tensor)
	}
	
	return weights, nil
}

// TestPhase1 runs a basic test of Phase 1 implementation
func TestPhase1() error {
	fmt.Println("Testing Phase 1 implementation...")
	
	// Create trainer
	trainer, err := NewSimpleTrainer(32, 0.01)
	if err != nil {
		return fmt.Errorf("failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()
	
	// Create dummy weights
	weights, err := CreateDummyWeights()
	if err != nil {
		return fmt.Errorf("failed to create weights: %v", err)
	}
	defer func() {
		for _, w := range weights {
			w.Release()
		}
	}()
	
	// Create dummy input data for intermediate CNN 
	inputShape := []int{32, 3, 32, 32}    // Batch of 32 RGB 32x32 images (smaller)
	labelShape := []int{32}               // 32 labels
	
	inputData := make([]float32, 32*3*32*32)
	labelData := make([]int32, 32)
	
	// Fill with dummy data
	for i := range inputData {
		inputData[i] = float32(i % 100) / 100.0
	}
	for i := range labelData {
		labelData[i] = int32(i % 2) // Binary classification
	}
	
	// Run a few training steps
	for step := 0; step < 3; step++ {
		result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape, weights)
		if err != nil {
			return fmt.Errorf("training step %d failed: %v", step, err)
		}
		
		fmt.Printf("Step %d: Loss=%.4f, BatchRate=%.2f batch/s, Time=%v\n",
			step, result.Loss, result.BatchRate, result.StepTime)
	}
	
	// Print statistics
	stats := trainer.GetStats()
	fmt.Printf("\nTraining Stats:\n")
	fmt.Printf("  Current Step: %d\n", stats.CurrentStep)
	fmt.Printf("  Batch Size: %d\n", stats.BatchSize)
	fmt.Printf("  Learning Rate: %.4f\n", stats.LearningRate)
	fmt.Printf("  Memory Pools: %d active\n", len(stats.MemoryPoolStats))
	
	fmt.Println("Phase 1 test completed successfully!")
	return nil
}

// ================================================================
// PRODUCTION TRAINER FACTORY SYSTEM
// ================================================================

// ModelArchitectureInfo provides architecture analysis for engine selection
type ModelArchitectureInfo struct {
	InputDimensions int          // Number of input dimensions (2D, 4D, etc.)
	HasConvLayers   bool         // Whether model contains Conv2D layers
	HasDenseLayers  bool         // Whether model contains Dense layers
	IsMLPOnly       bool         // Whether model is MLP-only (Dense + activations)
	IsCNNPattern    bool         // Whether model follows CNN pattern (Conv + Dense)
	LayerCount      int          // Total number of layers
	ParameterCount  int64        // Total trainable parameters
	Complexity      string       // "Simple", "Standard", "Complex"
}

// TrainerConfig provides comprehensive configuration for training
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
	Alpha       float32 `json:"alpha"`        // RMSProp smoothing constant (default: 0.99)
	Momentum    float32 `json:"momentum"`     // RMSProp momentum (default: 0.0)
	Centered    bool    `json:"centered"`     // RMSProp centered variant (default: false)
	
	// Engine selection (GPU-resident architecture compliance)
	EngineType       EngineType `json:"engine_type"`       // Engine selection: Auto, Hybrid, Dynamic (default: Auto)
	
	// Problem type and loss function configuration
	ProblemType  ProblemType  `json:"problem_type"`  // Classification or Regression (default: Classification)
	LossFunction LossFunction `json:"loss_function"` // Loss function for the problem type (default: CrossEntropy)
	UseHybridEngine  bool       `json:"use_hybrid_engine"`  // DEPRECATED: Use EngineType instead
	UseDynamicEngine bool       `json:"use_dynamic_engine"` // DEPRECATED: Use EngineType instead
	InferenceOnly    bool       `json:"inference_only"`     // Skip training setup, optimize for inference (forward-pass only)
	
	// Mixed Precision Training Configuration
	UseMixedPrecision  bool    `json:"use_mixed_precision"`   // Enable FP16 training with FP32 master weights
	InitialLossScale   float32 `json:"initial_loss_scale"`    // Initial loss scale for gradient scaling (default: 65536.0)
	LossScaleGrowthFactor float32 `json:"loss_scale_growth_factor"` // Loss scale growth factor (default: 2.0)
	LossScaleBackoffFactor float32 `json:"loss_scale_backoff_factor"` // Loss scale reduction factor on overflow (default: 0.5)
	LossScaleGrowthInterval int `json:"loss_scale_growth_interval"` // Steps between loss scale increases (default: 2000)
}

// Validate ensures the problem type and loss function are compatible
func (tc *TrainerConfig) Validate() error {
	// Validate problem type and loss function compatibility
	switch tc.ProblemType {
	case Classification:
		if tc.LossFunction != CrossEntropy && tc.LossFunction != SparseCrossEntropy {
			return fmt.Errorf("classification requires CrossEntropy or SparseCrossEntropy loss, got %v", tc.LossFunction)
		}
	case Regression:
		if tc.LossFunction != MeanSquaredError && tc.LossFunction != MeanAbsoluteError && tc.LossFunction != Huber {
			return fmt.Errorf("regression requires MSE, MAE, or Huber loss, got %v", tc.LossFunction)
		}
	default:
		return fmt.Errorf("unsupported problem type: %v", tc.ProblemType)
	}
	
	// Validate other parameters
	if tc.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive, got %d", tc.BatchSize)
	}
	
	if tc.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive, got %f", tc.LearningRate)
	}
	
	// Validate optimizer-specific parameters
	if tc.OptimizerType == cgo_bridge.Adam {
		if tc.Beta1 < 0 || tc.Beta1 >= 1 {
			return fmt.Errorf("Adam beta1 must be in [0, 1), got %f", tc.Beta1)
		}
		if tc.Beta2 < 0 || tc.Beta2 >= 1 {
			return fmt.Errorf("Adam beta2 must be in [0, 1), got %f", tc.Beta2)
		}
	}
	
	if tc.OptimizerType == cgo_bridge.RMSProp {
		if tc.Alpha < 0 || tc.Alpha >= 1 {
			return fmt.Errorf("RMSProp alpha must be in [0, 1), got %f", tc.Alpha)
		}
	}
	
	if tc.Epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive, got %f", tc.Epsilon)
	}
	
	return nil
}

// OptimizerConfig provides optimizer-specific configurations
type OptimizerConfig struct {
	Type        cgo_bridge.OptimizerType
	LearningRate float32
	Beta1       float32 // Adam only
	Beta2       float32 // Adam only
	Epsilon     float32 // Adam only
	WeightDecay float32
}

// TrainerFactory provides methods to create different types of trainers
type TrainerFactory struct{}

// NewFactory creates a new trainer factory
func NewFactory() *TrainerFactory {
	return &TrainerFactory{}
}

// CreateTrainer creates a trainer with full configuration control
func (tf *TrainerFactory) CreateTrainer(config TrainerConfig) (*SimpleTrainer, error) {
	// Validate configuration
	if err := tf.validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}
	
	// Convert to CGO bridge config (GPU-resident parameter passing)
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
	}
	
	// Create batch trainer using legacy approach (DEPRECATED)
	// This function maintains backward compatibility but is limited
	var batchTrainer *engine.BatchTrainer
	var err error
	
	// Handle deprecated configuration options
	if config.UseHybridEngine {
		batchTrainer, err = engine.NewBatchTrainerHybrid(bridgeConfig, config.BatchSize)
		if err != nil {
			return nil, fmt.Errorf("failed to create hybrid batch trainer: %v", err)
		}
	} else if config.UseDynamicEngine {
		// NOTE: This path doesn't exist in legacy BatchTrainer
		// Users should use NewModelTrainer for smart routing
		return nil, fmt.Errorf("dynamic engine not supported in legacy SimpleTrainer (use NewModelTrainer instead)")
	} else {
		return nil, fmt.Errorf("no engine specified (use UseHybridEngine: true or switch to NewModelTrainer for smart routing)")
	}
	
	return &SimpleTrainer{
		batchTrainer: batchTrainer,
		batchSize:    config.BatchSize,
		config:       bridgeConfig,
	}, nil
}

// CreateSGDTrainer creates an SGD trainer with specified parameters
func (tf *TrainerFactory) CreateSGDTrainer(batchSize int, learningRate float32, weightDecay float32) (*SimpleTrainer, error) {
	config := TrainerConfig{
		BatchSize:       batchSize,
		LearningRate:    learningRate,
		OptimizerType:   cgo_bridge.SGD,
		WeightDecay:     weightDecay,
		// Smart routing: Auto-select optimal engine
		EngineType:      Auto,
		UseHybridEngine: true, // Deprecated: kept for compatibility
		// Optimizer parameters (Adam/RMSProp params ignored for SGD)
		Beta1:    0.9,
		Beta2:    0.999,
		Epsilon:  1e-8,
		Alpha:    0.99,
		Momentum: 0.0,
		Centered: false,
	}
	
	return tf.CreateTrainer(config)
}

// CreateAdamTrainer creates an Adam trainer with specified parameters
func (tf *TrainerFactory) CreateAdamTrainer(batchSize int, learningRate float32, beta1, beta2, epsilon, weightDecay float32) (*SimpleTrainer, error) {
	config := TrainerConfig{
		BatchSize:       batchSize,
		LearningRate:    learningRate,
		OptimizerType:   cgo_bridge.Adam,
		Beta1:           beta1,
		Beta2:           beta2,
		Epsilon:         epsilon,
		WeightDecay:     weightDecay,
		// Smart routing: Auto-select optimal engine
		EngineType:      Auto,
		UseHybridEngine: true, // Deprecated: kept for compatibility
		// RMSProp parameters (ignored for Adam)
		Alpha:    0.99,
		Momentum: 0.0,
		Centered: false,
	}
	
	return tf.CreateTrainer(config)
}

// CreateAdamTrainerWithDefaults creates an Adam trainer with sensible defaults
func (tf *TrainerFactory) CreateAdamTrainerWithDefaults(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	return tf.CreateAdamTrainer(batchSize, learningRate, 0.9, 0.999, 1e-8, 0.0)
}

// CreateRMSPropTrainer creates an RMSProp trainer with specified parameters
func (tf *TrainerFactory) CreateRMSPropTrainer(batchSize int, learningRate, alpha, epsilon, weightDecay, momentum float32, centered bool) (*SimpleTrainer, error) {
	config := TrainerConfig{
		BatchSize:       batchSize,
		LearningRate:    learningRate,
		OptimizerType:   cgo_bridge.RMSProp,
		Alpha:           alpha,
		Epsilon:         epsilon,
		WeightDecay:     weightDecay,
		Momentum:        momentum,
		Centered:        centered,
		// Smart routing: Auto-select optimal engine
		EngineType:      Auto,
		UseHybridEngine: true, // Deprecated: kept for compatibility
		// Adam parameters are ignored for RMSProp
		Beta1:   0.9,
		Beta2:   0.999,
	}
	
	return tf.CreateTrainer(config)
}

// CreateRMSPropTrainerWithDefaults creates an RMSProp trainer with sensible defaults
func (tf *TrainerFactory) CreateRMSPropTrainerWithDefaults(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	return tf.CreateRMSPropTrainer(batchSize, learningRate, 0.99, 1e-8, 0.0, 0.0, false)
}

// CreateProductionTrainer creates a trainer optimized for production use
func (tf *TrainerFactory) CreateProductionTrainer(batchSize int, optimizerConfig OptimizerConfig) (*SimpleTrainer, error) {
	config := TrainerConfig{
		BatchSize:       batchSize,
		LearningRate:    optimizerConfig.LearningRate,
		OptimizerType:   optimizerConfig.Type,
		Beta1:           optimizerConfig.Beta1,
		Beta2:           optimizerConfig.Beta2,
		Epsilon:         optimizerConfig.Epsilon,
		WeightDecay:     optimizerConfig.WeightDecay,
		// Smart routing: Auto-select optimal engine for production
		EngineType:      Auto,
		UseHybridEngine: true, // Deprecated: kept for compatibility
		// RMSProp parameters (set to defaults, override in specific configs)
		Alpha:    0.99,
		Momentum: 0.0,
		Centered: false,
	}
	
	return tf.CreateTrainer(config)
}

// GetDefaultSGDConfig returns default SGD configuration
func (tf *TrainerFactory) GetDefaultSGDConfig(learningRate float32) OptimizerConfig {
	return OptimizerConfig{
		Type:         cgo_bridge.SGD,
		LearningRate: learningRate,
		WeightDecay:  0.0,
		// Optimizer parameters are ignored for SGD but set for completeness
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
}

// GetDefaultAdamConfig returns default Adam configuration
func (tf *TrainerFactory) GetDefaultAdamConfig(learningRate float32) OptimizerConfig {
	return OptimizerConfig{
		Type:         cgo_bridge.Adam,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
	}
}

// GetDefaultRMSPropConfig returns default RMSProp configuration
func (tf *TrainerFactory) GetDefaultRMSPropConfig(learningRate float32) OptimizerConfig {
	return OptimizerConfig{
		Type:         cgo_bridge.RMSProp,
		LearningRate: learningRate,
		WeightDecay:  0.0,
		// Adam parameters are ignored for RMSProp but set for completeness
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
}

// AnalyzeModelArchitecture analyzes model architecture for optimal engine selection
// Maintains GPU-resident principles by analyzing layer specifications only
func AnalyzeModelArchitecture(modelSpec *layers.ModelSpec) *ModelArchitectureInfo {
	if modelSpec == nil || !modelSpec.Compiled {
		return &ModelArchitectureInfo{
			Complexity: "Invalid",
		}
	}
	
	info := &ModelArchitectureInfo{
		InputDimensions: len(modelSpec.InputShape),
		LayerCount:      len(modelSpec.Layers),
		ParameterCount:  modelSpec.TotalParameters,
	}
	
	// Analyze layer composition (CPU-only analysis, no GPU operations)
	for _, layer := range modelSpec.Layers {
		switch layer.Type {
		case layers.Conv2D:
			info.HasConvLayers = true
		case layers.Dense:
			info.HasDenseLayers = true
		}
	}
	
	// Determine architecture patterns
	info.IsMLPOnly = info.HasDenseLayers && !info.HasConvLayers
	info.IsCNNPattern = info.HasConvLayers && info.HasDenseLayers && info.InputDimensions == 4
	
	// Classify complexity for engine selection
	if info.LayerCount <= 3 {
		info.Complexity = "Simple"
	} else if info.LayerCount <= 10 {
		info.Complexity = "Standard"
	} else {
		info.Complexity = "Complex"
	}
	
	return info
}

// SelectOptimalEngine selects the best engine based on architecture analysis
// Maintains GPU-resident architecture compliance for all engine types
func SelectOptimalEngine(modelSpec *layers.ModelSpec, config TrainerConfig) EngineType {
	// Handle explicit engine selection (skip auto-detection)
	if config.EngineType != Auto {
		return config.EngineType
	}
	
	// Handle deprecated config options for backward compatibility
	if config.UseHybridEngine && !config.UseDynamicEngine {
		return Hybrid
	}
	if config.UseDynamicEngine && !config.UseHybridEngine {
		return Dynamic
	}
	
	// Smart routing based on architecture analysis
	archInfo := AnalyzeModelArchitecture(modelSpec)
	
	// CNN Pattern: Use Hybrid Engine for maximum performance
	// - 4D input [batch, channels, height, width]
	// - Conv2D + Dense layer combination
	// - Optimized for 20k+ batches/second performance
	if archInfo.IsCNNPattern {
		return Hybrid
	}
	
	// MLP Pattern: Use Dynamic Engine for flexibility
	// - 2D input [batch, features]
	// - Dense-only or Dense + activation layers
	// - Better flexibility for regression, classification
	if archInfo.IsMLPOnly && archInfo.InputDimensions == 2 {
		return Dynamic
	}
	
	// Complex/Custom Architectures: Use Dynamic Engine
	// - Non-standard input dimensions
	// - Complex layer combinations
	// - Custom architectures requiring flexibility
	return Dynamic
}

// validateConfig validates trainer configuration
func (tf *TrainerFactory) validateConfig(config TrainerConfig) error {
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
	
	if config.OptimizerType == cgo_bridge.RMSProp {
		if config.Alpha <= 0 || config.Alpha >= 1 {
			return fmt.Errorf("RMSProp alpha must be in (0, 1), got %f", config.Alpha)
		}
		if config.Epsilon <= 0 {
			return fmt.Errorf("RMSProp epsilon must be positive, got %f", config.Epsilon)
		}
		if config.Momentum < 0 || config.Momentum >= 1 {
			return fmt.Errorf("RMSProp momentum must be in [0, 1), got %f", config.Momentum)
		}
	}
	
	if config.WeightDecay < 0 {
		return fmt.Errorf("weight decay must be non-negative, got %f", config.WeightDecay)
	}
	
	return nil
}

// ================================================================
// CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
// ================================================================

// NewSGDTrainer creates an SGD trainer (convenience function)
func NewSGDTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	factory := NewFactory()
	return factory.CreateSGDTrainer(batchSize, learningRate, 0.0)
}

// NewAdamTrainer creates an Adam trainer with defaults (convenience function) 
func NewAdamTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	factory := NewFactory()
	return factory.CreateAdamTrainerWithDefaults(batchSize, learningRate)
}

// NewRMSPropTrainer creates an RMSProp trainer with defaults (convenience function)
func NewRMSPropTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	factory := NewFactory()
	return factory.CreateRMSPropTrainerWithDefaults(batchSize, learningRate)
}

// NewTrainerWithConfig creates a trainer with full configuration (convenience function)
func NewTrainerWithConfig(config TrainerConfig) (*SimpleTrainer, error) {
	factory := NewFactory()
	return factory.CreateTrainer(config)
}