package training

import (
	"fmt"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/engine"
	"github.com/tsawler/go-metal/memory"
)

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

// TrainerConfig provides comprehensive configuration for training
type TrainerConfig struct {
	// Training parameters
	BatchSize    int     `json:"batch_size"`
	LearningRate float32 `json:"learning_rate"`
	
	// Optimizer configuration
	OptimizerType cgo_bridge.OptimizerType `json:"optimizer_type"`
	
	// Adam-specific parameters (ignored for SGD)
	Beta1       float32 `json:"beta1"`        // Adam momentum decay (default: 0.9)
	Beta2       float32 `json:"beta2"`        // Adam RMSprop decay (default: 0.999)
	Epsilon     float32 `json:"epsilon"`      // Adam numerical stability (default: 1e-8)
	WeightDecay float32 `json:"weight_decay"` // L2 regularization (default: 0.0)
	
	// Training behavior
	UseHybridEngine  bool `json:"use_hybrid_engine"`  // Use hybrid MPS/MPSGraph (recommended: true)
	UseDynamicEngine bool `json:"use_dynamic_engine"` // Use dynamic graph creation for any architecture (recommended: true)
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
	
	// Convert to CGO bridge config
	bridgeConfig := cgo_bridge.TrainingConfig{
		LearningRate:  config.LearningRate,
		Beta1:         config.Beta1,
		Beta2:         config.Beta2,
		WeightDecay:   config.WeightDecay,
		Epsilon:       config.Epsilon,
		OptimizerType: config.OptimizerType,
	}
	
	// Create batch trainer
	var batchTrainer *engine.BatchTrainer
	var err error
	
	if config.UseHybridEngine {
		batchTrainer, err = engine.NewBatchTrainerHybrid(bridgeConfig, config.BatchSize)
		if err != nil {
			return nil, fmt.Errorf("failed to create hybrid batch trainer: %v", err)
		}
	} else {
		return nil, fmt.Errorf("non-hybrid engine not supported (use UseHybridEngine: true)")
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
		UseHybridEngine: true,
		// Adam parameters are ignored for SGD
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
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
		UseHybridEngine: true,
	}
	
	return tf.CreateTrainer(config)
}

// CreateAdamTrainerWithDefaults creates an Adam trainer with sensible defaults
func (tf *TrainerFactory) CreateAdamTrainerWithDefaults(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	return tf.CreateAdamTrainer(batchSize, learningRate, 0.9, 0.999, 1e-8, 0.0)
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
		UseHybridEngine: true, // Always use hybrid for production
	}
	
	return tf.CreateTrainer(config)
}

// GetDefaultSGDConfig returns default SGD configuration
func (tf *TrainerFactory) GetDefaultSGDConfig(learningRate float32) OptimizerConfig {
	return OptimizerConfig{
		Type:         cgo_bridge.SGD,
		LearningRate: learningRate,
		WeightDecay:  0.0,
		// Adam parameters are ignored for SGD but set for completeness
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

// NewTrainerWithConfig creates a trainer with full configuration (convenience function)
func NewTrainerWithConfig(config TrainerConfig) (*SimpleTrainer, error) {
	factory := NewFactory()
	return factory.CreateTrainer(config)
}