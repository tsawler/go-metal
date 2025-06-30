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

// NewSimpleTrainer creates a new simple trainer
func NewSimpleTrainer(batchSize int, learningRate float32) (*SimpleTrainer, error) {
	config := cgo_bridge.TrainingConfig{
		LearningRate:  learningRate,
		Beta1:         0.9,
		Beta2:         0.999,
		WeightDecay:   0.0,
		Epsilon:       1e-8,
		OptimizerType: cgo_bridge.SGD, // Start with SGD for simplicity
	}
	
	// Use hybrid MPS/MPSGraph version to avoid MPSGraph convolution assertion
	batchTrainer, err := engine.NewBatchTrainerHybrid(config, batchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create hybrid batch trainer: %v", err)
	}
	
	return &SimpleTrainer{
		batchTrainer: batchTrainer,
		batchSize:    batchSize,
		config:       config,
	}, nil
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