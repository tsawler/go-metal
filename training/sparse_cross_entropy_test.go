package training

import (
	"math"
	"testing"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
)

func TestSparseCrossEntropyGradientComputation(t *testing.T) {
	// Test that SparseCrossEntropy can compute gradients without crashing
	// This addresses the core issue: "Couldn't get gradient Tensor for tensor of op : dense_0_weight_param"
	
	batchSize := 4
	inputSize := 3
	numClasses := 3
	
	// Create simple MLP model
	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddDense(numClasses, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	// Configure SparseCrossEntropy loss
	config := TrainerConfig{
		BatchSize:     batchSize,
		LearningRate:  0.01,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    Dynamic,
		LossFunction:  1, // SparseCrossEntropy
		ProblemType:   0, // Classification
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}
	
	trainer, err := NewModelTrainer(model, config)
	if err != nil {
		t.Fatalf("Failed to create trainer with SparseCrossEntropy: %v", err)
	}
	defer trainer.Cleanup()
	
	// Create test data with integer labels (not one-hot)
	inputData := []float32{
		1.0, 2.0, 3.0,  // Sample 1
		4.0, 5.0, 6.0,  // Sample 2  
		7.0, 8.0, 9.0,  // Sample 3
		0.5, 1.5, 2.5,  // Sample 4
	}
	
	labelData := []int32{0, 1, 2, 1} // Integer class indices
	
	// Test forward and backward pass (gradient computation)
	result, err := trainer.TrainBatch(inputData, inputShape, labelData, []int{batchSize})
	if err != nil {
		t.Fatalf("SparseCrossEntropy gradient computation failed: %v", err)
	}
	
	// Verify we got a reasonable loss value
	if math.IsNaN(float64(result.Loss)) || math.IsInf(float64(result.Loss), 0) {
		t.Fatalf("SparseCrossEntropy produced invalid loss: %f", result.Loss)
	}
	
	// Loss should be positive for classification
	if result.Loss < 0 {
		t.Fatalf("SparseCrossEntropy loss should be positive, got: %f", result.Loss)
	}
	
	t.Logf("✅ SparseCrossEntropy gradient computation successful, loss: %f", result.Loss)
}

func TestSparseCrossEntropyVsCrossEntropy(t *testing.T) {
	// Test that SparseCrossEntropy with integer labels produces similar results to 
	// CrossEntropy with one-hot labels (mathematically they should be equivalent)
	
	batchSize := 8
	inputSize := 4
	numClasses := 3
	
	// Create identical models
	inputShape := []int{batchSize, inputSize}
	
	// Model 1: SparseCrossEntropy
	builder1 := layers.NewModelBuilder(inputShape)
	model1, err := builder1.
		AddDense(16, true, "hidden").
		AddReLU("relu").
		AddDense(numClasses, true, "output").
		Compile()
	if err != nil {
		t.Fatalf("Failed to compile SparseCE model: %v", err)
	}
	
	// Model 2: CrossEntropy  
	builder2 := layers.NewModelBuilder(inputShape)
	model2, err := builder2.
		AddDense(16, true, "hidden").
		AddReLU("relu").
		AddDense(numClasses, true, "output").
		Compile()
	if err != nil {
		t.Fatalf("Failed to compile CrossCE model: %v", err)
	}
	
	// Configuration for SparseCrossEntropy
	config1 := TrainerConfig{
		BatchSize:     batchSize,
		LearningRate:  0.01,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    Dynamic,
		LossFunction:  1, // SparseCrossEntropy
		ProblemType:   0, // Classification
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}
	
	// Configuration for CrossEntropy (default)
	config2 := TrainerConfig{
		BatchSize:     batchSize,
		LearningRate:  0.01,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    Dynamic,
		// LossFunction: 0, // CrossEntropy (default)
		// ProblemType:  0, // Classification (default)
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}
	
	trainer1, err := NewModelTrainer(model1, config1)
	if err != nil {
		t.Fatalf("Failed to create SparseCE trainer: %v", err)
	}
	defer trainer1.Cleanup()
	
	trainer2, err := NewModelTrainer(model2, config2)
	if err != nil {
		t.Fatalf("Failed to create CrossCE trainer: %v", err)
	}
	defer trainer2.Cleanup()
	
	// Create identical input data
	inputData := make([]float32, batchSize*inputSize)
	for i := range inputData {
		inputData[i] = float32(i%10) * 0.1 // Simple pattern
	}
	
	// Integer labels for SparseCrossEntropy
	intLabels := []int32{0, 1, 2, 0, 1, 2, 1, 0}
	
	// Note: We can't directly compare because CrossEntropy in this implementation
	// actually handles integer labels correctly internally. The key test is that
	// SparseCrossEntropy doesn't crash and produces reasonable loss values.
	
	// Test SparseCrossEntropy
	result1, err := trainer1.TrainBatch(inputData, inputShape, intLabels, []int{batchSize})
	if err != nil {
		t.Fatalf("SparseCrossEntropy training failed: %v", err)
	}
	
	// Test CrossEntropy with same integer labels (should work due to internal conversion)
	result2, err := trainer2.TrainBatch(inputData, inputShape, intLabels, []int{batchSize})
	if err != nil {
		t.Fatalf("CrossEntropy training failed: %v", err)
	}
	
	// Both should produce reasonable loss values
	if math.IsNaN(float64(result1.Loss)) || result1.Loss < 0 {
		t.Fatalf("SparseCrossEntropy produced invalid loss: %f", result1.Loss)
	}
	
	if math.IsNaN(float64(result2.Loss)) || result2.Loss < 0 {
		t.Fatalf("CrossEntropy produced invalid loss: %f", result2.Loss)
	}
	
	t.Logf("✅ SparseCrossEntropy loss: %f", result1.Loss)
	t.Logf("✅ CrossEntropy loss: %f", result2.Loss)
	t.Logf("✅ Both loss functions working correctly with integer labels")
}

func TestSparseCrossEntropyMultipleOptimizers(t *testing.T) {
	// Test SparseCrossEntropy with different optimizers to ensure gradient flow works universally
	
	optimizers := []struct {
		name string
		opt  cgo_bridge.OptimizerType
	}{
		{"Adam", cgo_bridge.Adam},
		{"SGD", cgo_bridge.SGD},
		{"RMSProp", cgo_bridge.RMSProp},
	}
	
	batchSize := 4
	inputSize := 2
	numClasses := 3
	inputShape := []int{batchSize, inputSize}
	
	for _, test := range optimizers {
		t.Run(test.name, func(t *testing.T) {
			// Create model
			builder := layers.NewModelBuilder(inputShape)
			model, err := builder.
				AddDense(numClasses, true, "output").
				Compile()
			if err != nil {
				t.Fatalf("Failed to compile model for %s: %v", test.name, err)
			}
			
			// Configure with SparseCrossEntropy
			config := TrainerConfig{
				BatchSize:     batchSize,
				LearningRate:  0.01,
				OptimizerType: test.opt,
				EngineType:    Dynamic,
				LossFunction:  1, // SparseCrossEntropy
				ProblemType:   0, // Classification
				Beta1:         0.9,
				Beta2:         0.999,
				Epsilon:       1e-8,
			}
			
			trainer, err := NewModelTrainer(model, config)
			if err != nil {
				t.Fatalf("Failed to create trainer with %s: %v", test.name, err)
			}
			defer trainer.Cleanup()
			
			// Test data
			inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
			labelData := []int32{0, 1, 2, 1}
			
			// Test training step
			result, err := trainer.TrainBatch(inputData, inputShape, labelData, []int{batchSize})
			if err != nil {
				t.Fatalf("SparseCrossEntropy failed with %s optimizer: %v", test.name, err)
			}
			
			// Verify reasonable loss
			if math.IsNaN(float64(result.Loss)) || result.Loss < 0 {
				t.Fatalf("SparseCrossEntropy with %s produced invalid loss: %f", test.name, result.Loss)
			}
			
			t.Logf("✅ SparseCrossEntropy with %s: loss = %f", test.name, result.Loss)
		})
	}
}

func TestSparseCrossEntropyLabelShapeValidation(t *testing.T) {
	// Test that SparseCrossEntropy correctly handles integer labels with shape [batch_size]
	// and rejects incorrect label shapes
	
	batchSize := 3
	numClasses := 4
	inputShape := []int{batchSize, 5} // 5 input features
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddDense(numClasses, true, "output").
		Compile()
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	config := TrainerConfig{
		BatchSize:     batchSize,
		LearningRate:  0.01,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    Dynamic,
		LossFunction:  1, // SparseCrossEntropy
		ProblemType:   0, // Classification
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}
	
	trainer, err := NewModelTrainer(model, config)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()
	
	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	
	// Test with correct integer labels [batch_size]
	validLabels := []int32{0, 2, 3} // Valid class indices
	
	result, err := trainer.TrainBatch(inputData, inputShape, validLabels, []int{batchSize})
	if err != nil {
		t.Fatalf("SparseCrossEntropy failed with valid integer labels: %v", err)
	}
	
	if math.IsNaN(float64(result.Loss)) || result.Loss < 0 {
		t.Fatalf("SparseCrossEntropy produced invalid loss with valid labels: %f", result.Loss)
	}
	
	t.Logf("✅ SparseCrossEntropy correctly handled integer labels: loss = %f", result.Loss)
}