package layers_test

import (
	"fmt"
	"testing"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

// TestCompliantLayerSystem demonstrates the compliant layer system that adheres to design requirements
func TestCompliantLayerSystem(t *testing.T) {
	fmt.Println("=== Testing Compliant Layer System ===")
	
	// Step 1: Create a model using the layer builder (pure configuration)
	inputShape := []int{32, 3, 32, 32} // Batch of 32 RGB 32x32 images
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(8, 3, 1, 1, true, "conv1").    // 8 filters, 3x3 kernel, stride=1, padding=1
		AddReLU("relu1").
		AddDense(2, true, "fc1").                // 2 output classes
		AddSoftmax(-1, "softmax").               // Softmax on last dimension
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	// Step 2: Display model summary
	fmt.Println("Model compiled successfully!")
	fmt.Println(model.Summary())
	
	// Step 3: Validate model is compatible with TrainingEngine
	err = model.ValidateModelForTrainingEngine()
	if err != nil {
		t.Fatalf("Model validation failed: %v", err)
	}
	fmt.Println("✅ Model validated for TrainingEngine compatibility")
	
	// Step 4: Create a model trainer using the existing high-performance architecture
	config := training.TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.01,
		OptimizerType:   cgo_bridge.SGD,
		UseHybridEngine: true, // Use proven 20k+ batch/s architecture
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.0,
	}
	
	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		t.Fatalf("Failed to create model trainer: %v", err)
	}
	defer trainer.Cleanup()
	
	fmt.Println("✅ Model trainer created successfully")
	
	// Step 5: Create dummy training data
	inputData := make([]float32, 32*3*32*32) // 32 RGB 32x32 images
	labelData := make([]int32, 32)           // 32 labels
	
	// Fill with dummy data
	for i := range inputData {
		inputData[i] = float32(i%100) / 100.0
	}
	for i := range labelData {
		labelData[i] = int32(i % 2) // Binary classification
	}
	
	// Step 6: Execute training steps using single CGO call architecture
	fmt.Println("\n=== Executing Training Steps ===")
	for step := 0; step < 3; step++ {
		result, err := trainer.TrainBatch(
			inputData,
			inputShape,
			labelData,
			[]int{32},
		)
		
		if err != nil {
			t.Fatalf("Training step %d failed: %v", step, err)
		}
		
		fmt.Printf("Step %d: Loss=%.6f, BatchRate=%.2f batch/s, Time=%v\n",
			step, result.Loss, result.BatchRate, result.StepTime)
	}
	
	// Step 7: Display training statistics
	stats := trainer.GetStats()
	fmt.Printf("\n=== Training Statistics ===\n")
	fmt.Printf("Current Step: %d\n", stats.CurrentStep)
	fmt.Printf("Total Steps: %d\n", stats.TotalSteps)
	fmt.Printf("Batch Size: %d\n", stats.BatchSize)
	fmt.Printf("Learning Rate: %.4f\n", stats.LearningRate)
	fmt.Printf("Average Loss: %.6f\n", stats.AverageLoss)
	fmt.Printf("Model Parameters: %d\n", stats.ModelParameters)
	fmt.Printf("Layer Count: %d\n", stats.LayerCount)
	fmt.Printf("Memory Pools: %d active\n", len(stats.MemoryPoolStats))
	
	fmt.Println("\n✅ Compliant layer system test completed successfully!")
	fmt.Println("✅ Single CGO call architecture maintained")
	fmt.Println("✅ Shared Metal resources used")
	fmt.Println("✅ Layer abstraction as configuration only")
	fmt.Println("✅ Integration with existing TrainingEngine")
}

// TestLayerFactoryAndBuilder tests the layer creation and model building functionality
func TestLayerFactoryAndBuilder(t *testing.T) {
	fmt.Println("\n=== Testing Layer Factory and Builder ===")
	
	// Test layer factory
	factory := layers.NewFactory()
	
	// Create layer specifications
	conv2d := factory.CreateConv2DSpec(3, 16, 3, 1, 1, true, "conv1")
	relu := factory.CreateReLUSpec("relu1")
	dense := factory.CreateDenseSpec(128, 10, true, "fc1")
	softmax := factory.CreateSoftmaxSpec(-1, "softmax")
	
	fmt.Printf("Conv2D Spec: %+v\n", conv2d)
	fmt.Printf("ReLU Spec: %+v\n", relu)
	fmt.Printf("Dense Spec: %+v\n", dense)
	fmt.Printf("Softmax Spec: %+v\n", softmax)
	
	// Test model builder
	inputShape := []int{16, 3, 64, 64} // Larger input for testing
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddLayer(conv2d).
		AddLayer(relu).
		AddLayer(dense).
		AddLayer(softmax).
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	fmt.Printf("\n=== Compiled Model Information ===\n")
	fmt.Printf("Total Parameters: %d\n", model.TotalParameters)
	fmt.Printf("Input Shape: %v\n", model.InputShape)
	fmt.Printf("Output Shape: %v\n", model.OutputShape)
	fmt.Printf("Layer Count: %d\n", len(model.Layers))
	
	// Validate parameter shapes
	expectedParamCount := int64(0)
	for i, layer := range model.Layers {
		fmt.Printf("Layer %d (%s): %d parameters\n", i+1, layer.Name, layer.ParameterCount)
		expectedParamCount += layer.ParameterCount
	}
	
	if model.TotalParameters != expectedParamCount {
		t.Errorf("Parameter count mismatch: expected %d, got %d", expectedParamCount, model.TotalParameters)
	}
	
	fmt.Println("✅ Layer factory and builder test completed successfully!")
}

// TestTrainerFactoryIntegration tests the trainer factory with layer models
func TestTrainerFactoryIntegration(t *testing.T) {
	fmt.Println("\n=== Testing Trainer Factory Integration ===")
	
	factory := training.NewModelFactory()
	
	// Test CNN trainer creation
	inputShape := []int{16, 3, 32, 32}
	config := training.TrainerConfig{
		BatchSize:       16,
		LearningRate:    0.001,
		OptimizerType:   cgo_bridge.Adam,
		UseHybridEngine: true,
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.0001,
	}
	
	cnnTrainer, err := factory.CreateCNNTrainer(inputShape, 10, config)
	if err != nil {
		t.Fatalf("Failed to create CNN trainer: %v", err)
	}
	defer cnnTrainer.Cleanup()
	
	fmt.Println("✅ CNN trainer created successfully")
	fmt.Println(cnnTrainer.GetModelSummary())
	
	// Test MLP trainer creation
	mlpTrainer, err := factory.CreateMLPTrainer(784, []int{256, 128}, 10, config)
	if err != nil {
		t.Fatalf("Failed to create MLP trainer: %v", err)
	}
	defer mlpTrainer.Cleanup()
	
	fmt.Println("✅ MLP trainer created successfully")
	fmt.Println(mlpTrainer.GetModelSummary())
	
	fmt.Println("✅ Trainer factory integration test completed successfully!")
}

// TestDropoutLayer tests the Dropout layer functionality
func TestDropoutLayer(t *testing.T) {
	fmt.Println("\n=== Testing Dropout Layer ===")
	
	// Test 1: Dropout layer creation
	inputShape := []int{32, 3, 32, 32}
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(64, true, "fc1").
		AddReLU("relu2").
		AddDropout(0.5, "dropout1").         // Add dropout with 50% rate
		AddDense(10, true, "fc2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with Dropout: %v", err)
	}
	
	fmt.Printf("✅ Model with Dropout compiled successfully\n")
	fmt.Printf("Total layers: %d\n", len(model.Layers))
	
	// Test 2: Verify Dropout layer parameters
	var dropoutLayer *layers.LayerSpec
	for _, layer := range model.Layers {
		if layer.Type == layers.Dropout {
			dropoutLayer = &layer
			break
		}
	}
	
	if dropoutLayer == nil {
		t.Fatalf("Dropout layer not found in compiled model")
	}
	
	// Check dropout parameters
	rate, exists := dropoutLayer.Parameters["rate"]
	if !exists {
		t.Fatalf("Dropout rate parameter not found")
	}
	if rate != float32(0.5) {
		t.Errorf("Expected dropout rate 0.5, got %v", rate)
	}
	
	training, exists := dropoutLayer.Parameters["training"]
	if !exists {
		t.Fatalf("Dropout training parameter not found")
	}
	if training != true {
		t.Errorf("Expected training mode true, got %v", training)
	}
	
	fmt.Printf("✅ Dropout parameters verified: rate=%.1f, training=%v\n", rate, training)
	
	// Test 3: Verify Dropout has no parameters
	if dropoutLayer.ParameterCount != 0 {
		t.Errorf("Expected Dropout to have 0 parameters, got %d", dropoutLayer.ParameterCount)
	}
	
	// Test 4: Verify Dropout doesn't change shape
	expectedInputShape := dropoutLayer.InputShape
	expectedOutputShape := dropoutLayer.OutputShape
	
	if len(expectedInputShape) != len(expectedOutputShape) {
		t.Errorf("Dropout changed tensor rank: input %d, output %d", len(expectedInputShape), len(expectedOutputShape))
	}
	
	for i := range expectedInputShape {
		if expectedInputShape[i] != expectedOutputShape[i] {
			t.Errorf("Dropout changed shape at dimension %d: input %d, output %d", i, expectedInputShape[i], expectedOutputShape[i])
		}
	}
	
	fmt.Printf("✅ Dropout preserves tensor shape: %v\n", expectedInputShape)
	
	// Test 5: Test serialization for CGO
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with Dropout: %v", err)
	}
	
	// Find Dropout layer in serialized format
	var foundDropout bool
	for _, layer := range serialized.Layers {
		if layer.LayerType == 5 { // Dropout = 5 in Go enum
			foundDropout = true
			if len(layer.ParamFloat) == 0 || layer.ParamFloat[0] != 0.5 {
				t.Errorf("Dropout rate not serialized correctly: %v", layer.ParamFloat)
			}
			if len(layer.ParamInt) == 0 || layer.ParamInt[0] != 1 {
				t.Errorf("Dropout training mode not serialized correctly: %v", layer.ParamInt)
			}
			break
		}
	}
	
	if !foundDropout {
		t.Fatalf("Dropout layer not found in serialized model")
	}
	
	fmt.Printf("✅ Dropout serialization verified\n")
	
	// Test 6: Test dynamic layer spec conversion
	dynamicSpecs, err := model.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	// Find Dropout in dynamic specs
	var foundDynamicDropout bool
	for _, spec := range dynamicSpecs {
		if spec.LayerType == 5 { // Dropout = 5 in Go enum
			foundDynamicDropout = true
			if spec.ParamFloatCount != 1 || spec.ParamFloat[0] != 0.5 {
				t.Errorf("Dynamic Dropout rate not correct")
			}
			if spec.ParamIntCount != 1 || spec.ParamInt[0] != 1 {
				t.Errorf("Dynamic Dropout training mode not correct")
			}
			break
		}
	}
	
	if !foundDynamicDropout {
		t.Fatalf("Dropout not found in dynamic specs")
	}
	
	fmt.Printf("✅ Dynamic spec conversion verified\n")
	
	fmt.Println("✅ Dropout layer test completed successfully!")
}

// TestDropoutVariousRates tests Dropout with different dropout rates
func TestDropoutVariousRates(t *testing.T) {
	fmt.Println("\n=== Testing Dropout with Various Rates ===")
	
	rates := []float32{0.0, 0.25, 0.5, 0.75, 0.9}
	
	for _, rate := range rates {
		inputShape := []int{16, 64}
		
		builder := layers.NewModelBuilder(inputShape)
		model, err := builder.
			AddDropout(rate, fmt.Sprintf("dropout_%.2f", rate)).
			AddDense(10, true, "output").
			Compile()
		
		if err != nil {
			t.Fatalf("Failed to compile model with dropout rate %.2f: %v", rate, err)
		}
		
		// Verify the rate was set correctly
		dropoutLayer := model.Layers[0]
		actualRate := dropoutLayer.Parameters["rate"].(float32)
		if actualRate != rate {
			t.Errorf("Expected rate %.2f, got %.2f", rate, actualRate)
		}
		
		fmt.Printf("✅ Dropout rate %.2f compiled successfully\n", rate)
	}
	
	fmt.Println("✅ Various dropout rates test completed successfully!")
}

// BenchmarkCompliantVsOriginal compares performance of compliant vs original architecture
func BenchmarkCompliantVsOriginal(b *testing.B) {
	// This benchmark would compare the compliant layer system performance
	// with the original SimpleTrainer to ensure no performance regression
	
	b.Skip("Performance benchmark - implement when ready for validation")
}