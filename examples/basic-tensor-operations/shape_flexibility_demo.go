package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

func testShapeFlexibility() {
	fmt.Println("=== Shape Flexibility Test ===")
	fmt.Println("Testing the enhanced shape handling capabilities")
	fmt.Println("Focuses on successfully implemented C-side transformations\n")

	rand.Seed(42)

	// Test Case 1: MLP with 2D input and dynamic batch sizes
	fmt.Println("--- Test 1: Enhanced MLP with 2D input flexibility ---")
	testMLP2D()

	// Test Case 2: Standard 2D CNN with enhanced shape handling
	fmt.Println("\n--- Test 2: Enhanced 2D CNN with flexible BatchNorm ---")
	test2DCNN()

	// Test Case 3: Adaptive tensor flattening demonstrations
	fmt.Println("\n--- Test 3: Adaptive tensor flattening capabilities ---")
	testAdaptiveFlattening()

	// Test Case 4: Enhanced BatchNorm broadcasting flexibility
	fmt.Println("\n--- Test 4: Enhanced BatchNorm broadcasting ---")
	testEnhancedBatchNormBroadcasting()

	fmt.Println("\n🎉 Shape Flexibility Test Complete!")
	fmt.Println("✅ All C-side shape transformations working successfully")
	fmt.Println("✅ Enhanced flexibility within architectural constraints")
}

func testMLP2D() {
	// Pure MLP: 2D input → Dense layers
	batchSize := 8
	features := 128
	inputShape := []int{batchSize, features}

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddDense(64, true, "fc1").
		AddReLU("relu1").
		AddBatchNorm(64, 1e-5, 0.1, true, "bn1").
		AddDense(32, true, "fc2").
		AddReLU("relu2").
		AddDense(4, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build MLP model: %v", err)
	}

	fmt.Printf("✅ MLP Model: %d layers, %d parameters\n", len(model.Layers), model.TotalParameters)

	// Test with different batch sizes
	testBatchSizes := []int{4, 8, 16}
	for _, testBatchSize := range testBatchSizes {
		inputData := make([]float32, testBatchSize*features)
		for i := range inputData {
			inputData[i] = rand.Float32()*2 - 1
		}

		labelData := make([]int32, testBatchSize)
		for i := range labelData {
			labelData[i] = int32(rand.Intn(4))
		}

		config := training.TrainerConfig{
			BatchSize:        testBatchSize,
			LearningRate:     0.01,
			OptimizerType:    cgo_bridge.Adam,
			UseDynamicEngine: true,
			Beta1:            0.9,
			Beta2:            0.999,
			Epsilon:          1e-8,
		}

		trainer, err := training.NewModelTrainer(model, config)
		if err != nil {
			log.Fatalf("Failed to create trainer: %v", err)
		}

		result, err := trainer.TrainBatchWithCommandPool(inputData, []int{testBatchSize, features}, labelData, []int{testBatchSize})
		if err != nil {
			log.Fatalf("Training failed: %v", err)
		}

		fmt.Printf("  Batch %d: Loss=%.4f\n", testBatchSize, result.Loss)
		trainer.Cleanup()
	}
}

func testAdaptiveFlattening() {
	// Test different scenarios where adaptive flattening is beneficial
	testConfigs := []struct {
		name       string
		inputShape []int
		description string
	}{
		{"Standard 2D", []int{8, 256}, "Already 2D - no flattening needed"},
		{"Small 4D CNN", []int{8, 8, 4, 4}, "4D→2D adaptive flattening (8*4*4=128)"},
		{"Large Feature Map", []int{8, 32, 8, 8}, "4D→2D adaptive flattening (32*8*8=2048)"},
	}

	for _, config := range testConfigs {
		fmt.Printf("  Testing %s: %v\n", config.name, config.inputShape)
		fmt.Printf("    %s\n", config.description)

		builder := layers.NewModelBuilder(config.inputShape)
		model, err := builder.
			AddDense(64, true, "fc1"). // Uses adaptive flattening for >2D inputs
			AddReLU("relu1").
			AddBatchNorm(64, 1e-5, 0.1, true, "bn1"). // 2D BatchNorm
			AddDense(32, true, "fc2").
			AddReLU("relu2").
			AddDense(4, true, "output").
			Compile()

		if err != nil {
			log.Fatalf("Failed to build adaptive flattening model for %s: %v", config.name, err)
		}

		trainConfig := training.TrainerConfig{
			BatchSize:        config.inputShape[0],
			LearningRate:     0.01,
			OptimizerType:    cgo_bridge.Adam,
			UseDynamicEngine: true,
			Beta1:            0.9,
			Beta2:            0.999,
			Epsilon:          1e-8,
		}

		trainer, err := training.NewModelTrainer(model, trainConfig)
		if err != nil {
			log.Fatalf("Failed to create trainer for %s: %v", config.name, err)
		}

		// Calculate total input size
		totalSize := 1
		for _, dim := range config.inputShape {
			totalSize *= dim
		}

		inputData := make([]float32, totalSize)
		for i := range inputData {
			inputData[i] = rand.Float32()*2 - 1
		}

		labelData := make([]int32, config.inputShape[0])
		for i := range labelData {
			labelData[i] = int32(rand.Intn(4))
		}

		result, err := trainer.TrainBatchWithCommandPool(inputData, config.inputShape, labelData, []int{config.inputShape[0]})
		if err != nil {
			log.Fatalf("Training failed for %s: %v", config.name, err)
		}

		fmt.Printf("    ✅ Training successful: Loss=%.4f, Params=%d\n", result.Loss, model.TotalParameters)
		trainer.Cleanup()
	}
}

func test2DCNN() {
	// Standard 2D CNN: 4D input → Conv2D → Dense
	batchSize := 8
	channels := 3
	height := 32
	width := 32
	inputShape := []int{batchSize, channels, height, width}

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(16, 3, 2, 1, true, "conv1").
		AddReLU("relu1").
		AddBatchNorm(16, 1e-5, 0.1, true, "bn1").
		AddConv2D(32, 3, 2, 1, true, "conv2").
		AddReLU("relu2").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn2").
		AddDense(16, true, "fc1").
		AddReLU("relu3").
		AddDense(4, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build 2D CNN model: %v", err)
	}

	fmt.Printf("✅ 2D CNN Model: %d layers, %d parameters\n", len(model.Layers), model.TotalParameters)

	inputData := make([]float32, batchSize*channels*height*width)
	for i := range inputData {
		inputData[i] = rand.Float32()*2 - 1
	}

	labelData := make([]int32, batchSize)
	for i := range labelData {
		labelData[i] = int32(rand.Intn(4))
	}

	config := training.TrainerConfig{
		BatchSize:        batchSize,
		LearningRate:     0.01,
		OptimizerType:    cgo_bridge.Adam,
		UseDynamicEngine: true,
		Beta1:            0.9,
		Beta2:            0.999,
		Epsilon:          1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create trainer: %v", err)
	}

	result, err := trainer.TrainBatchWithCommandPool(inputData, inputShape, labelData, []int{batchSize})
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("  Training successful: Loss=%.4f\n", result.Loss)
	trainer.Cleanup()
}

func testEnhancedBatchNormBroadcasting() {
	// Test enhanced BatchNorm broadcasting with complex CNN architectures
	fmt.Println("  Demonstrating flexible BatchNorm broadcasting in multi-layer CNNs")

	batchSize := 8
	inputShape := []int{batchSize, 16, 16, 16}

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		// First conv block with BatchNorm
		AddConv2D(32, 3, 1, 1, true, "conv1").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn1"). // 4D BatchNorm with flexible broadcasting
		AddReLU("relu1").
		// Second conv block with BatchNorm
		AddConv2D(64, 3, 2, 1, true, "conv2").
		AddBatchNorm(64, 1e-5, 0.1, true, "bn2"). // Different channel count
		AddReLU("relu2").
		// Transition to dense with BatchNorm
		AddDense(128, true, "fc1"). // Adaptive flattening: 4D→2D
		AddBatchNorm(128, 1e-5, 0.1, true, "bn3"). // 2D BatchNorm
		AddReLU("relu3").
		AddDense(4, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build enhanced BatchNorm model: %v", err)
	}

	fmt.Printf("  ✅ Complex model built: %d layers, %d parameters\n", len(model.Layers), model.TotalParameters)
	fmt.Println("  - Multiple BatchNorm layers with different channel counts")
	fmt.Println("  - 4D→4D BatchNorm (conv layers) + 2D→2D BatchNorm (dense layer)")
	fmt.Println("  - Adaptive flattening between conv and dense sections")

	config := training.TrainerConfig{
		BatchSize:        batchSize,
		LearningRate:     0.01,
		OptimizerType:    cgo_bridge.Adam,
		UseDynamicEngine: true,
		Beta1:            0.9,
		Beta2:            0.999,
		Epsilon:          1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()

	totalSize := batchSize * 16 * 16 * 16
	inputData := make([]float32, totalSize)
	for i := range inputData {
		inputData[i] = rand.Float32()*2 - 1
	}

	labelData := make([]int32, batchSize)
	for i := range labelData {
		labelData[i] = int32(rand.Intn(4))
	}

	result, err := trainer.TrainBatchWithCommandPool(inputData, inputShape, labelData, []int{batchSize})
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("  ✅ Enhanced BatchNorm training successful: Loss=%.4f\n", result.Loss)
	fmt.Println("  ✅ All shape transformations handled seamlessly by C-side implementation")
}

func main() {
	testShapeFlexibility()
}