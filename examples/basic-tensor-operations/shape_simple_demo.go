package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

func testSimpleShapeFlexibility() {
	fmt.Println("=== Simple Shape Flexibility Test ===")
	fmt.Println("Testing flexible shape transformations that are currently supported\n")

	rand.Seed(42)

	// Test Case 1: Enhanced BatchNorm flexibility
	fmt.Println("--- Test 1: Enhanced BatchNorm with different input ranks ---")
	testEnhancedBatchNorm()

	// Test Case 2: Adaptive tensor flattening
	fmt.Println("\n--- Test 2: Adaptive tensor flattening for Dense layers ---")
	testAdaptiveFlattening()

	fmt.Println("\n🎉 Simple Shape Flexibility Test Complete!")
}

func testEnhancedBatchNorm() {
	// Test standard 4D CNN with enhanced BatchNorm
	batchSize := 8
	inputShape := []int{batchSize, 16, 32, 32}

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(32, 3, 2, 1, true, "conv1").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn1"). // Should use flexible broadcasting
		AddReLU("relu1").
		AddConv2D(64, 3, 2, 1, true, "conv2").
		AddBatchNorm(64, 1e-5, 0.1, true, "bn2"). // Should use flexible broadcasting
		AddReLU("relu2").
		AddDense(32, true, "fc1"). // Should use adaptive flattening
		AddBatchNorm(32, 1e-5, 0.1, true, "bn3"). // 2D BatchNorm
		AddReLU("relu3").
		AddDense(4, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build enhanced BatchNorm model: %v", err)
	}

	fmt.Printf("✅ Enhanced BatchNorm Model: %d layers, %d parameters\n", len(model.Layers), model.TotalParameters)

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

	totalSize := batchSize * 16 * 32 * 32
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
	fmt.Println("  - 4D→4D BatchNorm with flexible broadcasting")
	fmt.Println("  - Adaptive flattening: 4D→2D for Dense layer")
	fmt.Println("  - 2D→2D BatchNorm with flexible broadcasting")
}

func testAdaptiveFlattening() {
	// Test models with different input configurations to show adaptive flattening
	testConfigs := []struct {
		name       string
		inputShape []int
		description string
	}{
		{"2D Input", []int{8, 128}, "Already 2D - no flattening needed"},
		{"4D Input", []int{8, 16, 8, 8}, "4D→2D adaptive flattening"},
	}

	for _, config := range testConfigs {
		fmt.Printf("  Testing %s: %v (%s)\n", config.name, config.inputShape, config.description)

		builder := layers.NewModelBuilder(config.inputShape)
		model, err := builder.
			AddDense(64, true, "fc1"). // Should handle any input dimensionality
			AddReLU("relu1").
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

		fmt.Printf("    ✅ %s training successful: Loss=%.4f\n", config.name, result.Loss)
		trainer.Cleanup()
	}
}

func main() {
	testSimpleShapeFlexibility()
}