package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

func simpleTest() {
	fmt.Println("=== Simple Dynamic Batch Size Test ===")
	fmt.Println("Testing that the model works with consistent batch size first...\n")

	// Use a fixed batch size to start
	batchSize := 8
	inputShape := []int{batchSize, 3, 32, 32}

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(4, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(4, true, "fc1").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build model: %v", err)
	}

	fmt.Printf("✅ Model built: %d layers, %d parameters\n", len(model.Layers), model.TotalParameters)

	config := training.TrainerConfig{
		BatchSize:        batchSize,
		LearningRate:     0.01,
		OptimizerType:    cgo_bridge.Adam,
		UseDynamicEngine: true,
		UseHybridEngine:  false,
		Beta1:            0.9,
		Beta2:            0.999,
		WeightDecay:      0.0001,
		Epsilon:          1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()

	fmt.Println("✅ Training engine created successfully")

	// Test with the same batch size (should work)
	fmt.Println("\n--- Testing with batch size 8 (model's native size) ---")
	inputData, labelData := generateTestData(8, 3, 32, 32, 4)
	
	result, err := trainer.TrainBatchWithCommandPool(inputData, []int{8, 3, 32, 32}, labelData, []int{8})
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("✅ Training successful: Loss=%.4f\n", result.Loss)
	if result.HasAccuracy {
		fmt.Printf("   Accuracy: %.2f%%\n", result.Accuracy*100)
	}

	// Test inference
	predictions, err := trainer.InferBatch(inputData, []int{8, 3, 32, 32})
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	fmt.Printf("✅ Inference successful: %d predictions generated\n", len(predictions.Predictions))

	fmt.Println("\n🎉 Basic functionality verified!")
}

func generateTestData(batchSize, channels, height, width, numClasses int) ([]float32, []int32) {
	rand.Seed(42) // Fixed seed for reproducibility

	inputSize := batchSize * channels * height * width
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = rand.Float32()*2 - 1
	}

	labelData := make([]int32, batchSize)
	for b := 0; b < batchSize; b++ {
		labelData[b] = int32(rand.Intn(numClasses))
	}

	return inputData, labelData
}

func main() {
	simpleTest()
}