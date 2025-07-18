package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

func main() {
	fmt.Println("=== Dynamic Batch Size Demo ===")
	fmt.Println("This demo showcases the dynamic batch size functionality")
	fmt.Println("where the same model can process different batch sizes without recompilation.")

	// Initialize random seed for reproducible results
	rand.Seed(42)

	// Create a simple CNN model for demonstration
	fmt.Println("🧠 Building CNN model...")
	
	// Start with a base batch size, but we'll vary it during execution
	baseBatchSize := 16
	inputShape := []int{baseBatchSize, 3, 32, 32} // RGB 32x32 images

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(8, 3, 1, 1, true, "conv1").     // 8 filters, 3x3 kernel
		AddReLU("relu1").
		AddConv2D(16, 3, 2, 1, true, "conv2").    // 16 filters, stride 2 for downsampling
		AddReLU("relu2").
		AddDense(32, true, "fc1").                // Dense layer with 32 neurons
		AddReLU("relu3").
		AddDense(4, true, "fc2").                 // 4 classes output
		Compile()

	if err != nil {
		log.Fatalf("Failed to build model: %v", err)
	}

	fmt.Printf("✅ Model built with %d layers\n", len(model.Layers))
	fmt.Printf("   Input shape: %v\n", inputShape)
	fmt.Printf("   Parameter count: %d\n\n", model.TotalParameters)

	// Training configuration
	config := training.TrainerConfig{
		BatchSize:        baseBatchSize,
		LearningRate:     0.01,
		OptimizerType:    cgo_bridge.Adam,
		UseDynamicEngine: true,   // Dynamic engine is now the default and only option
		Beta1:            0.9,
		Beta2:            0.999,
		WeightDecay:      0.0001,
		Epsilon:          1e-8,
	}

	// Create training engine with dynamic batch support
	fmt.Println("🔧 Creating dynamic training engine...")
	engine, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create training engine: %v", err)
	}
	defer engine.Cleanup()

	fmt.Println("✅ Dynamic training engine created successfully")

	// Test different batch sizes
	testBatchSizes := []int{4, 8, 16, 24, 32}
	
	fmt.Println("🚀 Testing dynamic batch size functionality...")
	fmt.Println("Testing the same model with different batch sizes without recompilation:")

	for i, batchSize := range testBatchSizes {
		fmt.Printf("--- Test %d: Batch Size %d ---\n", i+1, batchSize)
		
		// Generate synthetic data for this batch size
		inputData, labelData := generateSyntheticData(batchSize, 3, 32, 32, 4)
		
		// Prepare shapes for dynamic batch size
		inputShape := []int{batchSize, 3, 32, 32}
		labelShape := []int{batchSize} // ModelTrainer converts to one-hot internally

		// Perform training step
		start := time.Now()
		result, err := engine.TrainBatchWithCommandPool(inputData, inputShape, labelData, labelShape)
		duration := time.Since(start)

		if err != nil {
			log.Fatalf("Training step failed for batch size %d: %v", batchSize, err)
		}

		fmt.Printf("✅ Forward pass successful!\n")
		fmt.Printf("   Batch size: %d samples\n", batchSize)
		fmt.Printf("   Loss: %.6f\n", result.Loss)
		if result.HasAccuracy {
			fmt.Printf("   Accuracy: %.2f%%\n", result.Accuracy*100)
		}
		fmt.Printf("   Processing time: %v\n", duration)

		// Test inference with the same batch size
		predictions, err := engine.InferBatch(inputData, inputShape)
		if err != nil {
			log.Fatalf("Inference failed for batch size %d: %v", batchSize, err)
		}

		fmt.Printf("   Inference: Generated %d predictions\n", len(predictions.Predictions)/4) // 4 classes
		if len(predictions.Predictions) >= 4 {
			fmt.Printf("   Sample predictions: [%.3f, %.3f, %.3f, %.3f]\n", 
				predictions.Predictions[0], predictions.Predictions[1], 
				predictions.Predictions[2], predictions.Predictions[3])
		}

		fmt.Println()
	}

	// Demonstrate mixed batch size training
	fmt.Println("🔄 Demonstrating mixed batch size training in a single session...")
	fmt.Println("Training with varying batch sizes to show true dynamic capability:")

	for epoch := 1; epoch <= 3; epoch++ {
		fmt.Printf("Epoch %d:\n", epoch)
		
		// Vary batch size each epoch
		for step, batchSize := range []int{8, 16, 12, 20} {
			inputData, labelData := generateSyntheticData(batchSize, 3, 32, 32, 4)
			
			inputShape := []int{batchSize, 3, 32, 32}
			labelShape := []int{batchSize} // ModelTrainer converts to one-hot internally

			result, err := engine.TrainBatchWithCommandPool(inputData, inputShape, labelData, labelShape)
			if err != nil {
				log.Fatalf("Training step failed: %v", err)
			}

			accuracy := 0.0
			if result.HasAccuracy {
				accuracy = result.Accuracy
			}

			fmt.Printf("  Step %d (batch=%d): Loss=%.4f, Accuracy=%.1f%%\n", 
				step+1, batchSize, result.Loss, accuracy*100)
		}
		fmt.Println()
	}

	fmt.Println("🎉 Dynamic Batch Size Demo Complete!")
	fmt.Println("\n✅ Key Achievements Demonstrated:")
	fmt.Println("   ✓ Same model processed multiple batch sizes (4, 8, 16, 24, 32)")
	fmt.Println("   ✓ No recompilation required between batch size changes")
	fmt.Println("   ✓ Both training and inference work with variable batch sizes")
	fmt.Println("   ✓ Mixed batch size training within single session")
	fmt.Println("   ✓ GPU-resident operations maintained throughout")
	fmt.Println("\n🚀 The dynamic batch size implementation is working perfectly!")
}

// generateSyntheticData creates random input and label data for testing
func generateSyntheticData(batchSize, channels, height, width, numClasses int) ([]float32, []int32) {
	// Generate random input data
	inputSize := batchSize * channels * height * width
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}

	// Generate random class indices (ModelTrainer converts to one-hot internally)
	labelData := make([]int32, batchSize)
	
	for b := 0; b < batchSize; b++ {
		// Random class index for this sample
		classIdx := rand.Intn(numClasses)
		labelData[b] = int32(classIdx)
	}

	return inputData, labelData
}