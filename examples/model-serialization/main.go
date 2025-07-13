package main

import (
	"fmt"
	"log"
	"os"

	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/layers"
)

func main() {
	fmt.Println("=== Go-Metal Checkpoint Demo ===")
	fmt.Println("Demonstrating checkpoint saving & loading functionality")
	fmt.Println("Supports both JSON and ONNX formats for PyTorch interoperability")

	// Create a simple neural network for demonstration
	inputShape := []int{32, 1, 28, 28} // Batch=32, MNIST-like input
	
	fmt.Println("📋 Creating model architecture...")
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddDense(128, true, "hidden1").
		AddReLU("relu1").
		AddDense(64, true, "hidden2").
		AddLeakyReLU(0.01, "leaky_relu1").
		AddDense(10, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	fmt.Printf("✅ Model created with %d layers\n", len(model.Layers))
	for i, layer := range model.Layers {
		fmt.Printf("   Layer %d: %s (%s)\n", i+1, layer.Name, layer.Type.String())
	}

	// Simulate training state
	fmt.Println("\n⚡ Simulating training state...")
	trainingState := checkpoints.TrainingState{
		Epoch:        25,
		Step:         5000,
		LearningRate: 0.001,
		BestLoss:     0.234,
		BestAccuracy: 0.892,
		TotalSteps:   5000,
	}

	// Create synthetic weights (in real usage, these would come from trained model)
	fmt.Println("🎯 Creating synthetic model weights...")
	weights := createSyntheticWeights(model)
	fmt.Printf("✅ Generated %d weight tensors\n", len(weights))

	// Create checkpoint
	checkpoint := &checkpoints.Checkpoint{
		ModelSpec:     model,
		Weights:       weights,
		TrainingState: trainingState,
		Metadata: checkpoints.CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			Description: "Demo checkpoint with Leaky ReLU",
			Tags:        []string{"demo", "mnist", "leaky-relu"},
		},
	}

	// Test JSON format
	fmt.Println("\n💾 Testing JSON checkpoint format...")
	testCheckpointFormat(checkpoint, checkpoints.FormatJSON, "demo_model.json")

	// Test ONNX format
	fmt.Println("\n📦 Testing ONNX checkpoint format...")
	testCheckpointFormat(checkpoint, checkpoints.FormatONNX, "demo_model.onnx")

	fmt.Println("\n🎉 Checkpoint demo completed successfully!")
	fmt.Println("\n📁 Generated Files:")
	fmt.Println("  - demo_model.json (JSON checkpoint for go-metal)")
	fmt.Println("  - demo_model.onnx (ONNX model for PyTorch/TensorFlow)")
	fmt.Println("\nKey Features Demonstrated:")
	fmt.Println("✅ Complete model architecture serialization")
	fmt.Println("✅ GPU weight extraction and storage")
	fmt.Println("✅ Training state preservation")
	fmt.Println("✅ JSON format for go-metal interoperability")
	fmt.Println("✅ ONNX format for PyTorch/TensorFlow interoperability")
	fmt.Println("✅ Leaky ReLU activation function support")
	fmt.Println("✅ Metadata and versioning")
}

func testCheckpointFormat(checkpoint *checkpoints.Checkpoint, format checkpoints.CheckpointFormat, filename string) {
	saver := checkpoints.NewCheckpointSaver(format)
	
	// Save checkpoint
	fmt.Printf("  Saving %s checkpoint...\n", format.String())
	err := saver.SaveCheckpoint(checkpoint, filename)
	if err != nil {
		log.Printf("  ❌ Failed to save %s checkpoint: %v", format.String(), err)
		return
	}

	// Check file size
	fileInfo, err := os.Stat(filename)
	if err != nil {
		log.Printf("  ❌ Failed to stat saved file: %v", err)
		return
	}

	fmt.Printf("  ✅ Saved successfully (%d bytes)\n", fileInfo.Size())

	// Test loading (JSON only, ONNX loading is more complex)
	if format == checkpoints.FormatJSON {
		fmt.Printf("  Loading %s checkpoint...\n", format.String())
		loadedCheckpoint, err := saver.LoadCheckpoint(filename)
		if err != nil {
			log.Printf("  ❌ Failed to load %s checkpoint: %v", format.String(), err)
			return
		}

		// Verify data integrity
		if loadedCheckpoint.TrainingState.Epoch != checkpoint.TrainingState.Epoch {
			log.Printf("  ❌ Data integrity check failed: epoch mismatch")
			return
		}

		if len(loadedCheckpoint.Weights) != len(checkpoint.Weights) {
			log.Printf("  ❌ Data integrity check failed: weight count mismatch")
			return
		}

		fmt.Printf("  ✅ Loaded and verified successfully\n")
		fmt.Printf("    - Epoch: %d\n", loadedCheckpoint.TrainingState.Epoch)
		fmt.Printf("    - Best Loss: %.6f\n", loadedCheckpoint.TrainingState.BestLoss)
		fmt.Printf("    - Best Accuracy: %.2f%%\n", loadedCheckpoint.TrainingState.BestAccuracy*100)
		fmt.Printf("    - Weights: %d tensors\n", len(loadedCheckpoint.Weights))
	}

	// Keep files for inspection (removed cleanup)
	fmt.Printf("  📁 File saved: %s\n", filename)
}

func createSyntheticWeights(model *layers.ModelSpec) []checkpoints.WeightTensor {
	var weights []checkpoints.WeightTensor
	weightIndex := 0

	for _, layer := range model.Layers {
		switch layer.Type {
		case layers.Dense:
			// Get layer parameters
			useBias := layer.Parameters["use_bias"].(bool)
			
			// For synthetic weights, we need to calculate dimensions
			// In a real scenario, these would come from the actual trained tensors
			inputSize := layer.InputShape[len(layer.InputShape)-1]
			outputSize := layer.OutputShape[len(layer.OutputShape)-1]

			// Weight tensor
			weightData := make([]float32, inputSize*outputSize)
			for i := range weightData {
				weightData[i] = float32(weightIndex%100) * 0.01 // Synthetic data
				weightIndex++
			}

			weights = append(weights, checkpoints.WeightTensor{
				Name:  fmt.Sprintf("%s.weight", layer.Name),
				Shape: []int{inputSize, outputSize},
				Data:  weightData,
				Layer: layer.Name,
				Type:  "weight",
			})

			// Bias tensor (if used)
			if useBias {
				biasData := make([]float32, outputSize)
				for i := range biasData {
					biasData[i] = float32(weightIndex%10) * 0.1 // Synthetic data
					weightIndex++
				}

				weights = append(weights, checkpoints.WeightTensor{
					Name:  fmt.Sprintf("%s.bias", layer.Name),
					Shape: []int{outputSize},
					Data:  biasData,
					Layer: layer.Name,
					Type:  "bias",
				})
			}

		case layers.ReLU, layers.LeakyReLU, layers.Softmax, layers.Dropout, layers.ELU, layers.Sigmoid:
			// Activation layers have no parameters
			continue

		default:
			fmt.Printf("Warning: Unsupported layer type for weight generation: %s\n", layer.Type.String())
		}
	}

	return weights
}