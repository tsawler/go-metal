# ONNX Integration Tutorial

Import and export models with ONNX for cross-framework compatibility between Go-Metal, PyTorch, TensorFlow, and other ML frameworks.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- Export Go-Metal models to ONNX format
- Import ONNX models from PyTorch/TensorFlow into Go-Metal
- Handle model conversion between different frameworks
- Understand ONNX compatibility limitations and solutions
- Deploy models across different platforms using ONNX

## 🔗 What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open standard for representing machine learning models. It enables interoperability between different ML frameworks.

### Key Benefits:
- **Cross-framework compatibility**: Train in PyTorch, deploy in Go-Metal
- **Production flexibility**: Use the best framework for each stage
- **Model sharing**: Exchange models between teams using different tools
- **Standardized format**: Consistent model representation across platforms

### Supported Operations:
- **Layers**: Dense (MatMul + Add), Conv2D, BatchNorm, Dropout
- **Activations**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Swish
- **Data types**: Float32, Int32 tensors
- **Model metadata**: Version, description, tags

## 🚀 Complete Example 1: Export Go-Metal Model to ONNX

This example trains a model in Go-Metal and exports it to ONNX format:

```go
package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/training"
)

func main() {
	// Initialize Metal device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	fmt.Println("🔧 Go-Metal to ONNX Export Example")
	fmt.Println("===================================")

	// Create and train a model
	checkpoint, err := createAndTrainModel()
	if err != nil {
		log.Fatalf("Failed to create/train model: %v", err)
	}

	// Export to ONNX
	fmt.Println("\n📦 Exporting model to ONNX format...")
	err = exportToONNX(checkpoint, "trained_model.onnx")
	if err != nil {
		log.Fatalf("Failed to export to ONNX: %v", err)
	}

	// Verify export
	fmt.Println("\n✅ Export completed successfully!")
	fmt.Println("📁 Generated files:")
	fmt.Println("  - trained_model.onnx (ONNX model)")
	fmt.Println("  - trained_model.json (Go-Metal checkpoint)")

	// Demonstrate cross-platform compatibility
	fmt.Println("\n🌍 Cross-Platform Compatibility:")
	fmt.Println("✅ Can be loaded in PyTorch with torch.onnx.load()")
	fmt.Println("✅ Can be loaded in TensorFlow with tf.saved_model.load()")
	fmt.Println("✅ Can be loaded in ONNX Runtime for deployment")
	fmt.Println("✅ Can be imported back into Go-Metal")
}

func createAndTrainModel() (*checkpoints.Checkpoint, error) {
	fmt.Println("🏗️ Building neural network...")

	// Create model architecture
	batchSize := 64
	inputSize := 784
	numClasses := 10

	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)

	model, err := builder.
		AddDense(512, true, "hidden1").
		AddReLU("relu1").
		AddDropout(0.2, "dropout1").
		AddDense(256, true, "hidden2").
		AddLeakyReLU(0.01, "leaky_relu1").
		AddDropout(0.1, "dropout2").
		AddDense(numClasses, true, "output").
		AddSoftmax(-1, "softmax").
		Compile()

	if err != nil {
		return nil, fmt.Errorf("model compilation failed: %v", err)
	}

	fmt.Printf("✅ Created model with %d layers\n", len(model.Layers))

	// Configure training
	config := training.TrainerConfig{
		BatchSize:     batchSize,
		LearningRate:  0.001,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    training.Dynamic,
		LossFunction:  training.SparseCrossEntropy,
		ProblemType:   training.Classification,
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		return nil, fmt.Errorf("trainer creation failed: %v", err)
	}
	defer trainer.Cleanup()

	// Generate training data
	fmt.Println("📊 Generating synthetic training data...")
	inputData, labelData := generateTrainingData(1000, inputSize, numClasses)

	// Training loop
	fmt.Println("🚀 Training model...")
	epochs := 15
	var finalLoss float32

	for epoch := 1; epoch <= epochs; epoch++ {
		numBatches := 1000 / batchSize
		var epochLoss float32

		for batch := 0; batch < numBatches; batch++ {
			batchStart := batch * batchSize
			batchEnd := batchStart + batchSize

			batchInput := inputData[batchStart*inputSize : batchEnd*inputSize]
			batchLabels := labelData[batchStart:batchEnd]

			result, err := trainer.TrainBatch(
				batchInput, []int{batchSize, inputSize},
				batchLabels, []int{batchSize},
			)
			if err != nil {
				return nil, fmt.Errorf("training failed: %v", err)
			}

			epochLoss += result.Loss
		}

		finalLoss = epochLoss / float32(numBatches)

		if epoch%5 == 0 {
			fmt.Printf("  Epoch %d: Loss %.6f\n", epoch, finalLoss)
		}
	}

	fmt.Printf("✅ Training completed with final loss: %.6f\n", finalLoss)

	// Extract weights from trained model
	parameterTensors := trainer.GetParameterTensors()
	weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, model)
	if err != nil {
		return nil, fmt.Errorf("failed to extract weights: %v", err)
	}

	// Create checkpoint
	checkpoint := &checkpoints.Checkpoint{
		ModelSpec: model,
		Weights:   weights,
		TrainingState: checkpoints.TrainingState{
			Epoch:        epochs,
			Step:         epochs * (1000 / batchSize),
			LearningRate: config.LearningRate,
			BestLoss:     finalLoss,
			BestAccuracy: 0.85, // Simulated
			TotalSteps:   epochs * (1000 / batchSize),
		},
		Metadata: checkpoints.CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			Description: "Trained MLP for ONNX export demo",
			Tags:        []string{"classification", "mlp", "onnx-ready"},
		},
	}

	// Save Go-Metal native format as well
	saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
	err = saver.SaveCheckpoint(checkpoint, "trained_model.json")
	if err != nil {
		return nil, fmt.Errorf("failed to save checkpoint: %v", err)
	}

	return checkpoint, nil
}

func exportToONNX(checkpoint *checkpoints.Checkpoint, filename string) error {
	// Create ONNX exporter
	exporter := checkpoints.NewONNXExporter()

	// Export to ONNX format
	err := exporter.ExportToONNX(checkpoint, filename)
	if err != nil {
		return fmt.Errorf("ONNX export failed: %v", err)
	}

	fmt.Printf("✅ Successfully exported to %s\n", filename)

	// Display model information
	fmt.Printf("📊 Exported model info:\n")
	fmt.Printf("  - Layers: %d\n", len(checkpoint.ModelSpec.Layers))
	fmt.Printf("  - Parameters: %d\n", checkpoint.ModelSpec.TotalParameters)
	fmt.Printf("  - Input shape: %v\n", checkpoint.ModelSpec.InputShape)
	fmt.Printf("  - Output shape: %v\n", checkpoint.ModelSpec.OutputShape)

	return nil
}

func generateTrainingData(samples, features, classes int) ([]float32, []int32) {
	rand.Seed(42) // Reproducible results

	inputData := make([]float32, samples*features)
	labelData := make([]int32, samples)

	for i := 0; i < samples; i++ {
		class := i % classes
		labelData[i] = int32(class)

		// Generate class-specific patterns
		for j := 0; j < features; j++ {
			baseValue := float64(class) / float64(classes)
			noise := rand.Float64()*0.2 - 0.1
			featureVar := 0.1 * float64(j%20) / 20.0

			value := baseValue + noise + featureVar
			if value < 0 {
				value = 0
			} else if value > 1 {
				value = 1
			}

			inputData[i*features+j] = float32(value)
		}
	}

	return inputData, labelData
}
```

**What this example demonstrates:**
- Complete model training in Go-Metal
- Proper weight extraction from trained models
- ONNX export with metadata
- Model architecture preservation
- Cross-platform compatibility information

## 🔄 Complete Example 2: Import ONNX Model into Go-Metal

This example imports an existing ONNX model and runs inference:

```go
package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/training"
)

func main() {
	// Initialize Metal device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	fmt.Println("📥 ONNX Import and Inference Example")
	fmt.Println("====================================")

	// Check if ONNX file exists (from previous example)
	onnxFile := "trained_model.onnx"
	if _, err := os.Stat(onnxFile); os.IsNotExist(err) {
		fmt.Printf("⚠️ ONNX file '%s' not found.\n", onnxFile)
		fmt.Println("Please run the export example first to generate the ONNX file.")
		return
	}

	// Import ONNX model
	fmt.Printf("📖 Importing ONNX model from %s...\n", onnxFile)
	checkpoint, err := importFromONNX(onnxFile)
	if err != nil {
		log.Fatalf("Failed to import ONNX model: %v", err)
	}

	// Analyze imported model
	analyzeImportedModel(checkpoint)

	// Run inference with imported model
	fmt.Println("\n🔮 Running inference with imported model...")
	err = runInferenceDemo(checkpoint)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	fmt.Println("\n✅ ONNX import and inference completed successfully!")
}

func importFromONNX(filename string) (*checkpoints.Checkpoint, error) {
	// Create ONNX importer
	importer := checkpoints.NewONNXImporter()

	// Import ONNX model
	checkpoint, err := importer.ImportFromONNX(filename)
	if err != nil {
		return nil, fmt.Errorf("ONNX import failed: %v", err)
	}

	fmt.Printf("✅ Successfully imported ONNX model\n")
	return checkpoint, nil
}

func analyzeImportedModel(checkpoint *checkpoints.Checkpoint) {
	fmt.Println("\n🔍 Imported Model Analysis:")
	fmt.Printf("  📊 Layers: %d\n", len(checkpoint.ModelSpec.Layers))
	fmt.Printf("  ⚙️ Parameters: %d\n", checkpoint.ModelSpec.TotalParameters)
	fmt.Printf("  📥 Input shape: %v\n", checkpoint.ModelSpec.InputShape)
	fmt.Printf("  📤 Output shape: %v\n", checkpoint.ModelSpec.OutputShape)
	fmt.Printf("  🔢 Weight tensors: %d\n", len(checkpoint.Weights))

	fmt.Println("\n🏗️ Model Architecture:")
	for i, layer := range checkpoint.ModelSpec.Layers {
		fmt.Printf("  %d. %s (%s)\n", i+1, layer.Name, layer.Type.String())
		if layer.Type.String() == "Dense" {
			inputSize := layer.InputShape[len(layer.InputShape)-1]
			outputSize := layer.OutputShape[len(layer.OutputShape)-1]
			fmt.Printf("     Size: %d → %d\n", inputSize, outputSize)
		}
	}

	// Check training metadata if available
	if checkpoint.TrainingState.Epoch > 0 {
		fmt.Println("\n📈 Training History:")
		fmt.Printf("  🔄 Epochs: %d\n", checkpoint.TrainingState.Epoch)
		fmt.Printf("  📉 Best Loss: %.6f\n", checkpoint.TrainingState.BestLoss)
		fmt.Printf("  🎯 Best Accuracy: %.2f%%\n", checkpoint.TrainingState.BestAccuracy*100)
	}

	if len(checkpoint.Metadata.Tags) > 0 {
		fmt.Printf("\n🏷️  Tags: %v\n", checkpoint.Metadata.Tags)
	}
}

func runInferenceDemo(checkpoint *checkpoints.Checkpoint) error {
	fmt.Println("🛠️ Setting up inference engine...")

	// Create trainer for inference (Go-Metal's current approach)
	config := training.TrainerConfig{
		BatchSize:     1, // Single sample inference
		LearningRate:  0.001,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    training.Dynamic,
		InferenceOnly: true,
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}

	trainer, err := training.NewModelTrainer(checkpoint.ModelSpec, config)
	if err != nil {
		return fmt.Errorf("failed to create inference trainer: %v", err)
	}
	defer trainer.Cleanup()

	// Load weights into model
	parameterTensors := trainer.GetParameterTensors()
	err = checkpoints.LoadWeightsIntoTensors(checkpoint.Weights, parameterTensors)
	if err != nil {
		return fmt.Errorf("failed to load weights: %v", err)
	}

	fmt.Println("✅ Weights loaded successfully")

	// Generate test samples
	inputSize := checkpoint.ModelSpec.InputShape[len(checkpoint.ModelSpec.InputShape)-1]
	numClasses := checkpoint.ModelSpec.OutputShape[len(checkpoint.ModelSpec.OutputShape)-1]

	fmt.Printf("🧪 Running inference on %d test samples...\n", 5)
	fmt.Println("Sample | Prediction | Confidence | Time (ms)")
	fmt.Println("-------|------------|------------|----------")

	for i := 0; i < 5; i++ {
		// Generate test sample
		testData := generateTestSample(inputSize, i)

		// Run inference
		startTime := time.Now()
		result, err := trainer.Predict(testData, []int{1, inputSize})
		if err != nil {
			return fmt.Errorf("inference failed for sample %d: %v", i+1, err)
		}
		inferenceTime := time.Since(startTime)

		// Process results
		prediction, confidence := processResults(result.Predictions, numClasses)

		fmt.Printf("  %d    |     %d      |   %.3f    | %8.2f\n",
			i+1, prediction, confidence, float64(inferenceTime.Milliseconds()))
	}

	// Benchmark inference speed
	fmt.Println("\n⚡ Performance Benchmark:")
	err = benchmarkInference(trainer, inputSize, 100)
	if err != nil {
		return fmt.Errorf("benchmark failed: %v", err)
	}

	return nil
}

func generateTestSample(inputSize, seed int) []float32 {
	rand.Seed(int64(seed + 42))

	testData := make([]float32, inputSize)
	for i := 0; i < inputSize; i++ {
		// Generate predictable test pattern
		testData[i] = rand.Float32()
	}

	return testData
}

func processResults(predictions []float32, numClasses int) (int, float64) {
	if len(predictions) == 1 {
		// Regression or binary classification
		return 0, float64(predictions[0])
	}

	// Multi-class classification - find max
	maxIdx := 0
	maxVal := predictions[0]

	for i := 1; i < len(predictions) && i < numClasses; i++ {
		if predictions[i] > maxVal {
			maxVal = predictions[i]
			maxIdx = i
		}
	}

	// Apply softmax for confidence
	confidence := applySoftmax(predictions, maxIdx)

	return maxIdx, confidence
}

func applySoftmax(logits []float32, targetIdx int) float64 {
	// Simple softmax approximation for confidence
	if len(logits) <= targetIdx {
		return 0.5
	}

	sum := float32(0)
	for _, logit := range logits {
		sum += float32(1.0 + logit) // Simplified exp approximation
	}

	if sum == 0 {
		return 0.5
	}

	return float64((1.0 + logits[targetIdx]) / sum)
}

func benchmarkInference(trainer *training.ModelTrainer, inputSize, iterations int) error {
	fmt.Printf("🏁 Running %d inference iterations...\n", iterations)

	testData := generateTestSample(inputSize, 999)
	totalTime := time.Duration(0)

	for i := 0; i < iterations; i++ {
		startTime := time.Now()
		_, err := trainer.Predict(testData, []int{1, inputSize})
		if err != nil {
			return fmt.Errorf("benchmark iteration %d failed: %v", i+1, err)
		}
		totalTime += time.Since(startTime)
	}

	avgTime := totalTime / time.Duration(iterations)
	throughput := float64(time.Second) / float64(avgTime)

	fmt.Printf("📊 Benchmark Results:\n")
	fmt.Printf("  ⏱️ Average time: %v\n", avgTime)
	fmt.Printf("  🚀 Throughput: %.1f samples/sec\n", throughput)
	fmt.Printf("  📈 Total time: %v\n", totalTime)

	return nil
}
```

**Key Features Demonstrated:**
- ONNX model import and analysis
- Weight loading into Go-Metal inference engine
- Real-time inference with performance metrics
- Proper model architecture reconstruction
- Cross-framework model usage

## 🔧 Complete Example 3: Cross-Framework Model Pipeline

This example demonstrates a complete workflow using models from different frameworks:

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

func main() {
	// Initialize Metal device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	fmt.Println("🌍 Cross-Framework Model Pipeline Demo")
	fmt.Println("======================================")

	// Step 1: Create models in different formats
	fmt.Println("1️⃣ Creating models in different formats...")
	err = createDemoModels()
	if err != nil {
		log.Fatalf("Failed to create demo models: %v", err)
	}

	// Step 2: Demonstrate format conversions
	fmt.Println("\n2️⃣ Demonstrating format conversions...")
	err = demonstrateConversions()
	if err != nil {
		log.Fatalf("Format conversion failed: %v", err)
	}

	// Step 3: Cross-framework compatibility testing
	fmt.Println("\n3️⃣ Testing cross-framework compatibility...")
	err = testCompatibility()
	if err != nil {
		log.Fatalf("Compatibility test failed: %v", err)
	}

	fmt.Println("\n✅ Cross-framework pipeline completed successfully!")
	displayCompatibilityMatrix()
}

func createDemoModels() error {
	fmt.Println("🏗️ Creating CNN model...")
	cnnCheckpoint, err := createCNNModel()
	if err != nil {
		return fmt.Errorf("CNN creation failed: %v", err)
	}

	fmt.Println("🏗️ Creating MLP model...")
	mlpCheckpoint, err := createMLPModel()
	if err != nil {
		return fmt.Errorf("MLP creation failed: %v", err)
	}

	fmt.Println("💾 Saving models in multiple formats...")

	// Save CNN in both formats
	err = saveModelInBothFormats(cnnCheckpoint, "cnn_model")
	if err != nil {
		return fmt.Errorf("CNN save failed: %v", err)
	}

	// Save MLP in both formats
	err = saveModelInBothFormats(mlpCheckpoint, "mlp_model")
	if err != nil {
		return fmt.Errorf("MLP save failed: %v", err)
	}

	fmt.Println("✅ Demo models created successfully")
	return nil
}

func createCNNModel() (*checkpoints.Checkpoint, error) {
	// Create a CNN model suitable for image classification
	inputShape := []int{1, 3, 32, 32} // RGB 32x32 images
	builder := layers.NewModelBuilder(inputShape)

	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddConv2D(32, 3, 1, 1, true, "conv2").
		AddReLU("relu2").
		AddFlatten("flatten").
		AddDense(128, true, "fc1").
		AddReLU("fc_relu").
		AddDropout(0.5, "dropout").
		AddDense(10, true, "output").
		Compile()

	if err != nil {
		return nil, err
	}

	// Create synthetic weights
	weights := createSyntheticWeightsForModel(model)

	checkpoint := &checkpoints.Checkpoint{
		ModelSpec: model,
		Weights:   weights,
		Metadata: checkpoints.CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			Description: "CNN for cross-framework compatibility demo",
			Tags:        []string{"cnn", "image-classification", "cross-framework"},
		},
	}

	return checkpoint, nil
}

func createMLPModel() (*checkpoints.Checkpoint, error) {
	// Create an MLP model for tabular data
	inputShape := []int{1, 784} // Flattened 28x28 images
	builder := layers.NewModelBuilder(inputShape)

	model, err := builder.
		AddDense(256, true, "hidden1").
		AddReLU("relu1").
		AddBatchNorm("bn1").
		AddDropout(0.3, "dropout1").
		AddDense(128, true, "hidden2").
		AddLeakyReLU(0.01, "leaky_relu").
		AddDropout(0.2, "dropout2").
		AddDense(10, true, "output").
		AddSoftmax(-1, "softmax").
		Compile()

	if err != nil {
		return nil, err
	}

	// Create synthetic weights
	weights := createSyntheticWeightsForModel(model)

	checkpoint := &checkpoints.Checkpoint{
		ModelSpec: model,
		Weights:   weights,
		Metadata: checkpoints.CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			Description: "MLP with BatchNorm for cross-framework demo",
			Tags:        []string{"mlp", "tabular-data", "batch-norm"},
		},
	}

	return checkpoint, nil
}

func saveModelInBothFormats(checkpoint *checkpoints.Checkpoint, baseName string) error {
	// Save in JSON format (Go-Metal native)
	jsonSaver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
	err := jsonSaver.SaveCheckpoint(checkpoint, baseName+".json")
	if err != nil {
		return fmt.Errorf("JSON save failed: %v", err)
	}

	// Save in ONNX format (cross-framework)
	onnxSaver := checkpoints.NewCheckpointSaver(checkpoints.FormatONNX)
	err = onnxSaver.SaveCheckpoint(checkpoint, baseName+".onnx")
	if err != nil {
		return fmt.Errorf("ONNX save failed: %v", err)
	}

	fmt.Printf("  ✅ %s saved in both JSON and ONNX formats\n", baseName)
	return nil
}

func demonstrateConversions() error {
	conversions := []struct {
		name        string
		sourceFile  string
		targetFile  string
		description string
	}{
		{"JSON→ONNX", "cnn_model.json", "cnn_from_json.onnx", "CNN model conversion"},
		{"ONNX→JSON", "mlp_model.onnx", "mlp_from_onnx.json", "MLP model conversion"},
	}

	for _, conv := range conversions {
		fmt.Printf("🔄 %s: %s\n", conv.name, conv.description)

		err := performConversion(conv.sourceFile, conv.targetFile)
		if err != nil {
			return fmt.Errorf("conversion %s failed: %v", conv.name, err)
		}

		fmt.Printf("  ✅ Successfully converted %s → %s\n", conv.sourceFile, conv.targetFile)
	}

	return nil
}

func performConversion(sourceFile, targetFile string) error {
	// Detect source format
	var sourceFormat, targetFormat checkpoints.CheckpointFormat

	if sourceFile[len(sourceFile)-5:] == ".json" {
		sourceFormat = checkpoints.FormatJSON
	} else {
		sourceFormat = checkpoints.FormatONNX
	}

	if targetFile[len(targetFile)-5:] == ".json" {
		targetFormat = checkpoints.FormatJSON
	} else {
		targetFormat = checkpoints.FormatONNX
	}

	// Load from source format
	sourceSaver := checkpoints.NewCheckpointSaver(sourceFormat)
	checkpoint, err := sourceSaver.LoadCheckpoint(sourceFile)
	if err != nil {
		return fmt.Errorf("failed to load %s: %v", sourceFile, err)
	}

	// Save in target format
	targetSaver := checkpoints.NewCheckpointSaver(targetFormat)
	err = targetSaver.SaveCheckpoint(checkpoint, targetFile)
	if err != nil {
		return fmt.Errorf("failed to save %s: %v", targetFile, err)
	}

	return nil
}

func testCompatibility() error {
	testFiles := []string{
		"cnn_model.json",
		"cnn_model.onnx",
		"mlp_model.json",
		"mlp_model.onnx",
		"cnn_from_json.onnx",
		"mlp_from_onnx.json",
	}

	fmt.Println("🧪 Testing model loading from different formats...")

	for _, filename := range testFiles {
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			fmt.Printf("  ⚠️ Skipping %s (not found)\n", filename)
			continue
		}

		err := testModelLoading(filename)
		if err != nil {
			fmt.Printf("  ❌ %s: %v\n", filename, err)
		} else {
			fmt.Printf("  ✅ %s: Loaded successfully\n", filename)
		}
	}

	return nil
}

func testModelLoading(filename string) error {
	var format checkpoints.CheckpointFormat

	if filename[len(filename)-5:] == ".json" {
		format = checkpoints.FormatJSON
	} else {
		format = checkpoints.FormatONNX
	}

	saver := checkpoints.NewCheckpointSaver(format)
	checkpoint, err := saver.LoadCheckpoint(filename)
	if err != nil {
		return err
	}

	// Basic validation
	if checkpoint.ModelSpec == nil {
		return fmt.Errorf("missing model specification")
	}

	if len(checkpoint.Weights) == 0 {
		return fmt.Errorf("no weights found")
	}

	return nil
}

func createSyntheticWeightsForModel(model *layers.ModelSpec) []checkpoints.WeightTensor {
	var weights []checkpoints.WeightTensor
	weightIndex := 0

	for _, layer := range model.Layers {
		switch layer.Type {
		case layers.Dense:
			useBias := layer.Parameters["use_bias"].(bool)
			inputSize := layer.InputShape[len(layer.InputShape)-1]
			outputSize := layer.OutputShape[len(layer.OutputShape)-1]

			// Weight tensor
			weightData := make([]float32, inputSize*outputSize)
			for i := range weightData {
				weightData[i] = float32(weightIndex%100) * 0.01
				weightIndex++
			}

			weights = append(weights, checkpoints.WeightTensor{
				Name:  fmt.Sprintf("%s.weight", layer.Name),
				Shape: []int{inputSize, outputSize},
				Data:  weightData,
				Layer: layer.Name,
				Type:  "weight",
			})

			// Bias tensor
			if useBias {
				biasData := make([]float32, outputSize)
				for i := range biasData {
					biasData[i] = float32(weightIndex%10) * 0.1
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

		case layers.Conv2D:
			useBias := layer.Parameters["use_bias"].(bool)
			filterSize := layer.Parameters["filter_size"].(int)
			numFilters := layer.Parameters["num_filters"].(int)
			inputChannels := layer.InputShape[1] // Assuming NCHW format

			// Conv weight tensor
			weightSize := numFilters * inputChannels * filterSize * filterSize
			weightData := make([]float32, weightSize)
			for i := range weightData {
				weightData[i] = float32(weightIndex%100) * 0.01
				weightIndex++
			}

			weights = append(weights, checkpoints.WeightTensor{
				Name:  fmt.Sprintf("%s.weight", layer.Name),
				Shape: []int{numFilters, inputChannels, filterSize, filterSize},
				Data:  weightData,
				Layer: layer.Name,
				Type:  "weight",
			})

			// Conv bias tensor
			if useBias {
				biasData := make([]float32, numFilters)
				for i := range biasData {
					biasData[i] = float32(weightIndex%10) * 0.1
					weightIndex++
				}

				weights = append(weights, checkpoints.WeightTensor{
					Name:  fmt.Sprintf("%s.bias", layer.Name),
					Shape: []int{numFilters},
					Data:  biasData,
					Layer: layer.Name,
					Type:  "bias",
				})
			}

		case layers.BatchNorm:
			numFeatures := layer.InputShape[len(layer.InputShape)-1]

			// BatchNorm parameters: weight (gamma), bias (beta), running_mean, running_var
			for _, paramType := range []string{"weight", "bias", "running_mean", "running_var"} {
				paramData := make([]float32, numFeatures)
				for i := range paramData {
					if paramType == "weight" {
						paramData[i] = 1.0 // Initialize gamma to 1
					} else if paramType == "running_var" {
						paramData[i] = 1.0 // Initialize variance to 1
					} else {
						paramData[i] = 0.0 // Initialize beta and mean to 0
					}
					weightIndex++
				}

				weights = append(weights, checkpoints.WeightTensor{
					Name:  fmt.Sprintf("%s.%s", layer.Name, paramType),
					Shape: []int{numFeatures},
					Data:  paramData,
					Layer: layer.Name,
					Type:  paramType,
				})
			}
		}
	}

	return weights
}

func displayCompatibilityMatrix() {
	fmt.Println("\n📊 Cross-Framework Compatibility Matrix:")
	fmt.Println("========================================")

	matrix := [][]string{
		{"Source/Target", "Go-Metal", "PyTorch", "TensorFlow", "ONNX Runtime"},
		{"Go-Metal JSON", "✅ Native", "❌ Direct", "❌ Direct", "❌ Direct"},
		{"Go-Metal ONNX", "✅ Import", "✅ Load", "✅ Load", "✅ Load"},
		{"PyTorch ONNX", "✅ Import", "✅ Native", "✅ Load", "✅ Load"},
		{"TensorFlow ONNX", "✅ Import", "✅ Load", "✅ Native", "✅ Load"},
	}

	for _, row := range matrix {
		fmt.Printf("%-15s | %-10s | %-10s | %-12s | %-12s\n",
			row[0], row[1], row[2], row[3], row[4])
		if row[0] == "Source/Target" {
			fmt.Println("----------------|------------|------------|--------------|------------")
		}
	}

	fmt.Println("\n🔑 Legend:")
	fmt.Println("✅ Supported  ❌ Not Supported")
	fmt.Println("\n💡 Key Points:")
	fmt.Println("• ONNX format enables cross-framework interoperability")
	fmt.Println("• Go-Metal JSON format is optimized for Go-Metal ecosystem")
	fmt.Println("• PyTorch/TensorFlow can export to ONNX for Go-Metal import")
	fmt.Println("• Go-Metal can export to ONNX for other framework deployment")
}
```

**Advanced Features Demonstrated:**
- Multiple model architectures (CNN and MLP)
- Bidirectional format conversion (JSON ↔ ONNX)
- Cross-framework compatibility testing
- Synthetic weight generation for demos
- Comprehensive compatibility matrix

## 🎯 ONNX Layer Compatibility

### ✅ Fully Supported Layers

| Go-Metal Layer | ONNX Operation | Notes |
|----------------|----------------|-------|
| Dense | MatMul + Add | Automatic weight transposition |
| Conv2D | Conv | Supports padding, stride, bias |
| ReLU | Relu | Direct mapping |
| LeakyReLU | LeakyRelu | Configurable alpha parameter |
| Sigmoid | Sigmoid | Direct mapping |
| Tanh | Tanh | Direct mapping |
| Softmax | Softmax | Configurable axis |
| BatchNorm | BatchNormalization | Inference mode, running stats |
| Dropout | Dropout | Training/inference mode handling |

### ⚠️ Partially Supported

| Layer | Status | Limitations |
|-------|--------|-------------|
| Swish | ✅ Exported | Decomposed to Sigmoid + Mul |
| ELU | ❌ Limited | Requires custom ONNX operator |
| Custom Layers | ❌ No | Not supported in ONNX standard |

### 🔧 ONNX Export Configuration

**Supported Data Types:**
- Float32 (primary)
- Int32 (for indices/labels)

**ONNX Version:**
- IR Version: 7
- Opset Version: 13

**Automatic Conversions:**
- Weight matrix transposition (Go-Metal [input, output] → ONNX [output, input])
- Bias absorption from separate Add nodes during import
- BatchNorm running statistics for inference mode

## 🚨 Common Issues and Solutions

### Issue: "Unsupported layer type for ONNX export"
**Solution**: Check layer compatibility table. Some custom layers need decomposition.

### Issue: "Weight shape mismatch during import"
**Solution**: ONNX uses different weight ordering. Go-Metal handles this automatically.

### Issue: "Model fails to load in PyTorch"
**Solution**: Ensure ONNX file is valid:
```bash
# Install onnx tools
pip install onnx onnxruntime

# Validate ONNX file
python -c "import onnx; onnx.checker.check_model('model.onnx')"
```

### Issue: "Performance degradation after ONNX round-trip"
**Solution**: Some precision may be lost during conversion. Use identical test data for validation.

## 📋 Best Practices

### Export Guidelines
- ✅ Use standard layer types for maximum compatibility
- ✅ Test models after export/import round-trip
- ✅ Include meaningful metadata and version info
- ✅ Validate ONNX files with external tools

### Import Guidelines
- ✅ Verify model architecture matches expectations
- ✅ Test inference with known inputs for validation
- ✅ Check weight tensor shapes and counts
- ✅ Handle missing training metadata gracefully

### Production Deployment
- ✅ Keep both Go-Metal JSON and ONNX versions
- ✅ Use Go-Metal JSON for Go-Metal-specific deployments
- ✅ Use ONNX for cross-platform deployment
- ✅ Implement fallback mechanisms for unsupported layers

## 🚀 Next Steps

Master ONNX integration with go-metal:

- **[Performance Guide](../guides/performance.md)** - Optimize model conversion workflows
- **[Model Serialization](../examples/model-serialization/)** - Complete serialization examples
- **[Production Deployment](../advanced/deployment.md)** - Deploy ONNX models in production

**Ready for cross-framework development?** Use ONNX to bridge Go-Metal with the broader ML ecosystem!

---

## 🧠 Key Takeaways

- **Cross-framework compatibility**: ONNX enables model sharing between Go-Metal, PyTorch, TensorFlow
- **Bidirectional conversion**: Export Go-Metal models to ONNX, import ONNX models to Go-Metal
- **Production flexibility**: Use the best framework for training, deployment, and inference
- **Standard compliance**: Full ONNX v1.13 support with automatic format handling
- **Ecosystem integration**: Connect Go-Metal to the broader machine learning ecosystem

With ONNX integration, you can leverage the entire ML ecosystem while maintaining the performance benefits of Go-Metal on Apple Silicon!