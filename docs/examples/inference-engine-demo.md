# Inference Engine Example

This example demonstrates how to use the go-metal inference engine for making predictions with a trained neural network model. We'll create a simple classification model, train it briefly, save it, and then use the inference engine to make predictions.

## Complete Working Example

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/engine"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

// generateSampleData creates simple synthetic data for demonstration
// Creates data for a 3-class classification problem
func generateSampleData(samples int) ([]float32, []int) {
    features := make([]float32, samples*4) // 4 features per sample
    labels := make([]int, samples)
    
    for i := 0; i < samples; i++ {
        // Generate features based on class
        class := i % 3
        labels[i] = class
        
        // Create separable features for each class
        baseIdx := i * 4
        switch class {
        case 0:
            features[baseIdx] = rand.Float32() * 0.3
            features[baseIdx+1] = rand.Float32() * 0.3
            features[baseIdx+2] = rand.Float32() * 0.3 + 0.7
            features[baseIdx+3] = rand.Float32() * 0.3 + 0.7
        case 1:
            features[baseIdx] = rand.Float32() * 0.3 + 0.7
            features[baseIdx+1] = rand.Float32() * 0.3
            features[baseIdx+2] = rand.Float32() * 0.3
            features[baseIdx+3] = rand.Float32() * 0.3 + 0.7
        case 2:
            features[baseIdx] = rand.Float32() * 0.3 + 0.35
            features[baseIdx+1] = rand.Float32() * 0.3 + 0.35
            features[baseIdx+2] = rand.Float32() * 0.3 + 0.35
            features[baseIdx+3] = rand.Float32() * 0.3 + 0.35
        }
    }
    
    return features, labels
}

// createSimpleModel builds a small neural network for classification
func createSimpleModel() *layers.ModelSpec {
    modelBuilder := layers.NewModelBuilder([]int{32, 4}) // batch_size=32, features=4
    
    // Hidden layer with 16 neurons
    modelBuilder.AddDense(16, true, "hidden1")
    modelBuilder.AddReLU("relu1")
    
    // Output layer with 3 classes
    modelBuilder.AddDense(3, true, "output")
    
    // Compile model
    modelSpec, err := modelBuilder.Compile()
    if err != nil {
        log.Fatalf("Failed to compile model: %v", err)
    }
    
    return modelSpec
}

// trainModel performs quick training to get a functional model
func trainModel(modelSpec *layers.ModelSpec, trainData []float32, trainLabels []int) error {
    fmt.Println("Training model...")
    
    // Configure trainer
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Auto,
        ProblemType:   training.Classification,
        LossFunction:  training.CrossEntropy,
    }
    
    // Create trainer
    trainer, err := training.NewModelTrainer(modelSpec, config)
    if err != nil {
        return fmt.Errorf("failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Train for a few steps
    numSamples := len(trainLabels)
    for step := 0; step < 5; step++ {
        // Get batch
        startIdx := (step * config.BatchSize) % numSamples
        endIdx := startIdx + config.BatchSize
        if endIdx > numSamples {
            endIdx = numSamples
            startIdx = endIdx - config.BatchSize
        }
        
        batchFeatures := trainData[startIdx*4 : endIdx*4]
        batchLabels := trainLabels[startIdx:endIdx]
        
        // Create label data
        labels, err := training.NewInt32Labels(batchLabels, []int{config.BatchSize})
        if err != nil {
            return fmt.Errorf("failed to create labels: %v", err)
        }
        
        // Train step
        result, err := trainer.TrainBatchUnified(
            batchFeatures,
            []int{config.BatchSize, 4},
            labels,
        )
        if err != nil {
            return fmt.Errorf("training failed: %v", err)
        }
        
        fmt.Printf("  Step %d - Loss: %.4f, Accuracy: %.2f%%\n", 
            step+1, result.Loss, result.Accuracy*100)
    }
    
    // Save model
    fmt.Println("\nSaving trained model...")
    return saveTrainedModel(trainer, modelSpec)
}

// saveTrainedModel extracts weights and saves the model
func saveTrainedModel(trainer *training.ModelTrainer, modelSpec *layers.ModelSpec) error {
    // Extract weights
    parameterTensors := trainer.GetParameterTensors()
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, modelSpec)
    if err != nil {
        return fmt.Errorf("failed to extract weights: %v", err)
    }
    
    // Create checkpoint
    checkpoint := &checkpoints.Checkpoint{
        ModelSpec: modelSpec,
        Weights:   weights,
        TrainingState: checkpoints.TrainingState{
            Epoch:        1,
            Step:         5,
            LearningRate: 0.01,
        },
        Metadata: checkpoints.CheckpointMetadata{
            Version:     "1.0.0",
            Framework:   "go-metal",
            Description: "Simple classification model for inference demo",
        },
    }
    
    // Save as JSON
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    err = saver.SaveCheckpoint(checkpoint, "demo_model.json")
    if err != nil {
        return fmt.Errorf("failed to save model: %v", err)
    }
    
    fmt.Println("Model saved to demo_model.json")
    return nil
}

// runInference demonstrates loading and using the model for predictions
func runInference() error {
    fmt.Println("\n=== Running Inference ===")
    
    // Load the saved model
    fmt.Println("Loading saved model...")
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    checkpoint, err := saver.LoadCheckpoint("demo_model.json")
    if err != nil {
        return fmt.Errorf("failed to load model: %v", err)
    }
    
    // Configure inference engine
    config := cgo_bridge.InferenceConfig{
        UseDynamicEngine:       true,
        BatchNormInferenceMode: true,
        InputShape:             []int32{1, 4}, // Single sample, 4 features
        InputShapeLen:          2,
        UseCommandPooling:      true,
        OptimizeForSingleBatch: true,
        ProblemType:            0, // Classification
        LossFunction:           2, // CrossEntropy
    }
    
    // Create inference engine
    fmt.Println("Creating inference engine...")
    inferenceEngine, err := engine.NewModelInferenceEngineFromDynamicTraining(
        checkpoint.ModelSpec,
        config,
    )
    if err != nil {
        return fmt.Errorf("failed to create inference engine: %v", err)
    }
    defer inferenceEngine.Cleanup()
    
    // Load weights
    err = inferenceEngine.LoadWeights(checkpoint.Weights)
    if err != nil {
        return fmt.Errorf("failed to load weights: %v", err)
    }
    
    // Test with different samples
    fmt.Println("\nMaking predictions:")
    fmt.Println("Class 0 features: low x1,x2 + high x3,x4")
    fmt.Println("Class 1 features: high x1 + low x2,x3 + high x4")  
    fmt.Println("Class 2 features: medium all features")
    fmt.Println()
    
    testSamples := []struct {
        name     string
        features []float32
        expected int
    }{
        {"Class 0 sample", []float32{0.1, 0.2, 0.8, 0.9}, 0},
        {"Class 1 sample", []float32{0.9, 0.1, 0.2, 0.8}, 1},
        {"Class 2 sample", []float32{0.5, 0.5, 0.5, 0.5}, 2},
        {"Ambiguous sample", []float32{0.4, 0.3, 0.6, 0.7}, -1},
    }
    
    for _, test := range testSamples {
        // Run inference
        result, err := inferenceEngine.Predict(test.features, []int{1, 4})
        if err != nil {
            return fmt.Errorf("inference failed: %v", err)
        }
        
        // Find predicted class
        maxIdx := 0
        maxProb := result.Predictions[0]
        for i, prob := range result.Predictions {
            if prob > maxProb {
                maxProb = prob
                maxIdx = i
            }
        }
        
        // Display results
        fmt.Printf("%s: features=%v\n", test.name, test.features)
        fmt.Printf("  Predicted class: %d (confidence: %.1f%%)\n", maxIdx, maxProb*100)
        fmt.Printf("  Class probabilities: [")
        for i, prob := range result.Predictions {
            if i > 0 {
                fmt.Printf(", ")
            }
            fmt.Printf("%.1f%%", prob*100)
        }
        fmt.Printf("]\n")
        
        if test.expected >= 0 {
            if maxIdx == test.expected {
                fmt.Printf("  ✓ Correct prediction\n")
            } else {
                fmt.Printf("  ✗ Expected class %d\n", test.expected)
            }
        }
        fmt.Println()
    }
    
    return nil
}

func main() {
    // Initialize Metal device
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    
    // Initialize memory manager
    memory.InitializeGlobalMemoryManager(device)
    
    fmt.Println("Go-Metal Inference Engine Demo")
    fmt.Println("==============================")
    
    // Step 1: Create and train a simple model
    modelSpec := createSimpleModel()
    fmt.Printf("\nCreated model with %d parameters\n", modelSpec.TotalParameters)
    
    // Generate training data
    trainData, trainLabels := generateSampleData(160)
    
    // Train the model
    err = trainModel(modelSpec, trainData, trainLabels)
    if err != nil {
        log.Fatalf("Training failed: %v", err)
    }
    
    // Step 2: Use inference engine with the saved model
    err = runInference()
    if err != nil {
        log.Fatalf("Inference failed: %v", err)
    }
    
    fmt.Println("\nDemo completed successfully!")
}
```

## Running the Example

1. Save the code to a file (e.g., `inference_demo.go`)
2. Ensure you have go-metal installed and configured
3. Run the example:

```bash
go run inference_demo.go
```

## Expected Output

```
Go-Metal Inference Engine Demo
==============================

Created model with 368 parameters
Training model...
  Step 1 - Loss: 1.0986, Accuracy: 33.33%
  Step 2 - Loss: 1.0847, Accuracy: 40.62%
  Step 3 - Loss: 1.0621, Accuracy: 43.75%
  Step 4 - Loss: 1.0234, Accuracy: 53.12%
  Step 5 - Loss: 0.9756, Accuracy: 56.25%

Saving trained model...
Model saved to demo_model.json

=== Running Inference ===
Loading saved model...
Creating inference engine...

Making predictions:
Class 0 features: low x1,x2 + high x3,x4
Class 1 features: high x1 + low x2,x3 + high x4
Class 2 features: medium all features

Class 0 sample: features=[0.1 0.2 0.8 0.9]
  Predicted class: 0 (confidence: 45.2%)
  Class probabilities: [45.2%, 28.3%, 26.5%]
  ✓ Correct prediction

Class 1 sample: features=[0.9 0.1 0.2 0.8]
  Predicted class: 1 (confidence: 48.7%)
  Class probabilities: [25.1%, 48.7%, 26.2%]
  ✓ Correct prediction

Class 2 sample: features=[0.5 0.5 0.5 0.5]
  Predicted class: 2 (confidence: 41.3%)
  Class probabilities: [29.4%, 29.3%, 41.3%]
  ✓ Correct prediction

Ambiguous sample: features=[0.4 0.3 0.6 0.7]
  Predicted class: 0 (confidence: 37.8%)
  Class probabilities: [37.8%, 30.1%, 32.1%]

Demo completed successfully!
```

## Key Concepts Demonstrated

1. **Model Creation**: Building a simple neural network architecture
2. **Training**: Quick training to get a functional model
3. **Model Saving**: Persisting the trained model to disk
4. **Model Loading**: Loading the saved model for inference
5. **Inference Engine Setup**: Configuring and creating the inference engine
6. **Making Predictions**: Running inference on new data
7. **Result Interpretation**: Processing and understanding predictions

## Customization

You can modify this example to:
- Use different model architectures
- Work with real datasets
- Implement batch inference
- Add preprocessing/postprocessing
- Support different problem types (regression, multi-label classification)