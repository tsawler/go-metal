# Inference Engine Guide

The go-metal inference engine provides high-performance model inference using Apple's Metal Performance Shaders. This guide covers how to use the inference engine for making predictions with trained models.

## Overview

The inference engine is optimized for forward-pass execution without the overhead of training operations like gradient computation. It supports:

- Loading models from checkpoints (JSON or ONNX format)
- Batch inference for multiple inputs
- Single-sample prediction
- GPU-accelerated execution
- Memory-efficient processing

## Key Components

### ModelInferenceEngine

The `ModelInferenceEngine` is the primary interface for running inference. It extends the base `MPSInferenceEngine` with model-specific functionality.

```go
import (
    "github.com/tsawler/go-metal/engine"
    "github.com/tsawler/go-metal/cgo_bridge"
)
```

### Creating an Inference Engine

There are two ways to create an inference engine:

1. **From a Model Specification**:
```go
inferenceEngine, err := engine.NewModelInferenceEngine(modelSpec, config)
```

2. **From a Dynamic Training Engine** (recommended for compatibility):
```go
inferenceEngine, err := engine.NewModelInferenceEngineFromDynamicTraining(modelSpec, config)
```

## Configuration

The inference engine requires an `InferenceConfig` that specifies execution parameters:

```go
config := cgo_bridge.InferenceConfig{
    UseDynamicEngine:       true,    // Use dynamic graph construction
    BatchNormInferenceMode: true,    // Set BatchNorm to inference mode
    InputShape:             []int32{batch_size, input_features},
    InputShapeLen:          2,
    UseCommandPooling:      true,    // Optimize GPU command submission
    OptimizeForSingleBatch: true,    // Optimize for batch size 1
    ProblemType:            0,       // 0=Classification, 1=Regression
    LossFunction:           2,       // Loss function type (2=CrossEntropy, 5=MSE)
}
```

### Important Configuration Fields

- **ProblemType**: Set to `1` for regression problems, `0` for classification
- **LossFunction**: Choose appropriate loss (e.g., MSE for regression, CrossEntropy for classification)
- **InputShape**: Must match your model's expected input dimensions
- **BatchNormInferenceMode**: Always set to `true` for inference

## Basic Usage Pattern

1. **Load a saved model**
2. **Create inference engine**
3. **Load weights into the engine**
4. **Run prediction**
5. **Process results**
6. **Clean up resources**

## Complete Example

Here's a simple but complete example that demonstrates inference with a trained model:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/engine"
    "github.com/tsawler/go-metal/memory"
)

func main() {
    // Initialize Metal device
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    
    // Initialize memory manager
    memory.InitializeGlobalMemoryManager(device)
    
    // Load a saved model (supports JSON or ONNX)
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    checkpoint, err := saver.LoadCheckpoint("model.json")
    if err != nil {
        log.Fatalf("Failed to load model: %v", err)
    }
    
    // Configure inference engine
    config := cgo_bridge.InferenceConfig{
        UseDynamicEngine:       true,
        BatchNormInferenceMode: true,
        InputShape:             []int32{1, 4}, // Batch size 1, 4 features
        InputShapeLen:          2,
        UseCommandPooling:      true,
        OptimizeForSingleBatch: true,
        ProblemType:            0,  // Classification
        LossFunction:           2,  // CrossEntropy
    }
    
    // Create inference engine
    inferenceEngine, err := engine.NewModelInferenceEngineFromDynamicTraining(
        checkpoint.ModelSpec, 
        config,
    )
    if err != nil {
        log.Fatalf("Failed to create inference engine: %v", err)
    }
    defer inferenceEngine.Cleanup()
    
    // Load model weights
    err = inferenceEngine.LoadWeights(checkpoint.Weights)
    if err != nil {
        log.Fatalf("Failed to load weights: %v", err)
    }
    
    // Prepare input data
    inputData := []float32{0.5, 0.3, 0.8, 0.2} // Example features
    inputShape := []int{1, 4}                  // Shape matches config
    
    // Run inference
    result, err := inferenceEngine.Predict(inputData, inputShape)
    if err != nil {
        log.Fatalf("Inference failed: %v", err)
    }
    
    // Process results
    fmt.Printf("Predictions: %v\n", result.Predictions)
    fmt.Printf("Output shape: %v\n", result.OutputShape)
    
    // For classification, find the predicted class
    if config.ProblemType == 0 {
        maxIdx := 0
        maxVal := result.Predictions[0]
        for i, val := range result.Predictions {
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }
        fmt.Printf("Predicted class: %d (confidence: %.2f%%)\n", 
            maxIdx, maxVal*100)
    }
}
```

## Batch Inference

For processing multiple samples efficiently:

```go
// Prepare batch data
batchSize := 32
inputData := make([]float32, batchSize * inputFeatures)
// ... fill inputData ...

// Update config for batch
config.InputShape = []int32{int32(batchSize), int32(inputFeatures)}
config.OptimizeForSingleBatch = false

// Run batch inference
result, err := inferenceEngine.Predict(inputData, []int{batchSize, inputFeatures})
```

## Working with Different Model Types

### Regression Models

For regression problems, configure appropriately:

```go
config := cgo_bridge.InferenceConfig{
    // ... other settings ...
    ProblemType:  1,  // Regression
    LossFunction: 5,  // MSE (Mean Squared Error)
}
```

### Image Classification

For CNN models processing images:

```go
// Example for 224x224 RGB images
config.InputShape = []int32{1, 224, 224, 3}
config.InputShapeLen = 4
```

## Performance Optimization

1. **Enable Command Pooling**: Set `UseCommandPooling = true` for better GPU utilization
2. **Optimize for Single Batch**: When processing one sample at a time, set `OptimizeForSingleBatch = true`
3. **Reuse Engine Instance**: Create the engine once and reuse for multiple predictions
4. **Use Persistent Buffers**: For repeated inference, the engine maintains GPU-resident buffers

## Custom Normalization

For models with BatchNorm layers, you can set custom normalization:

```go
// List available BatchNorm layers
bnLayers := inferenceEngine.ListBatchNormLayers()

// Set custom normalization
mean := []float32{0.485, 0.456, 0.406}
variance := []float32{0.229, 0.224, 0.225}
err = inferenceEngine.SetCustomNormalization("batchnorm1", mean, variance)

// Or use standard normalization (mean=0, var=1)
err = inferenceEngine.SetStandardNormalization("batchnorm1")
```

## Error Handling

Always check for errors at each step:

```go
if err != nil {
    // Log error details
    log.Printf("Inference error: %v", err)
    
    // Clean up if needed
    if inferenceEngine != nil {
        inferenceEngine.Cleanup()
    }
    
    return fmt.Errorf("inference failed: %w", err)
}
```

## Memory Management

The inference engine manages GPU memory efficiently:

- Tensors remain GPU-resident between calls
- Automatic cleanup with `defer inferenceEngine.Cleanup()`
- Shared memory pool across operations

## Integration Tips

1. **Model Loading**: Support both JSON and ONNX formats for flexibility
2. **Input Preprocessing**: Normalize inputs consistently with training
3. **Output Postprocessing**: Apply appropriate transformations (e.g., softmax for probabilities)
4. **Error Recovery**: Implement retry logic for transient GPU errors

## Troubleshooting

Common issues and solutions:

- **Shape Mismatch**: Ensure input shape matches model expectations
- **Memory Errors**: Check available GPU memory, reduce batch size if needed
- **Performance**: Enable optimization flags in config
- **Accuracy**: Verify input preprocessing matches training pipeline