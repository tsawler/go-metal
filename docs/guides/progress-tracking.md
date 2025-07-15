# Progress Tracking Guide

PyTorch-style progress bars and training session management for go-metal.

## 🎯 Overview

Go-metal provides a comprehensive progress tracking system that includes:

- **PyTorch-style Progress Bars**: Real-time training visualization with customizable metrics
- **Model Architecture Display**: Complete model summaries with parameter counts and memory estimates
- **Training Session Management**: Structured training and validation phases with automatic progress tracking
- **Performance Metrics**: Batch rate, ETA calculation, loss, accuracy, and custom metrics

This system is designed to give you the same professional training experience as PyTorch, with minimal performance overhead.

## 🚀 Quick Start

### Basic Progress Bar

```go
package main

import (
    "math/rand"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create a basic progress bar
    pb := training.NewProgressBar("Training", 100)
    
    for step := 1; step <= 100; step++ {
        // Your training logic here
        loss := performTrainingStep()
        
        // Update progress with metrics
        metrics := map[string]float64{
            "loss": loss,
            "accuracy": calculateAccuracy(),
        }
        pb.Update(step, metrics)
    }
    
    pb.Finish()
}

// Helper function to simulate training step
func performTrainingStep() float64 {
    // Simulate training logic returning loss
    return 2.0 + rand.Float64()*0.5
}

// Helper function to calculate accuracy
func calculateAccuracy() float64 {
    // Simulate accuracy calculation
    return 0.5 + rand.Float64()*0.4
}
```

## 🏗️ Training Session Management

### Complete Training Session

The recommended approach is to use the `CreateTrainingSession` method for structured training:

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Build your model
    inputShape := []int{32, 3, 32, 32}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(32, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddConv2D(64, 3, 2, 1, true, "conv2").
        AddReLU("relu2").
        AddDense(128, true, "fc1").
        AddReLU("relu3").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    // Configure trainer
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.01,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers for better performance
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Training parameters
    epochs := 10
    stepsPerEpoch := 100
    validationSteps := 20
    
    // Create training session with progress visualization
    session := trainer.CreateTrainingSession("MyCNN", epochs, stepsPerEpoch, validationSteps)
    
    // Start training (displays model architecture automatically)
    session.StartTraining()
    
    // Training loop with automatic progress tracking
    for epoch := 1; epoch <= epochs; epoch++ {
        // Training phase
        session.StartEpoch(epoch)
        
        for step := 1; step <= stepsPerEpoch; step++ {
            // Generate or load your training data
            inputData, labelData := generateTrainingBatch()
            
            // Train batch using unified API
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training step failed: %v", err)
                continue
            }
            
            // Calculate accuracy (optional)
            accuracy := calculateAccuracy(epoch, step)
            
            // Update progress (loss and accuracy are displayed automatically)
            session.UpdateTrainingProgress(step, float64(result.Loss), accuracy)
        }
        
        session.FinishTrainingEpoch()
        
        // Validation phase
        session.StartValidation()
        
        for step := 1; step <= validationSteps; step++ {
            // Generate or load validation data
            valInputData, _ := generateValidationBatch()
            
            // Run validation
            _, err := trainer.InferBatch(valInputData, inputShape)
            if err != nil {
                log.Printf("Validation step failed: %v", err)
                continue
            }
            
            // Calculate validation metrics
            valLoss := calculateValidationLoss(epoch, step)
            valAccuracy := calculateValidationAccuracy(epoch, step)
            
            // Update validation progress
            session.UpdateValidationProgress(step, valLoss, valAccuracy)
        }
        
        session.FinishValidationEpoch()
        session.PrintEpochSummary()
    }
    
    fmt.Println("Training completed!")
}

// Helper function to generate training batch
func generateTrainingBatch() ([]float32, *training.Int32Labels) {
    batchSize := 32
    
    // Generate random input data
    inputData := make([]float32, batchSize*3*32*32)
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    
    // Generate random labels
    labelData := make([]int32, batchSize)
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    // Create label tensor
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}

// Helper function to generate validation batch
func generateValidationBatch() ([]float32, *training.Int32Labels) {
    return generateTrainingBatch()
}

// Helper function to calculate accuracy
func calculateAccuracy(epoch, step int) float64 {
    // Simulate improving accuracy over time
    progress := float64(epoch-1)*100 + float64(step)
    totalSteps := 1000.0 // 10 epochs * 100 steps
    
    baseAccuracy := 0.1 + 0.8*(progress/totalSteps)
    noise := (rand.Float64() - 0.5) * 0.1
    
    accuracy := baseAccuracy + noise
    if accuracy < 0 {
        accuracy = 0
    }
    if accuracy > 1 {
        accuracy = 1
    }
    
    return accuracy
}

// Helper function to calculate validation loss
func calculateValidationLoss(epoch, step int) float64 {
    // Simulate decreasing validation loss
    progress := float64(epoch-1)*20 + float64(step)
    totalSteps := 200.0 // 10 epochs * 20 steps
    
    baseLoss := 2.5 - 1.5*(progress/totalSteps)
    noise := (rand.Float64() - 0.5) * 0.2
    
    loss := baseLoss + noise
    if loss < 0.1 {
        loss = 0.1
    }
    
    return loss
}

// Helper function to calculate validation accuracy
func calculateValidationAccuracy(epoch, step int) float64 {
    return calculateAccuracy(epoch, step) * 0.9 // Slightly lower than training
}
```

## 📊 Model Architecture Display

### Automatic Architecture Printing

When you start a training session, the model architecture is automatically displayed:

```go
// This automatically prints the model architecture
session.StartTraining()
```

### Manual Architecture Display

```go
// Print model architecture manually
trainer.PrintModelArchitecture("MyModel")

// Or use the architecture printer directly
printer := training.NewModelArchitecturePrinter("MyModel")
printer.PrintArchitecture(modelSpec)
```

## 📈 Output Examples

### Model Architecture Display

```
Model Architecture:
MyCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=true)
  (relu1): ReLU()
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=true)
  (relu2): ReLU()
  (fc1): Linear(in_features=8192, out_features=128, bias=true)
  (relu3): ReLU()
  (output): Linear(in_features=128, out_features=10, bias=true)
)

Total parameters: 1.1M
Trainable parameters: 1.1M
Non-trainable parameters: 0
Input size (MB): 0.012
Forward/backward pass size (MB): 0.048
Params size (MB): 4.4
Estimated Total Size (MB): 4.5
```

### Training Progress Bars

```
Starting training...
Epoch 1/10 (Training): 100%|██████████████████████████████████████████████████| 100/100 [00:15<00:00, 6.45batch/s, loss=2.31, accuracy=12.50%]
Epoch 1/10 (Validation):  100%|████████████████████████████████████████████████| 20/20 [00:02<00:00, 9.87batch/s, loss=2.28, accuracy=15.20%]

Epoch 1/10 Summary:
  Training   - Loss: 2.3089, Accuracy: 12.50%
  Validation - Loss: 2.2834, Accuracy: 15.20%

Epoch 2/10 (Training): 100%|██████████████████████████████████████████████████| 100/100 [00:14<00:00, 6.98batch/s, loss=2.15, accuracy=24.80%]
Epoch 2/10 (Validation):  100%|████████████████████████████████████████████████| 20/20 [00:02<00:00, 10.12batch/s, loss=2.05, accuracy=28.60%]

Epoch 2/10 Summary:
  Training   - Loss: 2.1456, Accuracy: 24.80%
  Validation - Loss: 2.0512, Accuracy: 28.60%
```

## 🔧 Advanced Features

### Custom Metrics

```go
// Add custom metrics to progress display
for step := 1; step <= stepsPerEpoch; step++ {
    result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
    if err != nil {
        continue
    }
    
    // Calculate additional metrics
    accuracy := calculateAccuracy(result, labelData)
    precision := calculatePrecision(result, labelData)
    recall := calculateRecall(result, labelData)
    f1Score := calculateF1Score(precision, recall)
    
    // Update with multiple metrics
    session.UpdateTrainingProgressWithMetrics(step, float64(result.Loss), map[string]float64{
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1_score":  f1Score,
    })
}

// Helper functions for custom metrics
func calculatePrecision(epoch, step int) float64 {
    return 0.7 + 0.2*rand.Float64()
}

func calculateRecall(epoch, step int) float64 {
    return 0.6 + 0.3*rand.Float64()
}

func calculateF1Score(precision, recall float64) float64 {
    return 2 * (precision * recall) / (precision + recall)
}
```

### Progress Bar Customization

```go
// Create progress bar with custom settings
pb := training.NewProgressBar("Custom Training", totalSteps)
pb.SetWidth(80)        // Set progress bar width
pb.ShowRate(true)      // Show batch rate
pb.ShowETA(true)       // Show estimated time remaining

// Custom update with multiple metrics
metrics := map[string]float64{
    "loss":      loss,
    "accuracy":  accuracy,
    "precision": precision,
    "recall":    recall,
}
pb.Update(step, metrics)
```

### Multiple Progress Bars

```go
// Run multiple progress bars for different phases
trainProgress := training.NewProgressBar("Training", trainSteps)
valProgress := training.NewProgressBar("Validation", valSteps)

// Use them independently
trainProgress.Update(step, trainMetrics)
valProgress.Update(step, valMetrics)

trainProgress.Finish()
valProgress.Finish()
```

## ⚡ Performance Characteristics

The progress bar system is designed for minimal overhead:

- **Update Cost**: ~10-50 microseconds per update
- **Memory Usage**: <1KB additional memory per progress bar
- **Thread Safety**: Progress bars can be safely used from multiple goroutines
- **Smart Updates**: Only redraws when necessary to minimize terminal flicker

## 🎯 Integration Patterns

### Converting Existing Training Code

**Before (manual progress tracking):**
```go
for step := 1; step <= totalSteps; step++ {
    result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
    if err != nil {
        return err
    }
    
    fmt.Printf("Step %d: Loss=%.4f\n", step, result.Loss)
}
```

**After (with progress bars):**
```go
pb := training.NewProgressBar("Training", totalSteps)
for step := 1; step <= totalSteps; step++ {
    result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
    if err != nil {
        return err
    }
    
    // Enhanced with progress visualization
    metrics := map[string]float64{"loss": float64(result.Loss)}
    pb.Update(step, metrics)
}
pb.Finish()
```

### With Training Session (Recommended)

```go
// Full session-based approach (recommended)
session := trainer.CreateTrainingSession("MyModel", epochs, stepsPerEpoch, validationSteps)
session.StartTraining()

for epoch := 1; epoch <= epochs; epoch++ {
    session.StartEpoch(epoch)
    
    // Training loop with automatic progress tracking
    for step := 1; step <= stepsPerEpoch; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
        if err != nil {
            continue
        }
        
        accuracy := calculateAccuracy(epoch, step)
        session.UpdateTrainingProgress(step, float64(result.Loss), accuracy)
    }
    
    session.FinishTrainingEpoch()
    
    // Validation phase
    session.StartValidation()
    for step := 1; step <= validationSteps; step++ {
        // Validation logic...
        session.UpdateValidationProgress(step, valLoss, valAccuracy)
    }
    session.FinishValidationEpoch()
    session.PrintEpochSummary()
}
```

## 🔍 Configuration Options

### Progress Bar Settings

```go
type ProgressBarConfig struct {
    Width    int    // Character width of progress bar (default: 70)
    ShowRate bool   // Display batch processing rate (default: true)
    ShowETA  bool   // Display estimated time remaining (default: true)
    Prefix   string // Custom prefix for progress bar
}
```

### Training Session Settings

```go
type TrainingSessionConfig struct {
    ModelName         string // Name displayed in architecture summary
    ShowArchitecture  bool   // Whether to print model architecture at start
    ParameterFormat   string // Format for parameter counts (K/M suffixes)
    MemoryUnits      string // Units for memory display (MB/GB)
}
```

## 📋 API Reference

### Core Methods

#### TrainingSession Methods
```go
// Create training session
session := trainer.CreateTrainingSession(modelName, epochs, stepsPerEpoch, validationSteps)

// Training flow
session.StartTraining()                                    // Show architecture and start
session.StartEpoch(epoch)                                  // Begin epoch
session.UpdateTrainingProgress(step, loss, accuracy)       // Update training progress
session.FinishTrainingEpoch()                             // End training phase

// Validation flow
session.StartValidation()                                  // Begin validation
session.UpdateValidationProgress(step, valLoss, valAcc)    // Update validation progress
session.FinishValidationEpoch()                           // End validation phase
session.PrintEpochSummary()                               // Print epoch summary
```

#### ProgressBar Methods
```go
// Create and configure
pb := training.NewProgressBar(description, totalSteps)
pb.SetWidth(width)
pb.ShowRate(showRate)
pb.ShowETA(showETA)

// Update and finish
pb.Update(currentStep, metrics)
pb.Finish()
```

## 🎯 Best Practices

### Training Session Guidelines

1. **Use Training Sessions**: Prefer `CreateTrainingSession` over manual progress bars for structured training
2. **Clear Naming**: Use descriptive model names for architecture display
3. **Consistent Updates**: Always call `UpdateTrainingProgress` for every training step
4. **Complete Phases**: Always finish training and validation phases properly

### Performance Tips

1. **Batch Updates**: Update progress once per batch, not per sample
2. **Minimal Metrics**: Only calculate and display necessary metrics during training
3. **Smart Validation**: Use reasonable validation step counts (don't validate too frequently)

### Code Organization

```go
// Good: Structured approach
session := trainer.CreateTrainingSession("CIFAR10_CNN", epochs, stepsPerEpoch, validationSteps)
session.StartTraining()

for epoch := 1; epoch <= epochs; epoch++ {
    session.StartEpoch(epoch)
    
    // Training phase
    for step := 1; step <= stepsPerEpoch; step++ {
        // ... training logic ...
        session.UpdateTrainingProgress(step, loss, accuracy)
    }
    session.FinishTrainingEpoch()
    
    // Validation phase
    session.StartValidation()
    // ... validation logic ...
    session.FinishValidationEpoch()
    session.PrintEpochSummary()
}
```

## 🚀 Real-World Example

Here's a complete example from a CNN image classification project:

```go
func trainCNN() error {
    // Model setup
    model, trainer := setupModel()
    defer trainer.Cleanup()
    
    // Data setup
    trainLoader := NewDataLoader(trainDataset, batchSize, true)
    valLoader := NewDataLoader(valDataset, batchSize, false)
    
    epochs := 50
    stepsPerEpoch := len(trainLoader)
    validationSteps := len(valLoader)
    
    // Create training session
    session := trainer.CreateTrainingSession("CIFAR10_ResNet", epochs, stepsPerEpoch, validationSteps)
    session.StartTraining()
    
    for epoch := 1; epoch <= epochs; epoch++ {
        // Training phase
        session.StartEpoch(epoch)
        
        runningLoss := 0.0
        runningAccuracy := 0.0
        
        for step, batch := range trainLoader.Enumerate() {
            result, err := trainer.TrainBatchUnified(batch.Data, batch.Shape, batch.Labels)
            if err != nil {
                log.Printf("Training error: %v", err)
                continue
            }
            
            // Calculate running metrics
            runningLoss += float64(result.Loss)
            accuracy := calculateBatchAccuracy(result, batch.Labels)
            runningAccuracy += accuracy
            
            // Update progress with real-time metrics
            avgLoss := runningLoss / float64(step)
            avgAccuracy := runningAccuracy / float64(step)
            session.UpdateTrainingProgress(step, avgLoss, avgAccuracy)
        }
        
        session.FinishTrainingEpoch()
        
        // Validation phase
        session.StartValidation()
        
        valLoss := 0.0
        valAccuracy := 0.0
        
        for step, batch := range valLoader.Enumerate() {
            result, err := trainer.InferBatch(batch.Data, batch.Shape)
            if err != nil {
                continue
            }
            
            batchLoss := calculateLoss(result, batch.Labels)
            batchAccuracy := calculateBatchAccuracy(result, batch.Labels)
            
            valLoss += batchLoss
            valAccuracy += batchAccuracy
            
            avgValLoss := valLoss / float64(step)
            avgValAccuracy := valAccuracy / float64(step)
            session.UpdateValidationProgress(step, avgValLoss, avgValAccuracy)
        }
        
        session.FinishValidationEpoch()
        session.PrintEpochSummary()
    }
    
    return nil
}
```

## 📚 Complete Examples

For complete working examples, see:

- **[Training with Progress](../../examples/training-with-progress/)** - Complete training pipeline with progress bars
- **[CNN Image Classification](../../examples/cnn-image-classification/)** - Real-world CNN training with progress tracking
- **[Complete Training Pipeline](../../examples/complete-training-pipeline/)** - End-to-end training with all features

## 🎯 Summary

The go-metal progress tracking system provides:

- **🎨 PyTorch-Style UI**: Professional progress bars and model summaries
- **📊 Rich Metrics**: Loss, accuracy, batch rate, ETA, and custom metrics
- **🏗️ Structured Training**: Session-based training with automatic phase management
- **⚡ High Performance**: Minimal overhead with smart update strategies
- **🔧 Highly Configurable**: Customizable appearance and behavior

### Next Steps

1. **Try the Examples**: Run the complete examples to see the progress system in action
2. **Add to Your Code**: Convert existing training loops to use training sessions
3. **Customize**: Experiment with custom metrics and progress bar configurations

**Continue Learning:**
- **[Visualization Guide](visualization.md)** - Interactive plotting with sidecar service
- **[Performance Guide](performance.md)** - Optimize training speed
- **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Complete neural network training

---

*Professional training progress tracking makes debugging and monitoring your go-metal models effortless.*