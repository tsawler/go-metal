# PyTorch-Style Progress Bar for Go-Metal Training

The go-metal training system now includes a PyTorch-style progress bar that provides real-time visualization of training progress, model architecture, and performance metrics.

## Features

- **Model Architecture Display**: PyTorch-style model summary with parameter counts and memory estimates
- **Real-time Progress Bars**: Training and validation progress with customizable metrics
- **Performance Metrics**: Batch rate, loss, accuracy, and custom metrics display
- **ETA Calculation**: Estimated time remaining for each epoch
- **Memory Estimates**: Input size, parameter size, and total memory usage estimates

## Basic Usage

### 1. Simple Progress Bar

```go
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
```

### 2. Model Architecture Display

```go
// Print PyTorch-style model architecture
trainer.PrintModelArchitecture("MyModel")

// Or use the architecture printer directly
printer := training.NewModelArchitecturePrinter("MyModel")
printer.PrintArchitecture(modelSpec)
```

### 3. Complete Training Session

```go
// Create model and trainer
trainer, err := training.NewModelTrainer(model, config)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// Create training session with progress visualization
session := trainer.CreateTrainingSession("MyCNN", epochs, stepsPerEpoch, validationSteps)

// Start training (displays model architecture)
session.StartTraining()

// Training loop
for epoch := 1; epoch <= epochs; epoch++ {
    // Training phase
    session.StartEpoch(epoch)
    
    for step := 1; step <= stepsPerEpoch; step++ {
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
        if err != nil {
            log.Fatal(err)
        }
        
        // Update progress
        session.UpdateTrainingProgress(step, float64(result.Loss), accuracy)
    }
    
    session.FinishTrainingEpoch()
    
    // Validation phase
    session.StartValidation()
    
    for step := 1; step <= validationSteps; step++ {
        // Your validation logic
        valLoss, valAccuracy := performValidation()
        session.UpdateValidationProgress(step, valLoss, valAccuracy)
    }
    
    session.FinishValidationEpoch()
    session.PrintEpochSummary()
}
```

## Output Examples

### Model Architecture Display

```
Model Architecture:
CatDogCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=true)
  (relu1): ReLU()
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=true)
  (relu2): ReLU()
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=true)
  (relu3): ReLU()
  (fc1): Linear(in_features=32768, out_features=512, bias=true)
  (relu4): ReLU()
  (fc2): Linear(in_features=512, out_features=2, bias=true)
  (softmax): Softmax(dim=-1)
)

Total parameters: 16.8M
Trainable parameters: 16.8M
Non-trainable parameters: 0
Input size (MB): 0.012
Forward/backward pass size (MB): 0.048
Params size (MB): 67.2
Estimated Total Size (MB): 67.3
```

### Training Progress

```
Starting training...
Epoch 1/5 (Training): 100%|██████████████████████████████████████████████████████████████████████████| 620/620 [00:52<00:00, 11.91batch/s, loss=1.31, accuracy=65.23%]
Epoch 1/5 (Validation):  40%|████████████████████████████████████████████████▍                          | 62/155 [00:03<00:05, 17.78batch/s, loss=1.28, accuracy=66.51%]
```

### Epoch Summary

```
Epoch 1/5 Summary:
  Training   - Loss: 1.3087, Accuracy: 65.23%
  Validation - Loss: 1.2834, Accuracy: 66.51%
```

## Advanced Usage

### Custom Metrics

```go
// Add custom metrics to progress bar
metrics := map[string]float64{
    "loss": loss,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1Score,
}

session.UpdateTrainingProgress(step, metrics)
```

### Progress Bar Customization

```go
// Create progress bar with custom settings
pb := training.NewProgressBar("Custom Training", totalSteps)
pb.SetWidth(80)        // Set progress bar width
pb.ShowRate(true)      // Show batch rate
pb.ShowETA(true)       // Show estimated time remaining
```

### Multiple Progress Bars

```go
// You can run multiple progress bars for different phases
trainProgress := training.NewProgressBar("Training", trainSteps)
valProgress := training.NewProgressBar("Validation", valSteps)

// Use them independently
trainProgress.Update(step, trainMetrics)
valProgress.Update(step, valMetrics)
```

## Integration with Existing Code

The progress bar system is designed to integrate seamlessly with existing go-metal training code:

```go
// Existing code (no progress bar)
for step := 1; step <= totalSteps; step++ {
    result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
    if err != nil {
        return err
    }
    
    fmt.Printf("Step %d: Loss=%.4f\n", step, result.Loss)
}

// Enhanced with progress bar
pb := training.NewProgressBar("Training", totalSteps)
for step := 1; step <= totalSteps; step++ {
    result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
    if err != nil {
        return err
    }
    
    // Add progress visualization
    metrics := map[string]float64{"loss": float64(result.Loss)}
    pb.Update(step, metrics)
}
pb.Finish()
```

## Performance Impact

The progress bar has minimal performance impact:

- **Update Cost**: ~10-50 microseconds per update
- **Memory Usage**: <1KB additional memory
- **Thread Safety**: Progress bars can be used from multiple goroutines

## Configuration Options

### Progress Bar Settings

- `Width`: Character width of the progress bar (default: 70)
- `ShowRate`: Display batch processing rate (default: true)
- `ShowETA`: Display estimated time remaining (default: true)

### Model Architecture Display Settings

- `ModelName`: Custom name for the model display
- `ParameterFormat`: Format for parameter counts (K/M suffixes)
- `MemoryUnits`: Units for memory size display (MB/GB)

## Examples

See the complete examples in:
- `examples/training_with_progress.go` - Full training example with progress bars
- `training/progress_example.go` - Demonstration of progress bar features
- `training/progress_test.go` - Unit tests and usage examples

Run the example:

```bash
cd go-metal
go run examples/training_with_progress.go
```

This will demonstrate the complete PyTorch-style training visualization with a real CNN model.