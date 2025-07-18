# Model Checkpoints Guide

Complete guide to saving and loading models in go-metal with support for JSON and ONNX formats.

## 🎯 Overview

Go-metal provides comprehensive checkpointing capabilities that allow you to save and restore complete model state including weights, training progress, and optimizer state. This enables:

- **🔄 Resume Training**: Continue training from where you left off
- **📊 Model Versioning**: Track different versions of your models
- **🔄 Interoperability**: Export models to ONNX for use with PyTorch/TensorFlow
- **🎯 Best Model Tracking**: Automatically save best-performing models
- **📦 Deployment**: Save trained models for inference

### Key Features

- **🗂️ Multiple Formats**: JSON (native) and ONNX (interoperability)
- **📈 Training State**: Epoch, step, learning rate, and best metrics
- **🔧 Optimizer State**: Momentum, variance, and other optimizer parameters
- **🏷️ Metadata**: Version, tags, and descriptions
- **🤖 Automatic Management**: Periodic saves and cleanup

## 🚀 Quick Start

### Basic Checkpoint Saving

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create and train a model
    inputShape := []int{32, 784}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(128, true, "hidden1").
        AddReLU("relu1").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers for training
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Train for a few epochs
    fmt.Println("Training model...")
    for epoch := 1; epoch <= 3; epoch++ {
        epochLoss := 0.0
        for step := 1; step <= 10; step++ {
            inputData, labelData := generateTrainingData()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training failed: %v", err)
                continue
            }
            
            epochLoss += float64(result.Loss)
        }
        
        avgLoss := float32(epochLoss / 10.0)
        fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, avgLoss)
    }
    
    // Save checkpoint
    fmt.Println("\nSaving checkpoint...")
    checkpoint := createCheckpoint(trainer, 3, 30, 0.245, 0.89, "Final training checkpoint")
    
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    err = saver.SaveCheckpoint(checkpoint, "my_model.json")
    if err != nil {
        log.Fatalf("Failed to save checkpoint: %v", err)
    }
    
    fmt.Println("✅ Checkpoint saved successfully!")
    
    // Load checkpoint
    fmt.Println("\nLoading checkpoint...")
    loadedCheckpoint, err := saver.LoadCheckpoint("my_model.json")
    if err != nil {
        log.Fatalf("Failed to load checkpoint: %v", err)
    }
    
    fmt.Printf("✅ Loaded checkpoint: Epoch %d, Loss %.6f\n", 
        loadedCheckpoint.TrainingState.Epoch, loadedCheckpoint.TrainingState.BestLoss)
    fmt.Printf("   Model: %d layers, %d weight tensors\n", 
        len(loadedCheckpoint.ModelSpec.Layers), len(loadedCheckpoint.Weights))
}

func createCheckpoint(trainer *training.ModelTrainer, epoch int, step int, loss float32, accuracy float32, description string) *checkpoints.Checkpoint {
    // Extract model components
    modelSpec := trainer.GetModelSpec()
    parameterTensors := trainer.GetParameterTensors()
    
    // Extract weights from GPU tensors
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, modelSpec)
    if err != nil {
        log.Fatalf("Failed to extract weights: %v", err)
    }
    
    // Create checkpoint
    checkpoint := &checkpoints.Checkpoint{
        ModelSpec: modelSpec,
        Weights:   weights,
        TrainingState: checkpoints.TrainingState{
            Epoch:        epoch,
            Step:         step,
            LearningRate: trainer.GetCurrentLearningRate(),
            BestLoss:     loss,
            BestAccuracy: accuracy,
            TotalSteps:   step,
        },
        Metadata: checkpoints.CheckpointMetadata{
            Version:     "1.0.0",
            Framework:   "go-metal",
            Description: description,
            Tags:        []string{"demo", "mnist"},
        },
    }
    
    return checkpoint
}

func generateTrainingData() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*784)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

## 📦 Checkpoint Formats

### JSON Format

Native go-metal format with complete state preservation:

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create a CNN model
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
    
    config := training.TrainerConfig{
        BatchSize:     32,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Train briefly
    fmt.Println("Training CNN model...")
    for epoch := 1; epoch <= 2; epoch++ {
        for step := 1; step <= 5; step++ {
            inputData, labelData := generateCNNTrainingData()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training failed: %v", err)
                continue
            }
            
            if step%2 == 0 {
                fmt.Printf("Epoch %d, Step %d: Loss = %.6f\n", epoch, step, result.Loss)
            }
        }
    }
    
    // Create comprehensive checkpoint
    fmt.Println("\nCreating comprehensive JSON checkpoint...")
    checkpoint := createComprehensiveCheckpoint(trainer)
    
    // Save as JSON
    jsonSaver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    err = jsonSaver.SaveCheckpoint(checkpoint, "cnn_model.json")
    if err != nil {
        log.Fatalf("Failed to save JSON checkpoint: %v", err)
    }
    
    fmt.Println("✅ JSON checkpoint saved!")
    
    // Load and verify
    fmt.Println("Loading and verifying JSON checkpoint...")
    loadedCheckpoint, err := jsonSaver.LoadCheckpoint("cnn_model.json")
    if err != nil {
        log.Fatalf("Failed to load JSON checkpoint: %v", err)
    }
    
    fmt.Printf("✅ Verification successful:\n")
    fmt.Printf("   Model: %d layers, %d parameters\n", 
        len(loadedCheckpoint.ModelSpec.Layers), len(loadedCheckpoint.Weights))
    fmt.Printf("   Training: Epoch %d, Step %d\n", 
        loadedCheckpoint.TrainingState.Epoch, loadedCheckpoint.TrainingState.Step)
    fmt.Printf("   Metadata: %s\n", loadedCheckpoint.Metadata.Description)
    
    // Show layer details
    fmt.Println("\nModel Architecture:")
    for i, layer := range loadedCheckpoint.ModelSpec.Layers {
        fmt.Printf("   Layer %d: %s (%s)\n", i+1, layer.Name, layer.Type.String())
    }
}

func createComprehensiveCheckpoint(trainer *training.ModelTrainer) *checkpoints.Checkpoint {
    modelSpec := trainer.GetModelSpec()
    parameterTensors := trainer.GetParameterTensors()
    
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, modelSpec)
    if err != nil {
        log.Fatalf("Failed to extract weights: %v", err)
    }
    
    return &checkpoints.Checkpoint{
        ModelSpec: modelSpec,
        Weights:   weights,
        TrainingState: checkpoints.TrainingState{
            Epoch:        2,
            Step:         10,
            LearningRate: trainer.GetCurrentLearningRate(),
            BestLoss:     0.187,
            BestAccuracy: 0.943,
            TotalSteps:   10,
        },
        Metadata: checkpoints.CheckpointMetadata{
            Version:     "1.0.0",
            Framework:   "go-metal",
            Description: "CNN model with Conv2D and Dense layers",
            Tags:        []string{"cnn", "conv2d", "cifar10"},
        },
    }
}

func generateCNNTrainingData() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*3*32*32)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

### ONNX Format

Export models for PyTorch/TensorFlow interoperability:

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create a model for ONNX export
    inputShape := []int{1, 784} // Single batch for ONNX
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(256, true, "input_layer").
        AddReLU("relu1").
        AddDense(128, true, "hidden_layer").
        AddReLU("relu2").
        AddDense(64, true, "hidden_layer2").
        AddReLU("relu3").
        AddDense(10, true, "output_layer").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     1,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Train the model
    fmt.Println("Training model for ONNX export...")
    for epoch := 1; epoch <= 5; epoch++ {
        for step := 1; step <= 20; step++ {
            inputData, labelData := generateMNISTData()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training failed: %v", err)
                continue
            }
            
            if step%5 == 0 {
                fmt.Printf("Epoch %d, Step %d: Loss = %.6f\n", epoch, step, result.Loss)
            }
        }
    }
    
    // Create checkpoint for ONNX export
    fmt.Println("\nPreparing model for ONNX export...")
    checkpoint := createONNXCheckpoint(trainer)
    
    // Export to ONNX
    onnxSaver := checkpoints.NewCheckpointSaver(checkpoints.FormatONNX)
    err = onnxSaver.SaveCheckpoint(checkpoint, "mnist_model.onnx")
    if err != nil {
        log.Fatalf("Failed to save ONNX checkpoint: %v", err)
    }
    
    fmt.Println("✅ ONNX model exported successfully!")
    fmt.Println("\nONNX Export Features:")
    fmt.Printf("   • Model file: mnist_model.onnx\n")
    fmt.Printf("   • Input shape: %v\n", inputShape)
    fmt.Printf("   • Output classes: 10\n")
    fmt.Printf("   • Layers: %d\n", len(checkpoint.ModelSpec.Layers))
    fmt.Printf("   • Parameters: %d\n", len(checkpoint.Weights))
    
    fmt.Println("\nUsage with PyTorch:")
    fmt.Println("   import torch")
    fmt.Println("   import torch.onnx")
    fmt.Println("   model = torch.onnx.load('mnist_model.onnx')")
    fmt.Println("   output = model(input_tensor)")
    
    fmt.Println("\nUsage with TensorFlow:")
    fmt.Println("   import tensorflow as tf")
    fmt.Println("   import tf2onnx")
    fmt.Println("   # Convert ONNX to TensorFlow SavedModel")
}

func createONNXCheckpoint(trainer *training.ModelTrainer) *checkpoints.Checkpoint {
    modelSpec := trainer.GetModelSpec()
    parameterTensors := trainer.GetParameterTensors()
    
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, modelSpec)
    if err != nil {
        log.Fatalf("Failed to extract weights: %v", err)
    }
    
    return &checkpoints.Checkpoint{
        ModelSpec: modelSpec,
        Weights:   weights,
        TrainingState: checkpoints.TrainingState{
            Epoch:        5,
            Step:         100,
            LearningRate: trainer.GetCurrentLearningRate(),
            BestLoss:     0.123,
            BestAccuracy: 0.967,
            TotalSteps:   100,
        },
        Metadata: checkpoints.CheckpointMetadata{
            Version:     "1.0.0",
            Framework:   "go-metal",
            Description: "MNIST classifier for PyTorch/TensorFlow interoperability",
            Tags:        []string{"mnist", "classification", "onnx", "interop"},
        },
    }
}

func generateMNISTData() ([]float32, *training.Int32Labels) {
    batchSize := 1
    inputData := make([]float32, batchSize*784)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

## 🤖 Automatic Checkpoint Management

### Training Integration

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create model and trainer
    inputShape := []int{32, 784}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(512, true, "hidden1").
        AddReLU("relu1").
        AddDense(256, true, "hidden2").
        AddReLU("relu2").
        AddDense(128, true, "hidden3").
        AddReLU("relu3").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Configure automatic checkpointing
    checkpointConfig := training.CheckpointConfig{
        SaveDirectory:   "./checkpoints",
        SaveFrequency:   3,    // Save every 3 epochs
        SaveBest:        true, // Save best model
        MaxCheckpoints:  5,    // Keep only 5 recent checkpoints
        Format:          checkpoints.FormatJSON,
        FilenamePattern: "model_epoch_%d_step_%d",
    }
    
    checkpointManager := training.NewCheckpointManager(trainer, checkpointConfig)
    
    // Training loop with automatic checkpointing
    fmt.Println("Training with automatic checkpointing...")
    bestLoss := float32(1e9)
    
    for epoch := 1; epoch <= 10; epoch++ {
        epochLoss := 0.0
        epochSteps := 0
        
        for step := 1; step <= 15; step++ {
            inputData, labelData := generateLargeTrainingData()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training failed: %v", err)
                continue
            }
            
            epochLoss += float64(result.Loss)
            epochSteps++
            
            if step%5 == 0 {
                fmt.Printf("Epoch %d, Step %d: Loss = %.6f\n", epoch, step, result.Loss)
            }
        }
        
        // Calculate epoch metrics
        avgLoss := float32(epochLoss / float64(epochSteps))
        accuracy := calculateAccuracy(avgLoss) // Simulated accuracy
        
        fmt.Printf("Epoch %d completed: Avg Loss = %.6f, Accuracy = %.2f%%\n", 
            epoch, avgLoss, accuracy*100)
        
        totalStep := epoch * 15
        
        // Periodic checkpoint save
        saved, err := checkpointManager.SavePeriodicCheckpoint(epoch, totalStep, avgLoss, accuracy)
        if err != nil {
            log.Printf("Failed to save periodic checkpoint: %v", err)
        } else if saved {
            fmt.Printf("  📁 Periodic checkpoint saved for epoch %d\n", epoch)
        }
        
        // Best model checkpoint save
        if avgLoss < bestLoss {
            bestLoss = avgLoss
            saved, err := checkpointManager.SaveBestCheckpoint(epoch, totalStep, avgLoss, accuracy)
            if err != nil {
                log.Printf("Failed to save best checkpoint: %v", err)
            } else if saved {
                fmt.Printf("  🏆 New best model saved! Loss: %.6f\n", avgLoss)
            }
        }
    }
    
    fmt.Println("\n✅ Training completed with automatic checkpointing!")
    fmt.Println("📁 Checkpoints saved in: ./checkpoints/")
    fmt.Println("   • Periodic checkpoints every 3 epochs")
    fmt.Println("   • Best model checkpoint")
    fmt.Println("   • Maximum 5 checkpoints retained")
}

func generateLargeTrainingData() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*784)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}

func calculateAccuracy(loss float32) float32 {
    // Simulated accuracy calculation based on loss
    // In real training, you'd calculate actual accuracy
    return 1.0 - loss
}
```

## 🔄 Resume Training

### Loading and Continuing Training

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "os"
    
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    checkpointPath := "resume_training_model.json"
    
    // Check if checkpoint exists
    if _, err := os.Stat(checkpointPath); err == nil {
        fmt.Println("📂 Found existing checkpoint - resuming training...")
        resumeTraining(checkpointPath)
    } else {
        fmt.Println("🆕 No checkpoint found - starting fresh training...")
        startFreshTraining(checkpointPath)
    }
}

func startFreshTraining(checkpointPath string) {
    // Create new model
    inputShape := []int{32, 784}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(256, true, "input_layer").
        AddReLU("relu1").
        AddDense(128, true, "hidden_layer").
        AddReLU("relu2").
        AddDense(10, true, "output_layer").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Train for a few epochs
    fmt.Println("Starting fresh training...")
    lastEpoch := trainModel(trainer, inputShape, 1, 5)
    
    // Save checkpoint
    fmt.Println("Saving checkpoint for resume...")
    checkpoint := createResumeCheckpoint(trainer, lastEpoch, lastEpoch*20, 0.456, 0.823)
    
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    err = saver.SaveCheckpoint(checkpoint, checkpointPath)
    if err != nil {
        log.Fatalf("Failed to save checkpoint: %v", err)
    }
    
    fmt.Printf("✅ Fresh training completed and saved (epochs 1-%d)\n", lastEpoch)
}

func resumeTraining(checkpointPath string) {
    // Load checkpoint
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    loadedCheckpoint, err := saver.LoadCheckpoint(checkpointPath)
    if err != nil {
        log.Fatalf("Failed to load checkpoint: %v", err)
    }
    
    fmt.Printf("📋 Loaded checkpoint: Epoch %d, Loss %.6f\n", 
        loadedCheckpoint.TrainingState.Epoch, loadedCheckpoint.TrainingState.BestLoss)
    
    // Recreate model from checkpoint
    inputShape := []int{32, 784}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(256, true, "input_layer").
        AddReLU("relu1").
        AddDense(128, true, "hidden_layer").
        AddReLU("relu2").
        AddDense(10, true, "output_layer").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  loadedCheckpoint.TrainingState.LearningRate,
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
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Restore weights
    fmt.Println("🔄 Restoring model weights...")
    parameterTensors := trainer.GetParameterTensors()
    err = checkpoints.LoadWeightsIntoTensors(loadedCheckpoint.Weights, parameterTensors)
    if err != nil {
        log.Fatalf("Failed to load weights: %v", err)
    }
    
    fmt.Println("✅ Model weights restored!")
    
    // Continue training from where we left off
    startEpoch := loadedCheckpoint.TrainingState.Epoch + 1
    endEpoch := startEpoch + 5
    
    fmt.Printf("🚀 Resuming training from epoch %d to %d...\n", startEpoch, endEpoch)
    lastEpoch := trainModel(trainer, inputShape, startEpoch, endEpoch)
    
    // Save updated checkpoint
    fmt.Println("Updating checkpoint with new progress...")
    updatedCheckpoint := createResumeCheckpoint(trainer, lastEpoch, lastEpoch*20, 0.234, 0.891)
    
    err = saver.SaveCheckpoint(updatedCheckpoint, checkpointPath)
    if err != nil {
        log.Fatalf("Failed to save updated checkpoint: %v", err)
    }
    
    fmt.Printf("✅ Training resumed and completed (epochs %d-%d)\n", startEpoch, lastEpoch)
}

func trainModel(trainer *training.ModelTrainer, inputShape []int, startEpoch int, endEpoch int) int {
    for epoch := startEpoch; epoch <= endEpoch; epoch++ {
        epochLoss := 0.0
        for step := 1; step <= 20; step++ {
            inputData, labelData := generateResumeTrainingData()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training failed: %v", err)
                continue
            }
            
            epochLoss += float64(result.Loss)
            
            if step%5 == 0 {
                fmt.Printf("Epoch %d, Step %d: Loss = %.6f\n", epoch, step, result.Loss)
            }
        }
        
        avgLoss := epochLoss / 20.0
        fmt.Printf("Epoch %d completed: Avg Loss = %.6f\n", epoch, avgLoss)
    }
    
    return endEpoch
}

func createResumeCheckpoint(trainer *training.ModelTrainer, epoch int, step int, loss float32, accuracy float32) *checkpoints.Checkpoint {
    modelSpec := trainer.GetModelSpec()
    parameterTensors := trainer.GetParameterTensors()
    
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, modelSpec)
    if err != nil {
        log.Fatalf("Failed to extract weights: %v", err)
    }
    
    return &checkpoints.Checkpoint{
        ModelSpec: modelSpec,
        Weights:   weights,
        TrainingState: checkpoints.TrainingState{
            Epoch:        epoch,
            Step:         step,
            LearningRate: trainer.GetCurrentLearningRate(),
            BestLoss:     loss,
            BestAccuracy: accuracy,
            TotalSteps:   step,
        },
        Metadata: checkpoints.CheckpointMetadata{
            Version:     "1.0.0",
            Framework:   "go-metal",
            Description: "Resume training checkpoint",
            Tags:        []string{"resume", "training", "mnist"},
        },
    }
}

func generateResumeTrainingData() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*784)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

## 🔧 Advanced Features

### Model Inspection

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create and save a model
    checkpoint := createInspectionModel()
    
    saver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
    err := saver.SaveCheckpoint(checkpoint, "inspection_model.json")
    if err != nil {
        log.Fatalf("Failed to save checkpoint: %v", err)
    }
    
    // Load and inspect the model
    fmt.Println("🔍 Model Inspection Tool")
    fmt.Println("========================")
    
    loadedCheckpoint, err := saver.LoadCheckpoint("inspection_model.json")
    if err != nil {
        log.Fatalf("Failed to load checkpoint: %v", err)
    }
    
    inspectModel(loadedCheckpoint)
}

func createInspectionModel() *checkpoints.Checkpoint {
    inputShape := []int{32, 3, 32, 32}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(32, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddConv2D(64, 3, 2, 1, true, "conv2").
        AddReLU("relu2").
        AddConv2D(128, 3, 2, 1, true, "conv3").
        AddReLU("relu3").
        AddDense(256, true, "fc1").
        AddReLU("relu4").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
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
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Brief training
    for i := 0; i < 10; i++ {
        inputData := make([]float32, 32*3*32*32)
        for j := range inputData {
            inputData[j] = rand.Float32()
        }
        
        labelData := make([]int32, 32)
        for j := range labelData {
            labelData[j] = int32(rand.Intn(10))
        }
        
        labels, err := training.NewInt32Labels(labelData, []int{32})
        if err != nil {
            log.Fatalf("Failed to create label tensor: %v", err)
        }
        
        _, err = trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Printf("Training failed: %v", err)
        }
    }
    
    // Create checkpoint
    parameterTensors := trainer.GetParameterTensors()
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, model)
    if err != nil {
        log.Fatalf("Failed to extract weights: %v", err)
    }
    
    return &checkpoints.Checkpoint{
        ModelSpec: model,
        Weights:   weights,
        TrainingState: checkpoints.TrainingState{
            Epoch:        10,
            Step:         10,
            LearningRate: trainer.GetCurrentLearningRate(),
            BestLoss:     0.567,
            BestAccuracy: 0.789,
            TotalSteps:   10,
        },
        Metadata: checkpoints.CheckpointMetadata{
            Version:     "1.0.0",
            Framework:   "go-metal",
            Description: "CNN model for inspection",
            Tags:        []string{"cnn", "inspection", "demo"},
        },
    }
}

func inspectModel(checkpoint *checkpoints.Checkpoint) {
    // Model metadata
    fmt.Printf("📋 Model Metadata:\n")
    fmt.Printf("   Framework: %s\n", checkpoint.Metadata.Framework)
    fmt.Printf("   Version: %s\n", checkpoint.Metadata.Version)
    fmt.Printf("   Description: %s\n", checkpoint.Metadata.Description)
    fmt.Printf("   Tags: %v\n", checkpoint.Metadata.Tags)
    fmt.Printf("   Created: %s\n", checkpoint.Metadata.CreatedAt.Format("2006-01-02 15:04:05"))
    
    // Training state
    fmt.Printf("\n📊 Training State:\n")
    fmt.Printf("   Epoch: %d\n", checkpoint.TrainingState.Epoch)
    fmt.Printf("   Step: %d\n", checkpoint.TrainingState.Step)
    fmt.Printf("   Learning Rate: %.6f\n", checkpoint.TrainingState.LearningRate)
    fmt.Printf("   Best Loss: %.6f\n", checkpoint.TrainingState.BestLoss)
    fmt.Printf("   Best Accuracy: %.2f%%\n", checkpoint.TrainingState.BestAccuracy*100)
    fmt.Printf("   Total Steps: %d\n", checkpoint.TrainingState.TotalSteps)
    
    // Model architecture
    fmt.Printf("\n🏗️ Model Architecture:\n")
    fmt.Printf("   Total Layers: %d\n", len(checkpoint.ModelSpec.Layers))
    fmt.Printf("   Total Parameters: %d\n", checkpoint.ModelSpec.TotalParameters)
    
    fmt.Printf("\n📚 Layer Details:\n")
    for i, layer := range checkpoint.ModelSpec.Layers {
        fmt.Printf("   Layer %d: %s\n", i+1, layer.Name)
        fmt.Printf("     Type: %s\n", layer.Type.String())
        fmt.Printf("     Input Shape: %v\n", layer.InputShape)
        fmt.Printf("     Output Shape: %v\n", layer.OutputShape)
        
        // Show parameters if available
        if len(layer.Parameters) > 0 {
            fmt.Printf("     Parameters: ")
            for key, value := range layer.Parameters {
                fmt.Printf("%s=%v ", key, value)
            }
            fmt.Println()
        }
        
        // Show parameter shapes if available
        if len(layer.ParameterShapes) > 0 {
            fmt.Printf("     Parameter Shapes: %v\n", layer.ParameterShapes)
        }
        
        fmt.Println()
    }
    
    // Weight analysis
    fmt.Printf("🔢 Weight Analysis:\n")
    fmt.Printf("   Total Weight Tensors: %d\n", len(checkpoint.Weights))
    
    totalParams := 0
    for _, weight := range checkpoint.Weights {
        paramCount := 1
        for _, dim := range weight.Shape {
            paramCount *= dim
        }
        totalParams += paramCount
        
        fmt.Printf("   %s (%s): %v - %d parameters\n", 
            weight.Name, weight.Type, weight.Shape, paramCount)
    }
    
    fmt.Printf("   Total Parameters: %d\n", totalParams)
    
    // Memory usage estimate
    memoryMB := float64(totalParams*4) / 1024.0 / 1024.0 // 4 bytes per float32
    fmt.Printf("   Estimated Memory: %.2f MB\n", memoryMB)
}
```

## 📊 Best Practices

### 1. Checkpoint Naming Convention

```go
// ✅ Good: Descriptive naming
filename := fmt.Sprintf("model_%s_epoch_%d_loss_%.4f.json", 
    modelName, epoch, loss)

// ✅ Good: Timestamped checkpoints
filename := fmt.Sprintf("checkpoint_%s.json", 
    time.Now().Format("20060102_150405"))

// ❌ Bad: Generic naming
filename := "model.json"
```

### 2. Checkpoint Frequency

```go
// ✅ Good: Save based on validation improvement
if validationLoss < bestValidationLoss {
    bestValidationLoss = validationLoss
    saveCheckpoint(checkpoint, "best_model.json")
}

// ✅ Good: Periodic saves
if epoch%saveFrequency == 0 {
    saveCheckpoint(checkpoint, fmt.Sprintf("epoch_%d.json", epoch))
}

// ❌ Bad: Save every step (too frequent)
if step%1 == 0 {
    saveCheckpoint(checkpoint, fmt.Sprintf("step_%d.json", step))
}
```

### 3. Checkpoint Cleanup

```go
// ✅ Good: Automatic cleanup
config := training.CheckpointConfig{
    MaxCheckpoints: 10, // Keep only 10 most recent
    SaveDirectory:  "./checkpoints",
}

// ✅ Good: Manual cleanup
func cleanupOldCheckpoints(directory string, maxFiles int) {
    // Implementation to remove old checkpoints
}
```

### 4. Error Handling

```go
// ✅ Good: Robust error handling
checkpoint, err := createCheckpoint(trainer, epoch, step, loss, accuracy)
if err != nil {
    log.Printf("Warning: Failed to create checkpoint: %v", err)
    continue // Don't stop training
}

if err := saver.SaveCheckpoint(checkpoint, path); err != nil {
    log.Printf("Warning: Failed to save checkpoint: %v", err)
    // Continue training, don't fail
}
```

## 🔍 Troubleshooting

### Common Issues

**Checkpoint Loading Fails**:
```go
// Check file exists and permissions
if _, err := os.Stat(checkpointPath); os.IsNotExist(err) {
    log.Printf("Checkpoint file not found: %s", checkpointPath)
    return
}

// Verify checkpoint format
loadedCheckpoint, err := saver.LoadCheckpoint(checkpointPath)
if err != nil {
    log.Printf("Failed to load checkpoint: %v", err)
    return
}
```

**Model Architecture Mismatch**:
```go
// Verify model compatibility before loading weights
if !modelsCompatible(currentModel, loadedCheckpoint.ModelSpec) {
    log.Printf("Model architectures are incompatible")
    return
}
```

**Memory Issues**:
```go
// Check available memory before loading large models
var memStats runtime.MemStats
runtime.ReadMemStats(&memStats)
fmt.Printf("Available memory: %.2f MB", float64(memStats.Sys)/1024/1024)
```

## 🚀 Next Steps

- **[Training Guide](../tutorials/mlp-tutorial.md)** - Complete training workflows
- **[Performance Guide](performance.md)** - Optimization techniques  
- **[Memory Management](memory-management.md)** - Efficient GPU memory usage
- **[Visualization Guide](visualization.md)** - Training progress monitoring

---

## 🧠 Key Takeaways

Go-metal checkpoints provide:

- **🔄 Complete State Preservation**: Model weights, training progress, and optimizer state
- **📦 Multiple Formats**: JSON for go-metal, ONNX for interoperability  
- **🤖 Automatic Management**: Periodic saves, best model tracking, and cleanup
- **🔄 Resume Training**: Seamless continuation of interrupted training
- **🔍 Model Inspection**: Detailed analysis of saved models

### Essential Workflow

1. **Configure checkpointing** during trainer setup
2. **Save periodically** and on improvement
3. **Load checkpoints** to resume training
4. **Export to ONNX** for deployment
5. **Clean up old checkpoints** to manage storage

### Inference and Deployment

For using saved models in production, see the [Inference Engine Guide](inference-engine.md) which covers:
- Loading checkpoints for inference
- Configuring the high-performance inference engine
- Running predictions on new data
- Optimizing inference performance

*Proper checkpointing ensures your training progress is never lost and enables flexible model management workflows.*