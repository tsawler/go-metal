# Loss Functions Guide

Complete reference for all loss functions available in go-metal, with practical usage examples and selection guidance.

## 🎯 Overview

Loss functions in go-metal measure how well your model's predictions match the true targets. They provide the gradient signal that drives learning during training. Choosing the right loss function is crucial for effective model training.

## 📊 Classification Loss Functions

### CrossEntropy
**Standard loss for multi-class classification with one-hot encoded labels.**

```go
config := training.TrainerConfig{
    LossFunction: training.CrossEntropy,
    ProblemType:  training.Classification,
}
```

**Input format:**
- **Predictions**: Raw logits `[batch_size, num_classes]`
- **Labels**: One-hot vectors `[batch_size, num_classes]`

**Mathematical definition:**
```
CrossEntropy = -∑(y_true * log(softmax(y_pred)))
```

**Example usage:**
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
    fmt.Println("🔍 CrossEntropy Loss Demo")
    
    // Build model for 3-class classification
    inputShape := []int{8, 10}  // 8 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(3, true, "output").  // 3 classes, no softmax (handled by loss)
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Configure CrossEntropy loss
    config := training.TrainerConfig{
        BatchSize:     8,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.CrossEntropy,  // CrossEntropy for integer labels
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers for better performance
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 8*10)  // 8 samples, 10 features
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0  // Random values [-1, 1]
    }
    
    // Create integer class labels (NOT one-hot)
    // For classes [0, 1, 2, 1, 0, 2, 1, 0]
    labelData := []int32{0, 1, 2, 1, 0, 2, 1, 0}
    
    // Convert to Int32Labels for unified API
    labels, err := training.NewInt32Labels(labelData, []int{8})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ CrossEntropy configured for integer class labels")
    fmt.Println("   Input: Raw logits from model")
    fmt.Println("   Labels: Integer class indices [0, 1, 2, 1, 0, 2, 1, 0]")
    fmt.Println("   Output: Scalar loss value")
    fmt.Println("   Note: Softmax is applied internally by the loss function")
    
    // Actually run training to demonstrate CrossEntropy
    fmt.Println("\n🚀 Training with CrossEntropy loss:")
    fmt.Println("Step | Loss")
    fmt.Println("-----|--------")
    
    for step := 1; step <= 5; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("%4d | %.4f\n", step, result.Loss)
    }
    
    fmt.Println("\n✅ Successfully demonstrated CrossEntropy loss!")
    fmt.Println("   The loss decreases as the model learns to predict class indices")
    fmt.Println("   CrossEntropy automatically applies softmax to convert logits to probabilities")
}
```

**Best for:**
- ✅ Multi-class classification
- ✅ When you have one-hot encoded labels
- ✅ Balanced datasets
- ✅ Standard neural network outputs

### SparseCrossEntropy
**Cross-entropy loss for integer class labels (most convenient).**

```go
config := training.TrainerConfig{
    LossFunction: training.SparseCrossEntropy,
    ProblemType:  training.Classification,
}
```

**Input format:**
- **Predictions**: Raw logits `[batch_size, num_classes]`
- **Labels**: Integer class indices `[batch_size]`

**Mathematical definition:**
```
SparseCrossEntropy = -log(softmax(y_pred)[y_true])
```

**Example usage:**
```go
func demonstrateSparseCrossEntropy() {
    // Build model for 3-class classification
    inputShape := []int{8, 10}  // 8 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(3, true, "output").  // 3 classes
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Configure SparseCrossEntropy loss
    config := training.TrainerConfig{
        BatchSize:     8,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,  // Key difference
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 8*10)
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0
    }
    
    // Create integer class labels (much simpler!)
    labelData := []int32{0, 1, 2, 1, 0, 2, 1, 0}  // Direct class indices
    
    labels, err := training.NewInt32Labels(labelData, []int{8})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ SparseCrossEntropy configured for integer labels")
    fmt.Println("   Input: Raw logits from model")
    fmt.Println("   Labels: Integer class indices")
    fmt.Println("   Advantage: No need to one-hot encode")
    
    // Test with actual training
    for step := 1; step <= 3; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
}
```

**Best for:**
- ✅ Multi-class classification (most common choice)
- ✅ When you have integer class labels
- ✅ Memory efficiency (no one-hot encoding)
- ✅ Most convenient for typical classification

### BinaryCrossEntropy
**Binary classification with probability inputs.**

```go
config := training.TrainerConfig{
    LossFunction: training.BinaryCrossEntropy,
    ProblemType:  training.Classification,
}
```

**Input format:**
- **Predictions**: Probabilities `[batch_size, 1]` (after sigmoid)
- **Labels**: Binary targets `[batch_size, 1]` (0 or 1)

**Mathematical definition:**
```
BinaryCrossEntropy = -(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
```

**Example usage:**
```go
func demonstrateBinaryCrossEntropy() {
    // Build model for binary classification
    inputShape := []int{16, 8}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").     // Single output
        AddSigmoid("sigmoid").           // Convert to probability
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     16,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.BinaryCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 16*8)
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0
    }
    
    // Binary labels as integers (0 or 1)
    labelData := []int32{1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0}
    
    labels, err := training.NewInt32Labels(labelData, []int{16})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ BinaryCrossEntropy configured")
    fmt.Println("   Input: Probabilities (0-1) after sigmoid")
    fmt.Println("   Labels: Binary values (0 or 1) as integers")
    fmt.Println("   Use case: Binary classification")
    
    // Test with actual training
    for step := 1; step <= 3; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
}
```

**Best for:**
- ✅ Binary classification problems
- ✅ When model outputs probabilities
- ✅ Two-class problems (spam/not spam, etc.)

### BCEWithLogits
**Binary classification with raw logits (numerically stable).**

```go
config := training.TrainerConfig{
    LossFunction: training.BCEWithLogits,
    ProblemType:  training.Classification,
}
```

**Input format:**
- **Predictions**: Raw logits `[batch_size, 1]` (before sigmoid)
- **Labels**: Binary targets `[batch_size, 1]` (0 or 1)

**Mathematical definition:**
```
BCEWithLogits = max(x, 0) - x * z + log(1 + exp(-abs(x)))
where x = logits, z = targets
```

**Example usage:**
```go
func demonstrateBCEWithLogits() {
    // Build model for binary classification
    inputShape := []int{16, 8}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").     // Single output (raw logits)
        // No sigmoid - BCEWithLogits handles it internally
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     16,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.BCEWithLogits,  // More stable than BinaryCrossEntropy
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 16*8)
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0
    }
    
    // Binary labels as integers (0 or 1)
    labelData := []int32{1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0}
    
    labels, err := training.NewInt32Labels(labelData, []int{16})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ BCEWithLogits configured")
    fmt.Println("   Input: Raw logits (any real number)")
    fmt.Println("   Labels: Binary values (0 or 1)")
    fmt.Println("   Advantage: Numerically stable")
    
    // Test with actual training
    for step := 1; step <= 3; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
}
```

**Best for:**
- ✅ Binary classification (preferred over BinaryCrossEntropy)
- ✅ Numerical stability
- ✅ When you want to avoid explicit sigmoid layer

### CategoricalCrossEntropy
**Alternative to CrossEntropy for multi-class problems.**

```go
config := training.TrainerConfig{
    LossFunction: training.CategoricalCrossEntropy,
    ProblemType:  training.Classification,
}
```

**Similar to CrossEntropy but may have different numerical implementation details.**

**Best for:**
- ✅ Multi-class classification
- ✅ When CrossEntropy doesn't meet specific needs
- ✅ Framework compatibility

## 📈 Regression Loss Functions

### MeanSquaredError (MSE)
**Standard loss for regression problems.**

```go
config := training.TrainerConfig{
    LossFunction: training.MeanSquaredError,
    ProblemType:  training.Regression,
}
```

**Input format:**
- **Predictions**: Continuous values `[batch_size, output_dim]`
- **Targets**: Continuous ground truth `[batch_size, output_dim]`

**Mathematical definition:**
```
MSE = (1/n) * ∑(y_true - y_pred)²
```

**Example usage:**
```go
func demonstrateMSE() {
    // Build regression model
    inputShape := []int{32, 10}  // 32 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(64, true, "hidden1").
        AddReLU("relu1").
        AddDense(32, true, "hidden2").
        AddReLU("relu2").
        AddDense(1, true, "output").     // Single continuous output
        // No activation - raw output for regression
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Configure MSE loss
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.MeanSquaredError,
        ProblemType:   training.Regression,  // Important!
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 32*10)
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0
    }
    
    // Create regression targets
    labelData := make([]float32, 32)
    for i := range labelData {
        labelData[i] = rand.Float32() * 100.0  // Target values 0-100
    }
    
    labels, err := training.NewFloat32Labels(labelData, []int{32, 1})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ MSE configured for regression")
    fmt.Println("   Input: Continuous predictions")
    fmt.Println("   Targets: Continuous ground truth")
    fmt.Println("   Output: Mean squared difference")
    fmt.Println("   Use case: Standard regression")
    
    // Test with actual training
    for step := 1; step <= 3; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
}
```

**Properties:**
- Penalizes large errors heavily (quadratic)
- Sensitive to outliers
- Units: squared units of target variable
- Always non-negative

**Best for:**
- ✅ Standard regression problems
- ✅ When large errors should be heavily penalized
- ✅ Normally distributed residuals
- ✅ When you need smooth gradients

### MeanAbsoluteError (MAE)
**Robust regression loss less sensitive to outliers.**

```go
config := training.TrainerConfig{
    LossFunction: training.MeanAbsoluteError,
    ProblemType:  training.Regression,
}
```

**Mathematical definition:**
```
MAE = (1/n) * ∑|y_true - y_pred|
```

**Example usage:**
```go
func demonstrateMAE() {
    // Build regression model (same as MSE)
    inputShape := []int{32, 10}  // 32 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(64, true, "hidden1").
        AddReLU("relu1").
        AddDense(32, true, "hidden2").
        AddReLU("relu2").
        AddDense(1, true, "output").     // Single continuous output
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Configure MAE loss
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.MeanAbsoluteError,
        ProblemType:   training.Regression,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 32*10)
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0
    }
    
    // Create regression targets
    labelData := make([]float32, 32)
    for i := range labelData {
        labelData[i] = rand.Float32() * 100.0  // Target values 0-100
    }
    
    labels, err := training.NewFloat32Labels(labelData, []int{32, 1})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ MAE configured for robust regression")
    fmt.Println("   Input: Continuous predictions")
    fmt.Println("   Targets: Continuous ground truth")
    fmt.Println("   Output: Mean absolute difference")
    fmt.Println("   Advantage: Less sensitive to outliers")
    
    // Test with actual training
    for step := 1; step <= 3; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
}
```

**Properties:**
- Linear penalty for errors
- Robust to outliers
- Units: same units as target variable
- Less smooth gradients than MSE

**Best for:**
- ✅ Datasets with outliers
- ✅ When all errors should be weighted equally
- ✅ Robust regression
- ✅ When interpretability matters

### Huber Loss
**Combines benefits of MSE and MAE.**

```go
config := training.TrainerConfig{
    LossFunction: training.Huber,
    ProblemType:  training.Regression,
}
```

**Mathematical definition:**
- **Huber(δ) = 0.5 × (y_true - y_pred)²** if |y_true - y_pred| ≤ δ
- **Huber(δ) = δ × |y_true - y_pred| - 0.5 × δ²** otherwise

**Example usage:**
```go
func demonstrateHuber() {
    // Build regression model (same as MSE)
    inputShape := []int{32, 10}  // 32 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(64, true, "hidden1").
        AddReLU("relu1").
        AddDense(32, true, "hidden2").
        AddReLU("relu2").
        AddDense(1, true, "output").     // Single continuous output
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Configure Huber loss
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.Huber,
        ProblemType:   training.Regression,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    // Note: Huber loss may have gradient computation issues in current version
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Println("✅ Huber loss configured")
    fmt.Println("   Input: Continuous predictions")
    fmt.Println("   Targets: Continuous ground truth")
    fmt.Println("   Behavior: MSE for small errors, MAE for large errors")
    fmt.Println("   Advantage: Best of both worlds")
    fmt.Println("   Note: Currently may have gradient computation issues")
}
```

**Properties:**
- Quadratic for small errors (δ threshold)
- Linear for large errors
- Robust to outliers while maintaining smoothness
- Parameter δ controls the transition point

**Best for:**
- ✅ Datasets with some outliers
- ✅ When you want smooth gradients AND robustness
- ✅ Time series prediction
- ✅ Computer vision regression tasks

## 🎯 Loss Function Selection Guide

### Problem Type Recommendations

#### Multi-Class Classification
```go
// Most common choice (integer labels)
config := training.TrainerConfig{
    LossFunction: training.SparseCrossEntropy,
    ProblemType:  training.Classification,
}

// When you have one-hot labels
config := training.TrainerConfig{
    LossFunction: training.CrossEntropy,
    ProblemType:  training.Classification,
}
```

#### Binary Classification
```go
// Recommended (numerically stable)
config := training.TrainerConfig{
    LossFunction: training.BCEWithLogits,
    ProblemType:  training.Classification,
}

// Alternative (if using sigmoid activation)
config := training.TrainerConfig{
    LossFunction: training.BinaryCrossEntropy,
    ProblemType:  training.Classification,
}
```

#### Standard Regression
```go
// Default choice
config := training.TrainerConfig{
    LossFunction: training.MeanSquaredError,
    ProblemType:  training.Regression,
}
```

#### Robust Regression (with outliers)
```go
// For datasets with outliers
config := training.TrainerConfig{
    LossFunction: training.MeanAbsoluteError,
    ProblemType:  training.Regression,
}

// Best of both worlds
config := training.TrainerConfig{
    LossFunction: training.Huber,
    ProblemType:  training.Regression,
}
```

### Data Format Considerations

#### Label Format Decision Tree

**📊 Classification:**
- Have integer labels (0, 1, 2, ...) → **SparseCrossEntropy**
- Have one-hot labels ([1,0,0], [0,1,0], ...) → **CrossEntropy**
- Binary problem → **BCEWithLogits**

**📈 Regression:**
- Standard problem → **MeanSquaredError**
- Have outliers → **MeanAbsoluteError** or **Huber**
- Need robustness → **Huber**

### Dataset Size Considerations

| Dataset Size         | Recommendation     | Reasoning                      |
|----------------------|--------------------|--------------------------------|
| Small (< 1K samples) | MSE/CrossEntropy   | Simple losses, avoid overfitting |
| Medium (1K - 100K)   | Standard choices   | SparseCrossEntropy, MSE work well |
| Large (> 100K)       | Robust losses      | Huber, MAE handle outliers better |
| Noisy labels         | Robust losses      | MAE, Huber less sensitive to noise |

## 🔧 Practical Examples

### Complete Training Setup with Different Losses

#### Image Classification Example
```go
func imageClassificationExample() {
    // CNN for image classification
    inputShape := []int{32, 3, 32, 32}  // CIFAR-10 style
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(32, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddConv2D(64, 3, 1, 1, true, "conv2").
        AddReLU("relu2").
        AddDense(128, true, "dense").  // Dense automatically handles flattening
        AddReLU("relu3").
        AddDense(10, true, "output").  // 10 classes
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // SparseCrossEntropy for integer labels
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,  // Perfect for image classification
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Println("✅ Image classification with SparseCrossEntropy")
    fmt.Printf("   Model: CNN with %d parameters\n", model.TotalParameters)
    fmt.Printf("   Input: %v (CIFAR-10 style)\n", inputShape)
    fmt.Printf("   Output: 10 classes\n")
}
```

#### House Price Regression Example
```go
func housePriceRegressionExample() {
    // MLP for house price prediction
    inputShape := []int{100, 20}  // 100 houses, 20 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(128, true, "dense1").
        AddReLU("relu1").
        AddDense(64, true, "dense2").
        AddReLU("relu2").
        AddDense(32, true, "dense3").
        AddReLU("relu3").
        AddDense(1, true, "output").  // Price prediction
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // MSE loss for price prediction (Huber has current issues)
    config := training.TrainerConfig{
        BatchSize:     100,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.MeanSquaredError,  // Standard regression loss
        ProblemType:   training.Regression,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Println("✅ House price regression with MSE loss")
    fmt.Printf("   Model: MLP with %d parameters\n", model.TotalParameters)
    fmt.Printf("   Input: %v (100 houses, 20 features)\n", inputShape)
    fmt.Printf("   Output: Price prediction\n")
}
```

#### Binary Sentiment Classification
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

func sentimentClassificationExample() {
    fmt.Println("🔍 Binary Sentiment Classification Demo")
    
    // Text classification model
    inputShape := []int{64, 512}  // 64 reviews, 512 features
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(256, true, "dense1").
        AddReLU("relu1").
        AddDense(128, true, "dense2").
        AddReLU("relu2").
        AddDense(1, true, "output").  // Binary sentiment (raw logits)
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // BCEWithLogits for stable binary classification
    config := training.TrainerConfig{
        BatchSize:     64,
        LearningRate:  0.0005,  // Lower LR for text
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.BCEWithLogits,  // Stable binary classification
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Create sample input data
    inputData := make([]float32, 64*512)
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0
    }
    
    // Create binary sentiment labels (0 = negative, 1 = positive)
    labelData := make([]int32, 64)
    for i := range labelData {
        labelData[i] = int32(rand.Intn(2))  // Random 0 or 1
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{64})
    if err != nil {
        log.Fatalf("Failed to create labels: %v", err)
    }
    
    fmt.Println("✅ Sentiment classification with BCEWithLogits")
    fmt.Printf("   Input: %v (64 reviews, 512 features)\n", inputShape)
    fmt.Printf("   Output: Binary sentiment (logits)\n")
    fmt.Printf("   Loss: BCEWithLogits (numerically stable)\n")
    
    // Test with actual training
    for step := 1; step <= 3; step++ {
        result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
}

func main() {
    sentimentClassificationExample()
}
```

### Loss Function Comparison

| Loss Function      | Problem Type   | Input Format  | Labels     | Use Case            |
|--------------------|----------------|---------------|------------|---------------------|
| CrossEntropy       | Classification | Logits        | One-hot    | Multi-class         |
| SparseCrossEntropy | Classification | Logits        | Integers   | Multi-class (preferred) |
| BinaryCrossEntropy | Classification | Probabilities | Binary     | Binary classification |
| BCEWithLogits      | Classification | Logits        | Binary     | Binary (stable)     |
| MeanSquaredError   | Regression     | Continuous    | Continuous | Standard regression |
| MeanAbsoluteError  | Regression     | Continuous    | Continuous | Robust regression   |
| Huber              | Regression     | Continuous    | Continuous | Balanced robustness |

## 🎓 Advanced Topics

### Loss Function Properties

**📊 Classification Losses:**
- **CrossEntropy**: Probabilistic, smooth gradients
- **SparseCrossEntropy**: Memory efficient, same math  
- **BCEWithLogits**: Numerically stable sigmoid+BCE

**📈 Regression Losses:**
- **MSE**: Smooth, sensitive to outliers, L2 norm
- **MAE**: Robust, less smooth gradients, L1 norm
- **Huber**: Adaptive, combines MSE+MAE benefits

**⚡ Gradient Properties:**
- **MSE**: Linear gradients (proportional to error)
- **MAE**: Constant gradients (sign of error)
- **Huber**: Smooth transition between MSE and MAE

### Numerical Stability Considerations

**⚠️ Potential Issues:**
- Large logits → overflow in softmax
- Very small probabilities → log(0) = -∞
- Extreme predictions → gradient explosion

**✅ Go-Metal Safeguards:**
- Automatic logit clipping
- Numerically stable implementations
- MPSGraph optimizations

**🎯 Best Practices:**
- Use BCEWithLogits over BinaryCrossEntropy + Sigmoid
- Avoid extreme learning rates
- Monitor loss values for NaN/Inf
- Use appropriate loss scaling for mixed precision

### Custom Loss Function Concepts

**🎯 When You Might Need Custom Losses:**
- Domain-specific objectives
- Multi-task learning
- Imbalanced dataset handling
- Specialized metrics optimization

**📝 Implementation Notes:**
- Go-metal uses built-in MPSGraph loss functions
- Custom losses would require C/Objective-C extension
- Current losses cover 95% of use cases

**💡 Alternatives:**
- Weighted sampling for imbalanced data
- Data augmentation for bias correction
- Post-processing for specialized metrics

## 🎯 Quick Reference

### Loss Function Quick Selection

**🎯 Default Choices:**
- Multi-class classification → SparseCrossEntropy
- Binary classification → BCEWithLogits
- Regression → MeanSquaredError
- Robust regression → Huber

**⚙️ Configuration Templates:**

Multi-class classification:
```go
config := training.TrainerConfig{
    LossFunction: training.SparseCrossEntropy,
    ProblemType:  training.Classification,
}
```

Binary classification:
```go
config := training.TrainerConfig{
    LossFunction: training.BCEWithLogits,
    ProblemType:  training.Classification,
}
```

Regression:
```go
config := training.TrainerConfig{
    LossFunction: training.MeanSquaredError,
    ProblemType:  training.Regression,
}
```

### Debugging Loss Values

**✅ Healthy Loss Patterns:**
- Decreasing trend over epochs
- Smooth convergence (not jagged)
- Reasonable final values

**⚠️ Warning Signs:**
- Loss = NaN → learning rate too high
- Loss = 0 exactly → potential bug
- Loss increases → wrong loss function or LR
- Loss plateaus immediately → data/model issue

**🔧 Typical Loss Ranges:**
- CrossEntropy: 0.1-3.0 (lower is better)
- MSE: depends on target scale
- MAE: depends on target scale
- Huber: between MSE and MAE

## 🚀 Next Steps

Master loss functions in go-metal:

- **[Optimizers Guide](optimizers.md)** - Combine losses with optimizers
- **[Regression Tutorial](../tutorials/regression-tutorial.md)** - Apply regression losses
- **[Performance Guide](performance.md)** - Optimize loss computation
- **[Examples](../examples/)** - See losses in complete projects

**Ready for advanced training?** Check out the [Mixed Precision Tutorial](../tutorials/mixed-precision.md) for combining loss functions with FP16 training.

---

## 🧠 Key Takeaways

- **SparseCrossEntropy is the default**: Best for most classification problems
- **Match loss to problem type**: Classification vs regression determines loss family
- **Consider data properties**: Outliers suggest robust losses (MAE, Huber)
- **Numerical stability matters**: Use BCEWithLogits over manual sigmoid+BCE
- **Go-metal advantages**: Optimized implementations, automatic stability, GPU-resident computation

Understanding loss functions helps you train models effectively and achieve better results on Apple Silicon!