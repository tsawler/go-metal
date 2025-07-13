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
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

func demonstrateCrossEntropy() {
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Build model for 3-class classification
    inputShape := []int{8, 10}  // 8 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(3, true, "output").  // 3 classes, no softmax (handled by loss)
        Compile()
    
    // Configure CrossEntropy loss
    config := training.TrainerConfig{
        BatchSize:     8,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.CrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, _ := training.NewModelTrainer(model, config)
    defer trainer.Cleanup()
    
    // Create one-hot encoded labels
    // For classes [0, 1, 2, 1, 0, 2, 1, 0]
    labelData := []float32{
        1, 0, 0,  // Class 0
        0, 1, 0,  // Class 1  
        0, 0, 1,  // Class 2
        0, 1, 0,  // Class 1
        1, 0, 0,  // Class 0
        0, 0, 1,  // Class 2
        0, 1, 0,  // Class 1
        1, 0, 0,  // Class 0
    }
    
    fmt.Println("✅ CrossEntropy configured for one-hot labels")
    fmt.Println("   Input: Raw logits from model")
    fmt.Println("   Labels: One-hot encoded vectors")
    fmt.Println("   Output: Scalar loss value")
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
    // ... (same setup as CrossEntropy)
    
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
    
    trainer, _ := training.NewModelTrainer(model, config)
    defer trainer.Cleanup()
    
    // Create integer class labels (much simpler!)
    labelData := []int32{0, 1, 2, 1, 0, 2, 1, 0}  // Direct class indices
    
    fmt.Println("✅ SparseCrossEntropy configured for integer labels")
    fmt.Println("   Input: Raw logits from model")
    fmt.Println("   Labels: Integer class indices")
    fmt.Println("   Advantage: No need to one-hot encode")
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
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").     // Single output
        AddSigmoid("sigmoid").           // Convert to probability
        Compile()
    
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
    
    // Binary labels (0 or 1)
    labelData := []float32{1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0}
    
    fmt.Println("✅ BinaryCrossEntropy configured")
    fmt.Println("   Input: Probabilities (0-1) after sigmoid")
    fmt.Println("   Labels: Binary values (0 or 1)")
    fmt.Println("   Use case: Binary classification")
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
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").     // Single output (raw logits)
        // No sigmoid - BCEWithLogits handles it internally
        Compile()
    
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
    
    fmt.Println("✅ BCEWithLogits configured")
    fmt.Println("   Input: Raw logits (any real number)")
    fmt.Println("   Labels: Binary values (0 or 1)")
    fmt.Println("   Advantage: Numerically stable")
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
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Build regression model
    inputShape := []int{32, 10}  // 32 samples, 10 features
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(64, true, "hidden1").
        AddReLU("relu1").
        AddDense(32, true, "hidden2").
        AddReLU("relu2").
        AddDense(1, true, "output").     // Single continuous output
        // No activation - raw output for regression
        Compile()
    
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
    
    trainer, _ := training.NewModelTrainer(model, config)
    defer trainer.Cleanup()
    
    fmt.Println("✅ MSE configured for regression")
    fmt.Println("   Input: Continuous predictions")
    fmt.Println("   Targets: Continuous ground truth")
    fmt.Println("   Output: Mean squared difference")
    fmt.Println("   Use case: Standard regression")
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
    // ... (same model setup as MSE)
    
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
    
    fmt.Println("✅ MAE configured for robust regression")
    fmt.Println("   Input: Continuous predictions")
    fmt.Println("   Targets: Continuous ground truth")
    fmt.Println("   Output: Mean absolute difference")
    fmt.Println("   Advantage: Less sensitive to outliers")
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
```
Huber(δ) = {
    0.5 * (y_true - y_pred)²           if |y_true - y_pred| ≤ δ
    δ * |y_true - y_pred| - 0.5 * δ²   otherwise
}
```

**Example usage:**
```go
func demonstrateHuber() {
    // ... (same model setup as MSE)
    
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
    
    fmt.Println("✅ Huber loss configured")
    fmt.Println("   Input: Continuous predictions")
    fmt.Println("   Targets: Continuous ground truth")
    fmt.Println("   Behavior: MSE for small errors, MAE for large errors")
    fmt.Println("   Advantage: Best of both worlds")
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
```go
func labelFormatDecision() {
    fmt.Println("🎯 Label Format Decision Tree")
    
    fmt.Println("\n📊 Classification:")
    fmt.Println("   Have integer labels (0, 1, 2, ...) → SparseCrossEntropy")
    fmt.Println("   Have one-hot labels ([1,0,0], [0,1,0], ...) → CrossEntropy")
    fmt.Println("   Binary problem → BCEWithLogits")
    
    fmt.Println("\n📈 Regression:")
    fmt.Println("   Standard problem → MeanSquaredError")
    fmt.Println("   Have outliers → MeanAbsoluteError or Huber")
    fmt.Println("   Need robustness → Huber")
}
```

### Dataset Size Considerations

```go
func datasetSizeConsiderations() {
    fmt.Println("📊 Dataset Size Considerations")
    
    considerations := []struct {
        size string
        recommendation string
        reasoning string
    }{
        {
            "Small (< 1K samples)",
            "MSE/CrossEntropy",
            "Simple losses, avoid overfitting",
        },
        {
            "Medium (1K - 100K)",
            "Standard choices",
            "SparseCrossEntropy, MSE work well",
        },
        {
            "Large (> 100K)",
            "Robust losses",
            "Huber, MAE handle outliers better",
        },
        {
            "Noisy labels",
            "Robust losses",
            "MAE, Huber less sensitive to noise",
        },
    }
    
    fmt.Printf("%-20s | %-20s | %-30s\n", "Dataset Size", "Recommendation", "Reasoning")
    fmt.Println("---------------------|----------------------|------------------------------")
    
    for _, cons := range considerations {
        fmt.Printf("%-20s | %-20s | %-30s\n", 
                   cons.size, cons.recommendation, cons.reasoning)
    }
}
```

## 🔧 Practical Examples

### Complete Training Setup with Different Losses

#### Image Classification Example
```go
func imageClassificationExample() {
    // ... (device setup)
    
    // CNN for image classification
    inputShape := []int{32, 3, 32, 32}  // CIFAR-10 style
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddConv2D(32, 3, "conv1").AddReLU("relu1").
        AddConv2D(64, 3, "conv2").AddReLU("relu2").
        AddFlatten("flatten").
        AddDense(128, true, "dense").AddReLU("relu3").
        AddDense(10, true, "output").  // 10 classes
        Compile()
    
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
    
    fmt.Println("✅ Image classification with SparseCrossEntropy")
}
```

#### House Price Regression Example
```go
func housePriceRegressionExample() {
    // ... (device setup)
    
    // MLP for house price prediction
    inputShape := []int{100, 20}  // 100 houses, 20 features
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(128, true, "dense1").AddReLU("relu1").
        AddDense(64, true, "dense2").AddReLU("relu2").
        AddDense(32, true, "dense3").AddReLU("relu3").
        AddDense(1, true, "output").  // Price prediction
        Compile()
    
    // Huber loss for robust price prediction
    config := training.TrainerConfig{
        BatchSize:     100,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.Huber,  // Robust to outlier prices
        ProblemType:   training.Regression,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    fmt.Println("✅ House price regression with Huber loss")
}
```

#### Binary Sentiment Classification
```go
func sentimentClassificationExample() {
    // ... (device setup)
    
    // Text classification model
    inputShape := []int{64, 512}  // 64 reviews, 512 features
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(256, true, "dense1").AddReLU("relu1").
        AddDropout(0.5, "dropout1").
        AddDense(128, true, "dense2").AddReLU("relu2").
        AddDropout(0.3, "dropout2").
        AddDense(1, true, "output").  // Binary sentiment (raw logits)
        Compile()
    
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
    
    fmt.Println("✅ Sentiment classification with BCEWithLogits")
}
```

### Loss Function Comparison

```go
func compareLossFunctions() {
    fmt.Println("📊 Loss Function Comparison")
    
    losses := []struct {
        name string
        problem_type string
        input_format string
        label_format string
        use_case string
    }{
        {"CrossEntropy", "Classification", "Logits", "One-hot", "Multi-class"},
        {"SparseCrossEntropy", "Classification", "Logits", "Integers", "Multi-class (preferred)"},
        {"BinaryCrossEntropy", "Classification", "Probabilities", "Binary", "Binary classification"},
        {"BCEWithLogits", "Classification", "Logits", "Binary", "Binary (stable)"},
        {"MeanSquaredError", "Regression", "Continuous", "Continuous", "Standard regression"},
        {"MeanAbsoluteError", "Regression", "Continuous", "Continuous", "Robust regression"},
        {"Huber", "Regression", "Continuous", "Continuous", "Balanced robustness"},
    }
    
    fmt.Printf("%-18s | %-14s | %-13s | %-10s | %-20s\n",
               "Loss Function", "Problem Type", "Input Format", "Labels", "Use Case")
    fmt.Println("-------------------|----------------|---------------|------------|--------------------")
    
    for _, loss := range losses {
        fmt.Printf("%-18s | %-14s | %-13s | %-10s | %-20s\n",
                   loss.name, loss.problem_type, loss.input_format, 
                   loss.label_format, loss.use_case)
    }
}
```

## 🎓 Advanced Topics

### Loss Function Properties

```go
func lossFunctionProperties() {
    fmt.Println("🔍 Loss Function Properties")
    
    fmt.Println("\n📊 Classification Losses:")
    fmt.Println("   • CrossEntropy: Probabilistic, smooth gradients")
    fmt.Println("   • SparseCrossEntropy: Memory efficient, same math")
    fmt.Println("   • BCEWithLogits: Numerically stable sigmoid+BCE")
    
    fmt.Println("\n📈 Regression Losses:")
    fmt.Println("   • MSE: Smooth, sensitive to outliers, L2 norm")
    fmt.Println("   • MAE: Robust, less smooth gradients, L1 norm")
    fmt.Println("   • Huber: Adaptive, combines MSE+MAE benefits")
    
    fmt.Println("\n⚡ Gradient Properties:")
    fmt.Println("   • MSE: Linear gradients (proportional to error)")
    fmt.Println("   • MAE: Constant gradients (sign of error)")
    fmt.Println("   • Huber: Smooth transition between MSE and MAE")
}
```

### Numerical Stability Considerations

```go
func numericalStabilityConsiderations() {
    fmt.Println("🔒 Numerical Stability Considerations")
    
    fmt.Println("\n⚠️ Potential Issues:")
    fmt.Println("   • Large logits → overflow in softmax")
    fmt.Println("   • Very small probabilities → log(0) = -∞")
    fmt.Println("   • Extreme predictions → gradient explosion")
    
    fmt.Println("\n✅ Go-Metal Safeguards:")
    fmt.Println("   • Automatic logit clipping")
    fmt.Println("   • Numerically stable implementations")
    fmt.Println("   • MPSGraph optimizations")
    
    fmt.Println("\n🎯 Best Practices:")
    fmt.Println("   • Use BCEWithLogits over BinaryCrossEntropy + Sigmoid")
    fmt.Println("   • Avoid extreme learning rates")
    fmt.Println("   • Monitor loss values for NaN/Inf")
    fmt.Println("   • Use appropriate loss scaling for mixed precision")
}
```

### Custom Loss Function Concepts

```go
func customLossConcepts() {
    fmt.Println("🛠️ Custom Loss Function Concepts")
    
    fmt.Println("\n🎯 When You Might Need Custom Losses:")
    fmt.Println("   • Domain-specific objectives")
    fmt.Println("   • Multi-task learning")
    fmt.Println("   • Imbalanced dataset handling")
    fmt.Println("   • Specialized metrics optimization")
    
    fmt.Println("\n📝 Implementation Notes:")
    fmt.Println("   • Go-metal uses built-in MPSGraph loss functions")
    fmt.Println("   • Custom losses would require C/Objective-C extension")
    fmt.Println("   • Current losses cover 95% of use cases")
    
    fmt.Println("\n💡 Alternatives:")
    fmt.Println("   • Weighted sampling for imbalanced data")
    fmt.Println("   • Data augmentation for bias correction")
    fmt.Println("   • Post-processing for specialized metrics")
}
```

## 🎯 Quick Reference

### Loss Function Quick Selection

```go
func quickLossSelection() {
    fmt.Println("⚡ Quick Loss Function Selection")
    
    fmt.Println("\n🎯 Default Choices:")
    fmt.Println("   Multi-class classification → SparseCrossEntropy")
    fmt.Println("   Binary classification → BCEWithLogits")
    fmt.Println("   Regression → MeanSquaredError")
    fmt.Println("   Robust regression → Huber")
    
    fmt.Println("\n⚙️ Configuration Templates:")
    
    fmt.Println("\n// Multi-class classification")
    fmt.Println(`config := training.TrainerConfig{
    LossFunction: training.SparseCrossEntropy,
    ProblemType:  training.Classification,
}`)
    
    fmt.Println("\n// Binary classification")
    fmt.Println(`config := training.TrainerConfig{
    LossFunction: training.BCEWithLogits,
    ProblemType:  training.Classification,
}`)
    
    fmt.Println("\n// Regression")
    fmt.Println(`config := training.TrainerConfig{
    LossFunction: training.MeanSquaredError,
    ProblemType:  training.Regression,
}`)
}
```

### Debugging Loss Values

```go
func debuggingLossValues() {
    fmt.Println("🔍 Debugging Loss Values")
    
    fmt.Println("\n✅ Healthy Loss Patterns:")
    fmt.Println("   • Decreasing trend over epochs")
    fmt.Println("   • Smooth convergence (not jagged)")
    fmt.Println("   • Reasonable final values")
    
    fmt.Println("\n⚠️ Warning Signs:")
    fmt.Println("   • Loss = NaN → learning rate too high")
    fmt.Println("   • Loss = 0 exactly → potential bug")
    fmt.Println("   • Loss increases → wrong loss function or LR")
    fmt.Println("   • Loss plateaus immediately → data/model issue")
    
    fmt.Println("\n🔧 Typical Loss Ranges:")
    fmt.Println("   • CrossEntropy: 0.1-3.0 (lower is better)")
    fmt.Println("   • MSE: depends on target scale")
    fmt.Println("   • MAE: depends on target scale")
    fmt.Println("   • Huber: between MSE and MAE")
}
```

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