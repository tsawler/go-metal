# Optimizer Guide

Complete reference for all optimizers available in go-metal, with practical usage examples and tuning advice.

## 🎯 Overview

Optimizers in go-metal determine how your model learns from data by updating parameters based on computed gradients. Each optimizer has different strengths and is suited for different types of problems.

## 🧠 Available Optimizers

### First-Order Optimizers

**✨ Newly Integrated with Unified Training System**: AdaGrad, AdaDelta, and Nadam are now fully integrated with `ModelTrainingEngine`, providing seamless GPU-resident optimization with single CGO calls and automatic MPSGraph fusion.

#### Adam (Adaptive Moment Estimation)
**Most popular and versatile optimizer for deep learning.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.001,  // Standard starting point
    Beta1:         0.9,    // Momentum decay rate
    Beta2:         0.999,  // Squared gradient decay rate  
    Epsilon:       1e-8,   // Numerical stability
}
```

**How it works:**
- Combines momentum (Beta1) with adaptive learning rates (Beta2)
- Maintains exponential moving averages of gradients and squared gradients
- Automatically adjusts learning rate per parameter

**Best for:**
- ✅ Most deep learning problems
- ✅ Neural networks with many parameters
- ✅ Problems with sparse gradients
- ✅ When you want "set and forget" optimization

**Tuning tips:**
```go
// Conservative (stable training)
Beta1: 0.9, Beta2: 0.999, LearningRate: 0.0001

// Aggressive (faster convergence)
Beta1: 0.9, Beta2: 0.99, LearningRate: 0.003

// For very large models
Beta1: 0.9, Beta2: 0.999, LearningRate: 0.00001
```

#### SGD (Stochastic Gradient Descent)
**Simple and reliable, especially with momentum.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.SGD,
    LearningRate:  0.01,   // Higher than Adam typically
    // Note: Momentum support depends on implementation
}
```

**How it works:**
- Direct gradient descent: `weights = weights - learning_rate * gradients`
- Simple but effective with proper learning rate scheduling

**Best for:**
- ✅ Simple problems and small models
- ✅ When interpretability is important
- ✅ Fine-tuning pre-trained models
- ✅ When you want predictable behavior

**Tuning tips:**
```go
// For small models
LearningRate: 0.1

// For deep networks  
LearningRate: 0.01

// For fine-tuning
LearningRate: 0.001
```

#### RMSProp (Root Mean Square Propagation)
**Good for non-stationary objectives and RNNs.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.RMSProp,
    LearningRate:  0.001,
    // Additional RMSProp parameters handled internally
}
```

**How it works:**
- Maintains moving average of squared gradients
- Divides gradient by root mean square of recent gradients
- Helps with varying gradient scales

**Best for:**
- ✅ Recurrent neural networks (RNNs)
- ✅ Problems with varying gradient magnitudes
- ✅ Online learning scenarios
- ✅ Non-stationary objectives

#### AdaGrad (Adaptive Gradient Algorithm)
**Adapts learning rate based on historical gradients - excellent for sparse features.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.AdaGrad,
    LearningRate:  0.01,      // Can start higher than Adam
    Epsilon:      1e-10,      // Numerical stability (smaller than Adam)
    WeightDecay:  0.001,      // L2 regularization
    EngineType:   training.Dynamic,  // Full unified training support
}
```

**How it works:**
- Accumulates squared gradients over time: `G_t = G_{t-1} + g_t²`
- Adaptive per-parameter learning rates: `θ_t = θ_{t-1} - (α/√(G_t + ε)) * g_t`
- Automatically reduces learning rate for frequently updated parameters
- Perfect for sparse data where features have vastly different update frequencies

**Best for:**
- ✅ Sparse data and features (NLP, recommendation systems)
- ✅ Natural language processing with varying word frequencies
- ✅ When parameters have very different update frequencies
- ✅ Problems with infrequent but important features
- ✅ Convex optimization problems

**Unified Training Integration:**
```go
// Complete example with ModelTrainingEngine
trainer, err := training.NewModelTrainer(model, config)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// Single call handles: forward pass + AdaGrad optimization + backward pass
result, err := trainer.TrainStep(inputTensor, labelTensor)
```

**Architecture Benefits:**
- **GPU-Resident**: All AdaGrad state (accumulated gradients) stays on GPU
- **Single CGO Call**: Complete training step in one optimized call
- **MPSGraph Integration**: Automatic kernel fusion with other operations
- **Memory Efficient**: Buffer pooling and automatic cleanup

**Caution:**
- Learning rate can become too small over time
- May stop learning in very long training sessions
- Not ideal for non-stationary objectives

#### AdaDelta 
**Extension of AdaGrad that addresses learning rate decay - no manual learning rate tuning required.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.AdaDelta,
    LearningRate:  1.0,      // Set to 1.0 - AdaDelta handles scaling automatically
    Alpha:         0.95,     // Rho parameter (decay rate for moving averages)
    Epsilon:       1e-6,     // Numerical stability
    WeightDecay:   0.005,    // L2 regularization
    EngineType:    training.Dynamic,  // Full unified training support
}
```

**How it works:**
- Uses exponential moving averages instead of accumulating all history: `E[g²]_t = ρ·E[g²]_{t-1} + (1-ρ)·g_t²`
- Also tracks moving average of parameter updates: `E[Δx²]_t = ρ·E[Δx²]_{t-1} + (1-ρ)·Δx_t²`
- Automatically adapts learning rate: `Δx_t = -(√(E[Δx²]_{t-1} + ε) / √(E[g²]_t + ε)) · g_t`
- No manual learning rate tuning required - algorithm determines appropriate scale

**Best for:**
- ✅ When you want robust optimization without hyperparameter tuning
- ✅ Long training sessions (avoids AdaGrad's diminishing returns)
- ✅ Robust optimization across different problem types and scales
- ✅ When you don't know the appropriate learning rate
- ✅ Non-stationary objectives

**Unified Training Integration:**
```go
// Complete example with ModelTrainingEngine
trainer, err := training.NewModelTrainer(model, config)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// Single call handles: forward pass + AdaDelta optimization + backward pass
result, err := trainer.TrainStep(inputTensor, labelTensor)

// Note: UpdateLearningRate() will return error for AdaDelta 
// since it adapts learning rate automatically
```

**Architecture Benefits:**
- **GPU-Resident**: All AdaDelta state (gradient and update averages) stays on GPU
- **Single CGO Call**: Complete training step in one optimized call
- **MPSGraph Integration**: Automatic kernel fusion with dual accumulator updates
- **Memory Efficient**: Proper buffer pooling for both moving averages

**Key Advantages:**
- Addresses AdaGrad's learning rate decay problem
- No learning rate hyperparameter tuning needed
- Robust across different problem types
- Consistent performance throughout long training runs

#### Nadam (Nesterov-accelerated Adam)
**Combines Adam's adaptive learning rates with Nesterov momentum for superior convergence.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Nadam,
    LearningRate:  0.002,    // Typically slightly higher than Adam
    Beta1:         0.9,      // Momentum decay rate
    Beta2:         0.999,    // Squared gradient decay rate
    Epsilon:       1e-8,     // Numerical stability
    WeightDecay:   0.01,     // L2 regularization for better generalization
    EngineType:    training.Dynamic,  // Full unified training support
}
```

**How it works:**
- Combines Adam's adaptive learning rates with Nesterov momentum acceleration
- Uses momentum schedule: `μ_t = β₁ × (1 - 0.5 × 0.96^(t × 0.004))`
- Nesterov-accelerated momentum: `m̂_t = μ_{t+1} × m̂_t + (1 - μ_{t+1}) × g_t / (1 - β₁^t)`
- Parameter update: `θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)`
- Look-ahead gradient computation provides faster convergence than standard Adam

**Best for:**
- ✅ Modern deep learning applications requiring fast convergence
- ✅ Computer vision tasks (CNNs, image classification)
- ✅ Natural language processing (when you want faster than Adam)
- ✅ Large-scale training where convergence speed matters
- ✅ When Adam works but you want superior performance

**Unified Training Integration:**
```go
// Complete example with ModelTrainingEngine
trainer, err := training.NewModelTrainer(model, config)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// Single call handles: forward pass + Nadam optimization + backward pass
result, err := trainer.TrainStep(inputTensor, labelTensor)

// Supports learning rate scheduling during training
if epoch % 10 == 0 {
    newLR := config.LearningRate * 0.8
    trainer.UpdateLearningRate(newLR)
}
```

**Architecture Benefits:**
- **GPU-Resident**: All Nadam state (momentum, variance, step count) stays on GPU
- **Single CGO Call**: Complete training step with Nesterov momentum in one call
- **MPSGraph Integration**: Automatic kernel fusion with momentum schedule computation
- **Memory Efficient**: Proper buffer pooling and automatic cleanup

**Key Advantages over Adam:**
- **Faster Convergence**: Nesterov momentum provides faster convergence, especially in later stages
- **Better Generalization**: Improved momentum schedule leads to better generalization
- **Reduced Oscillations**: More stable optimization trajectory
- **Theoretical Foundation**: Strong theoretical backing for the momentum schedule
- **Modern Performance**: Often converges 10-20% faster than Adam on deep networks

### Second-Order Optimizers

#### L-BFGS (Limited-memory BFGS)
**Quasi-Newton method for smooth optimization.**

```go
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.LBFGS,
    LearningRate:  1.0,    // Often set to 1.0 for L-BFGS
    // L-BFGS specific parameters handled internally
}
```

**How it works:**
- Approximates second-order information (Hessian)
- Uses limited memory to store gradient history
- Very effective for smooth, well-conditioned problems

**Best for:**
- ✅ Small to medium datasets
- ✅ Smooth optimization landscapes
- ✅ When high precision is needed
- ✅ Batch optimization (not mini-batch)

**Limitations:**
- High memory usage for large models
- Not suitable for stochastic/mini-batch training
- Can be slow for very large networks

## 🎯 Optimizer Selection Guide

### Problem Type Recommendations

#### Image Classification (CNNs)
```go
// Primary choice: Adam
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.001,
    Beta1:         0.9,
    Beta2:         0.999,
}

// Alternative: SGD with momentum for longer training
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.SGD,
    LearningRate:  0.01,
}
```

#### Text/NLP Tasks
```go
// RMSProp for RNNs
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.RMSProp,
    LearningRate:  0.001,
}

// Adam for transformers
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.0001,  // Lower for large models
    Beta1:         0.9,
    Beta2:         0.98,    // Slightly different for NLP
}
```

#### Tabular Data (MLPs)
```go
// Adam for general use
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.001,
}

// L-BFGS for smaller, smooth problems
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.LBFGS,
    LearningRate:  1.0,
}
```

#### Sparse Features
```go
// AdaGrad for sparse data
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.AdaGrad,
    LearningRate:  0.01,
}
```

### Model Size Recommendations

#### Small Models (< 1M parameters)
```go
// SGD often works well
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.SGD,
    LearningRate:  0.01,
}

// L-BFGS for batch training
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.LBFGS,
    LearningRate:  1.0,
}
```

#### Medium Models (1M - 10M parameters)
```go
// Adam is usually best
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.001,
    Beta1:         0.9,
    Beta2:         0.999,
}
```

#### Large Models (> 10M parameters)
```go
// Adam with lower learning rate
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.0001,
    Beta1:         0.9,
    Beta2:         0.999,
}

// AdamW equivalent settings
config := training.TrainerConfig{
    OptimizerType: cgo_bridge.Adam,
    LearningRate:  0.00001,
    Beta1:         0.9,
    Beta2:         0.999,
}
```

## 🔧 Practical Examples

### Complete Training Setup Examples

#### Basic Adam Configuration

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

func setupAdamTraining() {
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Build a simple model
    inputShape := []int{32, 784}
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(128, true, "hidden").
        AddReLU("relu").
        AddDense(10, true, "output").
        Compile()
    
    // Configure Adam optimizer
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.CrossEntropy,
        ProblemType:   training.Classification,
        
        // Adam-specific parameters
        Beta1:   0.9,    // Momentum decay
        Beta2:   0.999,  // Squared gradient decay
        Epsilon: 1e-8,   // Numerical stability
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Println("✅ Adam optimizer configured successfully")
    fmt.Printf("   Learning Rate: %.4f\n", config.LearningRate)
    fmt.Printf("   Beta1 (momentum): %.3f\n", config.Beta1)
    fmt.Printf("   Beta2 (adaptive): %.3f\n", config.Beta2)
}
```

#### Optimizer Comparison Example

```go
func compareOptimizers() {
    fmt.Println("🔍 Optimizer Comparison Example")
    
    optimizers := []struct {
        name string
        config training.TrainerConfig
        use_case string
    }{
        {
            "Adam (Standard)",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.Adam,
                LearningRate:  0.001,
                Beta1:         0.9,
                Beta2:         0.999,
                Epsilon:       1e-8,
            },
            "General purpose, most problems",
        },
        {
            "SGD (Conservative)",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.SGD,
                LearningRate:  0.01,
            },
            "Simple problems, interpretability",
        },
        {
            "RMSProp",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.RMSProp,
                LearningRate:  0.001,
            },
            "RNNs, non-stationary objectives",
        },
        {
            "AdaGrad",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.AdaGrad,
                LearningRate:  0.01,
            },
            "Sparse features, NLP",
        },
        {
            "AdaDelta",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.AdaDelta,
                LearningRate:  1.0,
                Alpha:         0.95,
            },
            "No LR tuning needed, robust",
        },
        {
            "Nadam",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.Nadam,
                LearningRate:  0.002,
                Beta1:         0.9,
                Beta2:         0.999,
            },
            "Faster Adam, modern deep learning",
        },
        {
            "L-BFGS",
            training.TrainerConfig{
                OptimizerType: cgo_bridge.LBFGS,
                LearningRate:  1.0,
            },
            "Small datasets, batch optimization",
        },
    }
    
    fmt.Printf("%-18s | %-8s | %-35s\n", "Optimizer", "LR", "Best Use Case")
    fmt.Println("-------------------|----------|------------------------------------")
    
    for _, opt := range optimizers {
        fmt.Printf("%-18s | %-8.4f | %-35s\n", 
                   opt.name, opt.config.LearningRate, opt.use_case)
    }
}
```

### Learning Rate Scheduling Patterns

```go
func demonstrateLearningRateScheduling() {
    fmt.Println("📈 Learning Rate Scheduling Patterns")
    
    fmt.Println("\n🎯 Common LR Schedules:")
    
    // Step decay pattern
    fmt.Println("Step Decay:")
    fmt.Println("   Epochs 1-10:  LR = 0.01")
    fmt.Println("   Epochs 11-20: LR = 0.005")
    fmt.Println("   Epochs 21+:   LR = 0.001")
    
    // Exponential decay
    fmt.Println("\nExponential Decay:")
    fmt.Println("   LR = initial_lr * (decay_rate ^ epoch)")
    fmt.Println("   Example: 0.01 * (0.95 ^ epoch)")
    
    // Cosine annealing
    fmt.Println("\nCosine Annealing:")
    fmt.Println("   LR follows cosine curve from max to min")
    fmt.Println("   Smooth transitions, good for fine-tuning")
    
    // Warm-up + decay
    fmt.Println("\nWarm-up + Decay:")
    fmt.Println("   Epochs 1-5:  Linear increase 0 → 0.01")
    fmt.Println("   Epochs 6+:   Decay from 0.01")
}
```

## 🎓 Tuning Guidelines

### Learning Rate Selection

```go
func learningRateGuidelines() {
    fmt.Println("🎯 Learning Rate Selection Guidelines")
    
    fmt.Println("\n📊 Starting Points by Optimizer:")
    rates := map[string]string{
        "Adam":     "0.001 (safe) to 0.003 (aggressive)",
        "SGD":      "0.01 (typical) to 0.1 (small models)",
        "RMSProp":  "0.001 (standard)",
        "AdaGrad":  "0.01 (can start higher, adapts down)",
        "AdaDelta": "1.0 (algorithm handles scaling automatically)",
        "Nadam":    "0.002 (slightly higher than Adam)",
        "L-BFGS":   "1.0 (algorithm determines step size)",
    }
    
    for opt, rate := range rates {
        fmt.Printf("   %-8s: %s\n", opt, rate)
    }
    
    fmt.Println("\n🔧 Tuning Strategy:")
    fmt.Println("   1. Start with recommended default")
    fmt.Println("   2. If loss decreases too slowly → increase LR")
    fmt.Println("   3. If loss oscillates/explodes → decrease LR")
    fmt.Println("   4. Monitor first 10-20 epochs for trend")
}
```

### Batch Size Impact

```go
func batchSizeImpact() {
    fmt.Println("📦 Batch Size Impact on Optimizers")
    
    fmt.Println("\n🎯 General Rules:")
    fmt.Println("   • Larger batches → more stable gradients → can use higher LR")
    fmt.Println("   • Smaller batches → more noise → may need lower LR")
    fmt.Println("   • Adam handles batch size variations better than SGD")
    
    fmt.Println("\n📊 Recommended Adjustments:")
    adjustments := []struct {
        batch_size string
        lr_adjustment string
        notes string
    }{
        {"8-16", "Standard LR", "Good for CNNs, limited memory"},
        {"32-64", "Standard LR", "Sweet spot for most problems"},
        {"128-256", "1.5-2x higher LR", "Stable gradients, faster training"},
        {"512+", "2-4x higher LR", "Very stable, linear scaling"},
    }
    
    fmt.Printf("%-12s | %-18s | %-25s\n", "Batch Size", "LR Adjustment", "Notes")
    fmt.Println("-------------|-------------------|-------------------------")
    for _, adj := range adjustments {
        fmt.Printf("%-12s | %-18s | %-25s\n", 
                   adj.batch_size, adj.lr_adjustment, adj.notes)
    }
}
```

### Convergence Diagnostics

```go
func convergenceDiagnostics() {
    fmt.Println("🔍 Convergence Diagnostics")
    
    fmt.Println("\n✅ Good Signs:")
    fmt.Println("   • Loss decreases steadily")
    fmt.Println("   • Validation loss tracks training loss")
    fmt.Println("   • Learning rate feels 'just right'")
    
    fmt.Println("\n⚠️ Warning Signs:")
    fmt.Println("   • Loss plateaus early → try higher LR")
    fmt.Println("   • Loss oscillates wildly → try lower LR") 
    fmt.Println("   • Validation loss diverges → overfitting")
    fmt.Println("   • Very slow progress → wrong optimizer/LR")
    
    fmt.Println("\n🔧 Fixes:")
    fmt.Println("   • Plateau: Increase LR by 2-5x")
    fmt.Println("   • Oscillation: Decrease LR by 2-10x")
    fmt.Println("   • Overfitting: Add regularization, lower LR")
    fmt.Println("   • Slow: Try Adam instead of SGD")
}
```

## 🚀 Advanced Optimization Techniques

### Gradient Clipping Concepts

```go
func gradientClippingConcepts() {
    fmt.Println("✂️ Gradient Clipping Concepts")
    
    fmt.Println("\n🎯 When to Use:")
    fmt.Println("   • Training RNNs (exploding gradients)")
    fmt.Println("   • Very deep networks")
    fmt.Println("   • When loss occasionally spikes")
    
    fmt.Println("\n📊 Types:")
    fmt.Println("   • Norm clipping: Limit gradient magnitude")
    fmt.Println("   • Value clipping: Limit individual gradient values")
    
    fmt.Println("\n⚙️ Implementation Note:")
    fmt.Println("   Currently handled internally by go-metal optimizers")
    fmt.Println("   Automatic stabilization for numerical stability")
}
```

### Optimizer State Management

```go
func optimizerStateManagement() {
    fmt.Println("💾 Optimizer State Management")
    
    fmt.Println("\n🎯 What Optimizers Remember:")
    fmt.Println("   • Adam: Momentum and squared gradient moving averages")
    fmt.Println("   • SGD: Previous gradients (if momentum enabled)")
    fmt.Println("   • RMSProp: Squared gradient moving averages")
    fmt.Println("   • AdaGrad: Accumulated squared gradients")
    
    fmt.Println("\n🔧 Implications:")
    fmt.Println("   • First few iterations may behave differently")
    fmt.Println("   • Warm-up periods help stabilize adaptive optimizers")
    fmt.Println("   • State is reset when creating new trainer")
    
    fmt.Println("\n💡 Best Practices:")
    fmt.Println("   • Don't change optimizers mid-training")
    fmt.Println("   • Consider warm-up for large learning rates")
    fmt.Println("   • Monitor early training behavior")
}
```

## 📊 Performance Comparison

### Optimizer Performance Characteristics

```go
func optimizerPerformanceCharacteristics() {
    fmt.Println("⚡ Optimizer Performance Characteristics")
    
    characteristics := []struct {
        optimizer string
        memory string
        speed string
        convergence string
        stability string
    }{
        {"Adam", "High", "Fast", "Fast", "High"},
        {"SGD", "Low", "Fastest", "Slow", "Medium"},
        {"RMSProp", "Medium", "Fast", "Medium", "High"},
        {"AdaGrad", "Medium", "Fast", "Fast→Slow", "Medium"},
        {"AdaDelta", "Medium", "Fast", "Steady", "Very High"},
        {"Nadam", "High", "Fast", "Very Fast", "High"},
        {"L-BFGS", "Very High", "Slow", "Very Fast", "Very High"},
    }
    
    fmt.Printf("%-10s | %-9s | %-7s | %-11s | %-9s\n",
               "Optimizer", "Memory", "Speed", "Convergence", "Stability")
    fmt.Println("-----------|-----------|---------|-------------|----------")
    
    for _, char := range characteristics {
        fmt.Printf("%-10s | %-9s | %-7s | %-11s | %-9s\n",
                   char.optimizer, char.memory, char.speed, 
                   char.convergence, char.stability)
    }
}
```

## 🎯 Quick Reference

### Optimizer Quick Selection

```go
func quickOptimizerSelection() {
    fmt.Println("⚡ Quick Optimizer Selection")
    
    fmt.Println("\n🎯 Default Choice:")
    fmt.Println("   Adam with LR=0.001 - works for 80% of problems")
    
    fmt.Println("\n🔧 Special Cases:")
    fmt.Println("   • Simple model + small data → SGD")
    fmt.Println("   • RNN/LSTM → RMSProp")  
    fmt.Println("   • Sparse features → AdaGrad")
    fmt.Println("   • No LR tuning wanted → AdaDelta")
    fmt.Println("   • Need fastest convergence → Nadam")
    fmt.Println("   • Small dataset + smooth loss → L-BFGS")
    
    fmt.Println("\n⚙️ Configuration Template:")
    fmt.Println(`   config := training.TrainerConfig{
       OptimizerType: cgo_bridge.Adam,
       LearningRate:  0.001,
       Beta1:         0.9,
       Beta2:         0.999,
       Epsilon:       1e-8,
   }`)
}
```

## 🚀 Next Steps

Master optimization in go-metal:

- **[Loss Functions Guide](loss-functions.md)** - Complete loss function reference
- **[Performance Guide](performance.md)** - Advanced optimization techniques
- **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Apply optimizers in practice
- **[CNN Tutorial](../tutorials/cnn-tutorial.md)** - Optimization for computer vision

**Ready for advanced techniques?** Check out the [Mixed Precision Tutorial](../tutorials/mixed-precision.md) for combining optimizers with FP16 training.

---

## 🧠 Key Takeaways

- **Adam is the default choice**: Works well for most deep learning problems
- **Learning rate matters most**: More important than optimizer choice
- **Match optimizer to problem**: SGD for simple, Adam for complex, RMSProp for RNNs
- **Monitor early training**: First 10-20 epochs reveal optimization health
- **Go-metal advantages**: Optimized implementations, automatic numerical stability, GPU-resident updates

Understanding optimizers helps you train models efficiently and achieve better results on Apple Silicon!