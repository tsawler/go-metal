# Basic Concepts

Understanding the core concepts that make go-metal powerful and efficient.

## 🏗️ Architecture Overview

Go-Metal is built on four fundamental principles that deliver exceptional performance on Apple Silicon:

### 1. **GPU-Resident Everything**
Data stays on GPU memory throughout the entire training pipeline. No expensive CPU-GPU transfers during computation.

### 2. **Minimize CGO Calls** 
Operations are batched to reduce the overhead of crossing the Go-C boundary. Single training steps perform complete forward/backward passes.

### 3. **MPSGraph-Centric**
Leverages Apple's Metal Performance Shaders Graph for automatic optimization, kernel fusion, and GPU acceleration.

### 4. **Proper Memory Management**
Reference counting, buffer pooling, and automatic cleanup prevent memory leaks and optimize GPU resource usage.

## 🧠 Core Components

### Model Building

Models in go-metal are built using a **builder pattern** that provides type safety and clear error messages:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/layers"
)

func main() {
    // Define input shape: [batch_size, features]
    inputShape := []int{32, 784} // MNIST-like: 32 samples, 784 features
    
    // Build model using chainable methods
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(128, true, "hidden1").  // 784 → 128 with bias
        AddReLU("relu1").               // ReLU activation
        AddDense(64, true, "hidden2").   // 128 → 64 with bias
        AddReLU("relu2").               // ReLU activation  
        AddDense(10, true, "output").    // 64 → 10 (classes)
        Compile()                        // Validate and finalize
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    fmt.Printf("✅ Model created with %d layers\n", len(model.Layers))
    fmt.Printf("   Architecture: 784 → 128 → 64 → 10\n")
}
```

**Key Points:**
- **Input Shape**: Always specify `[batch_size, features]` for MLPs
- **Layer Names**: Required for debugging and visualization
- **Bias Parameter**: `true` adds bias terms, `false` omits them
- **Compile()**: Validates shapes and prepares for training

### Layer Types

Go-Metal supports common neural network layers:

```go
// Core Layers
.AddDense(units, useBias, "name")           // Fully connected layer
.AddConv2D(filters, kernelSize, "name")     // 2D convolution
.AddBatchNorm("name")                       // Batch normalization

// Activation Layers  
.AddReLU("name")                            // f(x) = max(0, x)
.AddLeakyReLU(alpha, "name")               // f(x) = max(αx, x)
.AddELU(alpha, "name")                     // ELU with configurable α
.AddSigmoid("name")                        // σ(x) = 1/(1+e^(-x))
.AddTanh("name")                           // tanh activation

// Regularization
.AddDropout(rate, "name")                  // Random neuron deactivation

// Utility Layers
.AddFlatten("name")                        // Reshape to 2D
.AddReshape(newShape, "name")              // Arbitrary reshaping
```

**Example with different layer types:**

```go
inputShape := []int{16, 8}
builder := layers.NewModelBuilder(inputShape)
model, err := builder.
    AddDense(16, true, "dense1").
    AddLeakyReLU(0.1, "leaky").      // 10% negative slope
    AddDense(8, false, "dense2").     // No bias
    AddELU(1.0, "elu").              // Standard ELU (α=1.0)
    AddDense(3, true, "output").
    Compile()
```

### Training Configuration

Training behavior is controlled through `TrainerConfig`:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create simple model
    inputShape := []int{8, 4}
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.AddDense(3, true, "output").Compile()
    
    // Configure training parameters
    config := training.TrainerConfig{
        // Basic Parameters
        BatchSize:    8,              // Samples per batch
        LearningRate: 0.001,          // Optimizer step size
        
        // Optimizer Selection
        OptimizerType: cgo_bridge.Adam,    // Adam, SGD, RMSProp, etc.
        
        // Problem Type
        ProblemType:  training.Classification,  // or training.Regression
        LossFunction: training.CrossEntropy,    // Loss function
        
        // Engine Type
        EngineType: training.Dynamic,     // Universal architecture support
        
        // Adam Parameters (when using Adam)
        Beta1:   0.9,    // Momentum decay rate
        Beta2:   0.999,  // Second moment decay rate  
        Epsilon: 1e-8,   // Numerical stability
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Println("✅ Trainer configured successfully")
}
```

### Optimizers

Go-Metal provides multiple optimizers for different use cases:

```go
// Adam: Adaptive learning rates (most popular)
OptimizerType: cgo_bridge.Adam,
Beta1: 0.9,      // Momentum term
Beta2: 0.999,    // Squared gradient term
Epsilon: 1e-8,   // Numerical stability

// SGD: Simple and reliable
OptimizerType: cgo_bridge.SGD,
// LearningRate is the main parameter

// RMSProp: Good for RNNs and varying gradients
OptimizerType: cgo_bridge.RMSProp,
// Adapts learning rate based on gradient history
```

**Optimizer Comparison:**

| Optimizer | Best For | Key Feature |
|-----------|----------|-------------|
| **Adam** | General purpose | Adaptive learning rates |
| **SGD** | Simple problems | Stable, predictable |
| **RMSProp** | RNNs, non-stationary | Gradient scaling |
| **AdaGrad** | Sparse features | Per-parameter rates |
| **L-BFGS** | Small datasets | Second-order optimization |

### Loss Functions

Choose loss functions based on your problem type:

```go
// Classification Problems
LossFunction: training.CrossEntropy,        // One-hot labels
LossFunction: training.SparseCrossEntropy,  // Integer labels
LossFunction: training.BinaryCrossEntropy,  // Binary (0/1)
LossFunction: training.BCEWithLogits,       // Binary with raw logits

// Regression Problems  
LossFunction: training.MeanSquaredError,    // MSE (L2 loss)
LossFunction: training.MeanAbsoluteError,   // MAE (L1 loss)
LossFunction: training.Huber,               // Robust to outliers
```

**Example with different loss functions:**

```go
// For multi-class classification with integer labels
config := training.TrainerConfig{
    ProblemType:  training.Classification,
    LossFunction: training.SparseCrossEntropy,  // Labels: [0, 1, 2, ...]
    // ... other parameters
}

// For regression problems
config := training.TrainerConfig{
    ProblemType:  training.Regression,
    LossFunction: training.MeanSquaredError,    // Continuous targets
    // ... other parameters
}
```

### Data Formats

Go-Metal uses specific data formats for optimal GPU performance:

```go
// Input Data: Always float32 slices
inputData := []float32{
    1.0, 2.0, 3.0, 4.0,  // Sample 1 features
    5.0, 6.0, 7.0, 8.0,  // Sample 2 features
    // ... more samples
}

// Classification Labels: int32 class indices
labelData := []int32{0, 1, 2, 1}  // Class IDs

// Regression Targets: float32 values
targetData := []float32{1.5, 2.3, 0.8, 1.9}  // Continuous values

// Shapes: Always specify dimensions
inputShape := []int{batchSize, features}
labelShape := []int{batchSize}  // For classification/regression
```

## 🚀 Training Loop

The basic training pattern in go-metal:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // 1. Build model
    inputShape := []int{4, 3}
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(8, true, "hidden").
        AddReLU("relu").
        AddDense(2, true, "output").
        Compile()
    
    // 2. Configure training
    config := training.TrainerConfig{
        BatchSize:     4,
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
    
    // 3. Create sample data
    inputData := []float32{
        1.0, 2.0, 3.0,  // Sample 1
        4.0, 5.0, 6.0,  // Sample 2
        7.0, 8.0, 9.0,  // Sample 3
        1.5, 2.5, 3.5,  // Sample 4
    }
    labelData := []int32{0, 1, 1, 0}  // Class labels
    
    // 4. Training loop
    fmt.Println("Training Progress:")
    for step := 1; step <= 5; step++ {
        // Single training step (forward + backward + update)
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, []int{4})
        if err != nil {
            log.Fatalf("Training step failed: %v", err)
        }
        
        fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
    }
    
    fmt.Println("✅ Training completed!")
}
```

**What happens in `TrainBatch()`:**
1. **Data Transfer**: Copy data to GPU (one-time cost)
2. **Forward Pass**: Compute predictions through network
3. **Loss Calculation**: Compare predictions to labels
4. **Backward Pass**: Compute gradients via automatic differentiation  
5. **Parameter Update**: Optimizer adjusts weights
6. **Return Results**: Loss and timing information

## 🎯 Key Concepts Summary

### GPU-Resident Benefits
- **Speed**: No CPU-GPU transfers during training
- **Memory**: Efficient GPU buffer management
- **Optimization**: Metal Performance Shaders acceleration

### Type Safety
- **Compile-time Errors**: Catch shape mismatches early
- **Clear APIs**: Explicit parameter types
- **Helpful Messages**: Detailed error descriptions

### Memory Management
- **Automatic Cleanup**: `defer trainer.Cleanup()`
- **Buffer Pooling**: Reuse GPU memory efficiently
- **Reference Counting**: Prevent memory leaks

### Performance Features
- **Mixed Precision**: FP16 training for 86% speedup
- **Dynamic Shapes**: Support variable batch sizes
- **Kernel Fusion**: MPSGraph optimizes operations automatically

## 🔧 Common Patterns

**Multi-Layer Perceptron (MLP):**
```go
builder.
    AddDense(128, true, "hidden1").AddReLU("relu1").
    AddDense(64, true, "hidden2").AddReLU("relu2").
    AddDense(numClasses, true, "output")
```

**Convolutional Neural Network (CNN):**
```go
builder.
    AddConv2D(32, 3, "conv1").AddReLU("relu1").
    AddConv2D(64, 3, "conv2").AddReLU("relu2").
    AddFlatten("flatten").
    AddDense(10, true, "output")
```

**Regularized Network:**
```go
builder.
    AddDense(256, true, "dense1").
    AddBatchNorm("bn1").AddReLU("relu1").
    AddDropout(0.3, "dropout1").
    AddDense(10, true, "output")
```

---

## 🚀 Next Steps

Now that you understand the core concepts, explore more advanced topics:

- **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Build complete multi-layer perceptrons
- **[CNN Tutorial](../tutorials/cnn-tutorial.md)** - Construct convolutional networks
- **[Performance Guide](../guides/performance.md)** - Optimize training speed
- **[Layer Reference](../guides/layers.md)** - Complete layer documentation

**Ready to build something real?** Try the [Cats & Dogs Classification](../examples/cats-dogs-classification.md) example for a complete project walkthrough.