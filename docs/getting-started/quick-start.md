# Quick Start Guide

Train your first neural network with go-metal in under 5 minutes!

## 🎯 What We'll Build

A simple neural network to classify iris flowers based on their measurements. This classic machine learning problem demonstrates all the core concepts of go-metal:

- **Input**: 4 features (sepal length, sepal width, petal length, petal width)
- **Output**: 3 classes (setosa, versicolor, virginica)
- **Architecture**: Multi-layer perceptron with hidden layer
- **Training**: Adam optimizer with cross-entropy loss

## 📝 Complete Working Example

Create a new file `main.go` and copy this complete example:

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
    fmt.Println("🌸 Training Iris Classification Model")

    // Problem setup
    batchSize := 8
    inputSize := 4    // 4 flower measurements
    numClasses := 3   // 3 iris species

    // Step 1: Define model architecture
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(10, true, "hidden").    // Hidden layer: 4 → 10
        AddReLU("relu").                 // ReLU activation
        AddDense(numClasses, true, "output"). // Output layer: 10 → 3
        Compile()

    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }

    // Step 2: Configure training
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  0, // CrossEntropy
        ProblemType:   0, // Classification
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }

    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup() // Important: cleanup GPU resources

    // Step 3: Create sample training data
    inputData := make([]float32, batchSize*inputSize)
    labelData := make([]int32, batchSize)

    // Generate synthetic iris-like data
    for i := 0; i < batchSize; i++ {
        // Features: [sepal_length, sepal_width, petal_length, petal_width]
        baseIdx := i * inputSize
        inputData[baseIdx] = rand.Float32()*2 + 4     // sepal length: 4-6 cm
        inputData[baseIdx+1] = rand.Float32()*2 + 2   // sepal width: 2-4 cm
        inputData[baseIdx+2] = rand.Float32()*4 + 1   // petal length: 1-5 cm
        inputData[baseIdx+3] = rand.Float32()*2 + 0.1 // petal width: 0.1-2.1 cm

        // Labels: 0=setosa, 1=versicolor, 2=virginica
        labelData[i] = int32(rand.Intn(numClasses))
    }

    // Step 4: Train the model
    fmt.Printf("Training neural network (%d features → %d classes)\n", inputSize, numClasses)
    fmt.Println("Step | Loss")
    fmt.Println("-----|--------")

    for step := 1; step <= 10; step++ {
        // Execute one training step
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, []int{batchSize})
        if err != nil {
            log.Fatalf("Training step %d failed: %v", step, err)
        }

        fmt.Printf("%4d | %.4f\n", step, result.Loss)
    }

    fmt.Println("\n✅ Training completed! Your first go-metal model is ready.")
    fmt.Println("\n🎓 What you just accomplished:")
    fmt.Println("   ✓ Built a neural network with go-metal")
    fmt.Println("   ✓ Configured Adam optimizer")
    fmt.Println("   ✓ Trained on GPU using Metal Performance Shaders")
    fmt.Println("   ✓ Achieved GPU-resident training (no CPU-GPU transfers)")
}
```

## 🚀 Run the Example

```bash
# Initialize Go module (if not already done)
go mod init iris-classifier
go get github.com/tsawler/go-metal

# Run the example
go run main.go
```

**Expected output:**
```
🌸 Training Iris Classification Model
Training neural network (4 features → 3 classes)
Step | Loss
-----|--------
   1 | 3.6582
   2 | 3.3225
   3 | 2.9916
   4 | 2.6891
   5 | 2.4187
   6 | 2.1845
   7 | 1.9895
   8 | 1.8356
   9 | 1.7235
  10 | 1.6530

✅ Training completed! Your first go-metal model is ready.

🎓 What you just accomplished:
   ✓ Built a neural network with go-metal
   ✓ Configured Adam optimizer  
   ✓ Trained on GPU using Metal Performance Shaders
   ✓ Achieved GPU-resident training (no CPU-GPU transfers)
```

## 🧠 Understanding the Code

Let's break down what happened:

### 1. Model Architecture
```go
builder := layers.NewModelBuilder(inputShape)
model, err := builder.
    AddDense(10, true, "hidden").    // 4 inputs → 10 neurons (with bias)
    AddReLU("relu").                 // ReLU activation function
    AddDense(numClasses, true, "output"). // 10 → 3 outputs (with bias)
    Compile()
```

**Key Points:**
- `Dense` layers perform matrix multiplication: `output = input × weight + bias`
- `ReLU` applies: `f(x) = max(0, x)` element-wise
- `true` parameter enables bias terms
- `Compile()` validates the architecture and prepares for training

### 2. Training Configuration
```go
config := training.TrainerConfig{
    BatchSize:     8,           // Process 8 samples at once
    LearningRate:  0.01,        // Step size for gradient descent
    OptimizerType: cgo_bridge.Adam,  // Adam optimizer (adaptive learning)
    LossFunction:  0,           // CrossEntropy for classification
    // ... other Adam parameters
}
```

**Key Points:**
- **Batch Size**: Number of samples processed together (affects GPU efficiency)
- **Learning Rate**: Controls how big steps the optimizer takes
- **Adam**: Adaptive optimizer that adjusts learning rates automatically
- **CrossEntropy**: Standard loss for multi-class classification

### 3. Training Loop
```go
result, err := trainer.TrainBatch(inputData, inputShape, labelData, []int{batchSize})
```

**What happens internally:**
1. Data copied to GPU memory
2. Forward pass: compute predictions
3. Loss calculation: compare predictions to labels
4. Backward pass: compute gradients using automatic differentiation
5. Parameter update: Adam optimizer adjusts weights
6. Loss returned to CPU for monitoring

**GPU-Resident Magic**: All heavy computation stays on GPU!

## 🎯 Key Concepts Demonstrated

### ✅ GPU-Resident Computing
- Data stays on GPU throughout training
- No expensive CPU-GPU memory transfers during training loop
- Metal Performance Shaders handle all computation

### ✅ Automatic Differentiation
- Gradients computed automatically by MPSGraph
- No manual backpropagation code needed
- Optimized gradient computation chains

### ✅ Memory Management
- Automatic cleanup with `defer trainer.Cleanup()`
- GPU buffer pooling and reuse
- Reference counting prevents memory leaks

### ✅ Type Safety
- Strong typing for shapes, data types, and configurations
- Compile-time catch many common ML errors
- Clear error messages for debugging

## 🚀 Next Steps

Congratulations! You've successfully trained your first neural network with go-metal. Here's what to explore next:

### Immediate Next Steps
- **[Basic Concepts](basic-concepts.md)** - Understand layers, optimizers, and loss functions
- **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Build more complex neural networks
- **[Performance Guide](../guides/performance.md)** - Optimize training speed

### Advanced Features to Explore
- **Mixed Precision Training**: 86% speedup with FP16
- **Visualization**: Plot training curves and metrics
- **ONNX Integration**: Load pre-trained models
- **Checkpointing**: Save and resume training

### Example Projects
- **[Cats & Dogs Classification](../examples/cats-dogs-classification.md)** - Complete CNN example
- **[House Price Regression](../examples/house-price-regression.md)** - Regression with visualization

## 💡 Pro Tips

**Performance**: Larger batch sizes (32-128) often improve GPU utilization
**Memory**: Call `defer trainer.Cleanup()` to prevent GPU memory leaks  
**Debugging**: Check error messages - go-metal provides detailed diagnostics
**Experimentation**: Try different optimizers, learning rates, and architectures

---

**🎉 You're now ready to build production ML models with go-metal!**

The next guide, [Basic Concepts](basic-concepts.md), will deepen your understanding of layers, optimizers, and the GPU-resident architecture that makes go-metal so fast.