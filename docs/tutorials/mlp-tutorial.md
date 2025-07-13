# Multi-Layer Perceptron (MLP) Tutorial

Learn to build powerful neural networks with go-metal step by step.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- Build MLPs of any size and complexity
- Use different activation functions effectively
- Apply regularization techniques (BatchNorm, Dropout)
- Train on real datasets with proper evaluation
- Optimize performance for Apple Silicon

## 🏗️ MLP Architecture Fundamentals

### What is an MLP?

A Multi-Layer Perceptron is a fully connected neural network where:
- **Input Layer**: Receives raw data features
- **Hidden Layers**: Transform data through learned representations
- **Output Layer**: Produces final predictions
- **Connections**: Every neuron connects to all neurons in the next layer

### Mathematical Foundation

Each layer performs: `output = activation(input × weights + bias)`

```
Input → Dense → Activation → Dense → Activation → Output
  ↓        ↓        ↓        ↓        ↓         ↓
 [4]  → [4×10] →  [10]  → [10×3] →  [3]   → [3]
```

## 🚀 Tutorial 1: Simple Classification MLP

Let's start with a complete working example for iris flower classification.

### Step 1: Setup and Data Preparation

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "time"

    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("🌸 MLP Tutorial: Iris Classification")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Set random seed for reproducibility
    rand.Seed(42)
    
    // Problem setup
    batchSize := 16
    inputSize := 4     // 4 flower measurements
    numClasses := 3    // 3 iris species
    
    // Continue with model building...
}
```

### Step 2: Build the MLP Architecture

```go
func buildSimpleMLP(batchSize, inputSize, numClasses int) (*layers.Model, error) {
    fmt.Println("🏗️ Building Simple MLP Architecture")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Hidden layer 1: 4 → 16 with ReLU
        AddDense(16, true, "hidden1").
        AddReLU("relu1").
        
        // Hidden layer 2: 16 → 8 with ReLU
        AddDense(8, true, "hidden2").
        AddReLU("relu2").
        
        // Output layer: 8 → 3 (no activation, handled by loss function)
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("model compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: %d → 16 → 8 → %d\n", inputSize, numClasses)
    fmt.Printf("   📊 Total layers: %d\n", len(model.Layers))
    
    return model, nil
}
```

### Step 3: Generate Training Data

```go
func generateIrisData(batchSize int) ([]float32, []int32, []int, []int) {
    fmt.Println("📊 Generating Synthetic Iris Dataset")
    
    inputData := make([]float32, batchSize*4)
    labelData := make([]int32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        baseIdx := i * 4
        class := rand.Intn(3)
        
        // Generate class-specific features
        switch class {
        case 0: // Setosa - smaller flowers
            inputData[baseIdx] = rand.Float32()*1.0 + 4.0   // sepal length: 4.0-5.0
            inputData[baseIdx+1] = rand.Float32()*1.0 + 3.0 // sepal width: 3.0-4.0
            inputData[baseIdx+2] = rand.Float32()*0.8 + 1.0 // petal length: 1.0-1.8
            inputData[baseIdx+3] = rand.Float32()*0.5 + 0.1 // petal width: 0.1-0.6
            
        case 1: // Versicolor - medium flowers
            inputData[baseIdx] = rand.Float32()*1.0 + 5.5   // sepal length: 5.5-6.5
            inputData[baseIdx+1] = rand.Float32()*0.8 + 2.2 // sepal width: 2.2-3.0
            inputData[baseIdx+2] = rand.Float32()*1.0 + 3.5 // petal length: 3.5-4.5
            inputData[baseIdx+3] = rand.Float32()*0.5 + 1.0 // petal width: 1.0-1.5
            
        case 2: // Virginica - larger flowers
            inputData[baseIdx] = rand.Float32()*1.0 + 6.0   // sepal length: 6.0-7.0
            inputData[baseIdx+1] = rand.Float32()*0.8 + 2.8 // sepal width: 2.8-3.6
            inputData[baseIdx+2] = rand.Float32()*1.2 + 4.8 // petal length: 4.8-6.0
            inputData[baseIdx+3] = rand.Float32()*0.8 + 1.8 // petal width: 1.8-2.6
        }
        
        labelData[i] = int32(class)
    }
    
    inputShape := []int{batchSize, 4}
    labelShape := []int{batchSize}
    
    fmt.Printf("   ✅ Generated %d samples with %d features\n", batchSize, 4)
    
    return inputData, labelData, inputShape, labelShape
}
```

### Step 4: Configure Training

```go
func setupTrainer(model *layers.Model, batchSize int) (*training.ModelTrainer, error) {
    fmt.Println("⚙️ Configuring Training Setup")
    
    config := training.TrainerConfig{
        // Basic parameters
        BatchSize:    batchSize,
        LearningRate: 0.01,
        
        // Optimizer: Adam for adaptive learning rates
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,   // Momentum term
        Beta2:         0.999, // Squared gradient term
        Epsilon:       1e-8,  // Numerical stability
        
        // Problem configuration
        EngineType:   training.Dynamic,           // Universal engine
        ProblemType:  training.Classification,    // Classification task
        LossFunction: training.SparseCrossEntropy, // Integer class labels
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        return nil, fmt.Errorf("trainer creation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Optimizer: Adam (lr=%.3f)\n", config.LearningRate)
    fmt.Printf("   ✅ Loss: SparseCrossEntropy\n")
    
    return trainer, nil
}
```

### Step 5: Training Loop with Progress Tracking

```go
func trainModel(trainer *training.ModelTrainer, inputData []float32, inputShape []int, 
                labelData []int32, labelShape []int, epochs int) error {
    fmt.Printf("🚀 Training for %d epochs\n", epochs)
    fmt.Println("Epoch | Loss     | Progress")
    fmt.Println("------|----------|----------")
    
    for epoch := 1; epoch <= epochs; epoch++ {
        // Execute training step
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
        if err != nil {
            return fmt.Errorf("training epoch %d failed: %v", epoch, err)
        }
        
        // Progress indicator
        var progressBar string
        progress := float64(epoch) / float64(epochs)
        barLength := 10
        filled := int(progress * float64(barLength))
        
        for i := 0; i < barLength; i++ {
            if i < filled {
                progressBar += "█"
            } else {
                progressBar += "░"
            }
        }
        
        fmt.Printf("%5d | %.6f | %s %.1f%%\n", 
                   epoch, result.Loss, progressBar, progress*100)
        
        // Check for convergence
        if result.Loss < 0.1 {
            fmt.Printf("🎉 Early convergence achieved! (loss < 0.1)\n")
            break
        }
    }
    
    return nil
}
```

### Step 6: Complete Training Program

```go
func main() {
    fmt.Println("🌸 MLP Tutorial: Iris Classification")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Set random seed for reproducibility
    rand.Seed(42)
    
    // Problem setup
    batchSize := 16
    inputSize := 4
    numClasses := 3
    epochs := 50
    
    // Step 1: Build model
    model, err := buildSimpleMLP(batchSize, inputSize, numClasses)
    if err != nil {
        log.Fatalf("Model building failed: %v", err)
    }
    
    // Step 2: Setup trainer
    trainer, err := setupTrainer(model, batchSize)
    if err != nil {
        log.Fatalf("Trainer setup failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Step 3: Generate data
    inputData, labelData, inputShape, labelShape := generateIrisData(batchSize)
    
    // Step 4: Train model
    err = trainModel(trainer, inputData, inputShape, labelData, labelShape, epochs)
    if err != nil {
        log.Fatalf("Training failed: %v", err)
    }
    
    fmt.Println("\n🎓 Tutorial Complete!")
    fmt.Println("   ✅ Successfully built and trained an MLP")
    fmt.Println("   ✅ Used GPU-resident training on Apple Silicon")
    fmt.Println("   ✅ Applied modern optimization techniques")
}
```

## 🧠 Tutorial 2: Advanced MLP with Regularization

Now let's build a more sophisticated MLP with modern regularization techniques.

### Advanced Architecture with BatchNorm and Dropout

```go
func buildAdvancedMLP(batchSize, inputSize, numClasses int) (*layers.Model, error) {
    fmt.Println("🏗️ Building Advanced MLP with Regularization")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Input → Hidden 1: 784 → 256
        AddDense(256, true, "dense1").
        AddBatchNorm(256, 1e-5, 0.1, true, "bn1").
        AddReLU("relu1").
        AddDropout(0.3, "dropout1").
        
        // Hidden 1 → Hidden 2: 256 → 128
        AddDense(128, true, "dense2").
        AddBatchNorm(128, 1e-5, 0.1, true, "bn2").
        AddReLU("relu2").
        AddDropout(0.4, "dropout2").
        
        // Hidden 2 → Hidden 3: 128 → 64
        AddDense(64, true, "dense3").
        AddBatchNorm(64, 1e-5, 0.1, true, "bn3").
        AddReLU("relu3").
        AddDropout(0.2, "dropout3").
        
        // Output layer: 64 → classes
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("advanced model compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: %d → 256 → 128 → 64 → %d\n", inputSize, numClasses)
    fmt.Printf("   🔧 Regularization: BatchNorm + Dropout\n")
    fmt.Printf("   📊 Total layers: %d\n", len(model.Layers))
    
    return model, nil
}
```

### Advanced Training with Learning Rate Scheduling

```go
func trainAdvancedModel(trainer *training.ModelTrainer, inputData []float32, inputShape []int,
                       labelData []int32, labelShape []int, epochs int) error {
    fmt.Printf("🚀 Advanced Training for %d epochs\n", epochs)
    fmt.Println("Epoch | Loss     | LR      | Status")
    fmt.Println("------|----------|---------|--------")
    
    initialLR := 0.01
    
    for epoch := 1; epoch <= epochs; epoch++ {
        // Learning rate scheduling (step decay)
        var currentLR float64
        if epoch <= 20 {
            currentLR = initialLR
        } else if epoch <= 40 {
            currentLR = initialLR * 0.5
        } else {
            currentLR = initialLR * 0.1
        }
        
        // Update learning rate (conceptual - would need trainer.SetLearningRate())
        
        // Execute training step
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
        if err != nil {
            return fmt.Errorf("training epoch %d failed: %v", epoch, err)
        }
        
        // Training status
        var status string
        if result.Loss < 0.05 {
            status = "Excellent"
        } else if result.Loss < 0.2 {
            status = "Good"
        } else if result.Loss < 0.5 {
            status = "Learning"
        } else {
            status = "Starting"
        }
        
        fmt.Printf("%5d | %.6f | %.5f | %s\n", 
                   epoch, result.Loss, currentLR, status)
        
        // Early stopping
        if result.Loss < 0.01 {
            fmt.Printf("🎉 Training converged! (loss < 0.01)\n")
            break
        }
    }
    
    return nil
}
```

## 🎯 Tutorial 3: Real-World MNIST-Style Problem

Let's tackle a more realistic problem with higher-dimensional data.

### MNIST-Style Digit Classification

```go
func generateMNISTStyleData(batchSize int) ([]float32, []int32, []int, []int) {
    fmt.Println("🔢 Generating MNIST-Style Digit Dataset")
    
    inputSize := 784 // 28x28 pixels
    numClasses := 10 // 0-9 digits
    
    inputData := make([]float32, batchSize*inputSize)
    labelData := make([]int32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        // Generate synthetic "digit" patterns
        digit := rand.Intn(numClasses)
        labelData[i] = int32(digit)
        
        baseIdx := i * inputSize
        
        // Create digit-specific patterns
        for j := 0; j < inputSize; j++ {
            // Simplified pattern: different digits have different intensity distributions
            baseIntensity := float32(digit) / 10.0
            noise := (rand.Float32() - 0.5) * 0.2
            
            // Simulate "pixels" with some structure
            row := j / 28
            col := j % 28
            centerDistance := float32((row-14)*(row-14) + (col-14)*(col-14)) / 400.0
            
            intensity := baseIntensity + noise - centerDistance*0.3
            if intensity < 0 {
                intensity = 0
            }
            if intensity > 1 {
                intensity = 1
            }
            
            inputData[baseIdx+j] = intensity
        }
    }
    
    inputShape := []int{batchSize, inputSize}
    labelShape := []int{batchSize}
    
    fmt.Printf("   ✅ Generated %d samples of %d pixels each\n", batchSize, inputSize)
    fmt.Printf("   📊 Classes: 0-9 (10 digits)\n")
    
    return inputData, labelData, inputShape, labelShape
}
```

### High-Performance MLP for Image Data

```go
func buildImageMLP(batchSize int) (*layers.Model, error) {
    fmt.Println("🏗️ Building High-Performance Image MLP")
    
    inputSize := 784  // 28x28 pixels
    numClasses := 10  // 10 digits
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Input layer with dropout to prevent overfitting
        AddDropout(0.2, "input_dropout").
        
        // Hidden layer 1: 784 → 512
        AddDense(512, true, "dense1").
        AddBatchNorm(512, 1e-5, 0.1, true, "bn1").
        AddReLU("relu1").
        AddDropout(0.5, "dropout1").
        
        // Hidden layer 2: 512 → 256
        AddDense(256, true, "dense2").
        AddBatchNorm(256, 1e-5, 0.1, true, "bn2").
        AddReLU("relu2").
        AddDropout(0.5, "dropout2").
        
        // Hidden layer 3: 256 → 128
        AddDense(128, true, "dense3").
        AddBatchNorm(128, 1e-5, 0.1, true, "bn3").
        AddReLU("relu3").
        AddDropout(0.3, "dropout3").
        
        // Output layer: 128 → 10
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("image MLP compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: 784 → 512 → 256 → 128 → 10\n")
    fmt.Printf("   🔧 Regularization: Input Dropout + BatchNorm + Dropout\n")
    fmt.Printf("   📊 Parameters: ~500K (high capacity)\n")
    
    return model, nil
}
```

## 🔧 Advanced Techniques

### Activation Function Comparison

```go
func buildMLPWithDifferentActivations(batchSize, inputSize, numClasses int, 
                                    activationType string) (*layers.Model, error) {
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    // Build base architecture
    builder = builder.AddDense(64, true, "dense1")
    
    // Add different activation functions
    switch activationType {
    case "relu":
        builder = builder.AddReLU("activation1")
    case "leaky_relu":
        builder = builder.AddLeakyReLU(0.1, "activation1")
    case "elu":
        builder = builder.AddELU(1.0, "activation1")
    case "tanh":
        builder = builder.AddTanh("activation1")
    default:
        return nil, fmt.Errorf("unknown activation type: %s", activationType)
    }
    
    model, err := builder.
        AddDense(32, true, "dense2").
        AddReLU("relu2").
        AddDense(numClasses, true, "output").
        Compile()
    
    return model, err
}
```

### Architecture Search Example

```go
func findBestArchitecture(batchSize, inputSize, numClasses int) {
    fmt.Println("🔍 Architecture Search")
    
    architectures := []struct {
        name   string
        sizes  []int
        activation string
    }{
        {"Small", []int{32, 16}, "relu"},
        {"Medium", []int{64, 32}, "relu"},
        {"Large", []int{128, 64}, "relu"},
        {"Deep", []int{64, 64, 32}, "relu"},
        {"LeakyReLU", []int{64, 32}, "leaky_relu"},
        {"ELU", []int{64, 32}, "elu"},
    }
    
    for _, arch := range architectures {
        fmt.Printf("   Testing %s: %v with %s\n", 
                   arch.name, arch.sizes, arch.activation)
        
        // Build and test each architecture
        // (Implementation would involve building and training each)
    }
}
```

## 📊 Performance Optimization

### Memory-Efficient Training

```go
func optimizedTrainingLoop(trainer *training.ModelTrainer, 
                          inputData []float32, inputShape []int,
                          labelData []int32, labelShape []int,
                          epochs int) error {
    
    fmt.Println("⚡ Optimized Training with Performance Monitoring")
    
    for epoch := 1; epoch <= epochs; epoch++ {
        startTime := time.Now()
        
        // Training step
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
        if err != nil {
            return err
        }
        
        elapsed := time.Since(startTime)
        
        // Performance metrics
        samplesPerSec := float64(len(labelData)) / elapsed.Seconds()
        
        if epoch%10 == 0 {
            fmt.Printf("Epoch %d: Loss=%.4f, Speed=%.1f samples/sec, Time=%v\n",
                       epoch, result.Loss, samplesPerSec, elapsed)
        }
    }
    
    return nil
}
```

## 🎓 Best Practices Summary

### Architecture Design
- **Start Small**: Begin with 1-2 hidden layers
- **Scale Up**: Add depth/width based on data complexity
- **Regularization**: Always use BatchNorm and Dropout for deep networks
- **Activation**: ReLU for hidden layers, no activation for output

### Training Configuration
- **Optimizer**: Adam for most cases, SGD for simple problems
- **Learning Rate**: Start with 0.01, adjust based on convergence
- **Batch Size**: 16-64 for most problems, 8-16 for large models
- **Epochs**: Monitor validation loss for early stopping

### Performance Tips
- **GPU Efficiency**: Use larger batch sizes when memory allows
- **Memory Management**: Always call `defer trainer.Cleanup()`
- **Monitoring**: Track loss, learning rate, and training speed
- **Debugging**: Use meaningful layer names for easier debugging

## 🚀 Next Steps

You've mastered MLP building with go-metal! Continue your journey:

- **[CNN Tutorial](cnn-tutorial.md)** - Learn convolutional networks for images
- **[Performance Guide](../guides/performance.md)** - Optimize training speed
- **[Mixed Precision Tutorial](mixed-precision.md)** - 86% speedup with FP16
- **[Real Projects](../examples/)** - Complete end-to-end applications

**Ready for production?** Check out the [deployment guide](../guides/deployment.md) for best practices on serving go-metal models.

---

## 🧠 Key Takeaways

- **MLPs are versatile**: Great for tabular data, embeddings, and feature learning
- **Regularization matters**: BatchNorm + Dropout prevent overfitting
- **Architecture scales**: Start simple, add complexity gradually
- **Go-metal advantages**: GPU-resident training, automatic optimization, type safety

With these skills, you can tackle any supervised learning problem using MLPs in go-metal!