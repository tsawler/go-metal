# Activation Functions Demo

This example demonstrates the usage of Sigmoid, Tanh, and Swish activation functions in go-metal.

## Complete Example

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

func main() {
    fmt.Println("=== Go-Metal Activation Functions Demo ===")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Create models with different activation functions
    batchSize := 16
    inputSize := 128
    numClasses := 10
    
    // Model 1: Using Sigmoid activation
    sigmoidModel := buildModelWithSigmoid(batchSize, inputSize, numClasses)
    
    // Model 2: Using Tanh activation
    tanhModel := buildModelWithTanh(batchSize, inputSize, numClasses)
    
    // Model 3: Using Swish activation
    swishModel := buildModelWithSwish(batchSize, inputSize, numClasses)
    
    // Model 4: Using mixed activations
    mixedModel := buildModelWithMixedActivations(batchSize, inputSize, numClasses)
    
    fmt.Println("✅ All activation function models created successfully!")
}

// Sigmoid activation model - ideal for binary classification
func buildModelWithSigmoid(batchSize, inputSize, numClasses int) *layers.ModelSpec {
    fmt.Println("🔵 Building model with Sigmoid activation...")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(64, true, "dense1").
        AddSigmoid("sigmoid1").           // σ(x) = 1/(1+e^(-x))
        AddDense(32, true, "dense2").
        AddSigmoid("sigmoid2").
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to build sigmoid model: %v", err)
    }
    
    fmt.Printf("   Architecture: %d → 64 → 32 → %d (Sigmoid activations)\n", inputSize, numClasses)
    return model
}

// Tanh activation model - zero-centered outputs
func buildModelWithTanh(batchSize, inputSize, numClasses int) *layers.ModelSpec {
    fmt.Println("🟡 Building model with Tanh activation...")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(64, true, "dense1").
        AddTanh("tanh1").                 // tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
        AddDense(32, true, "dense2").
        AddTanh("tanh2").
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to build tanh model: %v", err)
    }
    
    fmt.Printf("   Architecture: %d → 64 → 32 → %d (Tanh activations)\n", inputSize, numClasses)
    return model
}

// Swish activation model - smooth, self-gating activation
func buildModelWithSwish(batchSize, inputSize, numClasses int) *layers.ModelSpec {
    fmt.Println("🟢 Building model with Swish activation...")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(64, true, "dense1").
        AddSwish("swish1").               // swish(x) = x * σ(x)
        AddDense(32, true, "dense2").
        AddSwish("swish2").
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to build swish model: %v", err)
    }
    
    fmt.Printf("   Architecture: %d → 64 → 32 → %d (Swish activations)\n", inputSize, numClasses)
    return model
}

// Mixed activations model - combining different activation functions
func buildModelWithMixedActivations(batchSize, inputSize, numClasses int) *layers.ModelSpec {
    fmt.Println("🌈 Building model with mixed activations...")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(128, true, "dense1").
        AddSwish("swish1").               // Swish for first layer
        AddDense(64, true, "dense2").
        AddTanh("tanh1").                 // Tanh for middle layer
        AddDense(32, true, "dense3").
        AddSigmoid("sigmoid1").           // Sigmoid for final hidden layer
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to build mixed model: %v", err)
    }
    
    fmt.Printf("   Architecture: %d → 128 → 64 → 32 → %d (Mixed: Swish, Tanh, Sigmoid)\n", inputSize, numClasses)
    return model
}
```

## Activation Function Properties

### Sigmoid: σ(x) = 1/(1+e^(-x))
- **Output Range**: (0, 1)
- **Use Cases**: Binary classification, probability outputs
- **Pros**: Smooth, differentiable, bounded
- **Cons**: Vanishing gradients, not zero-centered

### Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- **Output Range**: (-1, 1)
- **Use Cases**: Hidden layers in traditional networks
- **Pros**: Zero-centered, stronger gradients than sigmoid
- **Cons**: Still suffers from vanishing gradients

### Swish: swish(x) = x * σ(x)
- **Output Range**: Unbounded above, bounded below at 0
- **Use Cases**: Deep networks, modern architectures
- **Pros**: Smooth, self-gating, better than ReLU in deep networks
- **Cons**: Slightly more computationally expensive

## Training Example

```go
func trainWithDifferentActivations() {
    // Generate sample data
    inputData := generateSampleData(batchSize, inputSize)
    labelData := generateSampleLabels(batchSize, numClasses)
    
    models := []struct {
        name  string
        model *layers.ModelSpec
    }{
        {"Sigmoid", sigmoidModel},
        {"Tanh", tanhModel},
        {"Swish", swishModel},
        {"Mixed", mixedModel},
    }
    
    for _, m := range models {
        fmt.Printf("Training %s model...\n", m.name)
        
        config := training.TrainerConfig{
            BatchSize:     batchSize,
            LearningRate:  0.01,
            OptimizerType: cgo_bridge.Adam,
            ProblemType:   training.Classification,
            LossFunction:  training.SparseCrossEntropy,
        }
        
        trainer, err := training.NewModelTrainer(m.model, config)
        if err != nil {
            log.Printf("Failed to create trainer for %s: %v", m.name, err)
            continue
        }
        defer trainer.Cleanup()
        
        // Train for a few epochs
        for epoch := 1; epoch <= 5; epoch++ {
            result, err := trainer.TrainBatch(inputData, []int{batchSize, inputSize}, 
                                            labelData, []int{batchSize})
            if err != nil {
                log.Printf("Training failed for %s epoch %d: %v", m.name, epoch, err)
                break
            }
            
            fmt.Printf("  Epoch %d: Loss = %.4f\n", epoch, result.Loss)
        }
        
        fmt.Printf("✅ %s model training completed\n\n", m.name)
    }
}
```

This example demonstrates the practical usage of all three activation functions in go-metal, showing how they can be used individually or combined in the same model.