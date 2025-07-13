# Layer Reference

Complete documentation for all layers available in go-metal.

## 🧠 Overview

Go-Metal provides a comprehensive set of neural network layers optimized for Apple Silicon. Each layer is implemented using Metal Performance Shaders Graph (MPSGraph) for maximum performance and automatic optimization.

## 🏗️ Model Building Pattern

All layers are added using the builder pattern for type safety and clear error messages:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
)

func main() {
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Define input shape: [batch_size, features]
    inputShape := []int{32, 784}
    
    // Build model using chainable methods
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(128, true, "hidden1").
        AddReLU("relu1").
        AddDense(64, true, "hidden2").
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    fmt.Printf("✅ Model created with %d layers\n", len(model.Layers))
    fmt.Printf("   Architecture: 784 → 128 → 64 → 10\n")
}
```

## 📋 Layer Categories

### Core Layers

#### Dense (Fully Connected)
Performs linear transformation: `output = input × weight + bias`

```go
.AddDense(units, useBias, "name")
```

**Parameters:**
- `units`: Number of output neurons
- `useBias`: Whether to add bias term (`true`/`false`)
- `name`: Layer identifier for debugging

**Shape Transformation:**
- Input: `[batch_size, input_features]`
- Output: `[batch_size, units]`
- Weights: `[input_features, units]`
- Bias: `[units]` (if enabled)

**Example:**
```go
// 784 → 128 transformation with bias
.AddDense(128, true, "hidden1")

// 128 → 64 transformation without bias  
.AddDense(64, false, "hidden2")
```

#### Conv2D (2D Convolution)
Applies 2D convolution for image processing.

```go
.AddConv2D(filters, kernelSize, "name")
```

**Parameters:**
- `filters`: Number of output channels
- `kernelSize`: Size of convolution kernel (square)
- `name`: Layer identifier

**Shape Transformation:**
- Input: `[batch_size, channels, height, width]`
- Output: `[batch_size, filters, out_height, out_width]`
- Kernel: `[filters, channels, kernelSize, kernelSize]`

**Example:**
```go
// 32 filters, 3x3 kernel
.AddConv2D(32, 3, "conv1")

// 64 filters, 5x5 kernel
.AddConv2D(64, 5, "conv2")
```

#### BatchNorm (Batch Normalization)
Normalizes layer inputs to improve training stability.

```go
.AddBatchNorm(numFeatures, eps, momentum, affine, "name")
```

**Parameters:**
- `numFeatures`: Number of features to normalize (matches previous layer output)
- `eps`: Small value for numerical stability (typically 1e-5)
- `momentum`: Moving average momentum (typically 0.1)
- `affine`: Whether to learn scale and shift parameters (`true`/`false`)
- `name`: Layer identifier

**Benefits:**
- Faster convergence
- Higher learning rates
- Reduced internal covariate shift
- Regularization effect

**Example:**
```go
.AddDense(256, true, "dense1").
.AddBatchNorm(256, 1e-5, 0.1, true, "bn1").
.AddReLU("relu1")
```

### Activation Layers

#### ReLU (Rectified Linear Unit)
Applies: `f(x) = max(0, x)`

```go
.AddReLU("name")
```

**Properties:**
- Most popular activation
- Solves vanishing gradient problem
- Computationally efficient
- Zero for negative inputs

#### LeakyReLU
Applies: `f(x) = max(αx, x)` where α is the negative slope

```go
.AddLeakyReLU(alpha, "name")
```

**Parameters:**
- `alpha`: Slope for negative values (typically 0.01-0.3)

**Benefits:**
- Prevents dying ReLU problem
- Non-zero gradients for negative inputs

**Example:**
```go
// 10% slope for negative values
.AddLeakyReLU(0.1, "leaky1")
```

#### ELU (Exponential Linear Unit)
Applies: `f(x) = x` if x > 0, else `α(e^x - 1)`

```go
.AddELU(alpha, "name")
```

**Parameters:**
- `alpha`: Scale for negative saturation (typically 1.0)

**Benefits:**
- Smooth for negative values
- Zero-centered outputs
- Robust to noise

#### Sigmoid
Applies: `f(x) = 1 / (1 + e^(-x))`

```go
.AddSigmoid("name")
```

**Use Cases:**
- Binary classification output
- Gates in RNNs/LSTMs
- When outputs need to be in (0, 1)

#### Tanh (Hyperbolic Tangent)
Applies: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

```go
.AddTanh("name")
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered
- Stronger gradients than sigmoid

### Regularization Layers

#### Dropout
Randomly sets input elements to zero during training.

```go
.AddDropout(rate, "name")
```

**Parameters:**
- `rate`: Fraction of inputs to drop (0.0 to 1.0)

**Benefits:**
- Prevents overfitting
- Forces network to use all neurons
- Ensemble effect

**Common Rates:**
- 0.2-0.3 for input layers
- 0.5 for hidden layers
- 0.0 for output layers

**Example:**
```go
.AddDense(512, true, "dense1").
.AddReLU("relu1").
.AddDropout(0.5, "dropout1").
.AddDense(10, true, "output")
```

### Utility Layers

#### Flatten
Reshapes multi-dimensional input to 2D.

```go
.AddFlatten("name")
```

**Shape Transformation:**
- Input: `[batch_size, dim1, dim2, ..., dimN]`
- Output: `[batch_size, dim1 × dim2 × ... × dimN]`

**Common Use:**
- Transition from Conv2D to Dense layers

**Example:**
```go
.AddConv2D(64, 3, "conv").    // Output: [32, 64, 26, 26]
.AddFlatten("flatten").       // Output: [32, 43264]
.AddDense(128, true, "dense") // Input: [32, 43264]
```

#### Reshape
Transforms tensor to specified shape.

```go
.AddReshape(newShape, "name")
```

**Parameters:**
- `newShape`: Target shape `[]int{dim1, dim2, ...}`

**Requirements:**
- Total elements must match
- First dimension (batch_size) is preserved

**Example:**
```go
// Reshape from [32, 784] to [32, 28, 28]
.AddReshape([]int{28, 28}, "reshape")
```

## 🎯 Layer Combinations

### Common Patterns

#### MLP Block
```go
.AddDense(units, true, "dense").
.AddBatchNorm(units, 1e-5, 0.1, true, "bn").
.AddReLU("relu").
.AddDropout(0.3, "dropout")
```

#### CNN Block
```go
.AddConv2D(filters, kernelSize, "conv").
.AddBatchNorm(filters, 1e-5, 0.1, true, "bn").
.AddReLU("relu")
```

#### Residual-Style Block
```go
// Main path
.AddDense(256, true, "dense1").
.AddReLU("relu1").
.AddDense(256, false, "dense2")
// Note: Residual connections require manual implementation
```

### Architecture Examples

#### Simple MLP
```go
inputShape := []int{32, 784} // MNIST-like
builder := layers.NewModelBuilder(inputShape)
model, _ := builder.
    AddDense(128, true, "hidden1").AddReLU("relu1").
    AddDense(64, true, "hidden2").AddReLU("relu2").
    AddDense(10, true, "output").
    Compile()
```

#### Regularized Deep Network
```go
inputShape := []int{64, 1024}
builder := layers.NewModelBuilder(inputShape)
model, _ := builder.
    AddDense(512, true, "dense1").
    AddBatchNorm(512, 1e-5, 0.1, true, "bn1").AddReLU("relu1").AddDropout(0.3, "drop1").
    AddDense(256, true, "dense2").
    AddBatchNorm(256, 1e-5, 0.1, true, "bn2").AddReLU("relu2").AddDropout(0.3, "drop2").
    AddDense(128, true, "dense3").
    AddBatchNorm(128, 1e-5, 0.1, true, "bn3").AddReLU("relu3").AddDropout(0.2, "drop3").
    AddDense(10, true, "output").
    Compile()
```

#### Simple CNN
```go
inputShape := []int{32, 3, 32, 32} // CIFAR-10 like
builder := layers.NewModelBuilder(inputShape)
model, _ := builder.
    AddConv2D(32, 3, "conv1").AddReLU("relu1").
    AddConv2D(64, 3, "conv2").AddReLU("relu2").
    AddFlatten("flatten").
    AddDense(128, true, "dense1").AddReLU("relu3").
    AddDense(10, true, "output").
    Compile()
```

## ⚙️ Technical Details

### Weight Initialization

All layers use appropriate default initialization:

- **Dense**: Xavier/Glorot uniform initialization
- **Conv2D**: He initialization for ReLU networks
- **BatchNorm**: Scale=1.0, Bias=0.0

### Memory Layout

**Dense Layer Memory:**
```
Weights: [input_features, output_features] (column-major)
Bias:    [output_features]
Input:   [batch_size, input_features] 
Output:  [batch_size, output_features]
```

**Conv2D Memory:**
```
Kernel:  [output_channels, input_channels, height, width]
Bias:    [output_channels]
Input:   [batch_size, input_channels, height, width]
Output:  [batch_size, output_channels, out_height, out_width]
```

### GPU Optimization

**Automatic Kernel Fusion:**
```
Dense + ReLU + Dropout → Single GPU kernel
Conv2D + BatchNorm + ReLU → Fused convolution
```

**Memory Efficiency:**
- Buffer pooling for intermediate results
- In-place operations where possible
- Optimal memory alignment for Metal

## 🔧 Debugging and Inspection

### Layer Information
```go
model, _ := builder.Compile()

// Inspect model structure
fmt.Printf("Model has %d layers:\n", len(model.Layers))
for i, layer := range model.Layers {
    fmt.Printf("  %d: %s\n", i, layer.Name)
}
```

### Shape Validation
```go
// The builder validates shapes automatically
model, err := builder.
    AddDense(128, true, "dense1").
    AddDense(64, true, "dense2").   // 128 → 64 ✓
    AddDense(10, true, "output").   // 64 → 10 ✓
    Compile()

if err != nil {
    // Detailed shape mismatch information
    log.Printf("Shape error: %v", err)
}
```

### Common Shape Errors
```go
// ❌ Wrong: Shape mismatch
.AddDense(128, true, "dense1").      // Output: [batch, 128]
.AddConv2D(32, 3, "conv")            // Expects 4D input!

// ✅ Correct: Add reshape
.AddDense(128, true, "dense1").      // Output: [batch, 128]
.AddReshape([]int{8, 16}, "reshape"). // Output: [batch, 8, 16, 1]
.AddConv2D(32, 3, "conv")            // Now works!
```

## 🎯 Performance Tips

### Layer Ordering
```go
// ✅ Efficient: BatchNorm before activation
.AddDense(256, true, "dense").
.AddBatchNorm(256, 1e-5, 0.1, true, "bn").
.AddReLU("relu")

// ❌ Less efficient: Activation before BatchNorm
.AddDense(256, true, "dense").
.AddReLU("relu").
.AddBatchNorm(256, 1e-5, 0.1, true, "bn")
```

### Dropout Placement
```go
// ✅ Good: Dropout after activation
.AddDense(512, true, "dense").
.AddReLU("relu").
.AddDropout(0.5, "drop")

// ❌ Suboptimal: Dropout before activation
.AddDense(512, true, "dense").
.AddDropout(0.5, "drop").
.AddReLU("relu")
```

### Batch Size Considerations
- **Small models**: 32-64 samples
- **Large models**: 16-32 samples  
- **CNNs**: 8-16 samples (memory intensive)
- **Memory limited**: Start with 8, increase gradually

## 🚀 Next Steps

Now that you understand all available layers:

- **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Build complete multi-layer perceptrons
- **[CNN Tutorial](../tutorials/cnn-tutorial.md)** - Construct convolutional networks
- **[Architecture Guide](architecture.md)** - Understand go-metal's design principles
- **[Performance Guide](performance.md)** - Optimize training speed

**Ready for advanced projects?** Check out the [Examples Directory](../examples/) for complete implementations using these layers.

---

## 📚 API Quick Reference

| Layer | Method | Parameters | Output Shape |
|-------|--------|------------|--------------|
| **Dense** | `.AddDense(units, useBias, name)` | units, bias | `[batch, units]` |
| **Conv2D** | `.AddConv2D(filters, kernel, name)` | filters, kernel | `[batch, filters, h, w]` |
| **BatchNorm** | `.AddBatchNorm(features, eps, momentum, affine, name)` | features, eps, momentum, affine | Same as input |
| **ReLU** | `.AddReLU(name)` | - | Same as input |
| **LeakyReLU** | `.AddLeakyReLU(alpha, name)` | alpha | Same as input |
| **ELU** | `.AddELU(alpha, name)` | alpha | Same as input |
| **Sigmoid** | `.AddSigmoid(name)` | - | Same as input |
| **Tanh** | `.AddTanh(name)` | - | Same as input |
| **Dropout** | `.AddDropout(rate, name)` | rate | Same as input |
| **Flatten** | `.AddFlatten(name)` | - | `[batch, features]` |
| **Reshape** | `.AddReshape(shape, name)` | newShape | `[batch, ...shape]` |

Understanding these layers gives you the building blocks to create sophisticated neural networks optimized for Apple Silicon performance.