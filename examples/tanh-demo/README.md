# Tanh Activation Function Demo

This example demonstrates Go-Metal's Tanh activation function implementation, showcasing its integration with the MPSGraph-centric architecture and its advantages for zero-centered neural networks.

## What This Demonstrates

### 🔢 Tanh Function Properties
- **Mathematical Definition**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- **Output Range**: (-1, 1) - zero-centered, bounded
- **Key Characteristics**: S-shaped curve, odd function (tanh(-x) = -tanh(x))
- **Zero-Centered**: Unlike sigmoid, outputs have zero mean

### 🎯 Use Cases
- **Hidden Layers**: Better than sigmoid due to zero-centered outputs
- **RNN/LSTM**: Traditional activation for recurrent neural networks
- **Feature Normalization**: Zero-mean outputs reduce internal covariate shift
- **Multi-class Classification**: Hidden layers in classification networks

## Architecture Integration

### 🏗️ MPSGraph Implementation
```objective-c
// Optimal Metal Performance Shaders implementation
currentTensor = [engine->graph tanhWithTensor:currentTensor
                                         name:[NSString stringWithFormat:@"tanh_%d", layerIdx]];
```

### 🚀 Architecture Compliance
- **✅ GPU-Resident**: Uses MPSGraph's built-in `tanhWithTensor` operation
- **✅ Minimal CGO**: Single CGO call per activation layer
- **✅ MPSGraph-Centric**: Native MPSGraph operation with automatic differentiation
- **✅ Memory Management**: No additional GPU buffer allocations

## Running the Demo

```bash
cd examples/tanh-demo
go run *.go
```

### Expected Output

```
=== Go-Metal Tanh Activation Function Demo ===
Demonstrating Tanh activation: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
Output range: (-1, 1) - zero-centered outputs

📋 Creating zero-centered neural network with Tanh...
✅ Model created successfully!
   Architecture: Flatten → Dense(128) → Tanh → Dense(64) → Tanh → Dense(10) → Softmax
   Total layers: 6
   Parameters: 108801 (trainable)

🔍 Layer Details:
   Layer 1: hidden1 (Dense)
   Layer 2: tanh1 (Tanh)
      → Tanh activation: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
      → Output range: (-1, 1)
      → Use case: Hidden layer activation (zero-centered outputs)
   Layer 3: hidden2 (Dense)
   Layer 4: tanh2 (Tanh)
      → Tanh activation: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
      → Output range: (-1, 1)
      → Use case: Hidden layer activation (zero-centered outputs)
   Layer 5: output (Dense)
   Layer 6: softmax (Softmax)

📊 Tanh Function Properties:
   • tanh(0) = 0 (zero-centered)
   • tanh(+∞) → +1.0 (positive saturation)
   • tanh(-∞) → -1.0 (negative saturation)
   • tanh(-x) = -tanh(x) (odd function)
   • Derivative: tanh'(x) = 1 - tanh²(x)
   • Maximum gradient at x=0: tanh'(0) = 1.0

🎉 Tanh activation function implementation completed!

🚀 Ready for production use in zero-centered neural networks!
```

## Mathematical Properties

### 🧮 Function Characteristics
- **Domain**: (-∞, ∞) - accepts any real input
- **Range**: (-1, 1) - bounded, zero-centered
- **Monotonic**: Strictly increasing function
- **Odd Function**: tanh(-x) = -tanh(x)

### 📉 Gradient Properties
- **Derivative**: tanh'(x) = 1 - tanh²(x)
- **Maximum Gradient**: 1.0 at x = 0
- **Stronger than Sigmoid**: Maximum gradient 4x larger than sigmoid
- **Vanishing Gradients**: tanh'(x) → 0 as |x| → ∞

## Implementation Details

### 🔧 Go Layer Integration
```go
// Factory method
func (lf *LayerFactory) CreateTanhSpec(name string) LayerSpec {
    return LayerSpec{
        Type:       Tanh,
        Name:       name,
        Parameters: map[string]interface{}{},
    }
}

// ModelBuilder method
func (mb *ModelBuilder) AddTanh(name string) *ModelBuilder {
    layer := LayerSpec{
        Type:       Tanh,
        Name:       name,
        Parameters: map[string]interface{}{},
    }
    return mb.AddLayer(layer)
}
```

### 🔄 ONNX Compatibility
- **Export**: Creates ONNX "Tanh" nodes
- **Import**: Converts ONNX "Tanh" nodes to go-metal Tanh layers
- **Standard Compliance**: ONNX v1.7+ compatibility

### 💾 Checkpoint Support
- **Serialization**: No parameters to save (activation function only)
- **JSON Format**: Layer type and name preservation
- **Model Recovery**: Full architecture restoration

## Use Cases in Practice

### 🧠 Hidden Layer Activation
```go
// Better than sigmoid for hidden layers due to zero-centered outputs
model := builder.
    AddDense(256, true, "hidden1").
    AddTanh("tanh1").                     // Zero-centered activation
    AddDense(128, true, "hidden2").
    AddTanh("tanh2").                     // Reduces internal covariate shift
    AddDense(10, true, "output").
    AddSoftmax(-1, "softmax").
    Compile()
```

### 🔄 RNN/LSTM Networks
```go
// Traditional activation for recurrent architectures
lstmCell := training.NewLSTMCell(inputSize, hiddenSize, training.LSTMConfig{
    Activation:     layers.Tanh,          // Hidden state activation
    GateActivation: layers.Sigmoid,       // Gate activations
})
```

### 📊 Feature Normalization
```go
// Zero-mean outputs help with training stability
model := builder.
    AddDense(featuresSize, true, "feature_layer").
    AddTanh("feature_normalization").     // Zero-centered feature mapping
    AddDense(outputSize, true, "classifier").
    Compile()
```

## Performance Characteristics

### ⚡ GPU Performance
- **Metal Optimized**: Uses Apple's optimized MPS implementation
- **Kernel Fusion**: Automatically fused with adjacent operations
- **Memory Bandwidth**: Minimal memory overhead
- **Throughput**: High-performance hyperbolic tangent computation

### 🔄 Training Efficiency
- **Forward Pass**: Single Metal kernel execution
- **Backward Pass**: Automatic MPSGraph differentiation
- **Gradient Computation**: tanh'(x) = 1 - tanh²(x)
- **Memory Usage**: In-place computation when possible

## Comparison with Other Activations

| Activation | Range | Centered | Max Gradient | Use Case | Pros | Cons |
|------------|-------|----------|--------------|----------|------|------|
| **Tanh** | (-1, 1) | Yes | 1.0 | Hidden layers | Zero-centered, strong gradients | Vanishing gradients |
| **Sigmoid** | (0, 1) | No | 0.25 | Binary output | Bounded output | Not zero-centered |
| **ReLU** | [0, ∞) | No | 1.0 | Hidden layers | No vanishing gradients | Dead neurons |
| **LeakyReLU** | (-∞, ∞) | No | 1.0 | Hidden layers | No dead neurons | Unbounded negative |

## Best Practices

### ✅ When to Use Tanh
1. **Hidden Layers**: When you need zero-centered activations
2. **RNN/LSTM**: Traditional choice for recurrent networks
3. **Feature Normalization**: When zero-mean outputs are beneficial
4. **Legacy Models**: Reproducing older architectures

### ❌ When to Avoid Tanh
1. **Very Deep Networks**: Vanishing gradient problem
2. **Modern CNNs**: ReLU variants generally perform better
3. **Computational Speed**: ReLU is faster to compute
4. **Sparse Representations**: ReLU provides natural sparsity

### 🎯 Implementation Tips
1. **Weight Initialization**: Xavier/Glorot initialization works well
2. **Batch Normalization**: Helps mitigate vanishing gradients
3. **Learning Rate**: May need adjustment compared to ReLU networks
4. **Gradient Clipping**: Useful for very deep networks

## Mathematical Derivation

### 📐 Function Definition
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Alternative form:
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

Relationship to sigmoid:
tanh(x) = 2 * sigmoid(2x) - 1
```

### 📊 Derivative
```
d/dx tanh(x) = 1 - tanh²(x)

This provides efficient gradient computation:
gradient = 1 - output²
```

---

**This example demonstrates Go-Metal's comprehensive Tanh activation implementation, ready for production use in zero-centered neural networks and recurrent architectures!** 🚀