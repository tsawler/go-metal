# Sigmoid Activation Function Demo

This example demonstrates Go-Metal's Sigmoid activation function implementation, showcasing its integration with the MPSGraph-centric architecture and its use in binary classification models.

## What This Demonstrates

### 🔢 Sigmoid Function Properties
- **Mathematical Definition**: σ(x) = 1/(1+e^(-x))
- **Output Range**: (0, 1) - always positive, bounded
- **Key Characteristics**: S-shaped curve, differentiable everywhere
- **Symmetry**: σ(-x) = 1 - σ(x)

### 🎯 Use Cases
- **Binary Classification**: Final layer for probability output (0-1 range)
- **Legacy Networks**: Traditional neural networks before ReLU dominance
- **Gating Mechanisms**: LSTM forget gates and input gates
- **Probability Mapping**: Converting logits to probabilities

## Architecture Integration

### 🏗️ MPSGraph Implementation
```objective-c
// Optimal Metal Performance Shaders implementation
currentTensor = [engine->graph sigmoidWithTensor:currentTensor
                                            name:[NSString stringWithFormat:@"sigmoid_%d", layerIdx]];
```

### 🚀 Architecture Compliance
- **✅ GPU-Resident**: Uses MPSGraph's built-in `sigmoidWithTensor` operation
- **✅ Minimal CGO**: Single CGO call per activation layer
- **✅ MPSGraph-Centric**: Native MPSGraph operation with automatic differentiation
- **✅ Memory Management**: No additional GPU buffer allocations

## Running the Demo

```bash
cd examples/sigmoid-demo
go run main.go
```

### Expected Output

```
=== Go-Metal Sigmoid Activation Function Demo ===
Demonstrating Sigmoid activation: σ(x) = 1/(1+e^(-x))
Output range: (0, 1) - ideal for binary classification

📋 Creating binary classification model with Sigmoid...
✅ Model created successfully!
   Architecture: Flatten → Dense(128) → Sigmoid → Dense(64) → Sigmoid → Dense(1) → Sigmoid
   Total layers: 7
   Parameters: 109,057 (trainable)

🔍 Layer Details:
   Layer 1: hidden1 (Dense)
   Layer 2: sigmoid1 (Sigmoid)
      → Sigmoid activation: σ(x) = 1/(1+e^(-x))
      → Output range: (0, 1)
      → Use case: Hidden layer activation (legacy networks)
   Layer 3: hidden2 (Dense)
   Layer 4: sigmoid2 (Sigmoid)
      → Sigmoid activation: σ(x) = 1/(1+e^(-x))
      → Output range: (0, 1)
      → Use case: Hidden layer activation (legacy networks)
   Layer 5: output (Dense)
   Layer 6: output_sigmoid (Sigmoid)
      → Sigmoid activation: σ(x) = 1/(1+e^(-x))
      → Output range: (0, 1)
      → Use case: Binary classification probability output

📊 Sigmoid Function Properties:
   • σ(0) ≈ 0.5 (symmetric around origin)
   • σ(+∞) → 1.0 (positive saturation)
   • σ(-∞) → 0.0 (negative saturation)
   • Derivative: σ'(x) = σ(x) * (1 - σ(x))
   • Maximum gradient at x=0: σ'(0) = 0.25

🎉 Sigmoid activation function implementation completed!

🚀 Ready for production use in binary classification models!
```

## Mathematical Properties

### 🧮 Function Characteristics
- **Domain**: (-∞, ∞) - accepts any real input
- **Range**: (0, 1) - always positive, never exactly 0 or 1
- **Monotonic**: Strictly increasing function
- **Differentiable**: Smooth gradient everywhere

### 📉 Gradient Properties
- **Derivative**: σ'(x) = σ(x) × (1 - σ(x))
- **Maximum Gradient**: 0.25 at x = 0
- **Vanishing Gradients**: σ'(x) → 0 as |x| → ∞
- **Chain Rule Friendly**: Easy backpropagation computation

## Implementation Details

### 🔧 Go Layer Integration
```go
// Factory method
func (lf *LayerFactory) CreateSigmoidSpec(name string) LayerSpec {
    return LayerSpec{
        Type:       Sigmoid,
        Name:       name,
        Parameters: map[string]interface{}{},
    }
}

// ModelBuilder method
func (mb *ModelBuilder) AddSigmoid(name string) *ModelBuilder {
    layer := LayerSpec{
        Type:       Sigmoid,
        Name:       name,
        Parameters: map[string]interface{}{},
    }
    return mb.AddLayer(layer)
}
```

### 🔄 ONNX Compatibility
- **Export**: Creates ONNX "Sigmoid" nodes
- **Import**: Converts ONNX "Sigmoid" nodes to go-metal Sigmoid layers
- **Standard Compliance**: ONNX v1.7+ compatibility

### 💾 Checkpoint Support
- **Serialization**: No parameters to save (activation function only)
- **JSON Format**: Layer type and name preservation
- **Model Recovery**: Full architecture restoration

## Use Cases in Practice

### 🎯 Binary Classification
```go
// Final layer setup for binary classification
model := builder.
    AddDense(hiddenSize, true, "hidden").
    AddReLU("hidden_activation").           // ReLU for hidden layers
    AddDense(1, true, "output").           // Single output neuron
    AddSigmoid("output_activation").       // Sigmoid for probability
    Compile()

// Loss function: BinaryCrossEntropy expects probabilities [0,1]
loss := training.NewBinaryCrossEntropyLoss()
```

### 🧠 Legacy Network Architectures
```go
// Pre-ReLU era neural network design
model := builder.
    AddDense(256, true, "layer1").
    AddSigmoid("sigmoid1").               // Traditional activation
    AddDense(128, true, "layer2").
    AddSigmoid("sigmoid2").               // Hidden layer activation
    AddDense(10, true, "output").
    AddSoftmax(-1, "softmax").            // Multi-class output
    Compile()
```

## Performance Characteristics

### ⚡ GPU Performance
- **Metal Optimized**: Uses Apple's optimized MPS implementation
- **Kernel Fusion**: Automatically fused with adjacent operations
- **Memory Bandwidth**: Minimal memory overhead
- **Throughput**: High-performance sigmoid computation

### 🔄 Training Efficiency
- **Forward Pass**: Single Metal kernel execution
- **Backward Pass**: Automatic MPSGraph differentiation
- **Gradient Computation**: σ'(x) = σ(x) × (1 - σ(x))
- **Memory Usage**: In-place computation when possible

## Comparison with Other Activations

| Activation | Range | Gradient | Use Case | Pros | Cons |
|------------|-------|----------|----------|------|------|
| **Sigmoid** | (0, 1) | σ'(x) = σ(x)(1-σ(x)) | Binary classification | Bounded output, smooth | Vanishing gradients |
| **ReLU** | [0, ∞) | 1 if x>0, 0 if x≤0 | Hidden layers | No vanishing gradients | Dead neurons |
| **Tanh** | (-1, 1) | 1 - tanh²(x) | Hidden layers | Zero-centered | Vanishing gradients |
| **LeakyReLU** | (-∞, ∞) | 1 if x>0, α if x≤0 | Hidden layers | No dead neurons | Unbounded negative |

## Best Practices

### ✅ When to Use Sigmoid
1. **Binary Classification**: Final layer for probability output
2. **Gating Mechanisms**: LSTM/GRU gate functions
3. **Legacy Models**: Reproducing older architectures
4. **Probability Outputs**: When you need [0,1] range

### ❌ When to Avoid Sigmoid
1. **Deep Networks**: Vanishing gradient problem
2. **Hidden Layers**: ReLU generally performs better
3. **Multi-class**: Use Softmax instead
4. **Speed Critical**: ReLU is computationally faster

### 🎯 Implementation Tips
1. **Initialize Weights Carefully**: Xavier/Glorot initialization works well
2. **Learning Rate**: May need lower learning rates due to gradients
3. **Batch Normalization**: Helps mitigate vanishing gradients
4. **Loss Function**: Use BCEWithLogits for numerical stability

---

**This example demonstrates Go-Metal's comprehensive Sigmoid activation implementation, ready for production use in binary classification and legacy network architectures!** 🚀