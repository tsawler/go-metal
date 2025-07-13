# Complete Training Pipeline Demo

This example demonstrates a complete end-to-end neural network training pipeline using Go-Metal, showcasing advanced features like batch normalization, dropout, and early stopping.

## What This Demonstrates

Unlike the `cats-dogs` demo which uses a CNN architecture, this demo uses a **Pure MLP (Multi-Layer Perceptron)** architecture to prove that the library now supports **ANY neural network architecture**, not just CNNs.

### Architecture Comparison

| Demo | Architecture | Layers Used |
|------|-------------|-------------|
| `cats-dogs` | CNN | Conv2D, ReLU, Dense |
| `any-model-demo` | **Pure MLP** | **Dense, BatchNorm, ReLU, Dropout, Softmax** |

## Model Architecture

This demo implements a sophisticated MLP for 3-class iris flower classification:

```
Input (4 features) 
    ↓
Dense (4 → 32) 
    ↓
BatchNorm(32) 
    ↓
ReLU 
    ↓
Dropout(0.3) 
    ↓
Dense (32 → 16) 
    ↓
ReLU 
    ↓
Dropout(0.2) 
    ↓
Dense (16 → 8) 
    ↓
ReLU 
    ↓
Dense (8 → 3) 
    ↓
Softmax
```

## Key Features Demonstrated

1. **Pure MLP Architecture** - No convolutional layers at all
2. **BatchNorm Support** - Batch normalization for stable training
3. **Multiple Activation Functions** - ReLU and Softmax
4. **Dropout Regularization** - Different dropout rates in different layers
5. **Multi-class Classification** - 3-class iris species classification
6. **Feature Normalization** - Demonstrates preprocessing capabilities
7. **Early Stopping** - Automatic training termination on convergence

## Dataset

- **Synthetic Iris Dataset**: 600 samples (200 per class)
- **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **Classes**: Setosa, Versicolor, Virginica
- **Preprocessing**: Z-score normalization for better training stability

## Technical Significance

This demo proves that the **generic layer configuration** implementation successfully resolved the CNN-only limitation. The library can now handle:

- ✅ Pure MLPs (this demo)
- ✅ CNNs (cats-dogs demo)  
- ✅ Mixed architectures (CNN + MLP layers)
- ✅ Any combination of supported layer types

## Running the Demo

```bash
cd app/any-model-demo
go run main.go
```

## Expected Output

The demo will train a Pure MLP and achieve >95% validation accuracy on the iris classification task, demonstrating that:

1. The dynamic engine supports non-CNN architectures
2. Layer validation works for any architecture
3. Shape inference works correctly for MLP layers
4. BatchNorm running statistics are properly initialized
5. All layer types integrate seamlessly

## Architecture Flexibility

This demo represents a major architectural advancement - the go-metal library now supports **ANY neural network architecture** while maintaining its GPU-resident principles and MPSGraph-centric design.