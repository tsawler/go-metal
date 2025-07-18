# Go-Metal Documentation

High-performance machine learning framework optimized for Apple Silicon GPUs.

## 🚀 What is Go-Metal?

Go-Metal is a native Go machine learning framework that leverages Apple's Metal Performance Shaders Graph (MPSGraph) to deliver exceptional training and inference performance on Apple Silicon. Built from the ground up for GPU-resident computing, go-metal minimizes CPU-GPU data transfers while providing a clean, idiomatic Go API.

### Key Features

- **🏎️ Apple Silicon Optimized**: Native Metal Performance Shaders integration
- **⚡ High Performance**: Up to 86% speedup with mixed precision training  
- **🎯 GPU-Resident**: Data stays on GPU throughout the entire pipeline
- **🧠 Complete ML Pipeline**: Training, inference, visualization, and deployment
- **💾 Production Ready**: Comprehensive memory management and checkpointing
- **🔧 Developer Friendly**: Clean Go API with extensive documentation

### Supported Capabilities

**Layer Types**: Dense, Conv2D, BatchNorm, ReLU, LeakyReLU, ELU, Dropout, and more  
**Optimizers**: Adam, SGD, RMSProp, AdaGrad, AdaDelta, Nadam, L-BFGS  
**Loss Functions**: CrossEntropy, SparseCrossEntropy, MSE, MAE, Huber, BCE  
**Data Precision**: Float32, Float16 (mixed precision), Int32  
**Visualization**: 15+ plot types including ROC curves, confusion matrices, feature importance  
**Interoperability**: ONNX import/export, native checkpoints

## 📚 Documentation Sections

### 🏁 Getting Started
- **[Installation](getting-started/installation.md)** - Setup and requirements
- **[Quick Start](getting-started/quick-start.md)** - Train your first model in 5 minutes
- **[Basic Concepts](getting-started/basic-concepts.md)** - Core concepts and terminology

### 📖 Guides
- **[Architecture](guides/architecture.md)** - Design principles and GPU-resident computing
- **[Layers](guides/layers.md)** - Complete layer reference and usage
- **[Optimizers](guides/optimizers.md)** - Optimizer selection and configuration
- **[Loss Functions](guides/loss-functions.md)** - Loss function reference
- **[Inference Engine](guides/inference-engine.md)** - High-performance model inference
- **[Performance](guides/performance.md)** - Optimization techniques and mixed precision
- **[Memory Management](guides/memory-management.md)** - GPU memory optimization and best practices
- **[Checkpoints](guides/checkpoints.md)** - Model saving, loading, and resume training
- **[Progress Tracking](guides/progress-tracking.md)** - PyTorch-style progress bars and training sessions
- **[Visualization](guides/visualization.md)** - Training metrics and plotting

### 🎓 Tutorials
- **[MLP Tutorial](tutorials/mlp-tutorial.md)** - Building Multi-Layer Perceptrons step-by-step
- **[CNN Tutorial](tutorials/cnn-tutorial.md)** - Building Convolutional Neural Networks
- **[Regression Tutorial](tutorials/regression-tutorial.md)** - Solving regression problems
- **[Mixed Precision](tutorials/mixed-precision.md)** - FP16 training for maximum performance

### 💡 Examples
- **[Cats & Dogs Classification](examples/cats-dogs-classification.md)** - Complete CNN example
- **[House Price Regression](examples/house-price-regression.md)** - Regression with visualization
- **[SparseCrossEntropy Demo](examples/sparse-cross-entropy.md)** - Integer label classification
- **[Inference Engine Demo](examples/inference-engine-demo.md)** - Complete inference workflow example

### 📋 Reference
- **[API Overview](reference/api-overview.md)** - High-level API tour
- **[Complete API Reference](https://pkg.go.dev/github.com/tsawler/go-metal)** - Detailed API documentation

### 🔬 Advanced Topics
- **[Performance Tuning](advanced/performance-tuning.md)** - Advanced optimization techniques
- **[Custom Layers](tutorials/custom-layers.md)** - Implementing custom layer types
- **[Contributing](advanced/contributing.md)** - Contributing to go-metal

## 🚀 Quick Example

```go
// Train a neural network in just a few lines
package main

import (
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/cgo_bridge"
)

func main() {
    // Define model architecture
    inputShape := []int{32, 784} // batch_size=32, features=784
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddDense(128, true, "hidden1").
        AddReLU("relu1").
        AddDense(10, true, "output").
        Compile()
    
    // Configure training
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        LossFunction:  0, // CrossEntropy
    }
    
    trainer, _ := training.NewModelTrainer(model, config)
    defer trainer.Cleanup()
    
    // Train on your data
    // result, _ := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
}
```

## 🏗️ Architecture Overview

Go-Metal is built on four core principles:

1. **GPU-Resident Everything**: Data stays on GPU throughout the pipeline
2. **Minimize CGO Calls**: Batched operations reduce bridge overhead  
3. **MPSGraph-Centric**: Leverage Apple's optimized compute graph
4. **Proper Memory Management**: Reference counting and buffer pooling

This architecture delivers exceptional performance while maintaining a clean, Go-idiomatic API.

## 🎯 Perfect For

- **iOS/macOS ML Applications**: Native Apple Silicon optimization
- **Research & Prototyping**: Fast iteration with comprehensive visualization
- **Production Deployment**: Robust memory management and error handling
- **Performance-Critical Applications**: GPU-resident computing eliminates bottlenecks
- **Computer Vision**: Optimized CNN performance with mixed precision

## 🤝 Community & Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/tsawler/go-metal/issues)
- **Examples Repository**: Working code examples for all major use cases
- **API Documentation**: [pkg.go.dev/github.com/tsawler/go-metal](https://pkg.go.dev/github.com/tsawler/go-metal)

---

**Ready to get started?** Jump into the [Quick Start guide](getting-started/quick-start.md) and train your first model in 5 minutes!