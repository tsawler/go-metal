# Go-Metal Tutorials

Welcome to the Go-Metal tutorials! These hands-on guides will teach you how to build, train, and optimize neural networks using Go-Metal on Apple Silicon.

## 📚 Tutorial Overview

Our tutorials are designed to take you from beginner to advanced usage, with practical examples you can run immediately.

### 🎯 Getting Started Tutorials

#### [MLP Tutorial](mlp-tutorial.md)
Build your first neural network with Go-Metal! This tutorial covers:
- Creating a Multi-Layer Perceptron from scratch
- Understanding layer composition
- Training on real datasets
- Evaluating model performance
- **Difficulty**: Beginner
- **Time**: 30 minutes

#### [CNN Tutorial](cnn-tutorial.md)
Master computer vision with Convolutional Neural Networks:
- Building CNN architectures
- Image preprocessing and data loading
- Conv2D and pooling layers
- Training on image datasets
- **Difficulty**: Intermediate
- **Time**: 45 minutes

#### [Regression Tutorial](regression-tutorial.md)
Solve real-world prediction problems:
- Linear and non-linear regression
- Feature engineering
- Loss functions for regression
- Model evaluation metrics
- **Difficulty**: Beginner
- **Time**: 25 minutes

### ⚡ Performance & Optimization

#### [Mixed Precision Training](mixed-precision.md) 🔥
**Achieve up to 86% speedup** with FP16 training on Apple Silicon:
- Understanding mixed precision fundamentals
- Automatic loss scaling for numerical stability
- Model-specific optimization strategies
- Debugging and validation techniques
- Production deployment considerations
- **Difficulty**: Intermediate
- **Time**: 40 minutes
- **Performance Gain**: 60-86% speedup

### 🛠️ Advanced Tutorials

#### [Custom Layers](custom-layers.md)
Extend Go-Metal with your own layer types:
- Layer interface and requirements
- Implementing forward and backward passes
- Metal shader integration
- Testing custom layers
- **Difficulty**: Advanced
- **Time**: 60 minutes

#### [ONNX Integration](onnx-integration.md)
Import and export models with ONNX:
- Loading pre-trained ONNX models
- Converting Go-Metal models to ONNX
- Cross-framework compatibility
- Deployment strategies
- **Difficulty**: Intermediate
- **Time**: 35 minutes

## 🚀 Tutorial Learning Path

### For Beginners
1. Start with **[MLP Tutorial](mlp-tutorial.md)** - Learn the basics
2. Try **[Regression Tutorial](regression-tutorial.md)** - Apply to real problems
3. Move to **[CNN Tutorial](cnn-tutorial.md)** - Computer vision fundamentals

### For Performance Optimization
1. Complete any beginner tutorial first
2. Jump to **[Mixed Precision Training](mixed-precision.md)** - Maximize speed
3. Check the **[Performance Guide](../guides/performance.md)** - Advanced techniques

### For Advanced Users
1. Master the basics with beginner tutorials
2. Implement **[Custom Layers](custom-layers.md)** - Extend the framework
3. Use **[ONNX Integration](onnx-integration.md)** - Cross-platform models

## 📋 Before You Start

### Prerequisites
- Go 1.19 or later installed
- macOS with Apple Silicon (M1/M2/M3)
- Basic understanding of neural networks
- Familiarity with Go programming

### Setup Check
```bash
# Verify Go installation
go version

# Clone go-metal (if not already done)
git clone https://github.com/tsawler/go-metal.git
cd go-metal

# Run tests to verify setup
go test ./...
```

## 💡 Tutorial Tips

### Running Examples
Each tutorial includes complete, runnable code examples. To run an example:

```bash
# Save the example code to a file (e.g., main.go)
# Run it with:
go run main.go
```

### Common Patterns
All tutorials follow these patterns:
1. **Import necessary packages**
2. **Initialize Metal device and memory**
3. **Build and compile model**
4. **Prepare data**
5. **Train and evaluate**
6. **Clean up resources**

### Debugging Help
If you encounter issues:
- Check error messages carefully
- Ensure proper cleanup with `defer` statements
- Monitor memory usage
- Use smaller batch sizes if needed
- Refer to **[Troubleshooting Guide](../guides/troubleshooting.md)**

## 🎯 What You'll Learn

By completing these tutorials, you'll be able to:
- ✅ Build various neural network architectures
- ✅ Train models efficiently on Apple Silicon
- ✅ Achieve professional-level performance with mixed precision
- ✅ Debug and optimize your models
- ✅ Deploy models to production
- ✅ Extend the framework with custom components

## 📚 Additional Resources

- **[API Reference](../reference/)** - Detailed API documentation
- **[Examples](../examples/)** - Complete applications
- **[Guides](../guides/)** - In-depth topic coverage
- **[Contributing](../advanced/contributing.md)** - Help improve Go-Metal

## 🤝 Getting Help

- **Issues**: [GitHub Issues](https://github.com/tsawler/go-metal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tsawler/go-metal/discussions)
- **Examples**: Check the `examples/` directory

---

Ready to start? Pick a tutorial above and begin your Go-Metal journey! 🚀