# Getting Started with Go-Metal

Welcome to Go-Metal, a high-performance deep learning library built specifically for Apple Silicon! This guide will get you up and running with your first neural network in minutes.

## 🎯 What You'll Learn

By the end of this guide, you'll have:
- ✅ Trained your first neural network with Go-Metal
- ✅ Understood the core concepts of GPU-resident computing
- ✅ Learned how to use different optimizers and loss functions
- ✅ Built both classification and regression models
- ✅ Explored advanced features like visualization and mixed precision

## 📚 Quick Navigation

### 🚀 [Installation Guide](installation.md)
Complete setup instructions for macOS, Xcode, and Go dependencies.

### ⚡ [Quick Start Tutorial](quick-start.md)
Train your first neural network in under 5 minutes with a complete working example.

### 🧠 [Basic Concepts](basic-concepts.md)
Deep dive into layers, optimizers, loss functions, and GPU-resident architecture.

## 🌟 Why Go-Metal?

Go-Metal is designed with four core principles that make it uniquely powerful:

### 1. **GPU-Resident Everything**
All computations happen on the GPU using Metal Performance Shaders. Your data stays on the GPU throughout the entire training pipeline, eliminating expensive CPU-GPU transfers.

### 2. **Apple Silicon Optimized**
Built specifically for M1, M2, and M3 chips, Go-Metal leverages Apple's MPSGraph for maximum performance with up to **121x speedup** for matrix operations.

### 3. **Type-Safe and Memory-Safe**
Go's type system catches ML errors at compile time, while automatic memory management prevents GPU memory leaks.

### 4. **Production Ready**
Complete training infrastructure including checkpointing, visualization, mixed precision, and ONNX compatibility.

## 🔥 Performance Highlights

| Feature | Performance Benefit |
|---------|-------------------|
| **Matrix Operations** | 121x faster than CPU |
| **Training Pipeline** | 13.5x faster than CPU |
| **Mixed Precision** | 86% training speedup |
| **Memory Management** | Zero memory leaks with buffer pooling |

## 📖 Learning Path

### 1. **Start Here: Basic Setup**
```bash
# Install Go-Metal
go mod init my-ml-project
go get github.com/tsawler/go-metal
```

### 2. **Your First Model (5 minutes)**
```go
// Create a simple neural network
builder := layers.NewModelBuilder([]int{8, 4}) // batch_size=8, features=4
model, _ := builder.
    AddDense(10, true, "hidden").    // 4 → 10 hidden layer
    AddReLU("relu").                 // ReLU activation
    AddDense(3, true, "output").     // 10 → 3 output layer
    Compile()

fmt.Printf("Model has %d parameters\n", model.TotalParameters)
```

### 3. **Training Configuration**
```go
// Configure training with Adam optimizer
config := training.TrainerConfig{
    BatchSize:     8,
    LearningRate:  0.01,
    OptimizerType: cgo_bridge.Adam,
    LossFunction:  training.CrossEntropy,
    ProblemType:   training.Classification,
    Beta1:         0.9,
    Beta2:         0.999,
    Epsilon:       1e-8,
}
```

### 4. **Complete Training Loop**
```go
// Create trainer and run training
trainer, _ := training.NewModelTrainer(model, config)
defer trainer.Cleanup()

// Train for 10 steps
for step := 1; step <= 10; step++ {
    result, _ := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
    fmt.Printf("Step %d: Loss = %.4f\n", step, result.Loss)
}
```

## 🎮 Interactive Examples

### 🌸 Classification Example
Train a neural network to classify iris flowers:
```bash
cd examples/iris-classification
go run main.go
```

### 🏠 Regression Example  
Predict house prices using neural regression:
```bash
cd examples/house-price-prediction
go run main.go
```

### 🖼️ Computer Vision Example
Build a CNN for image classification with full data pipeline:
```bash
cd examples/computer-vision-pipeline
go run main.go
```

## 🛠️ Available Components

### **Neural Network Layers**
- **Dense** - Fully connected layers with optional bias
- **Conv2D** - 2D convolutions for image processing
- **ReLU, Sigmoid, Tanh** - Activation functions
- **BatchNorm** - Batch normalization
- **Dropout** - Regularization

### **Optimizers**
- **Adam** - Adaptive moment estimation (recommended)
- **SGD** - Stochastic gradient descent with momentum
- **RMSprop** - Root mean square propagation
- **AdaGrad** - Adaptive learning rates
- **L-BFGS** - Limited-memory quasi-Newton method

### **Loss Functions**
- **Classification**: CrossEntropy, SparseCrossEntropy, BinaryCrossEntropy
- **Regression**: MSE, MAE, Huber loss

### **Advanced Features**
- **Mixed Precision Training** - FP16 for 86% speedup
- **Visualization** - Real-time training plots
- **Checkpointing** - Save and resume training
- **ONNX Integration** - Import/export models

## 🚀 Next Steps

Ready to dive deeper? Here's your learning path:

1. **[Quick Start](quick-start.md)** - Train your first model in 5 minutes
2. **[Basic Concepts](basic-concepts.md)** - Understand the architecture
3. **[Computer Vision](../guides/computer-vision.md)** - Image classification pipeline
4. **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Build complex networks
5. **[CNN Tutorial](../tutorials/cnn-tutorial.md)** - Convolutional networks
6. **[Performance Guide](../guides/performance.md)** - Optimize training speed

## 💡 Pro Tips

- **Start with Adam optimizer** - It adapts learning rates automatically
- **Use larger batch sizes** - Better GPU utilization (try 32-128)
- **Enable mixed precision** - 86% speedup with minimal accuracy loss
- **Always call `defer trainer.Cleanup()`** - Prevents GPU memory leaks
- **Monitor training curves** - Enable visualization for insights

## 🆘 Need Help?

- **📖 Documentation**: Complete API reference in `docs/`
- **💬 Examples**: Working code in `examples/`
- **🐛 Issues**: Report bugs on GitHub
- **🎯 Tutorials**: Step-by-step guides in `docs/tutorials/`

## 🎉 What's Next?

You're now ready to build production-quality machine learning models with Go-Metal! The library provides everything you need from basic tensor operations to advanced training techniques.

**Ready to start?** Jump into the **[Quick Start Tutorial](quick-start.md)** and train your first neural network in under 5 minutes!

---

*Built with ❤️ for Apple Silicon developers who want the performance of Metal with the safety of Go.*