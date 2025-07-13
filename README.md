# Go-Metal

A high-performance deep learning library for Go that leverages Apple's Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon.

[![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)](https://golang.org)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-brightgreen.svg)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

## 🚀 Overview

Go-Metal is a PyTorch-inspired deep learning library built specifically for Apple Silicon. It provides GPU-accelerated tensor operations, automatic differentiation, and a complete training pipeline - all with the safety and simplicity of Go. It's only dependency apart from the standard library is google.golang.org/protobuf, so the package and read and write ONNX files.

Why did I build this? I wanted to increase my knowledge in Machine Learning, and building something like this from scratch seemed like a good way to take a deep dive into that topic. 

**Key Performance Achievement**: 13.5x faster GPU training compared to CPU, with up to 121x speedup for matrix operations.

### Why Go-Metal?

- **🍎 Native Apple Silicon**: Built specifically for Apple's M-series chips using Metal Performance Shaders
- **⚡ High Performance**: GPU-accelerated operations with persistent GPU memory management
- **🛡️ Memory Safe**: Go's garbage collection combined with careful GPU resource management
- **🔧 PyTorch-like API**: Familiar interface for machine learning practitioners
- **📦 Complete Package**: Everything from tensors to training loops in one library

## ✨ Features

### Core Tensor Operations
- **Multi-device Support**: CPU, GPU, and PersistentGPU tensor types
- **Broadcasting**: NumPy/PyTorch-style broadcasting for all operations
- **Element-wise Operations**: Add, Sub, Mul, Div with GPU acceleration
- **Matrix Operations**: High-performance GPU matrix multiplication
- **Activation Functions**: ReLU, Softmax, LeakyReLU, ELU, Sigmoid with Metal implementations

### Deep Learning Components
- **Neural Network Layers**: Linear, Conv2D, MaxPool2D, BatchNorm, Flatten
- **Optimizers**: SGD, Adam, AdaGrad, RMSprop, AdaDelta, NAdam, L-BFGS
- **Loss Functions**: MSE (regression), CrossEntropy (classification)
- **Automatic Differentiation**: Complete autograd engine with gradient computation
- **Model Containers**: Sequential models for easy layer composition

### GPU Acceleration
- **Metal Performance Shaders**: Integration with Apple's optimized MPS framework
- **Asynchronous Execution**: Non-blocking GPU operations with completion handlers
- **Memory Management**: Advanced GPU memory allocation with buffer pooling
- **Operation Fusion**: Fused kernels for optimal performance (47x speedup)
- **Persistent GPU Tensors**: Keep model parameters on GPU throughout training

### Training Infrastructure
- **Complete Training Pipeline**: From data loading to model evaluation
- **DataLoader**: Efficient batching, shuffling, and GPU data transfer
- **Metrics Collection**: Training progress tracking and visualization support
- **Checkpointing**: Model saving and loading (ONNX format supported)
- **Mixed Precision**: Float16 support for memory-efficient training

## 📋 Requirements

- **Operating System**: macOS 12.0+ (Monterey or later)
- **Hardware**: Apple Silicon (M1, M1 Pro, M1 Max, M1 Ultra, M2, M3, etc.)
- **Development**: 
  - Go 1.21 or later
  - Xcode 14.0+ with Command Line Tools
  - Metal support (included with macOS)

## 🛠️ Installation

1. [**Install Go**](http://go.dev) (if not already installed)
   

2. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

3. **Add to your Go project**:
   ```bash
   go mod init your-project
   go get github.com/tsawler/go-metal
   ```

4. **Verify installation** with a simple test:
   ```go
   package main
   
   import (
       "fmt"
       "github.com/tsawler/go-metal/tensor"
   )
   
   func main() {
       // Create tensors
       a, _ := tensor.NewTensor([]int{2, 3}, tensor.Float32, tensor.CPU, []float32{1, 2, 3, 4, 5, 6})
       b, _ := tensor.NewTensor([]int{2, 3}, tensor.Float32, tensor.CPU, []float32{6, 5, 4, 3, 2, 1})
       
       // GPU acceleration
       result, _ := tensor.AddMPS(a, b)
       fmt.Printf("Result: %v\n", result.Data)
   }
   ```

## 🚦 Quick Start

### Basic Tensor Operations

```go
package main

import (
    "fmt"
    "github.com/tsawler/go-metal/tensor"
)

func main() {
    // Create tensors on different devices
    cpuTensor, _ := tensor.Zeros([]int{3, 3}, tensor.Float32, tensor.CPU)
    gpuTensor, _ := tensor.Ones([]int{3, 3}, tensor.Float32, tensor.PersistentGPU)
    
    // GPU-accelerated operations
    result, _ := tensor.MatMulMPS(cpuTensor, gpuTensor)
    activated, _ := tensor.ReLUMPS(result)
    
    fmt.Printf("Shape: %v\n", activated.Shape)
}
```

### Neural Network Training

```go
package main

import (
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/tensor"
)

func main() {
    // Create a neural network
    model := training.NewSequential(
        training.NewLinear(784, 128, true),  // Input layer
        training.NewReLU(),
        training.NewLinear(128, 64, true),   // Hidden layer
        training.NewReLU(), 
        training.NewLinear(64, 10, true),    // Output layer
    )
    
    // Setup training components
    optimizer := training.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8)
    criterion := training.NewCrossEntropyLoss()
    
    config := training.TrainingConfig{
        Device:       tensor.PersistentGPU,  // Keep model on GPU
        LearningRate: 0.001,
        BatchSize:    32,
        Epochs:       10,
    }
    
    // Create trainer and start training
    trainer := training.NewTrainer(model, optimizer, criterion, config)
    trainer.Train(trainLoader, validLoader)
}
```

### Convolutional Neural Networks

```go
package main

import (
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/tensor"
)

func main() {
    // Create a CNN for image classification
    model := training.NewSequential(
        training.NewConv2D(3, 32, 3, []int{1, 1}, []int{1, 1}, true),  // Conv layer
        training.NewReLU(),
        training.NewMaxPool2D([]int{2, 2}, []int{2, 2}, []int{0, 0}),  // Pooling
        training.NewConv2D(32, 64, 3, []int{1, 1}, []int{1, 1}, true), // Conv layer
        training.NewReLU(),
        training.NewMaxPool2D([]int{2, 2}, []int{2, 2}, []int{0, 0}),  // Pooling
        training.NewFlatten(),                                          // Flatten for FC
        training.NewLinear(64*7*7, 128, true),                         // Fully connected
        training.NewReLU(),
        training.NewLinear(128, 10, true),                             // Output
    )
    
    // Training setup identical to above...
}
```

## 📊 Performance

Go-Metal delivers exceptional performance on Apple Silicon:

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Matrix Multiplication (1024×1024) | 867ms | 7ms | **121x** |
| Training Pipeline | 4.6ms/batch | 0.34ms/batch | **13.5x** |
| Fused Operations | 3 kernel calls | 1 kernel call | **47x** |

### Device Types

- **`tensor.CPU`**: Traditional CPU tensors for compatibility
- **`tensor.GPU`**: Temporary GPU tensors (automatically copy back to CPU)
- **`tensor.PersistentGPU`**: High-performance GPU-resident tensors for training

## 📚 Documentation

### Core Documentation
- **[Getting Started Guide](docs/getting-started/)** - Basic concepts and installation
- **[Architecture Overview](docs/guides/architecture.md)** - Library design and components
- **[Performance Guide](docs/guides/performance.md)** - Optimization tips and benchmarks

### Tutorials
- **[MLP Tutorial](docs/tutorials/mlp-tutorial.md)** - Multi-layer perceptron from scratch
- **[CNN Tutorial](docs/tutorials/cnn-tutorial.md)** - Convolutional neural networks
- **[Regression Tutorial](docs/tutorials/regression-tutorial.md)** - Linear and non-linear regression

### API Reference
- **[Tensor Operations](docs/guides/layers.md)** - Complete tensor operation reference
- **[Training API](docs/guides/optimizers.md)** - Optimizers and loss functions
- **[Visualization](docs/guides/visualization.md)** - Training progress visualization

## 🎯 Examples

### Complete Examples
Explore the `/examples` directory for comprehensive examples:

- **[Basic Tensor Operations](examples/basic-tensor-operations/)** - Fundamental tensor operations and dynamic batch sizes
- **[Complete Training Pipeline](examples/complete-training-pipeline/)** - End-to-end MLP training with BatchNorm and Dropout
- **[CNN Image Classification](examples/cnn-image-classification/)** - Convolutional neural networks for image classification
- **[Model Serialization](examples/model-serialization/)** - Saving and loading trained models (ONNX format)

### Sample Applications
```bash
# Run the complete training demo
cd examples/complete-training-pipeline
go run main.go

# Explore CNN training
cd examples/cnn-image-classification
go run main.go

# Test basic tensor operations
cd examples/basic-tensor-operations
go run main.go

# Try model serialization
cd examples/model-serialization
go run main.go
```

## 🔧 Advanced Features

### Visualization Support
Go-Metal includes a [sidecar](https://github.com/tsawler/go-metal-sidecar-plots) service for real-time training visualization:

```bash
# Start visualization service
cd go-metal-sidecar-plots
docker-compose up -d

# Enable in your training code
trainer.EnableVisualization()
plottingService := training.NewPlottingService(training.DefaultPlottingServiceConfig())
plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
```

### Memory Management
Efficient GPU memory management with automatic pooling:

```go
// Automatic buffer reuse
config := metal_bridge.BufferAllocatorConfig{
    MaxPoolSize:     100,
    MaxTotalMemory:  1024 * 1024 * 1024, // 1GB
}
allocator := metal_bridge.NewBufferAllocator(device, config)
```

### Async GPU Operations
Non-blocking GPU execution for maximum performance:

```go
// Asynchronous operation chains
graph := tensor.NewGPUComputationGraph()
opID1, _ := graph.AddOperation("MatMul", []*tensor.Tensor{a, b}, nil, nil)
opID2, _ := graph.AddOperation("ReLU", []*tensor.Tensor{nil}, []tensor.OperationID{opID1}, nil)
result, _ := graph.WaitForOperation(opID2)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Install development dependencies:
   ```bash
   # Install task runner for documentation
   go install github.com/go-task/task/v3/cmd/task@latest
   
   # Generate documentation
   task docs
   ```
3. Make your changes and ensure tests pass:
   ```bash
   go test ./...
   ```

### Areas for Contribution
- More neural network layers (LSTM, Transformer blocks)
- Additional activation functions (Tanh, Swish, GELU)
- Model serialization formats
- Performance optimizations
- Advanced optimization techniques (learning rate scheduling, gradient clipping)

## 📄 License

Go-Metal is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

- **Apple** for Metal Performance Shaders and excellent Apple Silicon architecture
- **PyTorch** team for API design inspiration
- **Go team** for the fantastic Go language and runtime
- Open source contributors and the machine learning community

---

**Ready to build high-performance ML applications on Apple Silicon with Go?** 

[Get Started](docs/getting-started/quick-start.md) | [View Examples](app/) | [API Documentation](docs/)