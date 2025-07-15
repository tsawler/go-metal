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
- **Activation Functions**: ReLU, Softmax, LeakyReLU, ELU, Sigmoid, Tanh, Swish with Metal implementations

### Deep Learning Components
- **Neural Network Layers**: Linear, Conv2D, MaxPool2D, BatchNorm, Dropout, ReLU, Softmax, LeakyReLU, ELU, Sigmoid, Tanh, Swish (Flattening is handled implicitly by Dense layers)
- **Optimizers**: SGD, Adam, AdaGrad, RMSprop, AdaDelta, NAdam, L-BFGS
- **Loss Functions**: CrossEntropy, SparseCrossEntropy, BinaryCrossEntropy, BCEWithLogits, CategoricalCrossEntropy (for classification); Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber (for regression)
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
       "github.com/tsawler/go-metal/layers"
       "github.com/tsawler/go-metal/training"
   )
   
   func main() {
       // Create a simple neural network
       builder := layers.NewModelBuilder([]int{1, 10}) // batch_size=1, features=10
       model, _ := builder.
           AddDense(5, true, "dense1").
           AddReLU("relu1").
           AddDense(1, true, "output").
           Compile()
       
       fmt.Printf("Model created with %d parameters\n", model.TotalParameters)
   }
   ```

## 🚦 Quick Start

### Basic Tensor Operations

```go
package main

import (
    "fmt"
    "math/rand"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create synthetic data
    batchSize := 32
    inputData := make([]float32, batchSize*10)
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    
    // Build a model
    builder := layers.NewModelBuilder([]int{batchSize, 10})
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").
        Compile()
    
    // Configure training
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    fmt.Printf("Model ready for training with %d parameters\n", model.TotalParameters)
}
```

### Neural Network Training

```go
package main

import (
	"fmt"
	"log"
	"math/rand"
	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

func main() {
	// Create synthetic training data
	numSamples := 1000
	inputFeatures := 784 // e.g., flattened 28x28 images
	outputClasses := 10  // e.g., digits 0-9

	// Generate random input data
	trainData := make([]float32, numSamples*inputFeatures)
	for i := range trainData {
		trainData[i] = rand.Float32()
	}

	// Generate random labels (class indices)
	trainLabels := make([]int32, numSamples)
	for i := range trainLabels {
		trainLabels[i] = int32(rand.Intn(outputClasses))
	}

	// Build the model using ModelBuilder
	builder := layers.NewModelBuilder([]int{32, inputFeatures}) // batch_size=32
	model, err := builder.
		AddDense(128, true, "fc1").
		AddReLU("relu1").
		AddDense(64, true, "fc2").
		AddReLU("relu2").
		AddDense(outputClasses, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build model: %v", err)
	}

	// Configure training
	config := training.TrainerConfig{
		BatchSize:     32,
		LearningRate:  0.001,
		OptimizerType: cgo_bridge.Adam,
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
		LossFunction:  training.CrossEntropy,
		ProblemType:   training.Classification,
	}

	// Create trainer
	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create trainer: %v", err)
	}

	// Train the model
	loss, accuracy := trainer.TrainStep(trainData, trainLabels)
	fmt.Printf("Loss: %.4f, Accuracy: %.2f%%\n", loss, accuracy*100)

	fmt.Println("Training complete!")
}
```

### Convolutional Neural Networks

```go
package main

import (
    "log"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/cgo_bridge"
)

func main() {
    // Create a CNN for 32x32 RGB image classification
    batchSize := 32
    inputShape := []int{batchSize, 3, 32, 32} // NCHW format
    
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        // First convolutional block
        AddConv2D(32, 3, 1, 1, true, "conv1").     // 32 filters, 3x3 kernel
        AddReLU("relu1").
        AddConv2D(32, 3, 2, 1, true, "conv2").     // Stride 2 for downsampling
        AddReLU("relu2").
        
        // Second convolutional block
        AddConv2D(64, 3, 1, 1, true, "conv3").     // 64 filters
        AddReLU("relu3").
        AddConv2D(64, 3, 2, 1, true, "conv4").     // Stride 2 for downsampling
        AddReLU("relu4").
        
        // Classification head
        AddDense(128, true, "fc1").                // Dense automatically handles flattening
        AddReLU("relu5").
        AddDropout(0.5, "dropout").
        AddDense(10, true, "output").              // 10 classes
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to build CNN: %v", err)
    }
    
    // Configure training
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
        LossFunction:  training.CrossEntropy,
        ProblemType:   training.Classification,
    }
    
    // Training would proceed as in the previous example...
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

- **`memory.CPU`** or **`cgo_bridge.CPU`**: Traditional CPU tensors for compatibility
- **`memory.GPU`** or **`cgo_bridge.GPU`**: Temporary GPU tensors (automatically copy back to CPU)
- **`memory.PersistentGPU`** or **`cgo_bridge.PersistentGPU`**: High-performance GPU-resident tensors for training

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
trainer.EnablePlottingService()

# Optional: Configure custom plotting service
config := training.DefaultPlottingServiceConfig()
trainer.ConfigurePlottingService(config)
```

### Memory Management
The library automatically manages GPU memory with efficient buffer pooling. Memory is handled transparently when you use the high-level training API:

```go
// Memory is automatically managed by the trainer
config := training.TrainerConfig{
    BatchSize: 32,
    LearningRate: 0.001,
    OptimizerType: cgo_bridge.Adam,
    // EngineType: training.Auto, // Auto-selects best engine
}

// The trainer handles all GPU memory allocation and pooling internally
trainer, _ := training.NewModelTrainer(model, config)
```

### Async GPU Operations
The library uses asynchronous GPU execution internally for maximum performance. Operations are automatically optimized:

```go
// Async execution is handled automatically by the trainer
trainer, _ := training.NewModelTrainer(model, config)

// Training steps are automatically pipelined for GPU efficiency
loss, accuracy := trainer.TrainStep(data, labels)

// For custom async operations, use the async package
import "github.com/tsawler/go-metal/async"

// Create a command buffer pool for efficient GPU command submission
pool := async.NewCommandBufferPool(device, maxBuffers)
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
- Additional activation functions (GELU)
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
- The CNN classification example uses the Kaggle Dogs vs. Cats imageset: Will Cukierski. Dogs vs. Cats. https://kaggle.com/competitions/dogs-vs-cats, 2013. Kaggle.

---

**Ready to build high-performance ML applications on Apple Silicon with Go?** 

[Get Started](docs/getting-started/quick-start.md) | [View Examples](app/) | [API Documentation](docs/)