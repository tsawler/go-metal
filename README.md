# Go-Metal

A high-performance deep learning library for Go that leverages Apple's Metal Performance Shaders (specifically [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph?language=objc) for GPU acceleration on Apple Silicon.

[![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)](https://golang.org)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-brightgreen.svg)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

## 🚀 Overview

Go-Metal is a PyTorch-inspired deep learning library built specifically for Apple Silicon. It provides GPU-accelerated tensor operations, automatic differentiation, and a complete training pipeline - all with the safety and simplicity of Go. It's only dependency apart from the standard library is google.golang.org/protobuf, so the package can read and write ONNX files.

Why did I build this? I wanted to increase my knowledge in Machine Learning, and building something like this from scratch seemed like a good way to take a deep dive into that topic. 

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
- **Metal Performance**: Integration with Apple's optimized MPS framework (MPS and MPSGraph)
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
    "github.com/tsawler/go-metal/layers"
)

func main() {
    // Basic tensor operations in go-metal are performed through neural network layers
    // This example demonstrates the actual tensor operations available
    
    fmt.Println("=== Basic Tensor Operations in Go-Metal ===")
    
    // Create input data (this represents our "tensors")
    batchSize := 4
    inputSize := 6
    inputData := make([]float32, batchSize*inputSize)
    
    // Fill with sample data
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0  // Random values between -1 and 1
    }
    
    fmt.Printf("Input data shape: [%d, %d]\n", batchSize, inputSize)
    
    // 1. Matrix multiplication (through Dense layer)
    builder := layers.NewModelBuilder([]int{batchSize, inputSize})
    
    // This Dense layer performs: output = input * weight + bias
    // Which is matrix multiplication + addition
    model, _ := builder.
        AddDense(3, true, "matrix_multiply").  // 6->3 matrix multiplication
        Compile()
    
    fmt.Printf("Created matrix multiplication layer: %dx%d -> %dx%d\n", 
        batchSize, inputSize, batchSize, 3)
    fmt.Printf("Matrix model has %d parameters\n", model.TotalParameters)
    
    // 2. Element-wise operations (through activation layers)
    activationBuilder := layers.NewModelBuilder([]int{batchSize, inputSize})
    
    activationModel, _ := activationBuilder.
        AddReLU("relu_activation").           // Element-wise: max(0, x)
        AddDense(inputSize, false, "dense").  // Matrix multiplication
        AddSigmoid("sigmoid_activation").     // Element-wise: 1/(1+exp(-x))
        Compile()
    
    fmt.Printf("Created activation layers for element-wise operations\n")
    fmt.Printf("Activation model has %d parameters\n", activationModel.TotalParameters)
    
    // 3. Available tensor operations
    fmt.Println("\n=== Available Tensor Operations ===")
    fmt.Println("✓ Matrix Multiplication (Dense layers)")
    fmt.Println("✓ Element-wise Addition (bias in Dense layers)")
    fmt.Println("✓ Element-wise Activations (ReLU, Sigmoid, Tanh, etc.)")
    fmt.Println("✓ 2D Convolution (Conv2D layers)")
    fmt.Println("✓ Tensor Reshaping (automatic in layer transitions)")
    fmt.Println("✓ Batch Normalization (BatchNorm layers)")
    fmt.Println("✓ Dropout (element-wise masking)")
    
    fmt.Println("\nNote: Go-Metal focuses on neural network operations.")
    fmt.Println("Tensor math operations are performed within neural network layers.")
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
	numSamples := 100  // Smaller dataset for demonstration
	inputFeatures := 10 // Simpler input for demonstration
	outputClasses := 2  // Binary classification

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
	batchSize := 8  // Smaller batch size for demonstration
	builder := layers.NewModelBuilder([]int{batchSize, inputFeatures})
	model, err := builder.
		AddDense(16, true, "fc1").
		AddReLU("relu1").
		AddDense(outputClasses, true, "output").
		Compile()

	if err != nil {
		log.Fatalf("Failed to build model: %v", err)
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

	// Create trainer
	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create trainer: %v", err)
	}

	fmt.Printf("Model created successfully with %d parameters\n", model.TotalParameters)
	fmt.Printf("Training configured with batch size %d and learning rate %.4f\n", config.BatchSize, config.LearningRate)
	fmt.Println("Ready for training!")

	// Cleanup
	trainer.Cleanup()
}
```

### Convolutional Neural Networks

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/layers"
)

func main() {
    // Create a CNN for 32x32 RGB image classification
    batchSize := 8
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
    
    fmt.Printf("CNN model built successfully!\n")
    fmt.Printf("Total parameters: %d\n", model.TotalParameters)
    fmt.Printf("Model has %d layers\n", len(model.Layers))
    
    // Print layer information
    for i, layer := range model.Layers {
        fmt.Printf("Layer %d: %s\n", i+1, layer.Name)
    }
    
    fmt.Println("CNN architecture ready for training!")
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
- **[Memory Management Guide](docs/guides/memory-management.md)** - GPU memory optimization and best practices
- **[Checkpoints Guide](docs/guides/checkpoints.md)** - Model saving, loading, and resume training

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

```go
package main

import (
    "fmt"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Build a simple model
    builder := layers.NewModelBuilder([]int{8, 10})
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").
        Compile()
    
    config := training.TrainerConfig{
        BatchSize: 8,
        LearningRate: 0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1: 0.9,
        Beta2: 0.999,
        Epsilon: 1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        fmt.Printf("Error creating trainer: %v\n", err)
        return
    }
    
    fmt.Println("Model ready for visualization!")
    fmt.Printf("Model ready with %d parameters\n", model.TotalParameters)
    
    // Visualization features can be enabled:
    // trainer.EnableVisualization()
    // trainer.EnablePlottingService()
    
    // Optional: Configure custom plotting service
    // plotConfig := training.DefaultPlottingServiceConfig()
    // trainer.ConfigurePlottingService(plotConfig)
    
    fmt.Println("Visualization features available!")
    
    // Cleanup
    if trainer != nil {
        trainer.Cleanup()
    }
}
```

### Memory Management
The library automatically manages GPU memory with efficient buffer pooling. Memory is handled transparently when you use the high-level training API:

```go
package main

import (
    "fmt"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Build a simple model
    builder := layers.NewModelBuilder([]int{8, 10})
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").
        Compile()
    
    // Memory is automatically managed by the trainer
    config := training.TrainerConfig{
        BatchSize: 8,
        LearningRate: 0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1: 0.9,
        Beta2: 0.999,
        Epsilon: 1e-8,
        EngineType: training.Auto, // Auto-selects best engine
    }
    
    // The trainer handles all GPU memory allocation and pooling internally
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        fmt.Printf("Error creating trainer: %v\n", err)
        return
    }
    
    fmt.Println("Memory management is handled automatically!")
    fmt.Printf("Model uses %d parameters\n", model.TotalParameters)
    
    // Always cleanup when done
    if trainer != nil {
        trainer.Cleanup()
    }
}
```

### Async GPU Operations
The library uses asynchronous GPU execution internally for maximum performance. Operations are automatically optimized:

```go
package main

import (
    "fmt"
    "log"
    "math"
    "math/rand"
    
    "github.com/tsawler/go-metal/async"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/checkpoints"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

// generateSineWaveData creates a synthetic dataset for demonstrating async operations
func generateSineWaveData(numSamples int) ([]float32, []float32) {
    inputs := make([]float32, numSamples)
    outputs := make([]float32, numSamples)
    
    for i := 0; i < numSamples; i++ {
        x := float32(i) * 2.0 * math.Pi / float32(numSamples)
        inputs[i] = x
        outputs[i] = float32(math.Sin(float64(x))) + (rand.Float32()-0.5)*0.1 // Add noise
    }
    
    return inputs, outputs
}

// SimpleDataSource implements the async.DataSource interface
type SimpleDataSource struct {
    inputs  []float32
    outputs []float32
    index   int
}

func (s *SimpleDataSource) GetBatch(batchSize int) ([]float32, []int, []float32, []int, error) {
    remaining := len(s.inputs) - s.index
    if remaining <= 0 {
        return nil, nil, nil, nil, fmt.Errorf("no more data")
    }
    
    actualBatchSize := batchSize
    if remaining < batchSize {
        actualBatchSize = remaining
    }
    
    // Create batch data
    batchInputs := make([]float32, batchSize)
    batchOutputs := make([]float32, batchSize)
    
    // Copy actual data
    for i := 0; i < actualBatchSize; i++ {
        batchInputs[i] = s.inputs[s.index+i]
        batchOutputs[i] = s.outputs[s.index+i]
    }
    
    // Pad with zeros if needed
    for i := actualBatchSize; i < batchSize; i++ {
        batchInputs[i] = 0
        batchOutputs[i] = 0
    }
    
    s.index += actualBatchSize
    
    return batchInputs, []int{batchSize, 1}, batchOutputs, []int{batchSize, 1}, nil
}

func (s *SimpleDataSource) Size() int {
    return len(s.inputs)
}

func (s *SimpleDataSource) Reset() error {
    s.index = 0
    return nil
}

func main() {
    fmt.Println("🚀 Go-Metal Complete Example: Async GPU Operations & Full Training Pipeline")
    fmt.Println("=========================================================================")
    
    // Initialize Metal device
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    
    // Initialize global memory manager (returns void)
    memory.InitializeGlobalMemoryManager(device)
    
    // Generate synthetic data
    fmt.Println("\n📊 Generating synthetic sine wave dataset...")
    inputs, outputs := generateSineWaveData(1000)
    dataSource := &SimpleDataSource{inputs: inputs, outputs: outputs}
    fmt.Printf("✅ Generated %d samples\n", len(inputs))
    
    // Build a model with various layer types
    fmt.Println("\n🧠 Building neural network model...")
    batchSize := 32
    builder := layers.NewModelBuilder([]int{batchSize, 1})
    
    model, err := builder.
        AddDense(64, true, "hidden1").
        AddReLU("relu1").
        AddDropout(0.2, "dropout1").
        AddDense(128, true, "hidden2").
        AddLeakyReLU(0.01, "leaky_relu").
        AddBatchNorm(128, 1e-5, 0.9, true, "batch_norm"). // Fixed: added numFeatures and affine
        AddDense(64, true, "hidden3").
        AddELU(1.0, "elu").
        AddDense(1, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to compile model: %v", err)
    }
    
    fmt.Printf("✅ Model compiled with %d parameters\n", model.TotalParameters)
    fmt.Printf("📐 Architecture: %d layers\n", len(model.Layers))
    
    // Configure training with advanced options
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
        WeightDecay:   0.0001,
        EngineType:    training.Dynamic,
        ProblemType:   training.Regression,
        LossFunction:  training.MeanSquaredError,
    }
    
    // Create trainer
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers for better performance
    trainer.EnablePersistentBuffers([]int{batchSize, 1})
    
    // Set up learning rate scheduler
    scheduler := training.NewCosineAnnealingLRScheduler(10, 0.0001) // Fixed: tMax is int, etaMin is float64
    trainer.SetLRScheduler(scheduler)
    
    fmt.Println("\n⚡ Demonstrating Async GPU Operations...")
    
    // Create async data loader for efficient GPU utilization
    memManager := memory.GetGlobalMemoryManager()
    asyncConfig := async.AsyncDataLoaderConfig{
        BatchSize:     batchSize,
        PrefetchDepth: 3,
        Workers:       2,
        MemoryManager: memManager,
    }
    
    asyncLoader, err := async.NewAsyncDataLoader(dataSource, asyncConfig)
    if err != nil {
        log.Printf("Note: Async data loader not available in this example")
    } else {
        // Start async loading
        asyncLoader.Start()
        defer asyncLoader.Stop()
        
        fmt.Println("✅ Async data loader started with 2 workers and prefetch depth of 3")
    }
    
    // Demonstrate command buffer pooling (if available)
    fmt.Println("\n🔄 Command Buffer Pool Management:")
    fmt.Println("  - Automatic command buffer reuse")
    fmt.Println("  - Async GPU operations handled internally")
    fmt.Println("  - Efficient memory transfers via staging buffers")
    
    // Training with progress tracking
    fmt.Println("\n🎯 Starting training with async GPU operations...")
    epochs := 10
    stepsPerEpoch := dataSource.Size() / batchSize
    
    session := trainer.CreateTrainingSession("Sine Wave Regression", epochs, stepsPerEpoch, 0)
    session.StartTraining()
    
    bestLoss := float64(math.Inf(1))
    
    for epoch := 0; epoch < epochs; epoch++ {
        trainer.SetEpoch(epoch)
        session.StartEpoch(epoch + 1)
        
        dataSource.Reset()
        epochLoss := 0.0
        
        for step := 0; step < stepsPerEpoch; step++ {
            // Get batch data
            batchInputs, inputShape, batchOutputs, outputShape, err := dataSource.GetBatch(batchSize)
            if err != nil {
                break
            }
            
            // Create labels for regression
            labels, _ := training.NewFloat32Labels(batchOutputs, outputShape)
            
            // Train on batch
            result, err := trainer.TrainBatchUnified(batchInputs, inputShape, labels)
            if err != nil {
                log.Printf("Training step failed: %v", err)
                continue
            }
            
            epochLoss += float64(result.Loss)
            
            if step < stepsPerEpoch {
                session.UpdateTrainingProgress(step+1, float64(result.Loss), result.Accuracy)
            }
        }
        
        session.FinishTrainingEpoch()
        
        avgLoss := epochLoss / float64(stepsPerEpoch)
        
        // Update learning rate
        trainer.StepSchedulerWithMetric(avgLoss)
        currentLR := trainer.GetCurrentLearningRate()
        
        fmt.Printf("Epoch %d/%d | Loss: %.6f | LR: %.6f\n", epoch+1, epochs, avgLoss, currentLR)
        
        if avgLoss < bestLoss {
            bestLoss = avgLoss
        }
        
        session.PrintEpochSummary()
    }
    
    fmt.Printf("\n🎉 Training completed! Best loss: %.6f\n", bestLoss)
    
    // Demonstrate model saving
    fmt.Println("\n💾 Saving trained model...")
    
    parameterTensors := trainer.GetParameterTensors()
    weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, model)
    if err != nil {
        log.Printf("Failed to extract weights: %v", err)
    } else {
        checkpoint := &checkpoints.Checkpoint{
            ModelSpec: model,
            Weights:   weights,
            TrainingState: checkpoints.TrainingState{
                Epoch:        epochs,
                LearningRate: trainer.GetCurrentLearningRate(),
                BestLoss:     float32(bestLoss),
            },
            Metadata: checkpoints.CheckpointMetadata{
                Version:     "1.0.0",
                Framework:   "go-metal",
                Description: "Complete example model with async operations",
                Tags:        []string{"async", "gpu", "regression"},
            },
        }
        
        // Save as both JSON and ONNX
        jsonSaver := checkpoints.NewCheckpointSaver(checkpoints.FormatJSON)
        jsonSaver.SaveCheckpoint(checkpoint, "demo_model.json")
        
        onnxSaver := checkpoints.NewCheckpointSaver(checkpoints.FormatONNX)
        onnxSaver.SaveCheckpoint(checkpoint, "demo_model.onnx")
        
        fmt.Println("✅ Model saved in JSON and ONNX formats")
    }
    
    // Display memory and performance statistics
    stats := trainer.GetStats()
    fmt.Println("\n📊 Performance Statistics:")
    fmt.Printf("  - Total training steps: %d\n", stats.CurrentStep)
    fmt.Printf("  - Model parameters: %d\n", stats.ModelParameters)
    fmt.Printf("  - Memory pool stats: %v\n", stats.MemoryPoolStats)
    
    if asyncLoader != nil {
        asyncStats := asyncLoader.Stats()
        fmt.Printf("\n⚡ Async Data Loader Statistics:\n")
        fmt.Printf("  - Batches produced: %d\n", asyncStats.BatchesProduced)
        fmt.Printf("  - Queue capacity: %d\n", asyncStats.QueueCapacity)
        fmt.Printf("  - Workers: %d\n", asyncStats.Workers)
    }
    
    fmt.Println("\n✅ Go-Metal Complete Example Finished!")
    fmt.Println("This example demonstrated:")
    fmt.Println("  • Async GPU operations with automatic optimization")
    fmt.Println("  • Various layer types (Dense, ReLU, LeakyReLU, ELU, Dropout, BatchNorm)")
    fmt.Println("  • Advanced optimizers with learning rate scheduling")
    fmt.Println("  • Model checkpointing in JSON and ONNX formats")
    fmt.Println("  • Memory management and performance monitoring")
    fmt.Println("  • Async data loading for efficient GPU utilization")
}
```

## 🤝 Contributing

Contributions are welcome.

### Development Setup
1. Fork the repository
2. Install development dependencies:
   ```bash
   # Install task for documentation
   go install github.com/go-task/task/v3/cmd/task@latest

   # Install godocdown
   go install github.com/robertkrimen/godocdown/godocdown@latest
   
   # This generates documentation (docs.md) in every directory in go-metal
   task docs
   ```
3. Make your changes and ensure tests pass:
   ```bash
   go test ./...
   ```

### Areas for Contribution
- More neural network layers (LSTM, Transformer blocks)
- Additional activation functions (GELU)
- Performance optimizations
- Advanced optimization techniques (learning rate scheduling, gradient clipping)

## 📄 License

Go-Metal is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

- **Apple** for Metal Performance Shaders and excellent Apple Silicon architecture
- **PyTorch** team for API design inspiration
- **Go team** for the fantastic Go language and runtime
- Open source contributors and the machine learning community
- The CNN classification example uses the Kaggle Dogs vs. Cats imageset: Will Cukierski. Dogs vs. Cats. [https://kaggle.com/competitions/dogs-vs-cats](https://kaggle.com/competitions/dogs-vs-cats), 2013. Kaggle.

---

**Ready to build high-performance ML applications on Apple Silicon with Go?** 

[Get Started](docs/getting-started/quick-start.md) | [View Examples](examples/) | [API Documentation](docs/)