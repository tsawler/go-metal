# Memory Management Guide

Efficient GPU memory management for high-performance training with go-metal.

## 🎯 Overview

Go-metal implements a sophisticated GPU memory management system designed for Apple Silicon's unified memory architecture. This guide covers best practices for optimal memory usage, performance optimization, and memory leak prevention.

### Key Features

- **🔄 Automatic Buffer Pooling**: Reduces allocation overhead through intelligent buffer reuse
- **📊 Reference Counting**: Prevents memory leaks with automatic cleanup
- **⚡ Persistent Buffers**: Pre-allocated tensors for maximum performance
- **🎯 Unified Memory**: Optimized for Apple Silicon's shared memory architecture
- **📈 Memory Statistics**: Real-time monitoring and debugging tools

## 🚀 Quick Start

> **⚠️ Important Note**: The global memory manager must be initialized before any memory operations can be performed. This is automatically done when you create a `ModelTrainer`, which initializes the Metal device and memory system. All examples in this guide include proper initialization.

### Basic Memory Management

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Build model
    inputShape := []int{32, 3, 32, 32}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(32, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    // Create trainer
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers for optimal performance
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Your training loop
    for epoch := 1; epoch <= 10; epoch++ {
        for step := 1; step <= 100; step++ {
            inputData, labelData := generateTrainingBatch()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training step failed: %v", err)
                continue
            }
            
            if step % 20 == 0 {
                fmt.Printf("Epoch %d, Step %d: Loss = %.6f\n", epoch, step, result.Loss)
            }
        }
    }
    
    // Memory statistics
    fmt.Printf("Memory stats: %v\n", memory.GetGlobalMemoryManager().Stats())
}

func generateTrainingBatch() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*3*32*32)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    for i := range labelData {
        labelData[i] = int32(rand.Intn(10))
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

## 🔧 Core Memory Components

### 1. Memory Manager

The global memory manager handles all GPU buffer allocation and pooling:

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Initialize the global memory manager by creating a trainer
    // This properly initializes the Metal device and memory system
    inputShape := []int{1, 10}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(5, true, "dense1").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     1,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Now we can safely use the global memory manager
    memManager := memory.GetGlobalMemoryManager()
    
    // Get buffer statistics
    stats := memManager.Stats()
    fmt.Printf("Memory manager stats: %v\n", stats)
    
    // Allocate a buffer directly (not recommended for normal use)
    bufferSize := 1024 * 1024 // 1MB
    buffer := memManager.AllocateBuffer(bufferSize)
    
    // Always release buffers when done
    defer memManager.ReleaseBuffer(buffer)
    
    fmt.Printf("Allocated buffer: %v\n", buffer)
}
```

### 2. GPU-Resident Tensors

Tensors are the primary way to work with GPU memory:

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Initialize the global memory manager by creating a trainer
    inputShape := []int{1, 10}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.AddDense(5, true, "dense1").Compile()
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     1,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Now we can create tensors - the memory manager is initialized
    shape := []int{4, 3, 32, 32}
    tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
    if err != nil {
        log.Fatalf("Failed to create tensor: %v", err)
    }
    defer tensor.Release() // Always release tensors
    
    // Tensor information
    fmt.Printf("Tensor shape: %v\n", tensor.Shape())
    fmt.Printf("Tensor size: %d bytes\n", tensor.Size())
    fmt.Printf("Data type: %v\n", tensor.DType())
    fmt.Printf("Device: %v\n", tensor.Device())
    fmt.Printf("Reference count: %d\n", tensor.RefCount())
    
    // Copy data to tensor
    inputData := make([]float32, 4*3*32*32)
    for i := range inputData {
        inputData[i] = float32(i) * 0.001
    }
    
    err = tensor.CopyFloat32Data(inputData)
    if err != nil {
        log.Fatalf("Failed to copy data: %v", err)
    }
    
    // Clone tensor (shares the same data, increments ref count)
    clonedTensor := tensor.Clone()
    defer clonedTensor.Release()
    
    fmt.Printf("Original ref count: %d\n", tensor.RefCount())
    fmt.Printf("Cloned ref count: %d\n", clonedTensor.RefCount())
    
    // Convert tensor to different data type
    int32Tensor, err := tensor.ConvertTo(memory.Int32)
    if err != nil {
        log.Fatalf("Failed to convert tensor: %v", err)
    }
    defer int32Tensor.Release()
    
    fmt.Printf("Converted tensor type: %v\n", int32Tensor.DType())
}
```

### 3. Reference Counting

Proper reference counting prevents memory leaks:

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Initialize the global memory manager by creating a trainer
    inputShape := []int{1, 10}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.AddDense(5, true, "dense1").Compile()
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     1,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Create tensor
    tensor, err := memory.NewTensor([]int{10, 10}, memory.Float32, memory.GPU)
    if err != nil {
        log.Fatalf("Failed to create tensor: %v", err)
    }
    
    fmt.Printf("Initial ref count: %d\n", tensor.RefCount())
    
    // Retain tensor (increment reference count)
    retainedTensor := tensor.Retain()
    fmt.Printf("After retain: %d\n", tensor.RefCount())
    
    // Clone tensor (also increments reference count)
    clonedTensor := tensor.Clone()
    fmt.Printf("After clone: %d\n", tensor.RefCount())
    
    // Release each reference
    retainedTensor.Release()
    fmt.Printf("After first release: %d\n", tensor.RefCount())
    
    clonedTensor.Release()
    fmt.Printf("After second release: %d\n", tensor.RefCount())
    
    tensor.Release()
    fmt.Printf("After final release: tensor is deallocated\n")
}
```

## ⚡ Performance Optimization

### 1. Persistent Buffers

Persistent buffers are required for training in go-metal and provide optimal performance by pre-allocating GPU memory:

```go
package main

import (
    "fmt"
    "log"
    "time"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/memory"
)

func main() {
    // Build model
    inputShape := []int{32, 3, 32, 32}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(64, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddDense(128, true, "fc1").
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Println("=== Persistent Buffers Demo ===")
    
    // Check initial memory state
    memManager := memory.GetGlobalMemoryManager()
    initialStats := memManager.Stats()
    fmt.Printf("Initial memory pools: %v\n", initialStats)
    
    // Enable persistent buffers - this is required for training
    fmt.Println("\nEnabling persistent buffers...")
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Check memory state after persistent buffers
    afterPersistentStats := memManager.Stats()
    fmt.Printf("After persistent buffers: %v\n", afterPersistentStats)
    
    // Run training to show efficient memory usage
    fmt.Println("\nRunning training with persistent buffers...")
    trainingTime := benchmarkTraining(trainer, inputShape)
    
    // Final memory state
    finalStats := memManager.Stats()
    fmt.Printf("\nFinal memory pools: %v\n", finalStats)
    fmt.Printf("Training performance: %.2f ms/batch\n", trainingTime.Seconds()*1000)
    fmt.Printf("Model parameters: %d\n", model.TotalParameters)
    
    // Show the benefit of persistent buffers
    fmt.Println("\n✅ Benefits of Persistent Buffers:")
    fmt.Println("   • Pre-allocated GPU memory reduces allocation overhead")
    fmt.Println("   • Tensors remain GPU-resident throughout training")
    fmt.Println("   • Eliminates CPU-GPU memory transfers during training")
    fmt.Println("   • Required for optimal performance in go-metal")
}

func benchmarkTraining(trainer *training.ModelTrainer, inputShape []int) time.Duration {
    numBatches := 20
    
    start := time.Now()
    for i := 0; i < numBatches; i++ {
        inputData, labelData := generateBenchmarkBatch()
        
        _, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
        if err != nil {
            log.Printf("Training failed: %v", err)
            continue
        }
        
        // Show progress occasionally
        if i%5 == 0 {
            fmt.Printf("   Batch %d/%d completed\n", i+1, numBatches)
        }
    }
    
    totalTime := time.Since(start)
    return totalTime / time.Duration(numBatches)
}

func generateBenchmarkBatch() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*3*32*32)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = float32(i%256) / 256.0
    }
    for i := range labelData {
        labelData[i] = int32(i % 10)
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

### 2. Buffer Pool Statistics

Monitor memory usage and optimize pool sizes:

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Build model
    inputShape := []int{32, 3, 32, 32}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(32, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Get initial memory statistics
    fmt.Println("=== Memory Statistics ===")
    memManager := memory.GetGlobalMemoryManager()
    initialStats := memManager.Stats()
    fmt.Printf("Initial memory stats: %v\n", initialStats)
    
    // Run training and monitor memory usage
    fmt.Println("\nRunning training...")
    for epoch := 1; epoch <= 3; epoch++ {
        for step := 1; step <= 20; step++ {
            inputData, labelData := generateTrainingBatch()
            
            _, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training step failed: %v", err)
                continue
            }
        }
        
        // Print memory statistics after each epoch
        stats := memManager.Stats()
        fmt.Printf("Epoch %d memory stats: %v\n", epoch, stats)
    }
    
    // Final memory statistics
    finalStats := memManager.Stats()
    fmt.Printf("\nFinal memory stats: %v\n", finalStats)
}

func generateTrainingBatch() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*3*32*32)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = float32(i%1000) / 1000.0
    }
    for i := range labelData {
        labelData[i] = int32(i % 10)
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

### 3. Memory Profiling

Monitor memory usage during training:

```go
package main

import (
    "fmt"
    "log"
    "runtime"
    "time"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("=== Memory Profiling ===")
    
    // Initial memory state
    var memBefore runtime.MemStats
    runtime.ReadMemStats(&memBefore)
    fmt.Printf("Initial Go memory: %.2f MB\n", float64(memBefore.Alloc)/1024/1024)
    
    // Create model and trainer
    inputShape := []int{32, 3, 32, 32}
    _, trainer := createModelAndTrainer(inputShape)
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err := trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Memory after model creation
    var memAfterModel runtime.MemStats
    runtime.ReadMemStats(&memAfterModel)
    fmt.Printf("After model creation: %.2f MB\n", float64(memAfterModel.Alloc)/1024/1024)
    
    // GPU memory statistics
    memManager := memory.GetGlobalMemoryManager()
    gpuStats := memManager.Stats()
    fmt.Printf("GPU memory pools: %v\n", gpuStats)
    
    // Training with memory monitoring
    fmt.Println("\nStarting training with memory monitoring...")
    
    for epoch := 1; epoch <= 2; epoch++ {
        epochStart := time.Now()
        
        for step := 1; step <= 10; step++ {
            inputData, labelData := generateTrainingBatch()
            
            _, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training step failed: %v", err)
                continue
            }
            
            // Monitor memory every 5 steps
            if step%5 == 0 {
                var memDuring runtime.MemStats
                runtime.ReadMemStats(&memDuring)
                gpuStatsDuring := memManager.Stats()
                
                fmt.Printf("Epoch %d, Step %d - Go: %.2f MB, GPU pools: %v\n",
                    epoch, step, float64(memDuring.Alloc)/1024/1024, gpuStatsDuring)
            }
        }
        
        epochTime := time.Since(epochStart)
        fmt.Printf("Epoch %d completed in %.2fs\n", epoch, epochTime.Seconds())
    }
    
    // Final memory state
    var memAfterTraining runtime.MemStats
    runtime.ReadMemStats(&memAfterTraining)
    finalGPUStats := memManager.Stats()
    
    fmt.Printf("\nFinal Go memory: %.2f MB\n", float64(memAfterTraining.Alloc)/1024/1024)
    fmt.Printf("Final GPU memory pools: %v\n", finalGPUStats)
    fmt.Printf("Memory growth: %.2f MB\n", 
        float64(memAfterTraining.Alloc-memBefore.Alloc)/1024/1024)
}

func createModelAndTrainer(inputShape []int) (*layers.ModelSpec, *training.ModelTrainer) {
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddConv2D(32, 3, 1, 1, true, "conv1").
        AddReLU("relu1").
        AddConv2D(64, 3, 2, 1, true, "conv2").
        AddReLU("relu2").
        AddDense(128, true, "fc1").
        AddReLU("relu3").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    
    return model, trainer
}

func generateTrainingBatch() ([]float32, *training.Int32Labels) {
    batchSize := 32
    inputData := make([]float32, batchSize*3*32*32)
    labelData := make([]int32, batchSize)
    
    for i := range inputData {
        inputData[i] = float32(i%256) / 256.0
    }
    for i := range labelData {
        labelData[i] = int32(i % 10)
    }
    
    labels, err := training.NewInt32Labels(labelData, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    return inputData, labels
}
```

## 🔍 Advanced Topics

### 1. Custom Buffer Pools

Create custom buffer pools for specialized use cases:

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("=== Custom Buffer Pool ===")
    
    // Initialize the global memory manager by creating a trainer
    inputShape := []int{1, 10}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.AddDense(5, true, "dense1").Compile()
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     1,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Create custom buffer pool
    bufferSize := 1024 * 1024 // 1MB buffers
    maxPoolSize := 10         // Maximum 10 buffers in pool
    
    pool := memory.NewBufferPool(bufferSize, maxPoolSize, memory.GPU)
    
    // Get buffer from pool
    buffer1, err := pool.Get()
    if err != nil {
        log.Fatalf("Failed to get buffer: %v", err)
    }
    
    buffer2, err := pool.Get()
    if err != nil {
        log.Fatalf("Failed to get buffer: %v", err)
    }
    
    // Check pool statistics
    available, allocated, maxSize := pool.Stats()
    fmt.Printf("Pool stats - Available: %d, Allocated: %d, Max: %d\n", 
        available, allocated, maxSize)
    
    // Return buffers to pool
    pool.Return(buffer1)
    pool.Return(buffer2)
    
    // Check statistics after return
    available, allocated, maxSize = pool.Stats()
    fmt.Printf("After return - Available: %d, Allocated: %d, Max: %d\n", 
        available, allocated, maxSize)
}
```

### 2. Tensor Operations with Memory Management

Efficient tensor operations demonstrating GPU-to-GPU memory transfers and different data types:

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("=== Efficient Tensor Operations ===")
    
    // Initialize the global memory manager by creating a trainer
    inputShape := []int{1, 10}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.AddDense(5, true, "dense1").Compile()
    if err != nil {
        log.Fatalf("Model compilation failed: %v", err)
    }
    
    config := training.TrainerConfig{
        BatchSize:     1,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Create source tensor
    srcTensor, err := memory.NewTensor([]int{4, 4}, memory.Float32, memory.GPU)
    if err != nil {
        log.Fatalf("Failed to create source tensor: %v", err)
    }
    defer srcTensor.Release()
    
    // Fill with data
    data := make([]float32, 16)
    for i := range data {
        data[i] = float32(i)
    }
    
    err = srcTensor.CopyFloat32Data(data)
    if err != nil {
        log.Fatalf("Failed to copy data: %v", err)
    }
    
    // Create destination tensor
    dstTensor, err := memory.NewTensor([]int{4, 4}, memory.Float32, memory.GPU)
    if err != nil {
        log.Fatalf("Failed to create destination tensor: %v", err)
    }
    defer dstTensor.Release()
    
    // Copy data between tensors (GPU-to-GPU)
    err = dstTensor.CopyFrom(srcTensor)
    if err != nil {
        log.Fatalf("Failed to copy tensor: %v", err)
    }
    
    // Verify data was copied
    copiedData, err := dstTensor.ToFloat32Slice()
    if err != nil {
        log.Fatalf("Failed to read tensor data: %v", err)
    }
    
    fmt.Printf("Original data: %v\n", data[:8])
    fmt.Printf("Copied data: %v\n", copiedData[:8])
    
    // Verify the copy was successful
    dataMatches := true
    for i := 0; i < len(data); i++ {
        if data[i] != copiedData[i] {
            dataMatches = false
            break
        }
    }
    
    if dataMatches {
        fmt.Printf("✅ GPU-to-GPU copy successful!\n")
    } else {
        fmt.Printf("❌ GPU-to-GPU copy failed!\n")
    }
    
    // Show tensor information
    fmt.Printf("Source tensor info: %s\n", srcTensor.String())
    fmt.Printf("Destination tensor info: %s\n", dstTensor.String())
    
    // Create an Int32 tensor separately to show different data types
    int32Data := make([]int32, 16)
    for i := range int32Data {
        int32Data[i] = int32(i)
    }
    
    int32Tensor, err := memory.NewTensor([]int{4, 4}, memory.Int32, memory.GPU)
    if err != nil {
        log.Fatalf("Failed to create Int32 tensor: %v", err)
    }
    defer int32Tensor.Release()
    
    err = int32Tensor.CopyInt32Data(int32Data)
    if err != nil {
        log.Fatalf("Failed to copy Int32 data: %v", err)
    }
    
    fmt.Printf("Int32 tensor type: %v\n", int32Tensor.DType())
    fmt.Printf("Int32 tensor shape: %v\n", int32Tensor.Shape())
}
```

## 📊 Memory Best Practices

### 1. Always Use Persistent Buffers

Persistent buffers are **required** for training in go-metal and provide optimal performance:

```go
// ✅ Required: Enable persistent buffers before training
err = trainer.EnablePersistentBuffers(inputShape)
if err != nil {
    log.Fatalf("Failed to enable persistent buffers: %v", err)
}

// Now you can train
result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)

// ❌ Error: Training without persistent buffers will fail
// result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
// // This will return: "persistent buffers not enabled - call EnablePersistentBuffers() first"
```

### 2. Proper Tensor Cleanup

Always release tensors when done:

```go
// ✅ Good: Proper cleanup
tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
if err != nil {
    return err
}
defer tensor.Release() // Always release

// ❌ Bad: Memory leak
tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
// Missing Release() call
```

### 3. Monitor Memory Usage

Regularly check memory statistics:

```go
// ✅ Good: Monitor memory usage
memManager := memory.GetGlobalMemoryManager()
stats := memManager.Stats()
fmt.Printf("Memory usage: %v\n", stats)

// Log memory usage periodically during training
if step%100 == 0 {
    fmt.Printf("Step %d memory: %v\n", step, memManager.Stats())
}
```

### 4. Use Reference Counting Correctly

Understand when to retain and release:

```go
// ✅ Good: Proper reference counting
original := tensor.Clone()  // Increments ref count
defer original.Release()   // Decrements ref count

retained := tensor.Retain() // Increments ref count
defer retained.Release()   // Decrements ref count

// ❌ Bad: Double release
tensor.Release()
tensor.Release() // This may cause issues
```

## 🎯 Performance Tips

### 1. Pre-allocate Buffers

Pre-allocate buffers for known sizes:

```go
// ✅ Good: Pre-allocation
err = trainer.EnablePersistentBuffers(inputShape)
```

### 2. Use Appropriate Data Types

Choose the right data type for your use case:

```go
// ✅ Good: Use Float32 for most ML operations
tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)

// ✅ Good: Use Int32 for labels
labelTensor, err := memory.NewTensor(labelShape, memory.Int32, memory.GPU)

// ✅ Good: Use Float16 for mixed precision (when available)
fp16Tensor, err := memory.NewTensor(shape, memory.Float16, memory.GPU)
```

### 3. Minimize CPU-GPU Transfers

Keep data on GPU when possible:

```go
// ✅ Good: GPU-to-GPU operations
err = dstTensor.CopyFrom(srcTensor)

// ❌ Bad: Unnecessary CPU transfer
data, _ := srcTensor.ToFloat32Slice()
dstTensor.CopyFloat32Data(data)
```

## 🐛 Troubleshooting

### Common Issues

**Memory Leaks**:
```go
// Check for unreleased tensors
memManager := memory.GetGlobalMemoryManager()
stats := memManager.Stats()
fmt.Printf("Memory pools: %v\n", stats)

// Look for growing buffer counts
```

**Out of Memory**:
```go
// Reduce batch size
config.BatchSize = 16 // Instead of 32

// Enable persistent buffers
err = trainer.EnablePersistentBuffers(inputShape)
```

**Performance Issues**:
```go
// Always enable persistent buffers
err = trainer.EnablePersistentBuffers(inputShape)

// Monitor memory statistics
stats := memory.GetGlobalMemoryManager().Stats()
```

## 🎯 Summary

The go-metal memory management system provides:

- **🔄 Automatic Pooling**: Intelligent buffer reuse for optimal performance
- **📊 Reference Counting**: Automatic memory cleanup prevents leaks
- **⚡ Persistent Buffers**: Pre-allocated tensors for maximum speed
- **🎯 Unified Memory**: Optimized for Apple Silicon architecture
- **📈 Monitoring**: Real-time statistics for debugging and optimization

### Key Takeaways

1. **Always use persistent buffers** for training loops
2. **Properly manage tensor lifetimes** with defer statements
3. **Monitor memory usage** during development
4. **Choose appropriate data types** for your use case
5. **Minimize CPU-GPU transfers** for better performance

---

*Efficient memory management is crucial for high-performance ML training on Apple Silicon.*