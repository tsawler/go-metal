# Performance Optimization Guide

Practical optimization techniques for maximizing go-metal performance on Apple Silicon.

## 🎯 Overview

This guide provides actionable performance optimization strategies with working examples. All code examples are functional and can be run to measure actual performance improvements.

## ⚡ Basic Performance Monitoring

### Simple Throughput Measurement

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "time"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    
    // Create a simple MLP for benchmarking
    inputShape := []int{32, 784} // Batch size 32, 784 features (28x28)
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(128, true, "hidden1").
        AddReLU("relu1").
        AddDense(64, true, "hidden2").
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Create trainer
    trainerConfig := training.TrainerConfig{
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
    
    trainer, err := training.NewModelTrainer(model, trainerConfig)
    if err != nil {
        log.Fatalf("Failed to create trainer: %v", err)
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers for better performance
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Generate sample data
    batchSize := 32
    inputSize := 784
    numClasses := 10
    
    inputs := make([]float32, batchSize*inputSize)
    labels := make([]int32, batchSize)
    
    // Fill with random data
    for i := range inputs {
        inputs[i] = rand.Float32()
    }
    for i := range labels {
        labels[i] = int32(rand.Intn(numClasses))
    }
    
    labelTensor, err := training.NewInt32Labels(labels, []int{batchSize})
    if err != nil {
        log.Fatalf("Failed to create label tensor: %v", err)
    }
    
    // Measure throughput
    fmt.Println("🚀 Measuring Training Throughput")
    fmt.Println("================================")
    
    numIterations := 100
    startTime := time.Now()
    
    for i := 0; i < numIterations; i++ {
        result, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
        if err != nil {
            log.Printf("Training step %d failed: %v", i, err)
            continue
        }
        
        if i%20 == 0 {
            fmt.Printf("Step %d: Loss = %.4f\n", i, result.Loss)
        }
    }
    
    elapsed := time.Since(startTime)
    samplesPerSecond := float64(numIterations*batchSize) / elapsed.Seconds()
    
    fmt.Printf("\n📊 Performance Results:\n")
    fmt.Printf("   Total time: %.2f seconds\n", elapsed.Seconds())
    fmt.Printf("   Samples/second: %.1f\n", samplesPerSecond)
    fmt.Printf("   Batch size: %d\n", batchSize)
    fmt.Printf("   Time per batch: %.2f ms\n", elapsed.Seconds()*1000/float64(numIterations))
}
```

### Memory Usage Profiling

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "runtime"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    
    fmt.Println("💾 Memory Usage Analysis")
    fmt.Println("========================")
    
    // Test different batch sizes
    batchSizes := []int{8, 16, 32, 64, 128}
    
    for _, batchSize := range batchSizes {
        fmt.Printf("\n🔍 Testing batch size: %d\n", batchSize)
        
        // Force garbage collection before test
        runtime.GC()
        
        var memBefore runtime.MemStats
        runtime.ReadMemStats(&memBefore)
        
        // Create model with current batch size
        inputShape := []int{batchSize, 784}
        builder := layers.NewModelBuilder(inputShape)
        
        model, err := builder.
            AddDense(256, true, "hidden1").
            AddReLU("relu1").
            AddDense(128, true, "hidden2").
            AddReLU("relu2").
            AddDense(10, true, "output").
            Compile()
        
        if err != nil {
            log.Printf("Failed to create model: %v", err)
            continue
        }
        
        trainerConfig := training.TrainerConfig{
            BatchSize:     batchSize,
            LearningRate:  0.001,
            OptimizerType: cgo_bridge.Adam,
            EngineType:    training.Dynamic,
            LossFunction:  training.SparseCrossEntropy,
            ProblemType:   training.Classification,
            Beta1:         0.9,
            Beta2:         0.999,
            Epsilon:       1e-8,
        }
        
        trainer, err := training.NewModelTrainer(model, trainerConfig)
        if err != nil {
            log.Printf("Failed to create trainer: %v", err)
            continue
        }
        
        // Enable persistent buffers
        err = trainer.EnablePersistentBuffers(inputShape)
        if err != nil {
            log.Printf("Failed to enable persistent buffers: %v", err)
            trainer.Cleanup()
            continue
        }
        
        // Measure memory after model creation
        var memAfter runtime.MemStats
        runtime.ReadMemStats(&memAfter)
        
        memUsedMB := float64(memAfter.Alloc-memBefore.Alloc) / 1024.0 / 1024.0
        
        // Run a few training steps to measure peak usage
        inputs := make([]float32, batchSize*784)
        labels := make([]int32, batchSize)
        
        for i := range inputs {
            inputs[i] = rand.Float32()
        }
        for i := range labels {
            labels[i] = int32(rand.Intn(10))
        }
        
        labelTensor, err := training.NewInt32Labels(labels, []int{batchSize})
        if err != nil {
            log.Printf("Failed to create label tensor: %v", err)
            trainer.Cleanup()
            continue
        }
        
        for i := 0; i < 10; i++ {
            _, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
            if err != nil {
                log.Printf("Training step %d failed: %v", i, err)
                break
            }
        }
        
        var memPeak runtime.MemStats
        runtime.ReadMemStats(&memPeak)
        
        peakUsedMB := float64(memPeak.Alloc-memBefore.Alloc) / 1024.0 / 1024.0
        
        fmt.Printf("   Model creation: %.1f MB\n", memUsedMB)
        fmt.Printf("   Peak during training: %.1f MB\n", peakUsedMB)
        fmt.Printf("   Memory per sample: %.2f KB\n", peakUsedMB*1024/float64(batchSize))
        
        trainer.Cleanup()
    }
}
```

## 🚀 Batch Size Optimization

### Automatic Batch Size Finder

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "runtime"
    "time"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    
    fmt.Println("📦 Optimal Batch Size Finder")
    fmt.Println("=============================")
    
    optimalBatchSize := findOptimalBatchSize()
    fmt.Printf("\n🎯 Recommended batch size: %d\n", optimalBatchSize)
    
    // Demonstrate performance with optimal batch size
    demonstrateOptimalPerformance(optimalBatchSize)
}

func findOptimalBatchSize() int {
    batchSizes := []int{8, 16, 32, 64, 128, 256}
    bestBatchSize := 8
    bestThroughput := 0.0
    
    fmt.Println("\n🔍 Testing different batch sizes...")
    fmt.Printf("%-10s | %-15s | %-12s | %-10s\n", "Batch Size", "Samples/Sec", "Memory (MB)", "Success")
    fmt.Println("-----------|-----------------|--------------|----------")
    
    for _, batchSize := range batchSizes {
        throughput, memUsage, success := benchmarkBatchSize(batchSize)
        
        status := "✅"
        if !success {
            status = "❌"
        }
        
        fmt.Printf("%-10d | %-15.1f | %-12.1f | %-10s\n", 
                   batchSize, throughput, memUsage, status)
        
        if success && throughput > bestThroughput {
            bestThroughput = throughput
            bestBatchSize = batchSize
        }
    }
    
    return bestBatchSize
}

func benchmarkBatchSize(batchSize int) (throughput float64, memUsage float64, success bool) {
    // Force garbage collection before test
    runtime.GC()
    
    var memBefore runtime.MemStats
    runtime.ReadMemStats(&memBefore)
    
    // Create model
    inputShape := []int{batchSize, 784}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(512, true, "hidden1").
        AddReLU("relu1").
        AddDense(256, true, "hidden2").
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        return 0, 0, false
    }
    
    trainerConfig := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, trainerConfig)
    if err != nil {
        return 0, 0, false
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        return 0, 0, false
    }
    
    // Measure memory usage
    var memAfter runtime.MemStats
    runtime.ReadMemStats(&memAfter)
    memUsage = float64(memAfter.Alloc-memBefore.Alloc) / 1024.0 / 1024.0
    
    // Generate test data
    inputs := make([]float32, batchSize*784)
    labels := make([]int32, batchSize)
    
    for i := range inputs {
        inputs[i] = rand.Float32()
    }
    for i := range labels {
        labels[i] = int32(rand.Intn(10))
    }
    
    labelTensor, err := training.NewInt32Labels(labels, []int{batchSize})
    if err != nil {
        return 0, memUsage, false
    }
    
    // Benchmark throughput
    numIterations := 50
    startTime := time.Now()
    
    for i := 0; i < numIterations; i++ {
        _, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
        if err != nil {
            return 0, memUsage, false
        }
    }
    
    elapsed := time.Since(startTime)
    throughput = float64(numIterations*batchSize) / elapsed.Seconds()
    
    return throughput, memUsage, true
}

func demonstrateOptimalPerformance(batchSize int) {
    fmt.Printf("\n🚀 Performance with optimal batch size (%d):\n", batchSize)
    
    inputShape := []int{batchSize, 784}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        AddDense(512, true, "hidden1").
        AddReLU("relu1").
        AddDense(256, true, "hidden2").
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        log.Printf("Failed to create model: %v", err)
        return
    }
    
    trainerConfig := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, trainerConfig)
    if err != nil {
        log.Printf("Failed to create trainer: %v", err)
        return
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Printf("Failed to enable persistent buffers: %v", err)
        return
    }
    
    // Generate data
    inputs := make([]float32, batchSize*784)
    labels := make([]int32, batchSize)
    
    for i := range inputs {
        inputs[i] = rand.Float32()
    }
    for i := range labels {
        labels[i] = int32(rand.Intn(10))
    }
    
    labelTensor, err := training.NewInt32Labels(labels, []int{batchSize})
    if err != nil {
        log.Printf("Failed to create label tensor: %v", err)
        return
    }
    
    // Train for several epochs
    fmt.Println("Training progress:")
    
    for epoch := 1; epoch <= 10; epoch++ {
        epochStart := time.Now()
        
        result, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
        if err != nil {
            log.Printf("Training failed at epoch %d: %v", epoch, err)
            continue
        }
        
        epochTime := time.Since(epochStart)
        samplesPerSec := float64(batchSize) / epochTime.Seconds()
        
        fmt.Printf("   Epoch %2d: Loss = %.4f, %.1f samples/sec\n", 
                   epoch, result.Loss, samplesPerSec)
    }
}
```

## 🔧 Model Architecture Optimization

### Efficient vs Inefficient Architectures

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    
    fmt.Println("🏗️ Architecture Performance Comparison")
    fmt.Println("======================================")
    
    batchSize := 32
    
    // Test different architectures
    architectures := []struct {
        name        string
        buildFunc   func([]int) (*layers.ModelSpec, error)
        description string
    }{
        {
            "Efficient",
            buildEfficientModel,
            "Optimized layer sizes and patterns",
        },
        {
            "Inefficient",
            buildInefficientModel,
            "Poor layer sizing and many small layers",
        },
        {
            "Balanced",
            buildBalancedModel,
            "Good balance of size and depth",
        },
    }
    
    fmt.Printf("%-12s | %-15s | %-12s | %-30s\n", 
               "Architecture", "Samples/Sec", "Memory (MB)", "Description")
    fmt.Println("-------------|-----------------|--------------|------------------------------")
    
    for _, arch := range architectures {
        throughput, memUsage := benchmarkArchitecture(arch.buildFunc, batchSize)
        fmt.Printf("%-12s | %-15.1f | %-12.1f | %-30s\n", 
                   arch.name, throughput, memUsage, arch.description)
    }
}

func buildEfficientModel(inputShape []int) (*layers.ModelSpec, error) {
    builder := layers.NewModelBuilder(inputShape)
    return builder.
        AddDense(256, true, "hidden1").  // Power of 2, reasonable size
        AddReLU("relu1").                // Fastest activation
        AddDense(128, true, "hidden2").  // Gradual reduction
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
}

func buildInefficientModel(inputShape []int) (*layers.ModelSpec, error) {
    builder := layers.NewModelBuilder(inputShape)
    return builder.
        AddDense(37, true, "hidden1").   // Odd size, not power of 2
        AddReLU("relu1").
        AddDense(23, true, "hidden2").   // Very small layer
        AddReLU("relu2").
        AddDense(19, true, "hidden3").   // Another small layer
        AddReLU("relu3").
        AddDense(41, true, "hidden4").   // Odd size again
        AddReLU("relu4").
        AddDense(10, true, "output").
        Compile()
}

func buildBalancedModel(inputShape []int) (*layers.ModelSpec, error) {
    builder := layers.NewModelBuilder(inputShape)
    return builder.
        AddDense(512, true, "hidden1").  // Larger first layer
        AddReLU("relu1").
        AddDense(256, true, "hidden2").  // Gradual reduction
        AddReLU("relu2").
        AddDense(128, true, "hidden3").  // Continue reduction
        AddReLU("relu3").
        AddDense(10, true, "output").
        Compile()
}

func benchmarkArchitecture(buildFunc func([]int) (*layers.ModelSpec, error), batchSize int) (float64, float64) {
    inputShape := []int{batchSize, 784}
    
    // Create model
    model, err := buildFunc(inputShape)
    if err != nil {
        return 0, 0
    }
    
    trainerConfig := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, trainerConfig)
    if err != nil {
        return 0, 0
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        return 0, 0
    }
    
    // Generate test data
    inputs := make([]float32, batchSize*784)
    labels := make([]int32, batchSize)
    
    for i := range inputs {
        inputs[i] = rand.Float32()
    }
    for i := range labels {
        labels[i] = int32(rand.Intn(10))
    }
    
    labelTensor, err := training.NewInt32Labels(labels, []int{batchSize})
    if err != nil {
        return 0, 0
    }
    
    // Warmup
    for i := 0; i < 5; i++ {
        _, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
        if err != nil {
            return 0, 0
        }
    }
    
    // Benchmark
    numIterations := 50
    startTime := time.Now()
    
    for i := 0; i < numIterations; i++ {
        _, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
        if err != nil {
            return 0, 0
        }
    }
    
    elapsed := time.Since(startTime)
    throughput := float64(numIterations*batchSize) / elapsed.Seconds()
    
    // Rough memory estimation (this is simplified)
    memUsage := float64(batchSize) * 784 * 4 / 1024.0 / 1024.0 // Input size
    
    return throughput, memUsage
}
```

## 📊 Performance Comparison Tool

### Optimizer Performance Comparison

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    
    fmt.Println("⚡ Optimizer Performance Comparison")
    fmt.Println("===================================")
    
    compareOptimizers()
}

func compareOptimizers() {
    batchSize := 32
    inputShape := []int{batchSize, 784}
    
    // Test different optimizers
    optimizers := []struct {
        name string
        typ  cgo_bridge.OptimizerType
        lr   float32
    }{
        {"Adam", cgo_bridge.Adam, 0.001},
        {"SGD", cgo_bridge.SGD, 0.01},
        {"RMSProp", cgo_bridge.RMSProp, 0.001},
    }
    
    fmt.Printf("%-8s | %-15s | %-12s | %-15s\n", 
               "Optimizer", "Samples/Sec", "Final Loss", "Convergence")
    fmt.Println("---------|-----------------|--------------|---------------")
    
    for _, opt := range optimizers {
        throughput, finalLoss, converged := benchmarkOptimizer(opt.typ, opt.lr, inputShape)
        
        convergenceStatus := "✅"
        if !converged {
            convergenceStatus = "❌"
        }
        
        fmt.Printf("%-8s | %-15.1f | %-12.4f | %-15s\n", 
                   opt.name, throughput, finalLoss, convergenceStatus)
    }
}

func benchmarkOptimizer(optimizerType cgo_bridge.OptimizerType, lr float32, inputShape []int) (float64, float64, bool) {
    // Create model
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(256, true, "hidden1").
        AddReLU("relu1").
        AddDense(128, true, "hidden2").
        AddReLU("relu2").
        AddDense(10, true, "output").
        Compile()
    
    if err != nil {
        return 0, 0, false
    }
    
    trainerConfig := training.TrainerConfig{
        BatchSize:     inputShape[0],
        LearningRate:  lr,
        OptimizerType: optimizerType,
        EngineType:    training.Dynamic,
        LossFunction:  training.SparseCrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, err := training.NewModelTrainer(model, trainerConfig)
    if err != nil {
        return 0, 0, false
    }
    defer trainer.Cleanup()
    
    // Enable persistent buffers
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        return 0, 0, false
    }
    
    batchSize := inputShape[0]
    
    // Generate test data
    inputs := make([]float32, batchSize*784)
    labels := make([]int32, batchSize)
    
    for i := range inputs {
        inputs[i] = rand.Float32()
    }
    for i := range labels {
        labels[i] = int32(rand.Intn(10))
    }
    
    labelTensor, err := training.NewInt32Labels(labels, []int{batchSize})
    if err != nil {
        return 0, 0, false
    }
    
    // Train and measure performance
    numIterations := 100
    startTime := time.Now()
    
    var finalLoss float64
    initialLoss := float64(0)
    
    for i := 0; i < numIterations; i++ {
        result, err := trainer.TrainBatchUnified(inputs, inputShape, labelTensor)
        if err != nil {
            return 0, 0, false
        }
        
        if i == 0 {
            initialLoss = float64(result.Loss)
        }
        finalLoss = float64(result.Loss)
    }
    
    elapsed := time.Since(startTime)
    throughput := float64(numIterations*batchSize) / elapsed.Seconds()
    
    // Check convergence (loss should decrease)
    converged := finalLoss < initialLoss*0.8 // 20% reduction
    
    return throughput, finalLoss, converged
}
```

## 🎯 Quick Performance Tips

### Essential Optimizations

1. **Use appropriate batch sizes**: Start with 32-64, increase until memory limits
2. **Choose efficient architectures**: Use power-of-2 layer sizes, avoid very small layers
3. **Monitor memory usage**: Use the tools above to profile your models
4. **Select the right optimizer**: Adam for most cases, SGD for simple models
5. **Leverage Apple Silicon**: Take advantage of unified memory architecture

### Command Line Monitoring

```bash
# Monitor GPU utilization
sudo powermetrics --samplers gpu_power -n 1

# Check memory pressure
memory_pressure

# Monitor specific process
top -pid $(pgrep -f your_program)
```

## 🚀 Next Steps

- **[Memory Management Guide](memory-management.md)** - GPU memory optimization and best practices
- **[Mixed Precision Tutorial](../tutorials/mixed-precision.md)** - Further speed improvements
- **[Visualization Guide](visualization.md)** - Monitor training progress
- **[Advanced Examples](../examples/)** - Production-ready optimizations

---

## 🧠 Key Takeaways

- **Measure first**: Use the benchmarking tools to establish baselines
- **Optimize systematically**: Focus on batch size, architecture, then fine-tuning
- **Monitor continuously**: Track throughput, memory usage, and convergence
- **Apple Silicon advantages**: Leverage unified memory and high bandwidth

With these tools and techniques, you can achieve optimal performance for your go-metal applications!