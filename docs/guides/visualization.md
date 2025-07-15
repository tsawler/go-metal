# Visualization Guide

Interactive visualization for go-metal models using the integrated sidecar plotting service.

## 🎯 Overview

This guide shows how to use the [**go-metal sidecar service**](https://github.com/tsawler/go-metal-sidecar-plots) for interactive visualization of your training progress, model metrics, and data analysis. The sidecar provides professional Plotly-based charts that open automatically in your browser.

The sidecar service is included with go-metal and provides seamless integration for all your visualization needs.

## 🚀 Quick Start

### 1. Start the Sidecar Service

Navigate to the sidecar directory and start the service:

```bash
# From the project root, go to the sidecar directory
cd sidecar

# Start with Docker (recommended)
docker-compose up -d

# Or start locally with Python
pip install -r requirements.txt
python app.py
```

The service starts on `http://localhost:8080` by default and provides a health check endpoint.

### 2. Enable Visualization in Your Training Code

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
    
    // Enable persistent buffers for better performance
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Enable visualization data collection
    trainer.EnableVisualization()
    trainer.EnableEvaluationMetrics()
    
    // Your training loop
    for epoch := 1; epoch <= 10; epoch++ {
        // Training phase
        for step := 1; step <= 100; step++ {
            inputData, labelData := generateTrainingBatch()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training step failed: %v", err)
                continue
            }
        }
        
        // Validation phase (for metrics collection)
        trainer.StartValidationPhase()
        for step := 1; step <= 20; step++ {
            valInputData, _ := generateValidationBatch()
            
            inferenceResult, err := trainer.InferBatch(valInputData, inputShape)
            if err != nil {
                log.Printf("Inference failed: %v", err)
                continue
            }
            
            // Generate labels for metrics (in real use, these would be your actual validation labels)
            labelData := make([]int32, 32)
            for i := range labelData {
                labelData[i] = int32(rand.Intn(10))
            }
            
            err = trainer.UpdateMetricsFromInference(
                inferenceResult.Predictions,
                labelData,
                32,
            )
            if err != nil {
                log.Printf("Metrics update failed: %v", err)
            }
        }
        trainer.RecordMetricsForVisualization()
        
        fmt.Printf("Epoch %d completed\n", epoch)
    }
    
    // Send plots to sidecar and open dashboard
    plottingService := training.NewPlottingService(training.DefaultPlottingServiceConfig())
    plottingService.Enable()
    
    if plottingService.CheckHealth() == nil {
        results := plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
        
        for plotType, result := range results {
            if result.Success {
                fmt.Printf("✅ %s: Generated successfully\n", plotType)
            } else {
                fmt.Printf("❌ %s: Failed - %s\n", plotType, result.Message)
            }
        }
        fmt.Println("🌐 Dashboard opened in browser")
    } else {
        fmt.Println("⚠️ Sidecar service not available")
    }
}

// Helper function to generate training batch
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

// Helper function to generate validation batch
func generateValidationBatch() ([]float32, *training.Int32Labels) {
    return generateTrainingBatch()
}
```

## 📊 Complete Training Example with Visualization

Here's a complete example showing sidecar integration:

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
    fmt.Println("🚀 Go-Metal Training with Sidecar Visualization")
    
    // Build model
    inputShape := []int{32, 3, 32, 32}
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
    
    // Enable persistent buffers for better performance
    err = trainer.EnablePersistentBuffers(inputShape)
    if err != nil {
        log.Fatalf("Failed to enable persistent buffers: %v", err)
    }
    
    // Enable visualization data collection
    trainer.EnableVisualization()
    trainer.EnableEvaluationMetrics()
    
    // Training loop
    fmt.Println("🎬 Starting training...")
    epochs := 10
    stepsPerEpoch := 100
    validationSteps := 20
    
    for epoch := 1; epoch <= epochs; epoch++ {
        // Training phase
        for step := 1; step <= stepsPerEpoch; step++ {
            // Generate or load training data
            inputData, labelData := generateTrainingBatch()
            
            result, err := trainer.TrainBatchUnified(inputData, inputShape, labelData)
            if err != nil {
                log.Printf("Training error: %v", err)
                continue
            }
        }
        
        // Validation phase for metrics collection
        trainer.StartValidationPhase()
        for step := 1; step <= validationSteps; step++ {
            // Generate or load validation data
            valInputData, _ := generateValidationBatch()
            
            // Run inference and update metrics
            inferenceResult, err := trainer.InferBatch(valInputData, inputShape)
            if err != nil {
                log.Printf("Inference error: %v", err)
                continue
            }
            
            // Generate labels for metrics (in real use, these would be your actual validation labels)
            labelData := make([]int32, 32)
            for i := range labelData {
                labelData[i] = int32(rand.Intn(10))
            }
            
            err = trainer.UpdateMetricsFromInference(
                inferenceResult.Predictions,
                labelData,
                32,
            )
            if err != nil {
                log.Printf("Metrics update error: %v", err)
            }
        }
        trainer.RecordMetricsForVisualization()
        
        fmt.Printf("Epoch %d/%d completed\n", epoch, epochs)
    }
    
    // Send plots to sidecar service
    fmt.Println("📊 Sending plots to sidecar service...")
    plottingService := training.NewPlottingService(training.DefaultPlottingServiceConfig())
    plottingService.Enable()
    
    if plottingService.CheckHealth() == nil {
        results := plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
        
        for plotType, result := range results {
            if result.Success {
                fmt.Printf("✅ %s: Generated successfully\n", plotType)
            } else {
                fmt.Printf("❌ %s: Failed - %s\n", plotType, result.Message)
            }
        }
        fmt.Println("🌐 Dashboard opened in browser")
    } else {
        fmt.Println("⚠️ Sidecar service not available")
    }
}

// Helper function to generate training batch
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

// Helper function to generate validation batch
func generateValidationBatch() ([]float32, *training.Int32Labels) {
    return generateTrainingBatch()
}
```

## 🎨 Supported Plot Types

The sidecar service automatically generates the following visualizations:

### Classification Tasks
- **Training Curves**: Loss and accuracy over training steps
- **Learning Rate Schedule**: Learning rate decay visualization  
- **ROC Curves**: True/False Positive Rate analysis
- **Precision-Recall Curves**: Precision vs Recall trade-offs
- **Confusion Matrix**: Classification results breakdown

### Regression Tasks
- **Training Curves**: Loss progression over time
- **Learning Rate Schedule**: Learning rate decay visualization
- **Regression Scatter**: Predicted vs actual values
- **Residual Plot**: Model diagnostic analysis
- **Q-Q Plot**: Distribution validation

### Advanced Plots (Available)
- **Feature Importance**: Feature contribution analysis
- **Learning Curve**: Performance vs training set size
- **Validation Curve**: Performance vs hyperparameter values
- **Prediction Interval**: Uncertainty and confidence intervals
- **Feature Correlation**: Multicollinearity detection
- **Partial Dependence**: Individual feature effects

## 🛠️ Configuration Options

### Sidecar Service Configuration

```go
// Custom sidecar configuration
config := training.PlottingServiceConfig{
    BaseURL:       "http://localhost:8080",  // Sidecar URL
    Timeout:       30 * time.Second,         // Request timeout
    RetryAttempts: 3,                        // Retry failed requests
    RetryDelay:    1 * time.Second,          // Delay between retries
}

plottingService := training.NewPlottingService(config)
```

### Manual Plot Generation

```go
// Generate specific plot types using the plotting service
plottingService := training.NewPlottingService(training.DefaultPlottingServiceConfig())
plottingService.Enable()

// Check if sidecar is available
if plottingService.CheckHealth() == nil {
    // Generate all available plots
    results := plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
    
    for plotType, result := range results {
        if result.Success {
            fmt.Printf("✅ %s: Generated successfully\n", plotType)
        } else {
            fmt.Printf("❌ %s: Failed - %s\n", plotType, result.Message)
        }
    }
    fmt.Println("🌐 Dashboard opened in browser")
} else {
    fmt.Println("⚠️ Sidecar service not available")
}
```

## 🔍 Troubleshooting

### Common Issues

**Sidecar service not found**:
```bash
# Check if sidecar is running
curl http://localhost:8080/health

# Start sidecar if needed
cd sidecar && docker-compose up -d
```

**No plots generated**:
```go
// Ensure visualization is enabled
trainer.EnableVisualization()
trainer.EnableEvaluationMetrics()
```

**Browser doesn't open automatically**:
The dashboard URL will be printed to console for manual access.

**Connection timeouts**:
```go
// Increase timeout for large datasets
config := training.PlottingServiceConfig{
    Timeout: 60 * time.Second,
}
```

## 🎯 Best Practices

### Training Visualization Guidelines

**📊 Data Collection**:
- Use `trainer.EnableVisualization()` to collect training metrics
- Use `trainer.EnableEvaluationMetrics()` for classification curves
- Call `trainer.RecordMetricsForVisualization()` after each epoch

**🌐 Sidecar Integration**:
- Start sidecar service before training: `cd sidecar && docker-compose up -d`
- Check service health: `plottingService.CheckHealth()`
- Use batch plotting: `GenerateAndSendAllPlotsWithBrowser()`

**⚡ Performance**:
- The sidecar handles plot generation efficiently
- Plots open automatically in your browser
- All data is processed client-side for responsiveness

**🔧 Configuration**:
- Adjust timeouts for large datasets
- Configure retry attempts for reliability
- Use custom base URLs for remote sidecar services

## 📋 Quick Reference

### Essential Commands

**Start Sidecar Service**:
```bash
cd sidecar && docker-compose up -d
```

**Enable Visualization in Code**:
```go
trainer.EnableVisualization()
trainer.EnableEvaluationMetrics()
```

**Send Plots to Sidecar**:
```go
plottingService := training.NewPlottingService(training.DefaultPlottingServiceConfig())
plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
```

**Check Service Health**:
```bash
curl http://localhost:8080/health
```

## 🌐 Production Deployment

### Docker Deployment

The sidecar service is production-ready with Docker:

```bash
# Production deployment
cd sidecar
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Configuration Options

**Environment Variables**:
- `PORT=8080` - Service port
- `FLASK_ENV=production` - Environment mode

**Security Features**:
- Non-root container user
- CORS enabled for cross-origin requests
- Request size limits (50MB max)
- Input validation on all endpoints

### Monitoring

**Health Checks**:
```bash
curl http://localhost:8080/health
```

**View Generated Plots**:
```bash
curl http://localhost:8080/api/plots
```

## 🎯 Summary

The go-metal sidecar service provides:

- **🚀 Easy Setup**: Docker-based deployment with health checks
- **📊 Complete Coverage**: All plot types for classification and regression
- **🌐 Browser Integration**: Automatic dashboard opening
- **⚡ High Performance**: Client-side rendering with Plotly
- **🔧 Production Ready**: Configurable timeouts, retries, and security

### Next Steps

1. **Start the sidecar**: `cd sidecar && docker-compose up -d`
2. **Enable visualization**: Add `trainer.EnableVisualization()` to your training code
3. **Generate plots**: Use `GenerateAndSendAllPlotsWithBrowser()` after training
4. **View results**: Dashboard opens automatically in your browser

**Continue Learning:**
- **[Performance Guide](performance.md)** - Optimize training performance
- **[Mixed Precision Tutorial](../tutorials/mixed-precision.md)** - Advanced training techniques
- **[Example Applications](../examples/)** - Real-world use cases

---

*The sidecar service handles all visualization complexity, letting you focus on building great models.*