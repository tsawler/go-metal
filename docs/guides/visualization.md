# Visualization Guide

Interactive visualization for go-metal models using the integrated sidecar plotting service.

## 🎯 Overview

This guide shows how to use the **go-metal sidecar service** for interactive visualization of your training progress, model metrics, and data analysis. The sidecar provides professional Plotly-based charts that open automatically in your browser.

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
    
    "github.com/tsawler/go-metal/training"
)

func main() {
    // ... your model and trainer setup ...
    
    // Enable visualization data collection
    trainer.EnableVisualization()
    trainer.EnableEvaluationMetrics()
    
    // Your training loop
    for epoch := 1; epoch <= 100; epoch++ {
        // Training phase
        result, err := trainer.TrainBatch(inputData, inputShape, targetData, targetShape)
        if err != nil {
            log.Fatal(err)
        }
        
        // Validation phase (for metrics collection)
        trainer.StartValidationPhase()
        // ... validation code ...
        trainer.RecordMetricsForVisualization()
        
        fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, result.Loss)
    }
    
    // Send plots to sidecar and open dashboard
    plottingService := training.NewPlottingService(training.DefaultPlottingServiceConfig())
    plottingService.Enable()
    
    if plottingService.CheckHealth() == nil {
        plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
        fmt.Println("🌐 Dashboard opened in browser")
    }
}
```

## 📊 Complete Training Example with Visualization

Here's a complete example showing sidecar integration:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("🚀 Go-Metal Training with Sidecar Visualization")
    
    // ... initialize device, create model, optimizer, criterion ...
    
    // Create trainer
    trainer := training.NewTrainer(model, optimizer, criterion, config)
    
    // Enable visualization data collection
    trainer.EnableVisualization()
    trainer.EnableEvaluationMetrics()
    
    // Training loop
    fmt.Println("🎬 Starting training...")
    for epoch := 1; epoch <= config.Epochs; epoch++ {
        // Training phase
        for batch := range dataLoader.Iterate() {
            result, err := trainer.TrainBatch(
                batch.Data.Data.([]float32),
                batch.Data.Shape,
                batch.Labels.Data.([]float32),
                batch.Labels.Shape,
            )
            if err != nil {
                log.Printf("Training error: %v", err)
                continue
            }
        }
        
        // Validation phase for metrics collection
        trainer.StartValidationPhase()
        for step := 1; step <= validationSteps; step++ {
            // Run inference and update metrics
            inferenceResult, err := trainer.InferBatch(validationData, validationShape)
            if err == nil {
                trainer.UpdateMetricsFromInference(
                    inferenceResult.Predictions,
                    validationLabels,
                    batchSize,
                )
            }
        }
        trainer.RecordMetricsForVisualization()
        
        fmt.Printf("Epoch %d/%d completed\n", epoch, config.Epochs)
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
// Generate specific plot types
trainingCurves := trainer.GenerateTrainingCurvesPlot()
rocCurve := trainer.GenerateROCCurvePlot()

// Send individual plots
resp, err := plottingService.SendPlotDataAndOpen(trainingCurves)
if err != nil {
    log.Printf("Failed to send plot: %v", err)
} else {
    fmt.Printf("Plot available at: %s\n", resp.ViewURL)
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