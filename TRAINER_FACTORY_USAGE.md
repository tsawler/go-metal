# Production Trainer Factory Usage Guide

The go-metal training system now includes a comprehensive factory system for creating trainers with full configuration control. This replaces the limited `NewSimpleTrainer` function with a robust, production-ready API.

## Quick Start

### Basic Usage (Convenience Functions)

```go
// Create SGD trainer
trainer, err := training.NewSGDTrainer(32, 0.01)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// Create Adam trainer with defaults
trainer, err := training.NewAdamTrainer(64, 0.001)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()
```

### Factory System (Production Recommended)

```go
factory := training.NewFactory()

// Create SGD trainer with weight decay
trainer, err := factory.CreateSGDTrainer(32, 0.01, 0.001)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// Create Adam trainer with custom parameters
trainer, err := factory.CreateAdamTrainer(
    128,    // batch size
    0.002,  // learning rate
    0.95,   // beta1 (momentum)
    0.9999, // beta2 (RMSprop)
    1e-7,   // epsilon
    0.01,   // weight decay
)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()
```

## Advanced Configuration

### Full Configuration Control

```go
config := training.TrainerConfig{
    BatchSize:       256,
    LearningRate:    0.0005,
    OptimizerType:   cgo_bridge.Adam,
    Beta1:           0.9,
    Beta2:           0.999,
    Epsilon:         1e-8,
    WeightDecay:     0.0001,
    UseHybridEngine: true, // Always recommended
}

trainer, err := training.NewTrainerWithConfig(config)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()
```

### Production Trainer Pattern

```go
factory := training.NewFactory()

// Get default configuration for your optimizer
optimizerConfig := factory.GetDefaultAdamConfig(0.001)
optimizerConfig.WeightDecay = 0.01  // Add L2 regularization

// Create production trainer
trainer, err := factory.CreateProductionTrainer(512, optimizerConfig)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()
```

## Configuration Options

### TrainerConfig Structure

```go
type TrainerConfig struct {
    // Training parameters
    BatchSize    int     // Batch size for training
    LearningRate float32 // Learning rate
    
    // Optimizer configuration
    OptimizerType cgo_bridge.OptimizerType // SGD or Adam
    
    // Adam-specific parameters (ignored for SGD)
    Beta1       float32 // Adam momentum decay (default: 0.9)
    Beta2       float32 // Adam RMSprop decay (default: 0.999)
    Epsilon     float32 // Adam numerical stability (default: 1e-8)
    WeightDecay float32 // L2 regularization (default: 0.0)
    
    // Training behavior
    UseHybridEngine bool // Use hybrid MPS/MPSGraph (recommended: true)
}
```

### Optimizer Types

```go
import "github.com/tsawler/go-metal/cgo_bridge"

// SGD optimizer
config.OptimizerType = cgo_bridge.SGD

// Adam optimizer (recommended for most use cases)
config.OptimizerType = cgo_bridge.Adam
```

## Factory Methods

### Basic Trainer Creation

- `CreateSGDTrainer(batchSize, learningRate, weightDecay)` - SGD with weight decay
- `CreateAdamTrainer(batchSize, learningRate, beta1, beta2, epsilon, weightDecay)` - Full Adam control
- `CreateAdamTrainerWithDefaults(batchSize, learningRate)` - Adam with sensible defaults

### Advanced Trainer Creation

- `CreateTrainer(config)` - Full configuration control
- `CreateProductionTrainer(batchSize, optimizerConfig)` - Production-optimized trainer

### Configuration Helpers

- `GetDefaultSGDConfig(learningRate)` - Default SGD configuration
- `GetDefaultAdamConfig(learningRate)` - Default Adam configuration

## Validation

The factory system includes comprehensive validation:

```go
// This will return an error
config := training.TrainerConfig{
    BatchSize:    -1,    // ❌ Must be positive
    LearningRate: 0.0,   // ❌ Must be positive
    Beta1:        1.5,   // ❌ Must be in (0, 1) for Adam
}

trainer, err := factory.CreateTrainer(config)
if err != nil {
    // Handle validation error
    log.Printf("Configuration error: %v", err)
}
```

## Backward Compatibility

The original `NewSimpleTrainer` function is still available but deprecated:

```go
// DEPRECATED: Still works but limited functionality
trainer, err := training.NewSimpleTrainer(32, 0.01)
```

## Performance Recommendations

### For Maximum Performance

1. **Always use `UseHybridEngine: true`** - This uses the optimized hybrid MPS/MPSGraph architecture
2. **Choose Adam for most use cases** - Better convergence than SGD
3. **Use appropriate batch sizes** - Larger batches (128-512) typically perform better
4. **Production trainer pattern** - Use `CreateProductionTrainer` for production deployments

### Example Production Configuration

```go
factory := training.NewFactory()

// High-performance production configuration
optimizerConfig := factory.GetDefaultAdamConfig(0.001)
optimizerConfig.WeightDecay = 0.01  // L2 regularization

trainer, err := factory.CreateProductionTrainer(256, optimizerConfig)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()

// This trainer is optimized for:
// - 20,000+ batch/s performance
// - MPSGraph-based Adam optimization
// - Hybrid MPS/MPSGraph architecture
// - Production-ready error handling
```

## Migration Guide

### From Old API

```go
// OLD (limited)
trainer, err := training.NewSimpleTrainer(32, 0.01)

// NEW (recommended)
trainer, err := training.NewSGDTrainer(32, 0.01)
// OR for Adam
trainer, err := training.NewAdamTrainer(32, 0.01)
```

### From Manual Configuration

```go
// OLD (manual config)
config := cgo_bridge.TrainingConfig{
    LearningRate:  0.01,
    OptimizerType: cgo_bridge.SGD,
    // ... manual setup
}

// NEW (factory with validation)
factory := training.NewFactory()
trainer, err := factory.CreateSGDTrainer(32, 0.01, 0.0)
```

This factory system provides the flexibility and robustness needed for production machine learning training while maintaining the high performance of the go-metal system (20,000+ batch/s).