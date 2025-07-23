# Mixed Precision Training Demo

This example demonstrates the performance benefits of mixed precision (FP16) training on Apple Silicon using Go-Metal.

## What This Demo Shows

- **Performance Comparison**: Direct comparison between FP32 and FP16 training
- **Loss Convergence**: Ensures FP16 maintains similar training dynamics
- **Automatic Mixed Precision**: Uses Go-Metal's automatic loss scaling
- **Real Training**: Complete training loop with synthetic data

## Expected Results

On Apple Silicon (M1/M2/M3), you should see:
- **50-70% speedup** for this MLP model
- **Similar loss convergence** between FP32 and FP16
- **Stable training** with automatic loss scaling

## Running the Demo

```bash
# From the go-metal directory
cd examples/mixed-precision-demo

# Run the demo
go run main.go
```

## Sample Output

```
🚀 Go-Metal Mixed Precision Training Demo
==========================================

📊 Phase 1: FP32 Baseline Training
==================================
FP32 Training Configuration:
  Batch Size: 64
  Input Size: 784
  Hidden Size: 512
  Classes: 10
  Samples: 1000
  Epochs: 20

Starting training...
Epoch | Loss     | Time (ms)
------|----------|----------
    1 | 2.301258 |    45.23
    5 | 1.523614 |    43.18
   10 | 0.892341 |    44.91
   20 | 0.421853 |    43.55

⚡ Phase 2: FP16 Mixed Precision Training
=========================================
FP16 Training Configuration:
  [Same as above]

Starting training...
[Similar output with faster times]

📈 Performance Comparison
========================

Training Time:
  FP32: 0.92 seconds
  FP16: 0.54 seconds
  Speedup: 70.4%

Final Loss:
  FP32: 0.421853
  FP16: 0.423917
  Loss Difference: 0.002064

✅ Summary
==========
🎉 Excellent speedup achieved: 70.4%
✅ Loss convergence well preserved

🏁 Mixed precision training demo completed!
```

## Key Observations

1. **Significant Speedup**: FP16 provides 50-70% faster training
2. **Maintained Convergence**: Loss values remain comparable between FP32 and FP16
3. **Stable Training**: Automatic loss scaling prevents numerical issues
4. **Memory Efficiency**: Lower memory usage allows larger batches

## Customization

Try modifying these parameters to see different results:

```go
// Increase model size for even better speedup
hiddenSize := 1024  // was 512

// Try different batch sizes
batchSize := 128    // was 64

// Add more layers for deeper network
builder.
    AddDense(1024, true, "dense1").
    AddReLU("relu1").
    AddDense(512, true, "dense2").
    AddReLU("relu2").
    AddDense(256, true, "dense3").
    AddReLU("relu3").
    // ...
```

## Understanding the Results

- **Larger models** see greater speedup (up to 86% for CNNs)
- **Batch size** affects GPU utilization and speedup
- **Apple Silicon** has dedicated FP16 units for optimal performance
- **Automatic loss scaling** maintains numerical stability

## Next Steps

- Try the full [Mixed Precision Tutorial](../../docs/tutorials/mixed-precision.md)
- Explore [Performance Optimization](../../docs/guides/performance.md)
- Apply to your own models for maximum training speed