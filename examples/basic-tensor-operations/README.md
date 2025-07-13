# Basic Tensor Operations Demo

This example demonstrates fundamental tensor operations and dynamic batch size support in Go-Metal. It shows how the same model can efficiently process different batch sizes using GPU acceleration.

## What This Demo Shows

### ✅ Core Dynamic Batch Size Features

1. **Multiple Batch Sizes**: Tests the same model with batch sizes: 4, 8, 16, 24, 32
2. **No Recompilation**: The model processes all batch sizes using the same compiled graph
3. **Training & Inference**: Both training steps and inference work with variable batch sizes
4. **Mixed Batch Training**: Demonstrates training with varying batch sizes in a single session
5. **GPU-Resident Operations**: All operations remain on GPU using MPSGraph

### 🔧 Technical Implementation Highlights

- **Dynamic Label Placeholders**: Uses `-1` for batch dimension in MPSGraph placeholders
- **Shape-Aware Loss Computation**: Automatically handles batch size variations in loss calculations
- **Tensor Flattening**: Preserves dynamic batch dimension when reshaping from 4D to 2D
- **Memory Efficient**: Reuses the same graph structure for all batch sizes

## Running the Demo

```bash
cd /Users/tcs/vs-projects/go-metal-new/app/dynamic-batch-demo
go run .
```

## Expected Output

The demo will:

1. **Build a CNN Model** - Creates a simple convolutional neural network
2. **Test Individual Batch Sizes** - Runs forward/backward passes with different batch sizes
3. **Show Performance Metrics** - Displays loss, accuracy, and processing time for each batch size
4. **Demonstrate Mixed Training** - Trains with varying batch sizes in sequence

### Sample Output Format

```
=== Dynamic Batch Size Demo ===

🧠 Building CNN model...
✅ Model built with 8 layers
   Input shape: [16 3 32 32]
   Parameter count: 42,148

--- Test 1: Batch Size 4 ---
✅ Forward pass successful!
   Batch size: 4 samples
   Loss: 1.386294
   Accuracy: 25.00%
   Processing time: 12.3ms

--- Test 2: Batch Size 8 ---
✅ Forward pass successful!
   Batch size: 8 samples
   Loss: 1.382156
   Accuracy: 12.50%
   Processing time: 15.7ms

[... continues for all batch sizes ...]

🔄 Demonstrating mixed batch size training...
Epoch 1:
  Step 1 (batch=8): Loss=1.3821, Accuracy=12.5%
  Step 2 (batch=16): Loss=1.3789, Accuracy=18.8%
  Step 3 (batch=12): Loss=1.3756, Accuracy=25.0%
  Step 4 (batch=20): Loss=1.3724, Accuracy=20.0%

🎉 Dynamic Batch Size Demo Complete!
```

## Architecture Details

### Model Structure
- **Conv2D Layer 1**: 8 filters, 3x3 kernel
- **ReLU Activation**
- **Conv2D Layer 2**: 16 filters, 3x3 kernel, stride 2
- **ReLU Activation**
- **Dense Layer 1**: 32 neurons
- **ReLU Activation**
- **Dense Layer 2**: 4 classes (output)

### Synthetic Data
- **Input**: Random RGB images (32x32 pixels)
- **Labels**: Random one-hot encoded (4 classes)
- **Data Range**: Input values normalized to [-1, 1]

## Key Benefits Demonstrated

### 🚀 Performance Benefits
- **No Graph Recompilation**: Saves significant initialization time
- **Memory Efficiency**: Single graph handles all batch sizes
- **GPU Utilization**: Optimal resource usage across different batch sizes

### 🔧 Flexibility Benefits
- **Dynamic Training**: Can adjust batch size based on available memory
- **Inference Scalability**: Handle single samples or large batches seamlessly
- **Production Ready**: Real-world applications can vary batch size as needed

## Technical Implementation

The dynamic batch size support is implemented through:

1. **MPSGraph Placeholders**: Using `-1` for dynamic dimensions
2. **Shape Operations**: Runtime batch size calculation using `shapeOfTensor`
3. **Reduction Operations**: Batch-aware loss computation with `reductionSumWithTensor`
4. **Tensor Reshaping**: Preserving dynamic dimensions during flattening

## Verification Points

This demo verifies that:
- ✅ Models accept variable batch sizes without errors
- ✅ Loss computation works correctly for all batch sizes
- ✅ Memory usage remains stable across batch size changes
- ✅ Performance scales appropriately with batch size
- ✅ GPU operations remain efficient and correct

## Shape Flexibility Demo Design

### Demo Scope and Limitations

The `shape_flexibility_demo.go` demonstrates **C-side shape transformations** that work within the current architectural constraints. The demo focuses on successfully implemented features rather than attempting operations that fail due to Go-side validation.

### Why the 1D CNN Test Was Removed

**Original Issue**: The initial demo included a test attempting to use 1D input `[batch, features]` with Conv2D layers, which failed with:
```
Conv2D layer requires 4D input [batch, channels, height, width], got [batch, features]
```

**Root Cause**: Go-side validation in the layer builder prevents incompatible shapes from reaching the C-side transformation logic. While the C-side code includes intelligent shape handling (like 3D→4D auto-expansion), Go-side validation occurs first and rejects configurations before C-side transformations can be applied.

**Design Decision**: Rather than attempt workarounds that violate the intended validation logic, the demo was updated to focus on **successfully working transformations** that demonstrate the enhanced shape flexibility.

### What the Demo Successfully Demonstrates

#### ✅ Adaptive Tensor Flattening
- **4D→2D transformations**: Conv2D output (4D) → Dense layer input (2D)
- **Dynamic calculation**: Flattened size computed at runtime: `flattenedSize = channels × height × width`
- **Batch preservation**: Dynamic batch dimension (-1) maintained throughout

#### ✅ Enhanced BatchNorm Broadcasting
- **Flexible channel support**: BatchNorm adapts to different channel counts (16→32→64→128)
- **Multi-dimensional support**: 4D BatchNorm (conv layers) and 2D BatchNorm (dense layers)
- **Broadcast pattern**: `[1, C, 1, 1, ...]` for arbitrary input ranks

#### ✅ Complex Architecture Support
- **Mixed architectures**: Conv2D → BatchNorm → Dense → BatchNorm chains
- **Shape transitions**: Seamless 4D→2D→2D transformations
- **Multiple layer types**: All layer types work together with enhanced shape handling

### Technical Justification

The demo design reflects the **layered validation approach** in go-metal:

1. **Go-side validation**: Ensures type safety and catches obvious incompatibilities early
2. **C-side transformations**: Provides intelligent shape adaptations for valid configurations
3. **GPU execution**: MPSGraph operations handle the actual tensor manipulations

This design ensures both **safety** (invalid configurations are caught early) and **flexibility** (valid configurations get intelligent shape handling).

### Future Improvements

To support more flexible input shapes (like 1D→4D auto-expansion), enhancements would need to be made at the Go-side validation layer to recognize and allow such transformations before delegating to C-side handling.

## Integration with Go-Metal Library

This functionality integrates seamlessly with:
- **Model Training**: `training.ModelTrainer` supports dynamic batches
- **Engine Creation**: Both dynamic and hybrid engines work
- **Optimizer Support**: All optimizers (Adam, SGD, RMSProp, etc.) handle variable batches
- **Memory Management**: Tensor creation and cleanup work with any batch size