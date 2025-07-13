# Go-Metal Examples

This directory contains comprehensive examples demonstrating various aspects of the Go-Metal library. Each example is self-contained and runnable.

## 📚 Available Examples

### 🔢 [Basic Tensor Operations](basic-tensor-operations/)
**What it demonstrates**: Fundamental tensor operations with dynamic batch size support
- Multi-layer perceptron (MLP) architecture
- Dynamic batch size processing (4, 8, 16, 24, 32 samples)
- GPU-accelerated operations with MPSGraph
- Shape flexibility and tensor transformations

**Run**: `cd basic-tensor-operations && go run main.go`

### 🎯 [Complete Training Pipeline](complete-training-pipeline/)
**What it demonstrates**: End-to-end neural network training with advanced features
- Pure MLP architecture for iris flower classification
- Batch normalization and dropout regularization
- Early stopping and convergence detection
- Feature preprocessing and normalization
- Multi-class classification (3 classes)

**Run**: `cd complete-training-pipeline && go run main.go`

### 🖼️ [CNN Image Classification](cnn-image-classification/)
**What it demonstrates**: Convolutional neural networks for image classification
- CNN architecture with Conv2D and pooling layers
- Real image dataset processing (cats vs dogs)
- GPU-accelerated convolution operations
- Image preprocessing and data loading
- Binary classification with visualization support

**Run**: `cd cnn-image-classification && go run main.go`

### 💾 [Model Serialization](model-serialization/)
**What it demonstrates**: Saving and loading trained models
- Model checkpoint creation and restoration
- ONNX format export and import
- Parameter persistence across sessions
- Model versioning and compatibility

**Run**: `cd model-serialization && go run main.go`

## 🚀 Quick Start

To run any example:

1. **Navigate to the example directory**:
   ```bash
   cd examples/[example-name]
   ```

2. **Install dependencies** (if needed):
   ```bash
   go mod tidy
   ```

3. **Run the example**:
   ```bash
   go run main.go
   ```

## 📊 Performance Comparison

| Example | Primary Focus | GPU Acceleration | Training Time | Key Features |
|---------|---------------|------------------|---------------|--------------|
| Basic Tensor Operations | Tensor fundamentals | ✅ MPSGraph | ~5 seconds | Dynamic batching, shape flexibility |
| Complete Training Pipeline | End-to-end ML | ✅ MPSGraph | ~10 seconds | BatchNorm, dropout, early stopping |
| CNN Image Classification | Computer vision | ✅ MPSGraph | ~30 seconds | Conv2D, pooling, real image data |
| Model Serialization | Model persistence | ✅ MPSGraph | ~5 seconds | ONNX export/import, checkpointing |

## 🛠️ System Requirements

All examples require:
- **macOS 12.0+** (Monterey or later)
- **Apple Silicon** (M1/M2/M3 series)
- **Go 1.21+**
- **Xcode Command Line Tools**

## 🎯 Learning Path

**Recommended order for learning**:

1. **Start here**: [Basic Tensor Operations](basic-tensor-operations/) - Learn fundamental concepts
2. **Next**: [Complete Training Pipeline](complete-training-pipeline/) - Understand full training workflow
3. **Then**: [CNN Image Classification](cnn-image-classification/) - Explore computer vision
4. **Finally**: [Model Serialization](model-serialization/) - Learn persistence and deployment

## 🔧 Customization

Each example can be easily modified:

- **Change architectures**: Modify layer configurations in model definition
- **Adjust hyperparameters**: Update learning rates, batch sizes, epochs
- **Try different optimizers**: Swap Adam for SGD or other optimizers
- **Add visualization**: Enable sidecar plotting service integration
- **Experiment with data**: Use your own datasets

## 🆘 Troubleshooting

**Common issues**:

1. **"Metal device not found"**: Ensure you're running on Apple Silicon
2. **"Module not found"**: Run `go mod tidy` in the example directory
3. **"Permission denied"**: Install Xcode Command Line Tools: `xcode-select --install`
4. **Slow performance**: Make sure you're using GPU device types (`tensor.PersistentGPU`)

**Getting help**:
- Check individual README files in each example directory
- Review the main [Go-Metal documentation](../docs/)
- Open an issue on the GitHub repository

## 🎉 Next Steps

After exploring these examples:

1. **Read the tutorials**: Check out [docs/tutorials/](../docs/tutorials/) for in-depth guides
2. **Explore the API**: Review [docs/guides/](../docs/guides/) for complete API documentation
3. **Build your own**: Start your own machine learning project with Go-Metal
4. **Contribute**: Help improve Go-Metal by contributing examples or features

---

**Happy learning with Go-Metal!** 🚀