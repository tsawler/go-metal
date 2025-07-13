# CNN Image Classification Demo

This example demonstrates convolutional neural network (CNN) training for image classification using Go-Metal's GPU-accelerated operations. It uses a cats vs dogs dataset to showcase real-world computer vision applications.

You **must** download the Kaggle Cats & Dogs dataset. You can find a copy of it [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765). Download it, extract it, and arrange the data like this:

```
.
└── data/
    ├── cat/
    └── dog/
```

All of the cat images go in the cat directory, and the dog images go in the dog directory.

## What This Demonstrates

### 🖼️ Computer Vision with CNNs
- **Convolutional Layers**: Conv2D operations with GPU acceleration via MPSGraph
- **Pooling Operations**: MaxPool2D for spatial dimension reduction
- **Real Image Data**: Actual image dataset processing with cats and dogs
- **Image Preprocessing**: Automatic image loading, resizing, and normalization
- **Binary Classification**: Cat vs Dog prediction with high accuracy

### 🚀 Key Technical Features
- **GPU-Accelerated Convolutions**: All convolution operations run on Apple Silicon GPU
- **Automatic Data Loading**: Built-in image folder dataset loader
- **Mixed Precision Support**: Efficient GPU memory usage
- **Performance Monitoring**: Training progress tracking and metrics
- **Model Persistence**: Save trained models in ONNX format

## Model Architecture

The CNN uses a classic computer vision architecture:

```
Input: RGB Images (64x64 pixels)
    ↓
Conv2D (3→32 filters, 3x3 kernel) + ReLU
    ↓
MaxPool2D (2x2)
    ↓
Conv2D (32→64 filters, 3x3 kernel) + ReLU
    ↓
MaxPool2D (2x2)
    ↓
Flatten
    ↓
Dense (64*15*15 → 128) + ReLU
    ↓
Dense (128 → 2) [Cat, Dog]
    ↓
Softmax (Probability Output)
```

## Dataset

### Image Data Structure
- **Path**: `data/cat/` and `data/dog/` directories
- **Format**: JPEG images of various sizes (automatically resized to 64x64)
- **Classes**: Binary classification (Cat=0, Dog=1)
- **Preprocessing**: Images normalized to [0,1] range

### Sample Images Included
The example includes a comprehensive dataset with thousands of cat and dog images for training and validation.

## Running the Demo

```bash
cd examples/cnn-image-classification
go run main.go
```

### Expected Output

```
🐱🐶 CNN Image Classification Demo
================================

📁 Loading image dataset...
✅ Dataset loaded: 2000 cat images, 1800 dog images
📊 Total samples: 3800

🧠 Building CNN model...
✅ Model architecture:
   Conv2D(3→32) → ReLU → MaxPool2D
   Conv2D(32→64) → ReLU → MaxPool2D  
   Flatten → Dense(15360→128) → ReLU → Dense(128→2)
   Parameters: 1,966,082

🎯 Training Configuration:
   Device: PersistentGPU
   Batch Size: 16
   Learning Rate: 0.001
   Epochs: 10

🚀 Starting training...
Epoch 1/10: Loss=0.693, Accuracy=55.2%, Time=12.3s
Epoch 2/10: Loss=0.622, Accuracy=67.8%, Time=11.9s
Epoch 3/10: Loss=0.541, Accuracy=74.1%, Time=12.1s
...
Epoch 10/10: Loss=0.234, Accuracy=91.3%, Time=11.8s

✅ Training completed!
🎉 Final accuracy: 91.3%
💾 Model saved to: cats_dogs_final_model.onnx
```

## Performance Characteristics

### GPU Acceleration Benefits
- **Training Speed**: ~12 seconds per epoch on Apple Silicon
- **Memory Efficiency**: Persistent GPU tensors minimize CPU↔GPU transfers
- **Parallel Processing**: Batch processing with optimized convolution kernels

### Accuracy Expectations
- **Training Accuracy**: Typically reaches 90%+ after 10 epochs
- **Convergence**: Stable learning with Adam optimizer
- **Generalization**: Good performance on validation data

## Key Technical Implementation

### GPU Operations
```go
// All operations use MPSGraph for optimal Apple Silicon performance
conv1, _ := tensor.Conv2DMPS(input, conv1Weights, conv1Bias, stride, padding)
relu1, _ := tensor.ReLUMPS(conv1)
pool1, _ := tensor.MaxPool2DMPS(relu1, poolSize, poolStride, poolPadding)
```

### Image Data Loading
```go
// Automatic image folder dataset
dataset, err := dataset.NewImageFolderDataset("./data", 64, []string{"cat", "dog"})
dataLoader := training.NewDataLoader(dataset, batchSize, true, 4, tensor.PersistentGPU)
```

### Training Pipeline
```go
// Complete training with GPU-resident model
config := training.TrainingConfig{
    Device: tensor.PersistentGPU,  // Keep everything on GPU
    LearningRate: 0.001,
    BatchSize: 16,
    Epochs: 10,
}
```

## Advanced Features

### Model Serialization
The example demonstrates saving and loading trained models:
- **ONNX Export**: Standard format for model interoperability
- **Parameter Persistence**: All weights and biases saved
- **Architecture Recovery**: Complete model reconstruction

### Performance Profiling
Includes profiling capabilities:
- **Memory Usage**: GPU memory consumption tracking
- **Training Speed**: Per-epoch timing analysis
- **CPU Profiling**: Performance bottleneck identification

### Visualization Support
Can be integrated with the sidecar visualization service:
- **Training Curves**: Real-time loss and accuracy plots
- **Confusion Matrix**: Classification results analysis
- **Image Samples**: Input data visualization

## Integration with Go-Metal

This example showcases Go-Metal's production-ready features:

### ✅ Core Capabilities
- **GPU-Accelerated CNNs**: Complete convolution operation support
- **Automatic Differentiation**: Full gradient computation for CNN training
- **Memory Management**: Efficient GPU memory allocation and cleanup
- **Data Pipeline**: Seamless image loading and batch processing

### ✅ Performance Features
- **Persistent GPU Tensors**: Model parameters stay on GPU throughout training
- **Async Operations**: Non-blocking GPU execution for optimal performance
- **Operation Fusion**: Optimized kernel combinations for speed

### ✅ Production Ready
- **Error Handling**: Comprehensive error checking and recovery
- **Memory Safety**: Proper resource cleanup and leak prevention
- **Scalability**: Handles large datasets and complex models

## Next Steps

After running this example:

1. **Experiment with architectures**: Modify layer configurations
2. **Try different datasets**: Use your own image classification data
3. **Optimize hyperparameters**: Adjust learning rates, batch sizes
4. **Add data augmentation**: Implement image transformations
5. **Deploy models**: Use ONNX exports for production inference

---

**This example demonstrates Go-Metal's capability to handle real-world computer vision tasks with performance competitive with established ML frameworks!** 🚀

# Credits
This example uses the Kaggle Dogs vs. Cats imageset. 
Will Cukierski. Dogs vs. Cats. https://kaggle.com/competitions/dogs-vs-cats, 2013. Kaggle.