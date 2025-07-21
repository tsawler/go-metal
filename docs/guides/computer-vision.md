# Computer Vision with Go-Metal

Go-Metal provides a comprehensive computer vision pipeline for loading, preprocessing, and training on image data. Built for Apple Silicon optimization, it offers high-performance image processing with GPU-resident data management.

## 📚 Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Quick Start](#quick-start)
- [Image Preprocessing](#image-preprocessing)
- [Datasets](#datasets)
- [Data Loaders](#data-loaders)
- [Training CNN Models](#training-cnn-models)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [Complete Examples](#complete-examples)

## Overview

The Go-Metal vision package (`github.com/tsawler/go-metal/vision`) provides three main components:

1. **Preprocessing** (`vision/preprocessing`) - Image loading, resizing, and normalization
2. **Datasets** (`vision/dataset`) - Dataset abstractions for various data sources
3. **DataLoaders** (`vision/dataloader`) - Efficient batch loading with caching

### Key Features

- 🚀 **High Performance**: GPU-optimized image preprocessing
- 🧠 **Smart Caching**: Automatic caching of preprocessed images
- 📊 **Flexible Datasets**: Support for custom and standard dataset formats  
- 🔄 **Efficient Loading**: Batched loading with memory reuse
- 🎯 **Multi-Class Support**: Works with any number of classes, not just binary
- 💾 **Memory Optimized**: Buffer reuse and shared caches

## Core Components

### Image Preprocessing

The preprocessing module handles image loading, resizing, and normalization:

```go
import "github.com/tsawler/go-metal/vision/preprocessing"

processor := preprocessing.NewImageProcessor(224) // 224x224 output size

file, _ := os.Open("image.jpg")
defer file.Close()

result, err := processor.DecodeAndPreprocess(file)
// result.Data contains normalized float32 RGB data in CHW format
// result.Channels, result.Width, result.Height contain dimensions
```

### Datasets

Datasets provide a standard interface for accessing image data:

```go
import "github.com/tsawler/go-metal/vision/dataset"

// ImageFolder dataset (discovers classes from directory structure)
dataset, err := dataset.NewImageFolderDataset("/path/to/data", []string{".jpg", ".png"})

// Cats vs Dogs dataset (specialized binary classification)
catsDogsDataset, err := dataset.NewCatsDogsDataset("/path/to/data", 1000)

// Custom dataset (implement the Dataset interface)
type CustomDataset struct { ... }
func (d *CustomDataset) Len() int { ... }
func (d *CustomDataset) GetItem(index int) (imagePath string, label int, err error) { ... }
```

### Data Loaders

Data loaders handle batching, shuffling, and caching:

```go
import "github.com/tsawler/go-metal/vision/dataloader"

config := dataloader.Config{
    BatchSize:    32,
    Shuffle:      true,
    ImageSize:    224,
    MaxCacheSize: 1000, // Cache up to 1000 preprocessed images
}

loader := dataloader.NewDataLoader(dataset, config)

// Get next batch
imageData, labelData, batchSize, err := loader.NextBatch()
// imageData: []float32 with shape [batch, channels, height, width]
// labelData: []int32 with class indices
```

## Quick Start

Here's a complete example that loads a dataset and trains a CNN:

```go
package main

import (
    "fmt"
    "log"

    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/vision/dataloader"
    "github.com/tsawler/go-metal/vision/dataset"
)

func main() {
    // Load dataset from directory structure:
    // data/
    //   class_a/
    //     image1.jpg, image2.jpg, ...
    //   class_b/
    //     image1.jpg, image2.jpg, ...
    dataset, err := dataset.NewImageFolderDataset("data", []string{".jpg"})
    if err != nil {
        log.Fatal(err)
    }

    // Split into train/validation
    trainDataset, valDataset := dataset.Split(0.8, true)

    // Create data loaders with shared cache
    batchSize := 16
    imageSize := 128
    
    trainLoader, valLoader := dataloader.CreateSharedDataLoaders(
        trainDataset, valDataset,
        dataloader.Config{
            BatchSize:    batchSize,
            ImageSize:    imageSize,
            MaxCacheSize: 500,
        },
    )

    // Build CNN model
    inputShape := []int{batchSize, 3, imageSize, imageSize}
    builder := layers.NewModelBuilder(inputShape)
    model, _ := builder.
        AddConv2D(32, 3, 2, 1, true, "conv1").
        AddReLU("relu1").
        AddConv2D(64, 3, 2, 1, true, "conv2"). 
        AddReLU("relu2").
        AddDense(128, true, "fc1").
        AddReLU("relu3").
        AddDense(dataset.NumClasses(), true, "output").
        Compile()

    // Configure trainer
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
        ProblemType:   training.Classification,
        LossFunction:  training.CrossEntropy,
        EngineType:    training.Dynamic,
    }

    trainer, _ := training.NewModelTrainer(model, config)
    defer trainer.Cleanup()

    // Train model
    epochs := 10
    stepsPerEpoch := trainDataset.Len() / batchSize
    
    session := trainer.CreateTrainingSession("VisionModel", epochs, stepsPerEpoch, 0)
    session.StartTraining()

    for epoch := 1; epoch <= epochs; epoch++ {
        session.StartEpoch(epoch)
        trainLoader.Reset()
        
        for step := 1; step <= stepsPerEpoch; step++ {
            imageData, labelData, actualBatchSize, err := trainLoader.NextBatch()
            if err != nil || actualBatchSize == 0 {
                break
            }

            inputShape := []int{actualBatchSize, 3, imageSize, imageSize}
            labels, _ := training.NewInt32Labels(labelData, []int{actualBatchSize, 1})
            
            result, err := trainer.TrainBatchUnified(imageData, inputShape, labels)
            if err != nil {
                continue
            }

            session.UpdateTrainingProgress(step, float64(result.Loss), 0.0)
        }

        session.FinishTrainingEpoch()
        session.PrintEpochSummary()
    }
}
```

## Image Preprocessing

### Basic Usage

```go
processor := preprocessing.NewImageProcessor(224)

file, err := os.Open("image.jpg")
if err != nil {
    log.Fatal(err)
}
defer file.Close()

result, err := processor.DecodeAndPreprocess(file)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Preprocessed %dx%dx%d image with %d elements\n",
    result.Width, result.Height, result.Channels, len(result.Data))
```

### Output Format

The preprocessor outputs images in **CHW format** (Channels, Height, Width):
- **Channels**: 3 (RGB)
- **Data Type**: `[]float32`
- **Value Range**: `[0.0, 1.0]` (normalized)
- **Memory Layout**: `[R_pixels..., G_pixels..., B_pixels...]`

### Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png) 
- Other formats supported by Go's image package

## Datasets

### ImageFolder Dataset

The ImageFolder dataset automatically discovers classes from directory structure:

```go
// Directory structure:
// root/
//   class1/
//     img1.jpg, img2.jpg, ...
//   class2/
//     img1.jpg, img2.jpg, ...

dataset, err := dataset.NewImageFolderDataset("root", []string{".jpg", ".png"})

// Dataset methods
fmt.Println("Total samples:", dataset.Len())
fmt.Println("Classes:", dataset.ClassNames()) 
fmt.Println("Distribution:", dataset.ClassDistribution())

// Get specific item
imagePath, label, err := dataset.GetItem(0)

// Split dataset
trainSet, valSet := dataset.Split(0.8, true) // 80% train, 20% validation
```

### Cats vs Dogs Dataset

Specialized dataset for binary classification:

```go
// Expects directory structure:
// data/
//   cat/
//     cat.1.jpg, cat.2.jpg, ...
//   dog/  
//     dog.1.jpg, dog.2.jpg, ...

dataset, err := dataset.NewCatsDogsDataset("data", 1000) // Max 1000 per class
summary := dataset.Summary()
fmt.Println(summary) // "Cats & Dogs Dataset: 2000 total images (1000 cats, 1000 dogs)"
```

### Custom Datasets

Implement the Dataset interface for custom data sources:

```go
type Dataset interface {
    Len() int
    GetItem(index int) (imagePath string, label int, err error)
}

type MyCustomDataset struct {
    imagePaths []string
    labels     []int
}

func (d *MyCustomDataset) Len() int {
    return len(d.imagePaths)
}

func (d *MyCustomDataset) GetItem(index int) (string, int, error) {
    if index >= len(d.imagePaths) {
        return "", 0, fmt.Errorf("index out of range")
    }
    return d.imagePaths[index], d.labels[index], nil
}
```

## Data Loaders

### Basic Configuration

```go
config := dataloader.Config{
    BatchSize:    32,           // Samples per batch
    Shuffle:      true,         // Shuffle data each epoch
    ImageSize:    224,          // Resize images to 224x224
    MaxCacheSize: 1000,         // Cache up to 1000 preprocessed images
    NumWorkers:   4,            // Parallel preprocessing workers (future)
}

loader := dataloader.NewDataLoader(dataset, config)
```

### Batch Loading

```go
// Reset loader at start of each epoch
loader.Reset()

for {
    imageData, labelData, batchSize, err := loader.NextBatch()
    if err != nil || batchSize == 0 {
        break // End of epoch
    }

    // imageData shape: [batchSize, 3, imageSize, imageSize]
    // labelData shape: [batchSize]
    
    // Use data for training/inference...
}
```

### Shared Caches

Use shared caches between train/validation loaders for better performance:

```go
// Option 1: Use convenience function
trainLoader, valLoader := dataloader.CreateSharedDataLoaders(
    trainDataset, valDataset,
    dataloader.Config{
        BatchSize:    32,
        ImageSize:    224,
        MaxCacheSize: 2000, // Shared cache size
    },
)

// Option 2: Manual shared cache
sharedCache := dataloader.NewCacheManager(2000, 3*224*224)

trainConfig := dataloader.Config{
    BatchSize:    32,
    ImageSize:    224,
    CacheManager: sharedCache,
}
valConfig := trainConfig // Same config
valConfig.Shuffle = false // Don't shuffle validation

trainLoader := dataloader.NewDataLoader(trainDataset, trainConfig)
valLoader := dataloader.NewDataLoader(valDataset, valConfig)
```

### Monitoring Performance

```go
// Check cache performance
stats := loader.Stats()
fmt.Println(stats) // "Cache: 150/1000 items, Hits: 45, Misses: 105, Hit Rate: 30.0%"

// Track progress
current, total := loader.Progress()
fmt.Printf("Progress: %d/%d (%.1f%%)\n", current, total, float64(current)/float64(total)*100)

// Clear cache if needed (only for owned caches)
loader.ClearCache()
```

## Training CNN Models

### Model Architecture

Build CNN models using the layers API:

```go
batchSize := 32
imageSize := 224
numClasses := 10

inputShape := []int{batchSize, 3, imageSize, imageSize}
builder := layers.NewModelBuilder(inputShape)

model, err := builder.
    // Feature extraction layers
    AddConv2D(32, 3, 1, 1, true, "conv1").  // 32 filters, 3x3 kernel, stride=1
    AddReLU("relu1").
    AddConv2D(32, 3, 2, 1, true, "conv2").  // stride=2 for downsampling
    AddReLU("relu2").
    AddConv2D(64, 3, 1, 1, true, "conv3").
    AddReLU("relu3").
    AddConv2D(64, 3, 2, 1, true, "conv4").  // Further downsampling
    AddReLU("relu4").
    
    // Classification layers  
    AddDense(128, true, "fc1").             // Fully connected
    AddReLU("relu5").
    AddDense(numClasses, true, "output").   // Output layer
    Compile()
```

### Training Configuration

```go
config := training.TrainerConfig{
    BatchSize:     batchSize,
    LearningRate:  0.001,
    OptimizerType: cgo_bridge.Adam,
    Beta1:         0.9,    // Adam parameters
    Beta2:         0.999,
    Epsilon:       1e-8,
    WeightDecay:   0.0001, // L2 regularization
    ProblemType:   training.Classification,
    LossFunction:  training.CrossEntropy,
    EngineType:    training.Dynamic, // Use Dynamic engine for CNNs
}

trainer, err := training.NewModelTrainer(model, config)
if err != nil {
    log.Fatal(err)
}
defer trainer.Cleanup()
```

### Training Loop

```go
// Enable optimizations
persistentInputShape := []int{batchSize, 3, imageSize, imageSize}
trainer.EnablePersistentBuffers(persistentInputShape)
trainer.SetAccuracyCheckInterval(1) // Check accuracy every step

epochs := 20
stepsPerEpoch := trainDataset.Len() / batchSize
valSteps := valDataset.Len() / batchSize

session := trainer.CreateTrainingSession("CNNModel", epochs, stepsPerEpoch, valSteps)
session.StartTraining()

for epoch := 1; epoch <= epochs; epoch++ {
    // Training phase
    session.StartEpoch(epoch)
    trainLoader.Reset()
    
    for step := 1; step <= stepsPerEpoch; step++ {
        imageData, labelData, actualBatchSize, err := trainLoader.NextBatch()
        if err != nil || actualBatchSize == 0 {
            break
        }

        // Prepare data
        inputShape := []int{actualBatchSize, 3, imageSize, imageSize}
        labels, err := training.NewInt32Labels(labelData, []int{actualBatchSize, 1})
        if err != nil {
            continue
        }

        // Train step
        result, err := trainer.TrainBatchUnified(imageData, inputShape, labels)
        if err != nil {
            continue
        }

        // Update progress
        accuracy := 0.0
        if result.HasAccuracy {
            accuracy = result.Accuracy
        }
        session.UpdateTrainingProgress(step, float64(result.Loss), accuracy)
    }

    session.FinishTrainingEpoch()

    // Validation phase
    session.StartValidation()
    valLoader.Reset()
    
    valCorrect := 0
    valTotal := 0

    for step := 1; step <= valSteps; step++ {
        imageData, labelData, actualBatchSize, err := valLoader.NextBatch()
        if err != nil || actualBatchSize == 0 {
            break
        }

        // Run inference
        inputShape := []int{actualBatchSize, 3, imageSize, imageSize}
        inferenceResult, err := trainer.InferBatch(imageData, inputShape)
        if err != nil {
            continue
        }

        // Calculate accuracy
        accuracy := trainer.CalculateAccuracy(
            inferenceResult.Predictions, labelData, actualBatchSize, numClasses)
        valCorrect += int(accuracy * float64(actualBatchSize))
        valTotal += actualBatchSize

        // Update validation progress
        currentAccuracy := float64(valCorrect) / float64(valTotal)
        session.UpdateValidationProgress(step, 0.0, currentAccuracy) // Loss estimated
    }

    session.FinishValidationEpoch()
    session.PrintEpochSummary()
}
```

## Performance Optimization

### Caching Strategy

```go
// Size cache appropriately
totalImages := dataset.Len()
cacheSize := min(totalImages, 2000) // Don't cache more than total images

config := dataloader.Config{
    MaxCacheSize: cacheSize,
    // ... other config
}

// Monitor cache hit rate
stats := loader.Stats()
fmt.Printf("Cache hit rate: %.1f%%\n", stats.HitRate*100)

// Hit rate > 80% is excellent
// Hit rate 50-80% is good  
// Hit rate < 50% may indicate cache too small
```

### Memory Management

```go
// Use appropriate batch sizes for your GPU memory
// M1/M2 Mac: Start with batch size 16-32 for 224x224 images
// M1 Pro/Max: Can handle 32-64
// M1 Ultra: Can handle 64+

batchSize := 32

// Use smaller images during development
imageSize := 128 // Instead of 224 for faster iteration

// Enable persistent buffers for better performance
trainer.EnablePersistentBuffers([]int{batchSize, 3, imageSize, imageSize})
```

### Training Speed Tips

```go
// 1. Use Dynamic engine for CNNs
config.EngineType = training.Dynamic

// 2. Enable accuracy checking only when needed
trainer.SetAccuracyCheckInterval(1) // Every step
// trainer.SetAccuracyCheckInterval(10) // Every 10 steps for speed

// 3. Limit steps during development
maxStepsPerEpoch := 100 // Limit for faster iteration
stepsPerEpoch := min(actualStepsPerEpoch, maxStepsPerEpoch)

// 4. Use shared caches
trainLoader, valLoader := dataloader.CreateSharedDataLoaders(...)
```

## Best Practices

### Dataset Organization

```
data/
├── train/
│   ├── class1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── img001.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

### Image Sizes

- **Development**: Start with 64x64 or 128x128 for fast iteration
- **Production**: Use 224x224 (standard) or 256x256 for better accuracy  
- **High Resolution**: 384x384 or 512x512 for fine-grained tasks

### Batch Sizes by Hardware

| Hardware | 224x224 | 128x128 | 64x64 |
|----------|---------|---------|-------|
| M1 Base  | 16-32   | 32-64   | 64-128|
| M1 Pro   | 32-64   | 64-128  | 128+  |
| M1 Max   | 64-128  | 128+    | 256+  |
| M1 Ultra | 128+    | 256+    | 512+  |

### Error Handling

```go
// Always check for errors in training loops
imageData, labelData, batchSize, err := loader.NextBatch()
if err != nil {
    log.Printf("Failed to load batch: %v", err)
    continue // Skip this batch
}

if batchSize == 0 {
    break // End of epoch
}

// Handle variable batch sizes (last batch may be smaller)
inputShape := []int{batchSize, 3, imageSize, imageSize} // Use actual batch size
```

## Complete Examples

### Multi-Class Classification

See the complete working example in [`docs/examples/vision-training-example.go`](../examples/vision-training-example.go). This example:

- Creates a synthetic 3-class dataset with distinct patterns
- Uses ImageFolderDataset for automatic class discovery
- Implements a CNN with Conv2D and Dense layers
- Shows proper training loop with validation
- Demonstrates cache usage and performance monitoring

### Binary Classification (Cats vs Dogs)

```go
// Use the specialized CatsDogsDataset
dataset, err := dataset.NewCatsDogsDataset("data", 5000) // 5000 per class max
if err != nil {
    log.Fatal(err)
}

fmt.Println(dataset.Summary()) // Shows distribution

// Build binary classifier
model, _ := builder.
    AddConv2D(32, 3, 2, 1, true, "conv1").
    AddReLU("relu1").
    AddConv2D(64, 3, 2, 1, true, "conv2").
    AddReLU("relu2").
    AddDense(128, true, "fc1").
    AddReLU("relu3").
    AddDense(2, true, "output"). // 2 classes: cat, dog
    Compile()
```

### Custom Preprocessing Pipeline

```go
// For advanced use cases, you can implement custom preprocessing
type CustomDataset struct {
    imagePaths []string
    labels     []int
    processor  *preprocessing.ImageProcessor
}

func (d *CustomDataset) GetItem(index int) (string, int, error) {
    // Apply custom augmentations, filtering, etc.
    imagePath := d.imagePaths[index]
    label := d.labels[index]
    
    // Custom validation
    if !d.isValidImage(imagePath) {
        return "", 0, fmt.Errorf("invalid image: %s", imagePath)
    }
    
    return imagePath, label, nil
}
```

## Next Steps

- 📖 **Learn More**: Check out [CNN Tutorial](../tutorials/cnn-tutorial.md) for deeper CNN concepts
- 🚀 **Performance**: See [Performance Guide](performance.md) for optimization techniques  
- 🔧 **Advanced**: Explore [Mixed Precision Training](../tutorials/mixed-precision.md) for maximum speed
- 📊 **Visualization**: Use [Visualization Guide](visualization.md) for training insights
- 💾 **Deployment**: Learn [Checkpoints](checkpoints.md) for model saving and loading

The Go-Metal vision pipeline provides everything needed for high-performance computer vision on Apple Silicon. Start with the quick examples and gradually build more complex models as you become familiar with the API.