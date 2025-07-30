# Convolutional Neural Network (CNN) Tutorial

Master building CNNs with go-metal for image processing and computer vision tasks.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- Build CNNs for image classification and processing
- Use Conv2D layers effectively with different kernel sizes
- Apply pooling and flattening for spatial reduction
- Combine CNNs with MLPs for complete architectures
- Train on image-like data with proper data formatting
- Optimize CNN performance for Apple Silicon

## 🏗️ CNN Architecture Fundamentals

### What is a CNN?

A Convolutional Neural Network uses specialized layers to process grid-like data (especially images):
- **Convolutional Layers**: Detect features using learnable filters
- **Activation Functions**: Introduce non-linearity after convolutions
- **Pooling Layers**: Reduce spatial dimensions while preserving features
- **Fully Connected**: Final classification/regression layers

### Mathematical Foundation

Each convolutional layer performs: `output = activation(input ⊛ kernel + bias)`

Where ⊛ represents the convolution operation.

```
Input Image → Conv2D → ReLU → Conv2D → ReLU → Flatten → Dense → Output
     ↓          ↓       ↓       ↓       ↓        ↓       ↓        ↓
  [32×32×3] → [30×30×16] → [30×30×16] → [28×28×32] → [28×28×32] → [25088] → [64] → [10]
```

## 🚀 Tutorial 1: Simple Image Classification CNN

Let's start with a complete working example for CIFAR-10-style image classification.

### Step 1: Setup and Dependencies

```go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "time"

    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("🖼️ CNN Tutorial: Image Classification")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Set random seed for reproducibility
    rand.Seed(42)
    
    // Problem setup
    batchSize := 8      // Smaller batch for CNN memory requirements
    imageHeight := 32   // Image dimensions
    imageWidth := 32
    channels := 3       // RGB channels
    numClasses := 10    // CIFAR-10 style (10 classes)
    
    // Continue with model building...
}
```

### Step 2: Build Simple CNN Architecture

```go
func buildSimpleCNN(batchSize, height, width, channels, numClasses int) (*layers.ModelSpec, error) {
    fmt.Println("🏗️ Building Simple CNN Architecture")
    
    // Input shape: [batch_size, channels, height, width]
    inputShape := []int{batchSize, channels, height, width}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // First convolutional block: 3×32×32 → 16×30×30
        AddConv2D(16, 3, "conv1").     // 16 filters, 3×3 kernel
        AddReLU("relu1").
        
        // Second convolutional block: 16×30×30 → 32×28×28  
        AddConv2D(32, 3, "conv2").     // 32 filters, 3×3 kernel
        AddReLU("relu2").
        
        // Flatten for fully connected layers: 32×28×28 → 25088
        AddFlatten("flatten").
        
        // Fully connected layers
        AddDense(64, true, "dense1").  // 25088 → 64
        AddReLU("relu3").
        AddDense(numClasses, true, "output"). // 64 → 10
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("CNN compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: %dx%dx%d → Conv(16) → Conv(32) → FC(64) → %d\n", 
               channels, height, width, numClasses)
    fmt.Printf("   📊 Total layers: %d\n", len(model.Layers))
    
    return model, nil
}
```

### Step 3: Generate Synthetic Image Data

```go
func generateImageData(batchSize, height, width, channels, numClasses int) ([]float32, []int32, []int, []int) {
    fmt.Println("🎨 Generating Synthetic Image Dataset")
    
    imageSize := height * width * channels
    inputData := make([]float32, batchSize*imageSize)
    labelData := make([]int32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        // Generate synthetic "images" with class-specific patterns
        class := rand.Intn(numClasses)
        labelData[i] = int32(class)
        
        baseIdx := i * imageSize
        
        // Create class-specific color patterns
        for c := 0; c < channels; c++ {
            // Different classes have different color intensities
            baseIntensity := float32(class) / float32(numClasses-1)
            channelBoost := float32(c) * 0.2 // RGB variation
            
            for h := 0; h < height; h++ {
                for w := 0; w < width; w++ {
                    // Create spatial patterns
                    centerH := float32(height) / 2.0
                    centerW := float32(width) / 2.0
                    distFromCenter := ((float32(h)-centerH)*(float32(h)-centerH) + 
                                     (float32(w)-centerW)*(float32(w)-centerW)) / 
                                     (centerH * centerH)
                    
                    // Pattern based on class and position
                    intensity := baseIntensity + channelBoost - distFromCenter*0.3
                    
                    // Add some noise
                    noise := (rand.Float32() - 0.5) * 0.2
                    intensity += noise
                    
                    // Clamp to [0, 1]
                    if intensity < 0 {
                        intensity = 0
                    }
                    if intensity > 1 {
                        intensity = 1
                    }
                    
                    // Store in NCHW format: [batch, channel, height, width]
                    pixelIdx := baseIdx + c*height*width + h*width + w
                    inputData[pixelIdx] = intensity
                }
            }
        }
    }
    
    inputShape := []int{batchSize, channels, height, width}
    labelShape := []int{batchSize}
    
    fmt.Printf("   ✅ Generated %d images of %dx%dx%d\n", 
               batchSize, channels, height, width)
    fmt.Printf("   🎨 Classes: 0-%d (synthetic patterns)\n", numClasses-1)
    
    return inputData, labelData, inputShape, labelShape
}
```

### Step 4: Configure CNN Training

```go
func setupCNNTrainer(model *layers.ModelSpec, batchSize int) (*training.ModelTrainer, error) {
    fmt.Println("⚙️ Configuring CNN Training Setup")
    
    config := training.TrainerConfig{
        // Basic parameters
        BatchSize:    batchSize,
        LearningRate: 0.001,  // Lower LR for CNNs typically
        
        // Optimizer: Adam for CNN training
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
        
        // Problem configuration
        EngineType:   training.Dynamic,           // Dynamic engine for CNNs
        ProblemType:  training.Classification,    // Image classification
        LossFunction: training.SparseCrossEntropy, // Integer class labels
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        return nil, fmt.Errorf("CNN trainer creation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Optimizer: Adam (lr=%.4f)\n", config.LearningRate)
    fmt.Printf("   ✅ Loss: SparseCrossEntropy\n")
    fmt.Printf("   🎯 Engine: Dynamic (CNN support)\n")
    
    return trainer, nil
}
```

### Step 5: CNN Training Loop with Progress Monitoring

```go
func trainCNN(trainer *training.ModelTrainer, inputData []float32, inputShape []int,
              labelData []int32, labelShape []int, epochs int) error {
    fmt.Printf("🚀 Training CNN for %d epochs\n", epochs)
    fmt.Println("Epoch | Loss     | Trend    | Status")
    fmt.Println("------|----------|----------|----------")
    
    var prevLoss float32 = 999.0
    
    for epoch := 1; epoch <= epochs; epoch++ {
        startTime := time.Now()
        
        // Execute training step
        result, err := trainer.TrainBatch(inputData, inputShape, labelData, labelShape)
        if err != nil {
            return fmt.Errorf("CNN training epoch %d failed: %v", epoch, err)
        }
        
        elapsed := time.Since(startTime)
        
        // Loss trend analysis
        var trend string
        if result.Loss < prevLoss {
            trend = "↓ Good"
        } else {
            trend = "↑ Check"
        }
        
        // Training status
        var status string
        if result.Loss < 0.5 {
            status = "Converging"
        } else if result.Loss < 1.5 {
            status = "Learning"
        } else {
            status = "Starting"
        }
        
        fmt.Printf("%5d | %.6f | %-8s | %s (%.2fs)\n", 
                   epoch, result.Loss, trend, status, elapsed.Seconds())
        
        prevLoss = result.Loss
        
        // Early stopping for demo
        if result.Loss < 0.1 {
            fmt.Printf("🎉 CNN converged! (loss < 0.1)\n")
            break
        }
    }
    
    return nil
}
```

### Step 6: Complete CNN Training Program

```go
func main() {
    fmt.Println("🖼️ CNN Tutorial: Image Classification")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Set random seed for reproducibility
    rand.Seed(42)
    
    // Problem setup
    batchSize := 8
    imageHeight := 32
    imageWidth := 32
    channels := 3
    numClasses := 10
    epochs := 20
    
    // Step 1: Build CNN model
    model, err := buildSimpleCNN(batchSize, imageHeight, imageWidth, channels, numClasses)
    if err != nil {
        log.Fatalf("CNN model building failed: %v", err)
    }
    
    // Step 2: Setup trainer
    trainer, err := setupCNNTrainer(model, batchSize)
    if err != nil {
        log.Fatalf("CNN trainer setup failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Step 3: Generate synthetic image data
    inputData, labelData, inputShape, labelShape := generateImageData(
        batchSize, imageHeight, imageWidth, channels, numClasses)
    
    // Step 4: Train CNN
    err = trainCNN(trainer, inputData, inputShape, labelData, labelShape, epochs)
    if err != nil {
        log.Fatalf("CNN training failed: %v", err)
    }
    
    fmt.Println("\n🎓 CNN Tutorial Complete!")
    fmt.Println("   ✅ Successfully built and trained a CNN")
    fmt.Println("   ✅ Used Conv2D layers for feature extraction")
    fmt.Println("   ✅ Applied GPU-accelerated convolution on Apple Silicon")
    fmt.Println("   ✅ Demonstrated image classification pipeline")
}
```

## 🧠 Tutorial 2: Advanced CNN with Modern Techniques

Now let's build a more sophisticated CNN with modern architecture patterns.

### Advanced CNN Architecture

```go
func buildAdvancedCNN(batchSize, height, width, channels, numClasses int) (*layers.ModelSpec, error) {
    fmt.Println("🏗️ Building Advanced CNN with Modern Techniques")
    
    inputShape := []int{batchSize, channels, height, width}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Block 1: Initial feature extraction
        AddConv2D(32, 3, "conv1_1").   // 3×32×32 → 32×30×30
        AddReLU("relu1_1").
        AddConv2D(32, 3, "conv1_2").   // 32×30×30 → 32×28×28
        AddReLU("relu1_2").
        
        // Block 2: Increased feature maps
        AddConv2D(64, 3, "conv2_1").   // 32×28×28 → 64×26×26
        AddReLU("relu2_1").
        AddConv2D(64, 3, "conv2_2").   // 64×26×26 → 64×24×24
        AddReLU("relu2_2").
        
        // Block 3: Deep feature extraction
        AddConv2D(128, 3, "conv3_1").  // 64×24×24 → 128×22×22
        AddReLU("relu3_1").
        AddConv2D(128, 3, "conv3_2").  // 128×22×22 → 128×20×20
        AddReLU("relu3_2").
        
        // Global feature aggregation
        AddFlatten("flatten").         // 128×20×20 → 51200
        
        // Classifier with regularization
        AddDense(256, true, "dense1"). // 51200 → 256
        AddReLU("relu_fc1").
        AddDropout(0.5, "dropout1").   // Prevent overfitting
        
        AddDense(128, true, "dense2"). // 256 → 128
        AddReLU("relu_fc2").
        AddDropout(0.3, "dropout2").
        
        AddDense(numClasses, true, "output"). // 128 → 10
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("advanced CNN compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: VGG-style CNN with %d layers\n", len(model.Layers))
    fmt.Printf("   🔧 Features: Deep convolution + dropout regularization\n")
    fmt.Printf("   📊 Progression: 32→64→128 filters\n")
    
    return model, nil
}
```

### Different Kernel Sizes and Patterns

```go
func buildMultiScaleCNN(batchSize, height, width, channels, numClasses int) (*layers.ModelSpec, error) {
    fmt.Println("🏗️ Building Multi-Scale CNN")
    
    inputShape := []int{batchSize, channels, height, width}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Multi-scale feature extraction
        AddConv2D(16, 1, "conv_1x1").  // 1×1 for point-wise features
        AddReLU("relu_1x1").
        
        AddConv2D(32, 3, "conv_3x3").  // 3×3 for local features
        AddReLU("relu_3x3").
        
        AddConv2D(32, 5, "conv_5x5").  // 5×5 for larger patterns
        AddReLU("relu_5x5").
        
        // Aggregate features
        AddConv2D(64, 3, "conv_agg").
        AddReLU("relu_agg").
        
        // Classification head
        AddFlatten("flatten").
        AddDense(128, true, "dense").
        AddReLU("relu_dense").
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("multi-scale CNN compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Multi-scale architecture: 1×1, 3×3, 5×5 kernels\n")
    fmt.Printf("   🔍 Captures features at different scales\n")
    
    return model, nil
}
```

## 🎯 Tutorial 3: Real-World Applications

### MNIST-Style Digit Recognition CNN

```go
func buildMNISTCNN(batchSize int) (*layers.ModelSpec, error) {
    fmt.Println("🔢 Building MNIST-Style CNN")
    
    // MNIST: 28×28 grayscale images
    height, width, channels := 28, 28, 1
    numClasses := 10
    
    inputShape := []int{batchSize, channels, height, width}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // First conv block
        AddConv2D(32, 3, "conv1").     // 1×28×28 → 32×26×26
        AddReLU("relu1").
        AddConv2D(32, 3, "conv2").     // 32×26×26 → 32×24×24
        AddReLU("relu2").
        
        // Second conv block  
        AddConv2D(64, 3, "conv3").     // 32×24×24 → 64×22×22
        AddReLU("relu3").
        AddConv2D(64, 3, "conv4").     // 64×22×22 → 64×20×20
        AddReLU("relu4").
        
        // Classification
        AddFlatten("flatten").         // 64×20×20 → 25600
        AddDense(128, true, "dense1"). // 25600 → 128
        AddReLU("relu5").
        AddDropout(0.5, "dropout").
        AddDense(numClasses, true, "output"). // 128 → 10
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("MNIST CNN compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ MNIST CNN: Optimized for 28×28 digit recognition\n")
    fmt.Printf("   📊 Architecture: 1→32→32→64→64→128→10\n")
    
    return model, nil
}
```

### CIFAR-10 Style Color Image CNN

```go
func buildCIFARCNN(batchSize int) (*layers.ModelSpec, error) {
    fmt.Println("🌈 Building CIFAR-10 Style CNN")
    
    // CIFAR-10: 32×32 RGB images
    height, width, channels := 32, 32, 3
    numClasses := 10
    
    inputShape := []int{batchSize, channels, height, width}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Block 1: Basic features
        AddConv2D(32, 3, "conv1_1").
        AddReLU("relu1_1").
        AddConv2D(32, 3, "conv1_2").
        AddReLU("relu1_2").
        
        // Block 2: Mid-level features
        AddConv2D(64, 3, "conv2_1").
        AddReLU("relu2_1").
        AddConv2D(64, 3, "conv2_2").
        AddReLU("relu2_2").
        
        // Block 3: High-level features
        AddConv2D(128, 3, "conv3_1").
        AddReLU("relu3_1").
        AddConv2D(128, 3, "conv3_2").
        AddReLU("relu3_2").
        
        // Global classification
        AddFlatten("flatten").
        AddDense(512, true, "dense1").
        AddReLU("relu_fc1").
        AddDropout(0.5, "dropout1").
        
        AddDense(256, true, "dense2").
        AddReLU("relu_fc2").
        AddDropout(0.3, "dropout2").
        
        AddDense(numClasses, true, "output").
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("CIFAR CNN compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ CIFAR-10 CNN: Color image classification\n")
    fmt.Printf("   🎨 RGB channels: 3×32×32 → 32→64→128 filters\n")
    
    return model, nil
}
```

## 🔧 Advanced CNN Techniques

### Data Format Understanding

```go
func demonstrateDataFormats() {
    fmt.Println("📊 CNN Data Format Guide")
    
    // Go-Metal uses NCHW format: [Batch, Channels, Height, Width]
    batchSize := 4
    channels := 3    // RGB
    height := 32
    width := 32
    
    fmt.Printf("Input Tensor Shape: [%d, %d, %d, %d]\n", 
               batchSize, channels, height, width)
    fmt.Println("   N (Batch): Number of images processed together")
    fmt.Println("   C (Channels): Color channels (1=grayscale, 3=RGB)")
    fmt.Println("   H (Height): Image height in pixels")
    fmt.Println("   W (Width): Image width in pixels")
    
    // Memory layout explanation
    totalPixels := batchSize * channels * height * width
    fmt.Printf("\nMemory Layout: %d total elements\n", totalPixels)
    fmt.Println("   Data stored as: [img0_R, img0_G, img0_B, img1_R, img1_G, ...]")
    fmt.Println("   Access pattern: batch→channel→height→width")
}
```

### Convolution Parameter Calculation

```go
func calculateConvOutput(inputH, inputW, kernelSize, stride, padding int) (int, int) {
    // Output size formula: (input + 2*padding - kernel) / stride + 1
    outputH := (inputH + 2*padding - kernelSize) / stride + 1
    outputW := (inputW + 2*padding - kernelSize) / stride + 1
    return outputH, outputW
}

func demonstrateConvMath() {
    fmt.Println("🧮 Convolution Math Examples")
    
    examples := []struct {
        name string
        inH, inW, kernel, stride, padding int
    }{
        {"Standard 3×3", 32, 32, 3, 1, 0},
        {"Same padding", 32, 32, 3, 1, 1},
        {"Large kernel", 32, 32, 5, 1, 0},
        {"Stride 2", 32, 32, 3, 2, 0},
    }
    
    for _, ex := range examples {
        outH, outW := calculateConvOutput(ex.inH, ex.inW, ex.kernel, ex.stride, ex.padding)
        fmt.Printf("   %s: %dx%d → %dx%d (k=%d, s=%d, p=%d)\n",
                   ex.name, ex.inH, ex.inW, outH, outW, 
                   ex.kernel, ex.stride, ex.padding)
    }
}
```

### Performance Optimization for CNNs

⚡ **CNN Performance Optimization Guide**

#### Batch Size Optimization

🎯 **Recommended Batch Sizes**:

| GPU Memory | CNN Type | Recommended Batch Size | Notes |
|------------|----------|----------------------|-------|
| 8GB | Small CNN (3-5 layers) | 32-64 | Good balance |
| 8GB | Medium CNN (10-15 layers) | 16-32 | Memory constrained |
| 8GB | Large CNN (20+ layers) | 8-16 | May need gradient accumulation |
| 16GB+ | Any size | 32-128 | More flexibility |

**Tips**:
- Start with smaller batch sizes for CNNs (more memory intensive than MLPs)
- Increase gradually while monitoring GPU memory
- Power of 2 sizes (8, 16, 32, 64) often perform better on GPUs
- Use gradient accumulation for effective larger batches

#### Channel Progression Strategy

🔢 **Optimal Filter Counts**:

```
First Layer:  16-32 filters (captures basic edges/colors)
     ↓
Middle Layers: Double when spatial size halves
     ↓
Deep Layers:  128-512 filters (high-level features)
```

**Common Progressions**:
- Small models: 16 → 32 → 64
- Medium models: 32 → 64 → 128
- Large models: 64 → 128 → 256 → 512

#### Architecture Optimization

🏗️ **Kernel Size Guidelines**:

| Pattern | Memory Cost | Computation | When to Use |
|---------|------------|-------------|-------------|
| Many 3×3 | Medium | Medium | Default choice - proven effective |
| Few 5×5 | High | High | Early layers for larger receptive field |
| 1×1 convs | Low | Low | Channel reduction, feature combination |
| Depthwise separable | Very Low | Low | Mobile/embedded deployment |

#### Memory Management

💾 **Memory Usage Estimation**:

```go
// Rough memory calculation for a conv layer
memoryMB := (batchSize * channels * height * width * 4) / (1024 * 1024)

// Example: 32×64×28×28 feature map
// Memory: 32 * 64 * 28 * 28 * 4 / (1024*1024) ≈ 6.25 MB
```

**Memory Optimization Techniques**:
1. **Gradient Checkpointing**: Trade computation for memory
2. **Mixed Precision (FP16)**: Halve memory usage
3. **In-place Operations**: Reuse tensors when possible
4. **Batch Size Reduction**: Most direct way to reduce memory

#### Performance Profiling

```go
// Profile CNN performance
type CNNProfile struct {
    LayerName       string
    ComputeTimeMs   float64
    MemoryMB        float64
    FLOPs          int64
}

// Key metrics to monitor:
// - Forward pass time
// - Backward pass time
// - Peak memory usage
// - GPU utilization percentage
```

## 📊 CNN Architecture Comparison

### Model Complexity Analysis

📊 **CNN Architecture Comparison**

| Architecture | Layers | Filter Progression | Use Case | Complexity | Parameters |
|--------------|--------|-------------------|----------|------------|------------|
| **Simple CNN** | 7 | 16→32 | Learning/Testing | Low | ~50K |
| **Advanced CNN** | 15 | 32→64→128 | Real datasets | Medium | ~500K |
| **Multi-Scale** | 11 | 16→32→32→64 | Complex patterns | Medium | ~200K |
| **MNIST CNN** | 13 | 32→32→64→64 | Digit recognition | Medium | ~400K |
| **CIFAR CNN** | 19 | 32→64→128 | Color images | High | ~1M |

#### Architecture Details

**Simple CNN** (Tutorial Model):
- 2 conv blocks + classifier
- Suitable for grayscale images
- Fast training, good for learning

**Advanced CNN** (General Purpose):
- 3-4 conv blocks with pooling
- Batch normalization layers
- Dropout regularization
- Handles RGB images well

**Multi-Scale CNN** (Feature Pyramid):
- Multiple kernel sizes
- Captures features at different scales
- Good for varied object sizes

**MNIST CNN** (Digit Specialist):
- Optimized for 28×28 grayscale
- High accuracy (>99%)
- Efficient architecture

**CIFAR CNN** (Complex Images):
- Deeper architecture
- More filters for color features
- Data augmentation recommended

## 🎓 Best Practices Summary

### CNN Design Principles

🎓 **CNN Best Practices**

#### Architecture Design

🏗️ **Design Principles**:

✅ **Start simple, add complexity gradually**
- Begin with 2-3 conv blocks
- Add layers only if performance plateaus
- Monitor validation accuracy improvements

✅ **Use proven patterns**
- VGG-style: Stacked 3×3 convolutions
- ResNet-style: Skip connections for deeper networks
- Inception-style: Multi-scale feature extraction

✅ **Ensure gradual spatial reduction**
- Avoid aggressive pooling early
- Typical reduction: 32→16→8→4
- Maintain aspect ratios when possible

✅ **Filter progression strategy**
- Start with 16-32 filters
- Double filters when halving spatial size
- Common: 32→64→128→256

#### Training Configuration

⚙️ **Optimal Settings**:

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Learning Rate | 0.001-0.01 | Lower than MLPs due to complexity |
| Optimizer | Adam | Good default, stable convergence |
| Weight Decay | 1e-4 to 1e-3 | Helps prevent overfitting |
| Dropout | 0.3-0.5 | Apply in FC layers only |
| Batch Size | 16-64 | Limited by GPU memory |

#### Data Handling

📊 **Data Best Practices**:

✅ **Image Preprocessing**:
```go
// Standard normalization
func normalizeImage(img []float32) []float32 {
    for i := range img {
        img[i] = img[i] / 255.0  // Scale to [0, 1]
    }
    return img
}

// Channel-wise normalization (ImageNet style)
func normalizeImageNet(img []float32, channels int) []float32 {
    mean := []float32{0.485, 0.456, 0.406}
    std := []float32{0.229, 0.224, 0.225}
    // Apply per channel
}
```

✅ **Data Format Consistency**:
- Always use NCHW format: [batch, channels, height, width]
- Ensure all images have same dimensions
- Handle different aspect ratios properly (crop/pad)

✅ **Data Augmentation** (for production models):
- Random crops
- Horizontal flips
- Color jittering
- Rotation (small angles)

#### Performance Optimization

🚀 **Performance Guidelines**:

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Smaller batch sizes | Reduces memory | Start with 16, increase carefully |
| Mixed precision | 2x speedup, 50% memory | Use FP16 for large models |
| Gradient accumulation | Effective larger batches | Accumulate over mini-batches |
| Profile first | Identify bottlenecks | Use profiling tools |

#### Common Pitfalls to Avoid

❌ **Avoid These Mistakes**:
1. Too aggressive pooling early (loses information)
2. Very deep networks without skip connections
3. Large kernels (7×7, 9×9) throughout network
4. Forgetting to normalize inputs
5. Batch size too large for GPU memory
6. Learning rate too high (NaN losses)

### Common CNN Patterns

🔧 **Proven Architectural Patterns**

#### Conv Block Pattern

📐 **Standard Convolutional Block**:
```
Conv2D → ReLU → Conv2D → ReLU → MaxPool2D
```

- Two convolutions extract features at different levels
- ReLU adds non-linearity between convolutions
- MaxPooling reduces spatial dimensions
- Pattern inspired by VGG architecture

#### Classifier Head Pattern

🎯 **Standard Classification Head**:
```
Flatten → Dense(256) → ReLU → Dropout(0.5) → Dense(num_classes)
```

- Flatten converts 2D features to 1D vector
- First Dense layer learns high-level combinations
- Dropout prevents overfitting
- Final Dense outputs class probabilities

#### Filter Progression Pattern

📊 **Typical Channel Progression**:

| Layer | Input Shape | Output Shape | Description |
|-------|-------------|--------------|-------------|
| Input | 3×32×32 | - | RGB image |
| Conv1 | 3×32×32 | 32×30×30 | Initial feature extraction |
| Pool1 | 32×30×30 | 32×15×15 | Spatial reduction |
| Conv2 | 32×15×15 | 64×13×13 | Deeper features |
| Pool2 | 64×13×13 | 64×6×6 | Further reduction |
| Conv3 | 64×6×6 | 128×4×4 | High-level features |

**Key principle**: Double channels when halving spatial dimensions

#### Optimization Patterns

⚡ **Kernel Size Selection**:

| Kernel Size | Use Case | Advantages | Disadvantages |
|------------|----------|------------|---------------|
| **1×1** | Channel reduction | Efficient, learns channel combinations | No spatial information |
| **3×3** | Standard feature extraction | Good balance of efficiency and effectiveness | May need multiple layers |
| **5×5** | Larger receptive field | Captures broader patterns | Higher computational cost |
| **7×7** | Initial layers only | Large context from input | Very expensive, rarely used |

#### Modern Architectural Patterns

**Residual Connections** (ResNet-inspired):
```go
// Skip connection pattern
input := previousLayer
conv1 := builder.AddConv2D(filters, 3, 1, 1, true, "conv1")
relu1 := builder.AddReLU("relu1")
conv2 := builder.AddConv2D(filters, 3, 1, 1, true, "conv2")
// Add input to conv2 output (residual connection)
```

**Inception Module** (GoogLeNet-inspired):
```go
// Multiple kernel sizes in parallel
branch1 := builder.AddConv2D(64, 1, 1, 0, true, "1x1")      // 1×1 conv
branch2 := builder.AddConv2D(128, 3, 1, 1, true, "3x3")    // 3×3 conv
branch3 := builder.AddConv2D(32, 5, 1, 2, true, "5x5")     // 5×5 conv
// Concatenate branches
```

## 🚀 Next Steps

You've mastered CNN building with go-metal! Continue your journey:

- **[Performance Guide](../guides/performance.md)** - Optimize CNN training speed
- **[Mixed Precision Tutorial](mixed-precision.md)** - FP16 for faster training
- **[Advanced Examples](../examples/)** - Real-world CNN applications
- **[Custom Layers Guide](custom-layers.md)** - Build specialized CNN layers

**Ready for production?** Check out the [Image Classification Example](../examples/cats-dogs-classification.md) for a complete CNN project.

---

## 🧠 Key Takeaways

- **CNNs excel at spatial data**: Perfect for images, maps, and grid-like data
- **Layer progression matters**: Start simple, build complexity gradually
- **Memory awareness**: CNNs require more GPU memory than MLPs
- **Proven patterns work**: 3×3 convolutions, filter doubling, dropout regularization
- **Go-metal advantages**: GPU-resident convolution, automatic optimization, Metal acceleration

With these skills, you can tackle any computer vision problem using CNNs in go-metal!