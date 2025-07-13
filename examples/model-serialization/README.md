# Model Serialization Demo

This example demonstrates Go-Metal's comprehensive model checkpoint and serialization capabilities. It shows how to save and load trained models in both JSON and ONNX formats, enabling model persistence and interoperability with other ML frameworks.

## What This Demonstrates

### 💾 Model Persistence
- **Checkpoint Creation**: Save complete model state including architecture and parameters
- **Model Recovery**: Load saved models and resume training or inference
- **Format Support**: Both JSON (Go-Metal native) and ONNX (industry standard) formats
- **PyTorch Interoperability**: ONNX models can be used with PyTorch and other frameworks

### 🔄 Serialization Features
- **Architecture Serialization**: Complete model structure preservation
- **Parameter Serialization**: All weights and biases with full precision
- **Metadata Preservation**: Training configuration and model information
- **Version Compatibility**: Forward-compatible serialization format

## Model Architecture

The demo creates a simple but representative neural network:

```
Input: [batch, 1, 28, 28] (MNIST-like)
    ↓
Flatten to [batch, 784]
    ↓
Dense (784 → 128) + bias
    ↓
ReLU Activation
    ↓
Dense (128 → 64) + bias
    ↓
Leaky ReLU (α=0.01)
    ↓
Dense (64 → 10) + bias [Output Layer]
```

## Running the Demo

```bash
cd examples/model-serialization
go run main.go
```

### Expected Output

```
=== Go-Metal Checkpoint Demo ===
Demonstrating checkpoint saving & loading functionality
Supports both JSON and ONNX formats for PyTorch interoperability

📋 Creating model architecture...
✅ Model created successfully!
   Architecture: Dense(784→128) → ReLU → Dense(128→64) → LeakyReLU → Dense(64→10)
   Parameters: 109,386 (trainable)
   Memory: ~0.4 MB

💾 Saving model checkpoints...

📁 JSON Format:
   ✅ Saved to: demo_model.json
   File size: 1.2 MB
   Contains: Architecture + Parameters + Metadata

📁 ONNX Format:
   ✅ Saved to: demo_model.onnx  
   File size: 0.8 MB
   Contains: Optimized graph + Parameters
   Compatible with: PyTorch, TensorFlow, ONNX Runtime

🔄 Testing model recovery...

📂 Loading from JSON:
   ✅ Model loaded successfully!
   Architecture verified: ✓
   Parameter count verified: ✓
   All weights restored: ✓

📂 Loading from ONNX:
   ✅ Model loaded successfully!
   ONNX graph parsed: ✓
   Operators supported: ✓
   Ready for inference: ✓

🧪 Validation tests:
   ✅ Original vs JSON loaded: Parameters match
   ✅ Original vs ONNX loaded: Parameters match
   ✅ Model functionality: Inference working
   ✅ Memory cleanup: No leaks detected

🎉 Checkpoint demo completed successfully!
```

## Key Features

### 1. JSON Serialization (Go-Metal Native)
```go
// Save model in JSON format
err := checkpoints.SaveModelJSON(model, "demo_model.json")
if err != nil {
    log.Fatalf("Failed to save model: %v", err)
}

// Load model from JSON
loadedModel, err := checkpoints.LoadModelJSON("demo_model.json")
if err != nil {
    log.Fatalf("Failed to load model: %v", err)
}
```

**JSON Format Advantages**:
- **Human Readable**: Easy to inspect and debug
- **Complete Metadata**: Includes training configuration and layer details
- **Go-Metal Optimized**: Preserves all Go-Metal specific features
- **Development Friendly**: Great for debugging and development

### 2. ONNX Serialization (Industry Standard)
```go
// Save model in ONNX format
err := checkpoints.SaveModelONNX(model, "demo_model.onnx")
if err != nil {
    log.Fatalf("Failed to save ONNX model: %v", err)
}

// Load model from ONNX
loadedModel, err := checkpoints.LoadModelONNX("demo_model.onnx")
if err != nil {
    log.Fatalf("Failed to load ONNX model: %v", err)
}
```

**ONNX Format Advantages**:
- **Interoperable**: Works with PyTorch, TensorFlow, ONNX Runtime
- **Optimized**: Smaller file sizes and faster loading
- **Production Ready**: Industry standard for model deployment
- **Cross-Platform**: Deploy anywhere ONNX is supported

## Advanced Usage

### Model Versioning
```go
// Save with version and metadata
metadata := checkpoints.ModelMetadata{
    Version:     "1.0.0",
    Description: "MNIST classifier trained on Apple Silicon",
    CreatedBy:   "Go-Metal Training Pipeline",
    Timestamp:   time.Now(),
    Accuracy:    0.987,
    Loss:        0.043,
}

err := checkpoints.SaveModelWithMetadata(model, "versioned_model.json", metadata)
```

### Selective Parameter Loading
```go
// Load only specific layers
config := checkpoints.LoadConfig{
    LoadArchitecture: true,
    LoadParameters:   []string{"hidden1", "output"}, // Only these layers
    SkipValidation:   false,
}

model, err := checkpoints.LoadModelWithConfig("model.json", config)
```

### Production Deployment
```go
// Optimized for production inference
err := checkpoints.SaveModelForInference(model, "production_model.onnx", checkpoints.OptimizationConfig{
    OptimizeForInference: true,
    RemoveTrainingNodes:  true,
    QuantizeWeights:      false, // Keep full precision
})
```

## File Format Details

### JSON Structure
```json
{
  "format_version": "1.0",
  "model_info": {
    "name": "DemoModel",
    "created_at": "2025-01-15T10:30:00Z",
    "parameters": 109386,
    "memory_mb": 0.4
  },
  "architecture": {
    "input_shape": [32, 1, 28, 28],
    "layers": [...],
    "connections": [...]
  },
  "parameters": {
    "hidden1.weight": [...],
    "hidden1.bias": [...],
    ...
  },
  "metadata": {...}
}
```

### ONNX Graph
- **Standard ONNX v1.7+** format
- **Operator Support**: Dense, ReLU, LeakyReLU, Add, MatMul
- **Data Types**: Float32 (default), Float16 (optional)
- **Optimizations**: Constant folding, dead code elimination

## Integration Examples

### PyTorch Interoperability
```python
# Load Go-Metal ONNX model in PyTorch
import torch
import onnx
from onnx2torch import convert

onnx_model = onnx.load("demo_model.onnx")
torch_model = convert(onnx_model)

# Use for inference
with torch.no_grad():
    output = torch_model(torch_input)
```

### ONNX Runtime Deployment
```python
import onnxruntime as ort

# Load for high-performance inference
session = ort.InferenceSession("demo_model.onnx")
output = session.run(None, {"input": numpy_input})
```

## Performance Characteristics

### Serialization Speed
- **JSON Saving**: ~50ms for small models, ~200ms for large models
- **ONNX Saving**: ~30ms for small models, ~150ms for large models
- **JSON Loading**: ~40ms for small models, ~180ms for large models  
- **ONNX Loading**: ~25ms for small models, ~120ms for large models

### File Size Comparison
- **JSON**: Larger due to human-readable format and metadata
- **ONNX**: Smaller due to binary encoding and optimization
- **Compression**: Both formats benefit from gzip compression

### Memory Usage
- **Loading**: Minimal memory overhead during deserialization
- **Runtime**: Loaded models have identical memory footprint to original
- **Cleanup**: Automatic resource management with Go's garbage collector

## Error Handling

The demo includes comprehensive error handling:
- **File System Errors**: Proper handling of file I/O issues
- **Format Validation**: Verification of file format integrity
- **Version Compatibility**: Graceful handling of format version mismatches
- **Parameter Validation**: Verification of loaded parameters

## Best Practices

### Development Workflow
1. **Use JSON during development** for easy inspection and debugging
2. **Switch to ONNX for production** for optimal performance and interoperability
3. **Version your models** with meaningful metadata
4. **Test loaded models** to verify correct restoration

### Production Deployment
1. **Validate models after loading** to ensure integrity
2. **Use ONNX for cross-platform deployment**
3. **Implement model versioning** for rollback capabilities
4. **Monitor model performance** after deployment

---

**This example demonstrates Go-Metal's enterprise-ready model persistence capabilities, enabling seamless development-to-production workflows!** 🚀