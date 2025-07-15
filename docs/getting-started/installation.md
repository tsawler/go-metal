# Installation Guide

Get go-metal up and running on your Apple Silicon Mac.

## 📋 Requirements

### System Requirements
- **macOS 12.0+** (Monterey or later)
- **Apple Silicon Mac** (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, M3, M3 Pro, M3 Max)
- **Xcode Command Line Tools** (for Metal development)
- **Go 1.19+** (recommended: Go 1.21 or later)

### Why Apple Silicon Only?
Go-Metal is specifically optimized for Apple's Metal Performance Shaders Graph (MPSGraph), which provides exceptional GPU acceleration on Apple Silicon. While Metal exists on Intel Macs, the performance benefits and architectural optimizations are designed for Apple's unified memory architecture.

## 🛠️ Installation Steps

### 1. Install Xcode Command Line Tools

```bash
# Install Xcode command line tools (required for Metal development)
xcode-select --install
```

If you already have Xcode installed, ensure it's up to date:

```bash
# Verify Xcode command line tools
xcode-select -p
# Should output: /Applications/Xcode.app/Contents/Developer
```

### 2. Verify Go Installation

```bash
# Check Go version (1.19+ required, 1.21+ recommended)
go version
# Should output: go version go1.21.x darwin/arm64
```

If you need to install or update Go:

```bash
# Download from https://go.dev/dl (recommended)

# Or, using Homebrew
brew install go
```

### 3. Install Go-Metal

```bash
# Initialize a new Go module for your project
mkdir my-ml-project
cd my-ml-project
go mod init my-ml-project

# Add go-metal dependency
go get github.com/tsawler/go-metal
```

### 4. Verify Installation

Create a simple test file to verify everything works:

```bash
# Create test file
cat > test_installation.go << 'EOF'
package main

import (
    "fmt"
    "log"
    
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
    "github.com/tsawler/go-metal/cgo_bridge"
)

func main() {
    fmt.Println("🧪 Testing go-metal installation...")
    
    // Test 1: Basic model creation
    fmt.Println("\n1. Testing model creation...")
    inputShape := []int{1, 4}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(2, true, "test_layer").
        Compile()
    
    if err != nil {
        log.Fatalf("❌ Model creation failed: %v", err)
    }
    
    fmt.Printf("✅ Model created successfully!\n")
    fmt.Printf("   - Layers: %d\n", len(model.Layers))
    fmt.Printf("   - Input shape: %v\n", inputShape)
    fmt.Printf("   - Parameters: %d\n", model.TotalParameters)
    
    // Test 2: Training configuration
    fmt.Println("\n2. Testing training configuration...")
    config := training.TrainerConfig{
        BatchSize:     1,
        LearningRate:  0.01,
        OptimizerType: cgo_bridge.Adam,
        LossFunction:  training.CrossEntropy,
        ProblemType:   training.Classification,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    fmt.Printf("✅ Training configuration created!\n")
    fmt.Printf("   - Optimizer: Adam\n")
    fmt.Printf("   - Learning rate: %.4f\n", config.LearningRate)
    fmt.Printf("   - Batch size: %d\n", config.BatchSize)
    
    // Test 3: Trainer creation (this will initialize Metal device)
    fmt.Println("\n3. Testing trainer creation...")
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        log.Fatalf("❌ Trainer creation failed: %v", err)
    }
    defer trainer.Cleanup()
    
    fmt.Printf("✅ Trainer created successfully!\n")
    fmt.Printf("   - Metal device initialized\n")
    fmt.Printf("   - GPU-resident training ready\n")
    
    fmt.Println("\n🎉 Go-Metal installation test PASSED!")
    fmt.Println("✅ All components working correctly")
    fmt.Println("🚀 Ready to build ML models with go-metal!")
}
EOF

# Run the test
go run test_installation.go
```

Expected output:
```
🧪 Testing go-metal installation...

1. Testing model creation...
✅ Model created successfully!
   - Layers: 1
   - Input shape: [1 4]
   - Parameters: 10

2. Testing training configuration...
✅ Training configuration created!
   - Optimizer: Adam
   - Learning rate: 0.0100
   - Batch size: 1

3. Testing trainer creation...
🧠 Smart Routing Analysis:
   - Input: 2D [1 4]
   - Architecture: Simple (1 layers, 10 params)
   - Pattern: CNN=false, MLP=true
   - Selected Engine: Dynamic
🔧 Creating Dynamic Engine (any architecture support)
✅ Created engine: isDynamic=true
✅ Trainer created successfully!
   - Metal device initialized
   - GPU-resident training ready

🎉 Go-Metal installation test PASSED!
✅ All components working correctly
🚀 Ready to build ML models with go-metal!
```

## 🔧 Development Setup (Optional)

For the best development experience, consider these additional tools:

### IDE Setup

**VS Code with Go Extension:**
```bash
# Install VS Code Go extension
code --install-extension golang.go
```

**GoLand**: JetBrains' Go IDE with excellent debugging support.

### Additional Tools

```bash
# Useful development tools
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install honnef.co/go/tools/cmd/staticcheck@latest
```

## 🚨 Troubleshooting

### Common Issues

**"Metal device not found"**
```bash
# Verify you're on Apple Silicon
uname -m
# Should output: arm64

# Check macOS version
sw_vers
# ProductVersion should be 12.0 or higher
```

**"CGO compilation failed"**
```bash
# Ensure Xcode command line tools are properly installed
sudo xcode-select --reset
xcode-select --install

# Verify CGO is enabled
go env CGO_ENABLED
# Should output: 1
```

**"Module not found" errors**
```bash
# Clean module cache and retry
go clean -modcache
go mod tidy
go mod download
```

**Performance issues**
```bash
# Ensure you're running on Apple Silicon (not Rosetta)
arch
# Should output: arm64

# If running under Rosetta, use native terminal:
# Open Terminal, Get Info, uncheck "Open using Rosetta"
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Go-Metal provides detailed error messages
2. **Verify system requirements**: Ensure Apple Silicon + macOS 12.0+
3. **Update dependencies**: `go get -u github.com/tsawler/go-metal`
4. **File an issue**: [GitHub Issues](https://github.com/tsawler/go-metal/issues)

## ✅ Installation Complete

Once the test passes, you're ready to start building ML models! 

**Next Steps:**
- **[Quick Start](quick-start.md)** - Train your first model in 5 minutes
- **[Basic Concepts](basic-concepts.md)** - Learn core go-metal concepts
- **[MLP Tutorial](../tutorials/mlp-tutorial.md)** - Build a complete neural network

## 📦 Project Structure Recommendation

For new projects, consider this structure:

```
my-ml-project/
├── go.mod
├── go.sum
├── main.go                 # Your main application
├── data/                   # Training data
├── models/                 # Saved model checkpoints
├── examples/               # Example scripts
└── README.md
```

This keeps your ML project organized and follows Go conventions.