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
# Using Homebrew (recommended)
brew install go

# Or download from https://golang.org/dl/
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
    "github.com/tsawler/go-metal/memory"
)

func main() {
    fmt.Println("🧪 Testing go-metal installation...")
    
    // Test basic functionality
    inputShape := []int{1, 4}
    builder := layers.NewModelBuilder(inputShape)
    model, err := builder.
        AddDense(2, true, "test_layer").
        Compile()
    
    if err != nil {
        log.Fatalf("❌ Installation test failed: %v", err)
    }
    
    fmt.Printf("✅ Go-Metal installed successfully!\n")
    fmt.Printf("   - Model created with %d layers\n", len(model.Layers))
    fmt.Printf("   - Input shape: %v\n", inputShape)
    fmt.Printf("   - Metal device available: %v\n", memory.GetDevice() != nil)
    
    fmt.Println("\n🚀 Ready to build ML models with go-metal!")
}
EOF

# Run the test
go run test_installation.go
```

Expected output:
```
🧪 Testing go-metal installation...
✅ Go-Metal installed successfully!
   - Model created with 1 layers
   - Input shape: [1 4]
   - Metal device available: true

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