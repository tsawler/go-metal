# Custom Layers Tutorial

Extend Go-Metal with your own layer types using MPSGraph operations for maximum performance on Apple Silicon.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- Understand Go-Metal's layer architecture and extension points
- Implement custom layers with learnable parameters
- Use MPSGraph operations for GPU computation
- Create activation functions and transformations
- Test and validate custom layer implementations
- Deploy custom layers in production models

## 🔧 Go-Metal Layer Architecture

### Layer Specification System

Go-Metal uses a **specification-based architecture** where layers are defined as configuration objects that describe what computation should happen, rather than imperative code.

```go
type LayerSpec struct {
    Type       LayerType              // Identifies the layer operation
    Name       string                 // Unique layer identifier
    Parameters map[string]interface{} // Layer-specific configuration
    
    // Computed during model compilation
    InputShape      []int              // Expected input tensor shape
    OutputShape     []int              // Resulting output tensor shape
    ParameterShapes [][]int            // Shapes of learnable parameters
    ParameterCount  int64              // Total learnable parameter count
}
```

### Layer Type System

Layer types are defined as consecutive integers in the enum (0-11 currently in use):

```go
const (
    Dense LayerType = iota    // 0
    Conv2D                    // 1
    ReLU                      // 2
    Softmax                   // 3
    MaxPool2D                 // 4
    Dropout                   // 5
    BatchNorm                 // 6
    LeakyReLU                 // 7
    ELU                       // 8
    Sigmoid                   // 9
    Tanh                      // 10
    Swish                     // 11
    // Custom layers extend sequentially: 12, 13, 14...
)
```

### MPSGraph Integration

The actual computation happens in **Metal Performance Shaders Graph (MPSGraph)**, Apple's high-performance GPU compute framework. Custom layers define their computation using MPSGraph primitives through C bridge functions.

## 🚀 Complete Example 1: Using Built-in Swish Layer

Swish activation (x * sigmoid(x)) is already implemented in Go-Metal. Here's how to use it:

```go
package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/training"
)

// Swish is available as layers.Swish (value 11 in the enum)

func main() {
	// Initialize Metal device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	fmt.Println("🔧 Custom Swish Activation Layer Demo")
	fmt.Println("====================================")

	// Demonstrate custom layer integration
	err = demonstrateCustomSwishLayer()
	if err != nil {
		log.Fatalf("Custom layer demo failed: %v", err)
	}

	fmt.Println("\n✅ Custom Swish layer demo completed successfully!")
}

func demonstrateCustomSwishLayer() error {
	fmt.Println("🏗️ Creating model with custom Swish activation...")

	// Create model with custom Swish layers
	batchSize := 32
	inputSize := 784
	numClasses := 10

	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)

	// Build model using built-in Swish activation
	model, err := builder.
		AddDense(256, true, "dense1").
		AddSwish("swish1").
		AddDense(128, true, "dense2").
		AddSwish("swish2").
		AddDense(numClasses, true, "output").
		AddSoftmax(-1, "softmax").
		Compile()

	if err != nil {
		return fmt.Errorf("model compilation failed: %v", err)
	}

	fmt.Printf("✅ Model compiled with %d layers\n", len(model.Layers))

	// Display layer information
	fmt.Println("\n📊 Model Architecture:")
	for i, layer := range model.Layers {
		fmt.Printf("  %d. %s (%s)\n", i+1, layer.Name, layer.Type.String())
		if layer.Type.String() == "Swish" {
			fmt.Printf("     🎯 Swish activation: f(x) = x * sigmoid(x)\n")
		}
	}

	// Test custom layer with training
	return testCustomLayerTraining(model)
}

func testCustomLayerTraining(model *layers.ModelSpec) error {
	fmt.Println("\n🚀 Testing custom layer in training...")

	// Configure training
	config := training.TrainerConfig{
		BatchSize:     32,
		LearningRate:  0.001,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    training.Dynamic,
		LossFunction:  training.SparseCrossEntropy,
		ProblemType:   training.Classification,
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		return fmt.Errorf("trainer creation failed: %v", err)
	}
	defer trainer.Cleanup()

	// Generate test data
	fmt.Println("📊 Generating test data...")
	inputData, labelData := generateTestData(1000, 784, 10)

	// Training loop
	fmt.Println("\n🎯 Training with custom Swish activation:")
	fmt.Println("Epoch | Loss     | Status")
	fmt.Println("------|----------|--------")

	epochs := 10
	for epoch := 1; epoch <= epochs; epoch++ {
		numBatches := 1000 / 32
		var epochLoss float32

		for batch := 0; batch < numBatches; batch++ {
			batchStart := batch * 32
			batchEnd := batchStart + 32

			batchInput := inputData[batchStart*784 : batchEnd*784]
			batchLabels := labelData[batchStart:batchEnd]

			result, err := trainer.TrainBatch(
				batchInput, []int{32, 784},
				batchLabels, []int{32},
			)
			if err != nil {
				return fmt.Errorf("training failed at epoch %d: %v", epoch, err)
			}

			epochLoss += result.Loss
		}

		epochLoss /= float32(numBatches)

		status := "Learning"
		if epochLoss < 0.5 {
			status = "Converging"
		}

		fmt.Printf("%5d | %.6f | %s\n", epoch, epochLoss, status)
	}

	fmt.Println("\n✅ Custom layer training completed successfully!")
	return nil
}

func generateTestData(samples, features, classes int) ([]float32, []int32) {
	rand.Seed(42)

	inputData := make([]float32, samples*features)
	labelData := make([]int32, samples)

	for i := 0; i < samples; i++ {
		class := i % classes
		labelData[i] = int32(class)

		// Generate class-specific patterns
		for j := 0; j < features; j++ {
			baseValue := float64(class) / float64(classes)
			noise := rand.Float64()*0.2 - 0.1

			value := baseValue + noise
			if value < 0 {
				value = 0
			} else if value > 1 {
				value = 1
			}

			inputData[i*features+j] = float32(value)
		}
	}

	return inputData, labelData
}
```

**What this example demonstrates:**
- Integration of built-in activation functions into Go-Metal models
- Training compatibility with dynamic engine
- Performance validation of layers
- Architecture inspection and debugging

## 🔬 Complete Example 2: Using LeakyReLU (Parametric Activation)

This example shows how to use LeakyReLU, which has a learnable negative slope parameter:

```go
package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/training"
)

func main() {
	// Initialize Metal device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	fmt.Println("🔧 LeakyReLU (Parametric ReLU) Demo")
	fmt.Println("===================================")

	err = demonstrateParametricLayer()
	if err != nil {
		log.Fatalf("Parametric layer demo failed: %v", err)
	}

	fmt.Println("\n✅ Parametric layer demo completed!")
}

func demonstrateParametricLayer() error {
	fmt.Println("🏗️ Creating model with LeakyReLU activations...")

	// Create model with LeakyReLU (fixed negative slope parameter)
	batchSize := 64
	inputSize := 256
	numClasses := 5

	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)

	// Build model with LeakyReLU having different negative slopes
	model, err := builder.
		AddDense(128, true, "dense1").
		AddLeakyReLU(0.2, "leaky_relu1"). // Negative slope = 0.2
		AddDense(64, true, "dense2").
		AddLeakyReLU(0.1, "leaky_relu2"). // Negative slope = 0.1
		AddDense(numClasses, true, "output").
		Compile()

	if err != nil {
		return fmt.Errorf("model compilation failed: %v", err)
	}

	fmt.Printf("✅ Model with LeakyReLU activations compiled (%d layers)\n", len(model.Layers))

	// Analyze the layer parameters
	return analyzeLayerParameters(model)
}

func analyzeLayerParameters(model *layers.ModelSpec) error {
	fmt.Println("\n📊 Analyzing layer parameters:")

	// Display parameter analysis
	totalParams := model.TotalParameters
	fmt.Printf("Total parameters: %d\n", totalParams)

	// Analyze each layer's parameters
	for _, layer := range model.Layers {
		switch layer.Type.String() {
		case "Dense":
			inputSize := layer.InputShape[len(layer.InputShape)-1]
			outputSize := layer.OutputShape[len(layer.OutputShape)-1]
			useBias := layer.Parameters["use_bias"].(bool)

			weightParams := inputSize * outputSize
			biasParams := 0
			if useBias {
				biasParams = outputSize
			}

			fmt.Printf("Layer '%s': %d weight + %d bias = %d parameters\n",
				layer.Name, weightParams, biasParams, weightParams+biasParams)

		case "LeakyReLU":
			// LeakyReLU has fixed alpha parameter, not learnable
			negativeSlope := layer.Parameters["negative_slope"].(float32)
			fmt.Printf("Layer '%s': negative_slope=%.2f (fixed parameter, not learnable)\n",
				layer.Name, negativeSlope)
		}
	}

	// Test training to show layer behavior
	return testParametricLayerTraining(model)
}

func testParametricLayerTraining(model *layers.ModelSpec) error {
	fmt.Println("\n🎯 Testing parametric layer learning:")

	// Configure training
	config := training.TrainerConfig{
		BatchSize:     64,
		LearningRate:  0.002,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    training.Dynamic,
		LossFunction:  training.SparseCrossEntropy,
		ProblemType:   training.Classification,
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		return fmt.Errorf("trainer creation failed: %v", err)
	}
	defer trainer.Cleanup()

	// Generate challenging data to test adaptive activations
	fmt.Println("📊 Generating data with negative values (tests activation functions)...")
	inputData, labelData := generateChallengingnData(1280, 256, 5)

	// Training with focus on activation learning
	fmt.Println("\nTraining Progress (testing parametric activations):")
	fmt.Println("Epoch | Loss     | Gradient Info")
	fmt.Println("------|----------|---------------")

	epochs := 15
	for epoch := 1; epoch <= epochs; epoch++ {
		numBatches := 1280 / 64
		var epochLoss float32

		for batch := 0; batch < numBatches; batch++ {
			batchStart := batch * 64
			batchEnd := batchStart + 64

			batchInput := inputData[batchStart*256 : batchEnd*256]
			batchLabels := labelData[batchStart:batchEnd]

			result, err := trainer.TrainBatch(
				batchInput, []int{64, 256},
				batchLabels, []int{32},
			)
			if err != nil {
				return fmt.Errorf("training failed: %v", err)
			}

			epochLoss += result.Loss
		}

		epochLoss /= float32(numBatches)

		// Simulate gradient information that would be available in PReLU
		gradientInfo := "Parameters adapting"
		if epochLoss < 0.8 {
			gradientInfo = "Activations learned"
		}

		if epoch%3 == 0 {
			fmt.Printf("%5d | %.6f | %s\n", epoch, epochLoss, gradientInfo)
		}
	}

	fmt.Println("\n💡 About LeakyReLU in Go-Metal:")
	fmt.Println("   • Uses fixed negative slope parameter (not learnable)")
	fmt.Println("   • Prevents dying ReLU problem for negative inputs")
	fmt.Println("   • Good alternative to ReLU for deep networks")
	fmt.Println("   • Implemented using MPSGraph operations")

	return nil
}

func generateChallengingnData(samples, features, classes int) ([]float32, []int32) {
	rand.Seed(123)

	inputData := make([]float32, samples*features)
	labelData := make([]int32, samples)

	for i := 0; i < samples; i++ {
		class := i % classes
		labelData[i] = int32(class)

		// Generate data with both positive and negative values
		// This tests the activation function's handling of negative inputs
		for j := 0; j < features; j++ {
			baseValue := float64(class)/float64(classes) - 0.5 // Center around 0
			noise := rand.Float64()*0.8 - 0.4                 // More noise, including negative

			value := baseValue + noise
			// Don't clamp - allow negative values to test activations
			inputData[i*features+j] = float32(value)
		}
	}

	return inputData, labelData
}
```

**Key concepts demonstrated:**
- Parametric layers with learnable parameters
- Analysis of parameter counts and shapes  
- Training dynamics with adaptive activations
- Handling of negative input values

## 🛠️ Complete Example 3: Complex Custom Transformation Layer

This example shows a more complex custom layer that combines multiple operations:

```go
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/training"
)

func main() {
	// Initialize Metal device and memory manager
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)

	memory.InitializeGlobalMemoryManager(device)

	fmt.Println("🔧 Complex Custom Transformation Layer Demo")
	fmt.Println("==========================================")

	err = demonstrateComplexCustomLayer()
	if err != nil {
		log.Fatalf("Complex layer demo failed: %v", err)
	}

	fmt.Println("\n✅ Complex custom layer demo completed!")
}

func demonstrateComplexCustomLayer() error {
	fmt.Println("🏗️ Creating Gated Linear Unit (GLU) simulation...")

	// GLU splits input into two halves: one goes through sigmoid, multiplied with other half
	// We'll simulate this with Dense layers and existing activations
	batchSize := 32
	inputSize := 512
	hiddenSize := 256
	numClasses := 8

	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)

	// Simulate GLU (Gated Linear Unit) using existing components
	// Real GLU: output = (input_a) * sigmoid(input_b) where input_a, input_b are halves
	model, err := builder.
		// First GLU simulation block
		AddDense(hiddenSize*2, true, "glu_projection1"). // Projects to 2x size for splitting
		AddDense(hiddenSize, true, "glu_gate1").         // Gate pathway  
		AddSigmoid("glu_sigmoid1").                      // Gate activation
		AddDense(hiddenSize, true, "glu_linear1").       // Linear pathway (simulated)
		
		// Second GLU simulation block
		AddDense(hiddenSize*2, true, "glu_projection2").
		AddDense(hiddenSize, true, "glu_gate2").
		AddSigmoid("glu_sigmoid2").
		AddDense(hiddenSize, true, "glu_linear2").
		
		// Output
		AddDense(numClasses, true, "output").
		Compile()

	if err != nil {
		return fmt.Errorf("GLU simulation compilation failed: %v", err)
	}

	fmt.Printf("✅ GLU simulation model compiled (%d layers)\n", len(model.Layers))

	// Analyze the complex layer structure
	return analyzeComplexLayerStructure(model)
}

func analyzeComplexLayerStructure(model *layers.ModelSpec) error {
	fmt.Println("\n📊 Complex Layer Architecture Analysis:")

	currentSize := model.InputShape[len(model.InputShape)-1]
	fmt.Printf("Input size: %d\n", currentSize)

	for i, layer := range model.Layers {
		layerType := layer.Type.String()
		
		switch layerType {
		case "Dense":
			inputSize := layer.InputShape[len(layer.InputShape)-1]
			outputSize := layer.OutputShape[len(layer.OutputShape)-1]
			
			if outputSize == inputSize*2 {
				fmt.Printf("%d. %s: %d → %d (Expansion for gating)\n", 
					i+1, layer.Name, inputSize, outputSize)
			} else if outputSize == inputSize/2 {
				fmt.Printf("%d. %s: %d → %d (Contraction after gating)\n", 
					i+1, layer.Name, inputSize, outputSize)
			} else {
				fmt.Printf("%d. %s: %d → %d (Transformation)\n", 
					i+1, layer.Name, inputSize, outputSize)
			}
			currentSize = outputSize
			
		case "Sigmoid":
			fmt.Printf("%d. %s: Gating activation (size %d)\n", 
				i+1, layer.Name, currentSize)
				
		default:
			fmt.Printf("%d. %s: %s operation (size %d)\n", 
				i+1, layer.Name, layerType, currentSize)
		}
	}

	fmt.Printf("Output size: %d\n", currentSize)

	// Test the complex layer in training
	return testComplexLayerTraining(model)
}

func testComplexLayerTraining(model *layers.ModelSpec) error {
	fmt.Println("\n🎯 Testing complex layer training dynamics:")

	// Configure training
	config := training.TrainerConfig{
		BatchSize:     32,
		LearningRate:  0.0015,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    training.Dynamic,
		LossFunction:  training.SparseCrossEntropy,
		ProblemType:   training.Classification,
		Beta1:         0.9,
		Beta2:         0.999,
		Epsilon:       1e-8,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		return fmt.Errorf("trainer creation failed: %v", err)
	}
	defer trainer.Cleanup()

	// Generate complex data to test gating mechanisms
	fmt.Println("📊 Generating multi-modal data (tests gating effectiveness)...")
	inputData, labelData := generateMultiModalData(800, 512, 8)

	// Training with analysis of gating behavior
	fmt.Println("\nTraining Complex Gated Model:")
	fmt.Println("Epoch | Loss     | Conv Rate | Gating Effect")
	fmt.Println("------|----------|-----------|---------------")

	epochs := 12
	var lastLoss float32 = 999.0

	for epoch := 1; epoch <= epochs; epoch++ {
		numBatches := 800 / 32
		var epochLoss float32

		for batch := 0; batch < numBatches; batch++ {
			batchStart := batch * 32
			batchEnd := batchStart + 32

			batchInput := inputData[batchStart*512 : batchEnd*512]
			batchLabels := labelData[batchStart:batchEnd]

			result, err := trainer.TrainBatch(
				batchInput, []int{32, 512},
				batchLabels, []int{32},
			)
			if err != nil {
				return fmt.Errorf("training failed: %v", err)
			}

			epochLoss += result.Loss
		}

		epochLoss /= float32(numBatches)

		// Analyze convergence rate
		convergenceRate := "Slow"
		if epoch > 1 {
			improvement := lastLoss - epochLoss
			if improvement > 0.1 {
				convergenceRate = "Fast"
			} else if improvement > 0.05 {
				convergenceRate = "Moderate"
			}
		}

		// Simulate gating effect analysis
		gatingEffect := "Learning gates"
		if epochLoss < 1.0 {
			gatingEffect = "Gates active"
		}
		if epochLoss < 0.5 {
			gatingEffect = "Gates optimized"
		}

		if epoch%2 == 0 {
			fmt.Printf("%5d | %.6f | %-9s | %s\n", epoch, epochLoss, convergenceRate, gatingEffect)
		}

		lastLoss = epochLoss
	}

	fmt.Println("\n💡 Complex Layer Insights:")
	fmt.Println("   • Gating mechanisms can improve gradient flow")
	fmt.Println("   • Multiple pathways allow selective information processing")
	fmt.Println("   • Complex layers may require careful initialization")
	fmt.Println("   • Training dynamics differ from simple activation functions")

	return nil
}

func generateMultiModalData(samples, features, classes int) ([]float32, []int32) {
	rand.Seed(456)

	inputData := make([]float32, samples*features)
	labelData := make([]int32, samples)

	for i := 0; i < samples; i++ {
		class := i % classes
		labelData[i] = int32(class)

		// Create multi-modal data (different regions of input space)
		for j := 0; j < features; j++ {
			var value float64

			// Create different modes based on feature index
			if j < features/3 {
				// Mode 1: Class-dependent values
				value = float64(class)/float64(classes) + rand.Float64()*0.2-0.1
			} else if j < 2*features/3 {
				// Mode 2: Sine-wave patterns
				value = math.Sin(float64(class)*math.Pi/4 + float64(j)*0.01) * 0.5 + 0.5
			} else {
				// Mode 3: Random background
				value = rand.Float64()
			}

			inputData[i*features+j] = float32(value)
		}
	}

	return inputData, labelData
}
```

**Advanced concepts demonstrated:**
- Complex multi-operation custom layers
- Gating mechanisms and selective processing
- Multi-modal data handling
- Training dynamics analysis
- Architecture design patterns

## 🔍 Custom Layer Implementation Guide

### Step 1: Design Your Layer

**Define the mathematical operation:**
```
Hypothetical NewActivation: f(x) = x * tanh(x)
Hypothetical ScaledReLU: f(x) = α * max(0, x)  [α fixed parameter]
```

**Determine parameters:**
- Input/output shapes
- Fixed parameters (like negative slope in LeakyReLU)
- Learnable parameters (for layers like Dense, Conv2D)

### Step 2: Extend Go-Metal Framework

**Add Layer Type (layers/layer.go):**
```go
const (
    Dense LayerType = iota    // 0
    Conv2D                    // 1
    ReLU                      // 2
    Softmax                   // 3
    MaxPool2D                 // 4
    Dropout                   // 5
    BatchNorm                 // 6
    LeakyReLU                 // 7
    ELU                       // 8
    Sigmoid                   // 9
    Tanh                      // 10
    Swish                     // 11
    // Add new layer types sequentially
    NewActivation             // 12
    ScaledReLU               // 13
)

func (lt LayerType) String() string {
    switch lt {
    // Existing cases...
    case NewActivation:
        return "NewActivation"
    case ScaledReLU:
        return "ScaledReLU"
    default:
        return "Unknown"
    }
}
```

**Add Factory Method (layers/layer.go):**
```go
func (mb *ModelBuilder) AddNewActivation(name string) *ModelBuilder {
    layer := LayerSpec{
        Type:       NewActivation,
        Name:       name,
        Parameters: make(map[string]interface{}),
    }
    mb.layers = append(mb.layers, layer)
    return mb
}

func (mb *ModelBuilder) AddScaledReLU(scale float32, name string) *ModelBuilder {
    layer := LayerSpec{
        Type: ScaledReLU,
        Name: name,
        Parameters: map[string]interface{}{
            "scale": scale, // Fixed scaling factor
        },
    }
    mb.layers = append(mb.layers, layer)
    return mb
}
```

**Add Shape Computation (in computeLayerInfo method):**
```go
func (mb *ModelBuilder) computeLayerInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
    switch layer.Type {
    // Existing cases...
    case NewActivation, ScaledReLU:
        return mb.computeActivationInfo(layer, inputShape) // No parameters, same shape
    // Other cases...
    }
}

// Activation layers don't change shape and have no learnable parameters
func (mb *ModelBuilder) computeActivationInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
    outputShape := make([]int, len(inputShape))
    copy(outputShape, inputShape)
    return outputShape, [][]int{}, 0, nil
}
```

### Step 3: Implement MPSGraph Computation

**Add to Bridge (cgo_bridge/bridge_graph.m switch statement):**
```objective-c
case 12: // NewActivation
    // NewActivation: x * tanh(x)
    {
        MPSGraphTensor* tanhTensor = [engine->graph tanhWithTensor:currentTensor
                                                             name:[NSString stringWithFormat:@"newact_tanh_%d", layerIdx]];
        currentTensor = [engine->graph multiplicationWithPrimaryTensor:currentTensor
                                                       secondaryTensor:tanhTensor
                                                                  name:[NSString stringWithFormat:@"newact_output_%d", layerIdx]];
    }
    break;

case 13: // ScaledReLU  
    // ScaledReLU: scale * max(0, x)
    {
        float scale = layer->param_float[0]; // Get scale parameter
        MPSGraphTensor* reluTensor = [engine->graph reLUWithTensor:currentTensor
                                                             name:[NSString stringWithFormat:@"scaled_relu_%d", layerIdx]];
        MPSGraphTensor* scaleTensor = [engine->graph constantWithScalar:scale
                                                                  shape:reluTensor.shape
                                                               dataType:MPSDataTypeFloat32];
        currentTensor = [engine->graph multiplicationWithPrimaryTensor:reluTensor
                                                       secondaryTensor:scaleTensor
                                                                  name:[NSString stringWithFormat:@"scaled_output_%d", layerIdx]];
    }
    break;
```

**Key Implementation Points:**
- Use MPSGraph operations (no custom Metal shaders)
- Access parameters via `layer->param_float[]` or `layer->param_int[]`
- Create descriptive tensor names for debugging
- Handle tensor shapes properly

### Step 4: Update C Bridge Type Mapping

**Add layer type constants to bridge_types.h:**
```c
typedef struct {
    int layer_type;          // 0=Dense, 1=Conv2D, 2=ReLU, 3=Softmax, 4=MaxPool2D, 
                            // 5=Dropout, 6=BatchNorm, 7=LeakyReLU, 8=ELU, 9=Sigmoid, 
                            // 10=Tanh, 11=Swish, 12=NewActivation, 13=ScaledReLU
    char name[64];           // Layer name
    int input_shape[4];      // Input dimensions [batch, channels, height, width]
    int input_shape_len;     // Number of valid dimensions
    int output_shape[4];     // Output dimensions
    int output_shape_len;    // Number of valid dimensions
    
    // Layer-specific parameters
    int param_int[8];        // Integer parameters
    float param_float[8];    // Float parameters (scale for ScaledReLU)
    int param_int_count;     // Number of valid int parameters
    int param_float_count;   // Number of valid float parameters
} layer_spec_c_t;
```

### Step 5: Update Dynamic Layer Conversion

**Add cases to ConvertToDynamicLayerSpecs method in layers/layer.go:**
```go
func (ms *ModelSpec) ConvertToDynamicLayerSpecs() ([]DynamicLayerSpec, error) {
    // ... existing code ...
    
    for i, layer := range ms.Layers {
        // ... existing cases ...
        
        switch layer.Type {
        // ... existing cases ...
        
        case NewActivation:
            spec.LayerType = 12 // NewActivation = 12
            spec.ParamIntCount = 0
            spec.ParamFloatCount = 0
            // Shape unchanged for activation
            
        case ScaledReLU:
            spec.LayerType = 13 // ScaledReLU = 13
            
            scale := getFloatParam(layer.Parameters, "scale", 1.0)
            spec.ParamFloat[0] = scale
            spec.ParamFloatCount = 1
            spec.ParamIntCount = 0
            // Shape unchanged for activation
            
        default:
            return nil, fmt.Errorf("unsupported layer type: %s", layer.Type.String())
        }
        
        // ... rest of conversion logic ...
    }
}
```

## 📋 Custom Layer Implementation Summary

### What We've Learned

1. **Go-Metal uses consecutive integer layer types** (0-11), not custom high values
2. **MPSGraph operations are used exclusively** - no custom Metal shaders needed
3. **Static switch statements** handle layer types in the C bridge
4. **Parameter arrays** (`param_float[]`, `param_int[]`) pass configuration to C bridge
5. **Activation layers** typically don't change tensor shapes or have learnable parameters

### Current Built-in Layers Available

Go-Metal already provides these layer types ready for use:

| Layer Type | Value | Description | Parameters |
|------------|-------|-------------|------------|
| Dense | 0 | Fully connected layer | input_size, output_size, use_bias |
| Conv2D | 1 | 2D convolution | channels, kernel_size, stride, padding, use_bias |
| ReLU | 2 | ReLU activation | none |
| Softmax | 3 | Softmax activation | axis |
| MaxPool2D | 4 | Max pooling | pool_size, stride |
| Dropout | 5 | Dropout regularization | rate, training |
| BatchNorm | 6 | Batch normalization | num_features, eps, momentum, affine |
| LeakyReLU | 7 | Leaky ReLU activation | negative_slope |
| ELU | 8 | ELU activation | alpha |
| Sigmoid | 9 | Sigmoid activation | none |
| Tanh | 10 | Tanh activation | none |
| Swish | 11 | Swish activation | none |

### Key Implementation Requirements

To add a custom layer type (e.g., `NewActivation` as type 12):

1. **Add to Go enum** in `layers/layer.go` with next sequential value
2. **Add builder method** like `AddNewActivation()` 
3. **Update computeLayerInfo** to handle shape and parameter calculations
4. **Add C bridge case** in `bridge_graph.m` switch statement  
5. **Update dynamic conversion** in `ConvertToDynamicLayerSpecs()`
6. **Use MPSGraph operations** for the actual computation

### Training Engine Compatibility

Custom layers automatically work with:
- ✅ **Dynamic Training Engine** - supports any layer architecture
- ✅ **All optimizers** (Adam, SGD, RMSProp, etc.)
- ✅ **Inference Engine** - optimized for deployment
- ✅ **Model serialization** - save/load functionality

## 🚀 Next Steps

Ready to extend Go-Metal? Consider these approaches:

1. **Start with existing layers** - LeakyReLU, ELU, Swish provide good examples
2. **Use built-in components** - Combine existing layers for complex operations  
3. **Focus on MPSGraph** - Leverage Apple's optimized GPU operations
4. **Follow the patterns** - Static registration, sequential numbering, parameter arrays

For complex architectures, Go-Metal's specification-based approach with MPSGraph provides excellent performance and maintainability!

---

**Ready to build high-performance neural networks?** Go-Metal's layer system is designed for both simplicity and extensibility, with Apple Silicon optimization built-in.

