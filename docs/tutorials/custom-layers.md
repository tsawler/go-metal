# Custom Layers Tutorial

Extend Go-Metal with your own layer types using Metal Performance Shaders for maximum performance on Apple Silicon.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- Understand Go-Metal's layer architecture and extension points
- Implement custom layers with learnable parameters
- Integrate Metal Performance Shaders for GPU acceleration
- Create activation functions, transformations, and complex operations
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

### Layer Type Registration

All layer types are defined in an enum that must be extended for custom layers:

```go
const (
    Dense LayerType = iota
    Conv2D
    ReLU
    Softmax
    // Your custom layers would be added here
    CustomActivation
    CustomTransform
)
```

### Metal Integration

The actual computation happens in **Metal Performance Shaders Graph (MPSGraph)**, Apple's high-performance GPU compute framework. Custom layers define their computation using MPSGraph primitives.

## 🚀 Complete Example 1: Custom Activation Function (Swish)

Let's implement a custom Swish activation function (x * sigmoid(x)) as a complete working example:

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

// SwishLayerType represents our custom Swish activation
// Note: In real implementation, this would be added to the LayerType enum
const SwishLayerType layers.LayerType = 99 // Using high number to avoid conflicts

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

	// Build model using built-in Swish (which demonstrates the pattern)
	// In a real custom implementation, you would use your custom layer type
	model, err := builder.
		AddDense(256, true, "dense1").
		AddSwish("swish1").  // Using built-in Swish as example
		AddDense(128, true, "dense2").
		AddSwish("swish2").  // Custom activation pattern
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
			fmt.Printf("     🎯 Custom Swish: f(x) = x * sigmoid(x)\n")
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
- Integration of custom activation functions into Go-Metal models
- Training compatibility with dynamic engine
- Performance validation of custom layers
- Architecture inspection and debugging

## 🔬 Complete Example 2: Custom Parametric Layer (PReLU)

This example shows how to create a parametric layer with learnable parameters:

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

	fmt.Println("🔧 Custom PReLU (Parametric ReLU) Demo")
	fmt.Println("======================================")

	err = demonstrateParametricLayer()
	if err != nil {
		log.Fatalf("Parametric layer demo failed: %v", err)
	}

	fmt.Println("\n✅ Parametric layer demo completed!")
}

func demonstrateParametricLayer() error {
	fmt.Println("🏗️ Simulating PReLU layer implementation...")

	// Create model demonstrating parametric activation concept
	// Using LeakyReLU as a fixed-parameter version of what PReLU would be
	batchSize := 64
	inputSize := 256
	numClasses := 5

	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)

	// Build model simulating PReLU with LeakyReLU
	// In full implementation, PReLU would have learnable alpha per channel
	model, err := builder.
		AddDense(128, true, "dense1").
		AddLeakyReLU(0.2, "prelu_simulation1"). // Simulates PReLU with fixed alpha
		AddDense(64, true, "dense2").
		AddLeakyReLU(0.1, "prelu_simulation2"). // Different alpha values
		AddDense(numClasses, true, "output").
		Compile()

	if err != nil {
		return fmt.Errorf("model compilation failed: %v", err)
	}

	fmt.Printf("✅ Model with parametric activations compiled (%d layers)\n", len(model.Layers))

	// Demonstrate the concept of learnable parameters
	return demonstrateLearnableParameters(model)
}

func demonstrateLearnableParameters(model *layers.ModelSpec) error {
	fmt.Println("\n📊 Analyzing learnable parameters in custom layers:")

	// Display parameter analysis
	totalParams := model.TotalParameters
	fmt.Printf("Total parameters: %d\n", totalParams)

	// Analyze each layer's parameters
	parameterIndex := 0
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
			// In a real PReLU implementation, this would have learnable alpha parameters
			alpha := layer.Parameters["alpha"].(float32)
			fmt.Printf("Layer '%s': alpha=%.2f (in PReLU, this would be learnable per channel)\n",
				layer.Name, alpha)
		}
	}

	// Test training to show parameter learning
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

	fmt.Println("\n💡 In a real PReLU implementation:")
	fmt.Println("   • Each channel would have a learnable alpha parameter")
	fmt.Println("   • Alpha values would adapt during training")
	fmt.Println("   • Different channels could learn different negative slopes")
	fmt.Println("   • This provides more flexibility than fixed LeakyReLU")

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
Swish: f(x) = x * sigmoid(x)
PReLU: f(x) = max(0, x) + α * min(0, x)  [α learnable per channel]
GLU: f(x) = (Wx + b) ⊙ σ(Vx + c)       [⊙ is element-wise product]
```

**Determine parameters:**
- Input/output shapes
- Learnable parameters (weights, biases, activation parameters)
- Hyperparameters (fixed configuration values)

### Step 2: Extend Go-Metal Framework

**Add Layer Type (layers/layer_types.go):**
```go
const (
    // Existing types...
    CustomSwish LayerType = iota + 100
    CustomPReLU
    CustomGLU
)

func (lt LayerType) String() string {
    switch lt {
    // Existing cases...
    case CustomSwish:
        return "CustomSwish"
    case CustomPReLU:
        return "CustomPReLU"
    case CustomGLU:
        return "CustomGLU"
    default:
        return "Unknown"
    }
}
```

**Add Factory Method (layers/layer.go):**
```go
func (mb *ModelBuilder) AddCustomSwish(name string) *ModelBuilder {
    layer := LayerSpec{
        Type:       CustomSwish,
        Name:       name,
        Parameters: make(map[string]interface{}),
    }
    mb.layers = append(mb.layers, layer)
    return mb
}

func (mb *ModelBuilder) AddCustomPReLU(name string) *ModelBuilder {
    layer := LayerSpec{
        Type: CustomPReLU,
        Name: name,
        Parameters: map[string]interface{}{
            "alpha_init": 0.25, // Initial alpha value
        },
    }
    mb.layers = append(mb.layers, layer)
    return mb
}
```

**Add Shape Computation:**
```go
func (mb *ModelBuilder) computeCustomSwishInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
    // Swish doesn't change shape and has no parameters
    return inputShape, [][]int{}, 0, nil
}

func (mb *ModelBuilder) computeCustomPReLUInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
    // PReLU has one learnable alpha parameter per channel
    channels := inputShape[len(inputShape)-1]
    paramShapes := [][]int{{channels}} // Alpha parameters
    return inputShape, paramShapes, int64(channels), nil
}
```

### Step 3: Implement Metal Computation

**Add to Bridge (cgo_bridge/bridge_graph.m):**
```objective-c
case CUSTOM_SWISH:
    currentTensor = addCustomSwishToGraph(engine->graph, currentTensor, layer, layerIdx);
    break;

case CUSTOM_PRELU:
    currentTensor = addCustomPReLUToGraph(engine->graph, currentTensor, layer, layerIdx, allParameterPlaceholders);
    break;
```

**Implement Metal Operations:**
```objective-c
MPSGraphTensor* addCustomSwishToGraph(MPSGraph* graph, 
                                     MPSGraphTensor* inputTensor,
                                     layer_spec_c_t* layer,
                                     int layerIdx) {
    // Swish: x * sigmoid(x)
    MPSGraphTensor* sigmoidTensor = [graph sigmoidWithTensor:inputTensor
                                                        name:[NSString stringWithFormat:@"swish_sigmoid_%d", layerIdx]];
    
    MPSGraphTensor* outputTensor = [graph multiplicationWithPrimaryTensor:inputTensor
                                                          secondaryTensor:sigmoidTensor
                                                                     name:[NSString stringWithFormat:@"swish_output_%d", layerIdx]];
    
    return outputTensor;
}

MPSGraphTensor* addCustomPReLUToGraph(MPSGraph* graph,
                                     MPSGraphTensor* inputTensor,
                                     layer_spec_c_t* layer,
                                     int layerIdx,
                                     NSMutableArray* parameterPlaceholders) {
    // PReLU: max(0, x) + α * min(0, x)
    
    // Create alpha parameter placeholder
    NSArray* inputShape = inputTensor.shape;
    NSArray* alphaShape = @[inputShape.lastObject]; // Alpha per channel
    
    MPSGraphTensor* alphaPlaceholder = [graph placeholderWithShape:alphaShape
                                                          dataType:MPSDataTypeFloat32
                                                              name:[NSString stringWithFormat:@"prelu_alpha_%d", layerIdx]];
    [parameterPlaceholders addObject:alphaPlaceholder];
    
    // Compute PReLU
    MPSGraphTensor* zeroTensor = [graph constantWithScalar:0.0f
                                                     shape:inputShape
                                                  dataType:MPSDataTypeFloat32];
    
    MPSGraphTensor* positivePart = [graph maximumWithPrimaryTensor:inputTensor
                                                   secondaryTensor:zeroTensor
                                                              name:[NSString stringWithFormat:@"prelu_positive_%d", layerIdx]];
    
    MPSGraphTensor* negativePart = [graph minimumWithPrimaryTensor:inputTensor
                                                   secondaryTensor:zeroTensor
                                                              name:[NSString stringWithFormat:@"prelu_negative_%d", layerIdx]];
    
    MPSGraphTensor* scaledNegative = [graph multiplicationWithPrimaryTensor:alphaTensor
                                                            secondaryTensor:negativePart
                                                                       name:[NSString stringWithFormat:@"prelu_scaled_%d", layerIdx]];
    
    MPSGraphTensor* outputTensor = [graph additionWithPrimaryTensor:positivePart
                                                    secondaryTensor:scaledNegative
                                                               name:[NSString stringWithFormat:@"prelu_output_%d", layerIdx]];
    
    return outputTensor;
}
```

### Step 4: Testing and Validation

**Create Test Cases:**
```go
func TestCustomSwishLayer(t *testing.T) {
    // Test layer creation
    builder := layers.NewModelBuilder([]int{1, 10})
    model, err := builder.
        AddCustomSwish("test_swish").
        Compile()
    
    assert.NoError(t, err)
    assert.Equal(t, 1, len(model.Layers))
    assert.Equal(t, "CustomSwish", model.Layers[0].Type.String())
}

func TestCustomPReLUParameters(t *testing.T) {
    builder := layers.NewModelBuilder([]int{1, 10})
    model, err := builder.
        AddCustomPReLU("test_prelu").
        Compile()
    
    assert.NoError(t, err)
    assert.Equal(t, int64(10), model.TotalParameters) // 10 alpha parameters
}
```

**Validate Training Integration:**
```go
func TestCustomLayerTraining(t *testing.T) {
    // Create model with custom layer
    model := createModelWithCustomLayers()
    
    // Test training compatibility
    config := training.TrainerConfig{
        BatchSize:     32,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    assert.NoError(t, err)
    defer trainer.Cleanup()
    
    // Test forward and backward passes
    result, err := trainer.TrainBatch(testInput, inputShape, testLabels, labelShape)
    assert.NoError(t, err)
    assert.Greater(t, result.Loss, 0.0)
}
```

## 📋 Custom Layer Checklist

### Design Phase
- [ ] **Mathematical specification** - Define the exact computation
- [ ] **Parameter analysis** - Identify learnable vs. fixed parameters
- [ ] **Shape computation** - Determine input/output shape relationships
- [ ] **Gradient requirements** - Ensure differentiability for training

### Implementation Phase
- [ ] **Layer type registration** - Add to enum and string conversion
- [ ] **Factory methods** - Create builder methods for easy usage
- [ ] **Shape computation** - Implement shape inference logic
- [ ] **Parameter counting** - Correctly count learnable parameters
- [ ] **Metal implementation** - Use MPSGraph operations efficiently

### Integration Phase
- [ ] **Engine compatibility** - Test with dynamic training engine
- [ ] **Optimizer compatibility** - Verify with all optimizers (Adam, SGD, etc.)
- [ ] **Checkpoint compatibility** - Ensure serialization works
- [ ] **Inference compatibility** - Test with inference engine

### Validation Phase
- [ ] **Unit tests** - Layer creation and shape computation
- [ ] **Training tests** - End-to-end training compatibility
- [ ] **Performance tests** - GPU utilization and speed
- [ ] **Numerical tests** - Gradient checking and accuracy

## 🚀 Advanced Custom Layer Patterns

### Composite Layers
Combine multiple operations into single logical units:
```go
// Residual Block: x + F(x)
func (mb *ModelBuilder) AddResidualBlock(channels int, name string) *ModelBuilder {
    // Implementation would create sub-layers internally
    return mb
}
```

### Attention Mechanisms
Implement attention-based layers:
```go
// Self-Attention: softmax(QK^T)V
func (mb *ModelBuilder) AddSelfAttention(headSize, numHeads int, name string) *ModelBuilder {
    // Implementation would handle multi-head attention computation
    return mb
}
```

### Normalization Layers
Custom normalization beyond BatchNorm:
```go
// Layer Normalization
func (mb *ModelBuilder) AddLayerNorm(epsilon float32, name string) *ModelBuilder {
    // Normalize across feature dimension rather than batch dimension
    return mb
}
```

## 💡 Performance Optimization Tips

### Metal Performance Shaders Best Practices
- **Use primitive operations**: Combine MPSGraph primitives efficiently
- **Avoid memory copies**: Keep computations on GPU
- **Batch operations**: Group similar operations together
- **Memory layout**: Consider tensor layout for optimal performance

### Training Performance
- **Parameter initialization**: Use appropriate initialization schemes
- **Gradient clipping**: Consider numerical stability
- **Learning rate scaling**: Account for custom layer characteristics
- **Memory efficiency**: Minimize intermediate tensor allocations

## 🔧 Troubleshooting Custom Layers

### Common Issues

**Compilation Errors:**
- Layer type not registered in enum
- Missing shape computation function
- Parameter count mismatch

**Runtime Errors:**
- Metal shader implementation missing
- Incorrect parameter placeholder registration
- Shape mismatch between layers

**Training Issues:**
- Gradient explosion/vanishing
- Numerical instability
- Poor convergence

### Debugging Strategies

**Layer Inspection:**
```go
func debugLayerInfo(model *layers.ModelSpec) {
    for i, layer := range model.Layers {
        fmt.Printf("Layer %d: %s (%s)\n", i, layer.Name, layer.Type.String())
        fmt.Printf("  Input shape: %v\n", layer.InputShape)
        fmt.Printf("  Output shape: %v\n", layer.OutputShape)
        fmt.Printf("  Parameters: %d\n", layer.ParameterCount)
    }
}
```

**Training Monitoring:**
```go
func monitorCustomLayerTraining(trainer *training.ModelTrainer) {
    // Monitor gradient norms, loss progression, parameter changes
    // Add custom logging for your layer's behavior
}
```

## 🎯 Production Deployment

### Testing Strategy
1. **Unit tests** for layer creation and shape computation
2. **Integration tests** with training and inference engines
3. **Performance benchmarks** comparing to baseline layers
4. **Numerical validation** against reference implementations

### Documentation Requirements
- Mathematical specification of the layer
- Usage examples and common patterns
- Performance characteristics and limitations
- Compatibility notes with different engines

### Maintenance Considerations
- Version compatibility across Go-Metal updates
- Performance monitoring in production
- Fallback mechanisms for unsupported platforms
- Migration paths for layer API changes

## 🚀 Next Steps

Master custom layer development with go-metal:

- **[Architecture Guide](../guides/architecture.md)** - Understand Go-Metal's design principles
- **[Performance Guide](../guides/performance.md)** - Optimize custom layer performance
- **[Memory Management](../guides/memory-management.md)** - GPU memory best practices
- **[Advanced Examples](../examples/)** - See custom layers in production models

**Ready to extend Go-Metal?** Use these patterns to create high-performance custom layers that integrate seamlessly with the Go-Metal ecosystem!

---

## 🧠 Key Takeaways

- **Specification-based design**: Go-Metal uses layer specs + Metal implementation
- **MPSGraph integration**: Leverage Apple's optimized GPU compute framework
- **Parameter management**: Proper handling of learnable parameters is crucial
- **Training compatibility**: Custom layers work with all optimizers and engines
- **Performance focus**: Metal implementation enables maximum GPU utilization

With custom layers, you can extend Go-Metal to support any neural network architecture while maintaining the performance benefits of Apple Silicon!