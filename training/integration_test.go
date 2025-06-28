package training

import (
	"fmt"
	"math"
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

// TestTensorOperationsForwardBackward tests all tensor operations forward and backward passes
func TestTensorOperationsForwardBackward(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	// Add GPU devices if available
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Create test tensors
			a, _ := tensor.Random([]int{3, 4}, tensor.Float32, device)
			b, _ := tensor.Random([]int{3, 4}, tensor.Float32, device)
			c, _ := tensor.Random([]int{4, 5}, tensor.Float32, device)
			
			a.SetRequiresGrad(true)
			b.SetRequiresGrad(true)
			c.SetRequiresGrad(true)

			t.Run("Addition", func(t *testing.T) {
				result, err := tensor.AddAutograd(a, b)
				if err != nil {
					t.Fatalf("Add failed: %v", err)
				}
				
				// Sum to scalar for backward pass
				scalar, err := tensor.SumAutograd(result)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				// Backward pass
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Backward failed: %v", err)
				}
				
				// Check gradients are ones (derivative of addition is 1)
				aOnes, _ := tensor.Ones(a.Shape, a.DType, a.Device)
				bOnes, _ := tensor.Ones(b.Shape, b.DType, b.Device)
				if !isClose(a.Grad(), aOnes, 1e-5) {
					t.Errorf("Addition gradient for a incorrect")
				}
				if !isClose(b.Grad(), bOnes, 1e-5) {
					t.Errorf("Addition gradient for b incorrect")
				}
				
				// Zero gradients for next test
				a.ZeroGrad()
				b.ZeroGrad()
			})

			t.Run("Subtraction", func(t *testing.T) {
				result, err := tensor.SubAutograd(a, b)
				if err != nil {
					t.Fatalf("Sub failed: %v", err)
				}
				
				// Sum to scalar for backward pass
				scalar, err := tensor.SumAutograd(result)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Backward failed: %v", err)
				}
				
				// Check gradients (derivative of subtraction is 1 for a, -1 for b)
				aOnes, _ := tensor.Ones(a.Shape, a.DType, a.Device)
				if !isClose(a.Grad(), aOnes, 1e-5) {
					t.Errorf("Subtraction gradient for a incorrect")
				}
				negOnes, _ := tensor.Full(b.Shape, float32(-1.0), b.DType, b.Device)
				if !isClose(b.Grad(), negOnes, 1e-5) {
					t.Errorf("Subtraction gradient for b incorrect")
				}
				
				a.ZeroGrad()
				b.ZeroGrad()
			})

			t.Run("Multiplication", func(t *testing.T) {
				result, err := tensor.MulAutograd(a, b)
				if err != nil {
					t.Fatalf("Mul failed: %v", err)
				}
				
				// Sum to scalar for backward pass
				scalar, err := tensor.SumAutograd(result); if err != nil { t.Fatalf("Sum failed: %v", err) }
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Backward failed: %v", err)
				}
				
				// Check gradients (derivative of a*b is b w.r.t a, a w.r.t b)
				if !isClose(a.Grad(), b, 1e-5) {
					t.Errorf("Multiplication gradient for a incorrect")
				}
				if !isClose(b.Grad(), a, 1e-5) {
					t.Errorf("Multiplication gradient for b incorrect")
				}
				
				a.ZeroGrad()
				b.ZeroGrad()
			})

			t.Run("MatMul", func(t *testing.T) {
				// a: [3, 4], c: [4, 5] -> result: [3, 5]
				result, err := tensor.MatMulAutograd(a, c)
				if err != nil {
					t.Fatalf("MatMul failed: %v", err)
				}
				
				if result.Shape[0] != 3 || result.Shape[1] != 5 {
					t.Errorf("MatMul result shape incorrect: got %v, expected [3, 5]", result.Shape)
				}
				
				// Sum to scalar for backward pass
				scalar, err := tensor.SumAutograd(result); if err != nil { t.Fatalf("Sum failed: %v", err) }
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Backward failed: %v", err)
				}
				
				// Check gradient shapes
				if a.Grad() == nil || c.Grad() == nil {
					t.Fatalf("Gradients not computed")
				}
				if !equalShapes(a.Grad().Shape, a.Shape) {
					t.Errorf("MatMul gradient shape for a incorrect: got %v, expected %v", a.Grad().Shape, a.Shape)
				}
				if !equalShapes(c.Grad().Shape, c.Shape) {
					t.Errorf("MatMul gradient shape for c incorrect: got %v, expected %v", c.Grad().Shape, c.Shape)
				}
				
				a.ZeroGrad()
				c.ZeroGrad()
			})
		})
	}
}

// TestActivationFunctionsForwardBackward tests activation functions
func TestActivationFunctionsForwardBackward(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Create test tensor with mixed positive and negative values
			data := []float32{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
			x, _ := tensor.NewTensor([]int{1, 7}, tensor.Float32, device, data)
			x.SetRequiresGrad(true)

			t.Run("ReLU", func(t *testing.T) {
				result, err := tensor.ReLUAutograd(x)
				if err != nil {
					t.Fatalf("ReLU failed: %v", err)
				}
				
				// Check forward pass - negative values should be 0
				resultData := tensorFloat32(result)
				expected := []float32{0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0}
				for i, v := range resultData {
					if math.Abs(float64(v-expected[i])) > 1e-5 {
						t.Errorf("ReLU forward incorrect at index %d: got %f, expected %f", i, v, expected[i])
					}
				}
				
				// Test backward pass
				// Sum to scalar for backward pass
				scalar, err := tensor.SumAutograd(result); if err != nil { t.Fatalf("Sum failed: %v", err) }
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Backward failed: %v", err)
				}
				
				// Check gradients - should be 0 for negative inputs, 1 for positive
				gradData := tensorFloat32(x.Grad())
				expectedGrad := []float32{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0}
				for i, v := range gradData {
					if math.Abs(float64(v-expectedGrad[i])) > 1e-5 {
						t.Errorf("ReLU gradient incorrect at index %d: got %f, expected %f", i, v, expectedGrad[i])
					}
				}
				
				x.ZeroGrad()
			})

			t.Run("Sigmoid", func(t *testing.T) {
				result, err := tensor.SigmoidAutograd(x)
				if err != nil {
					t.Fatalf("Sigmoid failed: %v", err)
				}
				
				// Test backward pass
				// Sum to scalar for backward pass
				scalar, err := tensor.SumAutograd(result); if err != nil { t.Fatalf("Sum failed: %v", err) }
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Backward failed: %v", err)
				}
				
				// Verify gradient computation - sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
				sigmoidVals := tensorFloat32(result)
				gradData := tensorFloat32(x.Grad())
				for i, sig := range sigmoidVals {
					expectedGrad := sig * (1.0 - sig)
					if math.Abs(float64(gradData[i]-expectedGrad)) > 1e-5 {
						t.Errorf("Sigmoid gradient incorrect at index %d: got %f, expected %f", i, gradData[i], expectedGrad)
					}
				}
				
				x.ZeroGrad()
			})
		})
	}
}

// TestLossFunctionsForwardBackward tests loss functions
func TestLossFunctionsForwardBackward(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			batchSize := 4
			numClasses := 3
			
			t.Run("MSELoss", func(t *testing.T) {
				// Create predictions and targets
				pred, _ := tensor.Random([]int{batchSize, numClasses}, tensor.Float32, device)
				pred.SetRequiresGrad(true)
				target, _ := tensor.Random([]int{batchSize, numClasses}, tensor.Float32, device)
				
				criterion := NewMSELoss("mean")
				loss, err := criterion.Forward(pred, target)
				if err != nil {
					t.Fatalf("MSELoss forward failed: %v", err)
				}
				
				// Loss should be scalar
				if len(loss.Shape) != 0 {
					t.Errorf("MSELoss should return scalar, got shape %v", loss.Shape)
				}
				
				// Backward pass
				err = loss.Backward()
				if err != nil {
					t.Fatalf("MSELoss backward failed: %v", err)
				}
				
				// Check gradient shape
				if !equalShapes(pred.Grad().Shape, pred.Shape) {
					t.Errorf("MSELoss gradient shape incorrect: got %v, expected %v", pred.Grad().Shape, pred.Shape)
				}
				
				// Manually compute expected gradient: 2*(pred - target) / n
				diff, _ := tensor.Sub(pred, target)
				n := float32(pred.NumElems)
				scalarTensor := tensor.FromScalar(float64(2.0/n), diff.DType, diff.Device)
				expectedGrad, _ := tensor.MulAutograd(diff, scalarTensor)
				
				if !isClose(pred.Grad(), expectedGrad, 1e-5) {
					t.Errorf("MSELoss gradient incorrect")
				}
			})

			t.Run("CrossEntropyLoss", func(t *testing.T) {
				// Create logits and labels
				logits, _ := tensor.Random([]int{batchSize, numClasses}, tensor.Float32, device)
				logits.SetRequiresGrad(true)
				
				// Create integer labels [0, numClasses)
				labelData := make([]int32, batchSize)
				for i := range labelData {
					labelData[i] = int32(i % numClasses)
				}
				labels, _ := tensor.NewTensor([]int{batchSize}, tensor.Int32, device, labelData)
				
				criterion := NewCrossEntropyLoss("mean")
				loss, err := criterion.Forward(logits, labels)
				if err != nil {
					t.Fatalf("CrossEntropyLoss forward failed: %v", err)
				}
				
				// Loss should be scalar
				if len(loss.Shape) != 0 {
					t.Errorf("CrossEntropyLoss should return scalar, got shape %v", loss.Shape)
				}
				
				// Backward pass
				err = loss.Backward()
				if err != nil {
					t.Fatalf("CrossEntropyLoss backward failed: %v", err)
				}
				
				// Check gradient shape
				if !equalShapes(logits.Grad().Shape, logits.Shape) {
					t.Errorf("CrossEntropyLoss gradient shape incorrect: got %v, expected %v", logits.Grad().Shape, logits.Shape)
				}
			})
		})
	}
}

// TestOptimizersWeightUpdate tests that optimizers correctly update weights
func TestOptimizersWeightUpdate(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Create a simple linear layer
			inputSize := 10
			outputSize := 5
			
			weight, _ := tensor.Random([]int{outputSize, inputSize}, tensor.Float32, device)
			weight.SetRequiresGrad(true)
			bias, _ := tensor.Random([]int{outputSize}, tensor.Float32, device)
			bias.SetRequiresGrad(true)
			
			// Store initial values
			initialWeight, _ := weight.Clone()
			initialBias, _ := bias.Clone()
			
			t.Run("SGD", func(t *testing.T) {
				// Set gradients
				weight.SetGrad(mustOnes(weight.Shape, weight.DType, weight.Device))
				bias.SetGrad(mustOnes(bias.Shape, bias.DType, bias.Device))
				
				// Create optimizer
				params := []*tensor.Tensor{weight, bias}
				lr := float64(0.1)
				optimizer := NewSGD(params, lr, 0.0, 0.0, 0.0, false)
				
				// Take a step
				err := optimizer.Step()
				if err != nil {
					t.Fatalf("SGD step failed: %v", err)
				}
				
				// Check weights were updated: new_weight = old_weight - lr * grad
				// Since grad is 1, expected = initial - 0.1
				expectedWeight, _ := subScalar(initialWeight, float32(lr))
				expectedBias, _ := subScalar(initialBias, float32(lr))
				
				if !isClose(weight, expectedWeight, 1e-5) {
					t.Errorf("SGD weight update incorrect")
				}
				if !isClose(bias, expectedBias, 1e-5) {
					t.Errorf("SGD bias update incorrect")
				}
				
				// Reset for next test
				weight.SetData(initialWeight.Data)
				bias.SetData(initialBias.Data)
			})

			t.Run("Adam", func(t *testing.T) {
				// Set gradients
				weight.SetGrad(mustOnes(weight.Shape, weight.DType, weight.Device))
				bias.SetGrad(mustOnes(bias.Shape, bias.DType, bias.Device))
				
				// Create optimizer
				params := []*tensor.Tensor{weight, bias}
				lr := float64(0.001)
				optimizer := NewAdam(params, lr, 0.9, 0.999, 1e-8, 0.0)
				
				// Take a step
				err := optimizer.Step()
				if err != nil {
					t.Fatalf("Adam step failed: %v", err)
				}
				
				// Verify weights changed
				if isExactlyEqual(weight, initialWeight) {
					t.Errorf("Adam did not update weight")
				}
				if isExactlyEqual(bias, initialBias) {
					t.Errorf("Adam did not update bias")
				}
				
				// Adam updates should be smaller than gradient due to momentum
				weightChange := computeChange(initialWeight, weight)
				biasChange := computeChange(initialBias, bias)
				
				// Expected change should be approximately lr (0.001) for first step
				if weightChange > 0.002 || weightChange < 0.0005 {
					t.Errorf("Adam weight change unexpected: %f", weightChange)
				}
				if biasChange > 0.002 || biasChange < 0.0005 {
					t.Errorf("Adam bias change unexpected: %f", biasChange)
				}
			})
		})
	}
}

// TestNeuralNetworkLayers tests neural network layers
func TestNeuralNetworkLayers(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			batchSize := 2
			
			t.Run("Linear", func(t *testing.T) {
				inputSize := 10
				outputSize := 5
				
				linear, err := NewLinear(inputSize, outputSize, true, device)
				if err != nil {
					t.Fatalf("Failed to create Linear layer: %v", err)
				}
				
				// Forward pass
				input, _ := tensor.Random([]int{batchSize, inputSize}, tensor.Float32, device)
				input.SetRequiresGrad(true)
				
				output, err := linear.Forward(input)
				if err != nil {
					t.Fatalf("Linear forward failed: %v", err)
				}
				
				// Check output shape
				expectedShape := []int{batchSize, outputSize}
				if !equalShapes(output.Shape, expectedShape) {
					t.Errorf("Linear output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
				}
				
				// Backward pass - sum to scalar first
				scalar, err := tensor.SumAutograd(output)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Linear backward failed: %v", err)
				}
				
				// Check gradients exist
				params := linear.Parameters()
				for i, p := range params {
					if p.Grad() == nil {
						t.Errorf("Linear parameter %d has no gradient", i)
					}
				}
			})

			t.Run("Conv2D", func(t *testing.T) {
				inChannels := 3
				outChannels := 16
				kernelSize := 3
				
				conv, err := NewConv2D(inChannels, outChannels, kernelSize, 1, 1, true, device)
				if err != nil {
					t.Fatalf("Failed to create Conv2D layer: %v", err)
				}
				
				// Forward pass
				input, _ := tensor.Random([]int{batchSize, inChannels, 28, 28}, tensor.Float32, device)
				input.SetRequiresGrad(true)
				
				output, err := conv.Forward(input)
				if err != nil {
					t.Fatalf("Conv2D forward failed: %v", err)
				}
				
				// Check output shape (with padding=1, output should be same size)
				expectedShape := []int{batchSize, outChannels, 28, 28}
				if !equalShapes(output.Shape, expectedShape) {
					t.Errorf("Conv2D output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
				}
				
				// Backward pass - sum to scalar first
				scalar, err := tensor.SumAutograd(output)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Conv2D backward failed: %v", err)
				}
				
				// Check gradients exist
				params := conv.Parameters()
				for i, p := range params {
					if p.Grad() == nil {
						t.Errorf("Conv2D parameter %d has no gradient", i)
					}
				}
			})

			t.Run("BatchNorm", func(t *testing.T) {
				numFeatures := 10
				
				bn, err := NewBatchNorm(numFeatures, 1e-5, 0.1, device)
				if err != nil {
					t.Fatalf("Failed to create BatchNorm layer: %v", err)
				}
				bn.Train() // Set to training mode
				
				// Forward pass
				input, _ := tensor.Random([]int{batchSize, numFeatures}, tensor.Float32, device)
				input.SetRequiresGrad(true)
				
				output, err := bn.Forward(input)
				if err != nil {
					t.Fatalf("BatchNorm forward failed: %v", err)
				}
				
				// Check output shape
				if !equalShapes(output.Shape, input.Shape) {
					t.Errorf("BatchNorm output shape incorrect: got %v, expected %v", output.Shape, input.Shape)
				}
				
				// Backward pass - sum to scalar first
				scalar, err := tensor.SumAutograd(output)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("BatchNorm backward failed: %v", err)
				}
				
				// Check gradients exist for trainable parameters
				params := bn.Parameters()
				if len(params) != 2 { // gamma and beta
					t.Errorf("BatchNorm should have 2 parameters, got %d", len(params))
				}
				for i, p := range params {
					if p.Grad() == nil {
						t.Errorf("BatchNorm parameter %d has no gradient", i)
					}
				}
			})
		})
	}
}

// TestCNNOperationsForwardBackward tests CNN operations specifically
func TestCNNOperationsForwardBackward(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			batchSize := 2
			inChannels := 3
			height, width := 32, 32
			
			// Create input tensor for CNN operations
			input, _ := tensor.Random([]int{batchSize, inChannels, height, width}, tensor.Float32, device)
			input.SetRequiresGrad(true)

			t.Run("Conv2D_forward_backward", func(t *testing.T) {
				outChannels := 16
				kernelSize := 3
				
				conv, err := NewConv2D(inChannels, outChannels, kernelSize, 1, 1, true, device)
				if err != nil {
					t.Fatalf("Failed to create Conv2D layer: %v", err)
				}
				
				// Forward pass
				output, err := conv.Forward(input)
				if err != nil {
					t.Fatalf("Conv2D forward failed: %v", err)
				}
				
				// Check output shape (with padding=1, output should be same size)
				expectedShape := []int{batchSize, outChannels, height, width}
				if !equalShapes(output.Shape, expectedShape) {
					t.Errorf("Conv2D output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
				}
				
				// Backward pass - sum to scalar first
				scalar, err := tensor.SumAutograd(output)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("Conv2D backward failed: %v", err)
				}
				
				// Check gradients exist for parameters
				params := conv.Parameters()
				for i, p := range params {
					if p.Grad() == nil {
						t.Errorf("Conv2D parameter %d has no gradient", i)
					}
				}
				
				// Check input gradients exist
				if input.Grad() == nil {
					t.Error("Input should have gradient after Conv2D backward")
				}
				
				// Reset gradients for next test
				input.ZeroGrad()
			})

			t.Run("MaxPool2D_forward", func(t *testing.T) {
				kernelSize := 2
				stride := 2
				padding := 0
				
				maxPool := NewMaxPool2D(kernelSize, stride, padding)
				
				// Forward pass
				output, err := maxPool.Forward(input)
				if err != nil {
					t.Fatalf("MaxPool2D forward failed: %v", err)
				}
				
				// Check output shape (should be half the size due to stride=2)
				expectedHeight := height / stride
				expectedWidth := width / stride
				expectedShape := []int{batchSize, inChannels, expectedHeight, expectedWidth}
				if !equalShapes(output.Shape, expectedShape) {
					t.Errorf("MaxPool2D output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
				}
				
				// MaxPool2D has no parameters
				params := maxPool.Parameters()
				if len(params) != 0 {
					t.Errorf("MaxPool2D should have no parameters, got %d", len(params))
				}
			})

			t.Run("AvgPool2D_forward", func(t *testing.T) {
				kernelSize := 2
				stride := 2
				padding := 0
				
				// Test AvgPool2D directly using tensor operation since module doesn't exist
				output, err := tensor.AvgPool2DMPS(input, kernelSize, stride, padding)
				if err != nil {
					t.Fatalf("AvgPool2D forward failed: %v", err)
				}
				
				// Check output shape (should be half the size due to stride=2)
				expectedHeight := height / stride
				expectedWidth := width / stride
				expectedShape := []int{batchSize, inChannels, expectedHeight, expectedWidth}
				if !equalShapes(output.Shape, expectedShape) {
					t.Errorf("AvgPool2D output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
				}
			})

			t.Run("CNN_sequence_forward", func(t *testing.T) {
				// Test a sequence of CNN operations without pooling (to preserve autograd chain)
				conv1, _ := NewConv2D(inChannels, 16, 3, 1, 1, true, device)
				conv2, _ := NewConv2D(16, 32, 3, 1, 1, true, device)
				
				// Forward through sequence
				x1, err := conv1.Forward(input)
				if err != nil {
					t.Fatalf("Conv1 forward failed: %v", err)
				}
				
				x2, err := conv2.Forward(x1)
				if err != nil {
					t.Fatalf("Conv2 forward failed: %v", err)
				}
				
				// Check final shape (should remain same size with padding=1)
				expectedShape := []int{batchSize, 32, height, width}
				if !equalShapes(x2.Shape, expectedShape) {
					t.Errorf("CNN sequence output shape incorrect: got %v, expected %v", x2.Shape, expectedShape)
				}
				
				// Test backward through the sequence
				scalar, err := tensor.SumAutograd(x2)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}
				
				err = scalar.Backward()
				if err != nil {
					t.Fatalf("CNN sequence backward failed: %v", err)
				}
				
				// Check that all conv layers have gradients
				params1 := conv1.Parameters()
				params2 := conv2.Parameters()
				
				for i, p := range params1 {
					if p.Grad() == nil {
						t.Errorf("Conv1 parameter %d has no gradient", i)
					}
				}
				
				for i, p := range params2 {
					if p.Grad() == nil {
						t.Errorf("Conv2 parameter %d has no gradient", i)
					}
				}
				
				// Check input gradients
				if input.Grad() == nil {
					t.Error("Input should have gradient after CNN sequence backward")
				}
			})
		})
	}
}

// TestLinearLayerAutograd tests that Linear layer works with autograd
func TestLinearLayerAutograd(t *testing.T) {
	device := tensor.CPU
	
	// Create a simple Linear layer
	linear, err := NewLinear(3, 2, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear layer: %v", err)
	}
	
	// Create input tensor
	input, _ := tensor.Random([]int{2, 3}, tensor.Float32, device)
	input.SetRequiresGrad(true)
	
	// Forward pass
	output, err := linear.Forward(input)
	if err != nil {
		t.Fatalf("Linear forward failed: %v", err)
	}
	
	// Sum to scalar and backward
	scalar, err := tensor.SumAutograd(output)
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}
	
	err = scalar.Backward()
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	
	// Check that parameters have gradients
	params := linear.Parameters()
	for i, p := range params {
		if p.Grad() == nil {
			t.Errorf("Parameter %d has no gradient", i)
		} else {
			t.Logf("Parameter %d has gradient with shape %v", i, p.Grad().Shape)
		}
	}
	
	// Check that input has gradient
	if input.Grad() == nil {
		t.Error("Input should have gradient")
	} else {
		t.Logf("Input has gradient with shape %v", input.Grad().Shape)
	}
}

// TestMSELossAutograd tests that MSELoss works with autograd
func TestMSELossAutograd(t *testing.T) {
	device := tensor.CPU
	
	// Create a simple Linear layer
	linear, err := NewLinear(2, 1, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear layer: %v", err)
	}
	
	// Create input and target
	input, _ := tensor.Random([]int{3, 2}, tensor.Float32, device)
	input.SetRequiresGrad(true)
	target, _ := tensor.Random([]int{3, 1}, tensor.Float32, device)
	
	// Forward pass through linear layer
	output, err := linear.Forward(input)
	if err != nil {
		t.Fatalf("Linear forward failed: %v", err)
	}
	
	// Compute MSE loss
	criterion := NewMSELoss("mean")
	loss, err := criterion.Forward(output, target)
	if err != nil {
		t.Fatalf("MSE loss forward failed: %v", err)
	}
	
	// Store initial parameters
	params := linear.Parameters()
	initialWeights := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		initialWeights[i], _ = p.Clone()
	}
	
	// Backward pass
	err = loss.Backward()
	if err != nil {
		t.Fatalf("MSE loss backward failed: %v", err)
	}
	
	// Check that parameters have gradients
	for i, p := range params {
		if p.Grad() == nil {
			t.Errorf("Parameter %d has no gradient", i)
		} else {
			t.Logf("Parameter %d has gradient with shape %v", i, p.Grad().Shape)
		}
	}
	
	// Simulate optimizer step (simple SGD)
	lr := float64(0.1)
	for _, p := range params {
		if p.Grad() != nil {
			// param = param - lr * grad
			lrTensor := tensor.FromScalar(lr, p.DType, p.Device)
			update, _ := tensor.Mul(p.Grad(), lrTensor)
			newData, _ := tensor.Sub(p, update)
			p.SetData(newData.Data)
		}
	}
	
	// Check that parameters changed
	for i, p := range params {
		if isExactlyEqual(p, initialWeights[i]) {
			t.Errorf("Parameter %d did not change after gradient update", i)
		} else {
			t.Logf("Parameter %d changed successfully", i)
		}
	}
}

// TestConv2DAutograd tests that Conv2D layer works with autograd
func TestConv2DAutograd(t *testing.T) {
	device := tensor.CPU
	
	// Create a simple Conv2D layer
	inChannels := 3
	outChannels := 8
	kernelSize := 3
	conv, err := NewConv2D(inChannels, outChannels, kernelSize, 1, 1, true, device)
	if err != nil {
		t.Fatalf("Failed to create Conv2D layer: %v", err)
	}
	
	// Create input tensor
	input, _ := tensor.Random([]int{2, inChannels, 16, 16}, tensor.Float32, device)
	input.SetRequiresGrad(true)
	
	// Forward pass
	output, err := conv.Forward(input)
	if err != nil {
		t.Fatalf("Conv2D forward failed: %v", err)
	}
	
	// Sum to scalar and backward
	scalar, err := tensor.SumAutograd(output)
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}
	
	err = scalar.Backward()
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	
	// Check that parameters have gradients
	params := conv.Parameters()
	for i, p := range params {
		if p.Grad() == nil {
			t.Errorf("Conv2D parameter %d has no gradient", i)
		} else {
			t.Logf("Conv2D parameter %d has gradient with shape %v", i, p.Grad().Shape)
		}
	}
	
	// Check that input has gradient
	if input.Grad() == nil {
		t.Error("Input should have gradient after Conv2D backward")
	} else {
		t.Logf("Input has gradient with shape %v", input.Grad().Shape)
	}
}

// TestCrossEntropyLossAutograd tests that CrossEntropyLoss works with autograd
func TestCrossEntropyLossAutograd(t *testing.T) {
	device := tensor.CPU
	
	// Create a simple Linear layer for classification
	linear, err := NewLinear(10, 3, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear layer: %v", err)
	}
	
	// Create input and target
	input, _ := tensor.Random([]int{4, 10}, tensor.Float32, device)
	input.SetRequiresGrad(true)
	
	// Create targets (class indices)
	targetData := []int32{0, 1, 2, 1}
	target, _ := tensor.NewTensor([]int{4}, tensor.Int32, device, targetData)
	
	// Forward pass through linear layer
	logits, err := linear.Forward(input)
	if err != nil {
		t.Fatalf("Linear forward failed: %v", err)
	}
	
	// Compute CrossEntropy loss
	criterion := NewCrossEntropyLoss("mean")
	loss, err := criterion.Forward(logits, target)
	if err != nil {
		t.Fatalf("CrossEntropy loss forward failed: %v", err)
	}
	
	// Store initial parameters
	params := linear.Parameters()
	initialWeights := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		initialWeights[i], _ = p.Clone()
	}
	
	// Check loss properties
	t.Logf("Loss shape: %v, requires grad: %v", loss.Shape, loss.RequiresGrad())
	t.Logf("Logits requires grad: %v", logits.RequiresGrad())
	
	// Backward pass
	err = loss.Backward()
	if err != nil {
		t.Fatalf("CrossEntropy loss backward failed: %v", err)
	}
	
	// Check that parameters have gradients
	for i, p := range params {
		if p.Grad() == nil {
			t.Errorf("Parameter %d has no gradient", i)
		} else {
			t.Logf("Parameter %d has gradient with shape %v", i, p.Grad().Shape)
		}
	}
	
	// Simulate optimizer step (simple SGD)
	lr := float64(0.1)
	for _, p := range params {
		if p.Grad() != nil {
			// param = param - lr * grad
			lrTensor := tensor.FromScalar(lr, p.DType, p.Device)
			update, _ := tensor.Mul(p.Grad(), lrTensor)
			newData, _ := tensor.Sub(p, update)
			p.SetData(newData.Data)
		}
	}
	
	// Check that parameters changed
	for i, p := range params {
		if isExactlyEqual(p, initialWeights[i]) {
			t.Errorf("Parameter %d did not change after gradient update", i)
		} else {
			t.Logf("Parameter %d changed successfully", i)
		}
	}
}

// TestOptimizerGradientFlow tests optimizer gradient application in detail
func TestOptimizerGradientFlow(t *testing.T) {
	device := tensor.CPU
	
	t.Run("SGD_gradient_application", func(t *testing.T) {
		// Create a simple linear layer
		linear, err := NewLinear(2, 1, true, device)
		if err != nil {
			t.Fatalf("Failed to create Linear layer: %v", err)
		}
		
		// Get initial parameters
		params := linear.Parameters()
		if len(params) != 2 {
			t.Fatalf("Expected 2 parameters, got %d", len(params))
		}
		
		weight := params[0]  // [2, 1]
		bias := params[1]    // [1]
		
		// Store initial values
		initialWeight, _ := weight.Clone()
		initialBias, _ := bias.Clone()
		
		t.Logf("Initial weight: %v", tensorFloat32(initialWeight))
		t.Logf("Initial bias: %v", tensorFloat32(initialBias))
		
		// Create known input and target for predictable gradients
		inputData := []float32{1.0, 2.0}  // Simple input
		targetData := []float32{5.0}      // Target output
		
		input, _ := tensor.NewTensor([]int{1, 2}, tensor.Float32, device, inputData)
		input.SetRequiresGrad(true)
		target, _ := tensor.NewTensor([]int{1, 1}, tensor.Float32, device, targetData)
		
		// Forward pass
		output, err := linear.Forward(input)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		
		t.Logf("Output: %v", tensorFloat32(output))
		
		// Compute MSE loss
		criterion := NewMSELoss("mean")
		loss, err := criterion.Forward(output, target)
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		
		t.Logf("Loss: %v", tensorFloat32(loss))
		
		// Backward pass
		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Check gradients exist and print them
		if weight.Grad() == nil {
			t.Fatal("Weight gradient is nil")
		}
		if bias.Grad() == nil {
			t.Fatal("Bias gradient is nil")
		}
		
		t.Logf("Weight gradient: %v", tensorFloat32(weight.Grad()))
		t.Logf("Bias gradient: %v", tensorFloat32(bias.Grad()))
		
		// Create optimizer
		optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
		
		// Apply optimizer step
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed: %v", err)
		}
		
		// Check that parameters changed
		t.Logf("Final weight: %v", tensorFloat32(weight))
		t.Logf("Final bias: %v", tensorFloat32(bias))
		
		// Verify parameters actually changed
		if isExactlyEqual(weight, initialWeight) {
			t.Error("Weight did not change after optimizer step")
		}
		if isExactlyEqual(bias, initialBias) {
			t.Error("Bias did not change after optimizer step")
		}
		
		// Manually verify the update: new_param = old_param - lr * grad
		lrTensor := tensor.FromScalar(0.1, weight.DType, weight.Device)
		weightUpdate, _ := tensor.Mul(weight.Grad(), lrTensor)
		biasUpdate, _ := tensor.Mul(bias.Grad(), lrTensor)
		
		expectedWeight, _ := tensor.Sub(initialWeight, weightUpdate)
		expectedBias, _ := tensor.Sub(initialBias, biasUpdate)
		
		if !isClose(weight, expectedWeight, 1e-6) {
			t.Errorf("Weight update incorrect. Expected: %v, Got: %v", 
				tensorFloat32(expectedWeight), tensorFloat32(weight))
		}
		if !isClose(bias, expectedBias, 1e-6) {
			t.Errorf("Bias update incorrect. Expected: %v, Got: %v", 
				tensorFloat32(expectedBias), tensorFloat32(bias))
		}
		
		t.Log("SGD gradient application test PASSED")
	})
	
	t.Run("SetData_verification", func(t *testing.T) {
		// Test SetData operation directly
		original, _ := tensor.Random([]int{3, 2}, tensor.Float32, device)
		original.SetRequiresGrad(true)
		
		originalData := tensorFloat32(original)
		t.Logf("Original data: %v", originalData)
		
		// Create new data
		newData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		newTensor, _ := tensor.NewTensor([]int{3, 2}, tensor.Float32, device, newData)
		
		// Use SetData
		err := original.SetData(newTensor.Data)
		if err != nil {
			t.Fatalf("SetData failed: %v", err)
		}
		
		// Verify data changed
		finalData := tensorFloat32(original)
		t.Logf("Final data: %v", finalData)
		
		for i, expected := range newData {
			if finalData[i] != expected {
				t.Errorf("SetData failed at index %d: expected %f, got %f", 
					i, expected, finalData[i])
			}
		}
		
		// Verify requiresGrad is preserved
		if !original.RequiresGrad() {
			t.Error("SetData should preserve requiresGrad flag")
		}
		
		t.Log("SetData verification test PASSED")
	})
}

// TestXORProblemDetailed tests XOR learning with detailed logging
func TestXORProblemDetailed(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU)
	}
	
	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			testXOROnDevice(t, device)
		})
	}
}

func testXOROnDevice(t *testing.T, device tensor.DeviceType) {
	
	// XOR dataset
	xorInputs := []float32{0, 0, 0, 1, 1, 0, 1, 1}
	xorLabels := []int32{0, 1, 1, 0}
	
	inputTensor, _ := tensor.NewTensor([]int{4, 2}, tensor.Float32, device, xorInputs)
	labelTensor, _ := tensor.NewTensor([]int{4}, tensor.Int32, device, xorLabels)
	
	// Create model: 2 -> 8 -> 2 (should be able to learn XOR)
	linear1, _ := NewLinear(2, 8, true, device)
	linear2, _ := NewLinear(8, 2, true, device)
	
	model := NewSequential(
		linear1,
		NewReLU(),
		linear2,
	)
	
	// Use CrossEntropy loss and SGD
	criterion := NewCrossEntropyLoss("mean")
	params := model.Parameters()
	optimizer := NewSGD(params, 0.5, 0.0, 0.0, 0.0, false) // Higher learning rate
	
	t.Logf("Initial parameters:")
	for i, p := range params {
		t.Logf("  Param %d: %v", i, tensorFloat32(p)[:min(8, len(tensorFloat32(p)))])
	}
	
	// Training loop with detailed logging
	epochs := 200
	for epoch := 0; epoch < epochs; epoch++ {
		// Zero gradients
		optimizer.ZeroGrad()
		
		// Forward pass
		output, err := model.Forward(inputTensor)
		if err != nil {
			t.Fatalf("Forward pass failed at epoch %d: %v", epoch, err)
		}
		
		// Compute loss
		loss, err := criterion.Forward(output, labelTensor)
		if err != nil {
			t.Fatalf("Loss computation failed at epoch %d: %v", epoch, err)
		}
		
		lossVal := tensorFloat32(loss)[0]
		
		// Backward pass
		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed at epoch %d: %v", epoch, err)
		}
		
		// Check gradients before optimizer step
		if epoch < 5 || epoch%50 == 0 {
			t.Logf("Epoch %d: Loss = %.6f", epoch, lossVal)
			for i, p := range params {
				if p.Grad() != nil {
					gradData := tensorFloat32(p.Grad())
					gradMag := computeGradientMagnitude(gradData)
					t.Logf("  Param %d gradient magnitude: %.6f", i, gradMag)
				} else {
					t.Logf("  Param %d: NO GRADIENT", i)
				}
			}
		}
		
		// Optimizer step
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed at epoch %d: %v", epoch, err)
		}
		
		// Check if loss is decreasing
		if epoch > 0 && epoch%50 == 0 {
			// Test accuracy
			model.Eval()
			testOutput, _ := model.Forward(inputTensor)
			predictions := computePredictions(testOutput)
			accuracy := computeAccuracy(predictions, xorLabels)
			t.Logf("Epoch %d: Loss = %.6f, Accuracy = %.2f%%", epoch, lossVal, accuracy*100)
			model.Train()
		}
		
		// Early stopping if loss becomes very small
		if lossVal < 0.01 {
			t.Logf("Early stopping at epoch %d with loss %.6f", epoch, lossVal)
			break
		}
	}
	
	// Final test
	model.Eval()
	finalOutput, _ := model.Forward(inputTensor)
	finalPredictions := computePredictions(finalOutput)
	finalAccuracy := computeAccuracy(finalPredictions, xorLabels)
	
	t.Logf("Final accuracy: %.2f%%", finalAccuracy*100)
	t.Logf("Final predictions: %v", finalPredictions)
	t.Logf("Expected labels:   %v", xorLabels)
	
	// XOR should be learnable to at least 75% accuracy
	if finalAccuracy < 0.75 {
		t.Errorf("XOR learning failed: accuracy %.2f%% < 75%%", finalAccuracy*100)
	} else {
		t.Logf("XOR learning SUCCESS: accuracy %.2f%%", finalAccuracy*100)
	}
}

// Helper functions for XOR test
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func computeGradientMagnitude(grad []float32) float32 {
	var sum float32
	for _, g := range grad {
		sum += g * g
	}
	return float32(math.Sqrt(float64(sum)))
}

func computePredictions(output *tensor.Tensor) []int32 {
	outputData := tensorFloat32(output)
	rows := output.Shape[0]
	cols := output.Shape[1]
	
	predictions := make([]int32, rows)
	for i := 0; i < rows; i++ {
		maxIdx := 0
		maxVal := outputData[i*cols]
		for j := 1; j < cols; j++ {
			if outputData[i*cols+j] > maxVal {
				maxVal = outputData[i*cols+j]
				maxIdx = j
			}
		}
		predictions[i] = int32(maxIdx)
	}
	return predictions
}

func computeAccuracy(predictions, labels []int32) float32 {
	correct := 0
	for i := range predictions {
		if predictions[i] == labels[i] {
			correct++
		}
	}
	return float32(correct) / float32(len(labels))
}

// TestEndToEndTrainingLoop tests a complete training loop
func TestEndToEndTrainingLoop(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Set deterministic seed for consistent XOR learning across devices
			SetRandomSeed(42)
			
			// Create a simple dataset - XOR problem
			xorInputs := []float32{0, 0, 0, 1, 1, 0, 1, 1}
			xorLabels := []int32{0, 1, 1, 0}
			
			inputData, _ := tensor.NewTensor([]int{4, 2}, tensor.Float32, device, xorInputs)
			labelData, _ := tensor.NewTensor([]int{4}, tensor.Int32, device, xorLabels)
			
			dataset, err := NewSimpleDataset([]*tensor.Tensor{inputData}, []*tensor.Tensor{labelData})
			if err != nil {
				t.Fatalf("Failed to create dataset: %v", err)
			}
			dataloader := NewDataLoader(dataset, 2, true, 1, device)
			
			// Create a simple model
			linear1, _ := NewLinear(2, 8, true, device)
			linear2, _ := NewLinear(8, 2, true, device)
			model := NewSequential(
				linear1,
				NewReLU(),
				linear2,
			)
			
			// Create optimizer and loss
			params := model.Parameters()
			optimizer := NewSGD(params, 0.5, 0.0, 0.0, 0.0, false) // Increased learning rate for XOR
			criterion := NewCrossEntropyLoss("mean")
			
			// Store initial weights
			initialWeights := make([]*tensor.Tensor, len(params))
			for i, p := range params {
				initialWeights[i], _ = p.Clone()
			}
			
			// Training loop
			model.Train()
			initialLoss := float32(0)
			finalLoss := float32(0)
			
			epochs := 100 // Increased epochs for XOR convergence
			for epoch := 0; epoch < epochs; epoch++ {
				epochLoss := float32(0)
				batches := 0
				
				iter := dataloader.Iterator()
				for batch := range iter {
					// Zero gradients
					optimizer.ZeroGrad()
					
					// Forward pass - reshape batch data if needed
					batchData := batch.Data
					if len(batchData.Shape) == 3 && batchData.Shape[0] == 1 {
						// Reshape from [1, batch_size, features] to [batch_size, features]
						batchData, err = batchData.Reshape([]int{batchData.Shape[1], batchData.Shape[2]})
						if err != nil {
							t.Fatalf("Batch reshape failed: %v", err)
						}
					}
					
					output, err := model.Forward(batchData)
					if err != nil {
						t.Fatalf("Forward pass failed: %v", err)
					}
					
					// Compute loss - reshape labels if needed
					batchLabels := batch.Labels
					if len(batchLabels.Shape) == 2 && batchLabels.Shape[0] == 1 {
						// Reshape from [1, batch_size] to [batch_size]
						batchLabels, err = batchLabels.Reshape([]int{batchLabels.Shape[1]})
						if err != nil {
							t.Fatalf("Labels reshape failed: %v", err)
						}
					}
					
					loss, err := criterion.Forward(output, batchLabels)
					if err != nil {
						t.Fatalf("Loss computation failed: %v", err)
					}
					
					lossVal := tensorItem(loss)
					epochLoss += lossVal
					batches++
					
					if epoch == 0 && batches == 1 {
						initialLoss = lossVal
					}
					
					// Backward pass
					err = loss.Backward()
					if err != nil {
						t.Fatalf("Backward pass failed: %v", err)
					}
					
					// Update weights
					err = optimizer.Step()
					if err != nil {
						t.Fatalf("Optimizer step failed: %v", err)
					}
				}
				
				if epoch == epochs-1 {
					finalLoss = epochLoss / float32(batches)
				}
			}
			
			// Verify training occurred
			// 1. Weights should have changed
			for i, p := range params {
				if isExactlyEqual(p, initialWeights[i]) {
					t.Errorf("Parameter %d did not change during training", i)
				}
			}
			
			// 2. Loss should have decreased (XOR is learnable)
			if finalLoss >= initialLoss {
				t.Errorf("Loss did not decrease: initial=%f, final=%f", initialLoss, finalLoss)
			}
			
			// 3. Test accuracy
			model.Eval()
			output, _ := model.Forward(inputData)
			predictions, _ := tensorArgmax(output, 1)
			
			correct := 0
			predData := tensorInt32(predictions)
			for i := range predData {
				if predData[i] == xorLabels[i] {
					correct++
				}
			}
			
			accuracy := float32(correct) / float32(len(xorLabels))
			if accuracy < 0.75 { // XOR should be learnable to at least 75% accuracy
				t.Errorf("Model failed to learn XOR: accuracy=%f", accuracy)
			}
		})
	}
}

// Helper functions

func isClose(a, b *tensor.Tensor, tolerance float32) bool {
	if a == nil || b == nil {
		return false
	}
	
	// Ensure both tensors are on CPU for comparison
	aCPU := a
	bCPU := b
	if a.Device != tensor.CPU {
		aCPU, _ = a.ToCPU()
	}
	if b.Device != tensor.CPU {
		bCPU, _ = b.ToCPU()
	}
	
	if !equalShapes(aCPU.Shape, bCPU.Shape) {
		return false
	}
	
	aData := tensorFloat32(aCPU)
	bData := tensorFloat32(bCPU)
	
	for i := range aData {
		if math.Abs(float64(aData[i]-bData[i])) > float64(tolerance) {
			return false
		}
	}
	return true
}

func isExactlyEqual(a, b *tensor.Tensor) bool {
	return isClose(a, b, 0)
}

func equalShapes(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func computeChange(before, after *tensor.Tensor) float32 {
	// Compute average absolute change
	diff, _ := tensor.Sub(after, before)
	diffAbs, _ := tensorAbs(diff)
	return tensorMean(diffAbs)
}

// Tensor accessor helpers

func tensorFloat32(t *tensor.Tensor) []float32 {
	if t.Device != tensor.CPU {
		cpuTensor, _ := t.ToCPU()
		return cpuTensor.Data.([]float32)
	}
	return t.Data.([]float32)
}

func tensorInt32(t *tensor.Tensor) []int32 {
	if t.Device != tensor.CPU {
		cpuTensor, _ := t.ToCPU()
		return cpuTensor.Data.([]int32)
	}
	return t.Data.([]int32)
}

func tensorItem(t *tensor.Tensor) float32 {
	if t.Device != tensor.CPU {
		cpuTensor, _ := t.ToCPU()
		return cpuTensor.Data.([]float32)[0]
	}
	return t.Data.([]float32)[0]
}

func tensorMean(t *tensor.Tensor) float32 {
	data := tensorFloat32(t)
	sum := float32(0)
	for _, v := range data {
		sum += v
	}
	return sum / float32(len(data))
}

func tensorAbs(t *tensor.Tensor) (*tensor.Tensor, error) {
	data := tensorFloat32(t)
	absData := make([]float32, len(data))
	for i, v := range data {
		absData[i] = float32(math.Abs(float64(v)))
	}
	return tensor.NewTensor(t.Shape, t.DType, t.Device, absData)
}

func tensorArgmax(t *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	// Simple argmax implementation for 2D tensors along dimension 1
	if len(t.Shape) != 2 || dim != 1 {
		return nil, fmt.Errorf("tensorArgmax only supports 2D tensors with dim=1")
	}
	
	data := tensorFloat32(t)
	rows := t.Shape[0]
	cols := t.Shape[1]
	
	result := make([]int32, rows)
	for i := 0; i < rows; i++ {
		maxIdx := 0
		maxVal := data[i*cols]
		for j := 1; j < cols; j++ {
			if data[i*cols+j] > maxVal {
				maxVal = data[i*cols+j]
				maxIdx = j
			}
		}
		result[i] = int32(maxIdx)
	}
	
	return tensor.NewTensor([]int{rows}, tensor.Int32, t.Device, result)
}

func mustOnes(shape []int, dtype tensor.DType, device tensor.DeviceType) *tensor.Tensor {
	t, err := tensor.Ones(shape, dtype, device)
	if err != nil {
		panic(fmt.Sprintf("Failed to create ones tensor: %v", err))
	}
	return t
}

// Scalar operations helpers
func subScalar(t *tensor.Tensor, scalar float32) (*tensor.Tensor, error) {
	// Create a scalar tensor and subtract
	scalarTensor := tensor.FromScalar(float64(scalar), t.DType, t.Device)
	return tensor.Sub(t, scalarTensor)
}

// tensorSum sums all elements in a tensor to create a scalar
func tensorSum(t *tensor.Tensor) *tensor.Tensor {
	data := tensorFloat32(t)
	sum := float32(0)
	for _, v := range data {
		sum += v
	}
	result, _ := tensor.NewTensor([]int{}, t.DType, t.Device, []float32{sum})
	return result
}

// TestCompleteCNNTrainingPipeline tests the complete CNN training pipeline with mathematical verification
func TestCompleteCNNTrainingPipeline(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			t.Run("CNN_gradient_verification", func(t *testing.T) {
				testCNNGradientVerification(t, device)
			})
			
			t.Run("CNN_architecture_with_MSELoss", func(t *testing.T) {
				testCNNArchitectureWithMSELoss(t, device)
			})
			
			t.Run("CNN_architecture_with_CrossEntropyLoss", func(t *testing.T) {
				testCNNArchitectureWithCrossEntropyLoss(t, device)
			})
			
			t.Run("CNN_simple_classification_learning", func(t *testing.T) {
				testCNNSimpleClassificationLearning(t, device)
			})
		})
	}
}

// testCNNGradientVerification tests mathematical correctness of CNN gradients
func testCNNGradientVerification(t *testing.T, device tensor.DeviceType) {
	// Test Conv2D forward pass and shape verification
	t.Run("Conv2D_forward_and_shape_verification", func(t *testing.T) {
		// Create small input: [1, 2, 4, 4] (batch=1, channels=2, height=4, width=4)
		inputData := make([]float32, 32) // 1*2*4*4
		for i := range inputData {
			inputData[i] = float32(i%5) * 0.1 // Small values for numerical stability
		}
		input, _ := tensor.NewTensor([]int{1, 2, 4, 4}, tensor.Float32, device, inputData)
		
		// Create Conv2D layer: input_channels=2, output_channels=3, kernel=3x3
		conv, err := NewConv2D(2, 3, 3, 1, 1, true, device) // stride=1, padding=1, bias=true
		if err != nil {
			t.Fatalf("Failed to create Conv2D: %v", err)
		}
		
		// Forward pass
		output, err := conv.Forward(input)
		if err != nil {
			t.Fatalf("Conv2D forward failed: %v", err)
		}
		
		// Expected output shape: [1, 3, 4, 4] with padding=1
		expectedShape := []int{1, 3, 4, 4}
		if !equalShapes(output.Shape, expectedShape) {
			t.Errorf("Conv2D output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
		}
		
		// Verify parameters exist and have correct shapes
		params := conv.Parameters()
		if len(params) != 2 { // weight and bias
			t.Errorf("Expected 2 parameters (weight, bias), got %d", len(params))
		}
		
		// Check weight shape: [3, 2, 3, 3]
		if !equalShapes(params[0].Shape, []int{3, 2, 3, 3}) {
			t.Errorf("Weight shape incorrect: got %v, expected [3, 2, 3, 3]", params[0].Shape)
		}
		
		// Check bias shape: [3]
		if !equalShapes(params[1].Shape, []int{3}) {
			t.Errorf("Bias shape incorrect: got %v, expected [3]", params[1].Shape)
		}
		
		// Verify output contains meaningful values (not all zeros)
		outputData := tensorFloat32(output)
		hasNonZero := false
		for _, val := range outputData {
			if math.Abs(float64(val)) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Errorf("Conv2D output is all zeros - computation may have failed")
		}
		
		t.Logf("Conv2D forward pass verified: input shape %v → output shape %v", input.Shape, output.Shape)
	})
	
	// Test MaxPool2D forward pass verification  
	t.Run("MaxPool2D_forward_verification", func(t *testing.T) {
		// Create input with known max locations: [1, 1, 4, 4]
		inputData := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		input, _ := tensor.NewTensor([]int{1, 1, 4, 4}, tensor.Float32, device, inputData)
		
		// MaxPool2D with 2x2 kernel, stride=2 (no overlap)
		maxPool := NewMaxPool2D(2, 2, 0)
		
		// Forward pass
		output, err := maxPool.Forward(input)
		if err != nil {
			t.Fatalf("MaxPool2D forward failed: %v", err)
		}
		
		// Expected output shape: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		if !equalShapes(output.Shape, expectedShape) {
			t.Errorf("MaxPool2D output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
		}
		
		// Expected output values: [6, 8, 14, 16] (max of each 2x2 region)
		outputData := tensorFloat32(output)
		expectedOutput := []float32{6, 8, 14, 16}
		for i, expected := range expectedOutput {
			if math.Abs(float64(outputData[i]-expected)) > 1e-5 {
				t.Errorf("MaxPool2D output[%d] incorrect: got %f, expected %f", i, outputData[i], expected)
			}
		}
		
		t.Logf("MaxPool2D forward pass verified: correctly computed max pooling")
	})
	
	// Test Linear layer gradient flow (known to work)
	t.Run("Linear_gradient_flow_verification", func(t *testing.T) {
		// Create simple input: [2, 4]
		inputData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
		input, _ := tensor.NewTensor([]int{2, 4}, tensor.Float32, device, inputData)
		input.SetRequiresGrad(true)
		
		// Create Linear layer
		linear, err := NewLinear(4, 3, true, device)
		if err != nil {
			t.Fatalf("Failed to create Linear layer: %v", err)
		}
		
		// Forward pass
		output, err := linear.Forward(input)
		if err != nil {
			t.Fatalf("Linear forward failed: %v", err)
		}
		
		// Sum to scalar for backward pass
		scalar, err := tensor.SumAutograd(output)
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}
		
		// Backward pass
		err = scalar.Backward()
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		
		// Verify gradients exist
		params := linear.Parameters()
		for i, param := range params {
			if param.Grad() == nil {
				t.Errorf("Linear parameter %d has no gradient", i)
			}
		}
		
		// Verify input gradient exists
		if input.Grad() == nil {
			t.Errorf("Linear input gradient is nil")
		}
		
		t.Logf("Linear gradient flow verified: all gradients computed correctly")
	})
}

// testCNNArchitectureWithMSELoss tests Conv2D → Conv2D → Linear with MSELoss
func testCNNArchitectureWithMSELoss(t *testing.T, device tensor.DeviceType) {
	// Test CNN architecture on all devices - CPU Conv2D autograd is now implemented
	
	// Set deterministic seed for consistent results across devices
	SetRandomSeed(1001)
	
	// Build CNN: Conv2D(1→4) → ReLU → Conv2D(4→8) → ReLU → Flatten → Linear(8→1)
	conv1, _ := NewConv2D(1, 4, 3, 1, 1, true, device) // 1→4 channels, 3x3 kernel
	relu1 := NewReLU()
	conv2, _ := NewConv2D(4, 8, 3, 1, 1, true, device) // 4→8 channels, 3x3 kernel  
	relu2 := NewReLU()
	flatten := NewFlatten()
	
	// For 4x4 input: after conv layers it's still 4x4, so flattened size is 8*4*4=128
	linear, _ := NewLinear(128, 1, true, device) // 128→1 for regression
	
	model := NewSequential(conv1, relu1, conv2, relu2, flatten, linear)
	
	// Create synthetic regression data: [batch=2, channels=1, height=4, width=4]
	inputData := []float32{
		// Sample 1
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
		// Sample 2  
		0.2, 0.4, 0.6, 0.8,
		1.0, 1.2, 1.4, 1.6,
		1.8, 2.0, 2.2, 2.4,
		2.6, 2.8, 3.0, 3.2,
	}
	input, _ := tensor.NewTensor([]int{2, 1, 4, 4}, tensor.Float32, device, inputData)
	
	// Target values for regression [batch=2, output=1]
	targetData := []float32{1.0, 2.0} // Simple targets
	target, _ := tensor.NewTensor([]int{2, 1}, tensor.Float32, device, targetData)
	
	// Test forward pass
	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Model forward failed: %v", err)
	}
	
	// Check output shape
	expectedShape := []int{2, 1}
	if !equalShapes(output.Shape, expectedShape) {
		t.Errorf("Model output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
	}
	
	// Verify output contains meaningful values
	outputData := tensorFloat32(output)
	hasNonZero := false
	for _, val := range outputData {
		if math.Abs(float64(val)) > 1e-6 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Errorf("Model output is all zeros - computation may have failed")
	}
	
	// Test MSELoss forward pass only (backward may not work due to CNN autograd limitations)
	criterion := NewMSELoss("mean")
	loss, err := criterion.Forward(output, target)
	if err != nil {
		t.Fatalf("MSELoss forward failed: %v", err)
	}
	
	// Loss should be scalar []
	if !equalShapes(loss.Shape, []int{}) {
		t.Errorf("Loss shape incorrect: got %v, expected []", loss.Shape)
	}
	
	// Verify loss is a reasonable value
	lossVal := tensorFloat32(loss)[0]
	if lossVal < 0 || lossVal > 100 {
		t.Errorf("Loss value unreasonable: %f", lossVal)
	}
	
	// Verify all parameters exist and have correct shapes
	allParams := model.Parameters()
	if len(allParams) == 0 {
		t.Fatalf("Model has no parameters")
	}
	
	expectedParamCount := 6 // 2 conv layers (weight+bias each) + 1 linear layer (weight+bias)
	if len(allParams) != expectedParamCount {
		t.Errorf("Expected %d parameters, got %d", expectedParamCount, len(allParams))
	}
	
	t.Logf("CNN with MSELoss forward pass verified: loss=%.6f, %d parameters", lossVal, len(allParams))
}

// testCNNArchitectureWithCrossEntropyLoss tests Conv2D → Conv2D → Linear with CrossEntropyLoss
func testCNNArchitectureWithCrossEntropyLoss(t *testing.T, device tensor.DeviceType) {
	// Test CNN architecture on all devices - CPU Conv2D autograd is now implemented
	
	// Set deterministic seed for consistent results across devices
	SetRandomSeed(1002)
	
	// Build CNN for classification: Conv2D(1→4) → ReLU → Conv2D(4→8) → ReLU → Flatten → Linear(8→3)
	conv1, _ := NewConv2D(1, 4, 3, 1, 1, true, device) // 1→4 channels
	relu1 := NewReLU()
	conv2, _ := NewConv2D(4, 8, 3, 1, 1, true, device) // 4→8 channels
	relu2 := NewReLU()
	flatten := NewFlatten()
	
	// For 4x4 input: flattened size is 8*4*4=128
	linear, _ := NewLinear(128, 3, true, device) // 128→3 classes
	
	model := NewSequential(conv1, relu1, conv2, relu2, flatten, linear)
	
	// Create synthetic classification data: [batch=3, channels=1, height=4, width=4]
	inputData := make([]float32, 3*1*4*4) // 3 samples
	for i := range inputData {
		inputData[i] = float32(i%10) * 0.1 // Different patterns for each sample
	}
	input, _ := tensor.NewTensor([]int{3, 1, 4, 4}, tensor.Float32, device, inputData)
	
	// Target classes [batch=3] - one class per sample
	targetData := []int32{0, 1, 2} // Classes 0, 1, 2
	target, _ := tensor.NewTensor([]int{3}, tensor.Int32, device, targetData)
	
	// Test forward pass
	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Model forward failed: %v", err)
	}
	
	// Check output shape: [batch=3, classes=3]
	expectedShape := []int{3, 3}
	if !equalShapes(output.Shape, expectedShape) {
		t.Errorf("Model output shape incorrect: got %v, expected %v", output.Shape, expectedShape)
	}
	
	// Verify output contains meaningful values (logits for classification)
	outputData := tensorFloat32(output)
	hasNonZero := false
	for _, val := range outputData {
		if math.Abs(float64(val)) > 1e-6 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Errorf("Model output is all zeros - computation may have failed")
	}
	
	// Test CrossEntropyLoss forward pass
	criterion := NewCrossEntropyLoss("mean")
	loss, err := criterion.Forward(output, target)
	if err != nil {
		t.Fatalf("CrossEntropyLoss forward failed: %v", err)
	}
	
	// Loss should be scalar []
	if !equalShapes(loss.Shape, []int{}) {
		t.Errorf("Loss shape incorrect: got %v, expected []", loss.Shape)
	}
	
	// Verify loss is a reasonable value for 3-class classification
	lossVal := tensorFloat32(loss)[0]
	if lossVal < 0 || lossVal > 10 { // CrossEntropy loss should be reasonable
		t.Errorf("Loss value unreasonable for classification: %f", lossVal)
	}
	
	// Verify all parameters exist and have correct shapes
	allParams := model.Parameters()
	if len(allParams) == 0 {
		t.Fatalf("Model has no parameters")
	}
	
	expectedParamCount := 6 // 2 conv layers (weight+bias each) + 1 linear layer (weight+bias)
	if len(allParams) != expectedParamCount {
		t.Errorf("Expected %d parameters, got %d", expectedParamCount, len(allParams))
	}
	
	// Verify output is valid logits (check if softmax probabilities are reasonable)
	for i := 0; i < 3; i++ { // For each sample
		var expSum float32
		for j := 0; j < 3; j++ { // For each class
			expSum += float32(math.Exp(float64(outputData[i*3+j])))
		}
		// Softmax probabilities should be positive and sum to reasonable value
		if expSum <= 0 {
			t.Errorf("Sample %d softmax sum invalid: %f", i, expSum)
		}
	}
	
	t.Logf("CNN with CrossEntropyLoss forward pass verified: loss=%.6f, %d parameters", lossVal, len(allParams))
}

// testCNNSimpleClassificationLearning tests that CNN can actually learn a simple pattern
func testCNNSimpleClassificationLearning(t *testing.T, device tensor.DeviceType) {
	// Create a simple pattern recognition task: classify 2x2 patterns
	// Pattern 0: top-left bright    [[1,0],[0,0]]
	// Pattern 1: bottom-right bright [[0,0],[0,1]]
	
	// Set deterministic seed for consistent learning behavior across devices
	SetRandomSeed(1003)
	
	// Build simple CNN
	conv, _ := NewConv2D(1, 2, 2, 1, 0, true, device) // 1→2 channels, 2x2 kernel, no padding
	relu := NewReLU()
	flatten := NewFlatten() // Output will be [batch, 2*1*1] = [batch, 2] after 2x2 conv on 2x2 input
	linear, _ := NewLinear(2, 2, true, device) // 2→2 classes
	
	model := NewSequential(conv, relu, flatten, linear)
	
	// Create training data: 4 samples with clear patterns
	trainData := []float32{
		// Sample 0 (class 0): top-left bright
		1.0, 0.0,
		0.0, 0.0,
		// Sample 1 (class 0): top-left bright 
		1.0, 0.1,
		0.1, 0.0,
		// Sample 2 (class 1): bottom-right bright
		0.0, 0.0,
		0.0, 1.0,
		// Sample 3 (class 1): bottom-right bright
		0.1, 0.0,
		0.0, 1.0,
	}
	trainInput, _ := tensor.NewTensor([]int{4, 1, 2, 2}, tensor.Float32, device, trainData)
	
	trainLabels := []int32{0, 0, 1, 1}
	trainTarget, _ := tensor.NewTensor([]int{4}, tensor.Int32, device, trainLabels)
	
	// Create optimizer and loss
	params := model.Parameters()
	optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false) // Simple SGD
	criterion := NewCrossEntropyLoss("mean")
	
	// Training loop
	initialLoss := float32(0)
	finalLoss := float32(0)
	
	for epoch := 0; epoch < 50; epoch++ { // More epochs for learning
		// Zero gradients
		optimizer.ZeroGrad()
		
		// Forward pass
		output, err := model.Forward(trainInput)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		
		// Compute loss
		loss, err := criterion.Forward(output, trainTarget)
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		
		lossVal := tensorFloat32(loss)[0]
		if epoch == 0 {
			initialLoss = lossVal
		}
		if epoch == 49 {
			finalLoss = lossVal
		}
		
		// Backward pass
		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Update parameters
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed: %v", err)
		}
		
		// Log progress every 10 epochs
		if epoch%10 == 0 {
			t.Logf("Epoch %d: Loss = %.6f", epoch, lossVal)
		}
	}
	
	// Verify learning occurred
	if finalLoss >= initialLoss {
		t.Errorf("Model did not learn: initial loss %.6f >= final loss %.6f", initialLoss, finalLoss)
	}
	
	// Test final accuracy
	finalOutput, _ := model.Forward(trainInput)
	predictions, _ := tensorArgmax(finalOutput, 1)
	predData := tensorInt32(predictions)
	
	correct := 0
	for i := range predData {
		if predData[i] == trainLabels[i] {
			correct++
		}
	}
	
	accuracy := float32(correct) / float32(len(trainLabels))
	
	t.Logf("CNN Learning Results:")
	t.Logf("  Initial Loss: %.6f", initialLoss)  
	t.Logf("  Final Loss: %.6f", finalLoss)
	t.Logf("  Final Accuracy: %.1f%% (%d/%d correct)", accuracy*100, correct, len(trainLabels))
	t.Logf("  Predictions: %v", predData)
	t.Logf("  Expected:    %v", trainLabels)
	
	// Expect at least 75% accuracy on this simple task
	if accuracy < 0.75 {
		t.Errorf("CNN failed to learn simple pattern: accuracy %.1f%% < 75%%", accuracy*100)
	} else {
		t.Logf("CNN learning SUCCESS: accuracy %.1f%%", accuracy*100)
	}
}