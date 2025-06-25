package training

import (
	"math"
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

func TestLinearModule(t *testing.T) {
	t.Run("Linear layer forward pass", func(t *testing.T) {
		// Create Linear layer: 3 input features -> 2 output features
		linear, err := NewLinear(3, 2, true, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create Linear layer: %v", err)
		}
		
		// Create input: batch_size=2, input_features=3
		input, err := tensor.NewTensor([]int{2, 3}, tensor.Float32, tensor.CPU, 
			[]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}
		
		// Forward pass
		output, err := linear.Forward(input)
		if err != nil {
			t.Fatalf("Linear forward pass failed: %v", err)
		}
		
		// Check output shape: [batch_size, output_features] = [2, 2]
		expectedShape := []int{2, 2}
		if len(output.Shape) != len(expectedShape) {
			t.Fatalf("Expected output shape %v, got %v", expectedShape, output.Shape)
		}
		
		for i, dim := range expectedShape {
			if output.Shape[i] != dim {
				t.Errorf("Output shape dimension %d: expected %d, got %d", i, dim, output.Shape[i])
			}
		}
		
		// Check that output contains valid values (not NaN or Inf)
		outputData := output.Data.([]float32)
		for i, val := range outputData {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Errorf("Output[%d] is invalid: %f", i, val)
			}
		}
	})
	
	t.Run("Linear layer without bias", func(t *testing.T) {
		linear, err := NewLinear(2, 1, false, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create Linear layer without bias: %v", err)
		}
		
		// Check that bias is nil
		if linear.bias != nil {
			t.Error("Linear layer without bias should have nil bias tensor")
		}
		
		// Forward pass should still work
		input, _ := tensor.NewTensor([]int{1, 2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		output, err := linear.Forward(input)
		if err != nil {
			t.Fatalf("Linear forward pass without bias failed: %v", err)
		}
		
		// Check output shape
		if len(output.Shape) != 2 || output.Shape[0] != 1 || output.Shape[1] != 1 {
			t.Errorf("Expected output shape [1, 1], got %v", output.Shape)
		}
	})
	
	t.Run("Linear layer parameters", func(t *testing.T) {
		linear, _ := NewLinear(3, 2, true, tensor.CPU)
		
		params := linear.Parameters()
		
		// Should have weight and bias
		if len(params) != 2 {
			t.Fatalf("Expected 2 parameters (weight and bias), got %d", len(params))
		}
		
		// Check weight shape: [output_features, input_features] = [2, 3]
		weight := params[0]
		if len(weight.Shape) != 2 || weight.Shape[0] != 2 || weight.Shape[1] != 3 {
			t.Errorf("Expected weight shape [2, 3], got %v", weight.Shape)
		}
		
		// Check bias shape: [output_features] = [2]
		bias := params[1]
		if len(bias.Shape) != 1 || bias.Shape[0] != 2 {
			t.Errorf("Expected bias shape [2], got %v", bias.Shape)
		}
		
		// Check that parameters require gradients
		if !weight.RequiresGrad() {
			t.Error("Weight should require gradients")
		}
		if !bias.RequiresGrad() {
			t.Error("Bias should require gradients")
		}
	})
}

func TestReLUModule(t *testing.T) {
	t.Run("ReLU forward pass", func(t *testing.T) {
		relu := NewReLU()
		
		// Create input with positive and negative values
		input, err := tensor.NewTensor([]int{2, 3}, tensor.Float32, tensor.CPU, 
			[]float32{-1.0, 0.0, 1.0, -2.0, 3.0, -0.5})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}
		
		// Forward pass
		output, err := relu.Forward(input)
		if err != nil {
			t.Fatalf("ReLU forward pass failed: %v", err)
		}
		
		// Check output values: negative values should be 0, positive values unchanged
		expectedData := []float32{0.0, 0.0, 1.0, 0.0, 3.0, 0.0}
		outputData := output.Data.([]float32)
		
		for i, expected := range expectedData {
			if math.Abs(float64(outputData[i]-expected)) > 1e-6 {
				t.Errorf("Output[%d]: expected %.6f, got %.6f", i, expected, outputData[i])
			}
		}
	})
	
	t.Run("ReLU has no parameters", func(t *testing.T) {
		relu := NewReLU()
		
		params := relu.Parameters()
		if len(params) != 0 {
			t.Errorf("ReLU should have no parameters, got %d", len(params))
		}
	})
}

func TestConv2DModule(t *testing.T) {
	t.Run("Conv2D layer creation", func(t *testing.T) {
		// Create Conv2D layer: 3 input channels, 16 output channels, 3x3 kernel
		conv, err := NewConv2D(3, 16, 3, 1, 1, true, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create Conv2D layer: %v", err)
		}
		
		// Check parameters
		params := conv.Parameters()
		if len(params) != 2 {
			t.Fatalf("Expected 2 parameters (weight and bias), got %d", len(params))
		}
		
		// Check weight shape: [output_channels, input_channels, kernel_height, kernel_width]
		weight := params[0]
		expectedWeightShape := []int{16, 3, 3, 3}
		if len(weight.Shape) != len(expectedWeightShape) {
			t.Fatalf("Expected weight shape %v, got %v", expectedWeightShape, weight.Shape)
		}
		
		for i, dim := range expectedWeightShape {
			if weight.Shape[i] != dim {
				t.Errorf("Weight shape dimension %d: expected %d, got %d", i, dim, weight.Shape[i])
			}
		}
		
		// Check bias shape: [output_channels]
		bias := params[1]
		if len(bias.Shape) != 1 || bias.Shape[0] != 16 {
			t.Errorf("Expected bias shape [16], got %v", bias.Shape)
		}
	})
	
	t.Run("Conv2D without bias", func(t *testing.T) {
		conv, err := NewConv2D(1, 1, 3, 1, 0, false, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create Conv2D layer without bias: %v", err)
		}
		
		// Should have only weight parameter
		params := conv.Parameters()
		if len(params) != 1 {
			t.Errorf("Conv2D without bias should have 1 parameter, got %d", len(params))
		}
		
		if conv.bias != nil {
			t.Error("Conv2D without bias should have nil bias tensor")
		}
	})
}

func TestBatchNormModule(t *testing.T) {
	t.Run("BatchNorm layer creation", func(t *testing.T) {
		bn, err := NewBatchNorm(10, 1e-5, 0.1, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create BatchNorm layer: %v", err)
		}
		
		// Check parameters
		params := bn.Parameters()
		if len(params) != 2 {
			t.Fatalf("Expected 2 parameters (gamma and beta), got %d", len(params))
		}
		
		// Check gamma and beta shapes
		gamma := params[0]
		beta := params[1]
		
		if len(gamma.Shape) != 1 || gamma.Shape[0] != 10 {
			t.Errorf("Expected gamma shape [10], got %v", gamma.Shape)
		}
		
		if len(beta.Shape) != 1 || beta.Shape[0] != 10 {
			t.Errorf("Expected beta shape [10], got %v", beta.Shape)
		}
		
		// Check initial values
		gammaData := gamma.Data.([]float32)
		betaData := beta.Data.([]float32)
		
		// Gamma should be initialized to 1.0
		for i, val := range gammaData {
			if math.Abs(float64(val-1.0)) > 1e-6 {
				t.Errorf("Gamma[%d] should be 1.0, got %.6f", i, val)
			}
		}
		
		// Beta should be initialized to 0.0
		for i, val := range betaData {
			if math.Abs(float64(val)) > 1e-6 {
				t.Errorf("Beta[%d] should be 0.0, got %.6f", i, val)
			}
		}
	})
	
	t.Run("BatchNorm forward pass", func(t *testing.T) {
		bn, err := NewBatchNorm(2, 1e-5, 0.1, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create BatchNorm layer: %v", err)
		}
		
		// Create input: batch_size=3, features=2
		input, err := tensor.NewTensor([]int{3, 2}, tensor.Float32, tensor.CPU, 
			[]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}
		
		// Forward pass in training mode
		bn.Train()
		output, err := bn.Forward(input)
		if err != nil {
			t.Fatalf("BatchNorm forward pass failed: %v", err)
		}
		
		// Check output shape
		if len(output.Shape) != 2 || output.Shape[0] != 3 || output.Shape[1] != 2 {
			t.Errorf("Expected output shape [3, 2], got %v", output.Shape)
		}
		
		// In training mode with batch normalization, output should have approximately zero mean and unit variance
		outputData := output.Data.([]float32)
		
		// Check that values are reasonable (not NaN or Inf)
		for i, val := range outputData {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Errorf("Output[%d] is invalid: %f", i, val)
			}
		}
	})
	
	t.Run("BatchNorm eval mode", func(t *testing.T) {
		bn, err := NewBatchNorm(2, 1e-5, 0.1, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create BatchNorm layer: %v", err)
		}
		
		// Set to eval mode
		bn.Eval()
		
		if bn.IsTraining() {
			t.Error("BatchNorm should be in eval mode")
		}
		
		// Create input
		input, _ := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, 
			[]float32{1.0, 2.0, 3.0, 4.0})
		
		// Forward pass should work in eval mode
		output, err := bn.Forward(input)
		if err != nil {
			t.Fatalf("BatchNorm forward pass in eval mode failed: %v", err)
		}
		
		// Check output shape
		if len(output.Shape) != 2 || output.Shape[0] != 2 || output.Shape[1] != 2 {
			t.Errorf("Expected output shape [2, 2], got %v", output.Shape)
		}
	})
}

func TestSequentialModule(t *testing.T) {
	t.Run("Sequential forward pass", func(t *testing.T) {
		// Create a simple sequential model: Linear -> ReLU -> Linear
		linear1, err := NewLinear(3, 5, true, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create first linear layer: %v", err)
		}
		
		relu := NewReLU()
		
		linear2, err := NewLinear(5, 2, true, tensor.CPU)
		if err != nil {
			t.Fatalf("Failed to create second linear layer: %v", err)
		}
		
		model := NewSequential(linear1, relu, linear2)
		
		// Create input
		input, err := tensor.NewTensor([]int{1, 3}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0, 3.0})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}
		
		// Forward pass
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Sequential forward pass failed: %v", err)
		}
		
		// Check output shape: [batch_size, final_output_features] = [1, 2]
		if len(output.Shape) != 2 || output.Shape[0] != 1 || output.Shape[1] != 2 {
			t.Errorf("Expected output shape [1, 2], got %v", output.Shape)
		}
	})
	
	t.Run("Sequential parameters", func(t *testing.T) {
		linear1, _ := NewLinear(2, 3, true, tensor.CPU)
		relu := NewReLU()
		linear2, _ := NewLinear(3, 1, false, tensor.CPU)
		
		model := NewSequential(linear1, relu, linear2)
		
		params := model.Parameters()
		
		// Should have parameters from both linear layers: weight1, bias1, weight2
		// ReLU has no parameters
		expectedParamCount := 3 // weight1, bias1, weight2 (linear2 has no bias)
		if len(params) != expectedParamCount {
			t.Errorf("Expected %d parameters, got %d", expectedParamCount, len(params))
		}
	})
	
	t.Run("Sequential train/eval mode", func(t *testing.T) {
		linear, _ := NewLinear(2, 2, true, tensor.CPU)
		bn, _ := NewBatchNorm(2, 1e-5, 0.1, tensor.CPU)
		
		model := NewSequential(linear, bn)
		
		// Test train mode
		model.Train()
		if !model.IsTraining() {
			t.Error("Sequential model should be in training mode")
		}
		if !linear.IsTraining() {
			t.Error("Linear layer should be in training mode")
		}
		if !bn.IsTraining() {
			t.Error("BatchNorm layer should be in training mode")
		}
		
		// Test eval mode
		model.Eval()
		if model.IsTraining() {
			t.Error("Sequential model should be in eval mode")
		}
		if linear.IsTraining() {
			t.Error("Linear layer should be in eval mode")
		}
		if bn.IsTraining() {
			t.Error("BatchNorm layer should be in eval mode")
		}
	})
	
	t.Run("Sequential Add method", func(t *testing.T) {
		model := NewSequential()
		
		// Initially should have no modules
		if len(model.modules) != 0 {
			t.Error("New sequential model should have no modules")
		}
		
		// Add modules
		linear, _ := NewLinear(2, 2, true, tensor.CPU)
		model.Add(linear)
		
		if len(model.modules) != 1 {
			t.Error("Sequential model should have 1 module after adding")
		}
		
		relu := NewReLU()
		model.Add(relu)
		
		if len(model.modules) != 2 {
			t.Error("Sequential model should have 2 modules after adding")
		}
	})
}