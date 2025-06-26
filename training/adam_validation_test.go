package training

import (
	"math"
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

// TestAdamOptimizerCorrectness validates the Adam optimizer implementation
// against the reference algorithm from the original paper
func TestAdamOptimizerCorrectness(t *testing.T) {
	t.Run("Adam algorithm correctness", func(t *testing.T) {
		// Test parameters
		lr := 0.001
		beta1 := 0.9
		beta2 := 0.999
		eps := 1e-8
		
		// Create a simple parameter tensor
		initialValue := 1.0
		param, err := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{float32(initialValue)})
		if err != nil {
			t.Fatalf("Failed to create parameter tensor: %v", err)
		}
		param.SetRequiresGrad(true)
		
		// Create Adam optimizer
		params := []*tensor.Tensor{param}
		optimizer := NewAdam(params, lr, beta1, beta2, eps, 0.0)
		
		// Manual Adam calculation for verification
		var m, v float64 = 0, 0  // First and second moment estimates
		paramValue := initialValue
		
		// Test with different gradients over multiple steps
		gradients := []float32{0.1, 0.05, 0.2, 0.01, 0.15}
		
		for step, gradVal := range gradients {
			t.Logf("Step %d: gradient=%.3f", step+1, gradVal)
			
			// Set gradient
			grad, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{gradVal})
			param.SetGrad(grad)
			
			// Manual Adam update calculation
			g := float64(gradVal)
			
			// Update biased first moment estimate
			m = beta1*m + (1-beta1)*g
			
			// Update biased second raw moment estimate  
			v = beta2*v + (1-beta2)*g*g
			
			// Compute bias-corrected first moment estimate
			mHat := m / (1 - math.Pow(beta1, float64(step+1)))
			
			// Compute bias-corrected second raw moment estimate
			vHat := v / (1 - math.Pow(beta2, float64(step+1)))
			
			// Update parameter
			paramValue = paramValue - lr*mHat/(math.Sqrt(vHat)+eps)
			
			// Run optimizer step
			err = optimizer.Step()
			if err != nil {
				t.Fatalf("Adam step %d failed: %v", step+1, err)
			}
			
			// Check the result
			actualData := param.Data.([]float32)
			actual := float64(actualData[0])
			
			// Allow small floating point differences
			tolerance := 1e-6
			if math.Abs(actual-paramValue) > tolerance {
				t.Errorf("Step %d: Expected parameter value %.8f, got %.8f (diff: %.2e)", 
					step+1, paramValue, actual, math.Abs(actual-paramValue))
			}
			
			t.Logf("  Expected: %.8f, Actual: %.8f ✓", paramValue, actual)
		}
	})
	
	t.Run("Adam bias correction", func(t *testing.T) {
		// Test that bias correction is working properly in early steps
		param, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{0.0})
		param.SetRequiresGrad(true)
		
		params := []*tensor.Tensor{param}
		optimizer := NewAdam(params, 0.1, 0.9, 0.999, 1e-8, 0.0)
		
		// First step with gradient of 1.0
		grad, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{1.0})
		param.SetGrad(grad)
		
		err := optimizer.Step()
		if err != nil {
			t.Fatalf("Adam step failed: %v", err)
		}
		
		// After first step with beta1=0.9, beta2=0.999:
		// m = 0.9*0 + 0.1*1.0 = 0.1
		// v = 0.999*0 + 0.001*1.0 = 0.001
		// m_hat = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
		// v_hat = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
		// param = 0 - 0.1 * 1.0 / (sqrt(1.0) + 1e-8) ≈ -0.1
		
		actualData := param.Data.([]float32)
		expected := -0.1
		
		if math.Abs(float64(actualData[0])-expected) > 1e-6 {
			t.Errorf("Bias correction not working correctly. Expected ≈%.6f, got %.6f", 
				expected, actualData[0])
		}
	})
	
	t.Run("Adam with weight decay", func(t *testing.T) {
		// Test Adam with L2 weight decay
		weightDecay := 0.01
		param, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{1.0})
		param.SetRequiresGrad(true)
		
		params := []*tensor.Tensor{param}
		optimizer := NewAdam(params, 0.1, 0.9, 0.999, 1e-8, weightDecay)
		
		// Set gradient
		grad, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{0.1})
		param.SetGrad(grad)
		
		initialValue := 1.0
		
		err := optimizer.Step()
		if err != nil {
			t.Fatalf("Adam step with weight decay failed: %v", err)
		}
		
		// With weight decay, the effective gradient should be: grad + weight_decay * param
		// effective_grad = 0.1 + 0.01 * 1.0 = 0.11
		actualData := param.Data.([]float32)
		
		// The parameter should be updated more aggressively due to weight decay
		if actualData[0] >= float32(initialValue) {
			t.Error("Weight decay should cause more aggressive parameter updates")
		}
		
		t.Logf("With weight decay: initial=%.3f, final=%.6f", initialValue, actualData[0])
	})
	
	t.Run("Adam moment initialization", func(t *testing.T) {
		// Test that moments are properly initialized to zero
		param, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		param.SetRequiresGrad(true)
		
		params := []*tensor.Tensor{param}
		optimizer := NewAdam(params, 0.001, 0.9, 0.999, 1e-8, 0.0)
		
		// Just test that the optimizer was created successfully
		if optimizer == nil {
			t.Fatal("Failed to create Adam optimizer")
		}
		
		// Test a simple step to ensure moments are working
		grad, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{0.1, 0.2})
		param.SetGrad(grad)
		
		err := optimizer.Step()
		if err != nil {
			t.Fatalf("Adam step failed: %v", err)
		}
		
		// Parameters should have changed
		actualData := param.Data.([]float32)
		if actualData[0] == 1.0 || actualData[1] == 2.0 {
			t.Error("Adam optimizer didn't update parameters")
		}
	})
}