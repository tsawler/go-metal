package training

import (
	"math"
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

func TestSGDOptimizer(t *testing.T) {
	t.Run("Basic SGD update", func(t *testing.T) {
		// Create a simple parameter tensor
		data := []float32{1.0, 2.0, 3.0}
		param, err := tensor.NewTensor([]int{3}, tensor.Float32, tensor.CPU, data)
		if err != nil {
			t.Fatalf("Failed to create parameter tensor: %v", err)
		}
		param.SetRequiresGrad(true)
		
		// Set gradient
		gradData := []float32{0.1, 0.2, 0.3}
		grad, err := tensor.NewTensor([]int{3}, tensor.Float32, tensor.CPU, gradData)
		if err != nil {
			t.Fatalf("Failed to create gradient tensor: %v", err)
		}
		param.SetGrad(grad)
		
		// Create SGD optimizer
		params := []*tensor.Tensor{param}
		optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
		
		// Perform one step
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("SGD step failed: %v", err)
		}
		
		// Check updated parameters: new_param = old_param - lr * grad
		expectedData := []float32{0.99, 1.98, 2.97} // 1.0 - 0.1*0.1, 2.0 - 0.1*0.2, 3.0 - 0.1*0.3
		actualData := param.Data.([]float32)
		
		for i, expected := range expectedData {
			if math.Abs(float64(actualData[i]-expected)) > 1e-6 {
				t.Errorf("Parameter %d: expected %.6f, got %.6f", i, expected, actualData[i])
			}
		}
	})
	
	t.Run("SGD with momentum", func(t *testing.T) {
		// Create parameter tensor
		data := []float32{1.0, 2.0}
		param, err := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, data)
		if err != nil {
			t.Fatalf("Failed to create parameter tensor: %v", err)
		}
		param.SetRequiresGrad(true)
		
		// Create SGD optimizer with momentum
		params := []*tensor.Tensor{param}
		optimizer := NewSGD(params, 0.1, 0.9, 0.0, 0.0, false)
		
		// First step
		gradData1 := []float32{0.1, 0.2}
		grad1, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, gradData1)
		param.SetGrad(grad1)
		
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("First SGD step failed: %v", err)
		}
		
		// Second step with different gradient
		gradData2 := []float32{0.2, 0.1}
		grad2, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, gradData2)
		param.SetGrad(grad2)
		
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Second SGD step failed: %v", err)
		}
		
		// Parameters should reflect momentum effects
		actualData := param.Data.([]float32)
		
		// After momentum, we expect different values than simple SGD
		if actualData[0] >= 0.98 || actualData[1] >= 1.97 {
			t.Error("Momentum doesn't appear to be working correctly")
		}
	})
}

func TestAdamOptimizer(t *testing.T) {
	t.Run("Basic Adam update", func(t *testing.T) {
		// Create a simple parameter tensor
		data := []float32{1.0, 2.0}
		param, err := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, data)
		if err != nil {
			t.Fatalf("Failed to create parameter tensor: %v", err)
		}
		param.SetRequiresGrad(true)
		
		// Set gradient
		gradData := []float32{0.1, 0.2}
		grad, err := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, gradData)
		if err != nil {
			t.Fatalf("Failed to create gradient tensor: %v", err)
		}
		param.SetGrad(grad)
		
		// Create Adam optimizer
		params := []*tensor.Tensor{param}
		optimizer := NewAdam(params, 0.001, 0.9, 0.999, 1e-8, 0.0)
		
		// Perform one step
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Adam step failed: %v", err)
		}
		
		// Check that parameters were updated
		actualData := param.Data.([]float32)
		
		// Parameters should be different from initial values
		if actualData[0] == 1.0 || actualData[1] == 2.0 {
			t.Error("Adam optimizer didn't update parameters")
		}
		
		// Parameters should be smaller than initial (gradient descent)
		if actualData[0] >= 1.0 || actualData[1] >= 2.0 {
			t.Error("Adam optimizer didn't decrease parameters as expected")
		}
	})
	
	t.Run("Adam with multiple steps", func(t *testing.T) {
		// Create parameter tensor
		data := []float32{1.0}
		param, err := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, data)
		if err != nil {
			t.Fatalf("Failed to create parameter tensor: %v", err)
		}
		param.SetRequiresGrad(true)
		
		// Create Adam optimizer
		params := []*tensor.Tensor{param}
		optimizer := NewAdam(params, 0.01, 0.9, 0.999, 1e-8, 0.0)
		
		// Perform multiple steps with consistent gradient
		for i := 0; i < 10; i++ {
			gradData := []float32{0.1}
			grad, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, gradData)
			param.SetGrad(grad)
			
			err = optimizer.Step()
			if err != nil {
				t.Fatalf("Adam step %d failed: %v", i, err)
			}
		}
		
		// After multiple steps with consistent positive gradient, parameter should decrease
		actualData := param.Data.([]float32)
		if actualData[0] >= 1.0 {
			t.Errorf("After 10 steps, parameter should be smaller than initial value 1.0, got %.6f", actualData[0])
		}
	})
}

func TestOptimizerLearningRate(t *testing.T) {
	t.Run("SGD learning rate getter/setter", func(t *testing.T) {
		param, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{1.0})
		params := []*tensor.Tensor{param}
		
		optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
		
		// Test getter
		if optimizer.GetLR() != 0.1 {
			t.Errorf("Expected learning rate 0.1, got %f", optimizer.GetLR())
		}
		
		// Test setter
		optimizer.SetLR(0.01)
		if optimizer.GetLR() != 0.01 {
			t.Errorf("Expected learning rate 0.01 after setting, got %f", optimizer.GetLR())
		}
	})
	
	t.Run("Adam learning rate getter/setter", func(t *testing.T) {
		param, _ := tensor.NewTensor([]int{1}, tensor.Float32, tensor.CPU, []float32{1.0})
		params := []*tensor.Tensor{param}
		
		optimizer := NewAdam(params, 0.001, 0.9, 0.999, 1e-8, 0.0)
		
		// Test getter
		if optimizer.GetLR() != 0.001 {
			t.Errorf("Expected learning rate 0.001, got %f", optimizer.GetLR())
		}
		
		// Test setter
		optimizer.SetLR(0.0001)
		if optimizer.GetLR() != 0.0001 {
			t.Errorf("Expected learning rate 0.0001 after setting, got %f", optimizer.GetLR())
		}
	})
}

func TestOptimizerZeroGrad(t *testing.T) {
	t.Run("ZeroGrad functionality", func(t *testing.T) {
		// Create parameters with gradients
		param1, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		param2, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{3.0, 4.0})
		param1.SetRequiresGrad(true)
		param2.SetRequiresGrad(true)
		
		// Set gradients
		grad1, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{0.1, 0.2})
		grad2, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{0.3, 0.4})
		param1.SetGrad(grad1)
		param2.SetGrad(grad2)
		
		// Create optimizer
		params := []*tensor.Tensor{param1, param2}
		optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
		
		// Zero gradients
		optimizer.ZeroGrad()
		
		// Check that gradients are zeroed
		if param1.Grad() != nil {
			grad1Data := param1.Grad().Data.([]float32)
			for i, val := range grad1Data {
				if val != 0.0 {
					t.Errorf("Gradient for param1[%d] should be 0, got %f", i, val)
				}
			}
		}
		
		if param2.Grad() != nil {
			grad2Data := param2.Grad().Data.([]float32)
			for i, val := range grad2Data {
				if val != 0.0 {
					t.Errorf("Gradient for param2[%d] should be 0, got %f", i, val)
				}
			}
		}
	})
}