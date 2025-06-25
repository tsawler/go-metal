package training

import (
	"math"
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

func TestMSELoss(t *testing.T) {
	t.Run("Basic MSE computation", func(t *testing.T) {
		// Create predicted and target tensors
		predicted, err := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0, 3.0, 4.0})
		if err != nil {
			t.Fatalf("Failed to create predicted tensor: %v", err)
		}
		
		target, err := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, []float32{1.5, 2.5, 2.5, 3.5})
		if err != nil {
			t.Fatalf("Failed to create target tensor: %v", err)
		}
		
		// Create MSE loss
		mse := NewMSELoss("mean")
		
		// Compute loss
		loss, err := mse.Forward(predicted, target)
		if err != nil {
			t.Fatalf("MSE forward failed: %v", err)
		}
		
		// Expected: ((1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-2.5)^2 + (4.0-3.5)^2) / 4 
		//         = (0.25 + 0.25 + 0.25 + 0.25) / 4 = 0.25
		expectedLoss := float32(0.25)
		actualLoss := loss.Data.([]float32)[0]
		
		if math.Abs(float64(actualLoss-expectedLoss)) > 1e-6 {
			t.Errorf("Expected loss %.6f, got %.6f", expectedLoss, actualLoss)
		}
	})
	
	t.Run("MSE backward pass", func(t *testing.T) {
		predicted, _ := tensor.NewTensor([]int{1, 2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		target, _ := tensor.NewTensor([]int{1, 2}, tensor.Float32, tensor.CPU, []float32{1.5, 1.5})
		
		mse := NewMSELoss("mean")
		
		// Compute gradient
		grad, err := mse.Backward(predicted, target)
		if err != nil {
			t.Fatalf("MSE backward failed: %v", err)
		}
		
		// Expected gradient: 2 * (predicted - target) / N
		// For predicted=[1.0, 2.0], target=[1.5, 1.5], N=2
		// grad = 2 * ([1.0, 2.0] - [1.5, 1.5]) / 2 = 2 * [-0.5, 0.5] / 2 = [-0.5, 0.5]
		expectedGrad := []float32{-0.5, 0.5}
		actualGrad := grad.Data.([]float32)
		
		for i, expected := range expectedGrad {
			if math.Abs(float64(actualGrad[i]-expected)) > 1e-6 {
				t.Errorf("Gradient[%d]: expected %.6f, got %.6f", i, expected, actualGrad[i])
			}
		}
	})
	
	t.Run("MSE with sum reduction", func(t *testing.T) {
		predicted, _ := tensor.NewTensor([]int{2, 1}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		target, _ := tensor.NewTensor([]int{2, 1}, tensor.Float32, tensor.CPU, []float32{0.0, 0.0})
		
		mse := NewMSELoss("sum")
		
		loss, err := mse.Forward(predicted, target)
		if err != nil {
			t.Fatalf("MSE forward with sum reduction failed: %v", err)
		}
		
		// Expected: (1.0-0.0)^2 + (2.0-0.0)^2 = 1.0 + 4.0 = 5.0 (no division by N)
		expectedLoss := float32(5.0)
		actualLoss := loss.Data.([]float32)[0]
		
		if math.Abs(float64(actualLoss-expectedLoss)) > 1e-6 {
			t.Errorf("Expected loss %.6f, got %.6f", expectedLoss, actualLoss)
		}
	})
}

func TestCrossEntropyLoss(t *testing.T) {
	t.Run("Basic CrossEntropy computation", func(t *testing.T) {
		// Create logits for 2 samples, 3 classes
		logits, err := tensor.NewTensor([]int{2, 3}, tensor.Float32, tensor.CPU, 
			[]float32{1.0, 2.0, 3.0, 0.5, 1.5, 0.1})
		if err != nil {
			t.Fatalf("Failed to create logits tensor: %v", err)
		}
		
		// Target classes
		target, err := tensor.NewTensor([]int{2}, tensor.Int32, tensor.CPU, []int32{2, 1})
		if err != nil {
			t.Fatalf("Failed to create target tensor: %v", err)
		}
		
		// Create CrossEntropy loss
		ce := NewCrossEntropyLoss("mean")
		
		// Compute loss
		loss, err := ce.Forward(logits, target)
		if err != nil {
			t.Fatalf("CrossEntropy forward failed: %v", err)
		}
		
		// Loss should be positive
		actualLoss := loss.Data.([]float32)[0]
		if actualLoss <= 0 {
			t.Errorf("CrossEntropy loss should be positive, got %.6f", actualLoss)
		}
	})
	
	t.Run("CrossEntropy backward pass", func(t *testing.T) {
		// Simple case: 1 sample, 2 classes
		logits, _ := tensor.NewTensor([]int{1, 2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		target, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{1})
		
		ce := NewCrossEntropyLoss("mean")
		
		// Compute gradient
		grad, err := ce.Backward(logits, target)
		if err != nil {
			t.Fatalf("CrossEntropy backward failed: %v", err)
		}
		
		// Check gradient shape
		if len(grad.Shape) != 2 || grad.Shape[0] != 1 || grad.Shape[1] != 2 {
			t.Errorf("Expected gradient shape [1, 2], got %v", grad.Shape)
		}
		
		// Gradient should be softmax probabilities minus one-hot target
		gradData := grad.Data.([]float32)
		
		// For target class, gradient should be negative (softmax_prob - 1)
		if gradData[1] >= 0 {
			t.Errorf("Gradient for target class should be negative, got %.6f", gradData[1])
		}
		
		// For non-target class, gradient should be positive (softmax_prob - 0)
		if gradData[0] <= 0 {
			t.Errorf("Gradient for non-target class should be positive, got %.6f", gradData[0])
		}
	})
	
	t.Run("CrossEntropy with invalid inputs", func(t *testing.T) {
		// Wrong shape logits
		logits, _ := tensor.NewTensor([]int{2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0})
		target, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{0})
		
		ce := NewCrossEntropyLoss("mean")
		
		_, err := ce.Forward(logits, target)
		if err == nil {
			t.Error("Expected error for 1D logits tensor")
		}
		
		// Wrong dtype
		logits2, _ := tensor.NewTensor([]int{1, 2}, tensor.Int32, tensor.CPU, []int32{1, 2})
		target2, _ := tensor.NewTensor([]int{1}, tensor.Int32, tensor.CPU, []int32{0})
		
		_, err = ce.Forward(logits2, target2)
		if err == nil {
			t.Error("Expected error for Int32 logits tensor")
		}
	})
}

func TestSoftmax(t *testing.T) {
	t.Run("Softmax numerical properties", func(t *testing.T) {
		// Create logits
		logits, _ := tensor.NewTensor([]int{2, 3}, tensor.Float32, tensor.CPU, 
			[]float32{1.0, 2.0, 3.0, 0.0, 1.0, 2.0})
		
		ce := NewCrossEntropyLoss("mean")
		
		// Apply softmax
		probs, err := ce.softmax(logits)
		if err != nil {
			t.Fatalf("Softmax failed: %v", err)
		}
		
		probsData := probs.Data.([]float32)
		
		// Check that probabilities sum to 1 for each sample
		for i := 0; i < 2; i++ {
			sum := float32(0.0)
			for j := 0; j < 3; j++ {
				prob := probsData[i*3+j]
				sum += prob
				
				// Each probability should be positive
				if prob <= 0 {
					t.Errorf("Probability should be positive, got %.6f at [%d, %d]", prob, i, j)
				}
			}
			
			// Sum should be approximately 1
			if math.Abs(float64(sum-1.0)) > 1e-6 {
				t.Errorf("Probabilities should sum to 1, got %.6f for sample %d", sum, i)
			}
		}
	})
}

func TestLossReduction(t *testing.T) {
	t.Run("Mean vs Sum reduction", func(t *testing.T) {
		predicted, _ := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, []float32{1.0, 2.0, 3.0, 4.0})
		target, _ := tensor.NewTensor([]int{2, 2}, tensor.Float32, tensor.CPU, []float32{0.0, 0.0, 0.0, 0.0})
		
		mseMean := NewMSELoss("mean")
		mseSum := NewMSELoss("sum")
		
		lossMean, err := mseMean.Forward(predicted, target)
		if err != nil {
			t.Fatalf("MSE mean forward failed: %v", err)
		}
		
		lossSum, err := mseSum.Forward(predicted, target)
		if err != nil {
			t.Fatalf("MSE sum forward failed: %v", err)
		}
		
		meanLoss := lossMean.Data.([]float32)[0]
		sumLoss := lossSum.Data.([]float32)[0]
		
		// Sum loss should be N times the mean loss (N=4 in this case)
		expectedRatio := float32(4.0)
		actualRatio := sumLoss / meanLoss
		
		if math.Abs(float64(actualRatio-expectedRatio)) > 1e-6 {
			t.Errorf("Expected ratio %.6f, got %.6f", expectedRatio, actualRatio)
		}
	})
}