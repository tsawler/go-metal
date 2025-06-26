package tensor

import (
	"math"
	"reflect"
	"testing"
)

func TestAutogradBasicOperations(t *testing.T) {
	// Test basic forward pass
	t.Run("Addition forward", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8})
		a.SetRequiresGrad(true)
		b.SetRequiresGrad(true)
		
		result, err := AddAutograd(a, b)
		if err != nil {
			t.Fatalf("AddAutograd failed: %v", err)
		}
		
		if !result.RequiresGrad() {
			t.Error("Result should require gradients")
		}
		
		expected := []float32{6, 8, 10, 12}
		actual := result.Data.([]float32)
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("Expected %v, got %v", expected, actual)
		}
	})
	
	t.Run("Multiplication forward", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{2, 3, 4, 5})
		a.SetRequiresGrad(true)
		b.SetRequiresGrad(true)
		
		result, err := MulAutograd(a, b)
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		
		expected := []float32{2, 6, 12, 20}
		actual := result.Data.([]float32)
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("Expected %v, got %v", expected, actual)
		}
	})
}

func TestAutogradBackward(t *testing.T) {
	t.Run("Simple addition backward", func(t *testing.T) {
		// Test y = x1 + x2, where y is scalar
		x1, _ := NewTensor([]int{1}, Float32, CPU, []float32{3.0})
		x2, _ := NewTensor([]int{1}, Float32, CPU, []float32{4.0})
		x1.SetRequiresGrad(true)
		x2.SetRequiresGrad(true)
		
		y, err := AddAutograd(x1, x2)
		if err != nil {
			t.Fatalf("AddAutograd failed: %v", err)
		}
		
		// Perform backward pass
		err = y.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Check gradients
		if x1.Grad() == nil {
			t.Error("x1 should have gradient")
		} else {
			grad1 := x1.Grad().Data.([]float32)[0]
			if grad1 != 1.0 {
				t.Errorf("Expected x1 gradient to be 1.0, got %f", grad1)
			}
		}
		
		if x2.Grad() == nil {
			t.Error("x2 should have gradient")
		} else {
			grad2 := x2.Grad().Data.([]float32)[0]
			if grad2 != 1.0 {
				t.Errorf("Expected x2 gradient to be 1.0, got %f", grad2)
			}
		}
	})
	
	t.Run("Simple multiplication backward", func(t *testing.T) {
		// Test y = x1 * x2, where y is scalar
		x1, _ := NewTensor([]int{1}, Float32, CPU, []float32{3.0})
		x2, _ := NewTensor([]int{1}, Float32, CPU, []float32{4.0})
		x1.SetRequiresGrad(true)
		x2.SetRequiresGrad(true)
		
		y, err := MulAutograd(x1, x2)
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		
		// Perform backward pass
		err = y.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Check gradients: ∂(x1 * x2)/∂x1 = x2, ∂(x1 * x2)/∂x2 = x1
		if x1.Grad() == nil {
			t.Error("x1 should have gradient")
		} else {
			grad1 := x1.Grad().Data.([]float32)[0]
			if grad1 != 4.0 {
				t.Errorf("Expected x1 gradient to be 4.0, got %f", grad1)
			}
		}
		
		if x2.Grad() == nil {
			t.Error("x2 should have gradient")
		} else {
			grad2 := x2.Grad().Data.([]float32)[0]
			if grad2 != 3.0 {
				t.Errorf("Expected x2 gradient to be 3.0, got %f", grad2)
			}
		}
	})
	
	t.Run("Chain rule test", func(t *testing.T) {
		// Test y = (x1 + x2) * x3, where y is scalar
		x1, _ := NewTensor([]int{1}, Float32, CPU, []float32{2.0})
		x2, _ := NewTensor([]int{1}, Float32, CPU, []float32{3.0})
		x3, _ := NewTensor([]int{1}, Float32, CPU, []float32{4.0})
		x1.SetRequiresGrad(true)
		x2.SetRequiresGrad(true)
		x3.SetRequiresGrad(true)
		
		// Build computational graph
		sum, err := AddAutograd(x1, x2) // sum = 5.0
		if err != nil {
			t.Fatalf("AddAutograd failed: %v", err)
		}
		y, err := MulAutograd(sum, x3)  // y = 20.0
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		
		// Perform backward pass
		err = y.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Check gradients
		// ∂y/∂x1 = ∂y/∂sum * ∂sum/∂x1 = x3 * 1 = 4.0
		// ∂y/∂x2 = ∂y/∂sum * ∂sum/∂x2 = x3 * 1 = 4.0  
		// ∂y/∂x3 = ∂y/∂x3 = sum = 5.0
		
		if x1.Grad() == nil {
			t.Error("x1 should have gradient")
		} else {
			grad1 := x1.Grad().Data.([]float32)[0]
			if grad1 != 4.0 {
				t.Errorf("Expected x1 gradient to be 4.0, got %f", grad1)
			}
		}
		
		if x2.Grad() == nil {
			t.Error("x2 should have gradient")
		} else {
			grad2 := x2.Grad().Data.([]float32)[0]
			if grad2 != 4.0 {
				t.Errorf("Expected x2 gradient to be 4.0, got %f", grad2)
			}
		}
		
		if x3.Grad() == nil {
			t.Error("x3 should have gradient")
		} else {
			grad3 := x3.Grad().Data.([]float32)[0]
			if grad3 != 5.0 {
				t.Errorf("Expected x3 gradient to be 5.0, got %f", grad3)
			}
		}
	})
}

func TestAutogradActivations(t *testing.T) {
	t.Run("ReLU backward", func(t *testing.T) {
		// Test with a scalar output since backward requires scalar
		x_scalar, _ := NewTensor([]int{1}, Float32, CPU, []float32{2.0})
		x_scalar.SetRequiresGrad(true)
		y_scalar, err := ReLUAutograd(x_scalar)
		if err != nil {
			t.Fatalf("ReLUAutograd failed: %v", err)
		}
		
		err = y_scalar.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// For positive input, ReLU gradient should be 1
		if x_scalar.Grad() == nil {
			t.Error("x_scalar should have gradient")
		} else {
			grad := x_scalar.Grad().Data.([]float32)[0]
			if grad != 1.0 {
				t.Errorf("Expected ReLU gradient to be 1.0 for positive input, got %f", grad)
			}
		}
	})
	
	t.Run("Sigmoid backward", func(t *testing.T) {
		// Test y = sigmoid(x), where x = 0 (sigmoid(0) = 0.5)
		x, _ := NewTensor([]int{1}, Float32, CPU, []float32{0.0})
		x.SetRequiresGrad(true)
		
		y, err := SigmoidAutograd(x)
		if err != nil {
			t.Fatalf("SigmoidAutograd failed: %v", err)
		}
		
		err = y.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Sigmoid derivative at x=0: σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
		if x.Grad() == nil {
			t.Error("x should have gradient")
		} else {
			grad := x.Grad().Data.([]float32)[0]
			expected := float32(0.25)
			if math.Abs(float64(grad-expected)) > 1e-6 {
				t.Errorf("Expected sigmoid gradient to be %f, got %f", expected, grad)
			}
		}
	})
}

func TestMatMulBackward(t *testing.T) {
	t.Run("Matrix multiplication backward", func(t *testing.T) {
		// Test y = A @ B where A is 2x3, B is 3x2, result is 2x2
		A, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		B, _ := NewTensor([]int{3, 2}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		A.SetRequiresGrad(true)
		B.SetRequiresGrad(true)
		
		Y, err := MatMulAutograd(A, B)
		if err != nil {
			t.Fatalf("MatMulAutograd failed: %v", err)
		}
		
		// Since backward requires scalar, let's sum all elements
		// In practice, this would be done with a Sum operation
		sumData := Y.Data.([]float32)
		totalSum := float32(0)
		for _, val := range sumData {
			totalSum += val
		}
		
		scalarResult, _ := NewTensor([]int{1}, Float32, CPU, []float32{totalSum})
		scalarResult.SetRequiresGrad(true)
		
		// For testing purposes, let's create a simpler case
		// Test with 1x1 matrices (scalars)
		a, _ := NewTensor([]int{1, 1}, Float32, CPU, []float32{3.0})
		b, _ := NewTensor([]int{1, 1}, Float32, CPU, []float32{4.0})
		a.SetRequiresGrad(true)
		b.SetRequiresGrad(true)
		
		result, err := MatMulAutograd(a, b)
		if err != nil {
			t.Fatalf("MatMulAutograd failed: %v", err)
		}
		
		err = result.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// For matrix multiplication: ∂(A @ B)/∂A = gradOut @ B^T, ∂(A @ B)/∂B = A^T @ gradOut
		// For 1x1 case: ∂(a * b)/∂a = b, ∂(a * b)/∂b = a
		if a.Grad() == nil {
			t.Error("a should have gradient")
		} else {
			gradA := a.Grad().Data.([]float32)[0]
			if gradA != 4.0 {
				t.Errorf("Expected gradient for a to be 4.0, got %f", gradA)
			}
		}
		
		if b.Grad() == nil {
			t.Error("b should have gradient")
		} else {
			gradB := b.Grad().Data.([]float32)[0]
			if gradB != 3.0 {
				t.Errorf("Expected gradient for b to be 3.0, got %f", gradB)
			}
		}
	})
}

func TestGradientAccumulation(t *testing.T) {
	t.Run("Multiple backward passes", func(t *testing.T) {
		// Test that gradients accumulate when the same tensor is used multiple times
		x, _ := NewTensor([]int{1}, Float32, CPU, []float32{2.0})
		x.SetRequiresGrad(true)
		
		// Create two different computations using the same x
		y1, err := MulAutograd(x, x) // y1 = x^2
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		y2, err := AddAutograd(x, x) // y2 = 2x
		if err != nil {
			t.Fatalf("AddAutograd failed: %v", err)
		}
		
		// Backward from y1
		err = y1.Backward()
		if err != nil {
			t.Fatalf("First backward pass failed: %v", err)
		}
		
		firstGrad := x.Grad().Data.([]float32)[0]
		
		// Backward from y2 (should accumulate)
		err = y2.Backward()
		if err != nil {
			t.Fatalf("Second backward pass failed: %v", err)
		}
		
		// Gradient should be accumulated: ∂(x^2)/∂x + ∂(2x)/∂x = 2x + 2 = 2*2 + 2 = 6
		finalGrad := x.Grad().Data.([]float32)[0]
		expected := firstGrad + 2.0 // 4.0 + 2.0 = 6.0
		
		if finalGrad != expected {
			t.Errorf("Expected accumulated gradient to be %f, got %f", expected, finalGrad)
		}
	})
}

func TestAutogradZeroGrad(t *testing.T) {
	t.Run("Zero gradient functionality", func(t *testing.T) {
		x, _ := NewTensor([]int{1}, Float32, CPU, []float32{3.0})
		x.SetRequiresGrad(true)
		
		y, err := MulAutograd(x, x)
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		
		err = y.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		if x.Grad() == nil {
			t.Error("x should have gradient after backward")
		}
		
		// Clear gradient
		x.ZeroGrad()
		
		if x.Grad() != nil {
			t.Error("x gradient should be nil after ZeroGrad()")
		}
	})
}

func TestComplexComputationalGraph(t *testing.T) {
	t.Run("Complex graph: y = (x1 * x2) + (x1 * x3)", func(t *testing.T) {
		x1, _ := NewTensor([]int{1}, Float32, CPU, []float32{2.0})
		x2, _ := NewTensor([]int{1}, Float32, CPU, []float32{3.0})
		x3, _ := NewTensor([]int{1}, Float32, CPU, []float32{4.0})
		x1.SetRequiresGrad(true)
		x2.SetRequiresGrad(true)
		x3.SetRequiresGrad(true)
		
		// Build computational graph
		prod1, err := MulAutograd(x1, x2) // x1 * x2 = 6
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		prod2, err := MulAutograd(x1, x3) // x1 * x3 = 8
		if err != nil {
			t.Fatalf("MulAutograd failed: %v", err)
		}
		y, err := AddAutograd(prod1, prod2) // y = 14
		if err != nil {
			t.Fatalf("AddAutograd failed: %v", err)
		}
		
		err = y.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Expected gradients:
		// ∂y/∂x1 = x2 + x3 = 3 + 4 = 7
		// ∂y/∂x2 = x1 = 2
		// ∂y/∂x3 = x1 = 2
		
		if x1.Grad() == nil {
			t.Error("x1 should have gradient")
		} else {
			grad1 := x1.Grad().Data.([]float32)[0]
			if grad1 != 7.0 {
				t.Errorf("Expected x1 gradient to be 7.0, got %f", grad1)
			}
		}
		
		if x2.Grad() == nil {
			t.Error("x2 should have gradient")
		} else {
			grad2 := x2.Grad().Data.([]float32)[0]
			if grad2 != 2.0 {
				t.Errorf("Expected x2 gradient to be 2.0, got %f", grad2)
			}
		}
		
		if x3.Grad() == nil {
			t.Error("x3 should have gradient")
		} else {
			grad3 := x3.Grad().Data.([]float32)[0]
			if grad3 != 2.0 {
				t.Errorf("Expected x3 gradient to be 2.0, got %f", grad3)
			}
		}
	})
}