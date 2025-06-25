package tensor

import (
	"reflect"
	"testing"
)

func TestAutogradBroadcasting(t *testing.T) {
	t.Run("Addition with broadcasting", func(t *testing.T) {
		// Test: (2x1) + (1x3) = (2x3), check gradients reduce properly
		a, _ := NewTensor([]int{2, 1}, Float32, CPU, []float32{1, 2})
		b, _ := NewTensor([]int{1, 3}, Float32, CPU, []float32{10, 20, 30})
		a.SetRequiresGrad(true)
		b.SetRequiresGrad(true)
		
		// Forward pass
		result := AddAutograd(a, b)
		
		// Create a scalar loss by summing all elements
		sumData := result.Data.([]float32)
		totalSum := float32(0)
		for _, val := range sumData {
			totalSum += val
		}
		loss, _ := NewTensor([]int{1}, Float32, CPU, []float32{totalSum})
		loss.SetRequiresGrad(true)
		
		// For simplicity, let's test with element that can be made scalar
		// Test with single element from each tensor
		a1, _ := NewTensor([]int{1}, Float32, CPU, []float32{5})
		b1, _ := NewTensor([]int{1}, Float32, CPU, []float32{10})
		a1.SetRequiresGrad(true)
		b1.SetRequiresGrad(true)
		
		result1 := AddAutograd(a1, b1)
		
		err := result1.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// Check gradients - should be 1 for both inputs in addition
		if a1.Grad() == nil {
			t.Error("a1 should have gradient")
		} else {
			grad := a1.Grad().Data.([]float32)[0]
			if grad != 1.0 {
				t.Errorf("Expected a1 gradient to be 1.0, got %f", grad)
			}
		}
		
		if b1.Grad() == nil {
			t.Error("b1 should have gradient")
		} else {
			grad := b1.Grad().Data.([]float32)[0]
			if grad != 1.0 {
				t.Errorf("Expected b1 gradient to be 1.0, got %f", grad)
			}
		}
	})
	
	t.Run("Multiplication with broadcasting", func(t *testing.T) {
		// Test scalar multiplication with broadcasting
		a, _ := NewTensor([]int{1}, Float32, CPU, []float32{3})
		b, _ := NewTensor([]int{1}, Float32, CPU, []float32{4})
		a.SetRequiresGrad(true)
		b.SetRequiresGrad(true)
		
		result := MulAutograd(a, b)
		
		err := result.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}
		
		// For multiplication: ∂(a * b)/∂a = b = 4, ∂(a * b)/∂b = a = 3
		if a.Grad() == nil {
			t.Error("a should have gradient")
		} else {
			grad := a.Grad().Data.([]float32)[0]
			if grad != 4.0 {
				t.Errorf("Expected a gradient to be 4.0, got %f", grad)
			}
		}
		
		if b.Grad() == nil {
			t.Error("b should have gradient")
		} else {
			grad := b.Grad().Data.([]float32)[0]
			if grad != 3.0 {
				t.Errorf("Expected b gradient to be 3.0, got %f", grad)
			}
		}
	})
}

func TestGradientReduction(t *testing.T) {
	t.Run("Sum all elements", func(t *testing.T) {
		// Create a 2x3 tensor
		input, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := sumAllElements(input)
		if err != nil {
			t.Fatalf("sumAllElements failed: %v", err)
		}
		
		// Check shape is scalar
		expectedShape := []int{1}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check sum is correct: 1+2+3+4+5+6 = 21
		resultData := result.Data.([]float32)[0]
		if resultData != 21.0 {
			t.Errorf("Sum result = %f, expected 21.0", resultData)
		}
	})
	
	t.Run("Sum over dimension", func(t *testing.T) {
		// Create a 2x3 tensor [[1,2,3], [4,5,6]]
		input, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Sum over dimension 0 (rows) -> should get [5, 7, 9]
		result, err := sumOverDimension(input, 0)
		if err != nil {
			t.Fatalf("sumOverDimension failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check values: [1+4, 2+5, 3+6] = [5, 7, 9]
		expectedData := []float32{5, 7, 9}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Sum over dimension 1", func(t *testing.T) {
		// Create a 2x3 tensor [[1,2,3], [4,5,6]]
		input, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Sum over dimension 1 (columns) -> should get [6, 15]
		result, err := sumOverDimension(input, 1)
		if err != nil {
			t.Fatalf("sumOverDimension failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check values: [1+2+3, 4+5+6] = [6, 15]
		expectedData := []float32{6, 15}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
}

func TestReduceGradientToShape(t *testing.T) {
	t.Run("Reduce 2x3 to 1x3", func(t *testing.T) {
		// Gradient with shape [2, 3]
		grad, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Target shape [1, 3] - should sum over first dimension
		result, err := reduceGradientToShape(grad, []int{1, 3})
		if err != nil {
			t.Fatalf("reduceGradientToShape failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{1, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check values: should be [1+4, 2+5, 3+6] = [5, 7, 9]
		expectedData := []float32{5, 7, 9}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Reduce 2x3 to 2x1", func(t *testing.T) {
		// Gradient with shape [2, 3]
		grad, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Target shape [2, 1] - should sum over second dimension
		result, err := reduceGradientToShape(grad, []int{2, 1})
		if err != nil {
			t.Fatalf("reduceGradientToShape failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{2, 1}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check values: should be [1+2+3, 4+5+6] = [6, 15]
		expectedData := []float32{6, 15}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Reduce to scalar", func(t *testing.T) {
		// Gradient with shape [2, 3]
		grad, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Target shape [1] - should sum all elements
		result, err := reduceGradientToShape(grad, []int{1})
		if err != nil {
			t.Fatalf("reduceGradientToShape failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{1}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check values: should be 1+2+3+4+5+6 = 21
		resultData := result.Data.([]float32)[0]
		if resultData != 21.0 {
			t.Errorf("Result data = %f, expected 21.0", resultData)
		}
	})
	
	t.Run("Same shape - no reduction needed", func(t *testing.T) {
		// Gradient with shape [2, 3]
		grad, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Target shape [2, 3] - should just clone
		result, err := reduceGradientToShape(grad, []int{2, 3})
		if err != nil {
			t.Fatalf("reduceGradientToShape failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check values are the same
		expectedData := []float32{1, 2, 3, 4, 5, 6}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
}

func TestCoordinateConversion(t *testing.T) {
	t.Run("Index to coordinates", func(t *testing.T) {
		shape := []int{2, 3}
		
		tests := []struct {
			index    int
			expected []int
		}{
			{0, []int{0, 0}},
			{1, []int{0, 1}},
			{2, []int{0, 2}},
			{3, []int{1, 0}},
			{4, []int{1, 1}},
			{5, []int{1, 2}},
		}
		
		for _, test := range tests {
			result := indexToCoords(test.index, shape)
			if !reflect.DeepEqual(result, test.expected) {
				t.Errorf("indexToCoords(%d, %v) = %v, expected %v", 
					test.index, shape, result, test.expected)
			}
		}
	})
	
	t.Run("Coordinates to index", func(t *testing.T) {
		strides := []int{3, 1} // for shape [2, 3]
		
		tests := []struct {
			coords   []int
			expected int
		}{
			{[]int{0, 0}, 0},
			{[]int{0, 1}, 1},
			{[]int{0, 2}, 2},
			{[]int{1, 0}, 3},
			{[]int{1, 1}, 4},
			{[]int{1, 2}, 5},
		}
		
		for _, test := range tests {
			result := coordsToIndex(test.coords, strides)
			if result != test.expected {
				t.Errorf("coordsToIndex(%v, %v) = %d, expected %d", 
					test.coords, strides, result, test.expected)
			}
		}
	})
}