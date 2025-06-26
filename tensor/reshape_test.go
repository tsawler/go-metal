package tensor

import (
	"testing"
)

func TestTensorReshape(t *testing.T) {
	t.Run("Basic reshape", func(t *testing.T) {
		// Create a 2x3 tensor
		data := []float32{1, 2, 3, 4, 5, 6}
		tensor, err := NewTensor([]int{2, 3}, Float32, CPU, data)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		// Reshape to 3x2
		reshaped, err := tensor.Reshape([]int{3, 2})
		if err != nil {
			t.Fatalf("Failed to reshape tensor: %v", err)
		}
		
		// Check shape
		if len(reshaped.Shape) != 2 || reshaped.Shape[0] != 3 || reshaped.Shape[1] != 2 {
			t.Errorf("Expected shape [3, 2], got %v", reshaped.Shape)
		}
		
		// Check data is shared
		originalData := tensor.Data.([]float32)
		reshapedData := reshaped.Data.([]float32)
		
		for i := range originalData {
			if originalData[i] != reshapedData[i] {
				t.Errorf("Data not shared properly at index %d", i)
			}
		}
	})
	
	t.Run("Reshape with -1", func(t *testing.T) {
		// Create a 12-element tensor
		data := make([]float32, 12)
		for i := range data {
			data[i] = float32(i)
		}
		
		tensor, err := NewTensor([]int{3, 4}, Float32, CPU, data)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		// Reshape to [2, -1] should become [2, 6]
		reshaped, err := tensor.Reshape([]int{2, -1})
		if err != nil {
			t.Fatalf("Failed to reshape tensor with -1: %v", err)
		}
		
		expectedShape := []int{2, 6}
		if len(reshaped.Shape) != len(expectedShape) {
			t.Fatalf("Expected shape length %d, got %d", len(expectedShape), len(reshaped.Shape))
		}
		
		for i, dim := range expectedShape {
			if reshaped.Shape[i] != dim {
				t.Errorf("Shape dimension %d: expected %d, got %d", i, dim, reshaped.Shape[i])
			}
		}
	})
	
	t.Run("Invalid reshape - size mismatch", func(t *testing.T) {
		data := []float32{1, 2, 3, 4}
		tensor, err := NewTensor([]int{2, 2}, Float32, CPU, data)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		// Try to reshape 4 elements into 6 elements
		_, err = tensor.Reshape([]int{2, 3})
		if err == nil {
			t.Error("Expected error for size mismatch, got none")
		}
	})
	
	t.Run("Invalid reshape - multiple -1", func(t *testing.T) {
		data := []float32{1, 2, 3, 4}
		tensor, err := NewTensor([]int{2, 2}, Float32, CPU, data)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		// Try to use multiple -1 dimensions
		_, err = tensor.Reshape([]int{-1, -1})
		if err == nil {
			t.Error("Expected error for multiple -1 dimensions, got none")
		}
	})
	
	t.Run("Flatten with -1", func(t *testing.T) {
		data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		tensor, err := NewTensor([]int{2, 2, 2}, Float32, CPU, data)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		// Flatten to 1D
		flattened, err := tensor.Reshape([]int{-1})
		if err != nil {
			t.Fatalf("Failed to flatten tensor: %v", err)
		}
		
		if len(flattened.Shape) != 1 || flattened.Shape[0] != 8 {
			t.Errorf("Expected shape [8], got %v", flattened.Shape)
		}
	})
}