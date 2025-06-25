package tensor

import (
	"reflect"
	"testing"
)

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		name     string
		shape1   []int
		shape2   []int
		expected []int
		wantErr  bool
	}{
		{
			name:     "Same shapes",
			shape1:   []int{3, 4},
			shape2:   []int{3, 4},
			expected: []int{3, 4},
			wantErr:  false,
		},
		{
			name:     "Scalar and tensor",
			shape1:   []int{1},
			shape2:   []int{3, 4},
			expected: []int{3, 4},
			wantErr:  false,
		},
		{
			name:     "Compatible broadcasting",
			shape1:   []int{3, 1},
			shape2:   []int{1, 4},
			expected: []int{3, 4},
			wantErr:  false,
		},
		{
			name:     "Different dimensions",
			shape1:   []int{5},
			shape2:   []int{3, 5},
			expected: []int{3, 5},
			wantErr:  false,
		},
		{
			name:     "Complex broadcasting",
			shape1:   []int{2, 3, 1},
			shape2:   []int{1, 4},
			expected: []int{2, 3, 4},
			wantErr:  false,
		},
		{
			name:     "Incompatible shapes",
			shape1:   []int{3, 4},
			shape2:   []int{2, 3},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Empty shapes",
			shape1:   []int{},
			shape2:   []int{},
			expected: []int{},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := BroadcastShapes(tt.shape1, tt.shape2)
			
			if tt.wantErr {
				if err == nil {
					t.Errorf("BroadcastShapes() expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Errorf("BroadcastShapes() unexpected error: %v", err)
				return
			}
			
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("BroadcastShapes() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestAreBroadcastable(t *testing.T) {
	tests := []struct {
		name     string
		shape1   []int
		shape2   []int
		expected bool
	}{
		{
			name:     "Compatible shapes",
			shape1:   []int{3, 1},
			shape2:   []int{1, 4},
			expected: true,
		},
		{
			name:     "Incompatible shapes",
			shape1:   []int{3, 4},
			shape2:   []int{2, 3},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AreBroadcastable(tt.shape1, tt.shape2)
			if result != tt.expected {
				t.Errorf("AreBroadcastable() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestBroadcastTensor(t *testing.T) {
	t.Run("Broadcast scalar to matrix", func(t *testing.T) {
		// Create scalar tensor
		scalar, _ := NewTensor([]int{1}, Float32, CPU, []float32{5.0})
		
		// Broadcast to 2x3
		result, err := BroadcastTensor(scalar, []int{2, 3})
		if err != nil {
			t.Fatalf("BroadcastTensor failed: %v", err)
		}
		
		// Check shape
		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check data
		expectedData := []float32{5, 5, 5, 5, 5, 5}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Broadcast row vector to matrix", func(t *testing.T) {
		// Create row vector [1, 2, 3]
		row, _ := NewTensor([]int{1, 3}, Float32, CPU, []float32{1, 2, 3})
		
		// Broadcast to 2x3
		result, err := BroadcastTensor(row, []int{2, 3})
		if err != nil {
			t.Fatalf("BroadcastTensor failed: %v", err)
		}
		
		// Check data - should repeat the row
		expectedData := []float32{1, 2, 3, 1, 2, 3}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Broadcast column vector to matrix", func(t *testing.T) {
		// Create column vector [1; 2]
		col, _ := NewTensor([]int{2, 1}, Float32, CPU, []float32{1, 2})
		
		// Broadcast to 2x3
		result, err := BroadcastTensor(col, []int{2, 3})
		if err != nil {
			t.Fatalf("BroadcastTensor failed: %v", err)
		}
		
		// Check data - should repeat each element across columns
		expectedData := []float32{1, 1, 1, 2, 2, 2}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
}

func TestBroadcastOperations(t *testing.T) {
	t.Run("Add with broadcasting", func(t *testing.T) {
		// Create tensors with different shapes
		a, _ := NewTensor([]int{2, 1}, Float32, CPU, []float32{1, 2})
		b, _ := NewTensor([]int{1, 3}, Float32, CPU, []float32{10, 20, 30})
		
		// Perform addition with broadcasting
		result, err := Add(a, b)
		if err != nil {
			t.Fatalf("Add with broadcasting failed: %v", err)
		}
		
		// Check result shape
		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check result data
		// a broadcasts to [[1, 1, 1], [2, 2, 2]]
		// b broadcasts to [[10, 20, 30], [10, 20, 30]]
		// result should be [[11, 21, 31], [12, 22, 32]]
		expectedData := []float32{11, 21, 31, 12, 22, 32}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Multiply with broadcasting", func(t *testing.T) {
		// Create tensors with different shapes
		a, _ := NewTensor([]int{3}, Float32, CPU, []float32{2, 3, 4})
		b, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		// Perform multiplication with broadcasting
		result, err := Mul(a, b)
		if err != nil {
			t.Fatalf("Mul with broadcasting failed: %v", err)
		}
		
		// Check result shape
		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check result data
		// a broadcasts to [[2, 3, 4], [2, 3, 4]]
		// b is [[1, 2, 3], [4, 5, 6]]
		// result should be [[2, 6, 12], [8, 15, 24]]
		expectedData := []float32{2, 6, 12, 8, 15, 24}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Subtract with broadcasting", func(t *testing.T) {
		// Create tensors with different shapes
		a, _ := NewTensor([]int{2, 1}, Float32, CPU, []float32{10, 20})
		b, _ := NewTensor([]int{3}, Float32, CPU, []float32{1, 2, 3})
		
		// Perform subtraction with broadcasting
		result, err := Sub(a, b)
		if err != nil {
			t.Fatalf("Sub with broadcasting failed: %v", err)
		}
		
		// Check result shape
		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Check result data
		// a broadcasts to [[10, 10, 10], [20, 20, 20]]
		// b broadcasts to [[1, 2, 3], [1, 2, 3]]
		// result should be [[9, 8, 7], [19, 18, 17]]
		expectedData := []float32{9, 8, 7, 19, 18, 17}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expectedData) {
			t.Errorf("Result data = %v, expected %v", resultData, expectedData)
		}
	})
	
	t.Run("Divide with broadcasting", func(t *testing.T) {
		// Create tensors with different shapes
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{12, 15, 20, 24})
		b, _ := NewTensor([]int{1, 2}, Float32, CPU, []float32{3, 5})
		
		// Perform division with broadcasting
		result, err := Div(a, b)
		if err != nil {
			t.Fatalf("Div with broadcasting failed: %v", err)
		}
		
		// Check result data
		// a is [[12, 15], [20, 24]]
		// b broadcasts to [[3, 5], [3, 5]]
		// result should be [[4, 3], [6.67, 4.8]]
		expectedData := []float32{4, 3, 6.666667, 4.8}
		resultData := result.Data.([]float32)
		
		// Check with tolerance for floating point comparison
		tolerance := float32(0.0001)
		for i, expected := range expectedData {
			if diff := resultData[i] - expected; diff < -tolerance || diff > tolerance {
				t.Errorf("Result data[%d] = %v, expected %v", i, resultData[i], expected)
			}
		}
	})
}

func TestBroadcastingErrors(t *testing.T) {
	t.Run("Incompatible shapes", func(t *testing.T) {
		a, _ := NewTensor([]int{3, 4}, Float32, CPU, make([]float32, 12))
		b, _ := NewTensor([]int{2, 3}, Float32, CPU, make([]float32, 6))
		
		_, err := Add(a, b)
		if err == nil {
			t.Error("Expected error for incompatible shapes")
		}
	})
	
	t.Run("Different data types", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 1}, Float32, CPU, []float32{1, 2})
		b, _ := NewTensor([]int{1, 3}, Int32, CPU, []int32{10, 20, 30})
		
		_, err := Add(a, b)
		if err == nil {
			t.Error("Expected error for different data types")
		}
	})
}

func TestComplexBroadcasting(t *testing.T) {
	t.Run("3D broadcasting", func(t *testing.T) {
		// Create 3D tensors with compatible shapes for broadcasting
		a, _ := NewTensor([]int{2, 1, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{1, 2, 1}, Float32, CPU, []float32{10, 20})
		
		// Perform addition with broadcasting
		result, err := Add(a, b)
		if err != nil {
			t.Fatalf("3D broadcasting failed: %v", err)
		}
		
		// Check result shape
		expectedShape := []int{2, 2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// The result should be 2x2x3 = 12 elements
		if result.NumElems != 12 {
			t.Errorf("Result NumElems = %d, expected 12", result.NumElems)
		}
	})
}