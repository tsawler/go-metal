package tensor

import (
	"reflect"
	"testing"
)

func TestGetIndex(t *testing.T) {
	tests := []struct {
		indices  []int
		strides  []int
		expected int
	}{
		{[]int{0, 0}, []int{3, 1}, 0},
		{[]int{0, 1}, []int{3, 1}, 1},
		{[]int{1, 0}, []int{3, 1}, 3},
		{[]int{1, 2}, []int{3, 1}, 5},
		{[]int{1, 2, 3}, []int{12, 4, 1}, 23},
	}

	for _, test := range tests {
		result := getIndex(test.indices, test.strides)
		if result != test.expected {
			t.Errorf("getIndex(%v, %v) = %d, expected %d", 
				test.indices, test.strides, result, test.expected)
		}
	}
}

func TestGetIndicesFromLinear(t *testing.T) {
	tests := []struct {
		linearIndex int
		shape       []int
		expected    []int
	}{
		{0, []int{2, 3}, []int{0, 0}},
		{1, []int{2, 3}, []int{0, 1}},
		{3, []int{2, 3}, []int{1, 0}},
		{5, []int{2, 3}, []int{1, 2}},
		{23, []int{2, 3, 4}, []int{1, 2, 3}},
	}

	for _, test := range tests {
		result := getIndicesFromLinear(test.linearIndex, test.shape)
		if !reflect.DeepEqual(result, test.expected) {
			t.Errorf("getIndicesFromLinear(%d, %v) = %v, expected %v", 
				test.linearIndex, test.shape, result, test.expected)
		}
	}
}

func TestMatMul(t *testing.T) {
	t.Run("2x3 * 3x2 = 2x2", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{3, 2}, Float32, CPU, []float32{7, 8, 9, 10, 11, 12})
		
		result, err := MatMul(a, b)
		if err != nil {
			t.Fatalf("MatMul failed: %v", err)
		}
		
		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Expected: [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
		//          = [58, 64, 139, 154]
		expected := []float32{58, 64, 139, 154}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Int32 matrix multiplication", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})
		
		result, err := MatMul(a, b)
		if err != nil {
			t.Fatalf("MatMul failed: %v", err)
		}
		
		// Expected: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
		expected := []int32{19, 22, 43, 50}
		resultData := result.Data.([]int32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible dimensions", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{7, 8, 9, 10})
		
		_, err := MatMul(a, b)
		if err == nil {
			t.Error("Expected error for incompatible dimensions")
		}
	})

	t.Run("1D tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{3}, Float32, CPU, []float32{1, 2, 3})
		b, _ := NewTensor([]int{3}, Float32, CPU, []float32{4, 5, 6})
		
		_, err := MatMul(a, b)
		if err == nil {
			t.Error("Expected error for 1D tensors")
		}
	})

	t.Run("Incompatible tensor types", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})
		
		_, err := MatMul(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensor types")
		}
	})
}

func TestTranspose(t *testing.T) {
	t.Run("2D transpose", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Transpose(a, 0, 1)
		if err != nil {
			t.Fatalf("Transpose failed: %v", err)
		}
		
		expectedShape := []int{3, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Test that the transpose is a true view
		if !result.isView {
			t.Error("Transpose result should be a view")
		}
		
		// Test individual element access using stride-aware methods
		expectedValues := [][]float32{
			{1, 4}, // First row: (0,0)=1, (0,1)=4
			{2, 5}, // Second row: (1,0)=2, (1,1)=5  
			{3, 6}, // Third row: (2,0)=3, (2,1)=6
		}
		
		for i := 0; i < result.Shape[0]; i++ {
			for j := 0; j < result.Shape[1]; j++ {
				actual, err := result.At(i, j)
				if err != nil {
					t.Fatalf("At(%d,%d) failed: %v", i, j, err)
				}
				expected := expectedValues[i][j]
				if actual.(float32) != expected {
					t.Errorf("At(%d,%d) = %f, expected %f", i, j, actual.(float32), expected)
				}
			}
		}
		
		// Test that modifying the original tensor affects the view
		setErr := a.SetAt(float32(99), 0, 0) // Change original (0,0) from 1 to 99
		if setErr != nil {
			t.Fatalf("SetAt failed: %v", setErr)
		}
		actual, getErr := result.At(0, 0)
		if getErr != nil {
			t.Fatalf("At failed: %v", getErr)
		}
		if actual.(float32) != 99 {
			t.Error("View should reflect changes to original tensor")
		}
	})

	t.Run("3D transpose", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2, 2}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6, 7, 8})
		
		result, err := Transpose(a, 0, 2)
		if err != nil {
			t.Fatalf("Transpose failed: %v", err)
		}
		
		expectedShape := []int{2, 2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("Invalid dimension", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Transpose(a, 0, 2)
		if err == nil {
			t.Error("Expected error for invalid dimension")
		}
	})

	t.Run("Negative dimension", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Transpose(a, -1, 1)
		if err == nil {
			t.Error("Expected error for negative dimension")
		}
	})
}

func TestReshape(t *testing.T) {
	t.Run("Valid reshape", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Reshape(a, []int{3, 2})
		if err != nil {
			t.Fatalf("Reshape failed: %v", err)
		}
		
		expectedShape := []int{3, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		// Data should remain the same
		originalData := a.Data.([]float32)
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, originalData) {
			t.Errorf("Data changed during reshape")
		}
	})

	t.Run("Invalid reshape - wrong size", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Reshape(a, []int{2, 2})
		if err == nil {
			t.Error("Expected error for invalid reshape size")
		}
	})

	t.Run("Invalid shape", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Reshape(a, []int{2, 0})
		if err == nil {
			t.Error("Expected error for invalid shape")
		}
	})
}

func TestFlatten(t *testing.T) {
	t.Run("2D flatten", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Flatten(a)
		if err != nil {
			t.Fatalf("Flatten failed: %v", err)
		}
		
		expectedShape := []int{6}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("3D flatten", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2, 2}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6, 7, 8})
		
		result, err := Flatten(a)
		if err != nil {
			t.Fatalf("Flatten failed: %v", err)
		}
		
		expectedShape := []int{8}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})
}

func TestSqueeze(t *testing.T) {
	t.Run("Valid squeeze", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 1, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Squeeze(a, 1)
		if err != nil {
			t.Fatalf("Squeeze failed: %v", err)
		}
		
		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("Invalid squeeze - dimension not 1", func(t *testing.T) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i + 1)
		}
		a, _ := NewTensor([]int{2, 3, 4}, Float32, CPU, data)
		
		_, err := Squeeze(a, 1)
		if err == nil {
			t.Error("Expected error for squeezing dimension that is not 1")
		}
	})

	t.Run("Invalid dimension", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 1, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Squeeze(a, 3)
		if err == nil {
			t.Error("Expected error for invalid dimension")
		}
	})
}

func TestUnsqueeze(t *testing.T) {
	t.Run("Valid unsqueeze", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Unsqueeze(a, 1)
		if err != nil {
			t.Fatalf("Unsqueeze failed: %v", err)
		}
		
		expectedShape := []int{2, 1, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("Unsqueeze at beginning", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Unsqueeze(a, 0)
		if err != nil {
			t.Fatalf("Unsqueeze failed: %v", err)
		}
		
		expectedShape := []int{1, 2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("Unsqueeze at end", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Unsqueeze(a, 2)
		if err != nil {
			t.Fatalf("Unsqueeze failed: %v", err)
		}
		
		expectedShape := []int{2, 3, 1}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("Invalid dimension", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Unsqueeze(a, 3)
		if err == nil {
			t.Error("Expected error for invalid dimension")
		}
	})
}

func TestSum(t *testing.T) {
	t.Run("Sum along dimension 0", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Sum(a, 0, false)
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}
		
		expectedShape := []int{3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		expected := []float32{5, 7, 9} // [1+4, 2+5, 3+6]
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Sum along dimension 1", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Sum(a, 1, false)
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}
		
		expectedShape := []int{2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		expected := []float32{6, 15} // [1+2+3, 4+5+6]
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Sum with keepDim=true", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		result, err := Sum(a, 1, true)
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}
		
		expectedShape := []int{2, 1}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
		
		expected := []float32{6, 15} // [1+2+3, 4+5+6]
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Sum Int32", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		result, err := Sum(a, 0, false)
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}
		
		expected := []int32{4, 6} // [1+3, 2+4]
		resultData := result.Data.([]int32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Invalid dimension", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := Sum(a, 2, false)
		if err == nil {
			t.Error("Expected error for invalid dimension")
		}
	})
}