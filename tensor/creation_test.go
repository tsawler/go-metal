package tensor

import (
	"reflect"
	"testing"
)

func TestNewTensor(t *testing.T) {
	t.Run("Valid Float32 tensor", func(t *testing.T) {
		shape := []int{2, 3}
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		
		tensor, err := NewTensor(shape, Float32, CPU, data)
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}
		
		if !reflect.DeepEqual(tensor.Shape, shape) {
			t.Errorf("Shape = %v, expected %v", tensor.Shape, shape)
		}
		
		if tensor.DType != Float32 {
			t.Errorf("DType = %v, expected %v", tensor.DType, Float32)
		}
		
		if tensor.Device != CPU {
			t.Errorf("Device = %v, expected %v", tensor.Device, CPU)
		}
		
		if tensor.NumElems != 6 {
			t.Errorf("NumElems = %d, expected 6", tensor.NumElems)
		}
		
		expectedStrides := []int{3, 1}
		if !reflect.DeepEqual(tensor.Strides, expectedStrides) {
			t.Errorf("Strides = %v, expected %v", tensor.Strides, expectedStrides)
		}
		
		resultData := tensor.Data.([]float32)
		if !reflect.DeepEqual(resultData, data) {
			t.Errorf("Data = %v, expected %v", resultData, data)
		}
	})

	t.Run("Valid Int32 tensor", func(t *testing.T) {
		shape := []int{2, 2}
		data := []int32{1, 2, 3, 4}
		
		tensor, err := NewTensor(shape, Int32, CPU, data)
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}
		
		if tensor.DType != Int32 {
			t.Errorf("DType = %v, expected %v", tensor.DType, Int32)
		}
		
		resultData := tensor.Data.([]int32)
		if !reflect.DeepEqual(resultData, data) {
			t.Errorf("Data = %v, expected %v", resultData, data)
		}
	})

	t.Run("Tensor without data", func(t *testing.T) {
		shape := []int{2, 3}
		
		tensor, err := NewTensor(shape, Float32, CPU, nil)
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}
		
		if tensor.Data != nil {
			t.Error("Expected Data to be nil")
		}
	})

	t.Run("Invalid shape", func(t *testing.T) {
		shape := []int{2, 0} // Invalid dimension
		data := []float32{1.0, 2.0}
		
		_, err := NewTensor(shape, Float32, CPU, data)
		if err == nil {
			t.Error("Expected error for invalid shape")
		}
	})

	t.Run("Wrong data length", func(t *testing.T) {
		shape := []int{2, 3}
		data := []float32{1.0, 2.0} // Wrong length
		
		_, err := NewTensor(shape, Float32, CPU, data)
		if err == nil {
			t.Error("Expected error for wrong data length")
		}
	})
}

func TestZeros(t *testing.T) {
	t.Run("Float32 zeros", func(t *testing.T) {
		shape := []int{2, 3}
		
		tensor, err := Zeros(shape, Float32, CPU)
		if err != nil {
			t.Fatalf("Zeros failed: %v", err)
		}
		
		if !reflect.DeepEqual(tensor.Shape, shape) {
			t.Errorf("Shape = %v, expected %v", tensor.Shape, shape)
		}
		
		data := tensor.Data.([]float32)
		expected := make([]float32, 6)
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Int32 zeros", func(t *testing.T) {
		shape := []int{2, 2}
		
		tensor, err := Zeros(shape, Int32, CPU)
		if err != nil {
			t.Fatalf("Zeros failed: %v", err)
		}
		
		data := tensor.Data.([]int32)
		expected := make([]int32, 4)
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Invalid shape", func(t *testing.T) {
		shape := []int{-1}
		
		_, err := Zeros(shape, Float32, CPU)
		if err == nil {
			t.Error("Expected error for invalid shape")
		}
	})

	t.Run("Unsupported dtype", func(t *testing.T) {
		shape := []int{2, 2}
		
		_, err := Zeros(shape, Float16, CPU)
		if err == nil {
			t.Error("Expected error for unsupported dtype")
		}
	})
}

func TestOnes(t *testing.T) {
	t.Run("Float32 ones", func(t *testing.T) {
		shape := []int{2, 3}
		
		tensor, err := Ones(shape, Float32, CPU)
		if err != nil {
			t.Fatalf("Ones failed: %v", err)
		}
		
		data := tensor.Data.([]float32)
		expected := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Int32 ones", func(t *testing.T) {
		shape := []int{2, 2}
		
		tensor, err := Ones(shape, Int32, CPU)
		if err != nil {
			t.Fatalf("Ones failed: %v", err)
		}
		
		data := tensor.Data.([]int32)
		expected := []int32{1, 1, 1, 1}
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Unsupported dtype", func(t *testing.T) {
		shape := []int{2, 2}
		
		_, err := Ones(shape, Float16, CPU)
		if err == nil {
			t.Error("Expected error for unsupported dtype")
		}
	})
}

func TestRandom(t *testing.T) {
	t.Run("Float32 random", func(t *testing.T) {
		shape := []int{2, 3}
		
		tensor, err := Random(shape, Float32, CPU)
		if err != nil {
			t.Fatalf("Random failed: %v", err)
		}
		
		if !reflect.DeepEqual(tensor.Shape, shape) {
			t.Errorf("Shape = %v, expected %v", tensor.Shape, shape)
		}
		
		data := tensor.Data.([]float32)
		if len(data) != 6 {
			t.Errorf("Data length = %d, expected 6", len(data))
		}
		
		// Check that values are in range [0, 1) for float32
		for i, val := range data {
			if val < 0 || val >= 1 {
				t.Errorf("Data[%d] = %f, expected in range [0, 1)", i, val)
			}
		}
	})

	t.Run("Int32 random", func(t *testing.T) {
		shape := []int{2, 2}
		
		tensor, err := Random(shape, Int32, CPU)
		if err != nil {
			t.Fatalf("Random failed: %v", err)
		}
		
		data := tensor.Data.([]int32)
		if len(data) != 4 {
			t.Errorf("Data length = %d, expected 4", len(data))
		}
		
		// Check that values are non-negative
		for i, val := range data {
			if val < 0 {
				t.Errorf("Data[%d] = %d, expected non-negative", i, val)
			}
		}
	})

	t.Run("Unsupported dtype", func(t *testing.T) {
		shape := []int{2, 2}
		
		_, err := Random(shape, Float16, CPU)
		if err == nil {
			t.Error("Expected error for unsupported dtype")
		}
	})
}

func TestRandomNormal(t *testing.T) {
	t.Run("Valid normal distribution", func(t *testing.T) {
		shape := []int{100} // Larger sample for statistical testing
		mean := float32(5.0)
		std := float32(2.0)
		
		tensor, err := RandomNormal(shape, mean, std, Float32, CPU)
		if err != nil {
			t.Fatalf("RandomNormal failed: %v", err)
		}
		
		data := tensor.Data.([]float32)
		if len(data) != 100 {
			t.Errorf("Data length = %d, expected 100", len(data))
		}
		
		// Simple sanity check: compute sample mean
		var sum float32
		for _, val := range data {
			sum += val
		}
		sampleMean := sum / float32(len(data))
		
		// Allow some tolerance for random variation
		if sampleMean < mean-1.0 || sampleMean > mean+1.0 {
			t.Errorf("Sample mean = %f, expected around %f", sampleMean, mean)
		}
	})

	t.Run("Non-Float32 dtype", func(t *testing.T) {
		shape := []int{10}
		
		_, err := RandomNormal(shape, 0.0, 1.0, Int32, CPU)
		if err == nil {
			t.Error("Expected error for non-Float32 dtype")
		}
	})

	t.Run("Invalid shape", func(t *testing.T) {
		shape := []int{0}
		
		_, err := RandomNormal(shape, 0.0, 1.0, Float32, CPU)
		if err == nil {
			t.Error("Expected error for invalid shape")
		}
	})
}

func TestFull(t *testing.T) {
	t.Run("Float32 full", func(t *testing.T) {
		shape := []int{2, 3}
		value := float32(7.5)
		
		tensor, err := Full(shape, value, Float32, CPU)
		if err != nil {
			t.Fatalf("Full failed: %v", err)
		}
		
		data := tensor.Data.([]float32)
		expected := []float32{7.5, 7.5, 7.5, 7.5, 7.5, 7.5}
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Int32 full", func(t *testing.T) {
		shape := []int{2, 2}
		value := int32(42)
		
		tensor, err := Full(shape, value, Int32, CPU)
		if err != nil {
			t.Fatalf("Full failed: %v", err)
		}
		
		data := tensor.Data.([]int32)
		expected := []int32{42, 42, 42, 42}
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})
}