package tensor

import (
	"math"
	"reflect"
	"testing"
)

func TestGetMPSGraphEngine(t *testing.T) {
	engine, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}
	
	if engine == nil {
		t.Fatal("GetMPSGraphEngine returned nil engine")
	}
	
	if engine.device == nil {
		t.Error("Engine device is nil")
	}
	
	if engine.graphDevice == nil {
		t.Error("Engine graph device is nil")
	}
	
	if engine.commandQueue == nil {
		t.Error("Engine command queue is nil")
	}
}

func TestSubMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Float32 subtraction", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5.0, 6.0, 7.0, 8.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})

		result, err := SubMPS(a, b)
		if err != nil {
			t.Fatalf("SubMPS failed: %v", err)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{4.0, 4.0, 4.0, 4.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("SubMPS result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - scalar from tensor", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8})
		b, _ := NewTensor([]int{1}, Float32, CPU, []float32{3})

		result, err := SubMPS(a, b)
		if err != nil {
			t.Fatalf("SubMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{2.0, 3.0, 4.0, 5.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("SubMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - vector from matrix", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{10, 20, 30, 40, 50, 60})
		b, _ := NewTensor([]int{3}, Float32, CPU, []float32{1, 2, 3})

		result, err := SubMPS(a, b)
		if err != nil {
			t.Fatalf("SubMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{9.0, 18.0, 27.0, 39.0, 48.0, 57.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("SubMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		_, err := SubMPS(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
		}
	})

	t.Run("Non-broadcastable shapes", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2}, Float32, CPU, []float32{10, 20})

		_, err := SubMPS(a, b)
		if err == nil {
			t.Error("Expected error for non-broadcastable shapes")
		}
	})
}

func TestMulMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Float32 multiplication", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{2.0, 3.0, 4.0, 5.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})

		result, err := MulMPS(a, b)
		if err != nil {
			t.Fatalf("MulMPS failed: %v", err)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{2.0, 6.0, 12.0, 20.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("MulMPS result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - scalar to tensor", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{1}, Float32, CPU, []float32{3})

		result, err := MulMPS(a, b)
		if err != nil {
			t.Fatalf("MulMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{3.0, 6.0, 9.0, 12.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("MulMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - vector to matrix", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{3}, Float32, CPU, []float32{2, 3, 4})

		result, err := MulMPS(a, b)
		if err != nil {
			t.Fatalf("MulMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{2.0, 6.0, 12.0, 8.0, 15.0, 24.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("MulMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		_, err := MulMPS(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
		}
	})

	t.Run("Non-broadcastable shapes", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2}, Float32, CPU, []float32{10, 20})

		_, err := MulMPS(a, b)
		if err == nil {
			t.Error("Expected error for non-broadcastable shapes")
		}
	})
}

func TestDivMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Float32 division", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{8.0, 12.0, 16.0, 20.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{2.0, 3.0, 4.0, 5.0})

		result, err := DivMPS(a, b)
		if err != nil {
			t.Fatalf("DivMPS failed: %v", err)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{4.0, 4.0, 4.0, 4.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("DivMPS result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - scalar division", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{6, 9, 12, 15})
		b, _ := NewTensor([]int{1}, Float32, CPU, []float32{3})

		result, err := DivMPS(a, b)
		if err != nil {
			t.Fatalf("DivMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{2.0, 3.0, 4.0, 5.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("DivMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - vector division", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{6, 12, 18, 8, 15, 24})
		b, _ := NewTensor([]int{3}, Float32, CPU, []float32{2, 3, 6})

		result, err := DivMPS(a, b)
		if err != nil {
			t.Fatalf("DivMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{3.0, 4.0, 3.0, 4.0, 5.0, 4.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("DivMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Division by zero handling", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 0, 3, 4})

		result, err := DivMPS(a, b)
		if err != nil {
			t.Fatalf("DivMPS failed: %v", err)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		resultData := cpuResult.Data.([]float32)
		// Check that division by zero produces infinity, not an error
		if !math.IsInf(float64(resultData[1]), 1) {
			t.Errorf("Division by zero should produce +Inf, got %v", resultData[1])
		}
	})

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		_, err := DivMPS(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
		}
	})

	t.Run("Non-broadcastable shapes", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2}, Float32, CPU, []float32{10, 20})

		_, err := DivMPS(a, b)
		if err == nil {
			t.Error("Expected error for non-broadcastable shapes")
		}
	})
}

func TestAddMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Float32 addition", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5.0, 6.0, 7.0, 8.0})

		result, err := AddMPS(a, b)
		if err != nil {
			t.Fatalf("AddMPS failed: %v", err)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{6.0, 8.0, 10.0, 12.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("AddMPS result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - scalar to tensor", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{1}, Float32, CPU, []float32{5})

		result, err := AddMPS(a, b)
		if err != nil {
			t.Fatalf("AddMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{6.0, 7.0, 8.0, 9.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("AddMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Broadcasting - vector to matrix", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{3}, Float32, CPU, []float32{10, 20, 30})

		result, err := AddMPS(a, b)
		if err != nil {
			t.Fatalf("AddMPS broadcasting failed: %v", err)
		}

		expectedShape := []int{2, 3}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{11.0, 22.0, 33.0, 14.0, 25.0, 36.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("AddMPS broadcast result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		_, err := AddMPS(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
		}
	})

	t.Run("Non-broadcastable shapes", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2}, Float32, CPU, []float32{10, 20})

		_, err := AddMPS(a, b)
		if err == nil {
			t.Error("Expected error for non-broadcastable shapes")
		}
	})
}

func TestMatMulMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("2x3 * 3x2 = 2x2", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{3, 2}, Float32, CPU, []float32{7, 8, 9, 10, 11, 12})

		result, err := MatMulMPS(a, b)
		if err != nil {
			t.Fatalf("MatMulMPS failed: %v", err)
		}

		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		// Expected: [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
		//          = [58, 64, 139, 154]
		expected := []float32{58, 64, 139, 154}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("MatMulMPS result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible dimensions", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{7, 8, 9, 10})

		_, err := MatMulMPS(a, b)
		if err == nil {
			t.Error("Expected error for incompatible dimensions")
		}
	})
}

func TestReLUMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Float32 ReLU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{-1.0, 2.0, -3.0, 4.0})

		result, err := ReLUMPS(a)
		if err != nil {
			t.Fatalf("ReLUMPS failed: %v", err)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		expected := []float32{0.0, 2.0, 0.0, 4.0}
		resultData := cpuResult.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("ReLUMPS result = %v, expected %v", resultData, expected)
		}
	})
}

func TestSigmoidMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Float32 Sigmoid", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{0.0, 1.0, -1.0, 2.0})

		result, err := SigmoidMPS(a)
		if err != nil {
			t.Fatalf("SigmoidMPS failed: %v", err)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		resultData := cpuResult.Data.([]float32)
		
		// Check that results are in sigmoid range [0, 1]
		for i, val := range resultData {
			if val < 0.0 || val > 1.0 {
				t.Errorf("Sigmoid result[%d] = %f, should be in range [0, 1]", i, val)
			}
		}

		// Check specific values with tolerance
		tolerance := float32(0.01)
		expectedApprox := []float32{0.5, 0.731, 0.269, 0.881} // Approximate sigmoid values
		
		for i, expected := range expectedApprox {
			diff := resultData[i] - expected
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				t.Errorf("Sigmoid result[%d] = %f, expected approximately %f", i, resultData[i], expected)
			}
		}
	})
}

func TestMPSPerformanceComparison(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	size := 1000
	data1 := make([]float32, size)
	data2 := make([]float32, size)

	for i := 0; i < size; i++ {
		data1[i] = float32(i)
		data2[i] = float32(i * 2)
	}

	a, _ := NewTensor([]int{size}, Float32, CPU, data1)
	b, _ := NewTensor([]int{size}, Float32, CPU, data2)

	// Test MPSGraph performance
	result, err := AddMPS(a, b)
	if err != nil {
		t.Fatalf("MPS addition failed: %v", err)
	}

	// Convert result back to CPU for verification
	cpuResult, err := result.ToCPU()
	if err != nil {
		t.Fatalf("Failed to convert result to CPU: %v", err)
	}

	// Verify some results
	resultData := cpuResult.Data.([]float32)
	for i := 0; i < 10; i++ {
		expected := float32(i * 3)
		if resultData[i] != expected {
			t.Errorf("Result[%d] = %f, expected %f", i, resultData[i], expected)
		}
	}

	t.Logf("Successfully processed %d elements using MPSGraph", size)
}

func TestConv2DMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Basic Conv2D without bias", func(t *testing.T) {
		// Input: 1 batch, 1 channel, 4x4 image
		input, _ := NewTensor([]int{1, 1, 4, 4}, Float32, CPU, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})
		
		// Weights: 1 output channel, 1 input channel, 3x3 kernel
		weights, _ := NewTensor([]int{1, 1, 3, 3}, Float32, CPU, []float32{
			1, 0, -1,
			1, 0, -1,
			1, 0, -1,
		})

		result, err := Conv2DMPS(input, weights, nil, 1, 1, 0, 0, 0, 0)
		if err != nil {
			t.Fatalf("Conv2DMPS failed: %v", err)
		}

		expectedShape := []int{1, 1, 2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		resultData := cpuResult.Data.([]float32)
		t.Logf("Conv2D result: %v", resultData)

		// Basic sanity check - result should have correct number of elements
		if len(resultData) != 4 {
			t.Errorf("Expected 4 elements, got %d", len(resultData))
		}
	})

	t.Run("Conv2D with bias", func(t *testing.T) {
		// Input: 1 batch, 1 channel, 3x3 image
		input, _ := NewTensor([]int{1, 1, 3, 3}, Float32, CPU, []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		})
		
		// Weights: 1 output channel, 1 input channel, 2x2 kernel
		weights, _ := NewTensor([]int{1, 1, 2, 2}, Float32, CPU, []float32{
			1, 1,
			1, 1,
		})

		// Bias: 1 element for 1 output channel
		bias, _ := NewTensor([]int{1}, Float32, CPU, []float32{10})

		result, err := Conv2DMPS(input, weights, bias, 1, 1, 0, 0, 0, 0)
		if err != nil {
			t.Fatalf("Conv2DMPS with bias failed: %v", err)
		}

		expectedShape := []int{1, 1, 2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}
	})

	t.Run("Invalid input dimensions", func(t *testing.T) {
		// 3D input (should fail)
		input, _ := NewTensor([]int{1, 4, 4}, Float32, CPU, make([]float32, 16))
		weights, _ := NewTensor([]int{1, 1, 3, 3}, Float32, CPU, make([]float32, 9))

		_, err := Conv2DMPS(input, weights, nil, 1, 1, 0, 0, 0, 0)
		if err == nil {
			t.Error("Expected error for 3D input tensor")
		}
	})
}

func TestMaxPool2DMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Basic MaxPool2D", func(t *testing.T) {
		// Input: 1 batch, 1 channel, 4x4 image
		input, _ := NewTensor([]int{1, 1, 4, 4}, Float32, CPU, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})

		result, err := MaxPool2DMPS(input, 2, 2, 0) // 2x2 kernel, stride 2, no padding
		if err != nil {
			t.Fatalf("MaxPool2DMPS failed: %v", err)
		}

		expectedShape := []int{1, 1, 2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		resultData := cpuResult.Data.([]float32)
		t.Logf("MaxPool2D result: %v", resultData)

		// Basic sanity check - result should have correct number of elements
		if len(resultData) != 4 {
			t.Errorf("Expected 4 elements, got %d", len(resultData))
		}
	})

	t.Run("Invalid input dimensions", func(t *testing.T) {
		// 3D input (should fail)
		input, _ := NewTensor([]int{1, 4, 4}, Float32, CPU, make([]float32, 16))

		_, err := MaxPool2DMPS(input, 2, 2, 0)
		if err == nil {
			t.Error("Expected error for 3D input tensor")
		}
	})
}

func TestAvgPool2DMPS(t *testing.T) {
	_, err := GetMPSGraphEngine()
	if err != nil {
		t.Skipf("MPSGraph not available on this system: %v", err)
	}

	t.Run("Basic AvgPool2D", func(t *testing.T) {
		// Input: 1 batch, 1 channel, 4x4 image
		input, _ := NewTensor([]int{1, 1, 4, 4}, Float32, CPU, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})

		result, err := AvgPool2DMPS(input, 2, 2, 0) // 2x2 kernel, stride 2, no padding
		if err != nil {
			t.Fatalf("AvgPool2DMPS failed: %v", err)
		}

		expectedShape := []int{1, 1, 2, 2}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("Result shape = %v, expected %v", result.Shape, expectedShape)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}

		// Convert result back to CPU for verification
		cpuResult, err := result.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert result to CPU: %v", err)
		}

		resultData := cpuResult.Data.([]float32)
		t.Logf("AvgPool2D result: %v", resultData)

		// Basic sanity check - result should have correct number of elements
		if len(resultData) != 4 {
			t.Errorf("Expected 4 elements, got %d", len(resultData))
		}
	})

	t.Run("Invalid input dimensions", func(t *testing.T) {
		// 3D input (should fail)
		input, _ := NewTensor([]int{1, 4, 4}, Float32, CPU, make([]float32, 16))

		_, err := AvgPool2DMPS(input, 2, 2, 0)
		if err == nil {
			t.Error("Expected error for 3D input tensor")
		}
	})
}