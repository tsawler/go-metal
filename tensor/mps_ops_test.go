package tensor

import (
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

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		_, err := AddMPS(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
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