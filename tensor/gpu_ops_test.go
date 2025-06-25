package tensor

import (
	"reflect"
	"testing"
)

func TestIsGPUAvailable(t *testing.T) {
	available := IsGPUAvailable()
	if !available {
		t.Skip("GPU not available on this system")
	}
	t.Log("GPU is available")
}

func TestGPUInfo(t *testing.T) {
	info, err := GPUInfo()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	
	if info == "" {
		t.Error("GPU info should not be empty")
	}
	
	t.Logf("GPU Info: %s", info)
}

func TestAddGPU(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available on this system")
	}

	t.Run("Float32 addition", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5.0, 6.0, 7.0, 8.0})

		result, err := AddGPU(a, b)
		if err != nil {
			t.Fatalf("AddGPU failed: %v", err)
		}

		expected := []float32{6.0, 8.0, 10.0, 12.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("AddGPU result = %v, expected %v", resultData, expected)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}
	})

	t.Run("Int32 addition", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		result, err := AddGPU(a, b)
		if err != nil {
			t.Fatalf("AddGPU failed: %v", err)
		}

		expected := []int32{6, 8, 10, 12}
		resultData := result.Data.([]int32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("AddGPU result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})

		_, err := AddGPU(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
		}
	})
}

func TestReLUGPU(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available on this system")
	}

	t.Run("Float32 ReLU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{-1.0, 2.0, -3.0, 4.0})

		result, err := ReLUGPU(a)
		if err != nil {
			t.Fatalf("ReLUGPU failed: %v", err)
		}

		expected := []float32{0.0, 2.0, 0.0, 4.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("ReLUGPU result = %v, expected %v", resultData, expected)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}
	})

	t.Run("Int32 ReLU (fallback to CPU)", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{-1, 2, -3, 4})

		result, err := ReLUGPU(a)
		if err != nil {
			t.Fatalf("ReLUGPU failed: %v", err)
		}

		expected := []int32{0, 2, 0, 4}
		resultData := result.Data.([]int32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("ReLUGPU result = %v, expected %v", resultData, expected)
		}
	})
}

func TestMatMulGPU(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available on this system")
	}

	t.Run("2x3 * 3x2 = 2x2", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{3, 2}, Float32, CPU, []float32{7, 8, 9, 10, 11, 12})

		result, err := MatMulGPU(a, b)
		if err != nil {
			t.Fatalf("MatMulGPU failed: %v", err)
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
			t.Errorf("MatMulGPU result = %v, expected %v", resultData, expected)
		}

		if result.Device != GPU {
			t.Errorf("Result device = %v, expected %v", result.Device, GPU)
		}
	})

	t.Run("Incompatible dimensions", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{7, 8, 9, 10})

		_, err := MatMulGPU(a, b)
		if err == nil {
			t.Error("Expected error for incompatible dimensions")
		}
	})
}

func TestToGPU(t *testing.T) {
	t.Run("CPU to GPU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})

		gpuTensor, err := a.ToGPU()
		if err != nil {
			t.Fatalf("ToGPU failed: %v", err)
		}

		if gpuTensor.Device != GPU {
			t.Errorf("GPU tensor device = %v, expected %v", gpuTensor.Device, GPU)
		}

		// Data should be the same
		originalData := a.Data.([]float32)
		gpuData := gpuTensor.Data.([]float32)
		if !reflect.DeepEqual(originalData, gpuData) {
			t.Error("Data should be preserved during device transfer")
		}
	})

	t.Run("Already GPU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, GPU, []float32{1, 2, 3, 4})

		gpuTensor, err := a.ToGPU()
		if err != nil {
			t.Fatalf("ToGPU failed: %v", err)
		}

		if gpuTensor != a {
			t.Error("ToGPU should return same tensor if already on GPU")
		}
	})
}

func TestToCPU(t *testing.T) {
	t.Run("GPU to CPU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, GPU, []float32{1, 2, 3, 4})

		cpuTensor, err := a.ToCPU()
		if err != nil {
			t.Fatalf("ToCPU failed: %v", err)
		}

		if cpuTensor.Device != CPU {
			t.Errorf("CPU tensor device = %v, expected %v", cpuTensor.Device, CPU)
		}

		// Data should be the same
		originalData := a.Data.([]float32)
		cpuData := cpuTensor.Data.([]float32)
		if !reflect.DeepEqual(originalData, cpuData) {
			t.Error("Data should be preserved during device transfer")
		}
	})

	t.Run("Already CPU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})

		cpuTensor, err := a.ToCPU()
		if err != nil {
			t.Fatalf("ToCPU failed: %v", err)
		}

		if cpuTensor != a {
			t.Error("ToCPU should return same tensor if already on CPU")
		}
	})
}

func TestGPUPerformance(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available on this system")
	}

	size := 10000
	data1 := make([]float32, size)
	data2 := make([]float32, size)

	for i := 0; i < size; i++ {
		data1[i] = float32(i)
		data2[i] = float32(i * 2)
	}

	a, _ := NewTensor([]int{size}, Float32, CPU, data1)
	b, _ := NewTensor([]int{size}, Float32, CPU, data2)

	// Test GPU performance
	result, err := AddGPU(a, b)
	if err != nil {
		t.Fatalf("GPU addition failed: %v", err)
	}

	// Verify some results
	resultData := result.Data.([]float32)
	for i := 0; i < 10; i++ {
		expected := float32(i * 3)
		if resultData[i] != expected {
			t.Errorf("Result[%d] = %f, expected %f", i, resultData[i], expected)
		}
	}

	t.Logf("Successfully processed %d elements on GPU", size)
}