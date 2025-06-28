package tensor

import (
	"testing"
)

// TestTensorGPUMethods tests GPU-specific tensor methods
func TestTensorGPUMethods(t *testing.T) {
	// Test CPU tensor (baseline)
	cpuTensor, err := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create CPU tensor: %v", err)
	}

	// Test IsOnGPU for CPU tensor
	if cpuTensor.IsOnGPU() {
		t.Error("CPU tensor incorrectly reports as being on GPU")
	}

	// Test IsPersistent for CPU tensor
	if cpuTensor.IsPersistent() {
		t.Error("CPU tensor incorrectly reports as persistent")
	}

	// Test GPU tensor
	gpuTensor, err := NewTensor([]int{2, 3}, Float32, GPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create GPU tensor: %v", err)
	}

	// Test IsOnGPU for GPU tensor
	if !gpuTensor.IsOnGPU() {
		t.Error("GPU tensor incorrectly reports as not being on GPU")
	}

	// Test IsPersistent for temporary GPU tensor
	if gpuTensor.IsPersistent() {
		t.Error("Temporary GPU tensor incorrectly reports as persistent")
	}

	// Test PersistentGPU tensor
	persistentGPUTensor, err := NewTensor([]int{2, 3}, Float32, PersistentGPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create PersistentGPU tensor: %v", err)
	}

	// Test IsOnGPU for PersistentGPU tensor
	if !persistentGPUTensor.IsOnGPU() {
		t.Error("PersistentGPU tensor incorrectly reports as not being on GPU")
	}

	// Test IsPersistent for PersistentGPU tensor
	if !persistentGPUTensor.IsPersistent() {
		t.Error("PersistentGPU tensor incorrectly reports as not persistent")
	}

	// Test ToPersistentGPU conversion
	if IsGPUAvailable() {
		persistentConverted, err := cpuTensor.ToPersistentGPU()
		if err != nil {
			t.Fatalf("Failed to convert CPU tensor to PersistentGPU: %v", err)
		}

		// Verify the converted tensor
		if persistentConverted.Device != PersistentGPU {
			t.Errorf("Expected device PersistentGPU, got %v", persistentConverted.Device)
		}

		if !persistentConverted.IsOnGPU() {
			t.Error("Converted tensor should be on GPU")
		}

		if !persistentConverted.IsPersistent() {
			t.Error("Converted tensor should be persistent")
		}

		// Verify data integrity
		if persistentConverted.Shape[0] != cpuTensor.Shape[0] || persistentConverted.Shape[1] != cpuTensor.Shape[1] {
			t.Error("Shape mismatch after conversion to PersistentGPU")
		}

		// Convert from GPU to PersistentGPU
		persistentFromGPU, err := gpuTensor.ToPersistentGPU()
		if err != nil {
			t.Fatalf("Failed to convert GPU tensor to PersistentGPU: %v", err)
		}

		if persistentFromGPU.Device != PersistentGPU {
			t.Errorf("Expected device PersistentGPU after GPU conversion, got %v", persistentFromGPU.Device)
		}

		// Test that PersistentGPU tensor returns itself
		samePersistent, err := persistentGPUTensor.ToPersistentGPU()
		if err != nil {
			t.Fatalf("Failed to convert PersistentGPU tensor to PersistentGPU: %v", err)
		}

		// Should be the same pointer
		if samePersistent != persistentGPUTensor {
			t.Error("ToPersistentGPU on PersistentGPU tensor should return the same tensor")
		}
	}
}

// TestSetDataAndSetGrad tests the SetData and SetGrad methods
func TestSetDataAndSetGrad(t *testing.T) {
	// Create a tensor
	tensor, err := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	// Test SetData with valid data
	newData := []float32{7, 8, 9, 10, 11, 12}
	tensor.SetData(newData)

	// Verify the data was set correctly
	floatData, err := tensor.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}
	for i, v := range newData {
		if floatData[i] != v {
			t.Errorf("SetData failed at index %d: expected %f, got %f", i, v, floatData[i])
		}
	}

	// Test SetData with nil
	tensor.SetData(nil)
	if tensor.Data != nil {
		t.Error("SetData(nil) should set Data to nil")
	}

	// Create a gradient tensor
	gradData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	gradTensor, err := NewTensor([]int{2, 3}, Float32, CPU, gradData)
	if err != nil {
		t.Fatalf("Failed to create gradient tensor: %v", err)
	}

	// Test SetGrad
	tensor.SetGrad(gradTensor)
	if tensor.grad != gradTensor {
		t.Error("SetGrad failed to set gradient tensor")
	}

	// Verify gradient data
	if tensor.grad == nil {
		t.Fatal("Gradient should not be nil after SetGrad")
	}

	gradFloatData, err := tensor.grad.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get gradient data: %v", err)
	}
	for i, v := range gradData {
		if gradFloatData[i] != v {
			t.Errorf("Gradient data mismatch at index %d: expected %f, got %f", i, v, gradFloatData[i])
		}
	}

	// Test SetGrad with nil
	tensor.SetGrad(nil)
	if tensor.grad != nil {
		t.Error("SetGrad(nil) should set grad to nil")
	}
}

// TestFromScalar tests the FromScalar utility function
func TestFromScalar(t *testing.T) {
	tests := []struct {
		name     string
		value    float32
		dtype    DType
		device   DeviceType
		wantErr  bool
	}{
		{"CPU Float32", 3.14, Float32, CPU, false},
		{"GPU Float32", 2.71, Float32, GPU, false},
		{"PersistentGPU Float32", 1.41, Float32, PersistentGPU, false},
		{"Zero value", 0.0, Float32, CPU, false},
		{"Negative value", -5.5, Float32, CPU, false},
		{"Large value", 1e10, Float32, CPU, false},
		{"Small value", 1e-10, Float32, CPU, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := FromScalar(float64(tt.value), tt.dtype, tt.device)
			
			if tensor == nil && !tt.wantErr {
				t.Error("FromScalar() returned nil tensor unexpectedly")
				return
			}

			if tensor != nil {
				// Verify tensor properties
				if len(tensor.Shape) != 0 {
					t.Errorf("Expected scalar shape [], got %v", tensor.Shape)
				}

				if tensor.DType != tt.dtype {
					t.Errorf("Expected dtype %v, got %v", tt.dtype, tensor.DType)
				}

				if tensor.Device != tt.device {
					t.Errorf("Expected device %v, got %v", tt.device, tensor.Device)
				}

				// Verify the value
				item, err := tensor.Item()
				if err != nil {
					t.Fatalf("Failed to get item: %v", err)
				}

				if item != tt.value {
					t.Errorf("Expected value %f, got %f", tt.value, item)
				}
			}
		})
	}
}

// TestSqrt tests the Sqrt utility function
func TestSqrt(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []float32
		shape    []int
		wantErr  bool
	}{
		{
			name:     "Simple square roots",
			input:    []float32{4, 9, 16, 25},
			expected: []float32{2, 3, 4, 5},
			shape:    []int{4},
			wantErr:  false,
		},
		{
			name:     "2D tensor",
			input:    []float32{1, 4, 9, 16},
			expected: []float32{1, 2, 3, 4},
			shape:    []int{2, 2},
			wantErr:  false,
		},
		{
			name:     "Zero values",
			input:    []float32{0, 1, 0, 4},
			expected: []float32{0, 1, 0, 2},
			shape:    []int{4},
			wantErr:  false,
		},
		{
			name:     "Fractional values",
			input:    []float32{0.25, 0.5, 2.25, 6.25},
			expected: []float32{0.5, 0.7071068, 1.5, 2.5},
			shape:    []int{4},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := NewTensor(tt.shape, Float32, CPU, tt.input)
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}

			result, err := Sqrt(input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Sqrt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Verify shape
				if len(result.Shape) != len(tt.shape) {
					t.Errorf("Shape dimension mismatch: expected %d, got %d", len(tt.shape), len(result.Shape))
				}

				for i, dim := range tt.shape {
					if result.Shape[i] != dim {
						t.Errorf("Shape mismatch at dim %d: expected %d, got %d", i, dim, result.Shape[i])
					}
				}

				// Verify values
				resultData, err := result.GetFloat32Data()
				if err != nil {
					t.Fatalf("Failed to get result data: %v", err)
				}
				for i, expected := range tt.expected {
					if !almostEqual(resultData[i], expected, 1e-6) {
						t.Errorf("Value mismatch at index %d: expected %f, got %f", i, expected, resultData[i])
					}
				}
			}
		})
	}

	// Test with Int32 tensor (should handle error gracefully)
	intTensor, err := NewTensor([]int{4}, Int32, CPU, []int32{4, 9, 16, 25})
	if err != nil {
		t.Fatalf("Failed to create int tensor: %v", err)
	}

	_, err = Sqrt(intTensor)
	if err == nil {
		t.Error("Sqrt on int tensor should fail with unsupported type error")
	}

	// Test negative values (should handle gracefully)
	negativeTensor, err := NewTensor([]int{2}, Float32, CPU, []float32{-4, -9})
	if err != nil {
		t.Fatalf("Failed to create negative tensor: %v", err)
	}

	negResult, err := Sqrt(negativeTensor)
	if err != nil {
		t.Fatalf("Sqrt on negative values failed: %v", err)
	}

	// Should produce NaN for negative values
	negData, err := negResult.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get negative result data: %v", err)
	}
	for i, v := range negData {
		if !isNaN(v) {
			t.Errorf("Expected NaN for sqrt of negative value at index %d, got %f", i, v)
		}
	}
}

// Helper function to check if float32 is NaN
func isNaN(f float32) bool {
	return f != f
}

// Helper function for float comparison
func almostEqual(a, b, epsilon float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < epsilon
}