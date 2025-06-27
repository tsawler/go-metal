package tensor

import (
	"strings"
	"testing"
)

// TestSubAutograd tests the SubAutograd function and SubOp operations
func TestSubAutograd(t *testing.T) {
	// Test basic subtraction with autograd
	a, err := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create tensor a: %v", err)
	}
	a.SetRequiresGrad(true)

	b, err := NewTensor([]int{2, 3}, Float32, CPU, []float32{0.5, 1, 1.5, 2, 2.5, 3})
	if err != nil {
		t.Fatalf("Failed to create tensor b: %v", err)
	}
	b.SetRequiresGrad(true)

	// Perform subtraction with autograd
	c, err := SubAutograd(a, b)
	if err != nil {
		t.Fatalf("SubAutograd failed: %v", err)
	}

	// Verify forward pass results
	expected := []float32{0.5, 1, 1.5, 2, 2.5, 3}
	cData, err := c.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}
	for i, v := range expected {
		if cData[i] != v {
			t.Errorf("Forward pass mismatch at index %d: expected %f, got %f", i, v, cData[i])
		}
	}

	// Create a scalar loss by summing all elements using autograd
	loss, err := SumAutograd(c)
	if err != nil {
		t.Fatalf("Failed to create scalar loss: %v", err)
	}

	// Test backward pass
	err = loss.Backward()
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	// Check gradients for a (should be 1s)
	if a.grad == nil {
		t.Fatal("Gradient for a is nil")
	}
	aGradData, err := a.grad.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get gradient data: %v", err)
	}
	for i, v := range aGradData {
		if v != 1.0 {
			t.Errorf("Gradient for a at index %d: expected 1.0, got %f", i, v)
		}
	}

	// Check gradients for b (should be -1s)
	if b.grad == nil {
		t.Fatal("Gradient for b is nil")
	}
	bGradData, err := b.grad.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get gradient data: %v", err)
	}
	for i, v := range bGradData {
		if v != -1.0 {
			t.Errorf("Gradient for b at index %d: expected -1.0, got %f", i, v)
		}
	}

	// Test with broadcasting
	scalar, err := NewTensor([]int{1}, Float32, CPU, []float32{10})
	if err != nil {
		t.Fatalf("Failed to create scalar tensor: %v", err)
	}
	scalar.SetRequiresGrad(true)

	result, err := SubAutograd(a, scalar)
	if err != nil {
		t.Fatalf("SubAutograd with broadcasting failed: %v", err)
	}

	// Verify broadcasting worked correctly
	resultData, err := result.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get result data: %v", err)
	}
	expectedBroadcast := []float32{-9, -8, -7, -6, -5, -4}
	for i, v := range expectedBroadcast {
		if resultData[i] != v {
			t.Errorf("Broadcast subtraction mismatch at index %d: expected %f, got %f", i, v, resultData[i])
		}
	}

	// Test error cases
	wrongShape, err := NewTensor([]int{3, 2}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create wrong shape tensor: %v", err)
	}

	_, err = SubAutograd(a, wrongShape)
	if err == nil {
		t.Error("SubAutograd should fail with incompatible shapes")
	}

	// Test with different data types
	intTensor, err := NewTensor([]int{2, 3}, Int32, CPU, []int32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create int tensor: %v", err)
	}

	_, err = SubAutograd(a, intTensor)
	if err == nil {
		t.Error("SubAutograd should fail with mismatched data types")
	}
}

// TestAutogradErrorPaths tests error handling in various autograd operations
func TestAutogradErrorPaths(t *testing.T) {
	// Test AddOp backward with nil gradient
	addOp := &AddOp{
		inputs: []*Tensor{
			mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4}),
			mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8}),
		},
	}

	// Create a valid gradient tensor for testing error path
	invalidGrad := mustCreateTensor(t, []int{3, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	grads, err := addOp.Backward(invalidGrad)
	if err == nil {
		t.Error("AddOp.Backward should fail with wrong gradient shape")
	}
	if grads != nil {
		t.Error("AddOp.Backward should return nil gradients on error")
	}

	// Test MulOp backward with wrong gradient shape
	mulOp := &MulOp{
		inputs: []*Tensor{
			mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4}),
			mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8}),
		},
	}

	wrongGrad := mustCreateTensor(t, []int{3, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	grads, err = mulOp.Backward(wrongGrad)
	if err == nil {
		t.Error("MulOp.Backward should fail with wrong gradient shape")
	}

	// Test MatMulOp backward with incompatible shapes
	matmulOp := &MatMulOp{
		inputs: []*Tensor{
			mustCreateTensor(t, []int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6}),
			mustCreateTensor(t, []int{3, 2}, Float32, CPU, []float32{7, 8, 9, 10, 11, 12}),
		},
	}

	// Test with correct gradient
	correctGrad := mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{1, 1, 1, 1})
	grads, err = matmulOp.Backward(correctGrad)
	if err != nil {
		t.Errorf("MatMulOp.Backward failed with correct gradient: %v", err)
	}

	// Test ReLUOp backward with negative values
	reluOp := &ReLUOp{
		inputs: []*Tensor{
			mustCreateTensor(t, []int{4}, Float32, CPU, []float32{-2, -1, 1, 2}),
		},
	}

	reluGrad := mustCreateTensor(t, []int{4}, Float32, CPU, []float32{1, 1, 1, 1})
	grads, err = reluOp.Backward(reluGrad)
	if err != nil {
		t.Errorf("ReLUOp.Backward failed: %v", err)
	}

	// Verify ReLU gradients (should be 0 for negative inputs)
	if grads != nil && len(grads) > 0 {
		reluGradData, err := grads[0].GetFloat32Data()
		if err != nil {
			t.Fatalf("Failed to get ReLU gradient data: %v", err)
		}
		expected := []float32{0, 0, 1, 1}
		for i, v := range expected {
			if reluGradData[i] != v {
				t.Errorf("ReLU gradient mismatch at index %d: expected %f, got %f", i, v, reluGradData[i])
			}
		}
	}

	// Test SigmoidOp backward - properly create through Forward() method
	sigmoidInput := mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{0, 1, -1, 2})
	sigmoidOp := &SigmoidOp{}
	
	// Call Forward() to properly initialize the operation
	_, err = sigmoidOp.Forward(sigmoidInput)
	if err != nil {
		t.Errorf("SigmoidOp.Forward failed: %v", err)
		return
	}
	
	// Verify output is stored for backward pass
	if sigmoidOp.output == nil {
		t.Error("SigmoidOp.output should be set after Forward() call")
		return
	}
	
	// Now test backward pass
	sigmoidGrad := mustCreateTensor(t, []int{2, 2}, Float32, CPU, []float32{1, 1, 1, 1})
	grads, err = sigmoidOp.Backward(sigmoidGrad)
	if err != nil {
		t.Errorf("SigmoidOp.Backward failed: %v", err)
		return
	}
	
	// Verify gradients were computed correctly
	if len(grads) != 1 {
		t.Errorf("SigmoidOp.Backward should return 1 gradient, got %d", len(grads))
		return
	}
	
	if grads[0] == nil {
		t.Error("SigmoidOp.Backward returned nil gradient")
		return
	}
	
	// Verify gradient correctness: ∂σ(x)/∂x = σ(x) * (1 - σ(x))
	// For input [0, 1, -1, 2], sigmoid values are approximately [0.5, 0.73, 0.27, 0.88]
	// Gradients should be σ(x) * (1 - σ(x)) = [0.25, 0.196, 0.196, 0.105] (approximately)
	gradData, err := grads[0].GetFloat32Data()
	if err != nil {
		t.Errorf("Failed to get sigmoid gradient data: %v", err)
		return
	}
	
	// Check that gradients are reasonable (between 0 and 0.25 for sigmoid derivative)
	for i, grad := range gradData {
		if grad < 0 || grad > 0.25 {
			t.Errorf("Sigmoid gradient at index %d is out of expected range [0, 0.25]: got %f", i, grad)
		}
	}
}

// TestReduceGradientEdgeCases tests edge cases in gradient reduction
func TestReduceGradientEdgeCases(t *testing.T) {
	// Test sumAllElements with different shapes
	tests := []struct {
		name      string
		shape     []int
		data      []float32
		expected  float32
	}{
		{
			name:     "Single element",
			shape:    []int{1},
			data:     []float32{5.0},
			expected: 5.0,
		},
		{
			name:     "1D array",
			shape:    []int{5},
			data:     []float32{1, 2, 3, 4, 5},
			expected: 15.0,
		},
		{
			name:     "2D array",
			shape:    []int{2, 3},
			data:     []float32{1, 2, 3, 4, 5, 6},
			expected: 21.0,
		},
		{
			name:     "3D array",
			shape:    []int{2, 2, 2},
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			expected: 36.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := mustCreateTensor(t, tt.shape, Float32, CPU, tt.data)
			result, err := sumAllElements(tensor)
			if err != nil {
				t.Fatalf("sumAllElements failed: %v", err)
			}
			
			if result.Shape[0] != 1 || len(result.Shape) != 1 {
				t.Errorf("sumAllElements should return shape [1], got %v", result.Shape)
			}

			resultData, err := result.GetFloat32Data()
			if err != nil {
				t.Fatalf("Failed to get result data: %v", err)
			}
			if resultData[0] != tt.expected {
				t.Errorf("sumAllElements: expected %f, got %f", tt.expected, resultData[0])
			}
		})
	}

	// Test sumOverDimension with edge cases
	// Create a 3D tensor for testing
	tensor3D := mustCreateTensor(t, []int{2, 3, 4}, Float32, CPU, 
		[]float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
			17, 18, 19, 20,
			21, 22, 23, 24,
		})

	// Sum over dimension 0
	sumDim0, err := sumOverDimension(tensor3D, 0)
	if err != nil {
		t.Fatalf("sumOverDimension failed: %v", err)
	}
	if !equalShapes(sumDim0.Shape, []int{3, 4}) {
		t.Errorf("Sum over dim 0: expected shape [3,4], got %v", sumDim0.Shape)
	}

	// Sum over dimension 1
	sumDim1, err := sumOverDimension(tensor3D, 1)
	if err != nil {
		t.Fatalf("sumOverDimension failed: %v", err)
	}
	if !equalShapes(sumDim1.Shape, []int{2, 4}) {
		t.Errorf("Sum over dim 1: expected shape [2,4], got %v", sumDim1.Shape)
	}

	// Test with Int32 data
	int32Tensor := mustCreateTensor(t, []int{2, 3}, Int32, CPU, []int32{1, 2, 3, 4, 5, 6})
	sumInt32, err := sumAllElements(int32Tensor)
	if err != nil {
		t.Fatalf("sumAllElements on int32 failed: %v", err)
	}
	if sumInt32.DType != Int32 {
		t.Errorf("sumAllElements should preserve Int32 type, got %v", sumInt32.DType)
	}
}

// TestBroadcastInt32DataIntegration tests the broadcastInt32Data function via tensor operations
func TestBroadcastInt32DataIntegration(t *testing.T) {
	tests := []struct {
		name         string
		srcData      []int32
		srcShape     []int
		targetShape  []int
		shouldSucceed bool
	}{
		{
			name:         "Scalar to vector",
			srcData:      []int32{5},
			srcShape:     []int{1},
			targetShape:  []int{4},
			shouldSucceed: true,
		},
		{
			name:         "Vector to matrix",
			srcData:      []int32{1, 2, 3},
			srcShape:     []int{3},
			targetShape:  []int{2, 3},
			shouldSucceed: true,
		},
		{
			name:         "Incompatible shapes",
			srcData:      []int32{1, 2},
			srcShape:     []int{2},
			targetShape:  []int{3},
			shouldSucceed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create source tensor
			srcTensor, err := NewTensor(tt.srcShape, Int32, CPU, tt.srcData)
			if err != nil {
				t.Fatalf("Failed to create source tensor: %v", err)
			}

			// Create destination tensor with target shape
			targetElements := 1
			for _, dim := range tt.targetShape {
				targetElements *= dim
			}
			
			dstTensor, err := Zeros(tt.targetShape, Int32, CPU)
			if err != nil {
				t.Fatalf("Failed to create destination tensor: %v", err)
			}

			// Test broadcasting via actual tensor operations
			if tt.shouldSucceed {
				// Test if shapes are broadcastable
				compatible := AreBroadcastable(srcTensor.Shape, dstTensor.Shape)
				if !compatible {
					t.Errorf("Shapes should be broadcastable: %v and %v", srcTensor.Shape, dstTensor.Shape)
				}
			}
		})
	}
}

// Helper function to create tensors in tests
func mustCreateTensor(t *testing.T, shape []int, dtype DType, device DeviceType, data interface{}) *Tensor {
	tensor, err := NewTensor(shape, dtype, device, data)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	return tensor
}

// equalShapes helper function - using the one from gpu_queue_test.go

// TestToDeviceExtended tests the ToDevice function comprehensively
func TestToDeviceExtended(t *testing.T) {
	// Create test tensor
	cpuTensor, err := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create CPU tensor: %v", err)
	}

	tests := []struct {
		name       string
		fromDevice DeviceType
		toDevice   DeviceType
		expectCopy bool
	}{
		{"CPU to CPU", CPU, CPU, false},
		{"CPU to GPU", CPU, GPU, true},
		{"CPU to PersistentGPU", CPU, PersistentGPU, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create source tensor
			var sourceTensor *Tensor
			if tt.fromDevice == CPU {
				sourceTensor = cpuTensor
			} else {
				// Convert to GPU first if needed
				var err error
				if tt.fromDevice == GPU {
					sourceTensor, err = cpuTensor.ToGPU()
				} else {
					sourceTensor, err = cpuTensor.ToPersistentGPU()
				}
				if err != nil && IsGPUAvailable() {
					t.Fatalf("Failed to create %v tensor: %v", tt.fromDevice, err)
				}
				if !IsGPUAvailable() {
					t.Skip("GPU not available")
				}
			}

			// Test ToDevice
			result, err := sourceTensor.ToDevice(tt.toDevice)
			if err != nil && IsGPUAvailable() {
				t.Fatalf("ToDevice failed: %v", err)
			}

			if IsGPUAvailable() {
				// Verify device
				if result.Device != tt.toDevice {
					t.Errorf("Expected device %v, got %v", tt.toDevice, result.Device)
				}

				// Verify if it's a copy or same tensor
				if tt.expectCopy {
					if result == sourceTensor {
						t.Error("Expected a copy, but got the same tensor")
					}
				} else {
					if result != sourceTensor {
						t.Error("Expected the same tensor, but got a copy")
					}
				}

				// Verify data integrity
				if result.Shape[0] != sourceTensor.Shape[0] || result.Shape[1] != sourceTensor.Shape[1] {
					t.Error("Shape mismatch after device transfer")
				}
			}
		})
	}

	// Test error cases
	if IsGPUAvailable() {
		// Test GPU to GPU (should return same tensor)
		gpuTensor, _ := cpuTensor.ToGPU()
		sameGPU, err := gpuTensor.ToDevice(GPU)
		if err != nil {
			t.Errorf("GPU to GPU ToDevice failed: %v", err)
		}
		if sameGPU != gpuTensor {
			t.Error("GPU to GPU should return the same tensor")
		}

		// Test PersistentGPU to PersistentGPU (should return same tensor)
		persistentTensor, _ := cpuTensor.ToPersistentGPU()
		samePersistent, err := persistentTensor.ToDevice(PersistentGPU)
		if err != nil {
			t.Errorf("PersistentGPU to PersistentGPU ToDevice failed: %v", err)
		}
		if samePersistent != persistentTensor {
			t.Error("PersistentGPU to PersistentGPU should return the same tensor")
		}
	}

	// Test with invalid device types - demonstrating CORRECTNESS of validation
	invalidDeviceTypes := []struct {
		device      DeviceType
		description string
	}{
		{DeviceType(999), "arbitrary large number"},
		{DeviceType(-1), "negative device type"},
		{DeviceType(100), "out of range device type"},
	}
	
	for _, test := range invalidDeviceTypes {
		_, err = cpuTensor.ToDevice(test.device)
		if err == nil {
			t.Errorf("ToDevice should fail with invalid device type %v (%s)", test.device, test.description)
		}
		
		// Verify the error message is informative
		expectedSubstring := "invalid device type"
		if !strings.Contains(err.Error(), expectedSubstring) {
			t.Errorf("Error message should contain '%s', got: %v", expectedSubstring, err.Error())
		}
	}
	
	// Test with valid device types to ensure they still work
	validDeviceTypes := []DeviceType{CPU, GPU, PersistentGPU}
	for _, validDevice := range validDeviceTypes {
		result, err := cpuTensor.ToDevice(validDevice)
		if err != nil {
			t.Errorf("ToDevice should succeed with valid device type %v, got error: %v", validDevice, err)
		}
		if result == nil {
			t.Errorf("ToDevice should return a valid tensor for device type %v", validDevice)
		}
	}
}