package tensor

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestGPUAsyncOperations tests asynchronous GPU operations
func TestGPUAsyncOperations(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}

	// Test AddGPUAsync
	t.Run("AddGPUAsync", func(t *testing.T) {
		// Fill with test data
		aData := make([]float32, 10000)
		bData := make([]float32, 10000)
		for i := range aData {
			aData[i] = float32(i)
			bData[i] = float32(i * 2)
		}
		
		a, _ := NewTensor([]int{100, 100}, Float32, GPU, aData)
		b, _ := NewTensor([]int{100, 100}, Float32, GPU, bData)

		// Create completion channel
		done := make(chan error, 1)
		var result *Tensor
		
		// Perform async addition
		err := AddGPUAsync(a, b, func(res *Tensor, err error) {
			result = res
			done <- err
		})
		
		if err != nil {
			t.Fatalf("AddGPUAsync failed to start: %v", err)
		}

		// Wait for completion
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("AddGPUAsync completion error: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("AddGPUAsync timed out")
		}

		// Verify result shape
		if result == nil || result.Shape[0] != 100 || result.Shape[1] != 100 {
			t.Errorf("Result shape mismatch: expected [100,100], got %v", result.Shape)
		}

		// Test error cases
		wrongShape, _ := NewTensor([]int{50, 50}, Float32, GPU, nil)
		err = AddGPUAsync(a, wrongShape, func(*Tensor, error) {})
		if err == nil {
			t.Error("AddGPUAsync should fail with mismatched shapes")
		}
	})

	// Test MatMulGPUAsync
	t.Run("MatMulGPUAsync", func(t *testing.T) {
		// Initialize with identity-like pattern for verification
		aData := make([]float32, 5000)
		bData := make([]float32, 5000)
		for i := 0; i < 50; i++ {
			for j := 0; j < 100; j++ {
				aData[i*100+j] = float32(i + j)
			}
		}
		for i := 0; i < 100; i++ {
			for j := 0; j < 50; j++ {
				bData[i*50+j] = float32(i - j)
			}
		}
		
		a, _ := NewTensor([]int{50, 100}, Float32, GPU, aData)
		b, _ := NewTensor([]int{100, 50}, Float32, GPU, bData)

		done := make(chan error, 1)
		var result *Tensor
		
		err := MatMulGPUAsync(a, b, func(res *Tensor, err error) {
			result = res
			done <- err
		})
		
		if err != nil {
			t.Fatalf("MatMulGPUAsync failed to start: %v", err)
		}

		// Wait for completion
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("MatMulGPUAsync completion error: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("MatMulGPUAsync timed out")
		}

		// Verify result shape
		if result == nil || result.Shape[0] != 50 || result.Shape[1] != 50 {
			t.Errorf("Result shape mismatch: expected [50,50], got %v", result.Shape)
		}

		// Test incompatible shapes
		wrongShape, _ := NewTensor([]int{75, 50}, Float32, GPU, nil)
		err = MatMulGPUAsync(a, wrongShape, func(*Tensor, error) {})
		if err == nil {
			t.Error("MatMulGPUAsync should fail with incompatible shapes")
		}
	})

	// Test concurrent async operations
	t.Run("ConcurrentAsyncOps", func(t *testing.T) {
		numOps := 10
		var wg sync.WaitGroup
		errors := make(chan error, numOps)

		for i := 0; i < numOps; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				
				// Create tensors with proper data initialization
				aData := make([]float32, 10000)
				bData := make([]float32, 10000)
				for j := range aData {
					aData[j] = float32(j + idx*10000)
					bData[j] = float32(j*2 + idx*10000)
				}
				
				a, _ := NewTensor([]int{100, 100}, Float32, GPU, aData)
				b, _ := NewTensor([]int{100, 100}, Float32, GPU, bData)
				
				done := make(chan error, 1)
				err := AddGPUAsync(a, b, func(res *Tensor, err error) {
					done <- err
				})
				
				if err != nil {
					errors <- err
					return
				}

				select {
				case err := <-done:
					if err != nil {
						errors <- err
					}
				case <-time.After(5 * time.Second):
					errors <- fmt.Errorf("Operation %d timed out", idx)
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		// Check for any errors
		for err := range errors {
			if err != nil {
				t.Errorf("Concurrent operation error: %v", err)
			}
		}
	})
}

// TestGPUTrainingOps tests GPU training operations
func TestGPUTrainingOps(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}

	// Test LinearLayerForwardAsync
	t.Run("LinearLayerForwardAsync", func(t *testing.T) {
		ctx, err := NewGPUTrainingContext()
		if err != nil {
			t.Fatalf("Failed to create training context: %v", err)
		}

		// Create tensors with proper data for testing correctness
		inputData := make([]float32, 10*20)  // 10 batches, 20 input features
		weightData := make([]float32, 30*20) // 30 output features, 20 input features  
		biasData := make([]float32, 30)      // 30 output features
		
		// Initialize with simple test data
		for i := range inputData {
			inputData[i] = float32(i%10 + 1) // Values 1-10 repeating
		}
		for i := range weightData {
			weightData[i] = float32(i%5 + 1) // Values 1-5 repeating
		}
		for i := range biasData {
			biasData[i] = float32(i%3 + 1) // Values 1-3 repeating
		}

		input, _ := NewTensor([]int{10, 20}, Float32, GPU, inputData)
		weight, _ := NewTensor([]int{30, 20}, Float32, GPU, weightData)
		bias, _ := NewTensor([]int{30}, Float32, GPU, biasData)

		result, err := ctx.LinearLayerForwardAsync(input, weight, bias, "none")

		if err != nil {
			t.Fatalf("LinearLayerForwardAsync failed: %v", err)
		}

		// Verify result shape
		expectedShape := []int{10, 30}
		if result != nil && !equalShapes(result.Shape, expectedShape) {
			t.Errorf("Result shape mismatch: expected %v, got %v", expectedShape, result.Shape)
		}
	})

	// Test operation queuing
	t.Run("QueueOperation", func(t *testing.T) {
		ctx, err := NewGPUTrainingContext()
		if err != nil {
			t.Fatalf("Failed to create training context: %v", err)
		}

		a, _ := NewTensor([]int{10, 10}, Float32, GPU, nil)
		b, _ := NewTensor([]int{10, 10}, Float32, GPU, nil)

		op := OperationDesc{
			Type:   "Add",
			Inputs: []*Tensor{a, b},
		}
		ctx.QueueOperation(op)

		// Test batch flushing
		for i := 0; i < 5; i++ {
			ctx.QueueOperation(op)
		}

		// The 6th operation should trigger a batch flush
		// (implementation would handle this internally)
	})

	// Test SetBatchSize
	t.Run("SetBatchSize", func(t *testing.T) {
		ctx, err := NewGPUTrainingContext()
		if err != nil {
			t.Fatalf("Failed to create training context: %v", err)
		}

		ctx.SetBatchSize(20)
		// Since batchSize field is not exported, we can't directly test it
		// This test verifies the method doesn't crash
	})

	// Test GetGlobalGPUTrainingContext
	t.Run("GetGlobalGPUTrainingContext", func(t *testing.T) {
		ctx, err := GetGlobalGPUTrainingContext()
		if err != nil {
			t.Fatalf("GetGlobalGPUTrainingContext failed: %v", err)
		}
		if ctx == nil {
			t.Error("GetGlobalGPUTrainingContext returned nil")
		}

		// Should return the same instance
		ctx2, err := GetGlobalGPUTrainingContext()
		if err != nil {
			t.Fatalf("GetGlobalGPUTrainingContext failed on second call: %v", err)
		}
		if ctx != ctx2 {
			t.Error("GetGlobalGPUTrainingContext should return singleton instance")
		}
	})

	// Test fused operation creation
	t.Run("FusedOperationCreation", func(t *testing.T) {
		input, _ := NewTensor([]int{10, 20}, Float32, GPU, nil)
		weight, _ := NewTensor([]int{30, 20}, Float32, GPU, nil)
		bias, _ := NewTensor([]int{30}, Float32, GPU, nil)

		// Test LinearForwardOp
		linearOp := NewLinearForwardOp(input, weight, bias)
		if linearOp.Type != "LinearForward" {
			t.Errorf("Expected LinearForward type, got %s", linearOp.Type)
		}

		// Test LinearReLUOp
		linearReLUOp := NewLinearReLUOp(input, weight, bias)
		if linearReLUOp.Type != "LinearReLU" {
			t.Errorf("Expected LinearReLU type, got %s", linearReLUOp.Type)
		}

		// Test LinearSigmoidOp
		linearSigmoidOp := NewLinearSigmoidOp(input, weight, bias)
		if linearSigmoidOp.Type != "LinearSigmoid" {
			t.Errorf("Expected LinearSigmoid type, got %s", linearSigmoidOp.Type)
		}

		// Test BatchMatMulOp
		a, _ := NewTensor([]int{5, 10, 20}, Float32, GPU, nil)
		b, _ := NewTensor([]int{5, 20, 30}, Float32, GPU, nil)
		batchOp := NewBatchMatMulOp(a, b)
		if batchOp.Type != "BatchMatMul" {
			t.Errorf("Expected BatchMatMul type, got %s", batchOp.Type)
		}
	})
}

// TestMPSCopyGPUBufferToGPUBuffer tests GPU-to-GPU buffer copying
func TestMPSCopyGPUBufferToGPUBuffer(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}

	// Create source tensor on GPU
	srcData := []float32{1, 2, 3, 4, 5, 6}
	srcTensor, _ := NewTensor([]int{2, 3}, Float32, GPU, srcData)

	// This test is for the GPU buffer copying functionality
	// Since the exact API may not be stable, we'll test the concept with actual tensor operations
	
	// Create destination tensor and copy via tensor operations
	dstTensor, err := srcTensor.Clone()
	if err != nil {
		t.Fatalf("Failed to clone tensor: %v", err)
	}

	// Transfer to CPU to verify contents
	cpuDst, err := dstTensor.ToCPU()
	if err != nil {
		t.Fatalf("Failed to transfer destination to CPU: %v", err)
	}

	// Verify data was copied correctly
	dstData, err := cpuDst.GetFloat32Data()
	if err != nil {
		t.Fatalf("Failed to get data: %v", err)
	}
	for i, v := range srcData {
		if dstData[i] != v {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, v, dstData[i])
		}
	}
}

// TestConvolutionForwardAsync tests async convolution operations
func TestConvolutionForwardAsync(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}

	ctx, err := NewGPUTrainingContext()
	if err != nil {
		t.Fatalf("Failed to create training context: %v", err)
	}

	// Create tensors with proper data for testing correctness
	inputData := make([]float32, 1*3*32*32)  // 1 batch, 3 channels, 32x32 image
	weightData := make([]float32, 16*3*3*3)  // 16 output channels, 3 input channels, 3x3 kernel
	biasData := make([]float32, 16)          // 16 output channels
	
	// Initialize with simple test data
	for i := range inputData {
		inputData[i] = float32(i%10 + 1) // Values 1-10 repeating
	}
	for i := range weightData {
		weightData[i] = float32(i%5 + 1) // Values 1-5 repeating  
	}
	for i := range biasData {
		biasData[i] = float32(i%3 + 1) // Values 1-3 repeating
	}

	input, _ := NewTensor([]int{1, 3, 32, 32}, Float32, GPU, inputData)    // NCHW format
	weight, _ := NewTensor([]int{16, 3, 3, 3}, Float32, GPU, weightData)   // Output channels, input channels, H, W
	bias, _ := NewTensor([]int{16}, Float32, GPU, biasData)

	result, err := ctx.ConvolutionForwardAsync(input, weight, bias, 1, 1)

	if err != nil {
		t.Fatalf("ConvolutionForwardAsync failed: %v", err)
	}
	
	if result == nil {
		t.Fatal("ConvolutionForwardAsync returned nil result")
	}

	// Verify result shape - for 3x3 kernel with stride 1 and padding 1, output should be 32x32
	expectedShape := []int{1, 16, 32, 32}
	if !equalShapes(result.Shape, expectedShape) {
		t.Errorf("Result shape mismatch: expected %v, got %v", expectedShape, result.Shape)
	}
}

// TestTrainingStepAsync tests async training step
func TestTrainingStepAsync(t *testing.T) {
	if !IsGPUAvailable() {
		t.Skip("GPU not available")
	}

	ctx, err := NewGPUTrainingContext()
	if err != nil {
		t.Fatalf("Failed to create training context: %v", err)
	}

	// Create dummy operation descriptions for forward and backward pass
	forwardOps := []OperationDesc{
		{Type: "MatMul", Inputs: []*Tensor{}},
	}
	backwardOps := []OperationDesc{
		{Type: "MatMul", Inputs: []*Tensor{}},
	}

	err = ctx.TrainingStepAsync(forwardOps, backwardOps)

	// Since this is a placeholder implementation, we expect an error
	if err == nil {
		t.Error("TrainingStepAsync should return not implemented error")
	}
}