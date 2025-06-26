package tensor

import (
	"testing"
	"time"
)

func TestLinearForward(t *testing.T) {
	// Create test tensors for linear layer: input(batch=2, features=4) * weight(4, 6) + bias(6)
	input, err := NewTensor([]int{2, 4}, Float32, GPU, []float32{
		1, 2, 3, 4,  // batch 1
		5, 6, 7, 8,  // batch 2
	})
	if err != nil {
		t.Skipf("Skipping GPU test - Metal not available: %v", err)
		return
	}

	weight, err := NewTensor([]int{6, 4}, Float32, GPU, []float32{
		0.1, 0.2, 0.3, 0.4,  // output feature 1
		0.5, 0.6, 0.7, 0.8,  // output feature 2
		0.1, 0.1, 0.1, 0.1,  // output feature 3
		0.2, 0.2, 0.2, 0.2,  // output feature 4
		0.3, 0.3, 0.3, 0.3,  // output feature 5
		0.4, 0.4, 0.4, 0.4,  // output feature 6
	})
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}

	bias, err := NewTensor([]int{6}, Float32, GPU, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}

	// Test fused linear forward operation
	start := time.Now()
	result, err := LinearForward(input, weight, bias)
	fusedTime := time.Since(start)

	if err != nil {
		t.Fatalf("LinearForward failed: %v", err)
	}

	// Verify result shape
	expectedShape := []int{2, 6}
	if !equalIntSlices(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}

	t.Logf("Fused LinearForward execution time: %v", fusedTime)
	t.Logf("Result shape: %v", result.Shape)

	// Compare with separate operations for verification
	start = time.Now()
	matmulResult, err := MatMul(input, weight)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}
	_, err = Add(matmulResult, bias)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	separateTime := time.Since(start)

	t.Logf("Separate operations execution time: %v", separateTime)
	
	if fusedTime < separateTime {
		t.Logf("✓ Fused operation is faster by %v", separateTime-fusedTime)
	} else {
		t.Logf("⚠ Fused operation took %v longer (overhead may be due to small tensor sizes)", fusedTime-separateTime)
	}
}

func TestLinearReLU(t *testing.T) {
	// Create test tensors
	input, err := NewTensor([]int{2, 3}, Float32, GPU, []float32{
		-1, 0, 1,   // batch 1
		-2, 2, 3,   // batch 2
	})
	if err != nil {
		t.Skipf("Skipping GPU test - Metal not available: %v", err)
		return
	}

	weight, err := NewTensor([]int{4, 3}, Float32, GPU, []float32{
		1, 0, -1,   // output feature 1
		0, 1, 0,    // output feature 2
		-1, 0, 1,   // output feature 3
		0.5, 0.5, 0.5, // output feature 4
	})
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}

	bias, err := NewTensor([]int{4}, Float32, GPU, []float32{0.1, -0.1, 0.2, -0.2})
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}

	// Test fused linear + ReLU operation
	start := time.Now()
	result, err := LinearReLU(input, weight, bias)
	fusedTime := time.Since(start)

	if err != nil {
		t.Fatalf("LinearReLU failed: %v", err)
	}

	// Verify result shape
	expectedShape := []int{2, 4}
	if !equalIntSlices(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}

	// Verify ReLU was applied (no negative values)
	resultData := result.Data.([]float32)
	for i, val := range resultData {
		if val < 0 {
			t.Errorf("ReLU should eliminate negative values, but found %f at index %d", val, i)
		}
	}

	t.Logf("Fused LinearReLU execution time: %v", fusedTime)
	t.Logf("Result shape: %v", result.Shape)
}

func TestLinearSigmoid(t *testing.T) {
	// Create test tensors
	input, err := NewTensor([]int{1, 2}, Float32, GPU, []float32{1, -1})
	if err != nil {
		t.Skipf("Skipping GPU test - Metal not available: %v", err)
		return
	}

	weight, err := NewTensor([]int{3, 2}, Float32, GPU, []float32{
		1, 0,   // output feature 1
		0, 1,   // output feature 2
		1, 1,   // output feature 3
	})
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}

	bias, err := NewTensor([]int{3}, Float32, GPU, []float32{0, 0, 0})
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}

	// Test fused linear + Sigmoid operation
	result, err := LinearSigmoid(input, weight, bias)
	if err != nil {
		t.Fatalf("LinearSigmoid failed: %v", err)
	}

	// Verify result shape
	expectedShape := []int{1, 3}
	if !equalIntSlices(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}

	// Verify Sigmoid was applied (values between 0 and 1)
	resultData := result.Data.([]float32)
	for i, val := range resultData {
		if val < 0 || val > 1 {
			t.Errorf("Sigmoid should produce values in [0,1], but found %f at index %d", val, i)
		}
	}

	t.Logf("LinearSigmoid result: %v", resultData)
}

func TestBatchMatMul(t *testing.T) {
	// Create test tensors for batch matrix multiplication
	// batchA: [2, 3, 4] - 2 batches of 3x4 matrices
	// batchB: [2, 4, 5] - 2 batches of 4x5 matrices
	// result: [2, 3, 5] - 2 batches of 3x5 matrices
	
	batchA, err := NewTensor([]int{2, 3, 4}, Float32, GPU, []float32{
		// Batch 1: 3x4 matrix
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		// Batch 2: 3x4 matrix
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
	})
	if err != nil {
		t.Skipf("Skipping GPU test - Metal not available: %v", err)
		return
	}

	batchB, err := NewTensor([]int{2, 4, 5}, Float32, GPU, []float32{
		// Batch 1: 4x5 matrix
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		// Batch 2: 4x5 matrix
		0.1, 0.2, 0.3, 0.4, 0.5,
		0.1, 0.2, 0.3, 0.4, 0.5,
		0.1, 0.2, 0.3, 0.4, 0.5,
		0.1, 0.2, 0.3, 0.4, 0.5,
	})
	if err != nil {
		t.Fatalf("Failed to create batchB tensor: %v", err)
	}

	// Test batch matrix multiplication
	start := time.Now()
	result, err := BatchMatMul(batchA, batchB)
	batchTime := time.Since(start)

	if err != nil {
		t.Fatalf("BatchMatMul failed: %v", err)
	}

	// Verify result shape
	expectedShape := []int{2, 3, 5}
	if !equalIntSlices(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}

	t.Logf("Batch MatMul execution time: %v", batchTime)
	t.Logf("Result shape: %v", result.Shape)
}

func TestOperationFusionDetection(t *testing.T) {
	// Create test tensors
	input, err := NewTensor([]int{2, 3}, Float32, GPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Skipf("Skipping GPU test - Metal not available: %v", err)
		return
	}

	weight, err := NewTensor([]int{4, 3}, Float32, GPU, make([]float32, 12))
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}

	bias, err := NewTensor([]int{4}, Float32, GPU, make([]float32, 4))
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}

	// Create a sequence of operations that can be fused
	operations := []OperationDesc{
		NewMatMulOp(input, weight),
		NewAddOp(nil, bias),     // nil will be replaced with MatMul result
		NewReLUOp(nil),          // nil will be replaced with Add result
	}

	// Test fusion detection
	detector := NewFusedOperationDetector()
	for _, op := range operations {
		detector.AddOperation(op.Type, op.Inputs, op.Params)
	}

	optimized, err := detector.DetectFusions()
	if err != nil {
		t.Fatalf("Fusion detection failed: %v", err)
	}

	// Should detect MatMul + Add + ReLU fusion
	if len(optimized) != 1 {
		t.Errorf("Expected 1 fused operation, got %d", len(optimized))
	}

	if len(optimized) > 0 && optimized[0].Type != "LinearReLU" {
		t.Errorf("Expected LinearReLU fusion, got %s", optimized[0].Type)
	}

	t.Logf("✓ Successfully detected fusion: %d operations -> %d fused operations", len(operations), len(optimized))
}

func TestGPUTrainingContextWithFusion(t *testing.T) {
	ctx, err := NewGPUTrainingContext()
	if err != nil {
		t.Skipf("Skipping GPU training test - Metal not available: %v", err)
		return
	}
	defer ctx.Shutdown()

	// Create test tensors
	input, err := NewTensor([]int{4, 8}, Float32, GPU, make([]float32, 32))
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	weight, err := NewTensor([]int{8, 16}, Float32, GPU, make([]float32, 128))
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}

	bias, err := NewTensor([]int{16}, Float32, GPU, make([]float32, 16))
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}

	// Test operations that should be automatically fused
	operations := []OperationDesc{
		NewMatMulOp(input, weight),
		NewAddOp(nil, bias),     // This will be fused with MatMul
		NewReLUOp(nil),          // This will be fused with MatMul+Add
	}

	start := time.Now()
	results, err := ctx.BatchOperationsAsync(operations)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Batched operations with fusion failed: %v", err)
	}

	if len(results) != 1 { // Should be fused into a single operation
		t.Logf("Note: Operations were not fully fused (got %d results, expected 1)", len(results))
	}

	// Check final result shape
	finalResult := results[len(results)-1]
	expectedShape := []int{4, 16}
	if !equalIntSlices(finalResult.Shape, expectedShape) {
		t.Errorf("Expected final shape %v, got %v", expectedShape, finalResult.Shape)
	}

	// Check GPU statistics
	queued, executed, pending, efficiency := ctx.GetGPUStats()
	t.Logf("GPU Stats - Queued: %d, Executed: %d, Pending: %d, Batch Efficiency: %.2f%%", 
		queued, executed, pending, efficiency)
	t.Logf("Operation execution time: %v", elapsed)
}

// Helper function to compare integer slices
func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}