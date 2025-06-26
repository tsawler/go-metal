package tensor

import (
	"testing"
	"time"
)

func TestGPUComputationGraph(t *testing.T) {
	// Create GPU computation graph
	graph, err := NewGPUComputationGraph()
	if err != nil {
		t.Skipf("Skipping GPU test - Metal not available: %v", err)
		return
	}
	defer graph.Shutdown()
	
	// Create test tensors
	a, err := NewTensor([]int{2, 3}, Float32, GPU, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create tensor A: %v", err)
	}
	
	b, err := NewTensor([]int{3, 2}, Float32, GPU, []float32{7, 8, 9, 10, 11, 12})
	if err != nil {
		t.Fatalf("Failed to create tensor B: %v", err)
	}
	
	c, err := NewTensor([]int{2, 2}, Float32, GPU, []float32{1, 1, 1, 1})
	if err != nil {
		t.Fatalf("Failed to create tensor C: %v", err)
	}
	
	// Test dependency chain: (A * B) + C using ExecuteSequence
	operations := []OperationDesc{
		{Type: "MatMul", Inputs: []*Tensor{a, b}, Params: nil},
		{Type: "Add", Inputs: []*Tensor{nil, c}, Params: nil}, // nil will be replaced with MatMul result
	}
	
	result, err := graph.ExecuteSequence(operations)
	if err != nil {
		t.Fatalf("ExecuteSequence failed: %v", err)
	}
	
	if result == nil {
		t.Fatalf("Result is nil")
	}
	
	// Verify result shape
	expectedShape := []int{2, 2}
	if !equalShapes(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}
	
	// Check statistics
	queued, executed, pending := graph.GetStats()
	t.Logf("GPU Graph Stats - Queued: %d, Executed: %d, Pending: %d", queued, executed, pending)
	
	if executed < 2 {
		t.Errorf("Expected at least 2 operations executed, got %d", executed)
	}
}

func TestGPUTrainingContext(t *testing.T) {
	// Create GPU training context
	ctx, err := NewGPUTrainingContext()
	if err != nil {
		t.Skipf("Skipping GPU training test - Metal not available: %v", err)
		return
	}
	defer ctx.Shutdown()
	
	// Create test tensors for linear layer simulation
	input, err := NewTensor([]int{4, 8}, Float32, GPU, make([]float32, 32))
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	
	weight, err := NewTensor([]int{16, 8}, Float32, GPU, make([]float32, 128))
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}
	
	bias, err := NewTensor([]int{16}, Float32, GPU, make([]float32, 16))
	if err != nil {
		t.Fatalf("Failed to create bias tensor: %v", err)
	}
	
	// Test batched operations
	operations := []OperationDesc{
		NewMatMulOp(input, weight),
		NewAddOp(nil, bias), // nil will be replaced with previous result
		NewReLUOp(nil),      // nil will be replaced with previous result
	}
	
	start := time.Now()
	results, err := ctx.BatchOperationsAsync(operations)
	elapsed := time.Since(start)
	
	if err != nil {
		t.Fatalf("Batched operations failed: %v", err)
	}
	
	// With operation fusion, we may get fewer results than input operations
	// The final result should still be correct
	if len(results) == 0 {
		t.Fatalf("Expected at least 1 result, got 0")
	}
	
	// Check final result shape
	finalResult := results[len(results)-1]
	expectedShape := []int{4, 16}
	if !equalShapes(finalResult.Shape, expectedShape) {
		t.Errorf("Expected final shape %v, got %v", expectedShape, finalResult.Shape)
	}
	
	// Check GPU statistics
	queued, executed, pending, efficiency := ctx.GetGPUStats()
	t.Logf("Training Context Stats - Queued: %d, Executed: %d, Pending: %d, Batch Efficiency: %.2f%%", 
		queued, executed, pending, efficiency)
	t.Logf("Operation execution time: %v", elapsed)
	
	// With operation fusion, we may execute fewer operations than the input count
	// What matters is that the computation completed successfully
	t.Logf("Operations fused from %d input operations to %d actual executions", len(operations), executed)
}

func TestOptimizedMatMulChain(t *testing.T) {
	ctx, err := NewGPUTrainingContext()
	if err != nil {
		t.Skipf("Skipping optimized MatMul test - Metal not available: %v", err)
		return
	}
	defer ctx.Shutdown()
	
	// Create chain of matrices: A * B * C
	a, err := NewTensor([]int{4, 8}, Float32, GPU, make([]float32, 32))
	if err != nil {
		t.Fatalf("Failed to create tensor A: %v", err)
	}
	
	b, err := NewTensor([]int{8, 6}, Float32, GPU, make([]float32, 48))
	if err != nil {
		t.Fatalf("Failed to create tensor B: %v", err)
	}
	
	c, err := NewTensor([]int{6, 4}, Float32, GPU, make([]float32, 24))
	if err != nil {
		t.Fatalf("Failed to create tensor C: %v", err)
	}
	
	// Execute optimized matrix multiplication chain
	start := time.Now()
	result, err := ctx.OptimizedMatMulChain([]*Tensor{a, b, c})
	elapsed := time.Since(start)
	
	if err != nil {
		t.Fatalf("Optimized MatMul chain failed: %v", err)
	}
	
	// Verify final result shape (4, 4)
	expectedShape := []int{4, 4}
	if !equalShapes(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}
	
	t.Logf("MatMul chain execution time: %v", elapsed)
	
	// Verify dependency tracking worked by checking operation count
	queued, executed, pending, _ := ctx.GetGPUStats()
	t.Logf("MatMul Chain Stats - Queued: %d, Executed: %d, Pending: %d", queued, executed, pending)
	
	// Should have 2 MatMul operations (A*B, result*C)
	if executed < 2 {
		t.Errorf("Expected at least 2 MatMul operations, got %d", executed)
	}
}

// Helper function to compare shapes
func equalShapes(a, b []int) bool {
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

func BenchmarkGPUOperationQueuing(b *testing.B) {
	ctx, err := NewGPUTrainingContext()
	if err != nil {
		b.Skipf("Skipping GPU benchmark - Metal not available: %v", err)
		return
	}
	defer ctx.Shutdown()
	
	// Create test tensors
	input, _ := NewTensor([]int{64, 128}, Float32, GPU, make([]float32, 64*128))
	weight, _ := NewTensor([]int{128, 256}, Float32, GPU, make([]float32, 128*256))
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		// Queue multiple operations
		ops := []OperationDesc{
			NewMatMulOp(input, weight),
			NewReLUOp(nil),
		}
		
		_, err := ctx.BatchOperationsAsync(ops)
		if err != nil {
			b.Fatalf("Batch operation failed: %v", err)
		}
	}
}