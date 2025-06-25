package metal_bridge

import (
	"reflect"
	"testing"
)

func TestNewComputeEngine(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	if engine == nil {
		t.Fatal("NewComputeEngine returned nil engine")
	}

	if engine.device == nil {
		t.Error("Engine device is nil")
	}

	if engine.queue == nil {
		t.Error("Engine queue is nil")
	}

	if engine.library == nil {
		t.Error("Engine library is nil")
	}
}

func TestAddArraysFloat32(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	inputA := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	inputB := []float32{2.0, 3.0, 4.0, 5.0, 6.0}
	expected := []float32{3.0, 5.0, 7.0, 9.0, 11.0}

	result, err := engine.AddArraysFloat32(inputA, inputB)
	if err != nil {
		t.Fatalf("AddArraysFloat32 failed: %v", err)
	}

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("AddArraysFloat32 result = %v, expected %v", result, expected)
	}
}

func TestAddArraysInt32(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	inputA := []int32{1, 2, 3, 4, 5}
	inputB := []int32{2, 3, 4, 5, 6}
	expected := []int32{3, 5, 7, 9, 11}

	result, err := engine.AddArraysInt32(inputA, inputB)
	if err != nil {
		t.Fatalf("AddArraysInt32 failed: %v", err)
	}

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("AddArraysInt32 result = %v, expected %v", result, expected)
	}
}

func TestReLUFloat32(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	expected := []float32{0.0, 0.0, 0.0, 1.0, 2.0}

	result, err := engine.ReLUFloat32(input)
	if err != nil {
		t.Fatalf("ReLUFloat32 failed: %v", err)
	}

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ReLUFloat32 result = %v, expected %v", result, expected)
	}
}

func TestMatMulFloat32(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	// Test 2x3 * 3x2 = 2x2 matrix multiplication
	matrixA := []float32{1, 2, 3, 4, 5, 6}  // 2x3
	matrixB := []float32{7, 8, 9, 10, 11, 12} // 3x2
	expected := []float32{58, 64, 139, 154}   // 2x2

	result, err := engine.MatMulFloat32(matrixA, matrixB, 2, 3, 2)
	if err != nil {
		t.Fatalf("MatMulFloat32 failed: %v", err)
	}

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("MatMulFloat32 result = %v, expected %v", result, expected)
	}
}

func TestAddArraysFloat32_DifferentSizes(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	inputA := []float32{1.0, 2.0, 3.0}
	inputB := []float32{2.0, 3.0} // Different size

	_, err = engine.AddArraysFloat32(inputA, inputB)
	if err == nil {
		t.Error("Expected error for arrays of different sizes")
	}
}

func TestLargeArrayAddition(t *testing.T) {
	engine, err := NewComputeEngine()
	if err != nil {
		t.Skipf("Metal not available on this system: %v", err)
	}

	size := 10000
	inputA := make([]float32, size)
	inputB := make([]float32, size)
	expected := make([]float32, size)

	for i := 0; i < size; i++ {
		inputA[i] = float32(i)
		inputB[i] = float32(i * 2)
		expected[i] = float32(i * 3)
	}

	result, err := engine.AddArraysFloat32(inputA, inputB)
	if err != nil {
		t.Fatalf("Large array addition failed: %v", err)
	}

	if len(result) != size {
		t.Errorf("Result length = %d, expected %d", len(result), size)
	}

	// Check a few values
	for i := 0; i < 10; i++ {
		if result[i] != expected[i] {
			t.Errorf("Result[%d] = %f, expected %f", i, result[i], expected[i])
		}
	}
}