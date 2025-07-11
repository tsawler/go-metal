package training

import (
	"testing"
	"unsafe"
)

// TestLabelDataType tests the LabelDataType string representation
func TestLabelDataType(t *testing.T) {
	tests := []struct {
		labelType LabelDataType
		expected  string
	}{
		{LabelTypeInt32, "Classification"},
		{LabelTypeFloat32, "Regression"},
		{LabelDataType(999), "Unknown(999)"},
	}

	for _, test := range tests {
		result := test.labelType.String()
		if result != test.expected {
			t.Errorf("LabelDataType(%d).String() = %s, expected %s", test.labelType, result, test.expected)
		}
	}
}

// TestNewInt32Labels tests creation and validation of Int32Labels
func TestNewInt32Labels(t *testing.T) {
	t.Run("ValidCreation", func(t *testing.T) {
		data := []int32{0, 1, 2, 1, 0, 2}
		shape := []int{6}
		
		labels, err := NewInt32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if labels.Size() != 6 {
			t.Errorf("Expected size 6, got %d", labels.Size())
		}
		
		if labels.DataType() != LabelTypeInt32 {
			t.Errorf("Expected LabelTypeInt32, got %v", labels.DataType())
		}
		
		returnedShape := labels.Shape()
		if len(returnedShape) != 1 || returnedShape[0] != 6 {
			t.Errorf("Expected shape [6], got %v", returnedShape)
		}
	})

	t.Run("BatchedLabels", func(t *testing.T) {
		data := []int32{0, 1, 1, 0, 2, 1}
		shape := []int{2, 3} // 2 batches of 3 labels each
		
		labels, err := NewInt32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if labels.Size() != 6 {
			t.Errorf("Expected size 6, got %d", labels.Size())
		}
		
		returnedShape := labels.Shape()
		if len(returnedShape) != 2 || returnedShape[0] != 2 || returnedShape[1] != 3 {
			t.Errorf("Expected shape [2, 3], got %v", returnedShape)
		}
	})

	t.Run("EmptyShape", func(t *testing.T) {
		data := []int32{0, 1, 2}
		shape := []int{}
		
		_, err := NewInt32Labels(data, shape)
		if err == nil {
			t.Error("Expected error for empty shape")
		}
	})

	t.Run("InvalidShapeDimension", func(t *testing.T) {
		data := []int32{0, 1, 2}
		shape := []int{3, 0, 2}
		
		_, err := NewInt32Labels(data, shape)
		if err == nil {
			t.Error("Expected error for invalid shape dimension")
		}
	})

	t.Run("SizeMismatch", func(t *testing.T) {
		data := []int32{0, 1, 2}
		shape := []int{2, 3} // Expected size 6, actual size 3
		
		_, err := NewInt32Labels(data, shape)
		if err == nil {
			t.Error("Expected error for size mismatch")
		}
	})
}

// TestInt32LabelsToFloat32Slice tests the conversion and caching
func TestInt32LabelsToFloat32Slice(t *testing.T) {
	data := []int32{0, 1, 2, 1, 0}
	shape := []int{5}
	
	labels, err := NewInt32Labels(data, shape)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// First call should create the cached version
	float32Data := labels.ToFloat32Slice()
	if len(float32Data) != 5 {
		t.Errorf("Expected length 5, got %d", len(float32Data))
	}

	expected := []float32{0, 1, 2, 1, 0}
	for i, v := range float32Data {
		if v != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], v)
		}
	}

	// Second call should return the same cached slice
	float32Data2 := labels.ToFloat32Slice()
	if &float32Data[0] != &float32Data2[0] {
		t.Error("Expected cached slice to be returned")
	}
}

// TestInt32LabelsUnsafePointer tests the unsafe pointer functionality
func TestInt32LabelsUnsafePointer(t *testing.T) {
	t.Run("ValidData", func(t *testing.T) {
		data := []int32{0, 1, 2}
		shape := []int{3}
		
		labels, err := NewInt32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		ptr := labels.UnsafePointer()
		if ptr == nil {
			t.Error("Expected non-nil pointer")
		}
		
		// Verify the pointer points to the first element
		if ptr != unsafe.Pointer(&data[0]) {
			t.Error("Pointer should point to first element of data")
		}
	})

	t.Run("EmptyData", func(t *testing.T) {
		// This test would require modifying the struct directly since
		// NewInt32Labels prevents empty data through validation
		labels := &Int32Labels{data: []int32{}, shape: []int{0}}
		
		ptr := labels.UnsafePointer()
		if ptr != nil {
			t.Error("Expected nil pointer for empty data")
		}
	})
}

// TestNewFloat32Labels tests creation and validation of Float32Labels
func TestNewFloat32Labels(t *testing.T) {
	t.Run("ValidCreation", func(t *testing.T) {
		data := []float32{0.1, 0.8, 0.3, 0.9, 0.2}
		shape := []int{5}
		
		labels, err := NewFloat32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if labels.Size() != 5 {
			t.Errorf("Expected size 5, got %d", labels.Size())
		}
		
		if labels.DataType() != LabelTypeFloat32 {
			t.Errorf("Expected LabelTypeFloat32, got %v", labels.DataType())
		}
		
		returnedShape := labels.Shape()
		if len(returnedShape) != 1 || returnedShape[0] != 5 {
			t.Errorf("Expected shape [5], got %v", returnedShape)
		}
	})

	t.Run("MultiDimensional", func(t *testing.T) {
		data := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
		shape := []int{2, 3} // 2 samples, 3 features each
		
		labels, err := NewFloat32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if labels.Size() != 6 {
			t.Errorf("Expected size 6, got %d", labels.Size())
		}
		
		returnedShape := labels.Shape()
		if len(returnedShape) != 2 || returnedShape[0] != 2 || returnedShape[1] != 3 {
			t.Errorf("Expected shape [2, 3], got %v", returnedShape)
		}
	})

	t.Run("EmptyShape", func(t *testing.T) {
		data := []float32{0.1, 0.2, 0.3}
		shape := []int{}
		
		_, err := NewFloat32Labels(data, shape)
		if err == nil {
			t.Error("Expected error for empty shape")
		}
	})

	t.Run("NegativeShapeDimension", func(t *testing.T) {
		data := []float32{0.1, 0.2, 0.3}
		shape := []int{3, -1}
		
		_, err := NewFloat32Labels(data, shape)
		if err == nil {
			t.Error("Expected error for negative shape dimension")
		}
	})

	t.Run("SizeMismatch", func(t *testing.T) {
		data := []float32{0.1, 0.2, 0.3}
		shape := []int{2, 2} // Expected size 4, actual size 3
		
		_, err := NewFloat32Labels(data, shape)
		if err == nil {
			t.Error("Expected error for size mismatch")
		}
	})
}

// TestFloat32LabelsToFloat32Slice tests zero-cost slice return
func TestFloat32LabelsToFloat32Slice(t *testing.T) {
	data := []float32{0.1, 0.8, 0.3, 0.9, 0.2}
	shape := []int{5}
	
	labels, err := NewFloat32Labels(data, shape)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Should return the same slice (zero-cost)
	returnedData := labels.ToFloat32Slice()
	
	if len(returnedData) != 5 {
		t.Errorf("Expected length 5, got %d", len(returnedData))
	}

	// Should be the exact same slice
	if &returnedData[0] != &data[0] {
		t.Error("Expected zero-cost return of original slice")
	}

	for i, v := range returnedData {
		if v != data[i] {
			t.Errorf("At index %d: expected %f, got %f", i, data[i], v)
		}
	}
}

// TestFloat32LabelsUnsafePointer tests the unsafe pointer functionality
func TestFloat32LabelsUnsafePointer(t *testing.T) {
	t.Run("ValidData", func(t *testing.T) {
		data := []float32{0.1, 0.2, 0.3}
		shape := []int{3}
		
		labels, err := NewFloat32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		ptr := labels.UnsafePointer()
		if ptr == nil {
			t.Error("Expected non-nil pointer")
		}
		
		// Verify the pointer points to the first element
		if ptr != unsafe.Pointer(&data[0]) {
			t.Error("Pointer should point to first element of data")
		}
	})

	t.Run("EmptyData", func(t *testing.T) {
		labels := &Float32Labels{data: []float32{}, shape: []int{0}}
		
		ptr := labels.UnsafePointer()
		if ptr != nil {
			t.Error("Expected nil pointer for empty data")
		}
	})
}

// TestShapeImmutability tests that returned shapes cannot modify internal state
func TestShapeImmutability(t *testing.T) {
	t.Run("Int32Labels", func(t *testing.T) {
		data := []int32{0, 1, 2}
		shape := []int{3}
		
		labels, err := NewInt32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		returnedShape := labels.Shape()
		returnedShape[0] = 999 // Try to modify
		
		// Original shape should be unchanged
		originalShape := labels.Shape()
		if originalShape[0] != 3 {
			t.Error("Shape should be immutable")
		}
	})

	t.Run("Float32Labels", func(t *testing.T) {
		data := []float32{0.1, 0.2, 0.3}
		shape := []int{3}
		
		labels, err := NewFloat32Labels(data, shape)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		returnedShape := labels.Shape()
		returnedShape[0] = 999 // Try to modify
		
		// Original shape should be unchanged
		originalShape := labels.Shape()
		if originalShape[0] != 3 {
			t.Error("Shape should be immutable")
		}
	})
}

// TestNewBatchedLabels tests batched label creation and validation
func TestNewBatchedLabels(t *testing.T) {
	t.Run("ValidCreation", func(t *testing.T) {
		// Create multiple label batches of the same type
		batch1, _ := NewInt32Labels([]int32{0, 1}, []int{2})
		batch2, _ := NewInt32Labels([]int32{1, 0}, []int{2})
		batch3, _ := NewInt32Labels([]int32{1, 1}, []int{2})
		
		batches := []LabelData{batch1, batch2, batch3}
		
		batchedLabels, err := NewBatchedLabels(batches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if batchedLabels.NumBatches() != 3 {
			t.Errorf("Expected 3 batches, got %d", batchedLabels.NumBatches())
		}
	})

	t.Run("EmptyBatches", func(t *testing.T) {
		batches := []LabelData{}
		
		_, err := NewBatchedLabels(batches)
		if err == nil {
			t.Error("Expected error for empty batches")
		}
	})

	t.Run("MixedTypes", func(t *testing.T) {
		batch1, _ := NewInt32Labels([]int32{0, 1}, []int{2})
		batch2, _ := NewFloat32Labels([]float32{0.1, 0.9}, []int{2})
		
		batches := []LabelData{batch1, batch2}
		
		_, err := NewBatchedLabels(batches)
		if err == nil {
			t.Error("Expected error for mixed label types")
		}
	})
}

// TestBatchedLabelsGetBatch tests batch retrieval
func TestBatchedLabelsGetBatch(t *testing.T) {
	batch1, _ := NewInt32Labels([]int32{0, 1}, []int{2})
	batch2, _ := NewInt32Labels([]int32{1, 0}, []int{2})
	batch3, _ := NewInt32Labels([]int32{1, 1}, []int{2})
	
	batches := []LabelData{batch1, batch2, batch3}
	batchedLabels, _ := NewBatchedLabels(batches)

	t.Run("ValidIndices", func(t *testing.T) {
		for i := 0; i < 3; i++ {
			batch, err := batchedLabels.GetBatch(i)
			if err != nil {
				t.Errorf("Unexpected error for index %d: %v", i, err)
			}
			if batch == nil {
				t.Errorf("Expected non-nil batch at index %d", i)
			}
		}
	})

	t.Run("NegativeIndex", func(t *testing.T) {
		_, err := batchedLabels.GetBatch(-1)
		if err == nil {
			t.Error("Expected error for negative index")
		}
	})

	t.Run("IndexOutOfRange", func(t *testing.T) {
		_, err := batchedLabels.GetBatch(3)
		if err == nil {
			t.Error("Expected error for index out of range")
		}
	})
}

// TestLabelDataInterface tests that all implementations satisfy the interface
func TestLabelDataInterface(t *testing.T) {
	var _ LabelData = &Int32Labels{}
	var _ LabelData = &Float32Labels{}
	
	// Test interface behavior
	int32Labels, _ := NewInt32Labels([]int32{0, 1, 2}, []int{3})
	float32Labels, _ := NewFloat32Labels([]float32{0.1, 0.2, 0.3}, []int{3})
	
	labels := []LabelData{int32Labels, float32Labels}
	
	for i, label := range labels {
		// Test all interface methods work
		if label.Size() != 3 {
			t.Errorf("Label %d: expected size 3, got %d", i, label.Size())
		}
		
		shape := label.Shape()
		if len(shape) != 1 || shape[0] != 3 {
			t.Errorf("Label %d: expected shape [3], got %v", i, shape)
		}
		
		float32Data := label.ToFloat32Slice()
		if len(float32Data) != 3 {
			t.Errorf("Label %d: expected float32 slice length 3, got %d", i, len(float32Data))
		}
		
		ptr := label.UnsafePointer()
		if ptr == nil {
			t.Errorf("Label %d: expected non-nil unsafe pointer", i)
		}
		
		dataType := label.DataType()
		if i == 0 && dataType != LabelTypeInt32 {
			t.Errorf("Label %d: expected LabelTypeInt32, got %v", i, dataType)
		}
		if i == 1 && dataType != LabelTypeFloat32 {
			t.Errorf("Label %d: expected LabelTypeFloat32, got %v", i, dataType)
		}
	}
}

// TestLabelCaching tests that Int32Labels properly caches float32 conversion
func TestLabelCaching(t *testing.T) {
	data := []int32{0, 1, 2, 3, 4}
	shape := []int{5}
	
	labels, err := NewInt32Labels(data, shape)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// First call - should create cache
	first := labels.ToFloat32Slice()
	
	// Second call - should return cached version
	second := labels.ToFloat32Slice()
	
	// Should be the same slice reference
	if &first[0] != &second[0] {
		t.Error("Expected cached slice to be returned on second call")
	}
	
	// Values should be correct
	for i, expected := range []float32{0, 1, 2, 3, 4} {
		if first[i] != expected {
			t.Errorf("At index %d: expected %f, got %f", i, expected, first[i])
		}
	}
}

// BenchmarkLabelCreation benchmarks label creation performance
func BenchmarkLabelCreation(b *testing.B) {
	data32 := make([]int32, 1000)
	dataFloat32 := make([]float32, 1000)
	shape := []int{1000}
	
	for i := range data32 {
		data32[i] = int32(i % 10)
		dataFloat32[i] = float32(i) * 0.1
	}

	b.Run("Int32Labels", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := NewInt32Labels(data32, shape)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Float32Labels", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := NewFloat32Labels(dataFloat32, shape)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkLabelConversion benchmarks conversion performance
func BenchmarkLabelConversion(b *testing.B) {
	data := make([]int32, 1000)
	for i := range data {
		data[i] = int32(i % 10)
	}
	
	labels, _ := NewInt32Labels(data, []int{1000})

	b.Run("Int32ToFloat32Conversion", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = labels.ToFloat32Slice()
		}
	})
}