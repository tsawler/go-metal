package training

import (
	"fmt"
	"unsafe"
)

// LabelData provides a flexible interface for different label types
// This enables zero-cost abstractions for both classification and regression
// while maintaining GPU-residency and minimizing CGO calls
type LabelData interface {
	// ToFloat32Slice returns the underlying data as []float32 for GPU consumption
	// For Float32Labels: returns slice directly (zero-cost)
	// For Int32Labels: converts to float32 (one-time cost)
	// PERFORMANCE: This method is designed to minimize allocations
	ToFloat32Slice() []float32
	
	// DataType returns the semantic type of labels for loss function selection
	DataType() LabelDataType
	
	// Size returns the number of label elements
	Size() int
	
	// Shape returns the tensor shape of labels [batch_size, num_classes/dims]
	Shape() []int
	
	// UnsafePointer returns a pointer to the underlying data for CGO
	// This enables zero-copy transfer to GPU
	UnsafePointer() unsafe.Pointer
}

// LabelDataType represents the semantic type of labels
type LabelDataType int

const (
	LabelTypeInt32   LabelDataType = iota // Classification labels
	LabelTypeFloat32                      // Regression targets
)

// String returns human-readable label type name
func (ldt LabelDataType) String() string {
	switch ldt {
	case LabelTypeInt32:
		return "Classification"
	case LabelTypeFloat32:
		return "Regression"
	default:
		return fmt.Sprintf("Unknown(%d)", ldt)
	}
}

// Int32Labels wraps []int32 for classification tasks
// Implements LabelData interface with minimal overhead
type Int32Labels struct {
	data  []int32
	shape []int
	// Cache the float32 conversion to avoid repeated allocations
	cachedFloat32 []float32
}

// NewInt32Labels creates classification labels with shape validation
func NewInt32Labels(data []int32, shape []int) (*Int32Labels, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("shape cannot be empty")
	}
	
	expectedSize := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid shape dimension: %d", dim)
		}
		expectedSize *= dim
	}
	
	if len(data) != expectedSize {
		return nil, fmt.Errorf("data size %d doesn't match shape %v (expected %d)", 
			len(data), shape, expectedSize)
	}
	
	// Make a copy of shape to prevent external modifications
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	
	return &Int32Labels{
		data:  data,
		shape: shapeCopy,
	}, nil
}

// ToFloat32Slice converts int32 labels to float32
// Uses cached conversion to minimize allocations
func (l *Int32Labels) ToFloat32Slice() []float32 {
	if l.cachedFloat32 == nil {
		l.cachedFloat32 = make([]float32, len(l.data))
		for i, v := range l.data {
			l.cachedFloat32[i] = float32(v)
		}
	}
	return l.cachedFloat32
}

// DataType returns LabelTypeInt32 for classification
func (l *Int32Labels) DataType() LabelDataType {
	return LabelTypeInt32
}

// Size returns the total number of labels
func (l *Int32Labels) Size() int {
	return len(l.data)
}

// Shape returns a copy of the label tensor shape
func (l *Int32Labels) Shape() []int {
	shapeCopy := make([]int, len(l.shape))
	copy(shapeCopy, l.shape)
	return shapeCopy
}

// UnsafePointer returns pointer to the underlying int32 data
// This enables zero-copy transfer to GPU for classification
func (l *Int32Labels) UnsafePointer() unsafe.Pointer {
	if len(l.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&l.data[0])
}

// Float32Labels wraps []float32 for regression tasks
// Implements LabelData interface with zero overhead
type Float32Labels struct {
	data  []float32
	shape []int
}

// NewFloat32Labels creates regression labels with shape validation
func NewFloat32Labels(data []float32, shape []int) (*Float32Labels, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("shape cannot be empty")
	}
	
	expectedSize := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid shape dimension: %d", dim)
		}
		expectedSize *= dim
	}
	
	if len(data) != expectedSize {
		return nil, fmt.Errorf("data size %d doesn't match shape %v (expected %d)", 
			len(data), shape, expectedSize)
	}
	
	// Make a copy of shape to prevent external modifications
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	
	return &Float32Labels{
		data:  data,
		shape: shapeCopy,
	}, nil
}

// ToFloat32Slice returns the underlying float32 slice directly
// ZERO-COST: No allocation, no copying
func (l *Float32Labels) ToFloat32Slice() []float32 {
	return l.data
}

// DataType returns LabelTypeFloat32 for regression
func (l *Float32Labels) DataType() LabelDataType {
	return LabelTypeFloat32
}

// Size returns the total number of labels
func (l *Float32Labels) Size() int {
	return len(l.data)
}

// Shape returns a copy of the label tensor shape
func (l *Float32Labels) Shape() []int {
	shapeCopy := make([]int, len(l.shape))
	copy(shapeCopy, l.shape)
	return shapeCopy
}

// UnsafePointer returns pointer to the underlying float32 data
// This enables zero-copy transfer to GPU for regression
func (l *Float32Labels) UnsafePointer() unsafe.Pointer {
	if len(l.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&l.data[0])
}

// BatchedLabels represents a collection of label batches
// Useful for multi-GPU training or pipeline parallelism
type BatchedLabels struct {
	batches []LabelData
}

// NewBatchedLabels creates a collection of label batches
func NewBatchedLabels(batches []LabelData) (*BatchedLabels, error) {
	if len(batches) == 0 {
		return nil, fmt.Errorf("batches cannot be empty")
	}
	
	// Validate all batches have the same type
	firstType := batches[0].DataType()
	for i, batch := range batches {
		if batch.DataType() != firstType {
			return nil, fmt.Errorf("batch %d has type %v, expected %v", 
				i, batch.DataType(), firstType)
		}
	}
	
	return &BatchedLabels{
		batches: batches,
	}, nil
}

// GetBatch returns the label batch at the specified index
func (bl *BatchedLabels) GetBatch(index int) (LabelData, error) {
	if index < 0 || index >= len(bl.batches) {
		return nil, fmt.Errorf("batch index %d out of range [0, %d)", index, len(bl.batches))
	}
	return bl.batches[index], nil
}

// NumBatches returns the number of label batches
func (bl *BatchedLabels) NumBatches() int {
	return len(bl.batches)
}