package tensor

import (
	"fmt"
)

// broadcastScalarToShape expands a scalar tensor to the target shape
func broadcastScalarToShape(scalar *Tensor, targetShape []int) (*Tensor, error) {
	if len(scalar.Shape) != 0 {
		return nil, fmt.Errorf("broadcastScalarToShape: input must be scalar, got shape %v", scalar.Shape)
	}
	
	// Calculate number of elements in target shape
	numElems := 1
	for _, dim := range targetShape {
		numElems *= dim
	}
	
	// Create data by replicating scalar value
	var newData interface{}
	switch scalar.DType {
	case Float32:
		scalarData := scalar.Data.([]float32)
		if len(scalarData) != 1 {
			return nil, fmt.Errorf("scalar tensor must have exactly 1 element")
		}
		value := scalarData[0]
		data := make([]float32, numElems)
		for i := range data {
			data[i] = value
		}
		newData = data
	case Int32:
		scalarData := scalar.Data.([]int32)
		if len(scalarData) != 1 {
			return nil, fmt.Errorf("scalar tensor must have exactly 1 element")
		}
		value := scalarData[0]
		data := make([]int32, numElems)
		for i := range data {
			data[i] = value
		}
		newData = data
	default:
		return nil, fmt.Errorf("unsupported data type for scalar broadcasting: %v", scalar.DType)
	}
	
	return NewTensor(targetShape, scalar.DType, scalar.Device, newData)
}

// BroadcastShapes determines if two shapes are broadcastable and returns the resulting shape
// Follows NumPy/PyTorch broadcasting rules:
// 1. Start from trailing dimensions and work backwards
// 2. Dimensions are compatible if they are equal, or one of them is 1, or one is missing
// 3. Result shape is the maximum of each dimension
func BroadcastShapes(shape1, shape2 []int) ([]int, error) {
	// Handle empty shapes
	if len(shape1) == 0 && len(shape2) == 0 {
		return []int{}, nil
	}
	if len(shape1) == 0 {
		return shape2, nil
	}
	if len(shape2) == 0 {
		return shape1, nil
	}
	
	// Determine the maximum number of dimensions
	maxDims := len(shape1)
	if len(shape2) > maxDims {
		maxDims = len(shape2)
	}
	
	// Create result shape with max dimensions
	resultShape := make([]int, maxDims)
	
	// Work backwards through dimensions
	for i := 0; i < maxDims; i++ {
		dim1Idx := len(shape1) - 1 - i
		dim2Idx := len(shape2) - 1 - i
		resultIdx := maxDims - 1 - i
		
		dim1 := 1 // Default for missing dimensions
		dim2 := 1
		
		if dim1Idx >= 0 {
			dim1 = shape1[dim1Idx]
		}
		if dim2Idx >= 0 {
			dim2 = shape2[dim2Idx]
		}
		
		// Check compatibility
		if dim1 == dim2 {
			resultShape[resultIdx] = dim1
		} else if dim1 == 1 {
			resultShape[resultIdx] = dim2
		} else if dim2 == 1 {
			resultShape[resultIdx] = dim1
		} else {
			return nil, fmt.Errorf("shapes %v and %v are not broadcastable: dimension %d (%d vs %d)", 
				shape1, shape2, i, dim1, dim2)
		}
	}
	
	return resultShape, nil
}

// AreBroadcastable checks if two shapes can be broadcast together
func AreBroadcastable(shape1, shape2 []int) bool {
	_, err := BroadcastShapes(shape1, shape2)
	return err == nil
}

// BroadcastTensor expands a tensor to a target shape using broadcasting rules
func BroadcastTensor(t *Tensor, targetShape []int) (*Tensor, error) {
	// Handle scalar tensors (empty shape) - they can broadcast to any shape
	if len(t.Shape) == 0 {
		// Scalar to any shape - replicate the scalar value
		return broadcastScalarToShape(t, targetShape)
	}
	if len(targetShape) == 0 {
		// Any shape to scalar - this should sum to a scalar but that's a reduction, not broadcasting
		// For broadcasting, we just return the original tensor
		return t.Clone()
	}
	
	// Check if broadcasting is needed
	if shapesEqual(t.Shape, targetShape) {
		return t.Clone()
	}
	
	// Verify shapes are broadcastable
	_, err := BroadcastShapes(t.Shape, targetShape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast tensor with shape %v to %v: %v", 
			t.Shape, targetShape, err)
	}
	
	// Create result tensor with target shape
	numElems := calculateNumElements(targetShape)
	result := &Tensor{
		Shape:    make([]int, len(targetShape)),
		Strides:  calculateStrides(targetShape),
		DType:    t.DType,
		Device:   t.Device,
		NumElems: numElems,
		requiresGrad: t.requiresGrad,
	}
	copy(result.Shape, targetShape)
	
	// Allocate data for result
	switch t.DType {
	case Float32:
		result.Data = make([]float32, numElems)
	case Int32:
		result.Data = make([]int32, numElems)
	default:
		return nil, fmt.Errorf("unsupported data type for broadcasting: %v", t.DType)
	}
	
	// Perform broadcasting expansion
	err = broadcastData(t, result, targetShape)
	if err != nil {
		return nil, fmt.Errorf("failed to broadcast data: %v", err)
	}
	
	return result, nil
}

// broadcastData copies data from source tensor to result tensor with broadcasting
func broadcastData(src, dst *Tensor, targetShape []int) error {
	switch src.DType {
	case Float32:
		return broadcastFloat32Data(src, dst, targetShape)
	case Int32:
		return broadcastInt32Data(src, dst, targetShape)
	default:
		return fmt.Errorf("unsupported data type for broadcasting: %v", src.DType)
	}
}

// broadcastFloat32Data performs the actual data broadcasting for Float32 tensors
func broadcastFloat32Data(src, dst *Tensor, targetShape []int) error {
	srcData := src.Data.([]float32)
	dstData := dst.Data.([]float32)
	
	// Create coordinate mappers
	numDims := len(targetShape)
	srcDims := len(src.Shape)
	
	// Calculate total number of elements in target
	totalElems := calculateNumElements(targetShape)
	
	// For each position in the target tensor
	for dstIdx := 0; dstIdx < totalElems; dstIdx++ {
		// Convert flat index to multi-dimensional coordinates
		coords := make([]int, numDims)
		remaining := dstIdx
		for i := numDims - 1; i >= 0; i-- {
			coords[i] = remaining % targetShape[i]
			remaining /= targetShape[i]
		}
		
		// Map coordinates to source tensor
		srcIdx := 0
		srcStride := 1
		
		for i := numDims - 1; i >= 0; i-- {
			srcDimIdx := i - (numDims - srcDims) // Map to source dimension
			
			if srcDimIdx >= 0 && srcDimIdx < srcDims {
				srcDim := src.Shape[srcDimIdx]
				coord := coords[i]
				
				// If source dimension is 1, use coordinate 0 (broadcasting)
				if srcDim == 1 {
					coord = 0
				}
				
				srcIdx += coord * srcStride
				srcStride *= srcDim
			}
			// If dimension doesn't exist in source, implicitly broadcast from 0
		}
		
		// Copy data
		if srcIdx < len(srcData) {
			dstData[dstIdx] = srcData[srcIdx]
		} else {
			return fmt.Errorf("broadcasting index out of bounds: srcIdx=%d, srcLen=%d", srcIdx, len(srcData))
		}
	}
	
	return nil
}

// broadcastInt32Data performs the actual data broadcasting for Int32 tensors
func broadcastInt32Data(src, dst *Tensor, targetShape []int) error {
	srcData := src.Data.([]int32)
	dstData := dst.Data.([]int32)
	
	// Create coordinate mappers
	numDims := len(targetShape)
	srcDims := len(src.Shape)
	
	// Calculate total number of elements in target
	totalElems := calculateNumElements(targetShape)
	
	// For each position in the target tensor
	for dstIdx := 0; dstIdx < totalElems; dstIdx++ {
		// Convert flat index to multi-dimensional coordinates
		coords := make([]int, numDims)
		remaining := dstIdx
		for i := numDims - 1; i >= 0; i-- {
			coords[i] = remaining % targetShape[i]
			remaining /= targetShape[i]
		}
		
		// Map coordinates to source tensor
		srcIdx := 0
		srcStride := 1
		
		for i := numDims - 1; i >= 0; i-- {
			srcDimIdx := i - (numDims - srcDims) // Map to source dimension
			
			if srcDimIdx >= 0 && srcDimIdx < srcDims {
				srcDim := src.Shape[srcDimIdx]
				coord := coords[i]
				
				// If source dimension is 1, use coordinate 0 (broadcasting)
				if srcDim == 1 {
					coord = 0
				}
				
				srcIdx += coord * srcStride
				srcStride *= srcDim
			}
			// If dimension doesn't exist in source, implicitly broadcast from 0
		}
		
		// Copy data
		if srcIdx < len(srcData) {
			dstData[dstIdx] = srcData[srcIdx]
		} else {
			return fmt.Errorf("broadcasting index out of bounds: srcIdx=%d, srcLen=%d", srcIdx, len(srcData))
		}
	}
	
	return nil
}

// shapesEqual checks if two shapes are identical
func shapesEqual(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

// BroadcastTensorsForOperation broadcasts two tensors to a common shape for element-wise operations
func BroadcastTensorsForOperation(a, b *Tensor) (*Tensor, *Tensor, error) {
	// Determine broadcast shape
	broadcastShape, err := BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, nil, fmt.Errorf("tensors cannot be broadcast together: %v", err)
	}
	
	// Broadcast both tensors to the common shape
	aBroadcast, err := BroadcastTensor(a, broadcastShape)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to broadcast first tensor: %v", err)
	}
	
	bBroadcast, err := BroadcastTensor(b, broadcastShape)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to broadcast second tensor: %v", err)
	}
	
	return aBroadcast, bBroadcast, nil
}