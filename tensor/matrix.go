package tensor

import (
	"fmt"
	"sync/atomic"
)

func getIndex(indices []int, strides []int) int {
	index := 0
	for i, idx := range indices {
		index += idx * strides[i]
	}
	return index
}

func getIndicesFromLinear(linearIndex int, shape []int) []int {
	indices := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		indices[i] = linearIndex % shape[i]
		linearIndex /= shape[i]
	}
	return indices
}

func MatMul(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	if len(t1.Shape) < 2 || len(t2.Shape) < 2 {
		return nil, fmt.Errorf("matmul requires tensors with at least 2 dimensions")
	}

	// Device-aware routing: Use GPU operations when tensors are on GPU
	if t1.Device == GPU || t2.Device == GPU {
		// Prefer MPS operations when available
		result, err := MatMulMPS(t1, t2)
		if err == nil {
			return result, nil
		}
		// If MPS fails, log the error and try GPU fallback
		fmt.Printf("MPS MatMul failed: %v\n", err)
		
		// Fallback to basic GPU operations
		result, err = MatMulGPU(t1, t2)
		if err == nil {
			return result, nil
		}
		// If GPU operations fail, log and continue with CPU implementation below
		fmt.Printf("GPU MatMul failed: %v\n", err)
	}

	shape1 := t1.Shape
	shape2 := t2.Shape
	
	rows1 := shape1[len(shape1)-2]
	cols1 := shape1[len(shape1)-1]
	rows2 := shape2[len(shape2)-2]
	cols2 := shape2[len(shape2)-1]

	if cols1 != rows2 {
		return nil, fmt.Errorf("incompatible dimensions for matmul: (%d, %d) x (%d, %d)", rows1, cols1, rows2, cols2)
	}

	outputShape := make([]int, len(shape1))
	copy(outputShape, shape1)
	outputShape[len(outputShape)-1] = cols2

	result, err := Zeros(outputShape, t1.DType, t1.Device)
	if err != nil {
		return nil, err
	}

	switch t1.DType {
	case Float32:
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]float32)
		data2 := t2.materializeView().([]float32)
		resultData := result.Data.([]float32)

		for i := 0; i < rows1; i++ {
			for j := 0; j < cols2; j++ {
				var sum float32
				for k := 0; k < cols1; k++ {
					idx1 := i*cols1 + k
					idx2 := k*cols2 + j
					sum += data1[idx1] * data2[idx2]
				}
				resultIdx := i*cols2 + j
				resultData[resultIdx] = sum
			}
		}
	case Int32:
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]int32)
		data2 := t2.materializeView().([]int32)
		resultData := result.Data.([]int32)

		for i := 0; i < rows1; i++ {
			for j := 0; j < cols2; j++ {
				var sum int32
				for k := 0; k < cols1; k++ {
					idx1 := i*cols1 + k
					idx2 := k*cols2 + j
					sum += data1[idx1] * data2[idx2]
				}
				resultIdx := i*cols2 + j
				resultData[resultIdx] = sum
			}
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for MatMul: %s", t1.DType)
	}

	return result, nil
}

func Transpose(t *Tensor, dim0, dim1 int) (*Tensor, error) {
	if dim0 < 0 || dim0 >= len(t.Shape) {
		return nil, fmt.Errorf("dim0 %d out of range for tensor with %d dimensions", dim0, len(t.Shape))
	}
	if dim1 < 0 || dim1 >= len(t.Shape) {
		return nil, fmt.Errorf("dim1 %d out of range for tensor with %d dimensions", dim1, len(t.Shape))
	}

	// Create transposed shape
	outputShape := make([]int, len(t.Shape))
	copy(outputShape, t.Shape)
	outputShape[dim0], outputShape[dim1] = outputShape[dim1], outputShape[dim0]

	// Create transposed strides - this is the key to view-based transpose
	outputStrides := make([]int, len(t.Strides))
	copy(outputStrides, t.Strides)
	outputStrides[dim0], outputStrides[dim1] = outputStrides[dim1], outputStrides[dim0]

	// Get the base data and base tensor for proper view management
	var baseData interface{}
	var baseTensor *Tensor
	var offset int
	
	if t.isView {
		// If the original tensor is already a view, chain to the base
		baseData = t.baseData
		baseTensor = t.baseTensor
		offset = t.offset
	} else {
		// If original tensor is not a view, it becomes the base
		baseData = t.Data
		baseTensor = t
		offset = 0
	}

	// Create result tensor as a TRUE VIEW - no data copying!
	result := &Tensor{
		Data:         nil, // Views don't have their own Data field
		Shape:        outputShape,
		Strides:      outputStrides,
		DType:        t.DType,
		Device:       t.Device,
		NumElems:     t.NumElems,
		requiresGrad: t.requiresGrad,
		
		// View-specific fields
		isView:       true,
		baseData:     baseData,
		baseTensor:   baseTensor,
		offset:       offset,
	}

	// For GPU tensors, share the Metal buffer with proper reference counting
	if t.Device == GPU || t.Device == PersistentGPU {
		if t.gpuBuffer != nil {
			result.gpuBuffer = t.gpuBuffer
			// Initialize reference count for the new tensor
			atomic.StoreInt32(&result.refCount, 1)
			// Retain the underlying Metal buffer to prevent premature release
			if buffer, ok := t.gpuBuffer.(interface{ Retain() }); ok {
				buffer.Retain()
			}
		}
	}

	// Keep a reference to the base tensor to prevent it from being garbage collected
	// This is critical for view lifecycle management
	if baseTensor != nil {
		// Increment reference count of base tensor if it's a GPU tensor
		if baseTensor.Device == GPU || baseTensor.Device == PersistentGPU {
			atomic.AddInt32(&baseTensor.refCount, 1)
		}
	}

	return result, nil
}

func Reshape(t *Tensor, newShape []int) (*Tensor, error) {
	if err := validateShape(newShape); err != nil {
		return nil, err
	}

	newNumElems := calculateNumElements(newShape)
	if newNumElems != t.NumElems {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v (size %d)", 
			t.NumElems, newShape, newNumElems)
	}

	newStrides := calculateStrides(newShape)

	result := &Tensor{
		Shape:        newShape,
		Strides:      newStrides,
		DType:        t.DType,
		Device:       t.Device,
		NumElems:     t.NumElems,
		requiresGrad: t.requiresGrad,
	}

	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		newData := make([]float32, len(data))
		copy(newData, data)
		result.Data = newData
	case Int32:
		data := t.Data.([]int32)
		newData := make([]int32, len(data))
		copy(newData, data)
		result.Data = newData
	default:
		return nil, fmt.Errorf("unsupported dtype for Reshape: %s", t.DType)
	}

	return result, nil
}

func Flatten(t *Tensor) (*Tensor, error) {
	return Reshape(t, []int{t.NumElems})
}

func Squeeze(t *Tensor, dim int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.Shape) {
		return nil, fmt.Errorf("dim %d out of range for tensor with %d dimensions", dim, len(t.Shape))
	}
	
	if t.Shape[dim] != 1 {
		return nil, fmt.Errorf("cannot squeeze dimension %d with size %d (must be 1)", dim, t.Shape[dim])
	}

	newShape := make([]int, 0, len(t.Shape)-1)
	for i, size := range t.Shape {
		if i != dim {
			newShape = append(newShape, size)
		}
	}

	return Reshape(t, newShape)
}

func Unsqueeze(t *Tensor, dim int) (*Tensor, error) {
	if dim < 0 || dim > len(t.Shape) {
		return nil, fmt.Errorf("dim %d out of range for unsqueeze operation", dim)
	}

	newShape := make([]int, len(t.Shape)+1)
	copy(newShape[:dim], t.Shape[:dim])
	newShape[dim] = 1
	copy(newShape[dim+1:], t.Shape[dim:])

	return Reshape(t, newShape)
}

func Sum(t *Tensor, dim int, keepDim bool) (*Tensor, error) {
	if dim < 0 || dim >= len(t.Shape) {
		return nil, fmt.Errorf("dim %d out of range for tensor with %d dimensions", dim, len(t.Shape))
	}

	var outputShape []int
	if keepDim {
		outputShape = make([]int, len(t.Shape))
		copy(outputShape, t.Shape)
		outputShape[dim] = 1
	} else {
		outputShape = make([]int, 0, len(t.Shape)-1)
		for i, size := range t.Shape {
			if i != dim {
				outputShape = append(outputShape, size)
			}
		}
	}

	result, err := Zeros(outputShape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}

	
	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		resultData := result.Data.([]float32)

		for i := 0; i < t.NumElems; i++ {
			indices := getIndicesFromLinear(i, t.Shape)
			
			var resultIndices []int
			if keepDim {
				resultIndices = make([]int, len(indices))
				copy(resultIndices, indices)
				resultIndices[dim] = 0
			} else {
				resultIndices = make([]int, 0, len(indices)-1)
				for j, idx := range indices {
					if j != dim {
						resultIndices = append(resultIndices, idx)
					}
				}
			}
			
			resultIdx := getIndex(resultIndices, result.Strides)
			resultData[resultIdx] += data[i]
		}
	case Int32:
		data := t.Data.([]int32)
		resultData := result.Data.([]int32)

		for i := 0; i < t.NumElems; i++ {
			indices := getIndicesFromLinear(i, t.Shape)
			
			var resultIndices []int
			if keepDim {
				resultIndices = make([]int, len(indices))
				copy(resultIndices, indices)
				resultIndices[dim] = 0
			} else {
				resultIndices = make([]int, 0, len(indices)-1)
				for j, idx := range indices {
					if j != dim {
						resultIndices = append(resultIndices, idx)
					}
				}
			}
			
			resultIdx := getIndex(resultIndices, result.Strides)
			resultData[resultIdx] += data[i]
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for Sum: %s", t.DType)
	}

	return result, nil
}