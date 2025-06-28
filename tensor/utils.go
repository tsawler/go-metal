package tensor

import (
	"fmt"
	"math"
	"strings"
	"sync/atomic"
)

// Reshape returns a new tensor with the same data but different shape
// The new shape must have the same total number of elements
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	// Calculate total elements in new shape
	newNumElems := 1
	hasNegOne := false
	negOneIdx := -1
	
	for i, dim := range newShape {
		if dim < 0 {
			if dim != -1 {
				return nil, fmt.Errorf("negative dimension %d at index %d is not allowed (only -1 is allowed)", dim, i)
			}
			if hasNegOne {
				return nil, fmt.Errorf("only one dimension can be -1")
			}
			hasNegOne = true
			negOneIdx = i
		} else if dim == 0 {
			return nil, fmt.Errorf("dimension %d cannot be 0", i)
		} else {
			newNumElems *= dim
		}
	}
	
	// If there's a -1, calculate what it should be
	if hasNegOne {
		if t.NumElems%newNumElems != 0 {
			return nil, fmt.Errorf("cannot reshape tensor of size %d into shape with -1: size must be divisible by %d", t.NumElems, newNumElems)
		}
		inferredDim := t.NumElems / newNumElems
		newShape[negOneIdx] = inferredDim
		newNumElems *= inferredDim
	}
	
	// Check that total elements match
	if newNumElems != t.NumElems {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v (size %d)", t.NumElems, newShape, newNumElems)
	}
	
	// Create new tensor with same data but new shape
	reshaped := &Tensor{
		Shape:        make([]int, len(newShape)),
		Strides:      calculateStrides(newShape),
		DType:        t.DType,
		Device:       t.Device,
		Data:         t.Data, // Share the same underlying data
		NumElems:     t.NumElems,
		requiresGrad: t.requiresGrad,
		grad:         nil, // Don't copy gradient
		creator:      nil, // Don't copy autograd graph
	}
	
	copy(reshaped.Shape, newShape)
	
	// If on GPU, share the same buffer with proper reference counting
	if t.Device == GPU {
		reshaped.gpuBuffer = t.gpuBuffer
		// Initialize the new tensor's reference count to 1
		atomic.StoreInt32(&reshaped.refCount, 1)
		// Retain the underlying Metal buffer to prevent premature release
		if buffer, ok := t.gpuBuffer.(interface{ Retain() }); ok {
			buffer.Retain()
		}
	}
	
	return reshaped, nil
}

func (t *Tensor) Clone() (*Tensor, error) {
	clone := &Tensor{
		Shape:        make([]int, len(t.Shape)),
		Strides:      make([]int, len(t.Strides)),
		DType:        t.DType,
		Device:       t.Device,
		NumElems:     t.NumElems,
		requiresGrad: t.requiresGrad,
	}
	
	copy(clone.Shape, t.Shape)
	copy(clone.Strides, t.Strides)

	// Handle GPU and PersistentGPU tensors with nil Data differently
	if (t.Device == GPU || t.Device == PersistentGPU) && t.Data == nil {
		// For pure GPU tensors, convert to CPU first, clone, then convert back to preserve device
		cpuTensor, err := t.ToCPU()
		if err != nil {
			return nil, fmt.Errorf("failed to convert GPU tensor to CPU for cloning: %v", err)
		}
		
		// Clone the CPU tensor
		cpuClone, err := cpuTensor.Clone()
		if err != nil {
			return nil, fmt.Errorf("failed to clone CPU tensor: %v", err)
		}
		
		// Convert back to original device
		if t.Device == PersistentGPU {
			return cpuClone.ToPersistentGPU()
		} else {
			return cpuClone.ToGPU()
		}
	}
	
	switch t.DType {
	case Float32:
		if t.Data == nil {
			return nil, fmt.Errorf("tensor has nil data")
		}
		data := t.Data.([]float32)
		cloneData := make([]float32, len(data))
		copy(cloneData, data)
		clone.Data = cloneData
	case Int32:
		if t.Data == nil {
			return nil, fmt.Errorf("tensor has nil data")
		}
		data := t.Data.([]int32)
		cloneData := make([]int32, len(data))
		copy(cloneData, data)
		clone.Data = cloneData
	default:
		return nil, fmt.Errorf("unsupported dtype for Clone: %s", t.DType)
	}

	return clone, nil
}

func (t *Tensor) GetFloat32Data() ([]float32, error) {
	if t.DType != Float32 {
		return nil, fmt.Errorf("tensor dtype is %s, not Float32", t.DType)
	}
	return t.Data.([]float32), nil
}

func (t *Tensor) GetInt32Data() ([]int32, error) {
	if t.DType != Int32 {
		return nil, fmt.Errorf("tensor dtype is %s, not Int32", t.DType)
	}
	return t.Data.([]int32), nil
}

func (t *Tensor) Item() (interface{}, error) {
	if t.NumElems != 1 {
		return nil, fmt.Errorf("item() can only be called on tensors with exactly one element, got %d", t.NumElems)
	}
	
	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		return data[0], nil
	case Int32:
		data := t.Data.([]int32)
		return data[0], nil
	default:
		return nil, fmt.Errorf("unsupported dtype for Item: %s", t.DType)
	}
}

func (t *Tensor) At(indices ...int) (interface{}, error) {
	if len(indices) != len(t.Shape) {
		return nil, fmt.Errorf("expected %d indices, got %d", len(t.Shape), len(indices))
	}
	
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			return nil, fmt.Errorf("index %d out of bounds for dimension %d (size %d)", idx, i, t.Shape[i])
		}
	}

	// Use view-aware data access
	linearIndex := t.getLinearIndex(indices)
	dataBuffer := t.getDataBuffer()
	
	switch t.DType {
	case Float32:
		return dataBuffer.([]float32)[linearIndex], nil
	case Int32:
		return dataBuffer.([]int32)[linearIndex], nil
	default:
		return nil, fmt.Errorf("unsupported dtype for At: %s", t.DType)
	}
}

func (t *Tensor) SetAt(value interface{}, indices ...int) error {
	if len(indices) != len(t.Shape) {
		return fmt.Errorf("expected %d indices, got %d", len(t.Shape), len(indices))
	}
	
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			return fmt.Errorf("index %d out of bounds for dimension %d (size %d)", idx, i, t.Shape[i])
		}
	}

	// Use view-aware data access
	linearIndex := t.getLinearIndex(indices)
	dataBuffer := t.getDataBuffer()
	
	switch t.DType {
	case Float32:
		val, ok := value.(float32)
		if !ok {
			return fmt.Errorf("expected float32 value for Float32 tensor")
		}
		dataBuffer.([]float32)[linearIndex] = val
	case Int32:
		val, ok := value.(int32)
		if !ok {
			return fmt.Errorf("expected int32 value for Int32 tensor")
		}
		dataBuffer.([]int32)[linearIndex] = val
	default:
		return fmt.Errorf("unsupported dtype for SetAt: %s", t.DType)
	}
	
	return nil
}

func (t *Tensor) Size() []int {
	result := make([]int, len(t.Shape))
	copy(result, t.Shape)
	return result
}

func (t *Tensor) Numel() int {
	return t.NumElems
}

func (t *Tensor) Dim() int {
	return len(t.Shape)
}

func (t *Tensor) Equal(other *Tensor) (bool, error) {
	if t.DType != other.DType {
		return false, nil
	}
	
	if len(t.Shape) != len(other.Shape) {
		return false, nil
	}
	
	for i, dim := range t.Shape {
		if dim != other.Shape[i] {
			return false, nil
		}
	}

	switch t.DType {
	case Float32:
		data1 := t.Data.([]float32)
		data2 := other.Data.([]float32)
		for i := 0; i < t.NumElems; i++ {
			if data1[i] != data2[i] {
				return false, nil
			}
		}
	case Int32:
		data1 := t.Data.([]int32)
		data2 := other.Data.([]int32)
		for i := 0; i < t.NumElems; i++ {
			if data1[i] != data2[i] {
				return false, nil
			}
		}
	default:
		return false, fmt.Errorf("unsupported dtype for Equal: %s", t.DType)
	}

	return true, nil
}

func (t *Tensor) ToDevice(device DeviceType) (*Tensor, error) {
	// Validate device type first
	if device != CPU && device != GPU && device != PersistentGPU {
		return nil, fmt.Errorf("invalid device type: %v (valid types: CPU, GPU, PersistentGPU)", device)
	}
	
	if t.Device == device {
		return t, nil
	}

	// Handle GPU device transfers
	if device == GPU || device == PersistentGPU {
		if t.Device == CPU {
			// CPU -> GPU: Create GPU tensor
			return NewTensor(t.Shape, t.DType, device, t.Data)
		} else if t.Device == GPU || t.Device == PersistentGPU {
			// GPU -> GPU: Convert through CPU if changing persistence mode
			if t.Device != device {
				cpuTensor, err := t.ToCPU()
				if err != nil {
					return nil, err
				}
				return cpuTensor.ToDevice(device)
			}
			return t, nil
		}
	}

	// GPU -> CPU conversion
	if (t.Device == GPU || t.Device == PersistentGPU) && device == CPU {
		return t.ToCPU()
	}

	// CPU -> CPU (just clone)
	if device == CPU {
		result, err := t.Clone()
		if err != nil {
			return nil, err
		}
		result.Device = device
		return result, nil
	}

	// Should never reach here due to validation above
	return nil, fmt.Errorf("unsupported device conversion from %v to %v", t.Device, device)
}

func (t *Tensor) PrintData(maxElements int) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s)\n", t.Shape, t.DType, t.Device))
	
	if maxElements <= 0 {
		maxElements = 20
	}
	
	elementsToShow := t.NumElems
	if elementsToShow > maxElements {
		elementsToShow = maxElements
	}

	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		sb.WriteString("[")
		for i := 0; i < elementsToShow; i++ {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%.4f", data[i]))
		}
		if t.NumElems > maxElements {
			sb.WriteString(fmt.Sprintf(", ... (%d more elements)", t.NumElems-maxElements))
		}
		sb.WriteString("]")
	case Int32:
		data := t.Data.([]int32)
		sb.WriteString("[")
		for i := 0; i < elementsToShow; i++ {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%d", data[i]))
		}
		if t.NumElems > maxElements {
			sb.WriteString(fmt.Sprintf(", ... (%d more elements)", t.NumElems-maxElements))
		}
		sb.WriteString("]")
	}
	
	return sb.String()
}

func ZeroGrad(tensors []*Tensor) {
	for _, t := range tensors {
		if t.requiresGrad && t.grad != nil {
			switch t.DType {
			case Float32:
				data := t.grad.Data.([]float32)
				for i := range data {
					data[i] = 0
				}
			case Int32:
				data := t.grad.Data.([]int32)
				for i := range data {
					data[i] = 0
				}
			}
		}
	}
}

func (t *Tensor) Cleanup() {
	t.Data = nil
	t.grad = nil
	t.creator = nil
}

// FromScalar creates a scalar tensor from a float64 value
func FromScalar(value float64, dtype DType, device DeviceType) *Tensor {
	switch dtype {
	case Float32:
		data := []float32{float32(value)}
		tensor, _ := NewTensor([]int{}, dtype, device, data)
		return tensor
	case Int32:
		data := []int32{int32(value)}
		tensor, _ := NewTensor([]int{}, dtype, device, data)
		return tensor
	default:
		// Default to Float32
		data := []float32{float32(value)}
		tensor, _ := NewTensor([]int{}, dtype, device, data)
		return tensor
	}
}

// Sqrt computes the square root of a tensor element-wise
func Sqrt(t *Tensor) (*Tensor, error) {
	if t.DType != Float32 {
		return nil, fmt.Errorf("sqrt only supports Float32 tensors")
	}
	
	data := t.Data.([]float32)
	result := make([]float32, len(data))
	
	for i, val := range data {
		if val < 0 {
			// Produce NaN for negative values instead of returning an error
			result[i] = float32(math.NaN())
		} else {
			result[i] = float32(math.Sqrt(float64(val)))
		}
	}
	
	return NewTensor(t.Shape, t.DType, t.Device, result)
}