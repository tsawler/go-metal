package tensor

import (
	"fmt"
	"strings"
)

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

	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		cloneData := make([]float32, len(data))
		copy(cloneData, data)
		clone.Data = cloneData
	case Int32:
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

	linearIndex := getIndex(indices, t.Strides)
	
	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		return data[linearIndex], nil
	case Int32:
		data := t.Data.([]int32)
		return data[linearIndex], nil
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

	linearIndex := getIndex(indices, t.Strides)
	
	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		val, ok := value.(float32)
		if !ok {
			return fmt.Errorf("expected float32 value for Float32 tensor")
		}
		data[linearIndex] = val
	case Int32:
		data := t.Data.([]int32)
		val, ok := value.(int32)
		if !ok {
			return fmt.Errorf("expected int32 value for Int32 tensor")
		}
		data[linearIndex] = val
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
	if t.Device == device {
		return t, nil
	}

	if device == GPU {
		return nil, fmt.Errorf("GPU device transfer not implemented in Phase 1")
	}

	result, err := t.Clone()
	if err != nil {
		return nil, err
	}
	
	result.Device = device
	return result, nil
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