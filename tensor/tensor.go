package tensor

import (
	"fmt"
)

type DType int

const (
	Float32 DType = iota
	Float16
	Int32
)

func (d DType) String() string {
	switch d {
	case Float32:
		return "Float32"
	case Float16:
		return "Float16"
	case Int32:
		return "Int32"
	default:
		return "Unknown"
	}
}

type DeviceType int

const (
	CPU DeviceType = iota
	GPU
)

func (d DeviceType) String() string {
	switch d {
	case CPU:
		return "CPU"
	case GPU:
		return "GPU"
	default:
		return "Unknown"
	}
}

type Operation interface {
	Forward(...*Tensor) *Tensor
	Backward(gradOut *Tensor) []*Tensor
}

type Tensor struct {
	Shape        []int
	Strides      []int
	DType        DType
	Device       DeviceType
	Data         interface{}
	NumElems     int
	requiresGrad bool
	grad         *Tensor
	creator      Operation
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s, elements=%d)", 
		t.Shape, t.DType, t.Device, t.NumElems)
}

func (t *Tensor) RequiresGrad() bool {
	return t.requiresGrad
}

func (t *Tensor) SetRequiresGrad(requires bool) {
	t.requiresGrad = requires
}

func (t *Tensor) Grad() *Tensor {
	return t.grad
}

func calculateStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func calculateNumElements(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	
	elements := 1
	for _, dim := range shape {
		elements *= dim
	}
	return elements
}

func validateShape(shape []int) error {
	for i, dim := range shape {
		if dim <= 0 {
			return fmt.Errorf("invalid shape: dimension %d has size %d, must be positive", i, dim)
		}
	}
	return nil
}

func getSizeForDType(dtype DType) int {
	switch dtype {
	case Float32:
		return 4
	case Float16:
		return 2
	case Int32:
		return 4
	default:
		return 4
	}
}