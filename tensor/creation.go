package tensor

import (
	"fmt"
	"math/rand"
	"time"
)

func NewTensor(shape []int, dtype DType, device DeviceType, data interface{}) (*Tensor, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}

	numElems := calculateNumElements(shape)
	strides := calculateStrides(shape)

	tensor := &Tensor{
		Shape:    shape,
		Strides:  strides,
		DType:    dtype,
		Device:   device,
		NumElems: numElems,
	}

	if data != nil {
		if err := tensor.setData(data); err != nil {
			return nil, err
		}
	}

	return tensor, nil
}

func (t *Tensor) setData(data interface{}) error {
	switch t.DType {
	case Float32:
		switch d := data.(type) {
		case []float32:
			if len(d) != t.NumElems {
				return fmt.Errorf("data length %d does not match tensor size %d", len(d), t.NumElems)
			}
			t.Data = d
		case float32:
			slice := make([]float32, t.NumElems)
			for i := range slice {
				slice[i] = d
			}
			t.Data = slice
		default:
			return fmt.Errorf("unsupported data type for Float32 tensor: %T", data)
		}
	case Int32:
		switch d := data.(type) {
		case []int32:
			if len(d) != t.NumElems {
				return fmt.Errorf("data length %d does not match tensor size %d", len(d), t.NumElems)
			}
			t.Data = d
		case int32:
			slice := make([]int32, t.NumElems)
			for i := range slice {
				slice[i] = d
			}
			t.Data = slice
		default:
			return fmt.Errorf("unsupported data type for Int32 tensor: %T", data)
		}
	default:
		return fmt.Errorf("unsupported dtype: %s", t.DType)
	}
	return nil
}

func Zeros(shape []int, dtype DType, device DeviceType) (*Tensor, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}

	numElems := calculateNumElements(shape)
	
	var data interface{}
	switch dtype {
	case Float32:
		data = make([]float32, numElems)
	case Int32:
		data = make([]int32, numElems)
	default:
		return nil, fmt.Errorf("unsupported dtype for Zeros: %s", dtype)
	}

	return NewTensor(shape, dtype, device, data)
}

func Ones(shape []int, dtype DType, device DeviceType) (*Tensor, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}

	numElems := calculateNumElements(shape)
	
	var data interface{}
	switch dtype {
	case Float32:
		slice := make([]float32, numElems)
		for i := range slice {
			slice[i] = 1.0
		}
		data = slice
	case Int32:
		slice := make([]int32, numElems)
		for i := range slice {
			slice[i] = 1
		}
		data = slice
	default:
		return nil, fmt.Errorf("unsupported dtype for Ones: %s", dtype)
	}

	return NewTensor(shape, dtype, device, data)
}

func Random(shape []int, dtype DType, device DeviceType) (*Tensor, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}

	numElems := calculateNumElements(shape)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	var data interface{}
	switch dtype {
	case Float32:
		slice := make([]float32, numElems)
		for i := range slice {
			slice[i] = rng.Float32()
		}
		data = slice
	case Int32:
		slice := make([]int32, numElems)
		for i := range slice {
			slice[i] = rng.Int31()
		}
		data = slice
	default:
		return nil, fmt.Errorf("unsupported dtype for Random: %s", dtype)
	}

	return NewTensor(shape, dtype, device, data)
}

func RandomNormal(shape []int, mean, std float32, dtype DType, device DeviceType) (*Tensor, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}

	if dtype != Float32 {
		return nil, fmt.Errorf("RandomNormal only supports Float32 dtype")
	}

	numElems := calculateNumElements(shape)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	slice := make([]float32, numElems)
	for i := range slice {
		slice[i] = float32(rng.NormFloat64())*std + mean
	}

	return NewTensor(shape, dtype, device, slice)
}

func Full(shape []int, value interface{}, dtype DType, device DeviceType) (*Tensor, error) {
	return NewTensor(shape, dtype, device, value)
}