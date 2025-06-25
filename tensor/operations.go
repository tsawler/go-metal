package tensor

import (
	"fmt"
	"math"
)

func checkCompatibility(t1, t2 *Tensor) error {
	if t1.DType != t2.DType {
		return fmt.Errorf("tensors must have same dtype: %s vs %s", t1.DType, t2.DType)
	}
	if t1.Device != t2.Device {
		return fmt.Errorf("tensors must be on same device: %s vs %s", t1.Device, t2.Device)
	}
	return nil
}

func checkShapesCompatible(shape1, shape2 []int) ([]int, error) {
	if len(shape1) == 0 || len(shape2) == 0 {
		return nil, fmt.Errorf("cannot operate on empty tensors")
	}

	if len(shape1) != len(shape2) {
		return nil, fmt.Errorf("tensor shapes must have same number of dimensions: %v vs %v", shape1, shape2)
	}

	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return nil, fmt.Errorf("tensor shapes must match: %v vs %v", shape1, shape2)
		}
	}

	return shape1, nil
}

func Add(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	outputShape, err := checkShapesCompatible(t1.Shape, t2.Shape)
	if err != nil {
		return nil, err
	}

	result, err := Zeros(outputShape, t1.DType, t1.Device)
	if err != nil {
		return nil, err
	}

	switch t1.DType {
	case Float32:
		data1 := t1.Data.([]float32)
		data2 := t2.Data.([]float32)
		resultData := result.Data.([]float32)
		
		for i := 0; i < t1.NumElems; i++ {
			resultData[i] = data1[i] + data2[i]
		}
	case Int32:
		data1 := t1.Data.([]int32)
		data2 := t2.Data.([]int32)
		resultData := result.Data.([]int32)
		
		for i := 0; i < t1.NumElems; i++ {
			resultData[i] = data1[i] + data2[i]
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for Add: %s", t1.DType)
	}

	return result, nil
}

func Sub(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	outputShape, err := checkShapesCompatible(t1.Shape, t2.Shape)
	if err != nil {
		return nil, err
	}

	result, err := Zeros(outputShape, t1.DType, t1.Device)
	if err != nil {
		return nil, err
	}

	switch t1.DType {
	case Float32:
		data1 := t1.Data.([]float32)
		data2 := t2.Data.([]float32)
		resultData := result.Data.([]float32)
		
		for i := 0; i < t1.NumElems; i++ {
			resultData[i] = data1[i] - data2[i]
		}
	case Int32:
		data1 := t1.Data.([]int32)
		data2 := t2.Data.([]int32)
		resultData := result.Data.([]int32)
		
		for i := 0; i < t1.NumElems; i++ {
			resultData[i] = data1[i] - data2[i]
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for Sub: %s", t1.DType)
	}

	return result, nil
}

func Mul(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	outputShape, err := checkShapesCompatible(t1.Shape, t2.Shape)
	if err != nil {
		return nil, err
	}

	result, err := Zeros(outputShape, t1.DType, t1.Device)
	if err != nil {
		return nil, err
	}

	switch t1.DType {
	case Float32:
		data1 := t1.Data.([]float32)
		data2 := t2.Data.([]float32)
		resultData := result.Data.([]float32)
		
		for i := 0; i < t1.NumElems; i++ {
			resultData[i] = data1[i] * data2[i]
		}
	case Int32:
		data1 := t1.Data.([]int32)
		data2 := t2.Data.([]int32)
		resultData := result.Data.([]int32)
		
		for i := 0; i < t1.NumElems; i++ {
			resultData[i] = data1[i] * data2[i]
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for Mul: %s", t1.DType)
	}

	return result, nil
}

func Div(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	outputShape, err := checkShapesCompatible(t1.Shape, t2.Shape)
	if err != nil {
		return nil, err
	}

	result, err := Zeros(outputShape, t1.DType, t1.Device)
	if err != nil {
		return nil, err
	}

	switch t1.DType {
	case Float32:
		data1 := t1.Data.([]float32)
		data2 := t2.Data.([]float32)
		resultData := result.Data.([]float32)
		
		for i := 0; i < t1.NumElems; i++ {
			if data2[i] == 0 {
				return nil, fmt.Errorf("division by zero at index %d", i)
			}
			resultData[i] = data1[i] / data2[i]
		}
	case Int32:
		data1 := t1.Data.([]int32)
		data2 := t2.Data.([]int32)
		resultData := result.Data.([]int32)
		
		for i := 0; i < t1.NumElems; i++ {
			if data2[i] == 0 {
				return nil, fmt.Errorf("division by zero at index %d", i)
			}
			resultData[i] = data1[i] / data2[i]
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for Div: %s", t1.DType)
	}

	return result, nil
}

func ReLU(t *Tensor) (*Tensor, error) {
	result, err := Zeros(t.Shape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}

	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		resultData := result.Data.([]float32)
		
		for i := 0; i < t.NumElems; i++ {
			if data[i] > 0 {
				resultData[i] = data[i]
			}
		}
	case Int32:
		data := t.Data.([]int32)
		resultData := result.Data.([]int32)
		
		for i := 0; i < t.NumElems; i++ {
			if data[i] > 0 {
				resultData[i] = data[i]
			}
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for ReLU: %s", t.DType)
	}

	return result, nil
}

func Sigmoid(t *Tensor) (*Tensor, error) {
	if t.DType != Float32 {
		return nil, fmt.Errorf("Sigmoid only supports Float32 dtype")
	}

	result, err := Zeros(t.Shape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}

	data := t.Data.([]float32)
	resultData := result.Data.([]float32)
	
	for i := 0; i < t.NumElems; i++ {
		resultData[i] = float32(1.0 / (1.0 + math.Exp(-float64(data[i]))))
	}

	return result, nil
}

func Tanh(t *Tensor) (*Tensor, error) {
	if t.DType != Float32 {
		return nil, fmt.Errorf("Tanh only supports Float32 dtype")
	}

	result, err := Zeros(t.Shape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}

	data := t.Data.([]float32)
	resultData := result.Data.([]float32)
	
	for i := 0; i < t.NumElems; i++ {
		resultData[i] = float32(math.Tanh(float64(data[i])))
	}

	return result, nil
}

func Exp(t *Tensor) (*Tensor, error) {
	if t.DType != Float32 {
		return nil, fmt.Errorf("Exp only supports Float32 dtype")
	}

	result, err := Zeros(t.Shape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}

	data := t.Data.([]float32)
	resultData := result.Data.([]float32)
	
	for i := 0; i < t.NumElems; i++ {
		resultData[i] = float32(math.Exp(float64(data[i])))
	}

	return result, nil
}

func Log(t *Tensor) (*Tensor, error) {
	if t.DType != Float32 {
		return nil, fmt.Errorf("Log only supports Float32 dtype")
	}

	result, err := Zeros(t.Shape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}

	data := t.Data.([]float32)
	resultData := result.Data.([]float32)
	
	for i := 0; i < t.NumElems; i++ {
		if data[i] <= 0 {
			return nil, fmt.Errorf("log of non-positive value at index %d: %f", i, data[i])
		}
		resultData[i] = float32(math.Log(float64(data[i])))
	}

	return result, nil
}