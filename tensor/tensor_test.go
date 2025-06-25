package tensor

import (
	"reflect"
	"testing"
)

func TestDTypeString(t *testing.T) {
	tests := []struct {
		dtype    DType
		expected string
	}{
		{Float32, "Float32"},
		{Float16, "Float16"},
		{Int32, "Int32"},
		{DType(999), "Unknown"},
	}

	for _, test := range tests {
		result := test.dtype.String()
		if result != test.expected {
			t.Errorf("DType.String() = %s, expected %s", result, test.expected)
		}
	}
}

func TestDeviceTypeString(t *testing.T) {
	tests := []struct {
		device   DeviceType
		expected string
	}{
		{CPU, "CPU"},
		{GPU, "GPU"},
		{DeviceType(999), "Unknown"},
	}

	for _, test := range tests {
		result := test.device.String()
		if result != test.expected {
			t.Errorf("DeviceType.String() = %s, expected %s", result, test.expected)
		}
	}
}

func TestCalculateStrides(t *testing.T) {
	tests := []struct {
		shape    []int
		expected []int
	}{
		{[]int{}, []int{}},
		{[]int{5}, []int{1}},
		{[]int{2, 3}, []int{3, 1}},
		{[]int{2, 3, 4}, []int{12, 4, 1}},
		{[]int{1, 5, 1, 3}, []int{15, 3, 3, 1}},
	}

	for _, test := range tests {
		result := calculateStrides(test.shape)
		if !reflect.DeepEqual(result, test.expected) {
			t.Errorf("calculateStrides(%v) = %v, expected %v", test.shape, result, test.expected)
		}
	}
}

func TestCalculateNumElements(t *testing.T) {
	tests := []struct {
		shape    []int
		expected int
	}{
		{[]int{}, 0},
		{[]int{5}, 5},
		{[]int{2, 3}, 6},
		{[]int{2, 3, 4}, 24},
		{[]int{1, 5, 1, 3}, 15},
	}

	for _, test := range tests {
		result := calculateNumElements(test.shape)
		if result != test.expected {
			t.Errorf("calculateNumElements(%v) = %d, expected %d", test.shape, result, test.expected)
		}
	}
}

func TestValidateShape(t *testing.T) {
	tests := []struct {
		shape   []int
		wantErr bool
	}{
		{[]int{}, false},
		{[]int{5}, false},
		{[]int{2, 3}, false},
		{[]int{2, 3, 4}, false},
		{[]int{0}, true},
		{[]int{2, 0}, true},
		{[]int{-1}, true},
		{[]int{2, -3}, true},
	}

	for _, test := range tests {
		err := validateShape(test.shape)
		if (err != nil) != test.wantErr {
			t.Errorf("validateShape(%v) error = %v, wantErr %v", test.shape, err, test.wantErr)
		}
	}
}

func TestGetSizeForDType(t *testing.T) {
	tests := []struct {
		dtype    DType
		expected int
	}{
		{Float32, 4},
		{Float16, 2},
		{Int32, 4},
		{DType(999), 4}, // default case
	}

	for _, test := range tests {
		result := getSizeForDType(test.dtype)
		if result != test.expected {
			t.Errorf("getSizeForDType(%v) = %d, expected %d", test.dtype, result, test.expected)
		}
	}
}

func TestTensorString(t *testing.T) {
	tensor := &Tensor{
		Shape:    []int{2, 3},
		DType:    Float32,
		Device:   CPU,
		NumElems: 6,
	}

	result := tensor.String()
	expected := "Tensor(shape=[2 3], dtype=Float32, device=CPU, elements=6)"
	if result != expected {
		t.Errorf("Tensor.String() = %s, expected %s", result, expected)
	}
}

func TestTensorRequiresGrad(t *testing.T) {
	tensor := &Tensor{requiresGrad: false}
	
	if tensor.RequiresGrad() {
		t.Error("RequiresGrad() should return false initially")
	}
	
	tensor.SetRequiresGrad(true)
	if !tensor.RequiresGrad() {
		t.Error("RequiresGrad() should return true after setting to true")
	}
	
	tensor.SetRequiresGrad(false)
	if tensor.RequiresGrad() {
		t.Error("RequiresGrad() should return false after setting to false")
	}
}

func TestTensorGrad(t *testing.T) {
	tensor := &Tensor{}
	
	if tensor.Grad() != nil {
		t.Error("Grad() should return nil initially")
	}
	
	gradTensor := &Tensor{Shape: []int{2, 2}}
	tensor.grad = gradTensor
	
	if tensor.Grad() != gradTensor {
		t.Error("Grad() should return the set gradient tensor")
	}
}

func TestTensorSetData(t *testing.T) {
	t.Run("Float32 slice", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Float32,
			NumElems: 4,
		}
		
		data := []float32{1.0, 2.0, 3.0, 4.0}
		err := tensor.setData(data)
		if err != nil {
			t.Errorf("setData failed: %v", err)
		}
		
		result := tensor.Data.([]float32)
		if !reflect.DeepEqual(result, data) {
			t.Errorf("Data = %v, expected %v", result, data)
		}
	})

	t.Run("Float32 scalar", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Float32,
			NumElems: 4,
		}
		
		err := tensor.setData(float32(5.0))
		if err != nil {
			t.Errorf("setData failed: %v", err)
		}
		
		result := tensor.Data.([]float32)
		expected := []float32{5.0, 5.0, 5.0, 5.0}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("Data = %v, expected %v", result, expected)
		}
	})

	t.Run("Int32 slice", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Int32,
			NumElems: 4,
		}
		
		data := []int32{1, 2, 3, 4}
		err := tensor.setData(data)
		if err != nil {
			t.Errorf("setData failed: %v", err)
		}
		
		result := tensor.Data.([]int32)
		if !reflect.DeepEqual(result, data) {
			t.Errorf("Data = %v, expected %v", result, data)
		}
	})

	t.Run("Wrong data length", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Float32,
			NumElems: 4,
		}
		
		data := []float32{1.0, 2.0} // Wrong length
		err := tensor.setData(data)
		if err == nil {
			t.Error("Expected error for wrong data length")
		}
	})

	t.Run("Unsupported data type", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Float32,
			NumElems: 4,
		}
		
		err := tensor.setData("invalid")
		if err == nil {
			t.Error("Expected error for unsupported data type")
		}
	})

	t.Run("Unsupported dtype", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Float16, // Not fully implemented
			NumElems: 4,
		}
		
		err := tensor.setData([]float32{1.0, 2.0, 3.0, 4.0})
		if err == nil {
			t.Error("Expected error for unsupported dtype")
		}
	})
}