package tensor

import (
	"reflect"
	"strings"
	"testing"
)

func TestClone(t *testing.T) {
	t.Run("Float32 clone", func(t *testing.T) {
		original, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		original.SetRequiresGrad(true)
		
		clone, err := original.Clone()
		if err != nil {
			t.Fatalf("Clone failed: %v", err)
		}
		
		// Check that all fields are copied
		if !reflect.DeepEqual(clone.Shape, original.Shape) {
			t.Errorf("Shape not cloned correctly")
		}
		
		if !reflect.DeepEqual(clone.Strides, original.Strides) {
			t.Errorf("Strides not cloned correctly")
		}
		
		if clone.DType != original.DType {
			t.Errorf("DType not cloned correctly")
		}
		
		if clone.Device != original.Device {
			t.Errorf("Device not cloned correctly")
		}
		
		if clone.NumElems != original.NumElems {
			t.Errorf("NumElems not cloned correctly")
		}
		
		if clone.requiresGrad != original.requiresGrad {
			t.Errorf("requiresGrad not cloned correctly")
		}
		
		// Check that data is deep copied
		originalData := original.Data.([]float32)
		cloneData := clone.Data.([]float32)
		if !reflect.DeepEqual(originalData, cloneData) {
			t.Errorf("Data not cloned correctly")
		}
		
		// Verify it's a deep copy by modifying original
		originalData[0] = 999
		if cloneData[0] == 999 {
			t.Error("Clone shares data with original (not deep copy)")
		}
	})

	t.Run("Int32 clone", func(t *testing.T) {
		original, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		clone, err := original.Clone()
		if err != nil {
			t.Fatalf("Clone failed: %v", err)
		}
		
		originalData := original.Data.([]int32)
		cloneData := clone.Data.([]int32)
		if !reflect.DeepEqual(originalData, cloneData) {
			t.Errorf("Data not cloned correctly")
		}
	})

	t.Run("Unsupported dtype", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{2, 2},
			DType:    Float16,
			NumElems: 4,
			Data:     []float32{1, 2, 3, 4},
		}
		
		_, err := tensor.Clone()
		if err == nil {
			t.Error("Expected error for unsupported dtype")
		}
	})
}

func TestGetFloat32Data(t *testing.T) {
	t.Run("Valid Float32 tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		data, err := tensor.GetFloat32Data()
		if err != nil {
			t.Fatalf("GetFloat32Data failed: %v", err)
		}
		
		expected := []float32{1, 2, 3, 4}
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Non-Float32 tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		_, err := tensor.GetFloat32Data()
		if err == nil {
			t.Error("Expected error for non-Float32 tensor")
		}
	})
}

func TestGetInt32Data(t *testing.T) {
	t.Run("Valid Int32 tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		data, err := tensor.GetInt32Data()
		if err != nil {
			t.Fatalf("GetInt32Data failed: %v", err)
		}
		
		expected := []int32{1, 2, 3, 4}
		if !reflect.DeepEqual(data, expected) {
			t.Errorf("Data = %v, expected %v", data, expected)
		}
	})

	t.Run("Non-Int32 tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		_, err := tensor.GetInt32Data()
		if err == nil {
			t.Error("Expected error for non-Int32 tensor")
		}
	})
}

func TestItem(t *testing.T) {
	t.Run("Scalar Float32 tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{1}, Float32, CPU, []float32{42.5})
		
		item, err := tensor.Item()
		if err != nil {
			t.Fatalf("Item failed: %v", err)
		}
		
		if item != float32(42.5) {
			t.Errorf("Item = %v, expected 42.5", item)
		}
	})

	t.Run("Scalar Int32 tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{1}, Int32, CPU, []int32{42})
		
		item, err := tensor.Item()
		if err != nil {
			t.Fatalf("Item failed: %v", err)
		}
		
		if item != int32(42) {
			t.Errorf("Item = %v, expected 42", item)
		}
	})

	t.Run("Multi-element tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		_, err := tensor.Item()
		if err == nil {
			t.Error("Expected error for multi-element tensor")
		}
	})

	t.Run("Unsupported dtype", func(t *testing.T) {
		tensor := &Tensor{
			Shape:    []int{1},
			DType:    Float16,
			NumElems: 1,
		}
		
		_, err := tensor.Item()
		if err == nil {
			t.Error("Expected error for unsupported dtype")
		}
	})
}

func TestAt(t *testing.T) {
	t.Run("Valid access Float32", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		val, err := tensor.At(1, 2)
		if err != nil {
			t.Fatalf("At failed: %v", err)
		}
		
		if val != float32(6) {
			t.Errorf("At(1, 2) = %v, expected 6", val)
		}
	})

	t.Run("Valid access Int32", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		val, err := tensor.At(0, 1)
		if err != nil {
			t.Fatalf("At failed: %v", err)
		}
		
		if val != int32(2) {
			t.Errorf("At(0, 1) = %v, expected 2", val)
		}
	})

	t.Run("Wrong number of indices", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := tensor.At(1)
		if err == nil {
			t.Error("Expected error for wrong number of indices")
		}
	})

	t.Run("Out of bounds access", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := tensor.At(2, 0)
		if err == nil {
			t.Error("Expected error for out of bounds access")
		}
	})

	t.Run("Negative index", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 3}, Float32, CPU, []float32{1, 2, 3, 4, 5, 6})
		
		_, err := tensor.At(-1, 0)
		if err == nil {
			t.Error("Expected error for negative index")
		}
	})
}

func TestSetAt(t *testing.T) {
	t.Run("Valid set Float32", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		err := tensor.SetAt(float32(99), 1, 1)
		if err != nil {
			t.Fatalf("SetAt failed: %v", err)
		}
		
		val, _ := tensor.At(1, 1)
		if val != float32(99) {
			t.Errorf("After SetAt, At(1, 1) = %v, expected 99", val)
		}
	})

	t.Run("Valid set Int32", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		err := tensor.SetAt(int32(99), 0, 1)
		if err != nil {
			t.Fatalf("SetAt failed: %v", err)
		}
		
		val, _ := tensor.At(0, 1)
		if val != int32(99) {
			t.Errorf("After SetAt, At(0, 1) = %v, expected 99", val)
		}
	})

	t.Run("Wrong value type", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		err := tensor.SetAt(int32(99), 0, 0)
		if err == nil {
			t.Error("Expected error for wrong value type")
		}
	})

	t.Run("Out of bounds", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		err := tensor.SetAt(float32(99), 2, 0)
		if err == nil {
			t.Error("Expected error for out of bounds access")
		}
	})
}

func TestTensorProperties(t *testing.T) {
	tensor, _ := NewTensor([]int{2, 3, 4}, Float32, CPU, make([]float32, 24))
	
	t.Run("Size", func(t *testing.T) {
		size := tensor.Size()
		expected := []int{2, 3, 4}
		if !reflect.DeepEqual(size, expected) {
			t.Errorf("Size() = %v, expected %v", size, expected)
		}
		
		// Verify it's a copy, not reference
		size[0] = 999
		if tensor.Shape[0] == 999 {
			t.Error("Size() should return a copy, not reference")
		}
	})

	t.Run("Numel", func(t *testing.T) {
		numel := tensor.Numel()
		if numel != 24 {
			t.Errorf("Numel() = %d, expected 24", numel)
		}
	})

	t.Run("Dim", func(t *testing.T) {
		dim := tensor.Dim()
		if dim != 3 {
			t.Errorf("Dim() = %d, expected 3", dim)
		}
	})
}

func TestEqual(t *testing.T) {
	t.Run("Equal Float32 tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		equal, err := a.Equal(b)
		if err != nil {
			t.Fatalf("Equal failed: %v", err)
		}
		
		if !equal {
			t.Error("Expected tensors to be equal")
		}
	})

	t.Run("Different data", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 5})
		
		equal, err := a.Equal(b)
		if err != nil {
			t.Fatalf("Equal failed: %v", err)
		}
		
		if equal {
			t.Error("Expected tensors to be different")
		}
	})

	t.Run("Different dtypes", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		equal, err := a.Equal(b)
		if err != nil {
			t.Fatalf("Equal failed: %v", err)
		}
		
		if equal {
			t.Error("Expected tensors with different dtypes to be different")
		}
	})

	t.Run("Different shapes", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		b, _ := NewTensor([]int{1, 4}, Float32, CPU, []float32{1, 2, 3, 4})
		
		equal, err := a.Equal(b)
		if err != nil {
			t.Fatalf("Equal failed: %v", err)
		}
		
		if equal {
			t.Error("Expected tensors with different shapes to be different")
		}
	})
}

func TestToDevice(t *testing.T) {
	t.Run("Same device", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		result, err := tensor.ToDevice(CPU)
		if err != nil {
			t.Fatalf("ToDevice failed: %v", err)
		}
		
		if result != tensor {
			t.Error("Expected same tensor when device doesn't change")
		}
	})

	t.Run("GPU transfer not implemented", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		
		_, err := tensor.ToDevice(GPU)
		if err == nil {
			t.Error("Expected error for GPU transfer in Phase 1")
		}
	})
}

func TestPrintData(t *testing.T) {
	t.Run("Float32 print", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.5, 2.7, 3.1, 4.9})
		
		result := tensor.PrintData(10)
		
		if !strings.Contains(result, "Tensor(shape=[2 2]") {
			t.Error("PrintData should contain tensor info")
		}
		
		if !strings.Contains(result, "1.5000") {
			t.Error("PrintData should contain formatted data")
		}
	})

	t.Run("Int32 print", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		
		result := tensor.PrintData(10)
		
		if !strings.Contains(result, "Int32") {
			t.Error("PrintData should show correct dtype")
		}
	})

	t.Run("Truncated output", func(t *testing.T) {
		data := make([]float32, 100)
		for i := range data {
			data[i] = float32(i)
		}
		tensor, _ := NewTensor([]int{100}, Float32, CPU, data)
		
		result := tensor.PrintData(5)
		
		if !strings.Contains(result, "... (95 more elements)") {
			t.Error("PrintData should show truncation message")
		}
	})
}

func TestZeroGrad(t *testing.T) {
	t.Run("Zero Float32 gradients", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		tensor.SetRequiresGrad(true)
		
		// Set up a gradient
		grad, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8})
		tensor.grad = grad
		
		ZeroGrad([]*Tensor{tensor})
		
		gradData := tensor.grad.Data.([]float32)
		for i, val := range gradData {
			if val != 0 {
				t.Errorf("Gradient[%d] = %f, expected 0", i, val)
			}
		}
	})

	t.Run("Skip tensors without requiresGrad", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		grad, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8})
		tensor.grad = grad
		
		ZeroGrad([]*Tensor{tensor})
		
		// Gradient should remain unchanged
		gradData := tensor.grad.Data.([]float32)
		if gradData[0] != 5 {
			t.Error("Gradient should not be zeroed for tensor without requiresGrad")
		}
	})
}

func TestRelease(t *testing.T) {
	t.Run("Release tensor", func(t *testing.T) {
		tensor, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1, 2, 3, 4})
		grad, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5, 6, 7, 8})
		tensor.grad = grad
		
		tensor.Release()
		
		if tensor.Data != nil {
			t.Error("Data should be nil after release")
		}
		
		if tensor.grad != nil {
			t.Error("Grad should be nil after release")
		}
		
		if tensor.creator != nil {
			t.Error("Creator should be nil after release")
		}
	})
}