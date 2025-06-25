package tensor

import (
	"math"
	"reflect"
	"testing"
)

func TestCheckCompatibility(t *testing.T) {
	t1 := &Tensor{DType: Float32, Device: CPU}
	t2 := &Tensor{DType: Float32, Device: CPU}
	t3 := &Tensor{DType: Int32, Device: CPU}
	t4 := &Tensor{DType: Float32, Device: GPU}

	t.Run("Compatible tensors", func(t *testing.T) {
		err := checkCompatibility(t1, t2)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("Different dtypes", func(t *testing.T) {
		err := checkCompatibility(t1, t3)
		if err == nil {
			t.Error("Expected error for different dtypes")
		}
	})

	t.Run("Different devices", func(t *testing.T) {
		err := checkCompatibility(t1, t4)
		if err == nil {
			t.Error("Expected error for different devices")
		}
	})
}

func TestCheckShapesCompatible(t *testing.T) {
	t.Run("Compatible shapes", func(t *testing.T) {
		shape1 := []int{2, 3}
		shape2 := []int{2, 3}
		
		result, err := checkShapesCompatible(shape1, shape2)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		
		if !reflect.DeepEqual(result, shape1) {
			t.Errorf("Result = %v, expected %v", result, shape1)
		}
	})

	t.Run("Different shapes", func(t *testing.T) {
		shape1 := []int{2, 3}
		shape2 := []int{3, 2}
		
		_, err := checkShapesCompatible(shape1, shape2)
		if err == nil {
			t.Error("Expected error for different shapes")
		}
	})

	t.Run("Different dimensions", func(t *testing.T) {
		shape1 := []int{2, 3}
		shape2 := []int{2, 3, 4}
		
		_, err := checkShapesCompatible(shape1, shape2)
		if err == nil {
			t.Error("Expected error for different number of dimensions")
		}
	})

	t.Run("Empty shapes", func(t *testing.T) {
		shape1 := []int{}
		shape2 := []int{2, 3}
		
		_, err := checkShapesCompatible(shape1, shape2)
		if err == nil {
			t.Error("Expected error for empty shapes")
		}
	})
}

func TestAdd(t *testing.T) {
	t.Run("Float32 addition", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5.0, 6.0, 7.0, 8.0})
		
		result, err := Add(a, b)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}
		
		expected := []float32{6.0, 8.0, 10.0, 12.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Int32 addition", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{1, 2, 3, 4})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})
		
		result, err := Add(a, b)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}
		
		expected := []int32{6, 8, 10, 12}
		resultData := result.Data.([]int32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Incompatible tensors", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})
		b, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{5, 6, 7, 8})
		
		_, err := Add(a, b)
		if err == nil {
			t.Error("Expected error for incompatible tensors")
		}
	})
}

func TestSub(t *testing.T) {
	t.Run("Float32 subtraction", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{5.0, 6.0, 7.0, 8.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})
		
		result, err := Sub(a, b)
		if err != nil {
			t.Fatalf("Sub failed: %v", err)
		}
		
		expected := []float32{4.0, 4.0, 4.0, 4.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})
}

func TestMul(t *testing.T) {
	t.Run("Float32 multiplication", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{2.0, 3.0, 4.0, 5.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{2.0, 2.0, 2.0, 2.0})
		
		result, err := Mul(a, b)
		if err != nil {
			t.Fatalf("Mul failed: %v", err)
		}
		
		expected := []float32{4.0, 6.0, 8.0, 10.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})
}

func TestDiv(t *testing.T) {
	t.Run("Float32 division", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{8.0, 6.0, 4.0, 2.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{2.0, 2.0, 2.0, 2.0})
		
		result, err := Div(a, b)
		if err != nil {
			t.Fatalf("Div failed: %v", err)
		}
		
		expected := []float32{4.0, 3.0, 2.0, 1.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Division by zero", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 2.0, 3.0, 4.0})
		b, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 0.0, 3.0, 4.0})
		
		_, err := Div(a, b)
		if err == nil {
			t.Error("Expected error for division by zero")
		}
	})
}

func TestReLU(t *testing.T) {
	t.Run("Float32 ReLU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{-1.0, 2.0, -3.0, 4.0})
		
		result, err := ReLU(a)
		if err != nil {
			t.Fatalf("ReLU failed: %v", err)
		}
		
		expected := []float32{0.0, 2.0, 0.0, 4.0}
		resultData := result.Data.([]float32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})

	t.Run("Int32 ReLU", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{-1, 2, -3, 4})
		
		result, err := ReLU(a)
		if err != nil {
			t.Fatalf("ReLU failed: %v", err)
		}
		
		expected := []int32{0, 2, 0, 4}
		resultData := result.Data.([]int32)
		if !reflect.DeepEqual(resultData, expected) {
			t.Errorf("Result = %v, expected %v", resultData, expected)
		}
	})
}

func TestSigmoid(t *testing.T) {
	t.Run("Float32 Sigmoid", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{0.0, 1.0, -1.0, 2.0})
		
		result, err := Sigmoid(a)
		if err != nil {
			t.Fatalf("Sigmoid failed: %v", err)
		}
		
		resultData := result.Data.([]float32)
		
		// Check sigmoid(0) = 0.5
		if math.Abs(float64(resultData[0])-0.5) > 1e-6 {
			t.Errorf("sigmoid(0) = %f, expected 0.5", resultData[0])
		}
		
		// Check sigmoid(1) ≈ 0.7311
		expected1 := float32(1.0 / (1.0 + math.Exp(-1.0)))
		if math.Abs(float64(resultData[1])-float64(expected1)) > 1e-4 {
			t.Errorf("sigmoid(1) = %f, expected %f", resultData[1], expected1)
		}
		
		// Check that all values are in (0, 1)
		for i, val := range resultData {
			if val <= 0 || val >= 1 {
				t.Errorf("sigmoid result[%d] = %f, expected in (0, 1)", i, val)
			}
		}
	})

	t.Run("Non-Float32 dtype", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{0, 1, -1, 2})
		
		_, err := Sigmoid(a)
		if err == nil {
			t.Error("Expected error for non-Float32 dtype")
		}
	})
}

func TestTanh(t *testing.T) {
	t.Run("Float32 Tanh", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{0.0, 1.0, -1.0, 2.0})
		
		result, err := Tanh(a)
		if err != nil {
			t.Fatalf("Tanh failed: %v", err)
		}
		
		resultData := result.Data.([]float32)
		
		// Check tanh(0) = 0
		if math.Abs(float64(resultData[0])) > 1e-6 {
			t.Errorf("tanh(0) = %f, expected 0", resultData[0])
		}
		
		// Check that all values are in (-1, 1)
		for i, val := range resultData {
			if val <= -1 || val >= 1 {
				t.Errorf("tanh result[%d] = %f, expected in (-1, 1)", i, val)
			}
		}
	})

	t.Run("Non-Float32 dtype", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Int32, CPU, []int32{0, 1, -1, 2})
		
		_, err := Tanh(a)
		if err == nil {
			t.Error("Expected error for non-Float32 dtype")
		}
	})
}

func TestExp(t *testing.T) {
	t.Run("Float32 Exp", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{0.0, 1.0, 2.0, -1.0})
		
		result, err := Exp(a)
		if err != nil {
			t.Fatalf("Exp failed: %v", err)
		}
		
		resultData := result.Data.([]float32)
		
		// Check exp(0) = 1
		if math.Abs(float64(resultData[0])-1.0) > 1e-6 {
			t.Errorf("exp(0) = %f, expected 1.0", resultData[0])
		}
		
		// Check exp(1) ≈ e
		if math.Abs(float64(resultData[1])-math.E) > 1e-4 {
			t.Errorf("exp(1) = %f, expected %f", resultData[1], math.E)
		}
		
		// Check that all values are positive
		for i, val := range resultData {
			if val <= 0 {
				t.Errorf("exp result[%d] = %f, expected positive", i, val)
			}
		}
	})
}

func TestLog(t *testing.T) {
	t.Run("Float32 Log", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, math.E, math.E * math.E, 0.5})
		
		result, err := Log(a)
		if err != nil {
			t.Fatalf("Log failed: %v", err)
		}
		
		resultData := result.Data.([]float32)
		
		// Check log(1) = 0
		if math.Abs(float64(resultData[0])) > 1e-6 {
			t.Errorf("log(1) = %f, expected 0", resultData[0])
		}
		
		// Check log(e) = 1
		if math.Abs(float64(resultData[1])-1.0) > 1e-4 {
			t.Errorf("log(e) = %f, expected 1.0", resultData[1])
		}
		
		// Check log(e²) = 2
		if math.Abs(float64(resultData[2])-2.0) > 1e-4 {
			t.Errorf("log(e²) = %f, expected 2.0", resultData[2])
		}
	})

	t.Run("Log of zero", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, 0.0, 3.0, 4.0})
		
		_, err := Log(a)
		if err == nil {
			t.Error("Expected error for log of zero")
		}
	})

	t.Run("Log of negative", func(t *testing.T) {
		a, _ := NewTensor([]int{2, 2}, Float32, CPU, []float32{1.0, -1.0, 3.0, 4.0})
		
		_, err := Log(a)
		if err == nil {
			t.Error("Expected error for log of negative number")
		}
	})
}