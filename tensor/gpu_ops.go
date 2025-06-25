package tensor

import (
	"fmt"
	"sync"

	"github.com/tsawler/go-metal/metal_bridge"
)

var (
	globalComputeEngine *metal_bridge.ComputeEngine
	engineMutex         sync.Mutex
)

// getComputeEngine returns the global compute engine, creating it if necessary
func getComputeEngine() (*metal_bridge.ComputeEngine, error) {
	engineMutex.Lock()
	defer engineMutex.Unlock()

	if globalComputeEngine == nil {
		engine, err := metal_bridge.NewComputeEngine()
		if err != nil {
			return nil, fmt.Errorf("failed to create Metal compute engine: %v", err)
		}
		globalComputeEngine = engine
	}

	return globalComputeEngine, nil
}

// AddGPU performs tensor addition on GPU
func AddGPU(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	outputShape, err := checkShapesCompatible(t1.Shape, t2.Shape)
	if err != nil {
		return nil, err
	}

	engine, err := getComputeEngine()
	if err != nil {
		return nil, err
	}

	var resultData interface{}

	switch t1.DType {
	case Float32:
		data1 := t1.Data.([]float32)
		data2 := t2.Data.([]float32)

		result, err := engine.AddArraysFloat32(data1, data2)
		if err != nil {
			return nil, fmt.Errorf("GPU addition failed: %v", err)
		}
		resultData = result

	case Int32:
		data1 := t1.Data.([]int32)
		data2 := t2.Data.([]int32)

		result, err := engine.AddArraysInt32(data1, data2)
		if err != nil {
			return nil, fmt.Errorf("GPU addition failed: %v", err)
		}
		resultData = result

	default:
		return nil, fmt.Errorf("unsupported dtype for GPU Add: %s", t1.DType)
	}

	resultTensor, err := NewTensor(outputShape, t1.DType, GPU, resultData)
	if err != nil {
		return nil, err
	}

	return resultTensor, nil
}

// ReLUGPU performs ReLU activation on GPU
func ReLUGPU(t *Tensor) (*Tensor, error) {
	engine, err := getComputeEngine()
	if err != nil {
		return nil, err
	}

	var resultData interface{}

	switch t.DType {
	case Float32:
		data := t.Data.([]float32)

		result, err := engine.ReLUFloat32(data)
		if err != nil {
			return nil, fmt.Errorf("GPU ReLU failed: %v", err)
		}
		resultData = result

	case Int32:
		// For now, fall back to CPU for Int32 ReLU
		return ReLU(t)

	default:
		return nil, fmt.Errorf("unsupported dtype for GPU ReLU: %s", t.DType)
	}

	resultTensor, err := NewTensor(t.Shape, t.DType, GPU, resultData)
	if err != nil {
		return nil, err
	}

	return resultTensor, nil
}

// MatMulGPU performs matrix multiplication on GPU
func MatMulGPU(t1, t2 *Tensor) (*Tensor, error) {
	if err := checkCompatibility(t1, t2); err != nil {
		return nil, err
	}

	if len(t1.Shape) < 2 || len(t2.Shape) < 2 {
		return nil, fmt.Errorf("matmul requires tensors with at least 2 dimensions")
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

	engine, err := getComputeEngine()
	if err != nil {
		return nil, err
	}

	var resultData interface{}

	switch t1.DType {
	case Float32:
		data1 := t1.Data.([]float32)
		data2 := t2.Data.([]float32)

		result, err := engine.MatMulFloat32(data1, data2, uint(rows1), uint(cols1), uint(cols2))
		if err != nil {
			return nil, fmt.Errorf("GPU matmul failed: %v", err)
		}
		resultData = result

	case Int32:
		// For now, fall back to CPU for Int32 matrix multiplication
		return MatMul(t1, t2)

	default:
		return nil, fmt.Errorf("unsupported dtype for GPU MatMul: %s", t1.DType)
	}

	resultTensor, err := NewTensor(outputShape, t1.DType, GPU, resultData)
	if err != nil {
		return nil, err
	}

	return resultTensor, nil
}

// ToGPU moves a tensor to GPU device (creates a copy with GPU device type)
func (t *Tensor) ToGPU() (*Tensor, error) {
	if t.Device == GPU {
		return t, nil
	}

	// For Phase 2, we just change the device type and keep the data on CPU
	// In future phases, this would actually transfer data to GPU memory
	gpuTensor, err := t.Clone()
	if err != nil {
		return nil, err
	}

	gpuTensor.Device = GPU
	return gpuTensor, nil
}

// ToCPU moves a tensor to CPU device (creates a copy with CPU device type)
func (t *Tensor) ToCPU() (*Tensor, error) {
	if t.Device == CPU {
		return t, nil
	}

	// For Phase 2, we just change the device type and keep the data on CPU
	// In future phases, this would actually transfer data from GPU memory
	cpuTensor, err := t.Clone()
	if err != nil {
		return nil, err
	}

	cpuTensor.Device = CPU
	return cpuTensor, nil
}

// IsGPUAvailable checks if Metal GPU compute is available
func IsGPUAvailable() bool {
	_, err := getComputeEngine()
	return err == nil
}

// GPUInfo returns information about the GPU device
func GPUInfo() (string, error) {
	engine, err := getComputeEngine()
	if err != nil {
		return "", err
	}

	// For now, just return that Metal is available
	_ = engine // Suppress unused variable warning
	return "Metal GPU compute available", nil
}