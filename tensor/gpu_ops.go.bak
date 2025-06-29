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

// AddGPUAsync performs tensor addition on GPU asynchronously
func AddGPUAsync(t1, t2 *Tensor, completion func(*Tensor, error)) error {
	if err := checkCompatibility(t1, t2); err != nil {
		completion(nil, err)
		return err
	}

	outputShape, err := checkShapesCompatible(t1.Shape, t2.Shape)
	if err != nil {
		completion(nil, err)
		return err
	}

	engine, err := getComputeEngine()
	if err != nil {
		completion(nil, err)
		return err
	}

	switch t1.DType {
	case Float32:
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]float32)
		data2 := t2.materializeView().([]float32)

		err := engine.AddArraysFloat32Async(data1, data2, func(result []float32, err error) {
			if err != nil {
				completion(nil, fmt.Errorf("GPU addition failed: %v", err))
				return
			}

			// Create result tensor
			resultTensor := &Tensor{
				Shape:    outputShape,
				Strides:  calculateStrides(outputShape),
				DType:    t1.DType,
				Device:   GPU,
				Data:     result,
				NumElems: calculateNumElements(outputShape),
			}

			completion(resultTensor, nil)
		})
		
		if err != nil {
			completion(nil, err)
			return err
		}

	case Int32:
		// Fall back to CPU for Int32 async operations for now
		result, err := Add(t1, t2)
		completion(result, err)
		return err

	default:
		err := fmt.Errorf("unsupported dtype for GPU addition: %s", t1.DType)
		completion(nil, err)
		return err
	}

	return nil
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
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]float32)
		data2 := t2.materializeView().([]float32)

		result, err := engine.AddArraysFloat32(data1, data2)
		if err != nil {
			return nil, fmt.Errorf("GPU addition failed: %v", err)
		}
		resultData = result

	case Int32:
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]int32)
		data2 := t2.materializeView().([]int32)

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
		// Materialize views if needed for contiguous data access
		data := t.materializeView().([]float32)

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
func MatMulGPUAsync(t1, t2 *Tensor, completion func(*Tensor, error)) error {
	if err := checkCompatibility(t1, t2); err != nil {
		completion(nil, err)
		return err
	}

	if len(t1.Shape) < 2 || len(t2.Shape) < 2 {
		err := fmt.Errorf("matmul requires tensors with at least 2 dimensions")
		completion(nil, err)
		return err
	}

	rows1 := t1.Shape[len(t1.Shape)-2]
	cols1 := t1.Shape[len(t1.Shape)-1]
	rows2 := t2.Shape[len(t2.Shape)-2]
	cols2 := t2.Shape[len(t2.Shape)-1]

	if cols1 != rows2 {
		err := fmt.Errorf("incompatible dimensions for matmul: (%d, %d) x (%d, %d)", rows1, cols1, rows2, cols2)
		completion(nil, err)
		return err
	}

	outputShape := make([]int, len(t1.Shape))
	copy(outputShape, t1.Shape)
	outputShape[len(outputShape)-1] = cols2

	engine, err := getComputeEngine()
	if err != nil {
		completion(nil, err)
		return err
	}

	switch t1.DType {
	case Float32:
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]float32)
		data2 := t2.materializeView().([]float32)

		err := engine.MatMulFloat32Async(data1, data2, uint(rows1), uint(cols1), uint(cols2), func(result []float32, err error) {
			if err != nil {
				completion(nil, fmt.Errorf("GPU matmul failed: %v", err))
				return
			}

			// Create result tensor
			resultTensor := &Tensor{
				Shape:    outputShape,
				Strides:  calculateStrides(outputShape),
				DType:    t1.DType,
				Device:   GPU,
				Data:     result,
				NumElems: calculateNumElements(outputShape),
			}

			completion(resultTensor, nil)
		})
		
		if err != nil {
			completion(nil, err)
			return err
		}

	case Int32:
		// Fall back to CPU for Int32 async operations
		result, err := MatMul(t1, t2)
		completion(result, err)
		return err

	default:
		err := fmt.Errorf("unsupported dtype for GPU matmul: %s", t1.DType)
		completion(nil, err)
		return err
	}

	return nil
}

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
		// Materialize views if needed for contiguous data access
		data1 := t1.materializeView().([]float32)
		data2 := t2.materializeView().([]float32)

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

// ToGPU moves a tensor to GPU device using the BufferAllocator
func (t *Tensor) ToGPU() (*Tensor, error) {
	if t.Device == GPU {
		// Already on GPU, just return self
		return t, nil
	}

	// Get the global allocator
	allocator := metal_bridge.GetGlobalAllocator()
	
	// Calculate buffer size needed
	elementSize := getSizeForDType(t.DType)
	bufferSize := uint64(t.NumElems * elementSize)
	
	// Allocate GPU buffer using the allocator
	// Use shared storage mode for CPU-GPU data sharing
	buffer, err := allocator.Allocate(bufferSize, 0) // 0 = MTLResourceStorageModeShared
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU buffer: %v", err)
	}

	// Create new GPU tensor
	gpuTensor := &Tensor{
		Shape:        make([]int, len(t.Shape)),
		Strides:      make([]int, len(t.Strides)),
		DType:        t.DType,
		Device:       GPU,
		Data:         t.Data, // Keep reference to original CPU data
		NumElems:     t.NumElems,
		requiresGrad: t.requiresGrad,
		refCount:     1,
	}
	
	copy(gpuTensor.Shape, t.Shape)
	copy(gpuTensor.Strides, t.Strides)
	
	// Set the GPU buffer and initialize reference counting
	gpuTensor.SetGPUBuffer(buffer)
	
	// Copy data from CPU to GPU buffer
	err = copyDataToGPUBuffer(t.Data, buffer, t.DType)
	if err != nil {
		buffer.Release() // Clean up on error
		return nil, fmt.Errorf("failed to copy data to GPU: %v", err)
	}

	return gpuTensor, nil
}

// copyDataToGPUBuffer copies CPU data to a GPU buffer
func copyDataToGPUBuffer(cpuData interface{}, buffer interface{}, dtype DType) error {
	// Type assert the buffer to metal_bridge.Buffer
	mtlBuffer, ok := buffer.(*metal_bridge.Buffer)
	if !ok {
		return fmt.Errorf("invalid buffer type for GPU copy")
	}
	
	// Check for nil data
	if cpuData == nil {
		return fmt.Errorf("CPU data is nil, cannot copy to GPU buffer")
	}
	
	// Get buffer contents pointer
	bufferPtr := mtlBuffer.Contents()
	
	switch dtype {
	case Float32:
		data, ok := cpuData.([]float32)
		if !ok {
			return fmt.Errorf("expected []float32 for Float32 dtype, got %T", cpuData)
		}
		// Copy data from Go slice to Metal buffer
		copy((*[1<<30]float32)(bufferPtr)[:len(data)], data)
		
	case Int32:
		data := cpuData.([]int32)
		copy((*[1<<30]int32)(bufferPtr)[:len(data)], data)
		
	default:
		return fmt.Errorf("unsupported data type for GPU copy: %v", dtype)
	}
	
	return nil
}

// ToCPU moves a tensor to CPU device (creates a copy with CPU device type)
func (t *Tensor) ToCPU() (*Tensor, error) {
	if t.Device == CPU {
		return t, nil
	}

	// Create CPU tensor with data copied from GPU buffer
	var cpuData interface{}
	
	if t.gpuBuffer != nil {
		// Copy data from GPU buffer to CPU
		var err error
		cpuData, err = copyDataFromGPUBuffer(t.gpuBuffer, t.DType, t.NumElems)
		if err != nil {
			return nil, fmt.Errorf("failed to copy data from GPU: %v", err)
		}
	} else {
		// Fallback to existing CPU data if no GPU buffer
		cpuData = t.Data
	}

	// Create new CPU tensor
	cpuTensor := &Tensor{
		Shape:        make([]int, len(t.Shape)),
		Strides:      make([]int, len(t.Strides)),
		DType:        t.DType,
		Device:       CPU,
		Data:         cpuData,
		NumElems:     t.NumElems,
		requiresGrad: t.requiresGrad,
	}
	
	copy(cpuTensor.Shape, t.Shape)
	copy(cpuTensor.Strides, t.Strides)

	return cpuTensor, nil
}

// copyDataFromGPUBuffer copies data from a GPU buffer to CPU
func copyDataFromGPUBuffer(buffer interface{}, dtype DType, numElems int) (interface{}, error) {
	// Type assert the buffer to metal_bridge.Buffer
	mtlBuffer, ok := buffer.(*metal_bridge.Buffer)
	if !ok {
		return nil, fmt.Errorf("invalid buffer type for GPU copy")
	}
	
	// Get buffer contents pointer
	bufferPtr := mtlBuffer.Contents()
	
	switch dtype {
	case Float32:
		// Create new CPU slice and copy data from GPU buffer
		cpuData := make([]float32, numElems)
		copy(cpuData, (*[1<<30]float32)(bufferPtr)[:numElems])
		return cpuData, nil
		
	case Int32:
		cpuData := make([]int32, numElems)
		copy(cpuData, (*[1<<30]int32)(bufferPtr)[:numElems])
		return cpuData, nil
		
	default:
		return nil, fmt.Errorf("unsupported data type for GPU copy: %v", dtype)
	}
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