package metal_bridge

import (
	"fmt"
)

// ComputeEngine manages Metal compute operations
type ComputeEngine struct {
	device   *Device
	queue    *CommandQueue
	library  *Library
	kernels  map[string]*Function
	pipelines map[string]*ComputePipelineState
}

// NewComputeEngine creates a new Metal compute engine
func NewComputeEngine() (*ComputeEngine, error) {
	device := CreateSystemDefaultDevice()
	if device == nil {
		return nil, fmt.Errorf("failed to create Metal device")
	}

	queue := device.NewCommandQueue()
	if queue == nil {
		return nil, fmt.Errorf("failed to create command queue")
	}

	library, err := device.CreateLibraryWithSource(MetalKernelSource)
	if err != nil {
		return nil, fmt.Errorf("failed to create Metal library: %v", err)
	}

	engine := &ComputeEngine{
		device:    device,
		queue:     queue,
		library:   library,
		kernels:   make(map[string]*Function),
		pipelines: make(map[string]*ComputePipelineState),
	}

	// Load commonly used kernels
	kernelNames := []string{
		"add_arrays_float32",
		"add_arrays_int32",
		"mul_arrays_float32",
		"mul_arrays_int32",
		"relu_float32",
		"relu_int32",
		"sigmoid_float32",
		"matmul_float32",
	}

	for _, name := range kernelNames {
		if err := engine.LoadKernel(name); err != nil {
			return nil, fmt.Errorf("failed to load kernel %s: %v", name, err)
		}
	}

	return engine, nil
}

// LoadKernel loads a specific kernel function and creates its pipeline state
func (e *ComputeEngine) LoadKernel(kernelName string) error {
	function, err := e.library.GetFunction(kernelName)
	if err != nil {
		return fmt.Errorf("failed to get function %s: %v", kernelName, err)
	}

	pipeline, err := e.device.NewComputePipelineStateWithFunction(function)
	if err != nil {
		return fmt.Errorf("failed to create pipeline state for %s: %v", kernelName, err)
	}

	e.kernels[kernelName] = function
	e.pipelines[kernelName] = pipeline
	return nil
}

// AddArraysFloat32 performs element-wise addition of two float32 arrays on GPU
func (e *ComputeEngine) AddArraysFloat32(inputA, inputB []float32) ([]float32, error) {
	if len(inputA) != len(inputB) {
		return nil, fmt.Errorf("input arrays must have same length")
	}

	length := len(inputA)
	result := make([]float32, length)

	// Create buffers
	bufferA, err := e.device.CreateBufferWithBytes(inputA, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer A: %v", err)
	}

	bufferB, err := e.device.CreateBufferWithBytes(inputB, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer B: %v", err)
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	// Execute kernel
	err = e.executeKernel("add_arrays_float32", uint(length), bufferA, bufferB, bufferResult)
	if err != nil {
		return nil, fmt.Errorf("failed to execute kernel: %v", err)
	}

	// Copy result back
	resultSlice := bufferResult.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// AddArraysInt32 performs element-wise addition of two int32 arrays on GPU
func (e *ComputeEngine) AddArraysInt32(inputA, inputB []int32) ([]int32, error) {
	if len(inputA) != len(inputB) {
		return nil, fmt.Errorf("input arrays must have same length")
	}

	length := len(inputA)
	result := make([]int32, length)

	// Create buffers
	bufferA, err := e.device.CreateBufferWithBytes(inputA, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer A: %v", err)
	}

	bufferB, err := e.device.CreateBufferWithBytes(inputB, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer B: %v", err)
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	// Execute kernel
	err = e.executeKernel("add_arrays_int32", uint(length), bufferA, bufferB, bufferResult)
	if err != nil {
		return nil, fmt.Errorf("failed to execute kernel: %v", err)
	}

	// Copy result back
	resultSlice := bufferResult.ContentsAsInt32()
	copy(result, resultSlice)

	return result, nil
}

// ReLUFloat32 applies ReLU activation to float32 array on GPU
func (e *ComputeEngine) ReLUFloat32(input []float32) ([]float32, error) {
	length := len(input)
	result := make([]float32, length)

	// Create buffers
	bufferInput, err := e.device.CreateBufferWithBytes(input, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create input buffer: %v", err)
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	// Execute kernel
	err = e.executeKernel("relu_float32", uint(length), bufferInput, bufferResult)
	if err != nil {
		return nil, fmt.Errorf("failed to execute kernel: %v", err)
	}

	// Copy result back
	resultSlice := bufferResult.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// executeKernel executes a Metal compute kernel with the given buffers
func (e *ComputeEngine) executeKernel(kernelName string, threadCount uint, buffers ...*Buffer) error {
	pipeline, exists := e.pipelines[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not loaded", kernelName)
	}

	// Create command buffer
	commandBuffer := e.queue.CommandBuffer()
	if commandBuffer == nil {
		return fmt.Errorf("failed to create command buffer")
	}

	// Create compute encoder
	encoder := commandBuffer.ComputeCommandEncoder()
	if encoder == nil {
		return fmt.Errorf("failed to create compute encoder")
	}

	// Set pipeline state
	encoder.SetComputePipelineState(pipeline)

	// Set buffers
	for i, buffer := range buffers {
		encoder.SetBuffer(buffer, 0, uint(i))
	}

	// Calculate thread group sizes
	threadgroupSize := MTLSize{Width: 64, Height: 1, Depth: 1} // Common threadgroup size
	gridSize := MTLSize{
		Width:  (threadCount + threadgroupSize.Width - 1) / threadgroupSize.Width * threadgroupSize.Width,
		Height: 1,
		Depth:  1,
	}

	// Dispatch threads
	encoder.DispatchThreads(gridSize, threadgroupSize)

	// End encoding
	encoder.EndEncoding()

	// Commit and wait for completion (synchronous for now)
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()

	return nil
}

// executeKernelAsync executes a Metal compute kernel asynchronously
func (e *ComputeEngine) executeKernelAsync(kernelName string, threadCount uint, completion func(), buffers ...*Buffer) error {
	pipeline, exists := e.pipelines[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not loaded", kernelName)
	}

	// Create command buffer
	commandBuffer := e.queue.CommandBuffer()
	if commandBuffer == nil {
		return fmt.Errorf("failed to create command buffer")
	}

	// Create compute encoder
	encoder := commandBuffer.ComputeCommandEncoder()
	if encoder == nil {
		return fmt.Errorf("failed to create compute encoder")
	}

	// Set pipeline state
	encoder.SetComputePipelineState(pipeline)

	// Set buffers
	for i, buffer := range buffers {
		encoder.SetBuffer(buffer, 0, uint(i))
	}

	// Calculate thread group sizes
	threadgroupSize := MTLSize{Width: 64, Height: 1, Depth: 1}
	gridSize := MTLSize{
		Width:  (threadCount + threadgroupSize.Width - 1) / threadgroupSize.Width * threadgroupSize.Width,
		Height: 1,
		Depth:  1,
	}

	// Dispatch threads
	encoder.DispatchThreads(gridSize, threadgroupSize)

	// End encoding
	encoder.EndEncoding()

	// Add completion handler if provided
	if completion != nil {
		commandBuffer.AddCompletedHandler(func(status int) {
			completion()
		})
	}

	// Commit (asynchronous)
	commandBuffer.Commit()

	return nil
}

// MatMulFloat32 performs matrix multiplication on GPU
func (e *ComputeEngine) MatMulFloat32(matrixA, matrixB []float32, M, N, P uint) ([]float32, error) {
	if len(matrixA) != int(M*N) {
		return nil, fmt.Errorf("matrix A size mismatch: expected %d, got %d", M*N, len(matrixA))
	}
	if len(matrixB) != int(N*P) {
		return nil, fmt.Errorf("matrix B size mismatch: expected %d, got %d", N*P, len(matrixB))
	}

	result := make([]float32, M*P)

	// Create buffers
	bufferA, err := e.device.CreateBufferWithBytes(matrixA, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer A: %v", err)
	}

	bufferB, err := e.device.CreateBufferWithBytes(matrixB, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer B: %v", err)
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	// Create parameter buffers
	mBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(M)}, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create M buffer: %v", err)
	}

	nBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(N)}, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create N buffer: %v", err)
	}

	pBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(P)}, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create P buffer: %v", err)
	}

	// Execute matrix multiplication kernel
	err = e.executeMatMulKernel("matmul_float32", M, P, bufferA, bufferB, bufferResult, mBuffer, nBuffer, pBuffer)
	if err != nil {
		return nil, fmt.Errorf("failed to execute matmul kernel: %v", err)
	}

	// Copy result back
	resultSlice := bufferResult.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// executeMatMulKernel executes matrix multiplication with 2D grid
func (e *ComputeEngine) executeMatMulKernel(kernelName string, M, P uint, buffers ...*Buffer) error {
	pipeline, exists := e.pipelines[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not loaded", kernelName)
	}

	// Create command buffer
	commandBuffer := e.queue.CommandBuffer()
	if commandBuffer == nil {
		return fmt.Errorf("failed to create command buffer")
	}

	// Create compute encoder
	encoder := commandBuffer.ComputeCommandEncoder()
	if encoder == nil {
		return fmt.Errorf("failed to create compute encoder")
	}

	// Set pipeline state
	encoder.SetComputePipelineState(pipeline)

	// Set buffers
	for i, buffer := range buffers {
		encoder.SetBuffer(buffer, 0, uint(i))
	}

	// Calculate thread group sizes for 2D dispatch
	threadgroupSize := MTLSize{Width: 8, Height: 8, Depth: 1}
	gridSize := MTLSize{
		Width:  (P + threadgroupSize.Width - 1) / threadgroupSize.Width * threadgroupSize.Width,
		Height: (M + threadgroupSize.Height - 1) / threadgroupSize.Height * threadgroupSize.Height,
		Depth:  1,
	}

	// Dispatch threads
	encoder.DispatchThreads(gridSize, threadgroupSize)

	// End encoding
	encoder.EndEncoding()

	// Commit and wait for completion
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()

	return nil
}