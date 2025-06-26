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
		// Fused operation kernels for Phase 6.3
		"linear_forward_float32",
		"linear_relu_float32",
		"linear_sigmoid_float32",
		"batch_matmul_float32",
		"adam_update_float32",
		"sgd_momentum_update_float32",
		"layer_norm_float32",
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

// GetDevice returns the Metal device
func (e *ComputeEngine) GetDevice() *Device {
	return e.device
}

// GetCommandQueue returns the command queue
func (e *ComputeEngine) GetCommandQueue() *CommandQueue {
	return e.queue
}

// AddArraysFloat32Async performs element-wise addition of two float32 arrays on GPU asynchronously
func (e *ComputeEngine) AddArraysFloat32Async(inputA, inputB []float32, completion func([]float32, error)) error {
	if len(inputA) != len(inputB) {
		completion(nil, fmt.Errorf("input arrays must have same length"))
		return fmt.Errorf("input arrays must have same length")
	}

	length := len(inputA)
	result := make([]float32, length)

	// Create buffers
	bufferA, err := e.device.CreateBufferWithBytes(inputA, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create buffer A: %v", err))
		return err
	}

	bufferB, err := e.device.CreateBufferWithBytes(inputB, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create buffer B: %v", err))
		return err
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create result buffer: %v", err))
		return err
	}

	// Execute kernel asynchronously
	err = e.executeKernelAsync("add_arrays_float32", uint(length), func() {
		// Copy result back when GPU computation completes
		resultSlice := bufferResult.ContentsAsFloat32()
		copy(result, resultSlice)
		completion(result, nil)
	}, bufferA, bufferB, bufferResult)
	
	if err != nil {
		completion(nil, fmt.Errorf("failed to execute kernel: %v", err))
		return err
	}

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

// ReLUFloat32Async applies ReLU activation to float32 array on GPU asynchronously
func (e *ComputeEngine) ReLUFloat32Async(input []float32, completion func([]float32, error)) error {
	length := len(input)
	result := make([]float32, length)

	// Create buffers
	bufferInput, err := e.device.CreateBufferWithBytes(input, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create input buffer: %v", err))
		return err
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create result buffer: %v", err))
		return err
	}

	// Execute kernel asynchronously
	err = e.executeKernelAsync("relu_float32", uint(length), func() {
		// Copy result back when GPU computation completes
		resultSlice := bufferResult.ContentsAsFloat32()
		copy(result, resultSlice)
		completion(result, nil)
	}, bufferInput, bufferResult)
	
	if err != nil {
		completion(nil, fmt.Errorf("failed to execute kernel: %v", err))
		return err
	}

	return nil
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

// MatMulFloat32Async performs matrix multiplication on GPU asynchronously  
func (e *ComputeEngine) MatMulFloat32Async(matrixA, matrixB []float32, M, N, P uint, completion func([]float32, error)) error {
	if len(matrixA) != int(M*N) {
		completion(nil, fmt.Errorf("matrix A size mismatch: expected %d, got %d", M*N, len(matrixA)))
		return fmt.Errorf("matrix A size mismatch: expected %d, got %d", M*N, len(matrixA))
	}
	if len(matrixB) != int(N*P) {
		completion(nil, fmt.Errorf("matrix B size mismatch: expected %d, got %d", N*P, len(matrixB)))
		return fmt.Errorf("matrix B size mismatch: expected %d, got %d", N*P, len(matrixB))
	}

	result := make([]float32, M*P)

	// Create buffers
	bufferA, err := e.device.CreateBufferWithBytes(matrixA, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create A buffer: %v", err))
		return err
	}

	bufferB, err := e.device.CreateBufferWithBytes(matrixB, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create B buffer: %v", err))
		return err
	}

	bufferResult, err := e.device.CreateBufferWithLength(uintptr(len(result)*4), ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create result buffer: %v", err))
		return err
	}

	// Create dimension buffers
	mBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(M)}, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create M buffer: %v", err))
		return err
	}

	nBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(N)}, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create N buffer: %v", err))
		return err
	}

	pBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(P)}, ResourceStorageModeShared)
	if err != nil {
		completion(nil, fmt.Errorf("failed to create P buffer: %v", err))
		return err
	}

	// Execute matrix multiplication kernel asynchronously
	err = e.executeMatMulKernelAsync("matmul_float32", M, P, func() {
		// Copy result back when GPU computation completes
		resultSlice := bufferResult.ContentsAsFloat32()
		copy(result, resultSlice)
		completion(result, nil)
	}, bufferA, bufferB, bufferResult, mBuffer, nBuffer, pBuffer)
	
	if err != nil {
		completion(nil, fmt.Errorf("failed to execute matmul kernel: %v", err))
		return err
	}

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

// executeMatMulKernelAsync executes matrix multiplication with 2D grid asynchronously
func (e *ComputeEngine) executeMatMulKernelAsync(kernelName string, M, P uint, completion func(), buffers ...*Buffer) error {
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

// ===== FUSED OPERATION METHODS FOR PHASE 6.3 =====

// LinearForwardFloat32 performs fused MatMul + Bias addition in one GPU call
func (e *ComputeEngine) LinearForwardFloat32(input, weight, bias []float32, batchSize, inputFeatures, outputFeatures uint) ([]float32, error) {
	if len(input) != int(batchSize*inputFeatures) {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", batchSize*inputFeatures, len(input))
	}
	if len(weight) != int(inputFeatures*outputFeatures) {
		return nil, fmt.Errorf("weight size mismatch: expected %d, got %d", inputFeatures*outputFeatures, len(weight))
	}
	if len(bias) != int(outputFeatures) {
		return nil, fmt.Errorf("bias size mismatch: expected %d, got %d", outputFeatures, len(bias))
	}

	result := make([]float32, batchSize*outputFeatures)

	// Create buffers
	inputBuffer, err := e.device.CreateBufferWithBytes(input, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create input buffer: %v", err)
	}

	weightBuffer, err := e.device.CreateBufferWithBytes(weight, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight buffer: %v", err)
	}

	biasBuffer, err := e.device.CreateBufferWithBytes(bias, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias buffer: %v", err)
	}

	outputBuffer, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Create parameter buffers
	batchSizeBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(batchSize)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	inputFeaturesBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(inputFeatures)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	outputFeaturesBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(outputFeatures)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	// Execute fused linear kernel
	err = e.executeFusedLinearKernel("linear_forward_float32", batchSize, outputFeatures,
		inputBuffer, weightBuffer, biasBuffer, outputBuffer,
		batchSizeBuffer, inputFeaturesBuffer, outputFeaturesBuffer)
	if err != nil {
		return nil, fmt.Errorf("failed to execute linear forward kernel: %v", err)
	}

	// Copy result back
	resultSlice := outputBuffer.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// LinearReLUFloat32 performs fused MatMul + Bias + ReLU activation in one GPU call
func (e *ComputeEngine) LinearReLUFloat32(input, weight, bias []float32, batchSize, inputFeatures, outputFeatures uint) ([]float32, error) {
	if len(input) != int(batchSize*inputFeatures) {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", batchSize*inputFeatures, len(input))
	}
	if len(weight) != int(inputFeatures*outputFeatures) {
		return nil, fmt.Errorf("weight size mismatch: expected %d, got %d", inputFeatures*outputFeatures, len(weight))
	}
	if len(bias) != int(outputFeatures) {
		return nil, fmt.Errorf("bias size mismatch: expected %d, got %d", outputFeatures, len(bias))
	}

	result := make([]float32, batchSize*outputFeatures)

	// Create buffers
	inputBuffer, err := e.device.CreateBufferWithBytes(input, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create input buffer: %v", err)
	}

	weightBuffer, err := e.device.CreateBufferWithBytes(weight, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight buffer: %v", err)
	}

	biasBuffer, err := e.device.CreateBufferWithBytes(bias, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias buffer: %v", err)
	}

	outputBuffer, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Create parameter buffers
	batchSizeBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(batchSize)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	inputFeaturesBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(inputFeatures)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	outputFeaturesBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(outputFeatures)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	// Execute fused linear + ReLU kernel
	err = e.executeFusedLinearKernel("linear_relu_float32", batchSize, outputFeatures,
		inputBuffer, weightBuffer, biasBuffer, outputBuffer,
		batchSizeBuffer, inputFeaturesBuffer, outputFeaturesBuffer)
	if err != nil {
		return nil, fmt.Errorf("failed to execute linear ReLU kernel: %v", err)
	}

	// Copy result back
	resultSlice := outputBuffer.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// LinearSigmoidFloat32 performs fused MatMul + Bias + Sigmoid activation in one GPU call
func (e *ComputeEngine) LinearSigmoidFloat32(input, weight, bias []float32, batchSize, inputFeatures, outputFeatures uint) ([]float32, error) {
	if len(input) != int(batchSize*inputFeatures) {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", batchSize*inputFeatures, len(input))
	}
	if len(weight) != int(inputFeatures*outputFeatures) {
		return nil, fmt.Errorf("weight size mismatch: expected %d, got %d", inputFeatures*outputFeatures, len(weight))
	}
	if len(bias) != int(outputFeatures) {
		return nil, fmt.Errorf("bias size mismatch: expected %d, got %d", outputFeatures, len(bias))
	}

	result := make([]float32, batchSize*outputFeatures)

	// Create buffers
	inputBuffer, err := e.device.CreateBufferWithBytes(input, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create input buffer: %v", err)
	}

	weightBuffer, err := e.device.CreateBufferWithBytes(weight, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight buffer: %v", err)
	}

	biasBuffer, err := e.device.CreateBufferWithBytes(bias, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias buffer: %v", err)
	}

	outputBuffer, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Create parameter buffers
	batchSizeBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(batchSize)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	inputFeaturesBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(inputFeatures)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	outputFeaturesBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(outputFeatures)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	// Execute fused linear + Sigmoid kernel
	err = e.executeFusedLinearKernel("linear_sigmoid_float32", batchSize, outputFeatures,
		inputBuffer, weightBuffer, biasBuffer, outputBuffer,
		batchSizeBuffer, inputFeaturesBuffer, outputFeaturesBuffer)
	if err != nil {
		return nil, fmt.Errorf("failed to execute linear sigmoid kernel: %v", err)
	}

	// Copy result back
	resultSlice := outputBuffer.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// BatchMatMulFloat32 performs batch matrix multiplication in one GPU call
func (e *ComputeEngine) BatchMatMulFloat32(batchA, batchB []float32, batchSize, M, N, P uint) ([]float32, error) {
	expectedSizeA := int(batchSize * M * N)
	expectedSizeB := int(batchSize * N * P)
	
	if len(batchA) != expectedSizeA {
		return nil, fmt.Errorf("batch A size mismatch: expected %d, got %d", expectedSizeA, len(batchA))
	}
	if len(batchB) != expectedSizeB {
		return nil, fmt.Errorf("batch B size mismatch: expected %d, got %d", expectedSizeB, len(batchB))
	}

	result := make([]float32, batchSize*M*P)

	// Create buffers
	bufferA, err := e.device.CreateBufferWithBytes(batchA, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch A buffer: %v", err)
	}

	bufferB, err := e.device.CreateBufferWithBytes(batchB, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch B buffer: %v", err)
	}

	bufferResult, err := e.device.CreateBufferWithBytes(result, ResourceStorageModeShared)
	if err != nil {
		return nil, fmt.Errorf("failed to create result buffer: %v", err)
	}

	// Create parameter buffers
	batchSizeBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(batchSize)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	mBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(M)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	nBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(N)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	pBuffer, err := e.device.CreateBufferWithBytes([]uint32{uint32(P)}, ResourceStorageModeShared)
	if err != nil {
		return nil, err
	}

	// Execute batch matrix multiplication kernel
	err = e.executeBatchMatMulKernel("batch_matmul_float32", batchSize, M, P,
		bufferA, bufferB, bufferResult, batchSizeBuffer, mBuffer, nBuffer, pBuffer)
	if err != nil {
		return nil, fmt.Errorf("failed to execute batch matmul kernel: %v", err)
	}

	// Copy result back
	resultSlice := bufferResult.ContentsAsFloat32()
	copy(result, resultSlice)

	return result, nil
}

// Helper functions for kernel execution

// executeFusedLinearKernel executes fused linear operations with 2D grid
func (e *ComputeEngine) executeFusedLinearKernel(kernelName string, batchSize, outputFeatures uint, buffers ...*Buffer) error {
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

	// Calculate thread group sizes for 2D dispatch (batch_size x output_features)
	threadgroupSize := MTLSize{Width: 16, Height: 16, Depth: 1}
	gridSize := MTLSize{
		Width:  (outputFeatures + threadgroupSize.Width - 1) / threadgroupSize.Width * threadgroupSize.Width,
		Height: (batchSize + threadgroupSize.Height - 1) / threadgroupSize.Height * threadgroupSize.Height,
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

// executeBatchMatMulKernel executes batch matrix multiplication with 3D grid
func (e *ComputeEngine) executeBatchMatMulKernel(kernelName string, batchSize, M, P uint, buffers ...*Buffer) error {
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

	// Calculate thread group sizes for 3D dispatch (P x M x batchSize)
	threadgroupSize := MTLSize{Width: 8, Height: 8, Depth: 1}
	gridSize := MTLSize{
		Width:  (P + threadgroupSize.Width - 1) / threadgroupSize.Width * threadgroupSize.Width,
		Height: (M + threadgroupSize.Height - 1) / threadgroupSize.Height * threadgroupSize.Height,
		Depth:  batchSize,
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