package training

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tsawler/go-metal/tensor"
)

// Global random source for deterministic initialization
var globalRng *rand.Rand = rand.New(rand.NewSource(1))

// SetRandomSeed sets the global random seed for deterministic weight initialization
func SetRandomSeed(seed int64) {
	globalRng = rand.New(rand.NewSource(seed))
}

// Module interface defines methods that all neural network layers must implement
type Module interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*tensor.Tensor // Returns trainable parameters (tensors with requiresGrad=true)
	Train()                       // Sets module to training mode
	Eval()                        // Sets module to evaluation mode
	IsTraining() bool             // Returns true if in training mode
}

// Linear implements a fully connected (dense) layer: y = xW^T + b
type Linear struct {
	weight   *tensor.Tensor
	bias     *tensor.Tensor
	training bool
}

// NewLinear creates a new Linear layer
func NewLinear(inputSize, outputSize int, bias bool, device tensor.DeviceType) (*Linear, error) {
	// Initialize weights using Xavier/Glorot uniform initialization
	// W ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
	bound := math.Sqrt(6.0 / float64(inputSize+outputSize))
	
	weightData := make([]float32, inputSize*outputSize)
	for i := range weightData {
		weightData[i] = float32((globalRng.Float64()*2.0 - 1.0) * bound)
	}
	
	// Create weight tensor with shape [inputSize, outputSize] to work with current MatMul
	weight, err := tensor.NewTensor([]int{inputSize, outputSize}, tensor.Float32, tensor.CPU, weightData)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
	}
	
	// Transfer to target device if needed
	if device == tensor.PersistentGPU {
		weight, err = weight.ToPersistentGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer weight to persistent GPU: %v", err)
		}
	} else if device == tensor.GPU {
		weight, err = weight.ToGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer weight to GPU: %v", err)
		}
	}
	
	weight.SetRequiresGrad(true)
	
	linear := &Linear{
		weight:   weight,
		training: true,
	}
	
	if bias {
		// Initialize bias to zeros
		biasData := make([]float32, outputSize)
		biasT, err := tensor.NewTensor([]int{outputSize}, tensor.Float32, tensor.CPU, biasData)
		if err != nil {
			return nil, fmt.Errorf("failed to create bias tensor: %v", err)
		}
		
		// Transfer bias to target device if needed
		if device == tensor.PersistentGPU {
			biasT, err = biasT.ToPersistentGPU()
			if err != nil {
				return nil, fmt.Errorf("failed to transfer bias to persistent GPU: %v", err)
			}
		} else if device == tensor.GPU {
			biasT, err = biasT.ToGPU()
			if err != nil {
				return nil, fmt.Errorf("failed to transfer bias to GPU: %v", err)
			}
		}
		
		biasT.SetRequiresGrad(true)
		linear.bias = biasT
	}
	
	return linear, nil
}

// Forward performs the forward pass: y = xW^T + b
func (l *Linear) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("Linear layer expects 2D input [batch_size, input_size], got shape %v", input.Shape)
	}
	
	inputSize := input.Shape[1]
	
	if inputSize != l.weight.Shape[0] {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", l.weight.Shape[0], inputSize)
	}
	
	// Compute input @ weight using autograd for gradient support
	// weight shape is [inputSize, outputSize], input shape is [batchSize, inputSize]
	output, err := tensor.MatMulAutograd(input, l.weight)
	
	// Add bias if present
	if l.bias != nil {
		// Try to add bias directly - AddAutograd should handle broadcasting
		output, err = tensor.AddAutograd(output, l.bias)
		if err != nil {
			return nil, fmt.Errorf("bias addition failed: %v", err)
		}
	}
	
	return output, nil
}

// expandBias expands bias from [output_size] to [batch_size, output_size]
func (l *Linear) expandBias(bias *tensor.Tensor, batchSize int) (*tensor.Tensor, error) {
	outputSize := bias.Shape[0]
	
	// Get bias data - convert to CPU if needed for expansion
	var biasData []float32
	if bias.Device == tensor.GPU || bias.Device == tensor.PersistentGPU {
		// Convert to CPU temporarily for bias expansion
		cpuBias, err := bias.ToCPU()
		if err != nil {
			return nil, fmt.Errorf("failed to convert bias to CPU: %v", err)
		}
		biasData = cpuBias.Data.([]float32)
	} else {
		biasData = bias.Data.([]float32)
	}
	
	expandedData := make([]float32, batchSize*outputSize)
	for i := 0; i < batchSize; i++ {
		copy(expandedData[i*outputSize:(i+1)*outputSize], biasData)
	}
	
	// Create expanded tensor and transfer to appropriate device
	expanded, err := tensor.NewTensor([]int{batchSize, outputSize}, bias.DType, tensor.CPU, expandedData)
	if err != nil {
		return nil, err
	}
	
	// Transfer to same device as bias
	if bias.Device == tensor.PersistentGPU {
		return expanded.ToPersistentGPU()
	} else if bias.Device == tensor.GPU {
		return expanded.ToGPU()
	}
	
	return expanded, nil
}

// Parameters returns the trainable parameters
func (l *Linear) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{l.weight}
	if l.bias != nil {
		params = append(params, l.bias)
	}
	return params
}

// Train sets the module to training mode
func (l *Linear) Train() {
	l.training = true
}

// Eval sets the module to evaluation mode
func (l *Linear) Eval() {
	l.training = false
}

// IsTraining returns true if in training mode
func (l *Linear) IsTraining() bool {
	return l.training
}

// ReLU implements ReLU activation function module
type ReLU struct {
	training bool
}

// NewReLU creates a new ReLU activation module
func NewReLU() *ReLU {
	return &ReLU{training: true}
}

// Forward performs ReLU activation
func (r *ReLU) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	// Use autograd ReLU to build computational graph
	return tensor.ReLUAutograd(input)
}

// Parameters returns empty slice (ReLU has no parameters)
func (r *ReLU) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// Train sets the module to training mode
func (r *ReLU) Train() {
	r.training = true
}

// Eval sets the module to evaluation mode
func (r *ReLU) Eval() {
	r.training = false
}

// IsTraining returns true if in training mode
func (r *ReLU) IsTraining() bool {
	return r.training
}

// Conv2D implements a 2D convolution layer
type Conv2D struct {
	weight     *tensor.Tensor
	bias       *tensor.Tensor
	stride     int
	padding    int
	training   bool
}

// NewConv2D creates a new Conv2D layer
func NewConv2D(inputChannels, outputChannels, kernelSize, stride, padding int, bias bool, device tensor.DeviceType) (*Conv2D, error) {
	// Initialize weights using Xavier/Glorot initialization for conv layers
	// fan_in = input_channels * kernel_size * kernel_size
	// fan_out = output_channels * kernel_size * kernel_size
	fanIn := float64(inputChannels * kernelSize * kernelSize)
	fanOut := float64(outputChannels * kernelSize * kernelSize)
	bound := math.Sqrt(6.0 / (fanIn + fanOut))
	
	weightData := make([]float32, outputChannels*inputChannels*kernelSize*kernelSize)
	for i := range weightData {
		weightData[i] = float32((globalRng.Float64()*2.0 - 1.0) * bound)
	}
	
	// Weight shape: [output_channels, input_channels, kernel_height, kernel_width]
	weight, err := tensor.NewTensor([]int{outputChannels, inputChannels, kernelSize, kernelSize}, tensor.Float32, tensor.CPU, weightData)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
	}
	
	// Transfer to target device if needed
	if device == tensor.PersistentGPU {
		weight, err = weight.ToPersistentGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer weight to persistent GPU: %v", err)
		}
	} else if device == tensor.GPU {
		weight, err = weight.ToGPU()
		if err != nil {
			return nil, fmt.Errorf("failed to transfer weight to GPU: %v", err)
		}
	}
	
	weight.SetRequiresGrad(true)
	
	conv := &Conv2D{
		weight:   weight,
		stride:   stride,
		padding:  padding,
		training: true,
	}
	
	if bias {
		// Initialize bias to zeros
		biasData := make([]float32, outputChannels)
		biasT, err := tensor.NewTensor([]int{outputChannels}, tensor.Float32, tensor.CPU, biasData)
		if err != nil {
			return nil, fmt.Errorf("failed to create bias tensor: %v", err)
		}
		
		// Transfer bias to target device if needed
		if device == tensor.PersistentGPU {
			biasT, err = biasT.ToPersistentGPU()
			if err != nil {
				return nil, fmt.Errorf("failed to transfer bias to persistent GPU: %v", err)
			}
		} else if device == tensor.GPU {
			biasT, err = biasT.ToGPU()
			if err != nil {
				return nil, fmt.Errorf("failed to transfer bias to GPU: %v", err)
			}
		}
		
		biasT.SetRequiresGrad(true)
		conv.bias = biasT
	}
	
	return conv, nil
}

// Forward performs 2D convolution
func (c *Conv2D) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("Conv2D expects 4D input [batch_size, channels, height, width], got shape %v", input.Shape)
	}
	
	// Use autograd Conv2D operation to build computational graph
	return tensor.Conv2DAutograd(input, c.weight, c.bias, c.stride, c.padding)
}

// Parameters returns the trainable parameters
func (c *Conv2D) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{c.weight}
	if c.bias != nil {
		params = append(params, c.bias)
	}
	return params
}

// Train sets the module to training mode
func (c *Conv2D) Train() {
	c.training = true
}

// Eval sets the module to evaluation mode
func (c *Conv2D) Eval() {
	c.training = false
}

// IsTraining returns true if in training mode
func (c *Conv2D) IsTraining() bool {
	return c.training
}

// BatchNorm implements Batch Normalization layer
type BatchNorm struct {
	numFeatures    int
	eps           float64
	momentum      float64
	gamma         *tensor.Tensor // Scale parameter
	beta          *tensor.Tensor // Shift parameter
	runningMean   *tensor.Tensor // Running mean for inference
	runningVar    *tensor.Tensor // Running variance for inference
	training      bool
}

// NewBatchNorm creates a new Batch Normalization layer
func NewBatchNorm(numFeatures int, eps, momentum float64, device tensor.DeviceType) (*BatchNorm, error) {
	if eps <= 0 {
		eps = 1e-5
	}
	if momentum <= 0 {
		momentum = 0.1
	}
	
	// Initialize gamma to ones
	gammaData := make([]float32, numFeatures)
	for i := range gammaData {
		gammaData[i] = 1.0
	}
	gamma, err := tensor.NewTensor([]int{numFeatures}, tensor.Float32, device, gammaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gamma tensor: %v", err)
	}
	gamma.SetRequiresGrad(true)
	
	// Initialize beta to zeros
	betaData := make([]float32, numFeatures)
	beta, err := tensor.NewTensor([]int{numFeatures}, tensor.Float32, device, betaData)
	if err != nil {
		return nil, fmt.Errorf("failed to create beta tensor: %v", err)
	}
	beta.SetRequiresGrad(true)
	
	// Initialize running statistics to zeros
	runningMeanData := make([]float32, numFeatures)
	runningMean, err := tensor.NewTensor([]int{numFeatures}, tensor.Float32, device, runningMeanData)
	if err != nil {
		return nil, fmt.Errorf("failed to create running mean tensor: %v", err)
	}
	
	runningVarData := make([]float32, numFeatures)
	for i := range runningVarData {
		runningVarData[i] = 1.0 // Initialize variance to 1
	}
	runningVar, err := tensor.NewTensor([]int{numFeatures}, tensor.Float32, device, runningVarData)
	if err != nil {
		return nil, fmt.Errorf("failed to create running variance tensor: %v", err)
	}
	
	return &BatchNorm{
		numFeatures: numFeatures,
		eps:         eps,
		momentum:    momentum,
		gamma:       gamma,
		beta:        beta,
		runningMean: runningMean,
		runningVar:  runningVar,
		training:    true,
	}, nil
}

// Forward performs batch normalization
func (bn *BatchNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if input.DType != tensor.Float32 {
		return nil, fmt.Errorf("BatchNorm only supports Float32 tensors")
	}
	
	// For now, implement a simplified version that works with 2D inputs [batch_size, features]
	// A full implementation would handle 4D inputs for conv layers as well
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("BatchNorm currently only supports 2D input [batch_size, features], got shape %v", input.Shape)
	}
	
	features := input.Shape[1]
	
	if features != bn.numFeatures {
		return nil, fmt.Errorf("input features mismatch: expected %d, got %d", bn.numFeatures, features)
	}
	
	// Use simplified batch norm that ensures gradient flow to gamma and beta
	return bn.simpleBatchNorm(input)
}

// simpleBatchNorm performs a simplified batch normalization that ensures gradient flow to gamma/beta
func (bn *BatchNorm) simpleBatchNorm(input *tensor.Tensor) (*tensor.Tensor, error) {
	// Compute batch statistics manually (for now)
	batchSize := input.Shape[0]
	features := input.Shape[1]
	inputData := input.Data.([]float32)
	
	// Calculate mean and variance manually
	meanData := make([]float32, features)
	varData := make([]float32, features)
	
	// Compute mean
	for j := 0; j < features; j++ {
		var sum float32
		for i := 0; i < batchSize; i++ {
			sum += inputData[i*features+j]
		}
		meanData[j] = sum / float32(batchSize)
	}
	
	// Compute variance
	for j := 0; j < features; j++ {
		var sumSq float32
		for i := 0; i < batchSize; i++ {
			diff := inputData[i*features+j] - meanData[j]
			sumSq += diff * diff
		}
		varData[j] = sumSq / float32(batchSize)
	}
	
	// Update running statistics
	if bn.training {
		momentum := float32(bn.momentum)
		runningMeanData := bn.runningMean.Data.([]float32)
		runningVarData := bn.runningVar.Data.([]float32)
		
		for i := range meanData {
			runningMeanData[i] = (1.0-momentum)*runningMeanData[i] + momentum*meanData[i]
			runningVarData[i] = (1.0-momentum)*runningVarData[i] + momentum*varData[i]
		}
	}
	
	// Normalize and apply gamma/beta using autograd operations to ensure gradient flow
	
	// Create normalized output manually first
	normalizedData := make([]float32, len(inputData))
	for i := 0; i < batchSize; i++ {
		for j := 0; j < features; j++ {
			idx := i*features + j
			// Normalize: (x - mean) / sqrt(var + eps)
			normalizedData[idx] = (inputData[idx] - meanData[j]) / 
				float32(math.Sqrt(float64(varData[j])+bn.eps))
		}
	}
	
	// Create normalized tensor
	normalized, err := tensor.NewTensor(input.Shape, input.DType, input.Device, normalizedData)
	if err != nil {
		return nil, fmt.Errorf("failed to create normalized tensor: %v", err)
	}
	
	// Apply affine transformation using autograd: gamma * normalized + beta
	scaled, err := tensor.MulAutograd(bn.gamma, normalized)
	if err != nil {
		return nil, fmt.Errorf("scaling failed: %v", err)
	}
	
	output, err := tensor.AddAutograd(scaled, bn.beta)
	if err != nil {
		return nil, fmt.Errorf("shift failed: %v", err)
	}
	
	return output, nil
}



// Parameters returns the trainable parameters
func (bn *BatchNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{bn.gamma, bn.beta}
}

// Train sets the module to training mode
func (bn *BatchNorm) Train() {
	bn.training = true
}

// Eval sets the module to evaluation mode
func (bn *BatchNorm) Eval() {
	bn.training = false
}

// IsTraining returns true if in training mode
func (bn *BatchNorm) IsTraining() bool {
	return bn.training
}

// Sequential allows chaining multiple modules together
type Sequential struct {
	modules  []Module
	training bool
}

// NewSequential creates a new Sequential container
func NewSequential(modules ...Module) *Sequential {
	return &Sequential{
		modules:  modules,
		training: true,
	}
}

// Forward passes input through all modules in sequence
func (s *Sequential) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	output := input
	var err error
	
	for i, module := range s.modules {
		output, err = module.Forward(output)
		if err != nil {
			return nil, fmt.Errorf("module %d forward failed: %v", i, err)
		}
	}
	
	return output, nil
}

// Parameters returns all trainable parameters from all modules
func (s *Sequential) Parameters() []*tensor.Tensor {
	var allParams []*tensor.Tensor
	for _, module := range s.modules {
		allParams = append(allParams, module.Parameters()...)
	}
	return allParams
}

// Train sets all modules to training mode
func (s *Sequential) Train() {
	s.training = true
	for _, module := range s.modules {
		module.Train()
	}
}

// Eval sets all modules to evaluation mode
func (s *Sequential) Eval() {
	s.training = false
	for _, module := range s.modules {
		module.Eval()
	}
}

// IsTraining returns true if in training mode
func (s *Sequential) IsTraining() bool {
	return s.training
}

// Add appends a module to the sequential container
func (s *Sequential) Add(module Module) {
	s.modules = append(s.modules, module)
}

// MaxPool2D implements a 2D max pooling layer
type MaxPool2D struct {
	kernelSize int
	stride     int
	padding    int
	training   bool
}

// NewMaxPool2D creates a new MaxPool2D layer
func NewMaxPool2D(kernelSize, stride, padding int) *MaxPool2D {
	return &MaxPool2D{
		kernelSize: kernelSize,
		stride:     stride,
		padding:    padding,
		training:   true,
	}
}

// Forward performs 2D max pooling
func (m *MaxPool2D) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("MaxPool2D expects 4D input [batch_size, channels, height, width], got shape %v", input.Shape)
	}
	
	// Use autograd MaxPool2D operation to maintain computational graph
	return tensor.MaxPool2DAutograd(input, m.kernelSize, m.stride, m.padding)
}

// Parameters returns empty slice (MaxPool2D has no parameters)
func (m *MaxPool2D) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// Train sets the module to training mode
func (m *MaxPool2D) Train() {
	m.training = true
}

// Eval sets the module to evaluation mode
func (m *MaxPool2D) Eval() {
	m.training = false
}

// IsTraining returns true if in training mode
func (m *MaxPool2D) IsTraining() bool {
	return m.training
}

// Flatten reshapes input tensor to [batch_size, -1]
type Flatten struct {
	training bool
}

// NewFlatten creates a new Flatten layer
func NewFlatten() *Flatten {
	return &Flatten{training: true}
}

// Forward flattens the input tensor to [batch_size, -1]
func (f *Flatten) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape) < 2 {
		return nil, fmt.Errorf("Flatten expects input with at least 2 dimensions, got shape %v", input.Shape)
	}
	
	batchSize := input.Shape[0]
	totalElements := input.NumElems
	flattenedSize := totalElements / batchSize
	
	// Use autograd reshape to maintain computational graph
	return tensor.ReshapeAutograd(input, []int{batchSize, flattenedSize})
}

// Parameters returns empty slice (Flatten has no parameters)
func (f *Flatten) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// Train sets the module to training mode
func (f *Flatten) Train() {
	f.training = true
}

// Eval sets the module to evaluation mode
func (f *Flatten) Eval() {
	f.training = false
}

// IsTraining returns true if in training mode
func (f *Flatten) IsTraining() bool {
	return f.training
}