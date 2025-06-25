package training

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tsawler/go-metal/tensor"
)

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
	
	weightData := make([]float32, outputSize*inputSize)
	for i := range weightData {
		weightData[i] = float32((rand.Float64()*2.0 - 1.0) * bound)
	}
	
	weight, err := tensor.NewTensor([]int{outputSize, inputSize}, tensor.Float32, device, weightData)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
	}
	weight.SetRequiresGrad(true)
	
	linear := &Linear{
		weight:   weight,
		training: true,
	}
	
	if bias {
		// Initialize bias to zeros
		biasData := make([]float32, outputSize)
		biasT, err := tensor.NewTensor([]int{outputSize}, tensor.Float32, device, biasData)
		if err != nil {
			return nil, fmt.Errorf("failed to create bias tensor: %v", err)
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
	
	batchSize := input.Shape[0]
	inputSize := input.Shape[1]
	
	if inputSize != l.weight.Shape[1] {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", l.weight.Shape[1], inputSize)
	}
	
	// Compute xW^T
	weightTransposed, err := l.weight.Transpose(0, 1)
	if err != nil {
		return nil, fmt.Errorf("weight transpose failed: %v", err)
	}
	
	output, err := tensor.MatMul(input, weightTransposed)
	if err != nil {
		return nil, fmt.Errorf("matrix multiplication failed: %v", err)
	}
	
	// Add bias if present
	if l.bias != nil {
		// Broadcast bias across batch dimension
		biasExpanded, err := l.expandBias(l.bias, batchSize)
		if err != nil {
			return nil, fmt.Errorf("bias expansion failed: %v", err)
		}
		
		output, err = tensor.Add(output, biasExpanded)
		if err != nil {
			return nil, fmt.Errorf("bias addition failed: %v", err)
		}
	}
	
	return output, nil
}

// expandBias expands bias from [output_size] to [batch_size, output_size]
func (l *Linear) expandBias(bias *tensor.Tensor, batchSize int) (*tensor.Tensor, error) {
	outputSize := bias.Shape[0]
	expandedData := make([]float32, batchSize*outputSize)
	biasData := bias.Data.([]float32)
	
	for i := 0; i < batchSize; i++ {
		copy(expandedData[i*outputSize:(i+1)*outputSize], biasData)
	}
	
	return tensor.NewTensor([]int{batchSize, outputSize}, bias.DType, bias.Device, expandedData)
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
	if input.Device == tensor.GPU {
		return tensor.ReLUMPS(input)
	}
	return tensor.ReLU(input)
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
		weightData[i] = float32((rand.Float64()*2.0 - 1.0) * bound)
	}
	
	// Weight shape: [output_channels, input_channels, kernel_height, kernel_width]
	weight, err := tensor.NewTensor([]int{outputChannels, inputChannels, kernelSize, kernelSize}, tensor.Float32, device, weightData)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
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
		biasT, err := tensor.NewTensor([]int{outputChannels}, tensor.Float32, device, biasData)
		if err != nil {
			return nil, fmt.Errorf("failed to create bias tensor: %v", err)
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
	
	// Use MPSGraph Conv2D operation
	return tensor.Conv2DMPS(input, c.weight, c.bias, c.stride, c.stride, c.padding, c.padding, c.padding, c.padding)
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
	
	batchSize := input.Shape[0]
	features := input.Shape[1]
	
	if features != bn.numFeatures {
		return nil, fmt.Errorf("input features mismatch: expected %d, got %d", bn.numFeatures, features)
	}
	
	if bn.training {
		// Training mode: compute batch statistics
		batchMean, err := bn.computeBatchMean(input)
		if err != nil {
			return nil, fmt.Errorf("batch mean computation failed: %v", err)
		}
		
		batchVar, err := bn.computeBatchVariance(input, batchMean)
		if err != nil {
			return nil, fmt.Errorf("batch variance computation failed: %v", err)
		}
		
		// Update running statistics
		err = bn.updateRunningStats(batchMean, batchVar)
		if err != nil {
			return nil, fmt.Errorf("running stats update failed: %v", err)
		}
		
		// Normalize using batch statistics
		return bn.normalize(input, batchMean, batchVar, batchSize)
	} else {
		// Evaluation mode: use running statistics
		return bn.normalize(input, bn.runningMean, bn.runningVar, batchSize)
	}
}

// computeBatchMean computes the mean across the batch dimension
func (bn *BatchNorm) computeBatchMean(input *tensor.Tensor) (*tensor.Tensor, error) {
	batchSize := float32(input.Shape[0])
	inputData := input.Data.([]float32)
	features := input.Shape[1]
	
	meanData := make([]float32, features)
	
	for j := 0; j < features; j++ {
		var sum float32
		for i := 0; i < int(batchSize); i++ {
			sum += inputData[i*features+j]
		}
		meanData[j] = sum / batchSize
	}
	
	return tensor.NewTensor([]int{features}, tensor.Float32, input.Device, meanData)
}

// computeBatchVariance computes the variance across the batch dimension
func (bn *BatchNorm) computeBatchVariance(input, mean *tensor.Tensor) (*tensor.Tensor, error) {
	batchSize := float32(input.Shape[0])
	inputData := input.Data.([]float32)
	meanData := mean.Data.([]float32)
	features := input.Shape[1]
	
	varData := make([]float32, features)
	
	for j := 0; j < features; j++ {
		var sumSq float32
		for i := 0; i < int(batchSize); i++ {
			diff := inputData[i*features+j] - meanData[j]
			sumSq += diff * diff
		}
		varData[j] = sumSq / batchSize
	}
	
	return tensor.NewTensor([]int{features}, tensor.Float32, input.Device, varData)
}

// updateRunningStats updates the running mean and variance using exponential moving average
func (bn *BatchNorm) updateRunningStats(batchMean, batchVar *tensor.Tensor) error {
	momentum := float32(bn.momentum)
	
	// running_mean = (1 - momentum) * running_mean + momentum * batch_mean
	runningMeanData := bn.runningMean.Data.([]float32)
	batchMeanData := batchMean.Data.([]float32)
	
	for i := range runningMeanData {
		runningMeanData[i] = (1.0-momentum)*runningMeanData[i] + momentum*batchMeanData[i]
	}
	
	// running_var = (1 - momentum) * running_var + momentum * batch_var
	runningVarData := bn.runningVar.Data.([]float32)
	batchVarData := batchVar.Data.([]float32)
	
	for i := range runningVarData {
		runningVarData[i] = (1.0-momentum)*runningVarData[i] + momentum*batchVarData[i]
	}
	
	return nil
}

// normalize performs the normalization: (x - mean) / sqrt(var + eps) * gamma + beta
func (bn *BatchNorm) normalize(input, mean, variance *tensor.Tensor, batchSize int) (*tensor.Tensor, error) {
	inputData := input.Data.([]float32)
	meanData := mean.Data.([]float32)
	varData := variance.Data.([]float32)
	gammaData := bn.gamma.Data.([]float32)
	betaData := bn.beta.Data.([]float32)
	
	features := input.Shape[1]
	outputData := make([]float32, len(inputData))
	
	for i := 0; i < batchSize; i++ {
		for j := 0; j < features; j++ {
			idx := i*features + j
			
			// Normalize: (x - mean) / sqrt(var + eps)
			normalized := (inputData[idx] - meanData[j]) / float32(math.Sqrt(float64(varData[j])+bn.eps))
			
			// Scale and shift: normalized * gamma + beta
			outputData[idx] = normalized*gammaData[j] + betaData[j]
		}
	}
	
	return tensor.NewTensor(input.Shape, input.DType, input.Device, outputData)
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