package tensor

import (
	"fmt"
)

// reduceGradientToShape reduces a gradient tensor to match the target shape
// This is needed when broadcasting occurred during forward pass
func reduceGradientToShape(grad *Tensor, targetShape []int) (*Tensor, error) {
	// If shapes are already the same, just clone
	if shapesEqual(grad.Shape, targetShape) {
		return grad.Clone()
	}
	
	// Handle scalar case (target shape is [1] or similar)
	if len(targetShape) == 1 && targetShape[0] == 1 {
		// Sum all elements to create scalar
		return sumAllElements(grad)
	}
	
	// General case: sum over broadcast dimensions
	result := grad
	var err error
	
	// Work backwards through dimensions
	gradDims := len(grad.Shape)
	targetDims := len(targetShape)
	
	// If target has fewer dimensions, sum over leading dimensions
	dimsToSum := gradDims - targetDims
	for i := 0; i < dimsToSum; i++ {
		result, err = sumOverDimension(result, 0) // Always sum over first dimension
		if err != nil {
			return nil, fmt.Errorf("failed to sum over dimension: %v", err)
		}
	}
	
	// Now handle remaining dimensions that might have been broadcast from size 1
	for i := 0; i < len(targetShape); i++ {
		resultDim := i
		if resultDim < len(result.Shape) && result.Shape[resultDim] != targetShape[i] {
			if targetShape[i] == 1 && result.Shape[resultDim] > 1 {
				// This dimension was broadcast from size 1, sum it
				result, err = sumOverDimension(result, resultDim)
				if err != nil {
					return nil, fmt.Errorf("failed to sum over broadcast dimension: %v", err)
				}
			}
		}
	}
	
	// Reshape to exact target shape if needed
	if !shapesEqual(result.Shape, targetShape) {
		result, err = Reshape(result, targetShape)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape gradient: %v", err)
		}
	}
	
	return result, nil
}

// sumAllElements sums all elements in a tensor to create a scalar
func sumAllElements(t *Tensor) (*Tensor, error) {
	switch t.DType {
	case Float32:
		data := t.Data.([]float32)
		sum := float32(0)
		for _, val := range data {
			sum += val
		}
		return NewTensor([]int{1}, t.DType, t.Device, []float32{sum})
	case Int32:
		data := t.Data.([]int32)
		sum := int32(0)
		for _, val := range data {
			sum += val
		}
		return NewTensor([]int{1}, t.DType, t.Device, []int32{sum})
	default:
		return nil, fmt.Errorf("unsupported data type for sum: %v", t.DType)
	}
}

// sumOverDimension sums a tensor over a specific dimension
func sumOverDimension(t *Tensor, dim int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.Shape) {
		return nil, fmt.Errorf("dimension %d out of bounds for tensor with %d dimensions", dim, len(t.Shape))
	}
	
	// Calculate output shape (remove the summed dimension)
	outputShape := make([]int, 0, len(t.Shape)-1)
	for i, size := range t.Shape {
		if i != dim {
			outputShape = append(outputShape, size)
		}
	}
	
	// Handle case where we're left with no dimensions (create scalar)
	if len(outputShape) == 0 {
		return sumAllElements(t)
	}
	
	// Create result tensor
	result, err := Zeros(outputShape, t.DType, t.Device)
	if err != nil {
		return nil, err
	}
	
	// Perform the summation
	switch t.DType {
	case Float32:
		inputData := t.Data.([]float32)
		outputData := result.Data.([]float32)
		
		// Calculate strides for input
		inputStrides := calculateStrides(t.Shape)
		
		// Sum over the specified dimension
		for outputIdx := 0; outputIdx < result.NumElems; outputIdx++ {
			// Convert flat output index to coordinates
			outputCoords := indexToCoords(outputIdx, outputShape)
			
			// Map to input coordinates (insert dimension being summed)
			inputCoords := make([]int, len(t.Shape))
			outputDim := 0
			for inputDim := 0; inputDim < len(t.Shape); inputDim++ {
				if inputDim == dim {
					inputCoords[inputDim] = 0 // Will iterate over this
				} else {
					inputCoords[inputDim] = outputCoords[outputDim]
					outputDim++
				}
			}
			
			// Sum over the dimension
			sum := float32(0)
			for k := 0; k < t.Shape[dim]; k++ {
				inputCoords[dim] = k
				inputIdx := coordsToIndex(inputCoords, inputStrides)
				sum += inputData[inputIdx]
			}
			outputData[outputIdx] = sum
		}
	case Int32:
		inputData := t.Data.([]int32)
		outputData := result.Data.([]int32)
		
		inputStrides := calculateStrides(t.Shape)
		
		for outputIdx := 0; outputIdx < result.NumElems; outputIdx++ {
			outputCoords := indexToCoords(outputIdx, outputShape)
			
			inputCoords := make([]int, len(t.Shape))
			outputDim := 0
			for inputDim := 0; inputDim < len(t.Shape); inputDim++ {
				if inputDim == dim {
					inputCoords[inputDim] = 0
				} else {
					inputCoords[inputDim] = outputCoords[outputDim]
					outputDim++
				}
			}
			
			sum := int32(0)
			for k := 0; k < t.Shape[dim]; k++ {
				inputCoords[dim] = k
				inputIdx := coordsToIndex(inputCoords, inputStrides)
				sum += inputData[inputIdx]
			}
			outputData[outputIdx] = sum
		}
	default:
		return nil, fmt.Errorf("unsupported data type for sum: %v", t.DType)
	}
	
	return result, nil
}

// Helper functions for coordinate conversion
func indexToCoords(index int, shape []int) []int {
	coords := make([]int, len(shape))
	remaining := index
	for i := len(shape) - 1; i >= 0; i-- {
		coords[i] = remaining % shape[i]
		remaining /= shape[i]
	}
	return coords
}

func coordsToIndex(coords []int, strides []int) int {
	index := 0
	for i, coord := range coords {
		index += coord * strides[i]
	}
	return index
}

// AddOp implements the Operation interface for tensor addition
type AddOp struct {
	inputs []*Tensor
}

func (op *AddOp) Forward(inputs ...*Tensor) *Tensor {
	if len(inputs) != 2 {
		panic("AddOp requires exactly 2 inputs")
	}
	
	a, b := inputs[0], inputs[1]
	op.inputs = inputs
	
	var result *Tensor
	var err error
	
	// Use GPU operation if either tensor is on GPU
	if a.Device == GPU || b.Device == GPU {
		result, err = AddMPS(a, b)
	} else {
		result, err = Add(a, b)
	}
	
	if err != nil {
		panic(fmt.Sprintf("Forward pass failed: %v", err))
	}
	
	// Set autograd properties
	result.creator = op
	result.requiresGrad = a.requiresGrad || b.requiresGrad
	
	return result
}

func (op *AddOp) Backward(gradOut *Tensor) []*Tensor {
	if len(op.inputs) != 2 {
		panic("AddOp inputs not properly stored")
	}
	
	// For addition: gradient flows unchanged to both inputs
	// ∂(a + b)/∂a = 1, ∂(a + b)/∂b = 1
	// However, if broadcasting occurred, we need to reduce gradients to original shapes
	gradA, err := reduceGradientToShape(gradOut, op.inputs[0].Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to reduce gradient for input A: %v", err))
	}
	
	gradB, err := reduceGradientToShape(gradOut, op.inputs[1].Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to reduce gradient for input B: %v", err))
	}
	
	return []*Tensor{gradA, gradB}
}

// SubOp implements the Operation interface for tensor subtraction
type SubOp struct {
	inputs []*Tensor
}

func (op *SubOp) Forward(inputs ...*Tensor) *Tensor {
	if len(inputs) != 2 {
		panic("SubOp requires exactly 2 inputs")
	}
	
	a, b := inputs[0], inputs[1]
	op.inputs = inputs
	
	// For now, only CPU subtraction is available
	result, err := Sub(a, b)
	if err != nil {
		panic(fmt.Sprintf("Forward pass failed: %v", err))
	}
	
	// Set autograd properties
	result.creator = op
	result.requiresGrad = a.requiresGrad || b.requiresGrad
	
	return result
}

func (op *SubOp) Backward(gradOut *Tensor) []*Tensor {
	// For subtraction: ∂(a - b)/∂a = 1, ∂(a - b)/∂b = -1
	// But we need to handle broadcasting by reducing gradients to original shapes
	gradA, err := reduceGradientToShape(gradOut, op.inputs[0].Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to reduce gradient for input A: %v", err))
	}
	
	// Create negative gradient for b
	negGradOut, err := gradOut.Clone()
	if err != nil {
		panic(fmt.Sprintf("Failed to clone gradient for negation: %v", err))
	}
	
	switch negGradOut.DType {
	case Float32:
		data := negGradOut.Data.([]float32)
		for i := range data {
			data[i] = -data[i]
		}
	case Int32:
		data := negGradOut.Data.([]int32)
		for i := range data {
			data[i] = -data[i]
		}
	}
	
	gradB, err := reduceGradientToShape(negGradOut, op.inputs[1].Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to reduce gradient for input B: %v", err))
	}
	
	return []*Tensor{gradA, gradB}
}

// MulOp implements the Operation interface for tensor multiplication
type MulOp struct {
	inputs []*Tensor
}

func (op *MulOp) Forward(inputs ...*Tensor) *Tensor {
	if len(inputs) != 2 {
		panic("MulOp requires exactly 2 inputs")
	}
	
	a, b := inputs[0], inputs[1]
	op.inputs = inputs
	
	// For now, only CPU multiplication is available
	result, err := Mul(a, b)
	if err != nil {
		panic(fmt.Sprintf("Forward pass failed: %v", err))
	}
	
	// Set autograd properties
	result.creator = op
	result.requiresGrad = a.requiresGrad || b.requiresGrad
	
	return result
}

func (op *MulOp) Backward(gradOut *Tensor) []*Tensor {
	a, b := op.inputs[0], op.inputs[1]
	
	// For multiplication: ∂(a * b)/∂a = b, ∂(a * b)/∂b = a
	// Need to broadcast b to gradOut shape first, then multiply
	bBroadcast, err := BroadcastTensor(b, gradOut.Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to broadcast b for gradA: %v", err))
	}
	
	gradAFull, err := Mul(gradOut, bBroadcast)
	if err != nil {
		panic(fmt.Sprintf("Backward pass failed for gradA: %v", err))
	}
	
	// Reduce to original shape of a
	gradA, err := reduceGradientToShape(gradAFull, a.Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to reduce gradient for input A: %v", err))
	}
	
	// Same for gradB
	aBroadcast, err := BroadcastTensor(a, gradOut.Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to broadcast a for gradB: %v", err))
	}
	
	gradBFull, err := Mul(gradOut, aBroadcast)
	if err != nil {
		panic(fmt.Sprintf("Backward pass failed for gradB: %v", err))
	}
	
	gradB, err := reduceGradientToShape(gradBFull, b.Shape)
	if err != nil {
		panic(fmt.Sprintf("Failed to reduce gradient for input B: %v", err))
	}
	
	return []*Tensor{gradA, gradB}
}

// MatMulOp implements the Operation interface for matrix multiplication
type MatMulOp struct {
	inputs []*Tensor
}

func (op *MatMulOp) Forward(inputs ...*Tensor) *Tensor {
	if len(inputs) != 2 {
		panic("MatMulOp requires exactly 2 inputs")
	}
	
	a, b := inputs[0], inputs[1]
	op.inputs = inputs
	
	var result *Tensor
	var err error
	
	// Use GPU operation if either tensor is on GPU
	if a.Device == GPU || b.Device == GPU {
		result, err = MatMulMPS(a, b)
	} else {
		result, err = MatMul(a, b)
	}
	
	if err != nil {
		panic(fmt.Sprintf("Forward pass failed: %v", err))
	}
	
	// Set autograd properties
	result.creator = op
	result.requiresGrad = a.requiresGrad || b.requiresGrad
	
	return result
}

func (op *MatMulOp) Backward(gradOut *Tensor) []*Tensor {
	a, b := op.inputs[0], op.inputs[1]
	
	// For matrix multiplication: ∂(A @ B)/∂A = gradOut @ B^T, ∂(A @ B)/∂B = A^T @ gradOut
	bT, err := Transpose(b, 0, 1)
	if err != nil {
		panic(fmt.Sprintf("Failed to transpose B: %v", err))
	}
	
	var gradA *Tensor
	if gradOut.Device == GPU || bT.Device == GPU {
		gradA, err = MatMulMPS(gradOut, bT)
	} else {
		gradA, err = MatMul(gradOut, bT)
	}
	if err != nil {
		panic(fmt.Sprintf("Backward pass failed for gradA: %v", err))
	}
	
	aT, err := Transpose(a, 0, 1)
	if err != nil {
		panic(fmt.Sprintf("Failed to transpose A: %v", err))
	}
	
	var gradB *Tensor
	if aT.Device == GPU || gradOut.Device == GPU {
		gradB, err = MatMulMPS(aT, gradOut)
	} else {
		gradB, err = MatMul(aT, gradOut)
	}
	if err != nil {
		panic(fmt.Sprintf("Backward pass failed for gradB: %v", err))
	}
	
	return []*Tensor{gradA, gradB}
}

// ReLUOp implements the Operation interface for ReLU activation
type ReLUOp struct {
	inputs []*Tensor
}

func (op *ReLUOp) Forward(inputs ...*Tensor) *Tensor {
	if len(inputs) != 1 {
		panic("ReLUOp requires exactly 1 input")
	}
	
	a := inputs[0]
	op.inputs = inputs
	
	var result *Tensor
	var err error
	
	// Use GPU operation if tensor is on GPU
	if a.Device == GPU {
		result, err = ReLUMPS(a)
	} else {
		result, err = ReLU(a)
	}
	
	if err != nil {
		panic(fmt.Sprintf("Forward pass failed: %v", err))
	}
	
	// Set autograd properties
	result.creator = op
	result.requiresGrad = a.requiresGrad
	
	return result
}

func (op *ReLUOp) Backward(gradOut *Tensor) []*Tensor {
	a := op.inputs[0]
	
	// For ReLU: ∂ReLU(x)/∂x = 1 if x > 0, else 0
	grad, err := gradOut.Clone()
	if err != nil {
		panic(fmt.Sprintf("Failed to clone gradient: %v", err))
	}
	
	switch a.DType {
	case Float32:
		inputData := a.Data.([]float32)
		gradData := grad.Data.([]float32)
		for i := range gradData {
			if inputData[i] <= 0 {
				gradData[i] = 0
			}
		}
	case Int32:
		inputData := a.Data.([]int32)
		gradData := grad.Data.([]int32)
		for i := range gradData {
			if inputData[i] <= 0 {
				gradData[i] = 0
			}
		}
	}
	
	return []*Tensor{grad}
}

// SigmoidOp implements the Operation interface for Sigmoid activation
type SigmoidOp struct {
	inputs []*Tensor
	output *Tensor // Store output for backward pass
}

func (op *SigmoidOp) Forward(inputs ...*Tensor) *Tensor {
	if len(inputs) != 1 {
		panic("SigmoidOp requires exactly 1 input")
	}
	
	a := inputs[0]
	op.inputs = inputs
	
	var result *Tensor
	var err error
	
	// Use GPU operation if tensor is on GPU
	if a.Device == GPU {
		result, err = SigmoidMPS(a)
	} else {
		result, err = Sigmoid(a)
	}
	
	if err != nil {
		panic(fmt.Sprintf("Forward pass failed: %v", err))
	}
	
	// Store output for backward pass
	op.output = result
	
	// Set autograd properties
	result.creator = op
	result.requiresGrad = a.requiresGrad
	
	return result
}

func (op *SigmoidOp) Backward(gradOut *Tensor) []*Tensor {
	if op.output == nil {
		panic("SigmoidOp: output not stored for backward pass")
	}
	
	// For Sigmoid: ∂σ(x)/∂x = σ(x) * (1 - σ(x))
	// gradOut * output * (1 - output)
	
	// First compute (1 - output)
	ones, err := Ones(op.output.Shape, op.output.DType, op.output.Device)
	if err != nil {
		panic(fmt.Sprintf("Failed to create ones tensor: %v", err))
	}
	
	oneMinusOutput, err := Sub(ones, op.output)
	if err != nil {
		panic(fmt.Sprintf("Failed to compute (1 - output): %v", err))
	}
	
	// output * (1 - output)
	sigmoidGrad, err := Mul(op.output, oneMinusOutput)
	if err != nil {
		panic(fmt.Sprintf("Failed to compute sigmoid gradient: %v", err))
	}
	
	// gradOut * sigmoidGrad
	grad, err := Mul(gradOut, sigmoidGrad)
	if err != nil {
		panic(fmt.Sprintf("Failed to apply chain rule: %v", err))
	}
	
	return []*Tensor{grad}
}

// High-level autograd functions that create and execute operations

// AddAutograd performs addition with automatic differentiation
func AddAutograd(a, b *Tensor) *Tensor {
	op := &AddOp{}
	return op.Forward(a, b)
}

// SubAutograd performs subtraction with automatic differentiation
func SubAutograd(a, b *Tensor) *Tensor {
	op := &SubOp{}
	return op.Forward(a, b)
}

// MulAutograd performs multiplication with automatic differentiation
func MulAutograd(a, b *Tensor) *Tensor {
	op := &MulOp{}
	return op.Forward(a, b)
}

// MatMulAutograd performs matrix multiplication with automatic differentiation
func MatMulAutograd(a, b *Tensor) *Tensor {
	op := &MatMulOp{}
	return op.Forward(a, b)
}

// ReLUAutograd performs ReLU activation with automatic differentiation
func ReLUAutograd(a *Tensor) *Tensor {
	op := &ReLUOp{}
	return op.Forward(a)
}

// SigmoidAutograd performs Sigmoid activation with automatic differentiation
func SigmoidAutograd(a *Tensor) *Tensor {
	op := &SigmoidOp{}
	return op.Forward(a)
}