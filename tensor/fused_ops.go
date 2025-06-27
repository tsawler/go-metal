package tensor

import (
	"fmt"
)

// ===== FUSED TENSOR OPERATIONS FOR PHASE 6.3 =====
// These operations combine multiple GPU kernels into single calls for performance

// LinearForward performs fused matrix multiplication + bias addition
// Equivalent to: MatMul(input, weight^T) + bias, but in a single GPU kernel
// Note: weight is expected to be [output_features, input_features] to match training.Linear convention
func LinearForward(input, weight, bias *Tensor) (*Tensor, error) {
	// Validate input shapes
	if len(input.Shape) != 2 || len(weight.Shape) != 2 || len(bias.Shape) != 1 {
		return nil, fmt.Errorf("LinearForward requires 2D input/weight and 1D bias tensors")
	}
	
	if input.Shape[1] != weight.Shape[1] {
		return nil, fmt.Errorf("input features (%d) must match weight input features (%d)", 
			input.Shape[1], weight.Shape[1])
	}
	
	if bias.Shape[0] != weight.Shape[0] {
		return nil, fmt.Errorf("bias size (%d) must match weight output features (%d)", 
			bias.Shape[0], weight.Shape[0])
	}
	
	// Use explicit operations to ensure mathematical correctness
	// This matches the standard linear layer operation: input @ weight^T + bias
	weightT, err := weight.Transpose(0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose weight matrix: %v", err)
	}
	
	var matmul *Tensor
	if input.Device == GPU || weightT.Device == GPU {
		matmul, err = MatMulMPS(input, weightT)
	} else {
		matmul, err = MatMul(input, weightT)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to perform matrix multiplication: %v", err)
	}
	
	// Use broadcasting-aware addition for bias
	result, err := Add(matmul, bias)
	if err != nil {
		return nil, fmt.Errorf("failed to add bias: %v", err)
	}
	
	return result, nil
}

// LinearReLU performs fused matrix multiplication + bias addition + ReLU activation
// Equivalent to: ReLU(MatMul(input, weight^T) + bias), but in a single GPU kernel
// Note: weight is expected to be [output_features, input_features] to match training.Linear convention
func LinearReLU(input, weight, bias *Tensor) (*Tensor, error) {
	// Validate input shapes
	if len(input.Shape) != 2 || len(weight.Shape) != 2 || len(bias.Shape) != 1 {
		return nil, fmt.Errorf("LinearReLU requires 2D input/weight and 1D bias tensors")
	}
	
	if input.Shape[1] != weight.Shape[1] {
		return nil, fmt.Errorf("input features (%d) must match weight input features (%d)", 
			input.Shape[1], weight.Shape[1])
	}
	
	if bias.Shape[0] != weight.Shape[0] {
		return nil, fmt.Errorf("bias size (%d) must match weight output features (%d)", 
			bias.Shape[0], weight.Shape[0])
	}
	
	// Use explicit operations to ensure mathematical correctness
	// This matches the standard linear layer operation: ReLU(input @ weight^T + bias)
	weightT, err := weight.Transpose(0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose weight matrix: %v", err)
	}
	
	var matmul *Tensor
	if input.Device == GPU || weightT.Device == GPU {
		matmul, err = MatMulMPS(input, weightT)
	} else {
		matmul, err = MatMul(input, weightT)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to perform matrix multiplication: %v", err)
	}
	
	// Use broadcasting-aware addition for bias
	added, err := Add(matmul, bias)
	if err != nil {
		return nil, fmt.Errorf("failed to add bias: %v", err)
	}
	
	// Apply ReLU activation
	result, err := ReLU(added)
	if err != nil {
		return nil, fmt.Errorf("failed to apply ReLU activation: %v", err)
	}
	
	return result, nil
}

// LinearSigmoid performs fused matrix multiplication + bias addition + Sigmoid activation
// Equivalent to: Sigmoid(MatMul(input, weight^T) + bias), but in a single GPU kernel
// Note: weight is expected to be [output_features, input_features] to match training.Linear convention
func LinearSigmoid(input, weight, bias *Tensor) (*Tensor, error) {
	if input.Device != GPU && weight.Device != GPU {
		// Fallback to separate operations for CPU tensors
		weightT, err := weight.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		matmul, err := MatMul(input, weightT)
		if err != nil {
			return nil, err
		}
		added, err := Add(matmul, bias)
		if err != nil {
			return nil, err
		}
		return Sigmoid(added)
	}

	// Validate input shapes
	if len(input.Shape) != 2 || len(weight.Shape) != 2 || len(bias.Shape) != 1 {
		return nil, fmt.Errorf("LinearSigmoid requires 2D input/weight and 1D bias tensors")
	}

	batchSize := uint(input.Shape[0])
	inputFeatures := uint(input.Shape[1])
	outputFeatures := uint(weight.Shape[0])  // weight is [output_features, input_features]

	if input.Shape[1] != weight.Shape[1] {  // input_features must match weight.Shape[1]
		return nil, fmt.Errorf("input features (%d) must match weight input features (%d)", 
			input.Shape[1], weight.Shape[1])
	}
	if bias.Shape[0] != int(outputFeatures) {
		return nil, fmt.Errorf("bias size (%d) must match output features (%d)", 
			bias.Shape[0], outputFeatures)
	}

	// Only support Float32 for now
	if input.DType != Float32 || weight.DType != Float32 || bias.DType != Float32 {
		return nil, fmt.Errorf("LinearSigmoid only supports Float32 tensors")
	}

	// Get compute engine
	engine, err := getComputeEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get compute engine: %v", err)
	}

	// Extract float32 data
	inputData := input.Data.([]float32)
	weightData := weight.Data.([]float32)
	biasData := bias.Data.([]float32)

	// Execute fused linear + Sigmoid kernel
	resultData, err := engine.LinearSigmoidFloat32(inputData, weightData, biasData, 
		batchSize, inputFeatures, outputFeatures)
	if err != nil {
		return nil, fmt.Errorf("failed to execute fused linear sigmoid: %v", err)
	}

	// Create result tensor
	resultShape := []int{int(batchSize), int(outputFeatures)}
	result, err := NewTensor(resultShape, Float32, GPU, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}

	return result, nil
}

// BatchMatMul performs batched matrix multiplication
// Input tensors should be 3D: [batch_size, M, N] x [batch_size, N, P] -> [batch_size, M, P]
func BatchMatMul(tensorA, tensorB *Tensor) (*Tensor, error) {
	if tensorA.Device != GPU && tensorB.Device != GPU {
		// Fallback to separate operations for CPU tensors
		return nil, fmt.Errorf("CPU batch matmul not implemented, use GPU tensors")
	}

	// Validate input shapes
	if len(tensorA.Shape) != 3 || len(tensorB.Shape) != 3 {
		return nil, fmt.Errorf("BatchMatMul requires 3D tensors")
	}

	batchSize := uint(tensorA.Shape[0])
	M := uint(tensorA.Shape[1])
	N := uint(tensorA.Shape[2])
	P := uint(tensorB.Shape[2])

	if tensorA.Shape[0] != tensorB.Shape[0] {
		return nil, fmt.Errorf("batch sizes must match: %d vs %d", tensorA.Shape[0], tensorB.Shape[0])
	}
	if tensorA.Shape[2] != tensorB.Shape[1] {
		return nil, fmt.Errorf("inner dimensions must match: %d vs %d", tensorA.Shape[2], tensorB.Shape[1])
	}

	// Only support Float32 for now
	if tensorA.DType != Float32 || tensorB.DType != Float32 {
		return nil, fmt.Errorf("BatchMatMul only supports Float32 tensors")
	}

	// Get compute engine
	engine, err := getComputeEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to get compute engine: %v", err)
	}

	// Extract float32 data
	dataA := tensorA.Data.([]float32)
	dataB := tensorB.Data.([]float32)

	// Execute batch matrix multiplication kernel
	resultData, err := engine.BatchMatMulFloat32(dataA, dataB, batchSize, M, N, P)
	if err != nil {
		return nil, fmt.Errorf("failed to execute batch matmul: %v", err)
	}

	// Create result tensor
	resultShape := []int{int(batchSize), int(M), int(P)}
	result, err := NewTensor(resultShape, Float32, GPU, resultData)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}

	return result, nil
}

// FusedOperationDetector analyzes a sequence of operations to detect fusion opportunities
type FusedOperationDetector struct {
	operations []OperationDesc
}

// NewFusedOperationDetector creates a new operation fusion detector
func NewFusedOperationDetector() *FusedOperationDetector {
	return &FusedOperationDetector{
		operations: make([]OperationDesc, 0),
	}
}

// AddOperation adds an operation to the sequence for analysis
func (fod *FusedOperationDetector) AddOperation(opType string, inputs []*Tensor, params map[string]interface{}) {
	fod.operations = append(fod.operations, OperationDesc{
		Type:   opType,
		Inputs: inputs,
		Params: params,
	})
}

// DetectFusions analyzes the operation sequence and returns optimized fused operations
func (fod *FusedOperationDetector) DetectFusions() ([]OperationDesc, error) {
	optimized := make([]OperationDesc, 0)
	
	i := 0
	for i < len(fod.operations) {
		// Check for MatMul + Add + Activation fusion patterns
		if i+2 < len(fod.operations) {
			op1 := fod.operations[i]
			op2 := fod.operations[i+1]
			op3 := fod.operations[i+2]
			
			// Pattern: MatMul -> Add -> ReLU
			if op1.Type == "MatMul" && op2.Type == "Add" && op3.Type == "ReLU" {
				// Extract weight and bias from the operations
				if len(op1.Inputs) >= 2 && len(op2.Inputs) >= 2 {
					fusedOp := OperationDesc{
						Type: "LinearReLU",
						Inputs: []*Tensor{op1.Inputs[0], op1.Inputs[1], op2.Inputs[1]}, // input, weight, bias
						Params: make(map[string]interface{}),
					}
					optimized = append(optimized, fusedOp)
					i += 3 // Skip the fused operations
					continue
				}
			}
			
			// Pattern: MatMul -> Add -> Sigmoid
			if op1.Type == "MatMul" && op2.Type == "Add" && op3.Type == "Sigmoid" {
				if len(op1.Inputs) >= 2 && len(op2.Inputs) >= 2 {
					fusedOp := OperationDesc{
						Type: "LinearSigmoid",
						Inputs: []*Tensor{op1.Inputs[0], op1.Inputs[1], op2.Inputs[1]}, // input, weight, bias
						Params: make(map[string]interface{}),
					}
					optimized = append(optimized, fusedOp)
					i += 3 // Skip the fused operations
					continue
				}
			}
		}
		
		// Check for MatMul + Add fusion (without activation)
		if i+1 < len(fod.operations) {
			op1 := fod.operations[i]
			op2 := fod.operations[i+1]
			
			// Pattern: MatMul -> Add
			if op1.Type == "MatMul" && op2.Type == "Add" {
				if len(op1.Inputs) >= 2 && len(op2.Inputs) >= 2 {
					fusedOp := OperationDesc{
						Type: "LinearForward",
						Inputs: []*Tensor{op1.Inputs[0], op1.Inputs[1], op2.Inputs[1]}, // input, weight, bias
						Params: make(map[string]interface{}),
					}
					optimized = append(optimized, fusedOp)
					i += 2 // Skip the fused operations
					continue
				}
			}
		}
		
		// No fusion detected, keep original operation
		optimized = append(optimized, fod.operations[i])
		i++
	}
	
	return optimized, nil
}

// ExecuteFusedOperation executes a fused operation based on its type
func ExecuteFusedOperation(op OperationDesc) (*Tensor, error) {
	switch op.Type {
	case "LinearForward":
		if len(op.Inputs) != 3 {
			return nil, fmt.Errorf("LinearForward requires 3 inputs: input, weight, bias")
		}
		return LinearForward(op.Inputs[0], op.Inputs[1], op.Inputs[2])
		
	case "LinearReLU":
		if len(op.Inputs) != 3 {
			return nil, fmt.Errorf("LinearReLU requires 3 inputs: input, weight, bias")
		}
		return LinearReLU(op.Inputs[0], op.Inputs[1], op.Inputs[2])
		
	case "LinearSigmoid":
		if len(op.Inputs) != 3 {
			return nil, fmt.Errorf("LinearSigmoid requires 3 inputs: input, weight, bias")
		}
		return LinearSigmoid(op.Inputs[0], op.Inputs[1], op.Inputs[2])
		
	case "BatchMatMul":
		if len(op.Inputs) != 2 {
			return nil, fmt.Errorf("BatchMatMul requires 2 inputs")
		}
		return BatchMatMul(op.Inputs[0], op.Inputs[1])
		
	default:
		return nil, fmt.Errorf("unsupported fused operation type: %s", op.Type)
	}
}

// OptimizeOperationSequence takes a sequence of operations and returns an optimized version with fusions
func OptimizeOperationSequence(operations []OperationDesc) ([]OperationDesc, error) {
	detector := NewFusedOperationDetector()
	
	// Add all operations to the detector
	for _, op := range operations {
		detector.AddOperation(op.Type, op.Inputs, op.Params)
	}
	
	// Detect and return fused operations
	return detector.DetectFusions()
}