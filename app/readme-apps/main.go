package main

import (
    "fmt"
    "math/rand"
    "github.com/tsawler/go-metal/layers"
)

func main() {
    // Basic tensor operations in go-metal are performed through neural network layers
    // This example demonstrates the actual tensor operations available
    
    fmt.Println("=== Basic Tensor Operations in Go-Metal ===")
    
    // Create input data (this represents our "tensors")
    batchSize := 4
    inputSize := 6
    inputData := make([]float32, batchSize*inputSize)
    
    // Fill with sample data
    for i := range inputData {
        inputData[i] = rand.Float32() * 2.0 - 1.0  // Random values between -1 and 1
    }
    
    fmt.Printf("Input data shape: [%d, %d]\n", batchSize, inputSize)
    
    // 1. Matrix multiplication (through Dense layer)
    builder := layers.NewModelBuilder([]int{batchSize, inputSize})
    
    // This Dense layer performs: output = input * weight + bias
    // Which is matrix multiplication + addition
    model, _ := builder.
        AddDense(3, true, "matrix_multiply").  // 6->3 matrix multiplication
        Compile()
    
    fmt.Printf("Created matrix multiplication layer: %dx%d -> %dx%d\n", 
        batchSize, inputSize, batchSize, 3)
    fmt.Printf("Matrix model has %d parameters\n", model.TotalParameters)
    
    // 2. Element-wise operations (through activation layers)
    activationBuilder := layers.NewModelBuilder([]int{batchSize, inputSize})
    
    activationModel, _ := activationBuilder.
        AddReLU("relu_activation").           // Element-wise: max(0, x)
        AddDense(inputSize, false, "dense").  // Matrix multiplication
        AddSigmoid("sigmoid_activation").     // Element-wise: 1/(1+exp(-x))
        Compile()
    
    fmt.Printf("Created activation layers for element-wise operations\n")
    fmt.Printf("Activation model has %d parameters\n", activationModel.TotalParameters)
    
    // 3. Available tensor operations
    fmt.Println("\n=== Available Tensor Operations ===")
    fmt.Println("✓ Matrix Multiplication (Dense layers)")
    fmt.Println("✓ Element-wise Addition (bias in Dense layers)")
    fmt.Println("✓ Element-wise Activations (ReLU, Sigmoid, Tanh, etc.)")
    fmt.Println("✓ 2D Convolution (Conv2D layers)")
    fmt.Println("✓ Tensor Reshaping (automatic in layer transitions)")
    fmt.Println("✓ Batch Normalization (BatchNorm layers)")
    fmt.Println("✓ Dropout (element-wise masking)")
    
    fmt.Println("\nNote: Go-Metal focuses on neural network operations.")
    fmt.Println("Tensor math operations are performed within neural network layers.")
}