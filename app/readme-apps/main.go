package main

import (
    "fmt"
    "math/rand"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Create synthetic data
    batchSize := 32
    inputData := make([]float32, batchSize*10)
    for i := range inputData {
        inputData[i] = rand.Float32()
    }
    
    // Build a model
    builder := layers.NewModelBuilder([]int{batchSize, 10})
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").
        Compile()
    
    // Configure training
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    fmt.Printf("Model ready for training with %d parameters\n", model.TotalParameters)
    fmt.Printf("Training config: batch_size=%d, learning_rate=%f\n", config.BatchSize, config.LearningRate)
}