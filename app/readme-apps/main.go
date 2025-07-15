package main

import (
    "fmt"
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/training"
)

func main() {
    // Build a simple model
    builder := layers.NewModelBuilder([]int{8, 10})
    model, _ := builder.
        AddDense(16, true, "hidden").
        AddReLU("relu").
        AddDense(1, true, "output").
        Compile()
    
    config := training.TrainerConfig{
        BatchSize: 8,
        LearningRate: 0.001,
        OptimizerType: cgo_bridge.Adam,
    }
    
    trainer, _ := training.NewModelTrainer(model, config)
    
    fmt.Println("Model ready for visualization!")
    fmt.Printf("Model ready with %d parameters\n", model.TotalParameters)
    
    // Visualization features can be enabled:
    // trainer.EnableVisualization()
    // trainer.EnablePlottingService()
    
    // Optional: Configure custom plotting service
    // plotConfig := training.DefaultPlottingServiceConfig()
    // trainer.ConfigurePlottingService(plotConfig)
    
    fmt.Println("Visualization features available!")
    
    // Cleanup
    trainer.Cleanup()
}