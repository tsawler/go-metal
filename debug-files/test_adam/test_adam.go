package main

import (
	"fmt"
	"os"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/engine"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/optimizer"
)

func main() {
	fmt.Println("=== Adam Optimizer Integration Test ===")
	
	// Test Adam optimizer with real gradients
	err := testAdamWithRealGradients()
	if err != nil {
		fmt.Printf("Adam test failed: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Println("Adam optimizer test completed successfully!")
}

func testAdamWithRealGradients() error {
	fmt.Println("Testing Adam optimizer with real gradients...")
	
	// Create Adam configuration
	config := cgo_bridge.TrainingConfig{
		LearningRate:  0.001,
		Beta1:         0.9,
		Beta2:         0.999,
		WeightDecay:   0.0,
		Epsilon:       1e-8,
		Alpha:         0.99,  // Not used for Adam
		Momentum:      0.0,   // Not used for Adam
		Centered:      false, // Not used for Adam
		OptimizerType: cgo_bridge.Adam,
	}
	
	// Define weight shapes
	weightShapes := [][]int{
		{8, 2}, // FC weights
		{2},    // FC bias
	}
	
	// Create Adam configuration
	adamConfig := optimizer.DefaultAdamConfig()
	adamConfig.LearningRate = config.LearningRate
	adamConfig.Beta1 = config.Beta1
	adamConfig.Beta2 = config.Beta2
	adamConfig.WeightDecay = config.WeightDecay
	adamConfig.Epsilon = config.Epsilon
	
	// Create training engine with Adam optimizer
	engine, err := engine.NewMPSTrainingEngineWithAdam(config, adamConfig, weightShapes)
	if err != nil {
		return fmt.Errorf("failed to create training engine with Adam: %v", err)
	}
	defer engine.Cleanup()
	
	// Create weight tensors
	weights := make([]*memory.Tensor, len(weightShapes))
	for i, shape := range weightShapes {
		tensor, err := memory.NewTensor(shape, memory.Float32, memory.GPU)
		if err != nil {
			return fmt.Errorf("failed to create weight tensor %d: %v", i, err)
		}
		weights[i] = tensor
		defer tensor.Release()
	}
	
	// Create input and label tensors
	inputShape := []int{32, 3, 32, 32}
	labelShape := []int{32, 2}
	
	inputTensor, err := memory.NewTensor(inputShape, memory.Float32, memory.GPU)
	if err != nil {
		return fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Release()
	
	labelTensor, err := memory.NewTensor(labelShape, memory.Float32, memory.GPU)
	if err != nil {
		return fmt.Errorf("failed to create label tensor: %v", err)
	}
	defer labelTensor.Release()
	
	// Create dummy training data
	inputData := make([]float32, 32*3*32*32)
	for i := range inputData {
		inputData[i] = float32(i%100) / 100.0
	}
	
	// Create one-hot labels
	labelData := make([]float32, 32*2)
	for i := 0; i < 32; i++ {
		label := i % 2
		labelData[i*2+0] = 0.0
		labelData[i*2+1] = 0.0
		labelData[i*2+label] = 1.0
	}
	
	// Copy data to GPU
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(inputTensor.MetalBuffer(), inputData)
	if err != nil {
		return fmt.Errorf("failed to copy input data: %v", err)
	}
	
	err = cgo_bridge.CopyFloat32ArrayToMetalBuffer(labelTensor.MetalBuffer(), labelData)
	if err != nil {
		return fmt.Errorf("failed to copy label data: %v", err)
	}
	
	// Run training steps with Adam optimizer
	fmt.Printf("Running training with Adam optimizer...\n")
	
	for step := 0; step < 5; step++ {
		// Execute complete training step with Adam optimizer
		loss, err := engine.ExecuteStepHybridFullWithAdam(inputTensor, labelTensor, weights)
		if err != nil {
			return fmt.Errorf("training step %d failed: %v", step, err)
		}
		
		fmt.Printf("Step %d: Loss=%.6f\n", step, loss)
	}
	
	// Print Adam optimizer statistics
	stats := engine.GetAdamStats()
	if stats != nil {
		fmt.Printf("\nAdam Optimizer Stats:\n")
		fmt.Printf("  Step Count: %d\n", stats.StepCount)
		fmt.Printf("  Learning Rate: %.6f\n", stats.LearningRate)
		fmt.Printf("  Beta1: %.3f, Beta2: %.3f\n", stats.Beta1, stats.Beta2)
		fmt.Printf("  Epsilon: %.2e\n", stats.Epsilon)
		fmt.Printf("  Weight Decay: %.6f\n", stats.WeightDecay)
		fmt.Printf("  Parameters: %d\n", stats.NumParameters)
		fmt.Printf("  Buffer Size: %d bytes\n", stats.TotalBufferSize)
	}
	
	// Print completion message
	fmt.Printf("Adam training completed successfully!")
	
	return nil
}