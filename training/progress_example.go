package training

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
)

// TrainingExample demonstrates the PyTorch-style progress bar in action
func TrainingExample() error {
	fmt.Println("=== Go-Metal Training with Progress Visualization ===\n")
	
	// Create a CNN model similar to the PyTorch example
	inputShape := []int{32, 3, 32, 32} // CIFAR-10 style input
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(32, 3, 1, 1, true, "conv1").
		AddConv2D(64, 3, 1, 1, true, "conv2").
		AddConv2D(128, 3, 1, 1, true, "conv3").
		AddDense(512, true, "fc1").
		AddReLU("relu").
		AddDense(2, true, "fc2"). // Binary classification (cats vs dogs)
		Compile()
	
	if err != nil {
		return fmt.Errorf("failed to compile model: %v", err)
	}
	
	// Create trainer configuration
	config := TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.001,
		OptimizerType:   cgo_bridge.Adam,
		UseHybridEngine: true,
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.0001,
	}
	
	// Create model trainer
	trainer, err := NewModelTrainer(model, config)
	if err != nil {
		return fmt.Errorf("failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()
	
	// Training parameters
	epochs := 3
	stepsPerEpoch := 620    // Simulating CIFAR-10 training set size
	validationSteps := 155  // Simulating CIFAR-10 test set size
	
	// Create training session with progress visualization
	session := NewTrainingSession(trainer, "CatDogCNN", epochs, stepsPerEpoch, validationSteps)
	
	// Start training (displays model architecture)
	session.StartTraining()
	
	// Training loop
	for epoch := 1; epoch <= epochs; epoch++ {
		// Start epoch
		session.StartEpoch(epoch)
		
		// Training phase
		trainLoss := 0.0
		trainAccuracy := 0.0
		
		for step := 1; step <= stepsPerEpoch; step++ {
			// Simulate training step
			result, err := simulateTrainingStep(trainer, config.BatchSize)
			if err != nil {
				return fmt.Errorf("training step failed: %v", err)
			}
			
			// Update running metrics
			trainLoss = 0.9*trainLoss + 0.1*float64(result.Loss)
			trainAccuracy = 0.9*trainAccuracy + 0.1*simulateAccuracy(epoch, step, stepsPerEpoch)
			
			// Update progress
			session.UpdateTrainingProgress(step, trainLoss, trainAccuracy)
			
			// Small delay to simulate realistic training speed
			time.Sleep(time.Millisecond * 5)
		}
		
		session.FinishTrainingEpoch()
		
		// Validation phase
		session.StartValidation()
		
		validationLoss := 0.0
		validationAccuracy := 0.0
		
		for step := 1; step <= validationSteps; step++ {
			// Simulate validation step (faster than training)
			validationLoss = 0.9*validationLoss + 0.1*simulateValidationLoss(epoch, step)
			validationAccuracy = 0.9*validationAccuracy + 0.1*simulateValidationAccuracy(epoch, step, validationSteps)
			
			// Update progress
			session.UpdateValidationProgress(step, validationLoss, validationAccuracy)
			
			// Smaller delay for validation
			time.Sleep(time.Millisecond * 2)
		}
		
		session.FinishValidationEpoch()
		
		// Print epoch summary
		session.PrintEpochSummary()
	}
	
	fmt.Println("Training completed successfully!")
	
	return nil
}

// simulateTrainingStep simulates a real training step
func simulateTrainingStep(trainer *ModelTrainer, batchSize int) (*TrainingResult, error) {
	// Create dummy data
	inputShape := []int{batchSize, 3, 32, 32}
	inputData := make([]float32, batchSize*3*32*32)
	labelData := make([]int32, batchSize)
	
	// Fill with random data
	for i := range inputData {
		inputData[i] = rand.Float32()
	}
	for i := range labelData {
		labelData[i] = int32(rand.Intn(2))
	}
	
	// Execute training step
	return trainer.TrainBatch(inputData, inputShape, labelData, []int{batchSize})
}

// simulateAccuracy simulates improving accuracy over training
func simulateAccuracy(epoch, step, totalSteps int) float64 {
	// Simulate accuracy improving over time
	progress := float64(epoch-1) + float64(step)/float64(totalSteps)
	baseAccuracy := 0.5 + 0.3*progress/3.0 // Improve from 50% to 80% over 3 epochs
	
	// Add some noise
	noise := (rand.Float64() - 0.5) * 0.1
	accuracy := baseAccuracy + noise
	
	// Clamp to reasonable range
	if accuracy < 0.4 {
		accuracy = 0.4
	}
	if accuracy > 0.9 {
		accuracy = 0.9
	}
	
	return accuracy
}

// simulateValidationLoss simulates validation loss
func simulateValidationLoss(epoch, step int) float64 {
	// Start higher and decrease
	baseLoss := 1.5 - 0.3*float64(epoch-1)
	noise := (rand.Float64() - 0.5) * 0.2
	loss := baseLoss + noise
	
	if loss < 0.3 {
		loss = 0.3
	}
	if loss > 2.0 {
		loss = 2.0
	}
	
	return loss
}

// simulateValidationAccuracy simulates validation accuracy
func simulateValidationAccuracy(epoch, step, totalSteps int) float64 {
	// Simulate accuracy improving over epochs
	progress := float64(epoch-1) + float64(step)/float64(totalSteps)
	baseAccuracy := 0.52 + 0.25*progress/3.0 // Improve from 52% to 77% over 3 epochs
	
	// Add some noise
	noise := (rand.Float64() - 0.5) * 0.08
	accuracy := baseAccuracy + noise
	
	// Clamp to reasonable range
	if accuracy < 0.45 {
		accuracy = 0.45
	}
	if accuracy > 0.85 {
		accuracy = 0.85
	}
	
	return accuracy
}

// RunProgressBarDemo runs a demonstration of the progress bar
func RunProgressBarDemo() {
	fmt.Println("Running Go-Metal Progress Bar Demo...")
	fmt.Println("This will show PyTorch-style training progress with a real model.\n")
	
	err := TrainingExample()
	if err != nil {
		fmt.Printf("Training example failed: %v\n", err)
		return
	}
	
	fmt.Println("Demo completed!")
}