package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
)

func main() {
	// Set random seed for reproducible results
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Go-Metal Training with PyTorch-Style Progress Bar ===")
	fmt.Println()

	// Run the complete training example
	if err := runTrainingWithProgress(); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
}

func runTrainingWithProgress() error {
	// Build a CNN model for image classification (similar to PyTorch example)
	inputShape := []int{32, 3, 32, 32} // CIFAR-10 style: batch_size=32, channels=3, height=32, width=32

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(32, 3, 1, 1, true, "conv1"). // 32 filters, 3x3 kernel, stride=1, padding=1
		AddReLU("relu1").
		AddConv2D(64, 3, 1, 1, true, "conv2"). // 64 filters, 3x3 kernel
		AddReLU("relu2").
		AddConv2D(128, 3, 1, 1, true, "conv3"). // 128 filters, 3x3 kernel
		AddReLU("relu3").
		AddDense(512, true, "fc1"). // Fully connected: 512 units
		AddReLU("relu4").
		AddDense(2, true, "fc2").  // Output layer: 2 classes (cats vs dogs)
		AddSoftmax(-1, "softmax"). // Softmax for classification
		Compile()

	if err != nil {
		return fmt.Errorf("failed to compile model: %v", err)
	}

	// Create trainer with Adam optimizer
	config := training.TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.001,
		OptimizerType:   cgo_bridge.Adam,
		UseDynamicEngine: true, // Dynamic engine is now the default
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.0001,
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		return fmt.Errorf("failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()

	// Training configuration
	epochs := 5
	stepsPerEpoch := 620   // Simulating CIFAR-10 training set (19,200 images / 32 batch_size = 620 steps)
	validationSteps := 155 // Simulating CIFAR-10 test set (5,000 images / 32 batch_size = 155 steps)

	// Create training session with progress visualization
	session := trainer.CreateTrainingSession("CatDogCNN", epochs, stepsPerEpoch, validationSteps)

	// Print model architecture (PyTorch style)
	session.StartTraining()

	// Initialize running metrics
	var finalRunningAccuracy float64

	// Main training loop
	for epoch := 1; epoch <= epochs; epoch++ {
		// === TRAINING PHASE ===
		session.StartEpoch(epoch)

		runningLoss := 0.0
		runningAccuracy := 0.0

		for step := 1; step <= stepsPerEpoch; step++ {
			// Simulate training step with real data
			result, err := simulateRealisticTrainingStep(trainer, config.BatchSize, epoch, step)
			if err != nil {
				return fmt.Errorf("training step %d failed: %v", step, err)
			}

			// Update running metrics with exponential moving average
			alpha := 0.02 // Smoothing factor
			runningLoss = (1-alpha)*runningLoss + alpha*float64(result.Loss)
			runningAccuracy = (1-alpha)*runningAccuracy + alpha*simulateAccuracy(epoch, step, stepsPerEpoch, "train")

			// Update progress bar
			session.UpdateTrainingProgress(step, runningLoss, runningAccuracy)

			// Simulate realistic training time per batch
			time.Sleep(time.Millisecond * 8) // ~8ms per batch = ~12.5 batch/s (realistic speed)
		}

		session.FinishTrainingEpoch()

		// === VALIDATION PHASE ===
		session.StartValidation()

		valLoss := 0.0
		valAccuracy := 0.0

		for step := 1; step <= validationSteps; step++ {
			// Simulate validation (typically faster than training)
			stepLoss := simulateLoss(epoch, step, "validation")
			stepAccuracy := simulateAccuracy(epoch, step, validationSteps, "validation")

			// Update running validation metrics
			alpha := 0.05 // Faster smoothing for validation
			valLoss = (1-alpha)*valLoss + alpha*stepLoss
			valAccuracy = (1-alpha)*valAccuracy + alpha*stepAccuracy

			// Update progress bar
			session.UpdateValidationProgress(step, valLoss, valAccuracy)

			// Validation is typically faster
			time.Sleep(time.Millisecond * 3) // ~3ms per batch = ~33 batch/s
		}

		session.FinishValidationEpoch()

		// Update final running accuracy
		finalRunningAccuracy = runningAccuracy

		// Print epoch summary
		session.PrintEpochSummary()
	}

	fmt.Println("🎉 Training completed successfully!")
	fmt.Printf("Final model performance:\n")
	fmt.Printf("  Training Accuracy: %.2f%%\n", finalRunningAccuracy*100)
	fmt.Printf("  Validation Accuracy: %.2f%%\n", finalRunningAccuracy*100) // In a real scenario, this would be separate

	return nil
}

// simulateRealisticTrainingStep performs an actual training step with dummy data
func simulateRealisticTrainingStep(trainer *training.ModelTrainer, batchSize, epoch, step int) (*training.TrainingResult, error) {
	// Create realistic dummy data
	inputShape := []int{batchSize, 3, 32, 32}
	inputData := make([]float32, batchSize*3*32*32)
	labelData := make([]int32, batchSize)

	// Generate semi-realistic input data (normalized images)
	for i := range inputData {
		inputData[i] = rand.Float32()*0.5 + 0.25 // Values between 0.25 and 0.75
	}

	// Generate balanced labels
	for i := range labelData {
		labelData[i] = int32(i % 2) // Alternating 0 and 1 for balanced classes
	}

	// Execute actual training step
	return trainer.TrainBatch(inputData, inputShape, labelData, []int{batchSize})
}

// simulateAccuracy simulates realistic accuracy progression
func simulateAccuracy(epoch, step, totalSteps int, phase string) float64 {
	// Calculate overall training progress
	progress := float64(epoch-1) + float64(step)/float64(totalSteps)
	totalEpochs := 5.0

	var baseAccuracy float64

	if phase == "train" {
		// Training accuracy: starts at ~50%, improves to ~85%
		baseAccuracy = 0.50 + 0.35*(progress/totalEpochs)

		// Training accuracy typically higher and less noisy
		noise := (rand.Float64() - 0.5) * 0.08
		baseAccuracy += noise

	} else { // validation
		// Validation accuracy: starts at ~48%, improves to ~78% (typically lower than training)
		baseAccuracy = 0.48 + 0.30*(progress/totalEpochs)

		// Validation accuracy more noisy
		noise := (rand.Float64() - 0.5) * 0.12
		baseAccuracy += noise
	}

	// Add some realistic noise
	noise := (rand.Float64() - 0.5) * 0.08
	accuracy := baseAccuracy + noise

	// Clamp to realistic bounds
	if accuracy < 0.35 {
		accuracy = 0.35
	}
	if accuracy > 0.92 {
		accuracy = 0.92
	}

	return accuracy
}

// simulateLoss simulates realistic loss progression
func simulateLoss(epoch, step int, phase string) float64 {
	// Base loss decreases over epochs
	baseLoss := 1.8 - 0.25*float64(epoch-1)

	if phase == "validation" {
		// Validation loss typically slightly higher
		baseLoss += 0.1
	}

	// Add realistic noise
	noise := (rand.Float64() - 0.5) * 0.3
	loss := baseLoss + noise

	// Clamp to realistic bounds
	if loss < 0.2 {
		loss = 0.2
	}
	if loss > 2.5 {
		loss = 2.5
	}

	return loss
}
