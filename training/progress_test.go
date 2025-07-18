package training

import (
	"fmt"
	"testing"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
)

// TestProgressBar tests the basic progress bar functionality
func TestProgressBar(t *testing.T) {
	fmt.Println("\n=== Testing Progress Bar ===")
	
	// Test basic progress bar
	pb := NewProgressBar("Testing", 10)
	
	for i := 1; i <= 10; i++ {
		metrics := map[string]float64{
			"loss":     1.0 - float64(i)*0.08,
			"accuracy": float64(i) * 0.09,
		}
		
		pb.Update(i, metrics)
		time.Sleep(time.Millisecond * 100) // Brief pause to see progress
	}
	
	pb.Finish()
	fmt.Println("✅ Basic progress bar test completed")
}

// TestModelArchitecturePrinting tests the model architecture printing
func TestModelArchitecturePrinting(t *testing.T) {
	fmt.Println("\n=== Testing Model Architecture Printing ===")
	
	// Create a test model
	inputShape := []int{32, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(32, 3, 1, 1, true, "conv1").
		AddConv2D(64, 3, 1, 1, true, "conv2").
		AddConv2D(128, 3, 1, 1, true, "conv3").
		AddDense(512, true, "fc1").
		AddReLU("relu").
		AddDense(10, true, "fc2").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile test model: %v", err)
	}
	
	// Print model architecture
	printer := NewModelArchitecturePrinter("TestCNN")
	printer.PrintArchitecture(model)
	
	fmt.Println("✅ Model architecture printing test completed")
}

// TestTrainingSessionDemo tests the complete training session workflow
func TestTrainingSessionDemo(t *testing.T) {
	fmt.Println("\n=== Testing Training Session Demo ===")
	
	// Create a simple model for testing
	inputShape := []int{16, 3, 28, 28}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(64, true, "fc1").
		AddReLU("relu2").
		AddDense(2, true, "fc2").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile test model: %v", err)
	}
	
	// Create trainer
	config := TrainerConfig{
		BatchSize:       16,
		LearningRate:    0.01,
		OptimizerType:   cgo_bridge.SGD,
		UseDynamicEngine: true, // Dynamic engine is now the default
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.0,
	}
	
	trainer, err := NewModelTrainer(model, config)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()
	
	// Create and run a short training session
	session := NewTrainingSession(trainer, "TestModel", 2, 5, 3)
	session.StartTraining()
	
	// Simulate 2 epochs of training
	for epoch := 1; epoch <= 2; epoch++ {
		session.StartEpoch(epoch)
		
		// Training phase
		for step := 1; step <= 5; step++ {
			loss := 1.0 - float64(epoch*step)*0.05
			accuracy := float64(epoch*step) * 0.15
			session.UpdateTrainingProgress(step, loss, accuracy)
			time.Sleep(time.Millisecond * 50)
		}
		session.FinishTrainingEpoch()
		
		// Validation phase
		session.StartValidation()
		for step := 1; step <= 3; step++ {
			loss := 0.8 - float64(epoch*step)*0.03
			accuracy := 0.6 + float64(epoch*step)*0.05
			session.UpdateValidationProgress(step, loss, accuracy)
			time.Sleep(time.Millisecond * 30)
		}
		session.FinishValidationEpoch()
		
		session.PrintEpochSummary()
	}
	
	fmt.Println("✅ Training session demo test completed")
}

// TestProgressBarFormatting tests various formatting scenarios
func TestProgressBarFormatting(t *testing.T) {
	fmt.Println("\n=== Testing Progress Bar Formatting ===")
	
	// Test with different metric types
	pb := NewProgressBar("Formatting Test", 5)
	
	testMetrics := []map[string]float64{
		{"loss": 1.234},
		{"loss": 0.987, "accuracy": 0.456},
		{"loss": 0.654, "accuracy": 0.789, "precision": 0.234},
		{"loss": 0.321, "acc": 0.876}, // Test different accuracy key
		{"val_loss": 0.123, "val_accuracy": 0.934},
	}
	
	for i, metrics := range testMetrics {
		pb.Update(i+1, metrics)
		time.Sleep(time.Millisecond * 100)
	}
	
	pb.Finish()
	fmt.Println("✅ Progress bar formatting test completed")
}

// BenchmarkProgressBar benchmarks progress bar performance
func BenchmarkProgressBar(b *testing.B) {
	pb := NewProgressBar("Benchmark", b.N)
	metrics := map[string]float64{
		"loss":     0.5,
		"accuracy": 0.8,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pb.Update(i+1, metrics)
	}
}