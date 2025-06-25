package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/tsawler/go-metal/tensor"
	"github.com/tsawler/go-metal/training"
)

func main() {
	fmt.Println("=== Go-Metal Phase 6 Demo: Complete Training Pipeline ===")
	fmt.Println()

	// Set random seed for reproducibility
	rand.Seed(42)

	// Device configuration
	device := tensor.CPU
	fmt.Printf("Using device: %s\n", device)
	fmt.Println()

	// Demo 1: Binary Classification with Neural Network
	fmt.Println("📊 Demo 1: Binary Classification with Neural Network")
	fmt.Println("---------------------------------------------------")
	err := runBinaryClassificationDemo(device)
	if err != nil {
		fmt.Printf("❌ Binary classification demo failed: %v\n", err)
		return
	}
	fmt.Println()

	// Demo 2: Multi-class Classification
	fmt.Println("🎯 Demo 2: Multi-class Classification")
	fmt.Println("-------------------------------------")
	err = runMultiClassDemo(device)
	if err != nil {
		fmt.Printf("❌ Multi-class demo failed: %v\n", err)
		return
	}
	fmt.Println()

	// Demo 3: Regression Task
	fmt.Println("📈 Demo 3: Regression Task")
	fmt.Println("-------------------------")
	err = runRegressionDemo(device)
	if err != nil {
		fmt.Printf("❌ Regression demo failed: %v\n", err)
		return
	}
	fmt.Println()

	// Demo 4: Optimizer Comparison
	fmt.Println("⚡ Demo 4: Optimizer Comparison")
	fmt.Println("------------------------------")
	err = runOptimizerComparisonDemo(device)
	if err != nil {
		fmt.Printf("❌ Optimizer comparison demo failed: %v\n", err)
		return
	}
	fmt.Println()

	fmt.Println("✅ All Phase 6 demos completed successfully!")
	fmt.Println("🎉 Go-Metal Phase 6 implementation is working correctly!")
}

// runBinaryClassificationDemo demonstrates binary classification with SGD optimizer
func runBinaryClassificationDemo(device tensor.DeviceType) error {
	fmt.Println("Creating binary classification dataset...")
	
	// Create a simple binary classification dataset
	// Data points in 2D space, labels 0 or 1
	numSamples := 1000
	inputDim := 2
	
	// Generate synthetic data
	data := make([]*tensor.Tensor, numSamples)
	labels := make([]*tensor.Tensor, numSamples)
	
	for i := 0; i < numSamples; i++ {
		// Generate random 2D point
		x1 := rand.Float32()*4.0 - 2.0 // Range [-2, 2]
		x2 := rand.Float32()*4.0 - 2.0 // Range [-2, 2]
		
		// Simple decision boundary: x1 + x2 > 0
		label := 0
		if x1+x2 > 0 {
			label = 1
		}
		
		dataTensor, err := tensor.NewTensor([]int{inputDim}, tensor.Float32, device, []float32{x1, x2})
		if err != nil {
			return fmt.Errorf("failed to create data tensor: %v", err)
		}
		
		labelTensor, err := tensor.NewTensor([]int{1}, tensor.Int32, device, []int32{int32(label)})
		if err != nil {
			return fmt.Errorf("failed to create label tensor: %v", err)
		}
		
		data[i] = dataTensor
		labels[i] = labelTensor
	}
	
	// Create dataset and dataloader
	dataset, err := training.NewSimpleDataset(data, labels)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %v", err)
	}
	
	trainLoader := training.NewDataLoader(dataset, 32, true, 1, device)
	fmt.Printf("Dataset created: %d samples, batch size: 32\n", dataset.Len())
	
	// Define model architecture
	model := training.NewSequential(
		createLinearLayer(inputDim, 16, true, device),
		training.NewReLU(),
		createLinearLayer(16, 8, true, device),
		training.NewReLU(),
		createLinearLayer(8, 2, true, device), // 2 classes for binary classification
	)
	
	// Create optimizer and loss function
	optimizer := training.NewSGD(model.Parameters(), 0.01, 0.9, 0.0, 1e-4, false)
	criterion := training.NewCrossEntropyLoss("mean")
	
	// Training configuration
	config := training.TrainingConfig{
		Epochs:        10,
		Device:        device,
		PrintEvery:    5,
		ValidateEvery: 0,
		EarlyStopping: false,
		Patience:      5,
	}
	
	// Create trainer and run training
	trainer := training.NewTrainer(model, optimizer, criterion, config)
	
	fmt.Println("Starting binary classification training...")
	err = trainer.Train(trainLoader, nil)
	if err != nil {
		return fmt.Errorf("training failed: %v", err)
	}
	
	// Evaluate on a few test samples
	fmt.Println("Testing model predictions...")
	model.Eval()
	
	testCases := [][]float32{
		{1.0, 1.0},   // Should predict class 1
		{-1.0, -1.0}, // Should predict class 0
		{1.0, -0.5},  // Should predict class 1
		{-0.5, -1.0}, // Should predict class 0
	}
	
	for i, testCase := range testCases {
		testInput, err := tensor.NewTensor([]int{1, inputDim}, tensor.Float32, device, testCase)
		if err != nil {
			continue
		}
		
		output, err := trainer.Predict(testInput)
		if err != nil {
			continue
		}
		
		outputData := output.Data.([]float32)
		predictedClass := 0
		if outputData[1] > outputData[0] {
			predictedClass = 1
		}
		
		fmt.Printf("Test %d: Input=[%.1f, %.1f] -> Predicted Class=%d\n", 
			i+1, testCase[0], testCase[1], predictedClass)
	}
	
	fmt.Println("✅ Binary classification demo completed!")
	return nil
}

// runMultiClassDemo demonstrates multi-class classification with Adam optimizer
func runMultiClassDemo(device tensor.DeviceType) error {
	fmt.Println("Creating multi-class classification dataset...")
	
	// Create a 3-class classification dataset
	numSamples := 900
	inputDim := 3
	numClasses := 3
	
	data := make([]*tensor.Tensor, numSamples)
	labels := make([]*tensor.Tensor, numSamples)
	
	for i := 0; i < numSamples; i++ {
		// Generate random 3D point
		x1 := rand.Float32()*2.0 - 1.0 // Range [-1, 1]
		x2 := rand.Float32()*2.0 - 1.0
		x3 := rand.Float32()*2.0 - 1.0
		
		// Simple decision boundaries for 3 classes
		var label int32
		if x1 > 0.3 {
			label = 0
		} else if x2 > 0.3 {
			label = 1
		} else {
			label = 2
		}
		
		dataTensor, err := tensor.NewTensor([]int{inputDim}, tensor.Float32, device, []float32{x1, x2, x3})
		if err != nil {
			return fmt.Errorf("failed to create data tensor: %v", err)
		}
		
		labelTensor, err := tensor.NewTensor([]int{1}, tensor.Int32, device, []int32{label})
		if err != nil {
			return fmt.Errorf("failed to create label tensor: %v", err)
		}
		
		data[i] = dataTensor
		labels[i] = labelTensor
	}
	
	// Create dataset and dataloader
	dataset, err := training.NewSimpleDataset(data, labels)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %v", err)
	}
	
	trainLoader := training.NewDataLoader(dataset, 64, true, 1, device)
	fmt.Printf("Dataset created: %d samples, %d classes, batch size: 64\n", dataset.Len(), numClasses)
	
	// Define model with batch normalization
	model := training.NewSequential(
		createLinearLayer(inputDim, 32, true, device),
		createBatchNormLayer(32, device),
		training.NewReLU(),
		createLinearLayer(32, 16, true, device),
		training.NewReLU(),
		createLinearLayer(16, numClasses, true, device),
	)
	
	// Create Adam optimizer
	optimizer := training.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8, 1e-4)
	criterion := training.NewCrossEntropyLoss("mean")
	
	// Training configuration
	config := training.TrainingConfig{
		Epochs:        15,
		Device:        device,
		PrintEvery:    3,
		ValidateEvery: 0,
		EarlyStopping: false,
		Patience:      5,
	}
	
	// Create trainer and run training
	trainer := training.NewTrainer(model, optimizer, criterion, config)
	
	fmt.Println("Starting multi-class classification training...")
	err = trainer.Train(trainLoader, nil)
	if err != nil {
		return fmt.Errorf("training failed: %v", err)
	}
	
	fmt.Println("✅ Multi-class classification demo completed!")
	return nil
}

// runRegressionDemo demonstrates regression with MSE loss
func runRegressionDemo(device tensor.DeviceType) error {
	fmt.Println("Creating regression dataset...")
	
	// Create a simple regression dataset: y = 2*x1 + 3*x2 + 1 + noise
	numSamples := 800
	inputDim := 2
	
	data := make([]*tensor.Tensor, numSamples)
	labels := make([]*tensor.Tensor, numSamples)
	
	for i := 0; i < numSamples; i++ {
		x1 := rand.Float32()*2.0 - 1.0 // Range [-1, 1]
		x2 := rand.Float32()*2.0 - 1.0
		
		// True function: y = 2*x1 + 3*x2 + 1 + small noise
		noise := (rand.Float32() - 0.5) * 0.1 // Small noise
		y := 2.0*x1 + 3.0*x2 + 1.0 + noise
		
		dataTensor, err := tensor.NewTensor([]int{inputDim}, tensor.Float32, device, []float32{x1, x2})
		if err != nil {
			return fmt.Errorf("failed to create data tensor: %v", err)
		}
		
		labelTensor, err := tensor.NewTensor([]int{1}, tensor.Float32, device, []float32{y})
		if err != nil {
			return fmt.Errorf("failed to create label tensor: %v", err)
		}
		
		data[i] = dataTensor
		labels[i] = labelTensor
	}
	
	// Create dataset and dataloader
	dataset, err := training.NewSimpleDataset(data, labels)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %v", err)
	}
	
	trainLoader := training.NewDataLoader(dataset, 64, true, 1, device)
	fmt.Printf("Dataset created: %d samples for regression, batch size: 64\n", dataset.Len())
	
	// Define model
	model := training.NewSequential(
		createLinearLayer(inputDim, 16, true, device),
		training.NewReLU(),
		createLinearLayer(16, 8, true, device),
		training.NewReLU(),
		createLinearLayer(8, 1, true, device), // Single output for regression
	)
	
	// Create optimizer and MSE loss
	optimizer := training.NewSGD(model.Parameters(), 0.01, 0.0, 0.0, 0.0, false)
	criterion := training.NewMSELoss("mean")
	
	// Training configuration
	config := training.TrainingConfig{
		Epochs:        20,
		Device:        device,
		PrintEvery:    5,
		ValidateEvery: 0,
		EarlyStopping: false,
		Patience:      5,
	}
	
	// Create trainer and run training
	trainer := training.NewTrainer(model, optimizer, criterion, config)
	
	fmt.Println("Starting regression training...")
	err = trainer.Train(trainLoader, nil)
	if err != nil {
		return fmt.Errorf("training failed: %v", err)
	}
	
	// Test predictions
	fmt.Println("Testing regression predictions...")
	model.Eval()
	
	testCases := [][]float32{
		{0.5, 0.5},   // Expected: ~2*0.5 + 3*0.5 + 1 = 3.5
		{-0.5, 0.5},  // Expected: ~2*(-0.5) + 3*0.5 + 1 = 1.5
		{0.0, 0.0},   // Expected: ~1.0
		{1.0, -1.0},  // Expected: ~2*1.0 + 3*(-1.0) + 1 = 0.0
	}
	
	for i, testCase := range testCases {
		expected := 2.0*testCase[0] + 3.0*testCase[1] + 1.0
		
		testInput, err := tensor.NewTensor([]int{1, inputDim}, tensor.Float32, device, testCase)
		if err != nil {
			continue
		}
		
		output, err := trainer.Predict(testInput)
		if err != nil {
			continue
		}
		
		predicted := output.Data.([]float32)[0]
		
		fmt.Printf("Test %d: Input=[%.1f, %.1f] -> Expected=%.2f, Predicted=%.2f\n", 
			i+1, testCase[0], testCase[1], expected, predicted)
	}
	
	fmt.Println("✅ Regression demo completed!")
	return nil
}

// runOptimizerComparisonDemo compares SGD vs Adam performance
func runOptimizerComparisonDemo(device tensor.DeviceType) error {
	fmt.Println("Comparing SGD vs Adam optimizers...")
	
	// Create a simple dataset for comparison
	numSamples := 500
	inputDim := 4
	
	data := make([]*tensor.Tensor, numSamples)
	labels := make([]*tensor.Tensor, numSamples)
	
	for i := 0; i < numSamples; i++ {
		features := make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			features[j] = rand.Float32()*2.0 - 1.0
		}
		
		// Binary classification based on sum of features
		label := 0
		if features[0]+features[1]+features[2]+features[3] > 0 {
			label = 1
		}
		
		dataTensor, err := tensor.NewTensor([]int{inputDim}, tensor.Float32, device, features)
		if err != nil {
			return fmt.Errorf("failed to create data tensor: %v", err)
		}
		
		labelTensor, err := tensor.NewTensor([]int{1}, tensor.Int32, device, []int32{int32(label)})
		if err != nil {
			return fmt.Errorf("failed to create label tensor: %v", err)
		}
		
		data[i] = dataTensor
		labels[i] = labelTensor
	}
	
	dataset, err := training.NewSimpleDataset(data, labels)
	if err != nil {
		return fmt.Errorf("failed to create dataset: %v", err)
	}
	
	// Test SGD
	fmt.Println("Training with SGD optimizer...")
	sgdLoader := training.NewDataLoader(dataset, 32, true, 1, device)
	sgdModel := training.NewSequential(
		createLinearLayer(inputDim, 16, true, device),
		training.NewReLU(),
		createLinearLayer(16, 2, true, device),
	)
	
	sgdOptimizer := training.NewSGD(sgdModel.Parameters(), 0.01, 0.9, 0.0, 0.0, false)
	sgdCriterion := training.NewCrossEntropyLoss("mean")
	
	sgdConfig := training.TrainingConfig{
		Epochs:        8,
		Device:        device,
		PrintEvery:    0, // No batch printing for comparison
		ValidateEvery: 0,
		EarlyStopping: false,
		Patience:      5,
	}
	
	sgdTrainer := training.NewTrainer(sgdModel, sgdOptimizer, sgdCriterion, sgdConfig)
	
	start := time.Now()
	err = sgdTrainer.Train(sgdLoader, nil)
	sgdTime := time.Since(start)
	if err != nil {
		return fmt.Errorf("SGD training failed: %v", err)
	}
	
	sgdMetrics := sgdTrainer.GetMetrics()
	sgdFinalLoss := sgdMetrics[len(sgdMetrics)-1].TrainLoss
	
	// Test Adam
	fmt.Println("Training with Adam optimizer...")
	adamLoader := training.NewDataLoader(dataset, 32, true, 2, device) // Different seed
	adamModel := training.NewSequential(
		createLinearLayer(inputDim, 16, true, device),
		training.NewReLU(),
		createLinearLayer(16, 2, true, device),
	)
	
	adamOptimizer := training.NewAdam(adamModel.Parameters(), 0.001, 0.9, 0.999, 1e-8, 0.0)
	adamCriterion := training.NewCrossEntropyLoss("mean")
	
	adamConfig := training.TrainingConfig{
		Epochs:        8,
		Device:        device,
		PrintEvery:    0, // No batch printing for comparison
		ValidateEvery: 0,
		EarlyStopping: false,
		Patience:      5,
	}
	
	adamTrainer := training.NewTrainer(adamModel, adamOptimizer, adamCriterion, adamConfig)
	
	start = time.Now()
	err = adamTrainer.Train(adamLoader, nil)
	adamTime := time.Since(start)
	if err != nil {
		return fmt.Errorf("Adam training failed: %v", err)
	}
	
	adamMetrics := adamTrainer.GetMetrics()
	adamFinalLoss := adamMetrics[len(adamMetrics)-1].TrainLoss
	
	// Compare results
	fmt.Println("📊 Optimizer Comparison Results:")
	fmt.Printf("SGD  - Final Loss: %.4f, Training Time: %v\n", sgdFinalLoss, sgdTime)
	fmt.Printf("Adam - Final Loss: %.4f, Training Time: %v\n", adamFinalLoss, adamTime)
	
	if adamFinalLoss < sgdFinalLoss {
		fmt.Println("🏆 Adam achieved lower final loss")
	} else {
		fmt.Println("🏆 SGD achieved lower final loss")
	}
	
	if adamTime < sgdTime {
		fmt.Println("⚡ Adam was faster")
	} else {
		fmt.Println("⚡ SGD was faster")
	}
	
	fmt.Println("✅ Optimizer comparison demo completed!")
	return nil
}

// Helper functions for creating layers
func createLinearLayer(inputSize, outputSize int, bias bool, device tensor.DeviceType) *training.Linear {
	layer, err := training.NewLinear(inputSize, outputSize, bias, device)
	if err != nil {
		panic(fmt.Sprintf("Failed to create linear layer: %v", err))
	}
	return layer
}

func createBatchNormLayer(numFeatures int, device tensor.DeviceType) *training.BatchNorm {
	layer, err := training.NewBatchNorm(numFeatures, 1e-5, 0.1, device)
	if err != nil {
		panic(fmt.Sprintf("Failed to create batch norm layer: %v", err))
	}
	return layer
}