package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
	"github.com/tsawler/go-metal/training"
)

// generateSyntheticData creates synthetic classification data
func generateSyntheticData(samples, features, classes int) ([]float32, []int32) {
	rand.Seed(time.Now().UnixNano())
	
	// Generate input data
	inputData := make([]float32, samples*features)
	labelData := make([]int32, samples)
	
	// Create synthetic patterns for each class
	for i := 0; i < samples; i++ {
		class := i % classes
		labelData[i] = int32(class)
		
		// Generate features with class-specific patterns
		for j := 0; j < features; j++ {
			// Base pattern for the class
			baseValue := float64(class) / float64(classes)
			// Add noise
			noise := rand.Float64()*0.3 - 0.15
			// Feature-specific variation
			featureVar := math.Sin(float64(j)*0.1+float64(class)) * 0.2
			
			value := baseValue + noise + featureVar
			// Clamp to [0,1]
			if value < 0 {
				value = 0
			} else if value > 1 {
				value = 1
			}
			
			inputData[i*features+j] = float32(value)
		}
	}
	
	// Shuffle the data
	for i := samples - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		// Swap samples
		for k := 0; k < features; k++ {
			inputData[i*features+k], inputData[j*features+k] = inputData[j*features+k], inputData[i*features+k]
		}
		labelData[i], labelData[j] = labelData[j], labelData[i]
	}
	
	return inputData, labelData
}

// trainModel trains a model with the specified configuration
func trainModel(useMixedPrecision bool, epochs int) (float32, time.Duration, error) {
	// Problem configuration
	batchSize := 64
	inputSize := 784  // Similar to MNIST
	hiddenSize := 512
	numClasses := 10
	samples := 1000
	
	fmt.Printf("\n%s Training Configuration:\n", map[bool]string{false: "FP32", true: "FP16"}[useMixedPrecision])
	fmt.Printf("  Batch Size: %d\n", batchSize)
	fmt.Printf("  Input Size: %d\n", inputSize)
	fmt.Printf("  Hidden Size: %d\n", hiddenSize)
	fmt.Printf("  Classes: %d\n", numClasses)
	fmt.Printf("  Samples: %d\n", samples)
	fmt.Printf("  Epochs: %d\n", epochs)
	
	// Build model
	inputShape := []int{batchSize, inputSize}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(hiddenSize, true, "dense1").
		AddReLU("relu1").
		AddDropout(0.2, "dropout1").
		AddDense(256, true, "dense2").
		AddReLU("relu2").
		AddDropout(0.2, "dropout2").
		AddDense(numClasses, true, "output").
		Compile()
	
	if err != nil {
		return 0, 0, fmt.Errorf("model compilation failed: %v", err)
	}
	
	// Configure training
	config := training.TrainerConfig{
		BatchSize:     batchSize,
		LearningRate:  0.001,
		OptimizerType: cgo_bridge.Adam,
		EngineType:    training.Dynamic,
		LossFunction:  training.SparseCrossEntropy,
		ProblemType:   training.Classification,
		
		// Mixed precision settings
		UseMixedPrecision: useMixedPrecision,
		InitialLossScale:  65536.0,
		
		// Adam parameters
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
	
	// Create trainer
	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		return 0, 0, fmt.Errorf("trainer creation failed: %v", err)
	}
	defer trainer.Cleanup()
	
	// Generate synthetic data
	inputData, labelData := generateSyntheticData(samples, inputSize, numClasses)
	
	// Training loop
	fmt.Println("\nStarting training...")
	fmt.Println("Epoch | Loss     | Time (ms)")
	fmt.Println("------|----------|----------")
	
	startTime := time.Now()
	var finalLoss float32
	
	for epoch := 1; epoch <= epochs; epoch++ {
		epochStart := time.Now()
		
		// Process batches
		numBatches := samples / batchSize
		var epochLoss float32
		
		for batch := 0; batch < numBatches; batch++ {
			// Get batch data
			batchStart := batch * batchSize
			batchEnd := batchStart + batchSize
			
			batchInput := inputData[batchStart*inputSize : batchEnd*inputSize]
			batchLabels := labelData[batchStart:batchEnd]
			
			// Train on batch
			result, err := trainer.TrainBatch(
				batchInput,
				[]int{batchSize, inputSize},
				batchLabels,
				[]int{batchSize},
			)
			if err != nil {
				return 0, 0, fmt.Errorf("training batch %d failed: %v", batch, err)
			}
			
			epochLoss += result.Loss
		}
		
		// Average loss
		epochLoss /= float32(numBatches)
		finalLoss = epochLoss
		
		epochTime := time.Since(epochStart)
		
		// Display progress for key epochs
		if epoch <= 5 || epoch%10 == 0 || epoch == epochs {
			fmt.Printf("%5d | %.6f | %8.2f\n",
				epoch, epochLoss, float64(epochTime.Milliseconds()))
		}
		
		// Early stopping based on loss
		if epochLoss < 0.1 {
			fmt.Printf("\nEarly stopping: loss reached %.6f\n", epochLoss)
			break
		}
	}
	
	totalTime := time.Since(startTime)
	
	return finalLoss, totalTime, nil
}

func main() {
	fmt.Println("🚀 Go-Metal Mixed Precision Training Demo")
	fmt.Println("==========================================")
	
	// Initialize Metal device
	device, err := cgo_bridge.CreateMetalDevice()
	if err != nil {
		log.Fatalf("Failed to create Metal device: %v", err)
	}
	defer cgo_bridge.DestroyMetalDevice(device)
	
	// Initialize memory manager
	memory.InitializeGlobalMemoryManager(device)
	
	// Training parameters
	epochs := 20
	
	// Train with FP32
	fmt.Println("\n📊 Phase 1: FP32 Baseline Training")
	fmt.Println("==================================")
	fp32Loss, fp32Time, err := trainModel(false, epochs)
	if err != nil {
		log.Fatalf("FP32 training failed: %v", err)
	}
	
	// Train with FP16
	fmt.Println("\n⚡ Phase 2: FP16 Mixed Precision Training")
	fmt.Println("=========================================")
	fp16Loss, fp16Time, err := trainModel(true, epochs)
	if err != nil {
		log.Fatalf("FP16 training failed: %v", err)
	}
	
	// Display comparison results
	fmt.Println("\n📈 Performance Comparison")
	fmt.Println("========================")
	
	fmt.Printf("\nTraining Time:\n")
	fmt.Printf("  FP32: %.2f seconds\n", fp32Time.Seconds())
	fmt.Printf("  FP16: %.2f seconds\n", fp16Time.Seconds())
	speedup := (fp32Time.Seconds()/fp16Time.Seconds() - 1) * 100
	fmt.Printf("  Speedup: %.1f%%\n", speedup)
	
	fmt.Printf("\nFinal Loss:\n")
	fmt.Printf("  FP32: %.6f\n", fp32Loss)
	fmt.Printf("  FP16: %.6f\n", fp16Loss)
	
	lossDiff := math.Abs(float64(fp32Loss - fp16Loss))
	fmt.Printf("  Loss Difference: %.6f\n", lossDiff)
	
	// Summary
	fmt.Println("\n✅ Summary")
	fmt.Println("==========")
	
	if speedup > 50 {
		fmt.Printf("🎉 Excellent speedup achieved: %.1f%%\n", speedup)
	} else if speedup > 20 {
		fmt.Printf("✅ Good speedup achieved: %.1f%%\n", speedup)
	} else {
		fmt.Printf("⚠️  Modest speedup achieved: %.1f%%\n", speedup)
	}
	
	if lossDiff < 0.05 {
		fmt.Println("✅ Loss convergence well preserved")
	} else if lossDiff < 0.1 {
		fmt.Println("⚠️  Minor loss difference")
	} else {
		fmt.Println("❌ Significant loss difference")
	}
	
	fmt.Println("\n🏁 Mixed precision training demo completed!")
}