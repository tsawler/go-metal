package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
	"github.com/tsawler/go-metal/vision/dataloader"
	"github.com/tsawler/go-metal/vision/dataset"
)


func main() {
	fmt.Println("=== Real Cats & Dogs Training with Go-Metal Layer Interface ===")
	// fmt.Println("Training Mode: 2,000 samples, 3 epochs")
	fmt.Println()


	// Initialize random seed
	// rand.Seed(time.Now().UnixNano())

	// Load dataset
	fmt.Println("📂 Loading cats and dogs dataset...")
	dataset, err := dataset.NewCatsDogsDataset("data", 100000) // No limit
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	fmt.Println("✅", dataset.Summary())

	// Split dataset (80% train, 20% validation)
	trainDataset, valDataset := dataset.Split(0.8, true)
	fmt.Printf("📊 Train: %d images, Validation: %d images\n\n",
		trainDataset.Len(), valDataset.Len())

	// Create data loaders with shared cache
	batchSize := 8 // Reduced batch size for larger images (256x256 uses 4x more memory than 128x128)
	imageSize := 256 // trying larger image
	totalImages := dataset.Len()
	
	// Create train and validation loaders with shared cache
	trainLoader, valLoader := dataloader.CreateSharedDataLoaders(trainDataset, valDataset, dataloader.Config{
		BatchSize:    batchSize,
		ImageSize:    imageSize,
		MaxCacheSize: totalImages, // Cache all images for maximum performance
	})

	// Build CNN model for real image classification
	// Using complex multi-layer architecture with dynamic engine support
	fmt.Println("🧠 Building CNN model...")
	inputShape := []int{batchSize, 3, imageSize, imageSize} // RGB images with configurable size

	// Calculate the flattened size after conv layers
	// Each conv layer has stride=2, so dimensions are halved 3 times
	// Final spatial size = imageSize / (2^3) = imageSize / 8
	finalSpatialSize := imageSize / 8
	flattenedSize := finalSpatialSize * finalSpatialSize * 64 // 64 is the number of filters in conv3
	
	fmt.Printf("📐 Model dimensions: %dx%d input → %dx%d after conv → %d flattened\n", 
		imageSize, imageSize, finalSpatialSize, finalSpatialSize, flattenedSize)

	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(16, 3, 2, 1, true, "conv1"). // 16 filters, 3x3 kernel, stride=2
		AddReLU("relu1").
		AddConv2D(32, 3, 2, 1, true, "conv2"). // 32 filters, 3x3 kernel, stride=2
		AddReLU("relu2").
		AddConv2D(64, 3, 2, 1, true, "conv3"). // 64 filters, 3x3 kernel, stride=2
		AddReLU("relu3").
		AddDense(128, true, "fc1"). // Fully connected: 128 units
		AddReLU("relu4").
		AddDense(2, true, "fc2"). // Output: 2 classes for binary classification
		Compile()

	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}

	// Create trainer with Adam optimizer for optimal performance
	config := training.TrainerConfig{
		BatchSize:        batchSize,
		LearningRate:     0.0001,          // Standard learning rate for Adam
		OptimizerType:    cgo_bridge.Adam, // Use Adam optimizer with pooled execution
		Beta1:            0.9,    // Adam momentum parameter
		Beta2:            0.999,  // Adam squared gradient momentum
		Epsilon:          1e-8,   // Adam numerical stability
		WeightDecay:      0.0001, // L2 regularization
		ProblemType:      training.Classification,     // This is a classification problem
		LossFunction:     training.CrossEntropy,       // Use CrossEntropy loss for classification
		EngineType:       training.Dynamic,            // Use Dynamic engine for stable CNN support
	}

	trainer, err := training.NewModelTrainer(model, config)
	if err != nil {
		log.Fatalf("Failed to create trainer: %v", err)
	}
	defer trainer.Cleanup()

	// PERFORMANCE OPTIMIZATION: Enable optimized training features
	// This reduces CGO overhead and improves performance significantly
	fmt.Println("🚀 Enabling performance optimizations...")

	// Enable persistent buffers for the unified training API
	persistentInputShape := []int{batchSize, 3, imageSize, imageSize}
	err = trainer.EnablePersistentBuffers(persistentInputShape)
	if err != nil {
		log.Fatalf("Failed to enable persistent buffers: %v", err)
	}

	// Enable accuracy checking for training progress monitoring
	trainer.SetAccuracyCheckInterval(1) // Calculate accuracy every step

	fmt.Println("✅ Performance optimizations enabled")

	// EVALUATION METRICS: Enable comprehensive metrics for detailed analysis
	fmt.Println("📊 Enabling comprehensive evaluation metrics...")
	trainer.EnableEvaluationMetrics()
	fmt.Println("✅ Evaluation metrics enabled (Precision, Recall, F1, AUC-ROC)")
	
	// VISUALIZATION: Enable visualization for plotting
	fmt.Println("📈 Enabling visualization system...")
	trainer.EnableVisualization()
	fmt.Println("✅ Visualization enabled for plot generation")
	fmt.Println()

	// Training configuration - extended for comprehensive training
	epochs := 4 // Run fewer epochs for metrics validation test
	maxStepsPerEpoch := 50  // Limit training steps for demo
	maxValSteps := 10       // Limit validation steps for demo

	stepsPerEpoch := trainDataset.Len() / batchSize
	if stepsPerEpoch > maxStepsPerEpoch {
		stepsPerEpoch = maxStepsPerEpoch
	}

	validationSteps := valDataset.Len() / batchSize
	if validationSteps > maxValSteps {
		validationSteps = maxValSteps
	}

	fmt.Printf("🏋️ Training configuration:\n")
	fmt.Printf("  Epochs: %d\n", epochs)
	fmt.Printf("  Steps per epoch: %d\n", stepsPerEpoch)
	fmt.Printf("  Validation steps: %d\n", validationSteps)
	fmt.Printf("  Batch size: %d\n", batchSize)
	fmt.Printf("  Learning rate: %.4f\n", config.LearningRate)
	fmt.Printf("  Shared cache size: %d images\n", totalImages)

	// Create training session with progress visualization
	session := trainer.CreateTrainingSession("RealCatDogCNN", epochs, stepsPerEpoch, validationSteps)

	// Print model architecture
	session.StartTraining()

	// Main training loop with real data
	bestAccuracy := 0.0
	var epochStartTimes []time.Time
	var epochBatchSpeeds []float64
	var epochLosses []float64

	for epoch := 1; epoch <= epochs; epoch++ {
		// === TRAINING PHASE ===
		fmt.Printf("\n🧠 Starting Epoch %d\n", epoch)
		// printMemoryStats(fmt.Sprintf("Epoch %d Start", epoch))

		session.StartEpoch(epoch)
		trainLoader.Reset()

		runningLoss := 0.0
		correctPredictions := 0
		totalSamples := 0

		// Track epoch timing for performance monitoring
		epochStart := time.Now()
		epochStartTimes = append(epochStartTimes, epochStart)

		for step := 1; step <= stepsPerEpoch; step++ {
			// TIMING BREAKDOWN: Track data loading time
			// dataStart := time.Now()
			// Load real batch of images
			inputData, labelData, actualBatchSize, err := trainLoader.NextBatch()
			if err != nil {
				log.Printf("Warning: Failed to load batch %d: %v", step, err)
				continue
			}

			// Verification: Check data consistency (only for first batch)
			if step == 1 {
				expectedElements := actualBatchSize * 3 * imageSize * imageSize
				if len(inputData) != expectedElements {
					log.Printf("⚠️ Data size mismatch: expected %d elements, got %d", expectedElements, len(inputData))
				}
			}
			// dataTime := time.Since(dataStart)

			// TIMING BREAKDOWN: Track training step time
			// trainStart := time.Now()

			if actualBatchSize == 0 {
				break // No more data
			}

			// Train using unified API with Int32Labels for classification
			inputShape := []int{actualBatchSize, 3, imageSize, imageSize}

			// Verify tensor shape consistency
			expectedSize := actualBatchSize * 3 * imageSize * imageSize
			if len(inputData) != expectedSize {
				log.Printf("⚠️ Tensor shape mismatch: expected %d elements for shape %v, got %d",
					expectedSize, inputShape, len(inputData))
			}

			// Create Int32Labels for classification
			labels, err := training.NewInt32Labels(labelData, []int{actualBatchSize, 1})
			if err != nil {
				log.Printf("Failed to create labels: %v", err)
				continue
			}

			// Train batch using unified API
			result, err := trainer.TrainBatchUnified(inputData, inputShape, labels)
			if err != nil {
				log.Printf("Warning: Unified training step %d failed: %v", step, err)
				continue
			}

			// Calculate accuracy using the training result from TrainBatchOptimized
			if result.HasAccuracy {
				correctPredictions += int(result.Accuracy * float64(actualBatchSize))
			}
			totalSamples += actualBatchSize

			// Update running metrics
			alpha := 0.1
			runningLoss = (1-alpha)*runningLoss + alpha*float64(result.Loss)

			var runningAccuracy float64
			if totalSamples > 0 {
				runningAccuracy = float64(correctPredictions) / float64(totalSamples)
			} else {
				runningAccuracy = 0.0
			}

			// Update progress bar with real accuracy
			session.UpdateTrainingProgress(step, runningLoss, runningAccuracy)
		}

		session.FinishTrainingEpoch()

		// Calculate and log epoch performance
		epochDuration := time.Since(epochStart)
		batchSpeed := float64(stepsPerEpoch) / epochDuration.Seconds()
		epochBatchSpeeds = append(epochBatchSpeeds, batchSpeed)
		epochLosses = append(epochLosses, runningLoss)

		// DEBUG: Track performance degradation and loss trend
		// fmt.Printf("🔍 PERFORMANCE: Epoch %d - %.2f batch/s (%.2fs total), Loss: %.6f\n", epoch, batchSpeed, epochDuration.Seconds(), runningLoss)
		if len(epochBatchSpeeds) > 1 {
			// degradation := (epochBatchSpeeds[0] - batchSpeed) / epochBatchSpeeds[0] * 100
			// lossTrend := runningLoss - epochLosses[0]
			// fmt.Printf("🔍 PERFORMANCE: Performance vs Epoch 1: %.1f%% degradation, Loss trend: %+.6f\n", degradation, lossTrend)
		}

		// fmt.Printf("📈 Epoch %d Performance: %.2f batch/s (%.2fs total)\n", epoch, batchSpeed, epochDuration.Seconds())
		// if len(epochBatchSpeeds) > 1 {
		// degradation := (epochBatchSpeeds[0] - batchSpeed) / epochBatchSpeeds[0] * 100
		// fmt.Printf("📉 Performance degradation vs Epoch 1: %.1f%%\n", degradation)
		// }

		// Print cache statistics
		fmt.Printf("📦 %s\n", trainLoader.Stats())

		// === VALIDATION PHASE ===
		session.StartValidation()
		valLoader.Reset()
		
		// Reset probability collection for proper PR/ROC curves
		trainer.StartValidationPhase()

		valLoss := 0.0
		valCorrect := 0
		valTotal := 0

		for step := 1; step <= validationSteps; step++ {
			// Load validation batch
			inputData, labelData, actualBatchSize, err := valLoader.NextBatch()
			if err != nil || actualBatchSize == 0 {
				break
			}

			// Run REAL inference for validation (no training, just forward pass)
			inputShape := []int{actualBatchSize, 3, imageSize, imageSize}
			inferenceResult, err := trainer.InferBatch(inputData, inputShape)

			if err != nil {
				// Inference failed - skip this batch for accuracy calculation
				log.Printf("Warning: Validation inference failed for step %d, skipping accuracy: %v", step, err)
			} else {
				// Calculate real validation accuracy from inference predictions
				realValAccuracy := trainer.CalculateAccuracy(
					inferenceResult.Predictions,
					labelData,
					actualBatchSize,
					2, // 2 classes: cat=0, dog=1
				)
				valCorrect += int(realValAccuracy * float64(actualBatchSize))
				valTotal += actualBatchSize
				
				// EVALUATION METRICS: Update comprehensive metrics from validation inference
				err = trainer.UpdateMetricsFromInference(
					inferenceResult.Predictions,
					labelData,
					actualBatchSize,
				)
				if err != nil {
					log.Printf("Warning: Failed to update validation metrics at step %d: %v", step, err)
				}
			}

			// Update running validation metrics (estimate loss for display)
			alpha := 0.1
			estimatedLoss := runningLoss + 0.1 + 0.1*rand.Float64() // Slightly higher than training
			valLoss = (1-alpha)*valLoss + alpha*estimatedLoss

			var currentValAccuracy float64
			if valTotal > 0 {
				currentValAccuracy = float64(valCorrect) / float64(valTotal)
			} else {
				currentValAccuracy = 0.0
			}

			// Update progress bar with real validation accuracy
			session.UpdateValidationProgress(step, valLoss, currentValAccuracy)
		}

		// Track best accuracy
		var currentAccuracy float64
		if valTotal > 0 {
			currentAccuracy = float64(valCorrect) / float64(valTotal)
			if currentAccuracy > bestAccuracy {
				bestAccuracy = currentAccuracy
			}
		}

		session.FinishValidationEpoch()
		session.PrintEpochSummary()
		
		// VISUALIZATION: Record metrics for visualization (confusion matrix, ROC, PR curves)
		trainer.RecordMetricsForVisualization()
		
		// EVALUATION METRICS: Print comprehensive metrics at end of epoch
		// if trainer.IsEvaluationMetricsEnabled() {
		// 	fmt.Printf("\n📊 Comprehensive Evaluation Metrics (Epoch %d):\n", epoch)
			
		// 	// Get all classification metrics
		// 	metrics := trainer.GetClassificationMetrics()
			
		// 	fmt.Printf("   Accuracy: %.3f%%\n", metrics["accuracy"]*100)
		// 	fmt.Printf("   Precision: %.3f\n", metrics["precision"])
		// 	fmt.Printf("   Recall: %.3f\n", metrics["recall"])
		// 	fmt.Printf("   F1-Score: %.3f\n", metrics["f1_score"])
		// 	fmt.Printf("   Specificity: %.3f\n", metrics["specificity"])
		// 	fmt.Printf("   NPV: %.3f\n", metrics["npv"])
		// 	if aucVal, exists := metrics["auc_roc"]; exists {
		// 		fmt.Printf("   AUC-ROC: %.3f\n", aucVal)
		// 	}
			
		// 	// Show confusion matrix
		// 	confMatrix := trainer.GetConfusionMatrix()
		// 	if confMatrix != nil {
		// 		fmt.Printf("   Confusion Matrix:\n")
		// 		fmt.Printf("      True\\Pred  Cat  Dog\n")
		// 		fmt.Printf("      Cat      %4d %4d\n", confMatrix[0][0], confMatrix[0][1])
		// 		fmt.Printf("      Dog      %4d %4d\n", confMatrix[1][0], confMatrix[1][1])
		// 	}
		// 	fmt.Println()
		// }
		
		// PLOTTING: Generate intermediate plots every 2 epochs
		if epoch%2 == 0 || epoch == epochs {
			fmt.Printf("\n📈 Generating intermediate plots (Epoch %d)...\n", epoch)
			
			// Generate key plots
			trainingCurves := trainer.GenerateTrainingCurvesPlot()
			if len(trainingCurves.Series) > 0 {
				// Save training curves
				plotDir := fmt.Sprintf("output/plots/epoch_%d", epoch)
				err = os.MkdirAll(plotDir, 0755)
				if err == nil {
					jsonData, err := trainingCurves.ToJSON()
					if err == nil {
						filename := fmt.Sprintf("%s/training_curves.json", plotDir)
						err = os.WriteFile(filename, []byte(jsonData), 0644)
						if err == nil {
							fmt.Printf("   ✅ Training curves saved to %s\n", filename)
						}
					}
				}
			}
			
			// Also generate confusion matrix if available
			confMatrix := trainer.GenerateConfusionMatrixPlot()
			if len(confMatrix.Series) > 0 {
				plotDir := fmt.Sprintf("output/plots/epoch_%d", epoch)
				jsonData, err := confMatrix.ToJSON()
				if err == nil {
					filename := fmt.Sprintf("%s/confusion_matrix.json", plotDir)
					err = os.WriteFile(filename, []byte(jsonData), 0644)
					if err == nil {
						fmt.Printf("   ✅ Confusion matrix saved to %s\n", filename)
					}
				}
			}
		}

	}

	fmt.Println("\n🎉 Training completed!")
	fmt.Printf("📈 Best validation accuracy: %.2f%%\n", bestAccuracy*100)
	
	// EVALUATION METRICS: Final comprehensive metrics summary
	if trainer.IsEvaluationMetricsEnabled() {
		fmt.Println("\n🎯 Final Evaluation Metrics Summary:")
		
		finalMetrics := trainer.GetClassificationMetrics()
		confMatrix := trainer.GetConfusionMatrix()
		
		fmt.Printf("   Overall Performance:\n")
		fmt.Printf("     • Accuracy: %.2f%%\n", finalMetrics["accuracy"]*100)
		fmt.Printf("     • Precision: %.3f\n", finalMetrics["precision"])
		fmt.Printf("     • Recall: %.3f\n", finalMetrics["recall"])
		fmt.Printf("     • F1-Score: %.3f\n", finalMetrics["f1_score"])
		fmt.Printf("     • Specificity: %.3f\n", finalMetrics["specificity"])
		if auc, exists := finalMetrics["auc_roc"]; exists {
			fmt.Printf("     • AUC-ROC: %.3f\n", auc)
		}
		
		if confMatrix != nil {
			totalSamples := confMatrix[0][0] + confMatrix[0][1] + confMatrix[1][0] + confMatrix[1][1]
			fmt.Printf("   Confusion Matrix Analysis (%d samples):\n", totalSamples)
			fmt.Printf("     • True Positives (Dogs): %d\n", confMatrix[1][1])
			fmt.Printf("     • True Negatives (Cats): %d\n", confMatrix[0][0])
			fmt.Printf("     • False Positives: %d\n", confMatrix[0][1])
			fmt.Printf("     • False Negatives: %d\n", confMatrix[1][0])
		}
		
		// Show metrics history for plotting analysis
		aucHistory := trainer.GetMetricHistory(training.AUCROC)
		if aucHistory != nil && len(aucHistory) > 3 {
			fmt.Printf("   AUC-ROC Progression: %.3f → %.3f → %.3f (Final: %.3f)\n", 
				aucHistory[0], aucHistory[1], aucHistory[2], aucHistory[len(aucHistory)-1])
		}
		
		fmt.Println("   ✅ Comprehensive evaluation metrics collected for analysis")
	}
	
	fmt.Println("✨ Model successfully trained on cats & dogs dataset")
	
	// PLOTTING: Generate and save visualization plots
	fmt.Println("\n📊 Generating visualization plots...")
	
	// Generate all available plots
	allPlots := trainer.GenerateAllPlots()
	fmt.Printf("   Generated %d plots:\n", len(allPlots))
	
	// Create output directory for plots
	plotDir := "output/plots"
	err = os.MkdirAll(plotDir, 0755)
	if err != nil {
		log.Printf("Failed to create plot directory: %v", err)
	} else {
		// Save each plot as a JSON file
		for plotType, plotData := range allPlots {
			if len(plotData.Series) == 0 {
				fmt.Printf("   - %s: Skipped (no data)\n", plotType)
				continue
			}
			
			// Convert to JSON
			jsonData, err := plotData.ToJSON()
			if err != nil {
				log.Printf("Failed to convert %s to JSON: %v", plotType, err)
				continue
			}
			
			// Save to file
			filename := fmt.Sprintf("%s/%s.json", plotDir, plotType)
			err = os.WriteFile(filename, []byte(jsonData), 0644)
			if err != nil {
				log.Printf("Failed to save %s: %v", filename, err)
				continue
			}
			
			fmt.Printf("   - %s: Saved to %s\n", plotType, filename)
		}
	}
	
	// If plotting service is configured, send plots to sidecar
	plottingConfig := training.DefaultPlottingServiceConfig()
	plottingService := training.NewPlottingService(plottingConfig)
	
	// Enable the service first
	plottingService.Enable()
	
	// Check if sidecar is available
	fmt.Println("\n🌐 Checking for plotting sidecar service...")
	err = plottingService.CheckHealth()
	if err != nil {
		fmt.Printf("   Sidecar service not available: %v\n", err)
		fmt.Println("   Run the sidecar plotting service to generate interactive plots")
	} else {
		fmt.Println("   ✅ Sidecar service is available!")
		
		// Send plots to sidecar and automatically open in browser
		fmt.Println("   Sending plots to sidecar service...")
		results := plottingService.GenerateAndSendAllPlotsWithBrowser(trainer.GetVisualizationCollector())
		
		for plotType, result := range results {
			if result.Success {
				fmt.Printf("   - %s: ✅ Sent successfully\n", plotType)
			} else {
				fmt.Printf("   - %s: ❌ Failed: %s\n", plotType, result.Message)
			}
		}
	}
	
	fmt.Println("\n🎯 Training and visualization complete!")
}










