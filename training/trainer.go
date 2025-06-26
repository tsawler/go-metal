package training

import (
	"fmt"
	"time"

	"github.com/tsawler/go-metal/tensor"
)

// TrainingConfig holds configuration for training
type TrainingConfig struct {
	Epochs        int
	Device        tensor.DeviceType
	PrintEvery    int  // Print training stats every N batches
	ValidateEvery int  // Run validation every N epochs (0 = no validation)
	EarlyStopping bool // Enable early stopping based on validation loss
	Patience      int  // Number of epochs to wait for improvement before stopping
}

// TrainingMetrics holds metrics for a single epoch
type TrainingMetrics struct {
	Epoch          int
	TrainLoss      float64
	TrainAccuracy  float64
	ValidLoss      float64
	ValidAccuracy  float64
	EpochDuration  time.Duration
	BatchCount     int
}

// Trainer manages the training process
type Trainer struct {
	model     Module
	optimizer Optimizer
	criterion Loss
	config    TrainingConfig
	metrics   []TrainingMetrics
}

// NewTrainer creates a new Trainer
func NewTrainer(model Module, optimizer Optimizer, criterion Loss, config TrainingConfig) *Trainer {
	return &Trainer{
		model:     model,
		optimizer: optimizer,
		criterion: criterion,
		config:    config,
		metrics:   make([]TrainingMetrics, 0),
	}
}

// Train runs the complete training loop
func (t *Trainer) Train(trainLoader, validLoader *DataLoader) error {
	fmt.Printf("Starting training for %d epochs on %s\n", t.config.Epochs, t.config.Device)
	
	// Move model parameters to PersistentGPU if using GPU training
	if t.config.Device == tensor.PersistentGPU {
		err := t.moveModelToPersistentGPU()
		if err != nil {
			return fmt.Errorf("failed to move model to persistent GPU: %v", err)
		}
		fmt.Println("Model parameters moved to persistent GPU for optimal training performance")
	}
	
	bestValidLoss := float64(1e10)
	patienceCounter := 0
	
	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		epochStart := time.Now()
		
		// Training phase
		t.model.Train()
		trainLoss, trainAcc, batchCount, err := t.trainEpoch(trainLoader, epoch)
		if err != nil {
			return fmt.Errorf("training epoch %d failed: %v", epoch, err)
		}
		
		epochDuration := time.Since(epochStart)
		
		// Create metrics for this epoch
		metrics := TrainingMetrics{
			Epoch:         epoch,
			TrainLoss:     trainLoss,
			TrainAccuracy: trainAcc,
			EpochDuration: epochDuration,
			BatchCount:    batchCount,
		}
		
		// Validation phase
		if validLoader != nil && (t.config.ValidateEvery > 0 && (epoch+1)%t.config.ValidateEvery == 0) {
			t.model.Eval()
			validLoss, validAcc, err := t.validateEpoch(validLoader, epoch)
			if err != nil {
				return fmt.Errorf("validation epoch %d failed: %v", epoch, err)
			}
			
			metrics.ValidLoss = validLoss
			metrics.ValidAccuracy = validAcc
			
			// Early stopping logic
			if t.config.EarlyStopping {
				if validLoss < bestValidLoss {
					bestValidLoss = validLoss
					patienceCounter = 0
				} else {
					patienceCounter++
					if patienceCounter >= t.config.Patience {
						fmt.Printf("Early stopping triggered after %d epochs\n", epoch+1)
						break
					}
				}
			}
		}
		
		// Store metrics
		t.metrics = append(t.metrics, metrics)
		
		// Print progress
		t.printEpochSummary(metrics)
	}
	
	return nil
}

// trainEpoch runs one training epoch
func (t *Trainer) trainEpoch(trainLoader *DataLoader, epoch int) (float64, float64, int, error) {
	var totalLoss float64
	var totalCorrect int
	var totalSamples int
	var batchCount int
	
	for batch := range trainLoader.Iterator() {
		batchStart := time.Now()
		
		// Zero gradients
		t.optimizer.ZeroGrad()
		
		// Ensure input data is on the same device as model
		input, err := t.ensureDeviceMatch(batch.Data)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("failed to match input device to model: %v", err)
		}
		
		// Forward pass
		output, err := t.model.Forward(input)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("forward pass failed: %v", err)
		}
		
		// Prepare labels for loss computation
		labels := batch.Labels
		
		// For CrossEntropyLoss, reshape labels from [batch_size, 1] to [batch_size] if needed
		if _, isCE := t.criterion.(*CrossEntropyLoss); isCE {
			if len(labels.Shape) == 2 && labels.Shape[1] == 1 {
				labels, err = labels.Reshape([]int{labels.Shape[0]})
				if err != nil {
					return 0, 0, 0, fmt.Errorf("failed to reshape labels: %v", err)
				}
			}
		}
		
		// Compute loss
		loss, err := t.criterion.Forward(output, labels)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("loss computation failed: %v", err)
		}
		
		// Backward pass
		lossValue, err := loss.Item()
		if err != nil {
			return 0, 0, 0, fmt.Errorf("failed to get loss value: %v", err)
		}
		
		err = loss.Backward()
		if err != nil {
			return 0, 0, 0, fmt.Errorf("backward pass failed: %v", err)
		}
		
		// Update parameters
		err = t.optimizer.Step()
		if err != nil {
			return 0, 0, 0, fmt.Errorf("optimizer step failed: %v", err)
		}
		
		// Accumulate metrics
		batchSize := batch.Data.Shape[0]
		lossFloat64 := float64(lossValue.(float32))
		totalLoss += lossFloat64 * float64(batchSize)
		totalSamples += batchSize
		batchCount++
		
		// Calculate accuracy (for classification tasks)
		if labels.DType == tensor.Int32 {
			correct, err := t.calculateAccuracy(output, labels)
			if err == nil {
				totalCorrect += correct
			}
		}
		
		// Print batch progress
		if t.config.PrintEvery > 0 && batchCount%t.config.PrintEvery == 0 {
			batchDuration := time.Since(batchStart)
			avgLoss := totalLoss / float64(totalSamples)
			accuracy := float64(totalCorrect) / float64(totalSamples) * 100.0
			
			fmt.Printf("Epoch %d, Batch %d: Loss=%.4f, Acc=%.2f%%, Time=%v\n",
				epoch, batchCount, avgLoss, accuracy, batchDuration)
		}
	}
	
	avgLoss := totalLoss / float64(totalSamples)
	accuracy := float64(totalCorrect) / float64(totalSamples) * 100.0
	
	return avgLoss, accuracy, batchCount, nil
}

// validateEpoch runs one validation epoch
func (t *Trainer) validateEpoch(validLoader *DataLoader, epoch int) (float64, float64, error) {
	var totalLoss float64
	var totalCorrect int
	var totalSamples int
	
	for batch := range validLoader.Iterator() {
		// Ensure input data is on the same device as model
		input, err := t.ensureDeviceMatch(batch.Data)
		if err != nil {
			return 0, 0, fmt.Errorf("failed to match input device to model: %v", err)
		}
		
		// Forward pass (no gradients needed)
		output, err := t.model.Forward(input)
		if err != nil {
			return 0, 0, fmt.Errorf("validation forward pass failed: %v", err)
		}
		
		// Prepare labels for loss computation
		labels := batch.Labels
		
		// For CrossEntropyLoss, reshape labels from [batch_size, 1] to [batch_size] if needed
		if _, isCE := t.criterion.(*CrossEntropyLoss); isCE {
			if len(labels.Shape) == 2 && labels.Shape[1] == 1 {
				labels, err = labels.Reshape([]int{labels.Shape[0]})
				if err != nil {
					return 0, 0, fmt.Errorf("failed to reshape validation labels: %v", err)
				}
			}
		}
		
		// Compute loss
		loss, err := t.criterion.Forward(output, labels)
		if err != nil {
			return 0, 0, fmt.Errorf("validation loss computation failed: %v", err)
		}
		
		// Accumulate metrics
		lossValue, err := loss.Item()
		if err != nil {
			return 0, 0, fmt.Errorf("failed to get validation loss value: %v", err)
		}
		
		batchSize := batch.Data.Shape[0]
		lossFloat64 := float64(lossValue.(float32))
		totalLoss += lossFloat64 * float64(batchSize)
		totalSamples += batchSize
		
		// Calculate accuracy (for classification tasks)
		if labels.DType == tensor.Int32 {
			correct, err := t.calculateAccuracy(output, labels)
			if err == nil {
				totalCorrect += correct
			}
		}
	}
	
	avgLoss := totalLoss / float64(totalSamples)
	accuracy := float64(totalCorrect) / float64(totalSamples) * 100.0
	
	return avgLoss, accuracy, nil
}

// calculateAccuracy computes classification accuracy
func (t *Trainer) calculateAccuracy(output, target *tensor.Tensor) (int, error) {
	if output.DType != tensor.Float32 || target.DType != tensor.Int32 {
		return 0, fmt.Errorf("accuracy calculation requires Float32 output and Int32 target")
	}
	
	if len(output.Shape) != 2 || len(target.Shape) != 1 {
		return 0, fmt.Errorf("accuracy calculation requires 2D output and 1D target")
	}
	
	batchSize := output.Shape[0]
	numClasses := output.Shape[1]
	
	if target.Shape[0] != batchSize {
		return 0, fmt.Errorf("batch size mismatch: output %d, target %d", batchSize, target.Shape[0])
	}
	
	outputData := output.Data.([]float32)
	targetData := target.Data.([]int32)
	
	correct := 0
	
	for i := 0; i < batchSize; i++ {
		// Find predicted class (argmax)
		maxIdx := 0
		maxVal := outputData[i*numClasses]
		
		for j := 1; j < numClasses; j++ {
			if outputData[i*numClasses+j] > maxVal {
				maxVal = outputData[i*numClasses+j]
				maxIdx = j
			}
		}
		
		// Check if prediction matches target
		if int32(maxIdx) == targetData[i] {
			correct++
		}
	}
	
	return correct, nil
}

// printEpochSummary prints a summary of the epoch results
func (t *Trainer) printEpochSummary(metrics TrainingMetrics) {
	fmt.Printf("Epoch %d/%d: ", metrics.Epoch+1, t.config.Epochs)
	fmt.Printf("Train Loss=%.4f, Train Acc=%.2f%%", metrics.TrainLoss, metrics.TrainAccuracy)
	
	if metrics.ValidLoss > 0 {
		fmt.Printf(", Valid Loss=%.4f, Valid Acc=%.2f%%", metrics.ValidLoss, metrics.ValidAccuracy)
	}
	
	fmt.Printf(", Time=%v, Batches=%d\n", metrics.EpochDuration, metrics.BatchCount)
}

// GetMetrics returns all training metrics
func (t *Trainer) GetMetrics() []TrainingMetrics {
	return t.metrics
}

// Evaluate runs the model on a dataset and returns overall metrics
func (t *Trainer) Evaluate(dataLoader *DataLoader) (float64, float64, error) {
	t.model.Eval()
	
	var totalLoss float64
	var totalCorrect int
	var totalSamples int
	
	for batch := range dataLoader.Iterator() {
		// Ensure input data is on the same device as model
		input, err := t.ensureDeviceMatch(batch.Data)
		if err != nil {
			return 0, 0, fmt.Errorf("failed to match input device to model: %v", err)
		}
		
		// Forward pass
		output, err := t.model.Forward(input)
		if err != nil {
			return 0, 0, fmt.Errorf("evaluation forward pass failed: %v", err)
		}
		
		// Prepare labels for loss computation
		labels := batch.Labels
		
		// For CrossEntropyLoss, reshape labels from [batch_size, 1] to [batch_size] if needed
		if _, isCE := t.criterion.(*CrossEntropyLoss); isCE {
			if len(labels.Shape) == 2 && labels.Shape[1] == 1 {
				labels, err = labels.Reshape([]int{labels.Shape[0]})
				if err != nil {
					return 0, 0, fmt.Errorf("failed to reshape evaluation labels: %v", err)
				}
			}
		}
		
		// Compute loss
		loss, err := t.criterion.Forward(output, labels)
		if err != nil {
			return 0, 0, fmt.Errorf("evaluation loss computation failed: %v", err)
		}
		
		// Accumulate metrics
		lossValue, err := loss.Item()
		if err != nil {
			return 0, 0, fmt.Errorf("failed to get evaluation loss value: %v", err)
		}
		
		batchSize := batch.Data.Shape[0]
		lossFloat64 := float64(lossValue.(float32))
		totalLoss += lossFloat64 * float64(batchSize)
		totalSamples += batchSize
		
		// Calculate accuracy (for classification tasks)
		if labels.DType == tensor.Int32 {
			correct, err := t.calculateAccuracy(output, labels)
			if err == nil {
				totalCorrect += correct
			}
		}
	}
	
	avgLoss := totalLoss / float64(totalSamples)
	accuracy := float64(totalCorrect) / float64(totalSamples) * 100.0
	
	return avgLoss, accuracy, nil
}

// Predict runs inference on a single batch
func (t *Trainer) Predict(input *tensor.Tensor) (*tensor.Tensor, error) {
	t.model.Eval()
	
	// Ensure input data is on the same device as model
	input, err := t.ensureDeviceMatch(input)
	if err != nil {
		return nil, fmt.Errorf("failed to match input device to model: %v", err)
	}
	
	return t.model.Forward(input)
}

// SaveCheckpoint saves model parameters (simplified version)
func (t *Trainer) SaveCheckpoint(filepath string) error {
	// This is a placeholder for model serialization
	// In a full implementation, this would save model parameters to disk
	return fmt.Errorf("checkpoint saving not implemented yet")
}

// LoadCheckpoint loads model parameters (simplified version)
func (t *Trainer) LoadCheckpoint(filepath string) error {
	// This is a placeholder for model deserialization
	// In a full implementation, this would load model parameters from disk
	return fmt.Errorf("checkpoint loading not implemented yet")
}

// moveModelToPersistentGPU moves all model parameters to PersistentGPU
func (t *Trainer) moveModelToPersistentGPU() error {
	params := t.model.Parameters()
	for i, param := range params {
		if param.Device != tensor.PersistentGPU {
			persistentParam, err := param.ToPersistentGPU()
			if err != nil {
				return fmt.Errorf("failed to move parameter %d to persistent GPU: %v", i, err)
			}
			// Update the parameter in place
			*param = *persistentParam
		}
	}
	return nil
}

// GetModelDevice returns the device type of the first model parameter
func (t *Trainer) GetModelDevice() tensor.DeviceType {
	params := t.model.Parameters()
	if len(params) > 0 {
		return params[0].Device
	}
	return tensor.CPU
}

// ensureDeviceMatch ensures input tensor is on the same device as the model
func (t *Trainer) ensureDeviceMatch(input *tensor.Tensor) (*tensor.Tensor, error) {
	modelDevice := t.GetModelDevice()
	
	// If input is already on the correct device, return as-is
	if input.Device == modelDevice {
		return input, nil
	}
	
	// Transfer input to match model device
	switch modelDevice {
	case tensor.PersistentGPU:
		return input.ToPersistentGPU()
	case tensor.GPU:
		return input.ToGPU()
	case tensor.CPU:
		return input.ToCPU()
	default:
		return input, nil
	}
}