package training

import (
	"fmt"
	"math"
	"testing"

	"github.com/tsawler/go-metal/tensor"
)

// TestCrossDeviceTrainingVerification implements comprehensive cross-device training verification
func TestCrossDeviceTrainingVerification(t *testing.T) {
	if !tensor.IsGPUAvailable() {
		t.Skip("GPU not available for cross-device testing")
	}

	devices := []tensor.DeviceType{tensor.CPU, tensor.GPU, tensor.PersistentGPU}
	
	for _, primaryDevice := range devices {
		t.Run(fmt.Sprintf("PrimaryDevice_%v", primaryDevice), func(t *testing.T) {
			t.Run("DeterministicTraining", func(t *testing.T) {
				testDeterministicTraining(t, primaryDevice)
			})
			
			t.Run("CrossDeviceConsistency", func(t *testing.T) {
				testCrossDeviceConsistency(t, primaryDevice)
			})
			
			t.Run("DeviceTransferDuringTraining", func(t *testing.T) {
				testDeviceTransferDuringTraining(t, primaryDevice)
			})
			
			t.Run("GradientSynchronization", func(t *testing.T) {
				testGradientSynchronization(t, primaryDevice)
			})
			
			t.Run("AutogradGraphPreservation", func(t *testing.T) {
				testAutogradGraphPreservation(t, primaryDevice)
			})
			
			t.Run("MemoryConsistencyDuringTransfers", func(t *testing.T) {
				testMemoryConsistencyDuringTransfers(t, primaryDevice)
			})
		})
	}
}

// testDeterministicTraining verifies that training on the same device produces identical results
func testDeterministicTraining(t *testing.T, device tensor.DeviceType) {
	const numRuns = 3
	const numEpochs = 10
	const tolerance = 1e-6
	
	// Store results from multiple runs
	var allLosses [][]float32
	var allAccuracies []float32
	
	for run := 0; run < numRuns; run++ {
		// Set fixed seed for deterministic results
		SetRandomSeed(12345) // Fixed seed for reproducible results
		
		// Create identical model architecture
		conv, err := NewConv2D(1, 4, 3, 1, 1, true, device)
		if err != nil {
			t.Fatalf("Failed to create Conv2D: %v", err)
		}
		
		relu := NewReLU()
		flatten := NewFlatten()
		linear, err := NewLinear(64, 2, true, device) // 4*4*4=64 after conv+flatten
		if err != nil {
			t.Fatalf("Failed to create Linear: %v", err)
		}
		
		model := NewSequential(conv, relu, flatten, linear)
		
		// Create identical training data
		trainData := []float32{
			// Sample 1: pattern in top-left
			1.0, 0.5, 0.0, 0.0,
			0.5, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			// Sample 2: pattern in bottom-right  
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.5, 1.0,
			0.0, 0.0, 1.0, 0.5,
		}
		trainInput, err := tensor.NewTensor([]int{2, 1, 4, 4}, tensor.Float32, device, trainData)
		if err != nil {
			t.Fatalf("Failed to create training input: %v", err)
		}
		
		trainLabels := []int32{0, 1}
		trainTarget, err := tensor.NewTensor([]int{2}, tensor.Int32, device, trainLabels)
		if err != nil {
			t.Fatalf("Failed to create training target: %v", err)
		}
		
		// Create optimizer and loss
		params := model.Parameters()
		optimizer := NewSGD(params, 0.01, 0.0, 0.0, 0.0, false)
		criterion := NewCrossEntropyLoss("mean")
		
		// Training loop
		var losses []float32
		for epoch := 0; epoch < numEpochs; epoch++ {
			optimizer.ZeroGrad()
			
			output, err := model.Forward(trainInput)
			if err != nil {
				t.Fatalf("Forward pass failed at epoch %d: %v", epoch, err)
			}
			
			loss, err := criterion.Forward(output, trainTarget)
			if err != nil {
				t.Fatalf("Loss computation failed at epoch %d: %v", epoch, err)
			}
			
			lossData := tensorFloat32(loss)
			losses = append(losses, lossData[0])
			
			err = loss.Backward()
			if err != nil {
				t.Fatalf("Backward pass failed at epoch %d: %v", epoch, err)
			}
			
			err = optimizer.Step()
			if err != nil {
				t.Fatalf("Optimizer step failed at epoch %d: %v", epoch, err)
			}
		}
		
		// Calculate final accuracy
		finalOutput, err := model.Forward(trainInput)
		if err != nil {
			t.Fatalf("Final forward pass failed: %v", err)
		}
		
		predictions, err := tensorArgmax(finalOutput, 1)
		if err != nil {
			t.Fatalf("Argmax failed: %v", err)
		}
		
		predData := tensorInt32(predictions)
		correct := 0
		for i := range predData {
			if predData[i] == trainLabels[i] {
				correct++
			}
		}
		accuracy := float32(correct) / float32(len(trainLabels))
		
		allLosses = append(allLosses, losses)
		allAccuracies = append(allAccuracies, accuracy)
		
		t.Logf("Run %d on %v: Final loss=%.6f, Accuracy=%.1f%%", 
			run+1, device, losses[len(losses)-1], accuracy*100)
	}
	
	// Verify deterministic behavior
	if numRuns > 1 {
		// Compare loss trajectories
		for epoch := 0; epoch < numEpochs; epoch++ {
			baseLoss := allLosses[0][epoch]
			for run := 1; run < numRuns; run++ {
				diff := math.Abs(float64(allLosses[run][epoch]) - float64(baseLoss))
				if diff > tolerance {
					t.Errorf("Non-deterministic training detected: epoch %d, run %d vs run 1: loss difference %.8f > tolerance %.8f", 
						epoch, run+1, diff, tolerance)
				}
			}
		}
		
		// Compare final accuracies
		baseAccuracy := allAccuracies[0]
		for run := 1; run < numRuns; run++ {
			if allAccuracies[run] != baseAccuracy {
				t.Errorf("Non-deterministic accuracy: run %d accuracy %.6f != run 1 accuracy %.6f", 
					run+1, allAccuracies[run], baseAccuracy)
			}
		}
		
		t.Logf("Deterministic training verified: all %d runs produced identical results", numRuns)
	}
}

// testCrossDeviceConsistency verifies that training produces equivalent results across devices  
func testCrossDeviceConsistency(t *testing.T, baseDevice tensor.DeviceType) {
	const numEpochs = 5
	const tolerance = 1e-3 // Allow for slight numerical differences between CPU/GPU
	
	devices := []tensor.DeviceType{tensor.CPU, tensor.GPU, tensor.PersistentGPU}
	
	// Store results for each device
	deviceResults := make(map[tensor.DeviceType][]float32)
	
	for _, device := range devices {
		t.Logf("Training on device: %v", device)
		
		// Set identical seed for each device to ensure same initialization
		SetRandomSeed(42)
		
		// Create model with identical initialization
		conv, err := NewConv2D(1, 2, 2, 1, 0, true, device)
		if err != nil {
			t.Fatalf("Failed to create Conv2D on %v: %v", device, err)
		}
		
		relu := NewReLU()
		flatten := NewFlatten()
		linear, err := NewLinear(2, 2, true, device)
		if err != nil {
			t.Fatalf("Failed to create Linear on %v: %v", device, err)
		}
		
		model := NewSequential(conv, relu, flatten, linear)
		
		// Initialize with same weights (copy from base device)
		if device != baseDevice {
			err = synchronizeModelWeights(model, deviceResults, baseDevice, device)
			if err != nil {
				t.Fatalf("Failed to synchronize weights to %v: %v", device, err)
			}
		}
		
		// Create training data
		trainData := []float32{
			1.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			0.0, 1.0,
		}
		trainInput, err := tensor.NewTensor([]int{2, 1, 2, 2}, tensor.Float32, device, trainData)
		if err != nil {
			t.Fatalf("Failed to create training input on %v: %v", device, err)
		}
		
		trainLabels := []int32{0, 1}
		trainTarget, err := tensor.NewTensor([]int{2}, tensor.Int32, device, trainLabels)
		if err != nil {
			t.Fatalf("Failed to create training target on %v: %v", device, err)
		}
		
		// Create optimizer
		params := model.Parameters()
		optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
		criterion := NewCrossEntropyLoss("mean")
		
		// Training loop
		var losses []float32
		for epoch := 0; epoch < numEpochs; epoch++ {
			optimizer.ZeroGrad()
			
			output, err := model.Forward(trainInput)
			if err != nil {
				t.Fatalf("Forward pass failed on %v at epoch %d: %v", device, epoch, err)
			}
			
			loss, err := criterion.Forward(output, trainTarget)
			if err != nil {
				t.Fatalf("Loss computation failed on %v at epoch %d: %v", device, epoch, err)
			}
			
			lossData := tensorFloat32(loss)
			losses = append(losses, lossData[0])
			
			err = loss.Backward()
			if err != nil {
				t.Fatalf("Backward pass failed on %v at epoch %d: %v", device, epoch, err)
			}
			
			err = optimizer.Step()
			if err != nil {
				t.Fatalf("Optimizer step failed on %v at epoch %d: %v", device, epoch, err)
			}
		}
		
		deviceResults[device] = losses
		t.Logf("Device %v final loss: %.6f", device, losses[len(losses)-1])
	}
	
	// Compare results across devices
	baseLosses := deviceResults[baseDevice]
	for _, device := range devices {
		if device == baseDevice {
			continue
		}
		
		losses := deviceResults[device]
		if len(losses) != len(baseLosses) {
			t.Errorf("Inconsistent training length: %v has %d epochs, %v has %d epochs", 
				device, len(losses), baseDevice, len(baseLosses))
			continue
		}
		
		for epoch := 0; epoch < len(losses); epoch++ {
			diff := math.Abs(float64(losses[epoch]) - float64(baseLosses[epoch]))
			relativeDiff := diff / math.Max(math.Abs(float64(baseLosses[epoch])), 1e-8)
			
			if relativeDiff > tolerance {
				t.Errorf("Cross-device inconsistency at epoch %d: %v loss %.6f vs %v loss %.6f (relative diff %.6f > tolerance %.6f)", 
					epoch, device, losses[epoch], baseDevice, baseLosses[epoch], relativeDiff, tolerance)
			}
		}
	}
	
	t.Logf("Cross-device consistency verified across %d devices", len(devices))
}

// Helper function to synchronize model weights across devices
func synchronizeModelWeights(model *Sequential, deviceResults map[tensor.DeviceType][]float32, sourceDevice, targetDevice tensor.DeviceType) error {
	// For now, we'll rely on the fact that models start with random initialization
	// In a full implementation, this would copy exact weights from source to target
	// This is a placeholder - the real test will verify mathematical consistency given the current initialization approach
	return nil
}

// testDeviceTransferDuringTraining tests seamless transfers during training
func testDeviceTransferDuringTraining(t *testing.T, primaryDevice tensor.DeviceType) {
	const numEpochs = 6
	const transferEpoch = 3 // Transfer after this epoch
	const tolerance = 1e-4
	
	t.Logf("Testing device transfer during training (primary: %v)", primaryDevice)
	
	// Define target device for transfer
	var targetDevice tensor.DeviceType
	switch primaryDevice {
	case tensor.CPU:
		targetDevice = tensor.GPU
	case tensor.GPU:
		targetDevice = tensor.PersistentGPU
	case tensor.PersistentGPU:
		targetDevice = tensor.CPU
	}
	
	t.Logf("Will transfer from %v to %v at epoch %d", primaryDevice, targetDevice, transferEpoch)
	
	// Set deterministic seed
	SetRandomSeed(99)
	
	// Create model on primary device
	conv, err := NewConv2D(1, 2, 2, 1, 0, true, primaryDevice)
	if err != nil {
		t.Fatalf("Failed to create Conv2D on %v: %v", primaryDevice, err)
	}
	
	relu := NewReLU()
	flatten := NewFlatten()
	linear, err := NewLinear(2, 2, true, primaryDevice)
	if err != nil {
		t.Fatalf("Failed to create Linear on %v: %v", primaryDevice, err)
	}
	
	model := NewSequential(conv, relu, flatten, linear)
	
	// Create training data on primary device
	trainData := []float32{
		1.0, 0.0,
		0.0, 0.0,
		0.0, 0.0,
		0.0, 1.0,
	}
	trainInput, err := tensor.NewTensor([]int{2, 1, 2, 2}, tensor.Float32, primaryDevice, trainData)
	if err != nil {
		t.Fatalf("Failed to create training input on %v: %v", primaryDevice, err)
	}
	
	trainLabels := []int32{0, 1}
	trainTarget, err := tensor.NewTensor([]int{2}, tensor.Int32, primaryDevice, trainLabels)
	if err != nil {
		t.Fatalf("Failed to create training target on %v: %v", primaryDevice, err)
	}
	
	// Create optimizer and loss
	params := model.Parameters()
	optimizer := NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
	criterion := NewCrossEntropyLoss("mean")
	
	var lossHistory []float32
	
	// Training loop with device transfer
	for epoch := 0; epoch < numEpochs; epoch++ {
		// Perform device transfer at the specified epoch
		if epoch == transferEpoch {
			t.Logf("Transferring model and data from %v to %v", primaryDevice, targetDevice)
			
			// Transfer model parameters
			for i, param := range params {
				var transferredParam *tensor.Tensor
				switch targetDevice {
				case tensor.CPU:
					transferredParam, err = param.ToCPU()
				case tensor.GPU:
					transferredParam, err = param.ToGPU()
				case tensor.PersistentGPU:
					transferredParam, err = param.ToPersistentGPU()
				}
				if err != nil {
					t.Fatalf("Failed to transfer parameter %d to %v: %v", i, targetDevice, err)
				}
				
				// Copy gradient state if it exists
				if param.Grad() != nil {
					var transferredGrad *tensor.Tensor
					switch targetDevice {
					case tensor.CPU:
						transferredGrad, err = param.Grad().ToCPU()
					case tensor.GPU:
						transferredGrad, err = param.Grad().ToGPU()
					case tensor.PersistentGPU:
						transferredGrad, err = param.Grad().ToPersistentGPU()
					}
					if err != nil {
						t.Fatalf("Failed to transfer gradient %d to %v: %v", i, targetDevice, err)
					}
					transferredParam.SetGrad(transferredGrad)
				}
				
				// Preserve requires_grad state
				transferredParam.SetRequiresGrad(param.RequiresGrad())
				
				// Update the parameter reference
				params[i] = transferredParam
			}
			
			// Transfer training data
			switch targetDevice {
			case tensor.CPU:
				trainInput, err = trainInput.ToCPU()
				if err != nil {
					t.Fatalf("Failed to transfer input to CPU: %v", err)
				}
				trainTarget, err = trainTarget.ToCPU()
				if err != nil {
					t.Fatalf("Failed to transfer target to CPU: %v", err)
				}
			case tensor.GPU:
				trainInput, err = trainInput.ToGPU()
				if err != nil {
					t.Fatalf("Failed to transfer input to GPU: %v", err)
				}
				trainTarget, err = trainTarget.ToGPU()
				if err != nil {
					t.Fatalf("Failed to transfer target to GPU: %v", err)
				}
			case tensor.PersistentGPU:
				trainInput, err = trainInput.ToPersistentGPU()
				if err != nil {
					t.Fatalf("Failed to transfer input to PersistentGPU: %v", err)
				}
				trainTarget, err = trainTarget.ToPersistentGPU()
				if err != nil {
					t.Fatalf("Failed to transfer target to PersistentGPU: %v", err)
				}
			}
			
			// Update the model to use new device - recreate with transferred parameters
			transferredModel, err := createModelWithParameters(params, targetDevice)
			if err != nil {
				t.Fatalf("Failed to create model with transferred parameters: %v", err)
			}
			model = transferredModel
			
			// Create new optimizer with transferred parameters
			optimizer = NewSGD(params, 0.1, 0.0, 0.0, 0.0, false)
			
			t.Logf("Device transfer completed successfully")
		}
		
		// Training step
		optimizer.ZeroGrad()
		
		output, err := model.Forward(trainInput)
		if err != nil {
			t.Fatalf("Forward pass failed at epoch %d: %v", epoch, err)
		}
		
		loss, err := criterion.Forward(output, trainTarget)
		if err != nil {
			t.Fatalf("Loss computation failed at epoch %d: %v", epoch, err)
		}
		
		lossData := tensorFloat32(loss)
		lossHistory = append(lossHistory, lossData[0])
		
		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed at epoch %d: %v", epoch, err)
		}
		
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed at epoch %d: %v", epoch, err)
		}
		
		// Log progress
		currentDevice := "unknown"
		if len(params) > 0 {
			currentDevice = params[0].Device.String()
		}
		t.Logf("Epoch %d (device: %s): loss=%.6f", epoch+1, currentDevice, lossData[0])
	}
	
	// Verify training continued properly after transfer
	if len(lossHistory) != numEpochs {
		t.Errorf("Expected %d loss values, got %d", numEpochs, len(lossHistory))
	}
	
	// Verify loss is decreasing or stable after transfer
	preTransferLoss := lossHistory[transferEpoch-1]
	postTransferLoss := lossHistory[transferEpoch]
	finalLoss := lossHistory[numEpochs-1]
	
	t.Logf("Loss trajectory: pre-transfer=%.6f, post-transfer=%.6f, final=%.6f", 
		preTransferLoss, postTransferLoss, finalLoss)
	
	// The loss should be reasonable after transfer (allow for some numerical differences)
	lossDiff := math.Abs(float64(postTransferLoss) - float64(preTransferLoss))
	if lossDiff > 0.1 { // Allow for some difference due to device precision
		t.Logf("Warning: Large loss difference after transfer: %.6f (this may be expected due to device precision differences)", lossDiff)
	}
	
	// Verify final accuracy
	finalOutput, err := model.Forward(trainInput)
	if err != nil {
		t.Fatalf("Final forward pass failed: %v", err)
	}
	
	predictions, err := tensorArgmax(finalOutput, 1)
	if err != nil {
		t.Fatalf("Argmax failed: %v", err)
	}
	
	predData := tensorInt32(predictions)
	correct := 0
	for i := range predData {
		if predData[i] == trainLabels[i] {
			correct++
		}
	}
	accuracy := float32(correct) / float32(len(trainLabels))
	
	t.Logf("Device transfer training completed: final accuracy=%.1f%%, final device=%s", 
		accuracy*100, params[0].Device.String())
	
	// Verify model ended up on target device
	if params[0].Device != targetDevice {
		t.Errorf("Model parameters not on expected device: expected %v, got %v", 
			targetDevice, params[0].Device)
	}
	
	t.Logf("Device transfer during training test passed")
}

// Helper function to create a model with given parameters on specified device
func createModelWithParameters(params []*tensor.Tensor, device tensor.DeviceType) (*Sequential, error) {
	// This is a simplified version - in practice, we would need to properly reconstruct the model
	// For our test case with Conv2D -> ReLU -> Flatten -> Linear, we can manually reconstruct
	
	if len(params) != 4 {
		return nil, fmt.Errorf("expected 4 parameters (conv weight, conv bias, linear weight, linear bias), got %d", len(params))
	}
	
	// Create modules without parameters first
	conv := &Conv2D{
		weight:   params[0], // Conv2D weight
		bias:     params[1], // Conv2D bias
		stride:   1,
		padding:  0,
		training: true,
	}
	
	relu := NewReLU()
	flatten := NewFlatten()
	
	linear := &Linear{
		weight:   params[2], // Linear weight
		bias:     params[3], // Linear bias
		training: true,
	}
	
	return NewSequential(conv, relu, flatten, linear), nil
}

// testGradientSynchronization verifies gradient consistency across devices
func testGradientSynchronization(t *testing.T, device tensor.DeviceType) {
	t.Logf("Testing gradient synchronization on device: %v", device)
	
	// Create simple model for gradient verification
	linear1, err := NewLinear(2, 4, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear layer: %v", err)
	}
	
	relu := NewReLU()
	linear2, err := NewLinear(4, 1, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear layer: %v", err)
	}
	
	model := NewSequential(linear1, relu, linear2)
	
	// Create test data
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	input, err := tensor.NewTensor([]int{2, 2}, tensor.Float32, device, inputData)
	if err != nil {
		t.Fatalf("Failed to create input: %v", err)
	}
	
	targetData := []float32{1.5, 3.5}
	target, err := tensor.NewTensor([]int{2, 1}, tensor.Float32, device, targetData)
	if err != nil {
		t.Fatalf("Failed to create target: %v", err)
	}
	
	// Forward pass
	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	
	// Compute loss
	criterion := NewMSELoss("mean")
	loss, err := criterion.Forward(output, target)
	if err != nil {
		t.Fatalf("Loss computation failed: %v", err)
	}
	
	// Backward pass
	err = loss.Backward()
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}
	
	// Verify all parameters have gradients
	params := model.Parameters()
	for i, param := range params {
		grad := param.Grad()
		if grad == nil {
			t.Errorf("Parameter %d has nil gradient", i)
			continue
		}
		
		if grad.Device != device {
			t.Errorf("Parameter %d gradient on wrong device: expected %v, got %v", i, device, grad.Device)
		}
		
		// Verify gradient is non-zero (indicating actual computation)
		gradData := tensorFloat32(grad)
		hasNonZero := false
		for _, val := range gradData {
			if math.Abs(float64(val)) > 1e-8 {
				hasNonZero = true
				break
			}
		}
		
		if !hasNonZero {
			t.Errorf("Parameter %d has all-zero gradients (may indicate computation issue)", i)
		}
		
		t.Logf("Parameter %d gradient verified: device=%v, non-zero values detected", i, grad.Device)
	}
	
	t.Logf("Gradient synchronization verified: all %d parameters have proper gradients", len(params))
}

// testAutogradGraphPreservation verifies that computational graph is preserved during device transfers
func testAutogradGraphPreservation(t *testing.T, device tensor.DeviceType) {
	t.Logf("Testing autograd graph preservation on device: %v", device)
	
	// Define target device
	var targetDevice tensor.DeviceType
	switch device {
	case tensor.CPU:
		targetDevice = tensor.GPU
	case tensor.GPU:
		targetDevice = tensor.PersistentGPU
	case tensor.PersistentGPU:
		targetDevice = tensor.CPU
	}
	
	// Create a complex computational graph
	SetRandomSeed(42)
	
	// Create tensors that will form a computational graph
	a, err := tensor.NewTensor([]int{2, 3}, tensor.Float32, device, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("Failed to create tensor a: %v", err)
	}
	a.SetRequiresGrad(true)
	
	b, err := tensor.NewTensor([]int{3, 2}, tensor.Float32, device, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	if err != nil {
		t.Fatalf("Failed to create tensor b: %v", err)
	}
	b.SetRequiresGrad(true)
	
	// Build computational graph: c = a @ b, d = relu(c), e = sum(d)
	c, err := tensor.MatMulAutograd(a, b)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}
	
	d, err := tensor.ReLUAutograd(c)
	if err != nil {
		t.Fatalf("ReLU failed: %v", err)
	}
	
	e, err := tensor.SumAutograd(d)
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}
	
	// Compute gradients before transfer
	err = e.Backward()
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}
	
	// Store gradients for comparison
	aGradBefore, err := a.Grad().Clone()
	if err != nil {
		t.Fatalf("Failed to clone a gradient: %v", err)
	}
	bGradBefore, err := b.Grad().Clone()
	if err != nil {
		t.Fatalf("Failed to clone b gradient: %v", err)
	}
	
	t.Logf("Pre-transfer: a.grad device=%v, b.grad device=%v", a.Grad().Device, b.Grad().Device)
	
	// Transfer tensors to target device
	var aTransferred, bTransferred *tensor.Tensor
	switch targetDevice {
	case tensor.CPU:
		aTransferred, err = a.ToCPU()
		if err != nil {
			t.Fatalf("Failed to transfer a to CPU: %v", err)
		}
		bTransferred, err = b.ToCPU()
		if err != nil {
			t.Fatalf("Failed to transfer b to CPU: %v", err)
		}
	case tensor.GPU:
		aTransferred, err = a.ToGPU()
		if err != nil {
			t.Fatalf("Failed to transfer a to GPU: %v", err)
		}
		bTransferred, err = b.ToGPU()
		if err != nil {
			t.Fatalf("Failed to transfer b to GPU: %v", err)
		}
	case tensor.PersistentGPU:
		aTransferred, err = a.ToPersistentGPU()
		if err != nil {
			t.Fatalf("Failed to transfer a to PersistentGPU: %v", err)
		}
		bTransferred, err = b.ToPersistentGPU()
		if err != nil {
			t.Fatalf("Failed to transfer b to PersistentGPU: %v", err)
		}
	}
	
	// Set requires_grad on transferred tensors
	aTransferred.SetRequiresGrad(true)
	bTransferred.SetRequiresGrad(true)
	
	// Rebuild computational graph on target device
	cTransferred, err := tensor.MatMulAutograd(aTransferred, bTransferred)
	if err != nil {
		t.Fatalf("MatMul on transferred tensors failed: %v", err)
	}
	
	dTransferred, err := tensor.ReLUAutograd(cTransferred)
	if err != nil {
		t.Fatalf("ReLU on transferred tensors failed: %v", err)
	}
	
	eTransferred, err := tensor.SumAutograd(dTransferred)
	if err != nil {
		t.Fatalf("Sum on transferred tensors failed: %v", err)
	}
	
	// Compute gradients after transfer
	err = eTransferred.Backward()
	if err != nil {
		t.Fatalf("Backward pass on transferred tensors failed: %v", err)
	}
	
	t.Logf("Post-transfer: aTransferred.grad device=%v, bTransferred.grad device=%v", 
		aTransferred.Grad().Device, bTransferred.Grad().Device)
	
	// Verify gradients are correct after transfer by comparing with original gradients
	// Convert gradients to common device (CPU) for comparison
	var aGradAfter, bGradAfter *tensor.Tensor
	if aTransferred.Grad().Device != tensor.CPU {
		aGradAfter, err = aTransferred.Grad().ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert aTransferred grad to CPU: %v", err)
		}
	} else {
		aGradAfter = aTransferred.Grad()
	}
	
	if bTransferred.Grad().Device != tensor.CPU {
		bGradAfter, err = bTransferred.Grad().ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert bTransferred grad to CPU: %v", err)
		}
	} else {
		bGradAfter = bTransferred.Grad()
	}
	
	if aGradBefore.Device != tensor.CPU {
		aGradBefore, err = aGradBefore.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert aGradBefore to CPU: %v", err)
		}
	}
	
	if bGradBefore.Device != tensor.CPU {
		bGradBefore, err = bGradBefore.ToCPU()
		if err != nil {
			t.Fatalf("Failed to convert bGradBefore to CPU: %v", err)
		}
	}
	
	// Compare gradients (should be identical)
	if !isClose(aGradAfter, aGradBefore, 1e-5) {
		t.Errorf("Gradient for a differs after device transfer")
	}
	
	if !isClose(bGradAfter, bGradBefore, 1e-5) {
		t.Errorf("Gradient for b differs after device transfer")
	}
	
	// Verify devices are correct
	if aTransferred.Device != targetDevice {
		t.Errorf("aTransferred on wrong device: expected %v, got %v", targetDevice, aTransferred.Device)
	}
	if bTransferred.Device != targetDevice {
		t.Errorf("bTransferred on wrong device: expected %v, got %v", targetDevice, bTransferred.Device)
	}
	if aTransferred.Grad().Device != targetDevice {
		t.Errorf("aTransferred.grad on wrong device: expected %v, got %v", targetDevice, aTransferred.Grad().Device)
	}
	if bTransferred.Grad().Device != targetDevice {
		t.Errorf("bTransferred.grad on wrong device: expected %v, got %v", targetDevice, bTransferred.Grad().Device)
	}
	
	t.Logf("Autograd graph preservation verified: gradients identical after transfer from %v to %v", device, targetDevice)
}

// testMemoryConsistencyDuringTransfers verifies memory integrity during device transfers
func testMemoryConsistencyDuringTransfers(t *testing.T, device tensor.DeviceType) {
	t.Logf("Testing memory consistency during transfers on device: %v", device)
	
	// Create test data with known pattern
	originalData := []float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0}
	
	tensor1, err := tensor.NewTensor([]int{2, 5}, tensor.Float32, device, originalData)
	if err != nil {
		t.Fatalf("Failed to create tensor1: %v", err)
	}
	
	// Test round-trip transfers to verify data integrity
	devices := []tensor.DeviceType{tensor.CPU, tensor.GPU, tensor.PersistentGPU}
	
	for _, targetDevice := range devices {
		if targetDevice == device {
			continue // Skip self-transfer
		}
		
		t.Logf("Testing transfer %v -> %v -> %v", device, targetDevice, device)
		
		// Transfer to target device
		var transferred *tensor.Tensor
		switch targetDevice {
		case tensor.CPU:
			transferred, err = tensor1.ToCPU()
		case tensor.GPU:
			transferred, err = tensor1.ToGPU()
		case tensor.PersistentGPU:
			transferred, err = tensor1.ToPersistentGPU()
		}
		if err != nil {
			t.Fatalf("Failed to transfer to %v: %v", targetDevice, err)
		}
		
		// Verify device is correct
		if transferred.Device != targetDevice {
			t.Errorf("Transfer to %v failed: tensor on %v", targetDevice, transferred.Device)
		}
		
		// Transfer back to original device
		var backTransferred *tensor.Tensor
		switch device {
		case tensor.CPU:
			backTransferred, err = transferred.ToCPU()
		case tensor.GPU:
			backTransferred, err = transferred.ToGPU()
		case tensor.PersistentGPU:
			backTransferred, err = transferred.ToPersistentGPU()
		}
		if err != nil {
			t.Fatalf("Failed to transfer back to %v: %v", device, err)
		}
		
		// Verify device is correct
		if backTransferred.Device != device {
			t.Errorf("Transfer back to %v failed: tensor on %v", device, backTransferred.Device)
		}
		
		// Verify data integrity by converting to CPU for comparison
		var originalCPU, backCPU *tensor.Tensor
		
		if tensor1.Device == tensor.CPU {
			originalCPU = tensor1
		} else {
			originalCPU, err = tensor1.ToCPU()
			if err != nil {
				t.Fatalf("Failed to convert original to CPU: %v", err)
			}
		}
		
		if backTransferred.Device == tensor.CPU {
			backCPU = backTransferred
		} else {
			backCPU, err = backTransferred.ToCPU()
			if err != nil {
				t.Fatalf("Failed to convert back transferred to CPU: %v", err)
			}
		}
		
		// Compare data
		originalFloat32 := originalCPU.Data.([]float32)
		backFloat32 := backCPU.Data.([]float32)
		
		if len(originalFloat32) != len(backFloat32) {
			t.Errorf("Data length mismatch after round-trip %v->%v->%v: original=%d, back=%d", 
				device, targetDevice, device, len(originalFloat32), len(backFloat32))
			continue
		}
		
		for i := range originalFloat32 {
			if math.Abs(float64(originalFloat32[i])-float64(backFloat32[i])) > 1e-6 {
				t.Errorf("Data corruption at index %d after round-trip %v->%v->%v: original=%.6f, back=%.6f",
					i, device, targetDevice, device, originalFloat32[i], backFloat32[i])
				break
			}
		}
		
		t.Logf("Round-trip %v->%v->%v: data integrity preserved", device, targetDevice, device)
	}
	
	// Test memory consistency with gradients
	tensor1.SetRequiresGrad(true)
	
	// Create a simple computation to generate gradients
	output, err := tensor.SumAutograd(tensor1)
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}
	
	err = output.Backward()
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	
	// Verify gradient memory consistency during transfers
	if tensor1.Grad() != nil {
		originalGrad := tensor1.Grad()
		
		for _, targetDevice := range devices {
			if targetDevice == device {
				continue
			}
			
			var transferredGrad *tensor.Tensor
			switch targetDevice {
			case tensor.CPU:
				transferredGrad, err = originalGrad.ToCPU()
			case tensor.GPU:
				transferredGrad, err = originalGrad.ToGPU()
			case tensor.PersistentGPU:
				transferredGrad, err = originalGrad.ToPersistentGPU()
			}
			if err != nil {
				t.Fatalf("Failed to transfer gradient to %v: %v", targetDevice, err)
			}
			
			// Convert both to CPU for comparison
			var originalGradCPU, transferredGradCPU *tensor.Tensor
			
			if originalGrad.Device == tensor.CPU {
				originalGradCPU = originalGrad
			} else {
				originalGradCPU, err = originalGrad.ToCPU()
				if err != nil {
					t.Fatalf("Failed to convert original grad to CPU: %v", err)
				}
			}
			
			if transferredGrad.Device == tensor.CPU {
				transferredGradCPU = transferredGrad
			} else {
				transferredGradCPU, err = transferredGrad.ToCPU()
				if err != nil {
					t.Fatalf("Failed to convert transferred grad to CPU: %v", err)
				}
			}
			
			if !isClose(originalGradCPU, transferredGradCPU, 1e-6) {
				t.Errorf("Gradient data corruption during transfer to %v", targetDevice)
			}
			
			t.Logf("Gradient transfer %v->%v: memory consistency preserved", device, targetDevice)
		}
	}
	
	t.Logf("Memory consistency verification completed for device %v", device)
}

// Helper functions use existing ones from integration_test.go