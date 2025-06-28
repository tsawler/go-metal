package training

import (
	"fmt"
	"runtime"
	"testing"
	"time"
	"math"

	"github.com/tsawler/go-metal/tensor"
)

// TestMemoryLeakDetection verifies no memory leaks during extended training
func TestMemoryLeakDetection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory leak test in short mode")
	}

	devices := []tensor.DeviceType{tensor.CPU}
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Record initial memory
			runtime.GC()
			runtime.GC() // Double GC to ensure cleanup
			var initialMem runtime.MemStats
			runtime.ReadMemStats(&initialMem)
			initialAllocMB := float64(initialMem.Alloc) / 1024 / 1024

			t.Logf("Initial memory usage: %.2f MB", initialAllocMB)

			// Run training for many iterations
			runExtendedTraining(t, device, 100) // 100 epochs

			// Force garbage collection and measure final memory
			runtime.GC()
			runtime.GC()
			time.Sleep(100 * time.Millisecond) // Allow GC to complete
			
			var finalMem runtime.MemStats
			runtime.ReadMemStats(&finalMem)
			finalAllocMB := float64(finalMem.Alloc) / 1024 / 1024

			memoryIncreaseMB := finalAllocMB - initialAllocMB
			t.Logf("Final memory usage: %.2f MB (increase: %.2f MB)", finalAllocMB, memoryIncreaseMB)

			// Check for excessive memory growth (allow 50MB for normal variation)
			maxAllowedIncreaseMB := 50.0
			if memoryIncreaseMB > maxAllowedIncreaseMB {
				t.Errorf("Excessive memory growth detected: %.2f MB > %.2f MB allowed", 
					memoryIncreaseMB, maxAllowedIncreaseMB)
			}

			// Check total allocations to detect allocation patterns
			totalAllocsMB := float64(finalMem.TotalAlloc-initialMem.TotalAlloc) / 1024 / 1024
			allocsPerEpoch := totalAllocsMB / 100
			t.Logf("Total allocations during training: %.2f MB (%.2f MB/epoch)", 
				totalAllocsMB, allocsPerEpoch)

			// Warn if allocations per epoch are excessive
			if allocsPerEpoch > 10.0 {
				t.Logf("WARNING: High allocation rate detected: %.2f MB/epoch", allocsPerEpoch)
			}
		})
	}
}

// runExtendedTraining runs a training loop for memory leak detection
func runExtendedTraining(t *testing.T, device tensor.DeviceType, epochs int) {
	// Set deterministic seed
	SetRandomSeed(42)

	// Create a CNN model
	conv1, err := NewConv2D(3, 16, 3, 1, 1, true, device)
	if err != nil {
		t.Fatalf("Failed to create Conv2D: %v", err)
	}

	relu1 := NewReLU()
	pool1 := NewMaxPool2D(2, 2, 0)
	
	conv2, err := NewConv2D(16, 32, 3, 1, 1, true, device)
	if err != nil {
		t.Fatalf("Failed to create Conv2D: %v", err)
	}

	relu2 := NewReLU()
	pool2 := NewMaxPool2D(2, 2, 0)
	flatten := NewFlatten()
	
	// After two 2x2 pools on 8x8 input: 8->4->2, so 32*2*2=128
	linear, err := NewLinear(128, 10, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear: %v", err)
	}

	model := NewSequential(conv1, relu1, pool1, conv2, relu2, pool2, flatten, linear)

	// Create synthetic data
	batchSize := 16
	inputData := make([]float32, batchSize*3*8*8)
	for i := range inputData {
		inputData[i] = float32(i%256) / 255.0
	}

	targetData := make([]int32, batchSize)
	for i := range targetData {
		targetData[i] = int32(i % 10)
	}

	// Create optimizer and loss
	params := model.Parameters()
	optimizer := NewSGD(params, 0.01, 0.9, 0.0, 0.0, false)
	criterion := NewCrossEntropyLoss("mean")

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		// Create tensors (these should be properly garbage collected)
		input, err := tensor.NewTensor([]int{batchSize, 3, 8, 8}, tensor.Float32, device, inputData)
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		target, err := tensor.NewTensor([]int{batchSize}, tensor.Int32, device, targetData)
		if err != nil {
			t.Fatalf("Failed to create target tensor: %v", err)
		}

		// Forward pass
		optimizer.ZeroGrad()
		
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward pass failed at epoch %d: %v", epoch, err)
		}

		loss, err := criterion.Forward(output, target)
		if err != nil {
			t.Fatalf("Loss computation failed at epoch %d: %v", epoch, err)
		}

		// Backward pass
		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed at epoch %d: %v", epoch, err)
		}

		// Optimizer step
		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed at epoch %d: %v", epoch, err)
		}

		// Log progress every 20 epochs
		if epoch%20 == 0 {
			lossVal := performanceTensorFloat32(loss)[0]
			t.Logf("Epoch %d: loss=%.6f", epoch, lossVal)
		}

		// Explicitly nil references to help GC (shouldn't be necessary but good practice)
		input = nil
		target = nil
		output = nil
		loss = nil
	}
}

// TestNumericalStability tests training with various hyperparameters
func TestNumericalStability(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	// Test various learning rates
	learningRates := []float32{1e-5, 1e-3, 0.1, 1.0, 10.0}
	
	// Test various momentum values
	momentums := []float32{0.0, 0.5, 0.9, 0.99}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			for _, lr := range learningRates {
				for _, momentum := range momentums {
					testName := fmt.Sprintf("LR_%.0e_Momentum_%.2f", lr, momentum)
					t.Run(testName, func(t *testing.T) {
						testNumericalStabilityWithParams(t, device, lr, momentum)
					})
				}
			}
		})
	}
}

func testNumericalStabilityWithParams(t *testing.T, device tensor.DeviceType, lr float32, momentum float32) {
	// Set deterministic seed
	SetRandomSeed(123)

	// Create simple model
	linear1, err := NewLinear(10, 20, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear: %v", err)
	}

	relu := NewReLU()
	
	linear2, err := NewLinear(20, 5, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear: %v", err)
	}

	model := NewSequential(linear1, relu, linear2)

	// Create data
	batchSize := 8
	inputData := make([]float32, batchSize*10)
	for i := range inputData {
		inputData[i] = float32(i%10) * 0.1
	}

	targetData := make([]int32, batchSize)
	for i := range targetData {
		targetData[i] = int32(i % 5)
	}

	input, err := tensor.NewTensor([]int{batchSize, 10}, tensor.Float32, device, inputData)
	if err != nil {
		t.Fatalf("Failed to create input: %v", err)
	}

	target, err := tensor.NewTensor([]int{batchSize}, tensor.Int32, device, targetData)
	if err != nil {
		t.Fatalf("Failed to create target: %v", err)
	}

	// Create optimizer and loss
	params := model.Parameters()
	optimizer := NewSGD(params, float64(lr), float64(momentum), 0.0, 0.0, false)
	criterion := NewCrossEntropyLoss("mean")

	// Track losses to check for NaN/Inf
	losses := make([]float32, 10)
	hasNaN := false
	hasInf := false

	// Training loop
	for epoch := 0; epoch < 10; epoch++ {
		optimizer.ZeroGrad()

		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}

		loss, err := criterion.Forward(output, target)
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}

		lossVal := performanceTensorFloat32(loss)[0]
		losses[epoch] = lossVal

		// Check for numerical issues
		if math.IsNaN(float64(lossVal)) {
			hasNaN = true
			t.Logf("NaN detected at epoch %d with lr=%.0e, momentum=%.2f", epoch, lr, momentum)
			break
		}
		if math.IsInf(float64(lossVal), 0) {
			hasInf = true
			t.Logf("Inf detected at epoch %d with lr=%.0e, momentum=%.2f", epoch, lr, momentum)
			break
		}

		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}

		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed: %v", err)
		}
	}

	// Analyze results
	if hasNaN || hasInf {
		if lr >= 10.0 {
			t.Logf("Expected instability with high learning rate %.0e", lr)
		} else {
			t.Errorf("Unexpected numerical instability with lr=%.0e, momentum=%.2f", lr, momentum)
		}
	} else {
		// Check if learning is happening (loss should change)
		firstLoss := losses[0]
		lastLoss := losses[9]
		
		if lr >= 0.001 && lr <= 1.0 {
			// With reasonable learning rates, we expect some change
			if math.Abs(float64(lastLoss-firstLoss)) < 1e-6 {
				t.Logf("Warning: No learning detected with lr=%.0e, momentum=%.2f", lr, momentum)
			}
		}
		
		t.Logf("Training stable: initial loss=%.6f, final loss=%.6f", firstLoss, lastLoss)
	}

	// Check parameter magnitudes for explosion
	for i, param := range params {
		paramData := performanceTensorFloat32(param)
		maxVal := float32(0)
		for _, val := range paramData {
			if absVal := float32(math.Abs(float64(val))); absVal > maxVal {
				maxVal = absVal
			}
		}
		
		if maxVal > 1e6 {
			t.Errorf("Parameter %d explosion detected: max magnitude %.0e", i, maxVal)
		}
	}
}

// TestBatchSizeScaling verifies correct behavior with various batch sizes
func TestBatchSizeScaling(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	// Test various batch sizes including edge cases
	batchSizes := []int{1, 2, 4, 8, 16, 32, 64, 128}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Store loss values for different batch sizes
			lossValues := make(map[int]float32)
			
			for _, batchSize := range batchSizes {
				t.Run(fmt.Sprintf("BatchSize_%d", batchSize), func(t *testing.T) {
					loss := testBatchSizeScaling(t, device, batchSize)
					lossValues[batchSize] = loss
				})
			}

			// Verify that loss computation is consistent across batch sizes
			// When using mean reduction, initial loss should be similar
			baseLoss := lossValues[batchSizes[0]]
			for _, batchSize := range batchSizes[1:] {
				loss := lossValues[batchSize]
				relDiff := math.Abs(float64(loss-baseLoss)) / float64(baseLoss)
				
				// Allow 10% variation due to different random initialization effects
				if relDiff > 0.1 {
					t.Logf("Warning: Large loss variation between batch sizes: "+
						"batch_size=1 loss=%.6f, batch_size=%d loss=%.6f (%.1f%% diff)",
						baseLoss, batchSize, loss, relDiff*100)
				}
			}
		})
	}
}

func testBatchSizeScaling(t *testing.T, device tensor.DeviceType, batchSize int) float32 {
	// Set deterministic seed
	SetRandomSeed(456)

	// Create model
	conv, err := NewConv2D(1, 4, 3, 1, 1, true, device)
	if err != nil {
		t.Fatalf("Failed to create Conv2D: %v", err)
	}

	relu := NewReLU()
	pool := NewMaxPool2D(2, 2, 0)
	flatten := NewFlatten()
	
	// 4x4 input -> 4x4 after conv -> 2x2 after pool -> 4*2*2=16
	linear, err := NewLinear(16, 2, true, device)
	if err != nil {
		t.Fatalf("Failed to create Linear: %v", err)
	}

	model := NewSequential(conv, relu, pool, flatten, linear)

	// Create data for the batch size
	inputData := make([]float32, batchSize*1*4*4)
	for i := range inputData {
		inputData[i] = float32(i%16) / 16.0
	}

	targetData := make([]int32, batchSize)
	for i := range targetData {
		targetData[i] = int32(i % 2)
	}

	input, err := tensor.NewTensor([]int{batchSize, 1, 4, 4}, tensor.Float32, device, inputData)
	if err != nil {
		t.Fatalf("Failed to create input: %v", err)
	}

	target, err := tensor.NewTensor([]int{batchSize}, tensor.Int32, device, targetData)
	if err != nil {
		t.Fatalf("Failed to create target: %v", err)
	}

	// Create optimizer and loss
	params := model.Parameters()
	optimizer := NewSGD(params, 0.01, 0.0, 0.0, 0.0, false)
	criterion := NewCrossEntropyLoss("mean")

	// Run one epoch and return initial loss
	optimizer.ZeroGrad()

	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed with batch_size=%d: %v", batchSize, err)
	}

	// Verify output shape
	expectedShape := []int{batchSize, 2}
	if !performanceEqualShapes(output.Shape, expectedShape) {
		t.Errorf("Output shape mismatch for batch_size=%d: got %v, expected %v", 
			batchSize, output.Shape, expectedShape)
	}

	loss, err := criterion.Forward(output, target)
	if err != nil {
		t.Fatalf("Loss computation failed with batch_size=%d: %v", batchSize, err)
	}

	lossVal := performanceTensorFloat32(loss)[0]

	// Run backward to ensure gradient computation works
	err = loss.Backward()
	if err != nil {
		t.Fatalf("Backward pass failed with batch_size=%d: %v", batchSize, err)
	}

	// Verify all parameters have gradients
	for i, param := range params {
		if param.Grad() == nil {
			t.Errorf("Parameter %d has nil gradient with batch_size=%d", i, batchSize)
		}
	}

	err = optimizer.Step()
	if err != nil {
		t.Fatalf("Optimizer step failed with batch_size=%d: %v", batchSize, err)
	}

	t.Logf("Batch size %d: initial loss=%.6f", batchSize, lossVal)
	
	// Run a few more epochs to check stability
	for epoch := 1; epoch < 5; epoch++ {
		optimizer.ZeroGrad()
		
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward pass failed at epoch %d with batch_size=%d: %v", 
				epoch, batchSize, err)
		}

		loss, err := criterion.Forward(output, target)
		if err != nil {
			t.Fatalf("Loss computation failed at epoch %d with batch_size=%d: %v", 
				epoch, batchSize, err)
		}

		err = loss.Backward()
		if err != nil {
			t.Fatalf("Backward pass failed at epoch %d with batch_size=%d: %v", 
				epoch, batchSize, err)
		}

		err = optimizer.Step()
		if err != nil {
			t.Fatalf("Optimizer step failed at epoch %d with batch_size=%d: %v", 
				epoch, batchSize, err)
		}
	}

	return lossVal
}

// TestGradientAccumulation verifies gradient behavior with small batch sizes
func TestGradientAccumulation(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Set deterministic seed
			SetRandomSeed(789)

			// Create simple model
			linear, err := NewLinear(5, 3, true, device)
			if err != nil {
				t.Fatalf("Failed to create Linear: %v", err)
			}

			// Create small batches
			batch1Data := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
			batch2Data := []float32{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

			batch1, err := tensor.NewTensor([]int{2, 5}, tensor.Float32, device, batch1Data)
			if err != nil {
				t.Fatalf("Failed to create batch1: %v", err)
			}

			batch2, err := tensor.NewTensor([]int{2, 5}, tensor.Float32, device, batch2Data)
			if err != nil {
				t.Fatalf("Failed to create batch2: %v", err)
			}

			target1, err := tensor.NewTensor([]int{2}, tensor.Int32, device, []int32{0, 1})
			if err != nil {
				t.Fatalf("Failed to create target1: %v", err)
			}

			target2, err := tensor.NewTensor([]int{2}, tensor.Int32, device, []int32{2, 0})
			if err != nil {
				t.Fatalf("Failed to create target2: %v", err)
			}

			criterion := NewCrossEntropyLoss("mean")

			// Process batch 1
			output1, err := linear.Forward(batch1)
			if err != nil {
				t.Fatalf("Forward pass 1 failed: %v", err)
			}

			loss1, err := criterion.Forward(output1, target1)
			if err != nil {
				t.Fatalf("Loss computation 1 failed: %v", err)
			}

			err = loss1.Backward()
			if err != nil {
				t.Fatalf("Backward pass 1 failed: %v", err)
			}

			// Store gradients after first batch
			weightGrad1, err := linear.weight.Grad().Clone()
			if err != nil {
				t.Fatalf("Failed to clone weight gradient: %v", err)
			}

			var biasGrad1 *tensor.Tensor
			if linear.bias != nil {
				biasGrad1, err = linear.bias.Grad().Clone()
				if err != nil {
					t.Fatalf("Failed to clone bias gradient: %v", err)
				}
			}
			_ = biasGrad1 // Mark as used

			// Process batch 2 (without zeroing gradients - accumulation)
			output2, err := linear.Forward(batch2)
			if err != nil {
				t.Fatalf("Forward pass 2 failed: %v", err)
			}

			loss2, err := criterion.Forward(output2, target2)
			if err != nil {
				t.Fatalf("Loss computation 2 failed: %v", err)
			}

			err = loss2.Backward()
			if err != nil {
				t.Fatalf("Backward pass 2 failed: %v", err)
			}

			// Verify gradients accumulated (should be different from batch 1 alone)
			weightGrad2 := linear.weight.Grad()
			weightData1 := performanceTensorFloat32(weightGrad1)
			weightData2 := performanceTensorFloat32(weightGrad2)

			accumulated := false
			for i := range weightData1 {
				if math.Abs(float64(weightData1[i]-weightData2[i])) > 1e-6 {
					accumulated = true
					break
				}
			}

			if !accumulated {
				t.Errorf("Gradients did not accumulate properly between batches")
			}

			t.Logf("Gradient accumulation verified on device %v", device)
		})
	}
}

// TestExtremeBatchSizes tests edge cases for batch size handling
func TestExtremeBatchSizes(t *testing.T) {
	devices := []tensor.DeviceType{tensor.CPU}
	if tensor.IsGPUAvailable() {
		devices = append(devices, tensor.GPU, tensor.PersistentGPU)
	}

	for _, device := range devices {
		t.Run(fmt.Sprintf("Device_%v", device), func(t *testing.T) {
			// Test batch size 1 (edge case for batch normalization, etc.)
			t.Run("BatchSize1", func(t *testing.T) {
				SetRandomSeed(100)
				
				linear, err := NewLinear(10, 5, true, device)
				if err != nil {
					t.Fatalf("Failed to create Linear: %v", err)
				}

				input, err := tensor.NewTensor([]int{1, 10}, tensor.Float32, device, 
					[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
				if err != nil {
					t.Fatalf("Failed to create input: %v", err)
				}

				output, err := linear.Forward(input)
				if err != nil {
					t.Fatalf("Forward pass failed with batch_size=1: %v", err)
				}

				if output.Shape[0] != 1 || output.Shape[1] != 5 {
					t.Errorf("Unexpected output shape: %v", output.Shape)
				}

				// Test backward pass
				sum, err := tensor.SumAutograd(output)
				if err != nil {
					t.Fatalf("Sum failed: %v", err)
				}

				err = sum.Backward()
				if err != nil {
					t.Fatalf("Backward pass failed with batch_size=1: %v", err)
				}

				if linear.weight.Grad() == nil {
					t.Errorf("Weight gradient is nil after backward pass")
				}
			})

			// Test large batch size (memory stress test)
			t.Run("LargeBatchSize", func(t *testing.T) {
				if testing.Short() {
					t.Skip("Skipping large batch test in short mode")
				}

				SetRandomSeed(101)
				
				conv, err := NewConv2D(3, 8, 3, 1, 1, true, device)
				if err != nil {
					t.Fatalf("Failed to create Conv2D: %v", err)
				}

				// Create large batch
				batchSize := 256
				inputData := make([]float32, batchSize*3*16*16)
				for i := range inputData {
					inputData[i] = float32(i%256) / 255.0
				}

				input, err := tensor.NewTensor([]int{batchSize, 3, 16, 16}, tensor.Float32, device, inputData)
				if err != nil {
					t.Fatalf("Failed to create large batch input: %v", err)
				}

				output, err := conv.Forward(input)
				if err != nil {
					t.Fatalf("Forward pass failed with large batch: %v", err)
				}

				expectedShape := []int{batchSize, 8, 16, 16}
				if !performanceEqualShapes(output.Shape, expectedShape) {
					t.Errorf("Output shape mismatch: got %v, expected %v", 
						output.Shape, expectedShape)
				}

				t.Logf("Large batch test passed with batch_size=%d", batchSize)
			})
		})
	}
}

// Helper functions - reuse existing ones to avoid redeclaration

func performanceEqualShapes(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func performanceTensorFloat32(t *tensor.Tensor) []float32 {
	if t.Device != tensor.CPU {
		cpuTensor, _ := t.ToCPU()
		return cpuTensor.Data.([]float32)
	}
	return t.Data.([]float32)
}