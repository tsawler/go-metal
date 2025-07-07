package checkpoints

import (
	"os"
	"testing"
	"time"

	"github.com/tsawler/go-metal/layers"
)

func TestCheckpointJSONSaveLoad(t *testing.T) {
	// Create a simple model spec for testing
	inputShape := []int{1, 28, 28, 1} // MNIST-like shape
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(128, true, "dense1").
		AddReLU("relu1").
		AddDense(10, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	// Create a test checkpoint
	checkpoint := &Checkpoint{
		ModelSpec: model,
		Weights: []WeightTensor{
			{
				Name:  "dense1.weight",
				Shape: []int{784, 128},
				Data:  make([]float32, 784*128),
				Layer: "dense1",
				Type:  "weight",
			},
			{
				Name:  "dense1.bias",
				Shape: []int{128},
				Data:  make([]float32, 128),
				Layer: "dense1",
				Type:  "bias",
			},
		},
		TrainingState: TrainingState{
			Epoch:        10,
			Step:         1000,
			LearningRate: 0.001,
			BestLoss:     0.5,
			BestAccuracy: 0.85,
			TotalSteps:   1000,
		},
		Metadata: CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			CreatedAt:   time.Now(),
			Description: "Test checkpoint",
			Tags:        []string{"test", "mnist"},
		},
	}

	// Fill test data
	for i := range checkpoint.Weights[0].Data {
		checkpoint.Weights[0].Data[i] = float32(i%100) * 0.01
	}
	for i := range checkpoint.Weights[1].Data {
		checkpoint.Weights[1].Data[i] = float32(i%10) * 0.1
	}

	// Test JSON save
	saver := NewCheckpointSaver(FormatJSON)
	testFile := "test_checkpoint.json"
	
	// Clean up
	defer os.Remove(testFile)
	
	err = saver.SaveCheckpoint(checkpoint, testFile)
	if err != nil {
		t.Fatalf("Failed to save JSON checkpoint: %v", err)
	}

	// Test JSON load
	loadedCheckpoint, err := saver.LoadCheckpoint(testFile)
	if err != nil {
		t.Fatalf("Failed to load JSON checkpoint: %v", err)
	}

	// Verify loaded data
	if loadedCheckpoint.TrainingState.Epoch != checkpoint.TrainingState.Epoch {
		t.Errorf("Epoch mismatch: expected %d, got %d", 
			checkpoint.TrainingState.Epoch, loadedCheckpoint.TrainingState.Epoch)
	}

	if len(loadedCheckpoint.Weights) != len(checkpoint.Weights) {
		t.Errorf("Weight count mismatch: expected %d, got %d", 
			len(checkpoint.Weights), len(loadedCheckpoint.Weights))
	}

	// Check first weight tensor
	if len(loadedCheckpoint.Weights) > 0 {
		originalWeight := checkpoint.Weights[0]
		loadedWeight := loadedCheckpoint.Weights[0]
		
		if originalWeight.Name != loadedWeight.Name {
			t.Errorf("Weight name mismatch: expected %s, got %s", 
				originalWeight.Name, loadedWeight.Name)
		}
		
		if len(originalWeight.Data) != len(loadedWeight.Data) {
			t.Errorf("Weight data length mismatch: expected %d, got %d", 
				len(originalWeight.Data), len(loadedWeight.Data))
		}
		
		// Check first few values
		for i := 0; i < 10 && i < len(originalWeight.Data); i++ {
			if originalWeight.Data[i] != loadedWeight.Data[i] {
				t.Errorf("Weight data mismatch at index %d: expected %f, got %f", 
					i, originalWeight.Data[i], loadedWeight.Data[i])
			}
		}
	}

	t.Logf("JSON checkpoint test passed successfully!")
}

func TestCheckpointONNXExport(t *testing.T) {
	// Create a simple model spec for testing
	inputShape := []int{1, 28, 28, 1} // MNIST-like shape
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "dense1").
		AddReLU("relu1").
		AddDense(10, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	// Create a test checkpoint
	checkpoint := &Checkpoint{
		ModelSpec: model,
		Weights: []WeightTensor{
			{
				Name:  "dense1.weight",
				Shape: []int{784, 64},
				Data:  make([]float32, 784*64),
				Layer: "dense1",
				Type:  "weight",
			},
			{
				Name:  "output.weight",
				Shape: []int{64, 10},
				Data:  make([]float32, 64*10),
				Layer: "output",
				Type:  "weight",
			},
		},
		Metadata: CheckpointMetadata{
			Version:   "1.0.0",
			Framework: "go-metal",
			CreatedAt: time.Now(),
		},
	}

	// Fill with test data
	for i := range checkpoint.Weights[0].Data {
		checkpoint.Weights[0].Data[i] = float32(i%100) * 0.01
	}
	for i := range checkpoint.Weights[1].Data {
		checkpoint.Weights[1].Data[i] = float32(i%10) * 0.1
	}

	// Test ONNX export
	saver := NewCheckpointSaver(FormatONNX)
	testFile := "test_model.onnx"
	
	// Clean up
	defer os.Remove(testFile)
	
	err = saver.SaveCheckpoint(checkpoint, testFile)
	if err != nil {
		t.Fatalf("Failed to export ONNX model: %v", err)
	}

	// Check that file was created and has content
	fileInfo, err := os.Stat(testFile)
	if err != nil {
		t.Fatalf("ONNX file was not created: %v", err)
	}

	if fileInfo.Size() == 0 {
		t.Errorf("ONNX file is empty")
	}

	t.Logf("ONNX export test passed! File size: %d bytes", fileInfo.Size())
}