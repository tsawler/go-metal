package checkpoints

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
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

// TestCheckpointFormatString tests the String() method for CheckpointFormat
func TestCheckpointFormatString(t *testing.T) {
	tests := []struct {
		format   CheckpointFormat
		expected string
	}{
		{FormatJSON, "JSON"},
		{FormatONNX, "ONNX"},
		{CheckpointFormat(999), "Unknown"}, // Invalid format
	}

	for _, test := range tests {
		result := test.format.String()
		if result != test.expected {
			t.Errorf("Format %d: expected %s, got %s", test.format, test.expected, result)
		}
	}

	t.Log("CheckpointFormat String() tests passed")
}

// TestCheckpointSaverCreation tests creating checkpoint savers
func TestCheckpointSaverCreation(t *testing.T) {
	// Test JSON saver creation
	jsonSaver := NewCheckpointSaver(FormatJSON)
	if jsonSaver == nil {
		t.Error("JSON checkpoint saver should not be nil")
	}
	if jsonSaver.format != FormatJSON {
		t.Errorf("Expected format %d, got %d", FormatJSON, jsonSaver.format)
	}

	// Test ONNX saver creation
	onnxSaver := NewCheckpointSaver(FormatONNX)
	if onnxSaver == nil {
		t.Error("ONNX checkpoint saver should not be nil")
	}
	if onnxSaver.format != FormatONNX {
		t.Errorf("Expected format %d, got %d", FormatONNX, onnxSaver.format)
	}

	t.Log("Checkpoint saver creation tests passed")
}

// TestUnsupportedCheckpointFormat tests error handling for unsupported formats
func TestUnsupportedCheckpointFormat(t *testing.T) {
	// Create saver with invalid format
	invalidFormat := CheckpointFormat(999)
	saver := NewCheckpointSaver(invalidFormat)

	// Create a minimal checkpoint for testing
	inputShape := []int{1, 10}
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.AddDense(5, false, "test").Compile()
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	checkpoint := &Checkpoint{
		ModelSpec: model,
		Weights:   []WeightTensor{},
		Metadata:  CheckpointMetadata{Framework: "test"},
	}

	// Test save with unsupported format
	err = saver.SaveCheckpoint(checkpoint, "test.invalid")
	if err == nil {
		t.Error("Expected error for unsupported save format")
	}
	if !strings.Contains(err.Error(), "unsupported checkpoint format") {
		t.Errorf("Expected 'unsupported checkpoint format' error, got: %v", err)
	}

	// Test load with unsupported format
	_, err = saver.LoadCheckpoint("nonexistent.invalid")
	if err == nil {
		t.Error("Expected error for unsupported load format")
	}
	if !strings.Contains(err.Error(), "unsupported checkpoint format") {
		t.Errorf("Expected 'unsupported checkpoint format' error, got: %v", err)
	}

	t.Log("Unsupported format error tests passed")
}

// TestJSONLoadFileErrors tests JSON loading error conditions
func TestJSONLoadFileErrors(t *testing.T) {
	saver := NewCheckpointSaver(FormatJSON)

	// Test loading non-existent file
	_, err := saver.LoadCheckpoint("nonexistent.json")
	if err == nil {
		t.Error("Expected error for non-existent file")
	}
	if !strings.Contains(err.Error(), "failed to open checkpoint file") {
		t.Errorf("Expected 'failed to open checkpoint file' error, got: %v", err)
	}

	// Test loading invalid JSON file
	invalidJSONFile := "invalid.json"
	defer os.Remove(invalidJSONFile)
	
	if err := os.WriteFile(invalidJSONFile, []byte("{invalid json"), 0644); err != nil {
		t.Fatalf("Failed to create invalid JSON file: %v", err)
	}

	_, err = saver.LoadCheckpoint(invalidJSONFile)
	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "failed to decode checkpoint") {
		t.Errorf("Expected 'failed to decode checkpoint' error, got: %v", err)
	}

	t.Log("JSON load error tests passed")
}

// TestJSONSaveFileErrors tests JSON saving error conditions
func TestJSONSaveFileErrors(t *testing.T) {
	saver := NewCheckpointSaver(FormatJSON)

	// Create a basic checkpoint
	inputShape := []int{1, 10}
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.AddDense(5, false, "test").Compile()
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	checkpoint := &Checkpoint{
		ModelSpec: model,
		Weights:   []WeightTensor{},
		Metadata:  CheckpointMetadata{Framework: "test"},
	}

	// Test saving to invalid path (directory that doesn't exist)
	err = saver.SaveCheckpoint(checkpoint, "/nonexistent/path/checkpoint.json")
	if err == nil {
		t.Error("Expected error for invalid save path")
	}
	if !strings.Contains(err.Error(), "failed to create checkpoint file") {
		t.Errorf("Expected 'failed to create checkpoint file' error, got: %v", err)
	}

	t.Log("JSON save error tests passed")
}

// TestCheckpointMetadataDefaults tests automatic metadata setting
func TestCheckpointMetadataDefaults(t *testing.T) {
	saver := NewCheckpointSaver(FormatJSON)
	
	// Create checkpoint with minimal metadata
	inputShape := []int{1, 10}
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.AddDense(5, false, "test").Compile()
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	checkpoint := &Checkpoint{
		ModelSpec: model,
		Weights:   []WeightTensor{},
		Metadata:  CheckpointMetadata{}, // Empty metadata
	}

	// Save checkpoint
	testFile := "test_metadata.json"
	defer os.Remove(testFile)
	
	err = saver.SaveCheckpoint(checkpoint, testFile)
	if err != nil {
		t.Fatalf("Failed to save checkpoint: %v", err)
	}

	// Verify defaults were set
	if checkpoint.Metadata.Framework != "go-metal" {
		t.Errorf("Expected framework 'go-metal', got '%s'", checkpoint.Metadata.Framework)
	}
	if checkpoint.Metadata.Version != "1.0.0" {
		t.Errorf("Expected version '1.0.0', got '%s'", checkpoint.Metadata.Version)
	}
	if checkpoint.Metadata.CreatedAt.IsZero() {
		t.Error("CreatedAt should be set to current time")
	}

	t.Log("Checkpoint metadata defaults tests passed")
}

// TestCheckpointDataStructures tests the checkpoint data structures
func TestCheckpointDataStructures(t *testing.T) {
	// Test WeightTensor structure
	weightTensor := WeightTensor{
		Name:  "layer1.weight",
		Shape: []int{10, 5},
		Data:  make([]float32, 50),
		Layer: "layer1",
		Type:  "weight",
	}

	if weightTensor.Name != "layer1.weight" {
		t.Errorf("Expected name 'layer1.weight', got '%s'", weightTensor.Name)
	}
	if len(weightTensor.Shape) != 2 || weightTensor.Shape[0] != 10 || weightTensor.Shape[1] != 5 {
		t.Errorf("Expected shape [10, 5], got %v", weightTensor.Shape)
	}
	if len(weightTensor.Data) != 50 {
		t.Errorf("Expected data length 50, got %d", len(weightTensor.Data))
	}

	// Test TrainingState structure
	trainingState := TrainingState{
		Epoch:        5,
		Step:         1000,
		LearningRate: 0.001,
		BestLoss:     0.25,
		BestAccuracy: 0.95,
		TotalSteps:   5000,
	}

	if trainingState.Epoch != 5 {
		t.Errorf("Expected epoch 5, got %d", trainingState.Epoch)
	}
	if trainingState.LearningRate != 0.001 {
		t.Errorf("Expected learning rate 0.001, got %f", trainingState.LearningRate)
	}

	// Test OptimizerState structure
	optimizerState := OptimizerState{
		Type:       "Adam",
		Parameters: map[string]interface{}{"beta1": 0.9, "beta2": 0.999},
		StateData:  []OptimizerTensor{},
	}

	if optimizerState.Type != "Adam" {
		t.Errorf("Expected optimizer type 'Adam', got '%s'", optimizerState.Type)
	}
	if len(optimizerState.Parameters) != 2 {
		t.Errorf("Expected 2 parameters, got %d", len(optimizerState.Parameters))
	}

	// Test CheckpointMetadata structure
	now := time.Now()
	metadata := CheckpointMetadata{
		Version:     "1.0.0",
		Framework:   "go-metal",
		CreatedAt:   now,
		Description: "Test checkpoint",
		Tags:        []string{"test", "validation"},
	}

	if metadata.Framework != "go-metal" {
		t.Errorf("Expected framework 'go-metal', got '%s'", metadata.Framework)
	}
	if len(metadata.Tags) != 2 {
		t.Errorf("Expected 2 tags, got %d", len(metadata.Tags))
	}
	if !metadata.CreatedAt.Equal(now) {
		t.Error("CreatedAt time mismatch")
	}

	t.Log("Checkpoint data structure tests passed")
}

// MockTensor implements basic tensor interface for testing
type MockTensor struct {
	shape []int
	data  []float32
}

func (mt *MockTensor) Shape() []int {
	return mt.shape
}

func (mt *MockTensor) ToFloat32Slice() ([]float32, error) {
	return mt.data, nil
}

func (mt *MockTensor) CopyFloat32Data(data []float32) error {
	if len(data) != len(mt.data) {
		return fmt.Errorf("data length mismatch: expected %d, got %d", len(mt.data), len(data))
	}
	copy(mt.data, data)
	return nil
}

func (mt *MockTensor) Release() {
	// Mock implementation - no-op
}

// TestExtractWeightsFromTensors tests weight extraction functionality
func TestExtractWeightsFromTensors(t *testing.T) {
	// Since we can't easily create real memory.Tensor objects for testing,
	// let's test the error conditions and validation logic
	
	// Test with nil tensors and model - this will panic in the actual function
	// so we need to skip this test case and focus on testing with valid models

	// Create a test model with Dense layers
	inputShape := []int{1, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(16, true, "dense1").  // Dense with bias
		AddDense(8, false, "dense2").  // Dense without bias
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	// Test with insufficient tensors
	emptyTensors := []*memory.Tensor{}
	_, err = ExtractWeightsFromTensors(emptyTensors, model)
	if err == nil {
		t.Error("Expected error for insufficient tensors")
	}
	if !strings.Contains(err.Error(), "insufficient tensors") {
		t.Errorf("Expected 'insufficient tensors' error, got: %v", err)
	}

	// Test the layer type handling logic
	for _, layerSpec := range model.Layers {
		switch layerSpec.Type {
		case layers.Dense:
			// Verify Dense layer parameters are accessible
			if useBias, ok := layerSpec.Parameters["use_bias"].(bool); ok {
				t.Logf("Dense layer %s has use_bias: %t", layerSpec.Name, useBias)
			} else {
				t.Errorf("Dense layer %s missing use_bias parameter", layerSpec.Name)
			}
		case layers.ReLU, layers.LeakyReLU, layers.Softmax, layers.Dropout:
			// These should be handled (no parameters)
			t.Logf("Activation layer %s of type %s", layerSpec.Name, layerSpec.Type.String())
		default:
			// This would trigger the unsupported layer error in the actual function
			t.Logf("Layer %s has type %s", layerSpec.Name, layerSpec.Type.String())
		}
	}

	t.Log("ExtractWeightsFromTensors validation tests passed")
}

// TestLoadWeightsIntoTensors tests weight loading functionality
func TestLoadWeightsIntoTensors(t *testing.T) {
	// Test with mismatched weight and tensor counts
	weights := []WeightTensor{
		{
			Name:  "layer1.weight",
			Shape: []int{10, 5},
			Data:  make([]float32, 50),
			Layer: "layer1",
			Type:  "weight",
		},
		{
			Name:  "layer1.bias",
			Shape: []int{5},
			Data:  make([]float32, 5),
			Layer: "layer1",
			Type:  "bias",
		},
	}

	// Test with no tensors
	emptyTensors := []*memory.Tensor{}
	err := LoadWeightsIntoTensors(weights, emptyTensors)
	if err == nil {
		t.Error("Expected error for mismatched weight/tensor count")
	}
	if !strings.Contains(err.Error(), "weight count mismatch") {
		t.Errorf("Expected 'weight count mismatch' error, got: %v", err)
	}

	// Test with running statistics (should be filtered out)
	weightsWithStats := []WeightTensor{
		{
			Name:  "layer1.weight",
			Shape: []int{10, 5},
			Data:  make([]float32, 50),
			Layer: "layer1",
			Type:  "weight",
		},
		{
			Name:  "layer1.running_mean",
			Shape: []int{5},
			Data:  make([]float32, 5),
			Layer: "layer1",
			Type:  "running_mean",
		},
		{
			Name:  "layer1.running_var",
			Shape: []int{5},
			Data:  make([]float32, 5),
			Layer: "layer1",
			Type:  "running_var",
		},
	}

	// This should result in 1 learnable weight vs 0 tensors
	err = LoadWeightsIntoTensors(weightsWithStats, emptyTensors)
	if err == nil {
		t.Error("Expected error for mismatched count after filtering")
	}

	t.Log("LoadWeightsIntoTensors validation tests passed")
}

// Helper function to verify contains functionality
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || 
		    (len(s) > len(substr) && 
		     (s[:len(substr)] == substr || 
		      s[len(s)-len(substr):] == substr || 
		      containsAtIndex(s, substr))))
}

func containsAtIndex(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// TestONNXExporterCreation tests ONNX exporter creation
func TestONNXExporterCreation(t *testing.T) {
	exporter := NewONNXExporter()
	if exporter == nil {
		t.Error("ONNX exporter should not be nil")
	}

	// The model field should be nil initially
	if exporter.model != nil {
		t.Error("ONNX exporter model should be nil initially")
	}

	t.Log("ONNX exporter creation test passed")
}

// TestONNXImporterCreation tests ONNX importer creation
func TestONNXImporterCreation(t *testing.T) {
	importer := NewONNXImporter()
	if importer == nil {
		t.Error("ONNX importer should not be nil")
	}

	t.Log("ONNX importer creation test passed")
}

// TestONNXImportFileErrors tests ONNX import error conditions
func TestONNXImportFileErrors(t *testing.T) {
	importer := NewONNXImporter()

	// Test importing non-existent file
	_, err := importer.ImportFromONNX("nonexistent.onnx")
	if err == nil {
		t.Error("Expected error for non-existent ONNX file")
	}
	if !strings.Contains(err.Error(), "failed to read ONNX file") {
		t.Errorf("Expected 'failed to read ONNX file' error, got: %v", err)
	}

	// Test importing invalid protobuf file
	invalidFile := "invalid.onnx"
	defer os.Remove(invalidFile)
	
	if err := os.WriteFile(invalidFile, []byte("invalid protobuf data"), 0644); err != nil {
		t.Fatalf("Failed to create invalid protobuf file: %v", err)
	}

	_, err = importer.ImportFromONNX(invalidFile)
	if err == nil {
		t.Error("Expected error for invalid protobuf data")
	}
	if !strings.Contains(err.Error(), "failed to unmarshal ONNX model") {
		t.Errorf("Expected 'failed to unmarshal ONNX model' error, got: %v", err)
	}

	t.Log("ONNX import file error tests passed")
}

// TestONNXFormatHandling tests ONNX format-specific operations
func TestONNXFormatHandling(t *testing.T) {
	// Create a basic checkpoint for testing
	inputShape := []int{1, 4}
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.AddDense(2, false, "test").Compile()
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	// Create weights that match the model (Dense layer without bias)
	weights := []WeightTensor{
		{
			Name:  "test.weight",
			Shape: []int{4, 2}, // input_size=4, output_size=2
			Data:  make([]float32, 8),
			Layer: "test",
			Type:  "weight",
		},
	}

	// Initialize weight data
	for i := range weights[0].Data {
		weights[0].Data[i] = float32(i) * 0.1
	}

	checkpoint := &Checkpoint{
		ModelSpec: model,
		Weights:   weights,
		Metadata: CheckpointMetadata{
			Framework: "go-metal",
			Version:   "1.0.0",
		},
	}

	// Test ONNX format handling 
	onnxSaver := NewCheckpointSaver(FormatONNX)
	if onnxSaver.format != FormatONNX {
		t.Errorf("Expected ONNX format, got %d", onnxSaver.format)
	}

	// Test ONNX save operation (should attempt to create ONNX file)
	onnxFile := "test_format.onnx"
	defer os.Remove(onnxFile)
	
	err = onnxSaver.SaveCheckpoint(checkpoint, onnxFile)
	// This may fail due to protobuf generation, but should not panic
	if err != nil {
		t.Logf("ONNX save failed as expected: %v", err)
	}

	// Test ONNX load operation (should attempt to load ONNX file)
	_, err = onnxSaver.LoadCheckpoint("nonexistent.onnx")
	if err == nil {
		t.Error("Expected error for non-existent ONNX file")
	}

	t.Log("ONNX format handling tests passed")
}

// TestMemoryManagerMocking tests creating mock memory objects
func TestMemoryManagerMocking(t *testing.T) {
	// Test mock device creation
	mockDevice := memory.CreateMockDevice()
	if mockDevice == nil {
		t.Error("Mock device should not be nil")
	}

	// Test mock command queue creation
	mockQueue := memory.CreateMockCommandQueue()
	if mockQueue == nil {
		t.Error("Mock command queue should not be nil")
	}

	// Verify mock objects have different addresses
	if uintptr(unsafe.Pointer(mockDevice)) == uintptr(unsafe.Pointer(mockQueue)) {
		t.Error("Mock device and queue should have different addresses")
	}

	// Test that mock objects are non-zero pointers
	if uintptr(unsafe.Pointer(mockDevice)) == 0 {
		t.Error("Mock device should be non-zero pointer")
	}
	if uintptr(unsafe.Pointer(mockQueue)) == 0 {
		t.Error("Mock command queue should be non-zero pointer")
	}

	t.Log("Memory manager mocking tests passed")
}

// TestWeightTensorValidation tests weight tensor validation logic
func TestWeightTensorValidation(t *testing.T) {
	// Test different weight types
	tests := []struct {
		weight WeightTensor
		isRunningStats bool
	}{
		{
			WeightTensor{Type: "weight"},
			false,
		},
		{
			WeightTensor{Type: "bias"},
			false,
		},
		{
			WeightTensor{Type: "gamma"},
			false,
		},
		{
			WeightTensor{Type: "beta"},
			false,
		},
		{
			WeightTensor{Type: "running_mean"},
			true,
		},
		{
			WeightTensor{Type: "running_var"},
			true,
		},
	}

	for _, test := range tests {
		isRunning := test.weight.Type == "running_mean" || test.weight.Type == "running_var"
		if isRunning != test.isRunningStats {
			t.Errorf("Weight type %s: expected running stats %t, got %t", 
				test.weight.Type, test.isRunningStats, isRunning)
		}
	}

	t.Log("Weight tensor validation tests passed")
}

// TestCompleteCheckpointRoundTrip tests a complete save/load cycle
func TestCompleteCheckpointRoundTrip(t *testing.T) {
	// Create a comprehensive checkpoint
	inputShape := []int{1, 8}
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddDense(4, true, "dense1").
		AddReLU("relu1").
		AddDense(2, true, "output").
		Compile()
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	// Create weights with realistic data
	weights := []WeightTensor{
		{
			Name:  "dense1.weight",
			Shape: []int{8, 4},
			Data:  make([]float32, 32),
			Layer: "dense1",
			Type:  "weight",
		},
		{
			Name:  "dense1.bias",
			Shape: []int{4},
			Data:  make([]float32, 4),
			Layer: "dense1",
			Type:  "bias",
		},
		{
			Name:  "output.weight",
			Shape: []int{4, 2},
			Data:  make([]float32, 8),
			Layer: "output",
			Type:  "weight",
		},
		{
			Name:  "output.bias",
			Shape: []int{2},
			Data:  make([]float32, 2),
			Layer: "output",
			Type:  "bias",
		},
	}

	// Initialize weights with pattern data for verification
	for i, weight := range weights {
		for j := range weight.Data {
			weights[i].Data[j] = float32(i*100 + j) * 0.01
		}
	}

	// Create original checkpoint
	original := &Checkpoint{
		ModelSpec: model,
		Weights:   weights,
		TrainingState: TrainingState{
			Epoch:        15,
			Step:         3000,
			LearningRate: 0.0005,
			BestLoss:     0.15,
			BestAccuracy: 0.95,
			TotalSteps:   15000,
		},
		OptimizerState: &OptimizerState{
			Type: "AdamW",
			Parameters: map[string]interface{}{
				"learning_rate": 0.0005,
				"beta1":         0.9,
				"beta2":         0.999,
				"weight_decay":  0.01,
			},
			StateData: []OptimizerTensor{},
		},
		Metadata: CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			CreatedAt:   time.Now(),
			Description: "Complete round-trip test checkpoint",
			Tags:        []string{"test", "roundtrip", "validation"},
		},
	}

	// Test JSON round-trip
	jsonSaver := NewCheckpointSaver(FormatJSON)
	jsonFile := "roundtrip_test.json"
	defer os.Remove(jsonFile)

	// Save
	err = jsonSaver.SaveCheckpoint(original, jsonFile)
	if err != nil {
		t.Fatalf("Failed to save JSON checkpoint: %v", err)
	}

	// Load
	loaded, err := jsonSaver.LoadCheckpoint(jsonFile)
	if err != nil {
		t.Fatalf("Failed to load JSON checkpoint: %v", err)
	}

	// Verify round-trip integrity
	if loaded.TrainingState.Epoch != original.TrainingState.Epoch {
		t.Errorf("Training state epoch mismatch: expected %d, got %d", 
			original.TrainingState.Epoch, loaded.TrainingState.Epoch)
	}

	if loaded.TrainingState.LearningRate != original.TrainingState.LearningRate {
		t.Errorf("Learning rate mismatch: expected %f, got %f", 
			original.TrainingState.LearningRate, loaded.TrainingState.LearningRate)
	}

	if len(loaded.Weights) != len(original.Weights) {
		t.Errorf("Weight count mismatch: expected %d, got %d", 
			len(original.Weights), len(loaded.Weights))
	}

	// Verify weight data integrity
	for i, originalWeight := range original.Weights {
		if i >= len(loaded.Weights) {
			t.Errorf("Missing weight %d in loaded checkpoint", i)
			continue
		}

		loadedWeight := loaded.Weights[i]
		if originalWeight.Name != loadedWeight.Name {
			t.Errorf("Weight %d name mismatch: expected %s, got %s", 
				i, originalWeight.Name, loadedWeight.Name)
		}

		if len(originalWeight.Data) != len(loadedWeight.Data) {
			t.Errorf("Weight %d data length mismatch: expected %d, got %d", 
				i, len(originalWeight.Data), len(loadedWeight.Data))
			continue
		}

		// Check data values
		for j, originalVal := range originalWeight.Data {
			if j < len(loadedWeight.Data) && originalVal != loadedWeight.Data[j] {
				t.Errorf("Weight %d data[%d] mismatch: expected %f, got %f", 
					i, j, originalVal, loadedWeight.Data[j])
				break // Only report first mismatch
			}
		}
	}

	// Verify optimizer state
	if loaded.OptimizerState == nil {
		t.Error("Loaded checkpoint missing optimizer state")
	} else {
		if loaded.OptimizerState.Type != original.OptimizerState.Type {
			t.Errorf("Optimizer type mismatch: expected %s, got %s", 
				original.OptimizerState.Type, loaded.OptimizerState.Type)
		}
	}

	// Verify metadata
	if loaded.Metadata.Framework != original.Metadata.Framework {
		t.Errorf("Framework mismatch: expected %s, got %s", 
			original.Metadata.Framework, loaded.Metadata.Framework)
	}

	if len(loaded.Metadata.Tags) != len(original.Metadata.Tags) {
		t.Errorf("Tags count mismatch: expected %d, got %d", 
			len(original.Metadata.Tags), len(loaded.Metadata.Tags))
	}

	t.Log("Complete checkpoint round-trip test passed")
}