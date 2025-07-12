package engine

import (
	"fmt"
	"testing"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
)

// TestCustomNormalization tests that the inference engine properly handles custom normalization values
func TestCustomNormalization(t *testing.T) {
	// Create a model with BatchNorm layer for testing
	builder := layers.NewModelBuilder([]int{1, 4, 32, 32}) // [batch, channels, height, width]
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddBatchNorm(16, 1e-5, 0.1, true, "bn1").
		AddReLU("relu1").
		AddDense(64, true, "fc1").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	// Create inference config
	config := cgo_bridge.InferenceConfig{
		UseDynamicEngine:       true,
		BatchNormInferenceMode: true,
	}
	
	// Create inference engine (this will fail without Metal device, but we can test the setup)
	_, err = NewModelInferenceEngine(model, config)
	
	// We expect this to fail due to Metal device not being available in CI/test environment
	// But we can verify the normalization logic is properly set up
	if err != nil {
		t.Logf("Expected Metal device error: %v", err)
		
		// Test the normalization logic directly on the model spec
		err = testNormalizationLogic(model)
		if err != nil {
			t.Fatalf("Normalization logic test failed: %v", err)
		}
		
		t.Log("✅ Normalization logic test passed")
	}
}

// testNormalizationLogic tests the normalization conversion without requiring Metal device
func testNormalizationLogic(model *layers.ModelSpec) error {
	// Test ConvertToInferenceLayerSpecs with default normalization
	inferenceSpecs, err := model.ConvertToInferenceLayerSpecs()
	if err != nil {
		return err
	}
	
	// Find the BatchNorm layer in the specs
	var batchNormSpec *layers.DynamicLayerSpec
	for i := range inferenceSpecs {
		if inferenceSpecs[i].LayerType == int32(layers.BatchNorm) {
			batchNormSpec = &inferenceSpecs[i]
			break
		}
	}
	
	if batchNormSpec == nil {
		return fmt.Errorf("BatchNorm layer not found in inference specs")
	}
	
	// Verify that running statistics are initialized
	if !batchNormSpec.HasRunningStats {
		return fmt.Errorf("BatchNorm layer should have running stats")
	}
	
	if len(batchNormSpec.RunningMean) == 0 {
		return fmt.Errorf("BatchNorm layer should have initialized running mean")
	}
	
	if len(batchNormSpec.RunningVar) == 0 {
		return fmt.Errorf("BatchNorm layer should have initialized running variance")
	}
	
	// Verify default values (mean=0, var=1)
	for i, mean := range batchNormSpec.RunningMean {
		if mean != 0.0 {
			return fmt.Errorf("Default running mean[%d] should be 0.0, got %f", i, mean)
		}
	}
	
	for i, variance := range batchNormSpec.RunningVar {
		if variance != 1.0 {
			return fmt.Errorf("Default running variance[%d] should be 1.0, got %f", i, variance)
		}
	}
	
	// Test custom normalization by modifying the model spec
	batchNormLayer := &model.Layers[1] // bn1 layer
	if batchNormLayer.RunningStatistics == nil {
		batchNormLayer.RunningStatistics = make(map[string][]float32)
	}
	
	// Set custom values
	customMean := []float32{0.5, -0.2, 0.1, 0.8, -0.1, 0.3, 0.7, -0.4, 0.2, 0.6, -0.3, 0.9, 0.0, 0.4, -0.5, 0.1}
	customVar := []float32{2.0, 1.5, 0.8, 1.2, 0.9, 1.8, 0.7, 1.3, 1.1, 0.6, 1.4, 0.5, 1.7, 0.4, 1.6, 1.0}
	
	batchNormLayer.RunningStatistics["running_mean"] = customMean
	batchNormLayer.RunningStatistics["running_var"] = customVar
	
	// Convert again with custom statistics
	customInferenceSpecs, err := model.ConvertToInferenceLayerSpecs()
	if err != nil {
		return err
	}
	
	// Find the BatchNorm layer again
	var customBatchNormSpec *layers.DynamicLayerSpec
	for i := range customInferenceSpecs {
		if customInferenceSpecs[i].LayerType == int32(layers.BatchNorm) {
			customBatchNormSpec = &customInferenceSpecs[i]
			break
		}
	}
	
	if customBatchNormSpec == nil {
		return fmt.Errorf("Custom BatchNorm layer not found in inference specs")
	}
	
	// Verify custom values were copied
	for i, mean := range customBatchNormSpec.RunningMean {
		if mean != customMean[i] {
			return fmt.Errorf("Custom running mean[%d] should be %f, got %f", i, customMean[i], mean)
		}
	}
	
	for i, variance := range customBatchNormSpec.RunningVar {
		if variance != customVar[i] {
			return fmt.Errorf("Custom running variance[%d] should be %f, got %f", i, customVar[i], variance)
		}
	}
	
	return nil
}

// TestBatchNormLayerListing tests the utility functions for managing normalization
func TestBatchNormLayerListing(t *testing.T) {
	// Create a model with multiple layers including BatchNorm
	builder := layers.NewModelBuilder([]int{1, 3, 64, 64})
	model, err := builder.
		AddConv2D(32, 3, 1, 1, true, "conv1").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn1").
		AddReLU("relu1").
		AddConv2D(64, 3, 1, 1, true, "conv2").
		AddBatchNorm(64, 1e-5, 0.1, true, "bn2").
		AddReLU("relu2").
		AddDense(128, true, "fc1").
		AddReLU("relu3").
		AddDense(10, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	// Count BatchNorm layers manually
	expectedBatchNormLayers := []string{"bn1", "bn2"}
	foundBatchNormLayers := 0
	
	for _, layer := range model.Layers {
		if layer.Type == layers.BatchNorm {
			foundBatchNormLayers++
			
			// Check if the layer name is in expected list
			found := false
			for _, expectedName := range expectedBatchNormLayers {
				if layer.Name == expectedName {
					found = true
					break
				}
			}
			
			if !found {
				t.Errorf("Unexpected BatchNorm layer name: %s", layer.Name)
			}
		}
	}
	
	if foundBatchNormLayers != len(expectedBatchNormLayers) {
		t.Errorf("Expected %d BatchNorm layers, found %d", len(expectedBatchNormLayers), foundBatchNormLayers)
	}
	
	t.Logf("✅ Found %d BatchNorm layers as expected", foundBatchNormLayers)
}