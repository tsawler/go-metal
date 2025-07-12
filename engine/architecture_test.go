package engine

import (
	"testing"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
)

// TestGenericArchitectureSupport tests that the engine supports various model architectures
func TestGenericArchitectureSupport(t *testing.T) {
	// Test Pure MLP Architecture (no CNN layers)
	t.Run("PureMLP", func(t *testing.T) {
		// Create model using ModelBuilder for proper compilation
		builder := layers.NewModelBuilder([]int{1, 784}) // Flattened 28x28 image
		builder.AddDense(128, true, "hidden1")
		builder.AddReLU("relu1")
		builder.AddDense(64, true, "hidden2")
		builder.AddReLU("relu2")
		builder.AddDense(10, true, "output")
		builder.AddSoftmax(-1, "softmax")
		
		mlpModel, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile MLP model: %v", err)
		}
		
		// Validate for dynamic engine (should work with any architecture)
		err = mlpModel.ValidateModelForDynamicEngine()
		if err != nil {
			t.Fatalf("Pure MLP should be valid for dynamic engine: %v", err)
		}
		
		// Test conversion to dynamic specs
		_, err = mlpModel.ConvertToDynamicLayerSpecs()
		if err != nil {
			t.Fatalf("Pure MLP should convert to dynamic specs: %v", err)
		}
		
		t.Logf("✅ Pure MLP architecture validated successfully")
	})
	
	// Test CNN Architecture
	t.Run("CNN", func(t *testing.T) {
		// Create CNN using ModelBuilder for proper compilation
		builder := layers.NewModelBuilder([]int{1, 1, 28, 28}) // [batch, channels, height, width]
		builder.AddConv2D(32, 3, 1, 1, true, "conv1")
		builder.AddReLU("relu1")
		builder.AddDense(64, true, "fc1") // Dense layer input size will be computed automatically
		builder.AddReLU("relu2")
		builder.AddDense(10, true, "output")
		builder.AddSoftmax(-1, "softmax")
		
		cnnModel, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile CNN model: %v", err)
		}
		
		// Validate for dynamic engine
		err = cnnModel.ValidateModelForDynamicEngine()
		if err != nil {
			t.Fatalf("CNN should be valid for dynamic engine: %v", err)
		}
		
		// Test conversion to dynamic specs
		_, err = cnnModel.ConvertToDynamicLayerSpecs()
		if err != nil {
			t.Fatalf("CNN should convert to dynamic specs: %v", err)
		}
		
		t.Logf("✅ CNN architecture validated successfully")
	})
	
	// Test Mixed Architecture with BatchNorm
	t.Run("MixedWithBatchNorm", func(t *testing.T) {
		// Create mixed architecture using ModelBuilder
		builder := layers.NewModelBuilder([]int{1, 3, 28, 28}) // RGB input
		builder.AddConv2D(16, 5, 1, 2, true, "conv1")
		builder.AddBatchNorm(16, 1e-5, 0.1, true, "bn1")
		builder.AddReLU("relu1")
		builder.AddDense(32, true, "fc1") // Dense input size computed automatically
		builder.AddDropout(0.5, "dropout1")
		builder.AddDense(5, true, "output")
		
		mixedModel, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile mixed model: %v", err)
		}
		
		// Validate for dynamic engine
		err = mixedModel.ValidateModelForDynamicEngine()
		if err != nil {
			t.Fatalf("Mixed architecture should be valid for dynamic engine: %v", err)
		}
		
		// Test conversion to dynamic specs
		specs, err := mixedModel.ConvertToDynamicLayerSpecs()
		if err != nil {
			t.Fatalf("Mixed architecture should convert to dynamic specs: %v", err)
		}
		
		// Verify BatchNorm layer has running stats
		foundBatchNorm := false
		for _, spec := range specs {
			if spec.LayerType == int32(layers.BatchNorm) {
				if !spec.HasRunningStats {
					t.Errorf("BatchNorm layer should have running stats")
				}
				if len(spec.RunningMean) == 0 || len(spec.RunningVar) == 0 {
					t.Errorf("BatchNorm layer should have initialized running mean/var")
				}
				foundBatchNorm = true
			}
		}
		
		if !foundBatchNorm {
			t.Errorf("Should have found BatchNorm layer in specs")
		}
		
		t.Logf("✅ Mixed architecture with BatchNorm validated successfully")
	})
}

// TestEngineCompilation tests that different architectures compile correctly
func TestEngineCompilation(t *testing.T) {
	// Test that dynamic engine creation works (we can't actually create it without Metal device)
	t.Run("DynamicEngineSupport", func(t *testing.T) {
		// Create simple MLP using ModelBuilder
		builder := layers.NewModelBuilder([]int{1, 10})
		builder.AddDense(5, true, "hidden")
		builder.AddReLU("relu")
		builder.AddDense(2, true, "output")
		
		mlpModel, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile test model: %v", err)
		}
		
		// Create training config
		config := cgo_bridge.TrainingConfig{
			LearningRate: 0.001,
			OptimizerType: cgo_bridge.Adam,
			ProblemType:   0, // Classification
			LossFunction:  0, // CrossEntropy
		}
		
		// The actual engine creation would require a Metal device
		// For this test, we just verify the architecture validation passes
		err = mlpModel.ValidateModelForDynamicEngine()
		if err != nil {
			t.Fatalf("MLP model should validate for dynamic engine: %v", err)
		}
		
		// Test layer specification conversion
		specs, err := mlpModel.ConvertToDynamicLayerSpecs()
		if err != nil {
			t.Fatalf("Should convert MLP to dynamic specs: %v", err)
		}
		
		if len(specs) != len(mlpModel.Layers) {
			t.Errorf("Expected %d specs, got %d", len(mlpModel.Layers), len(specs))
		}
		
		t.Logf("✅ Dynamic engine compilation support verified")
		
		// Verify config is properly structured
		if config.OptimizerType != cgo_bridge.Adam {
			t.Errorf("Config should preserve optimizer type")
		}
	})
}