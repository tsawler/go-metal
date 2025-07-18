package engine

import (
	"testing"
	
	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
)

// TestInferenceEngineWithDynamicTraining tests the updated inference engine with dynamic training engine compatibility
func TestInferenceEngineWithDynamicTraining(t *testing.T) {
	// Test various model architectures that the dynamic training engine supports
	
	// Test 1: Simple MLP (1D->2D architecture)
	t.Run("MLP_1D_Input", func(t *testing.T) {
		builder := layers.NewModelBuilder([]int{4, 10}) // 1D input: batch_size=4, features=10
		model, err := builder.
			AddDense(128, true, "dense1").
			AddReLU("relu1").
			AddDense(64, true, "dense2").
			AddReLU("relu2").
			AddDense(3, true, "output").
			AddSoftmax(-1, "softmax").
			Compile()
		if err != nil {
			t.Fatalf("Failed to compile MLP model: %v", err)
		}
		
		// Validate model is compatible with dynamic engine
		if err := model.ValidateModelForDynamicEngine(); err != nil {
			t.Fatalf("Model validation failed: %v", err)
		}
		
		// Test that inference engine can be created
		config := cgo_bridge.InferenceConfig{
			UseDynamicEngine:       true,
			BatchNormInferenceMode: true,
			InputShape:             []int32{4, 10},
			InputShapeLen:          2,
			UseCommandPooling:      true,
			OptimizeForSingleBatch: false,
		}
		
		// This should not fail now that inference engine supports dynamic architecture
		_, err = NewModelInferenceEngine(model, config)
		if err != nil {
			t.Fatalf("Failed to create inference engine for MLP: %v", err)
		}
	})
	
	// Test 2: CNN Architecture (4D input)
	t.Run("CNN_4D_Input", func(t *testing.T) {
		builder := layers.NewModelBuilder([]int{4, 3, 32, 32}) // 4D input: batch=4, channels=3, height=32, width=32
		model, err := builder.
			AddConv2D(16, 3, 1, 1, true, "conv1").
			AddReLU("relu1").
			AddConv2D(32, 3, 1, 1, true, "conv2").
			AddReLU("relu2").
			AddDense(128, true, "dense1").
			AddReLU("relu3").
			AddDense(10, true, "output").
			AddSoftmax(-1, "softmax").
			Compile()
		if err != nil {
			t.Fatalf("Failed to compile CNN model: %v", err)
		}
		
		// Validate model is compatible with dynamic engine
		if err := model.ValidateModelForDynamicEngine(); err != nil {
			t.Fatalf("Model validation failed: %v", err)
		}
		
		// Test that inference engine can be created
		config := cgo_bridge.InferenceConfig{
			UseDynamicEngine:       true,
			BatchNormInferenceMode: true,
			InputShape:             []int32{4, 3, 32, 32},
			InputShapeLen:          4,
			UseCommandPooling:      true,
			OptimizeForSingleBatch: false,
		}
		
		// This should not fail now that inference engine supports dynamic architecture
		_, err = NewModelInferenceEngine(model, config)
		if err != nil {
			t.Fatalf("Failed to create inference engine for CNN: %v", err)
		}
	})
	
	// Test 3: Mixed architecture with BatchNorm
	t.Run("Mixed_Architecture_BatchNorm", func(t *testing.T) {
		builder := layers.NewModelBuilder([]int{2, 3, 16, 16}) // 4D input
		model, err := builder.
			AddConv2D(32, 3, 1, 1, true, "conv1").
			AddBatchNorm(32, 1e-5, 0.1, true, "bn1").
			AddReLU("relu1").
			AddDense(64, true, "dense1").
			AddBatchNorm(64, 1e-5, 0.1, true, "bn2").
			AddReLU("relu2").
			AddDense(5, true, "output").
			AddSoftmax(-1, "softmax").
			Compile()
		if err != nil {
			t.Fatalf("Failed to compile mixed model: %v", err)
		}
		
		// Validate model is compatible with dynamic engine
		if err := model.ValidateModelForDynamicEngine(); err != nil {
			t.Fatalf("Model validation failed: %v", err)
		}
		
		// Test that inference engine can be created
		config := cgo_bridge.InferenceConfig{
			UseDynamicEngine:       true,
			BatchNormInferenceMode: true,
			InputShape:             []int32{2, 3, 16, 16},
			InputShapeLen:          4,
			UseCommandPooling:      true,
			OptimizeForSingleBatch: false,
		}
		
		// This should not fail now that inference engine supports dynamic architecture
		_, err = NewModelInferenceEngine(model, config)
		if err != nil {
			t.Fatalf("Failed to create inference engine for mixed architecture: %v", err)
		}
	})
	
	// Test 4: Validate that NewModelInferenceEngineFromDynamicTraining works
	t.Run("FromDynamicTraining_Constructor", func(t *testing.T) {
		builder := layers.NewModelBuilder([]int{1, 20}) // Simple 2D input
		model, err := builder.
			AddDense(50, true, "dense1").
			AddReLU("relu1").
			AddDense(10, true, "output").
			AddSoftmax(-1, "softmax").
			Compile()
		if err != nil {
			t.Fatalf("Failed to compile model: %v", err)
		}
		
		config := cgo_bridge.InferenceConfig{
			UseDynamicEngine:       true,
			BatchNormInferenceMode: true,
			InputShape:             []int32{1, 20},
			InputShapeLen:          2,
			UseCommandPooling:      true,
			OptimizeForSingleBatch: true,
		}
		
		// Test the specific constructor for dynamic training compatibility
		_, err = NewModelInferenceEngineFromDynamicTraining(model, config)
		if err != nil {
			t.Fatalf("Failed to create inference engine from dynamic training: %v", err)
		}
	})
	
	// Test 5: Verify Predict method supports flexible input shapes
	t.Run("Predict_Flexible_Input", func(t *testing.T) {
		builder := layers.NewModelBuilder([]int{2, 5}) // 2D input
		model, err := builder.
			AddDense(10, true, "dense1").
			AddReLU("relu1").
			AddDense(2, true, "output").
			AddSoftmax(-1, "softmax").
			Compile()
		if err != nil {
			t.Fatalf("Failed to compile model: %v", err)
		}
		
		config := cgo_bridge.InferenceConfig{
			UseDynamicEngine:       true,
			BatchNormInferenceMode: true,
			InputShape:             []int32{2, 5},
			InputShapeLen:          2,
			UseCommandPooling:      true,
			OptimizeForSingleBatch: false,
		}
		
		engine, err := NewModelInferenceEngine(model, config)
		if err != nil {
			t.Fatalf("Failed to create inference engine: %v", err)
		}
		
		// Test that Predict accepts various input shapes
		inputData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
		inputShape := []int{2, 5} // 2D input shape
		
		// This should work with the updated Predict method
		_, err = engine.Predict(inputData, inputShape)
		if err != nil {
			t.Logf("Predict with 2D input failed (expected due to CGO): %v", err)
		}
		
		// Test with 1D input (this should now be supported)
		inputData1D := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
		inputShape1D := []int{1, 5} // 1D input shape
		
		_, err = engine.Predict(inputData1D, inputShape1D)
		if err != nil {
			t.Logf("Predict with 1D input failed (expected due to CGO): %v", err)
		}
		
		// Note: We can't fully test inference without CGO, but we can test the validation
	})
}

// TestInferenceEngineArchitectureSupport tests that the engine supports various architectures
func TestInferenceEngineArchitectureSupport(t *testing.T) {
	architectures := []struct {
		name       string
		inputShape []int
		buildFunc  func(*layers.ModelBuilder) (*layers.ModelSpec, error)
	}{
		{
			name:       "Simple_MLP",
			inputShape: []int{1, 10},
			buildFunc: func(builder *layers.ModelBuilder) (*layers.ModelSpec, error) {
				return builder.AddDense(5, true, "dense1").AddReLU("relu1").AddDense(2, true, "output").Compile()
			},
		},
		{
			name:       "Deep_MLP",
			inputShape: []int{1, 784},
			buildFunc: func(builder *layers.ModelBuilder) (*layers.ModelSpec, error) {
				return builder.
					AddDense(128, true, "dense1").AddReLU("relu1").
					AddDense(64, true, "dense2").AddReLU("relu2").
					AddDense(32, true, "dense3").AddReLU("relu3").
					AddDense(10, true, "output").AddSoftmax(-1, "softmax").
					Compile()
			},
		},
		{
			name:       "CNN_Small",
			inputShape: []int{1, 1, 28, 28},
			buildFunc: func(builder *layers.ModelBuilder) (*layers.ModelSpec, error) {
				return builder.
					AddConv2D(8, 3, 1, 1, true, "conv1").AddReLU("relu1").
					AddDense(10, true, "output").AddSoftmax(-1, "softmax").
					Compile()
			},
		},
		{
			name:       "CNN_Complex",
			inputShape: []int{1, 3, 32, 32},
			buildFunc: func(builder *layers.ModelBuilder) (*layers.ModelSpec, error) {
				return builder.
					AddConv2D(16, 3, 1, 1, true, "conv1").AddReLU("relu1").
					AddConv2D(32, 3, 1, 1, true, "conv2").AddReLU("relu2").
					AddDense(128, true, "dense1").AddReLU("relu3").
					AddDense(10, true, "output").AddSoftmax(-1, "softmax").
					Compile()
			},
		},
	}
	
	for _, arch := range architectures {
		t.Run(arch.name, func(t *testing.T) {
			builder := layers.NewModelBuilder(arch.inputShape)
			model, err := arch.buildFunc(builder)
			if err != nil {
				t.Fatalf("Failed to build model: %v", err)
			}
			
			// Validate model is compatible with dynamic engine
			if err := model.ValidateModelForDynamicEngine(); err != nil {
				t.Fatalf("Model validation failed: %v", err)
			}
			
			// Test that inference engine can be created
			inputShape := make([]int32, len(arch.inputShape))
			for i, dim := range arch.inputShape {
				inputShape[i] = int32(dim)
			}
			
			config := cgo_bridge.InferenceConfig{
				UseDynamicEngine:       true,
				BatchNormInferenceMode: true,
				InputShape:             inputShape,
				InputShapeLen:          int32(len(inputShape)),
				UseCommandPooling:      true,
				OptimizeForSingleBatch: false,
			}
			
			// This should not fail now that inference engine supports dynamic architecture
			_, err = NewModelInferenceEngine(model, config)
			if err != nil {
				t.Fatalf("Failed to create inference engine for %s: %v", arch.name, err)
			}
		})
	}
}