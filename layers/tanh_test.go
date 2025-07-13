package layers

import (
	"testing"
)

func TestTanhLayerType(t *testing.T) {
	// Test that Tanh has correct string representation
	if Tanh.String() != "Tanh" {
		t.Errorf("Expected Tanh.String() to be 'Tanh', got %s", Tanh.String())
	}
}

func TestTanhFactory(t *testing.T) {
	factory := NewFactory()
	
	// Test CreateTanhSpec
	spec := factory.CreateTanhSpec("test_tanh")
	
	if spec.Type != Tanh {
		t.Errorf("Expected Type to be Tanh, got %v", spec.Type)
	}
	
	if spec.Name != "test_tanh" {
		t.Errorf("Expected Name to be 'test_tanh', got %s", spec.Name)
	}
	
	if len(spec.Parameters) != 0 {
		t.Errorf("Expected no parameters for Tanh, got %d", len(spec.Parameters))
	}
}

func TestTanhModelBuilder(t *testing.T) {
	inputShape := []int{32, 10}
	builder := NewModelBuilder(inputShape)
	
	// Test AddTanh method
	builder = builder.AddTanh("tanh_layer")
	
	if len(builder.layers) != 1 {
		t.Errorf("Expected 1 layer, got %d", len(builder.layers))
	}
	
	layer := builder.layers[0]
	if layer.Type != Tanh {
		t.Errorf("Expected layer type to be Tanh, got %v", layer.Type)
	}
	
	if layer.Name != "tanh_layer" {
		t.Errorf("Expected layer name to be 'tanh_layer', got %s", layer.Name)
	}
	
	if len(layer.Parameters) != 0 {
		t.Errorf("Expected no parameters for Tanh, got %d", len(layer.Parameters))
	}
}

func TestTanhCompilation(t *testing.T) {
	inputShape := []int{2, 5}
	builder := NewModelBuilder(inputShape)
	
	// Create a simple model with Tanh
	model, err := builder.
		AddDense(3, true, "dense1").
		AddTanh("tanh1").
		AddDense(2, true, "hidden").
		AddTanh("tanh_hidden").
		AddDense(1, true, "output").
		AddTanh("output_tanh").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	if len(model.Layers) != 6 {
		t.Errorf("Expected 6 layers, got %d", len(model.Layers))
	}
	
	// Check that Tanh layers have correct input/output shapes
	tanh1 := model.Layers[1]
	if tanh1.Type != Tanh {
		t.Errorf("Expected second layer to be Tanh, got %v", tanh1.Type)
	}
	
	expectedShape := []int{2, 3}
	if !equalIntSlices(tanh1.InputShape, expectedShape) || !equalIntSlices(tanh1.OutputShape, expectedShape) {
		t.Errorf("Expected Tanh input/output shape %v, got input: %v, output: %v", 
			expectedShape, tanh1.InputShape, tanh1.OutputShape)
	}
	
	tanhHidden := model.Layers[3]
	if tanhHidden.Type != Tanh {
		t.Errorf("Expected fourth layer to be Tanh, got %v", tanhHidden.Type)
	}
	
	expectedHiddenShape := []int{2, 2}
	if !equalIntSlices(tanhHidden.InputShape, expectedHiddenShape) || !equalIntSlices(tanhHidden.OutputShape, expectedHiddenShape) {
		t.Errorf("Expected hidden Tanh input/output shape %v, got input: %v, output: %v", 
			expectedHiddenShape, tanhHidden.InputShape, tanhHidden.OutputShape)
	}
	
	outputTanh := model.Layers[5]
	if outputTanh.Type != Tanh {
		t.Errorf("Expected sixth layer to be Tanh, got %v", outputTanh.Type)
	}
	
	expectedOutputShape := []int{2, 1}
	if !equalIntSlices(outputTanh.InputShape, expectedOutputShape) || !equalIntSlices(outputTanh.OutputShape, expectedOutputShape) {
		t.Errorf("Expected output Tanh input/output shape %v, got input: %v, output: %v", 
			expectedOutputShape, outputTanh.InputShape, outputTanh.OutputShape)
	}
}

func TestTanhParameterCounting(t *testing.T) {
	inputShape := []int{1, 10}
	builder := NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(5, true, "dense1").    // 10*5 + 5 = 55 parameters
		AddTanh("tanh1").               // 0 parameters
		AddDense(3, false, "hidden").   // 5*3 + 0 = 15 parameters
		AddTanh("tanh_hidden").         // 0 parameters
		AddDense(1, true, "output").    // 3*1 + 1 = 4 parameters
		AddTanh("output_tanh").         // 0 parameters
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	expectedParams := int64(55 + 15 + 4) // Total should be 74
	if model.TotalParameters != expectedParams {
		t.Errorf("Expected %d total parameters, got %d", expectedParams, model.TotalParameters)
	}
	
	// Check individual Tanh layers
	tanh1 := model.Layers[1]
	if tanh1.ParameterCount != 0 {
		t.Errorf("Expected Tanh layer to have 0 parameters, got %d", tanh1.ParameterCount)
	}
	
	tanhHidden := model.Layers[3]
	if tanhHidden.ParameterCount != 0 {
		t.Errorf("Expected hidden Tanh layer to have 0 parameters, got %d", tanhHidden.ParameterCount)
	}
	
	outputTanh := model.Layers[5]
	if outputTanh.ParameterCount != 0 {
		t.Errorf("Expected output Tanh layer to have 0 parameters, got %d", outputTanh.ParameterCount)
	}
}

func TestTanhZeroCentered(t *testing.T) {
	// This is a conceptual test - in practice, we can't easily test the actual
	// mathematical properties without running the network, but we can verify
	// the layer is properly configured for zero-centered outputs
	
	inputShape := []int{1, 5}
	builder := NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(3, true, "feature_layer").
		AddTanh("zero_centered_activation").  // This should produce outputs in [-1, 1]
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile zero-centered model: %v", err)
	}
	
	tanhLayer := model.Layers[1]
	if tanhLayer.Type != Tanh {
		t.Errorf("Expected zero-centered layer to be Tanh, got %v", tanhLayer.Type)
	}
	
	if tanhLayer.Name != "zero_centered_activation" {
		t.Errorf("Expected layer name to be 'zero_centered_activation', got %s", tanhLayer.Name)
	}
	
	// Verify shape preservation (Tanh doesn't change tensor dimensions)
	expectedShape := []int{1, 3}
	if !equalIntSlices(tanhLayer.InputShape, expectedShape) || !equalIntSlices(tanhLayer.OutputShape, expectedShape) {
		t.Errorf("Expected Tanh to preserve shape %v, got input: %v, output: %v", 
			expectedShape, tanhLayer.InputShape, tanhLayer.OutputShape)
	}
}

func TestTanhRNNUsage(t *testing.T) {
	// Test a typical RNN-style architecture where Tanh is commonly used
	inputShape := []int{1, 50} // Sequence input
	builder := NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "rnn_input").
		AddTanh("rnn_activation").      // Traditional RNN activation
		AddDense(32, true, "hidden").
		AddTanh("hidden_activation").   // Zero-centered hidden activation
		AddDense(10, true, "output").
		AddSoftmax(-1, "output_softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile RNN-style model: %v", err)
	}
	
	// Verify both Tanh layers
	rnnActivation := model.Layers[1]
	if rnnActivation.Type != Tanh || rnnActivation.Name != "rnn_activation" {
		t.Errorf("Expected RNN activation to be Tanh named 'rnn_activation', got %v named %s", 
			rnnActivation.Type, rnnActivation.Name)
	}
	
	hiddenActivation := model.Layers[3]
	if hiddenActivation.Type != Tanh || hiddenActivation.Name != "hidden_activation" {
		t.Errorf("Expected hidden activation to be Tanh named 'hidden_activation', got %v named %s", 
			hiddenActivation.Type, hiddenActivation.Name)
	}
}