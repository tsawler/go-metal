package layers

import (
	"testing"
)

func TestSigmoidLayerType(t *testing.T) {
	// Test that Sigmoid has correct string representation
	if Sigmoid.String() != "Sigmoid" {
		t.Errorf("Expected Sigmoid.String() to be 'Sigmoid', got %s", Sigmoid.String())
	}
}

func TestSigmoidFactory(t *testing.T) {
	factory := NewFactory()
	
	// Test CreateSigmoidSpec
	spec := factory.CreateSigmoidSpec("test_sigmoid")
	
	if spec.Type != Sigmoid {
		t.Errorf("Expected Type to be Sigmoid, got %v", spec.Type)
	}
	
	if spec.Name != "test_sigmoid" {
		t.Errorf("Expected Name to be 'test_sigmoid', got %s", spec.Name)
	}
	
	if len(spec.Parameters) != 0 {
		t.Errorf("Expected no parameters for Sigmoid, got %d", len(spec.Parameters))
	}
}

func TestSigmoidModelBuilder(t *testing.T) {
	inputShape := []int{32, 10}
	builder := NewModelBuilder(inputShape)
	
	// Test AddSigmoid method
	builder = builder.AddSigmoid("sigmoid_layer")
	
	if len(builder.layers) != 1 {
		t.Errorf("Expected 1 layer, got %d", len(builder.layers))
	}
	
	layer := builder.layers[0]
	if layer.Type != Sigmoid {
		t.Errorf("Expected layer type to be Sigmoid, got %v", layer.Type)
	}
	
	if layer.Name != "sigmoid_layer" {
		t.Errorf("Expected layer name to be 'sigmoid_layer', got %s", layer.Name)
	}
	
	if len(layer.Parameters) != 0 {
		t.Errorf("Expected no parameters for Sigmoid, got %d", len(layer.Parameters))
	}
}

func TestSigmoidCompilation(t *testing.T) {
	inputShape := []int{2, 5}
	builder := NewModelBuilder(inputShape)
	
	// Create a simple model with Sigmoid
	model, err := builder.
		AddDense(3, true, "dense1").
		AddSigmoid("sigmoid1").
		AddDense(1, true, "output").
		AddSigmoid("output_sigmoid").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	if len(model.Layers) != 4 {
		t.Errorf("Expected 4 layers, got %d", len(model.Layers))
	}
	
	// Check that Sigmoid layers have correct input/output shapes
	sigmoid1 := model.Layers[1]
	if sigmoid1.Type != Sigmoid {
		t.Errorf("Expected second layer to be Sigmoid, got %v", sigmoid1.Type)
	}
	
	expectedShape := []int{2, 3}
	if !equalIntSlices(sigmoid1.InputShape, expectedShape) || !equalIntSlices(sigmoid1.OutputShape, expectedShape) {
		t.Errorf("Expected Sigmoid input/output shape %v, got input: %v, output: %v", 
			expectedShape, sigmoid1.InputShape, sigmoid1.OutputShape)
	}
	
	outputSigmoid := model.Layers[3]
	if outputSigmoid.Type != Sigmoid {
		t.Errorf("Expected fourth layer to be Sigmoid, got %v", outputSigmoid.Type)
	}
	
	expectedOutputShape := []int{2, 1}
	if !equalIntSlices(outputSigmoid.InputShape, expectedOutputShape) || !equalIntSlices(outputSigmoid.OutputShape, expectedOutputShape) {
		t.Errorf("Expected output Sigmoid input/output shape %v, got input: %v, output: %v", 
			expectedOutputShape, outputSigmoid.InputShape, outputSigmoid.OutputShape)
	}
}

func TestSigmoidParameterCounting(t *testing.T) {
	inputShape := []int{1, 10}
	builder := NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(5, true, "dense1").    // 10*5 + 5 = 55 parameters
		AddSigmoid("sigmoid1").         // 0 parameters
		AddDense(1, false, "output").   // 5*1 + 0 = 5 parameters
		AddSigmoid("output_sigmoid").   // 0 parameters
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	expectedParams := int64(55 + 5) // Total should be 60
	if model.TotalParameters != expectedParams {
		t.Errorf("Expected %d total parameters, got %d", expectedParams, model.TotalParameters)
	}
	
	// Check individual layers
	sigmoid1 := model.Layers[1]
	if sigmoid1.ParameterCount != 0 {
		t.Errorf("Expected Sigmoid layer to have 0 parameters, got %d", sigmoid1.ParameterCount)
	}
	
	outputSigmoid := model.Layers[3]
	if outputSigmoid.ParameterCount != 0 {
		t.Errorf("Expected output Sigmoid layer to have 0 parameters, got %d", outputSigmoid.ParameterCount)
	}
}

func equalIntSlices(a, b []int) bool {
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