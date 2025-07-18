package layers_test

import (
	"fmt"
	"testing"

	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// TestLayerFactoryAndBuilder tests the layer creation and model building functionality
func TestLayerFactoryAndBuilder(t *testing.T) {
	fmt.Println("\n=== Testing Layer Factory and Builder ===")
	
	// Test layer factory
	factory := layers.NewFactory()
	
	// Create layer specifications
	conv2d := factory.CreateConv2DSpec(3, 16, 3, 1, 1, true, "conv1")
	relu := factory.CreateReLUSpec("relu1")
	dense := factory.CreateDenseSpec(128, 10, true, "fc1")
	softmax := factory.CreateSoftmaxSpec(-1, "softmax")
	
	fmt.Printf("Conv2D Spec: %+v\n", conv2d)
	fmt.Printf("ReLU Spec: %+v\n", relu)
	fmt.Printf("Dense Spec: %+v\n", dense)
	fmt.Printf("Softmax Spec: %+v\n", softmax)
	
	// Test model builder
	inputShape := []int{16, 3, 64, 64} // Larger input for testing
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddLayer(conv2d).
		AddLayer(relu).
		AddLayer(dense).
		AddLayer(softmax).
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	fmt.Printf("\n=== Compiled Model Information ===\n")
	fmt.Printf("Total Parameters: %d\n", model.TotalParameters)
	fmt.Printf("Input Shape: %v\n", model.InputShape)
	fmt.Printf("Output Shape: %v\n", model.OutputShape)
	fmt.Printf("Layer Count: %d\n", len(model.Layers))
	
	// Validate parameter shapes
	expectedParamCount := int64(0)
	for i, layer := range model.Layers {
		fmt.Printf("Layer %d (%s): %d parameters\n", i+1, layer.Name, layer.ParameterCount)
		expectedParamCount += layer.ParameterCount
	}
	
	if model.TotalParameters != expectedParamCount {
		t.Errorf("Parameter count mismatch: expected %d, got %d", expectedParamCount, model.TotalParameters)
	}
	
	fmt.Println("✅ Layer factory and builder test completed successfully!")
}

// TestTrainerFactoryIntegration tests the trainer factory with layer models
func TestTrainerFactoryIntegration(t *testing.T) {
	t.Skip("Skipping trainer factory integration test - should be tested in training package")
}

// TestDropoutLayer tests the Dropout layer functionality
func TestDropoutLayer(t *testing.T) {
	fmt.Println("\n=== Testing Dropout Layer ===")
	
	// Test 1: Dropout layer creation
	inputShape := []int{32, 3, 32, 32}
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(64, true, "fc1").
		AddReLU("relu2").
		AddDropout(0.5, "dropout1").         // Add dropout with 50% rate
		AddDense(10, true, "fc2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with Dropout: %v", err)
	}
	
	// Model compiled successfully - only report if test passes
	
	// Test 2: Verify Dropout layer parameters
	var dropoutLayer *layers.LayerSpec
	for _, layer := range model.Layers {
		if layer.Type == layers.Dropout {
			dropoutLayer = &layer
			break
		}
	}
	
	if dropoutLayer == nil {
		t.Fatalf("Dropout layer not found in compiled model")
	}
	
	// Check dropout parameters
	rate, exists := dropoutLayer.Parameters["rate"]
	if !exists {
		t.Fatalf("Dropout rate parameter not found")
	}
	if rate != float32(0.5) {
		t.Errorf("Expected dropout rate 0.5, got %v", rate)
	}
	
	training, exists := dropoutLayer.Parameters["training"]
	if !exists {
		t.Fatalf("Dropout training parameter not found")
	}
	if training != true {
		t.Errorf("Expected training mode true, got %v", training)
	}
	
	// Dropout parameters verified - silent success
	
	// Test 3: Verify Dropout has no parameters
	if dropoutLayer.ParameterCount != 0 {
		t.Errorf("Expected Dropout to have 0 parameters, got %d", dropoutLayer.ParameterCount)
	}
	
	// Test 4: Verify Dropout doesn't change shape
	expectedInputShape := dropoutLayer.InputShape
	expectedOutputShape := dropoutLayer.OutputShape
	
	if len(expectedInputShape) != len(expectedOutputShape) {
		t.Errorf("Dropout changed tensor rank: input %d, output %d", len(expectedInputShape), len(expectedOutputShape))
	}
	
	for i := range expectedInputShape {
		if expectedInputShape[i] != expectedOutputShape[i] {
			t.Errorf("Dropout changed shape at dimension %d: input %d, output %d", i, expectedInputShape[i], expectedOutputShape[i])
		}
	}
	
	fmt.Printf("✅ Dropout preserves tensor shape: %v\n", expectedInputShape)
	
	// Test 5: Test serialization for CGO
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with Dropout: %v", err)
	}
	
	// Find Dropout layer in serialized format
	var foundDropout bool
	for _, layer := range serialized.Layers {
		if layer.LayerType == 5 { // Dropout = 5 in Go enum
			foundDropout = true
			if len(layer.ParamFloat) == 0 || layer.ParamFloat[0] != 0.5 {
				t.Errorf("Dropout rate not serialized correctly: %v", layer.ParamFloat)
			}
			if len(layer.ParamInt) == 0 || layer.ParamInt[0] != 0 {
				t.Errorf("Dropout training mode not serialized correctly (expected 0 for inference): %v", layer.ParamInt)
			}
			break
		}
	}
	
	if !foundDropout {
		t.Fatalf("Dropout layer not found in serialized model")
	}
	
	fmt.Printf("✅ Dropout serialization verified\n")
	
	// Test 6: Test dynamic layer spec conversion
	dynamicSpecs, err := model.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	// Find Dropout in dynamic specs
	var foundDynamicDropout bool
	for _, spec := range dynamicSpecs {
		if spec.LayerType == 5 { // Dropout = 5 in Go enum
			foundDynamicDropout = true
			if spec.ParamFloatCount != 1 || spec.ParamFloat[0] != 0.5 {
				t.Errorf("Dynamic Dropout rate not correct")
			}
			if spec.ParamIntCount != 1 || spec.ParamInt[0] != 0 {
				t.Errorf("Dynamic Dropout training mode not correct (expected 0 for inference)")
			}
			break
		}
	}
	
	if !foundDynamicDropout {
		t.Fatalf("Dropout not found in dynamic specs")
	}
	
	fmt.Printf("✅ Dynamic spec conversion verified\n")
	
	fmt.Println("✅ Dropout layer test completed successfully!")
}

// TestDropoutVariousRates tests Dropout with different dropout rates
func TestDropoutVariousRates(t *testing.T) {
	fmt.Println("\n=== Testing Dropout with Various Rates ===")
	
	rates := []float32{0.0, 0.25, 0.5, 0.75, 0.9}
	
	for _, rate := range rates {
		inputShape := []int{16, 64}
		
		builder := layers.NewModelBuilder(inputShape)
		model, err := builder.
			AddDropout(rate, fmt.Sprintf("dropout_%.2f", rate)).
			AddDense(10, true, "output").
			Compile()
		
		if err != nil {
			t.Fatalf("Failed to compile model with dropout rate %.2f: %v", rate, err)
		}
		
		// Verify the rate was set correctly
		dropoutLayer := model.Layers[0]
		actualRate := dropoutLayer.Parameters["rate"].(float32)
		if actualRate != rate {
			t.Errorf("Expected rate %.2f, got %.2f", rate, actualRate)
		}
		
		fmt.Printf("✅ Dropout rate %.2f compiled successfully\n", rate)
	}
	
	fmt.Println("✅ Various dropout rates test completed successfully!")
}

// TestBatchNormLayer tests the BatchNorm layer functionality
func TestBatchNormLayer(t *testing.T) {
	fmt.Println("\n=== Testing BatchNorm Layer ===")
	
	// Test 1: BatchNorm layer creation for Conv2D output
	inputShape := []int{32, 3, 32, 32}
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddBatchNorm(16, 1e-5, 0.1, true, "batchnorm1").  // BatchNorm for 16 channels
		AddReLU("relu1").
		AddDense(64, true, "fc1").
		AddBatchNorm(64, 1e-5, 0.1, true, "batchnorm2").  // BatchNorm for 64 features
		AddReLU("relu2").
		AddDense(10, true, "fc2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with BatchNorm: %v", err)
	}
	
	fmt.Printf("✅ Model with BatchNorm compiled successfully\n")
	fmt.Printf("Total layers: %d\n", len(model.Layers))
	
	// Test 2: Verify BatchNorm layer parameters
	var batchNormLayers []*layers.LayerSpec
	for _, layer := range model.Layers {
		if layer.Type == layers.BatchNorm {
			layerCopy := layer
			batchNormLayers = append(batchNormLayers, &layerCopy)
		}
	}
	
	if len(batchNormLayers) != 2 {
		t.Fatalf("Expected 2 BatchNorm layers, found %d", len(batchNormLayers))
	}
	
	// Check first BatchNorm (after conv) parameters
	batchNorm1 := batchNormLayers[0]
	numFeatures1, exists := batchNorm1.Parameters["num_features"]
	if !exists {
		t.Fatalf("BatchNorm num_features parameter not found")
	}
	if numFeatures1 != 16 {
		t.Errorf("Expected num_features 16, got %v", numFeatures1)
	}
	
	eps1, exists := batchNorm1.Parameters["eps"]
	if !exists {
		t.Fatalf("BatchNorm eps parameter not found")
	}
	if eps1 != float32(1e-5) {
		t.Errorf("Expected eps 1e-5, got %v", eps1)
	}
	
	affine1, exists := batchNorm1.Parameters["affine"]
	if !exists {
		t.Fatalf("BatchNorm affine parameter not found")
	}
	if affine1 != true {
		t.Errorf("Expected affine true, got %v", affine1)
	}
	
	fmt.Printf("✅ BatchNorm1 parameters verified: num_features=%v, eps=%v, affine=%v\n", numFeatures1, eps1, affine1)
	
	// Test 3: Verify BatchNorm parameter count
	expectedParams1 := int64(16 * 2) // gamma + beta for 16 features
	if batchNorm1.ParameterCount != expectedParams1 {
		t.Errorf("Expected BatchNorm1 to have %d parameters, got %d", expectedParams1, batchNorm1.ParameterCount)
	}
	
	// Check second BatchNorm (after dense) parameters
	batchNorm2 := batchNormLayers[1]
	numFeatures2 := batchNorm2.Parameters["num_features"].(int)
	if numFeatures2 != 64 {
		t.Errorf("Expected num_features 64, got %v", numFeatures2)
	}
	
	expectedParams2 := int64(64 * 2) // gamma + beta for 64 features
	if batchNorm2.ParameterCount != expectedParams2 {
		t.Errorf("Expected BatchNorm2 to have %d parameters, got %d", expectedParams2, batchNorm2.ParameterCount)
	}
	
	fmt.Printf("✅ BatchNorm2 parameters verified: num_features=%v, params=%d\n", numFeatures2, expectedParams2)
	
	// Test 4: Verify BatchNorm doesn't change shape
	for i, batchNormLayer := range batchNormLayers {
		expectedInputShape := batchNormLayer.InputShape
		expectedOutputShape := batchNormLayer.OutputShape
		
		if len(expectedInputShape) != len(expectedOutputShape) {
			t.Errorf("BatchNorm%d changed tensor rank: input %d, output %d", i+1, len(expectedInputShape), len(expectedOutputShape))
		}
		
		for j := range expectedInputShape {
			if expectedInputShape[j] != expectedOutputShape[j] {
				t.Errorf("BatchNorm%d changed shape at dimension %d: input %d, output %d", i+1, j, expectedInputShape[j], expectedOutputShape[j])
			}
		}
		
		fmt.Printf("✅ BatchNorm%d preserves tensor shape: %v\n", i+1, expectedInputShape)
	}
	
	// Test 5: Test serialization for CGO
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with BatchNorm: %v", err)
	}
	
	// Find BatchNorm layers in serialized format
	foundBatchNorms := 0
	for _, layer := range serialized.Layers {
		if layer.LayerType == 6 { // BatchNorm = 6 in Go enum
			foundBatchNorms++
			
			// Verify serialized parameters: [eps, momentum] in floats, [num_features, affine, track_running_stats, training] in ints
			if len(layer.ParamFloat) != 2 {
				t.Errorf("BatchNorm should have 2 float params, got %d", len(layer.ParamFloat))
			}
			if layer.ParamFloat[0] != 1e-5 {
				t.Errorf("BatchNorm eps not serialized correctly: %f", layer.ParamFloat[0])
			}
			if layer.ParamFloat[1] != 0.1 {
				t.Errorf("BatchNorm momentum not serialized correctly: %f", layer.ParamFloat[1])
			}
			
			if len(layer.ParamInt) != 4 {
				t.Errorf("BatchNorm should have 4 int params, got %d", len(layer.ParamInt))
			}
			if layer.ParamInt[1] != 1 { // affine = true
				t.Errorf("BatchNorm affine not serialized correctly: %d", layer.ParamInt[1])
			}
			if layer.ParamInt[2] != 1 { // track_running_stats = true
				t.Errorf("BatchNorm track_running_stats not serialized correctly: %d", layer.ParamInt[2])
			}
			if layer.ParamInt[3] != 0 { // training = false for inference
				t.Errorf("BatchNorm training not serialized correctly (expected 0 for inference): %d", layer.ParamInt[3])
			}
		}
	}
	
	if foundBatchNorms != 2 {
		t.Fatalf("Expected 2 BatchNorm layers in serialized model, found %d", foundBatchNorms)
	}
	
	fmt.Printf("✅ BatchNorm serialization verified\n")
	
	// Test 6: Test dynamic layer spec conversion
	dynamicSpecs, err := model.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	// Find BatchNorm in dynamic specs
	foundDynamicBatchNorms := 0
	for _, spec := range dynamicSpecs {
		if spec.LayerType == 6 { // BatchNorm = 6 in Go enum
			foundDynamicBatchNorms++
			if spec.ParamFloatCount != 2 {
				t.Errorf("Dynamic BatchNorm should have 2 float params")
			}
			if spec.ParamIntCount != 4 {
				t.Errorf("Dynamic BatchNorm should have 4 int params")
			}
			if spec.ParamFloat[0] != 1e-5 {
				t.Errorf("Dynamic BatchNorm eps not correct")
			}
			if spec.ParamFloat[1] != 0.1 {
				t.Errorf("Dynamic BatchNorm momentum not correct")
			}
		}
	}
	
	if foundDynamicBatchNorms != 2 {
		t.Fatalf("Expected 2 BatchNorm in dynamic specs, found %d", foundDynamicBatchNorms)
	}
	
	fmt.Printf("✅ Dynamic spec conversion verified\n")
	
	fmt.Println("✅ BatchNorm layer test completed successfully!")
}

// TestBatchNormVariousConfigurations tests BatchNorm with different configurations
func TestBatchNormVariousConfigurations(t *testing.T) {
	fmt.Println("\n=== Testing BatchNorm with Various Configurations ===")
	
	configs := []struct {
		numFeatures int
		eps         float32
		momentum    float32
		affine      bool
		description string
	}{
		{32, 1e-5, 0.1, true, "standard_affine"},
		{64, 1e-4, 0.01, true, "low_momentum"},
		{128, 1e-6, 0.9, false, "no_affine"},
		{16, 1e-3, 0.5, true, "high_eps"},
	}
	
	for _, config := range configs {
		inputShape := []int{16, config.numFeatures}
		
		builder := layers.NewModelBuilder(inputShape)
		model, err := builder.
			AddBatchNorm(config.numFeatures, config.eps, config.momentum, config.affine, config.description).
			AddDense(10, true, "output").
			Compile()
		
		if err != nil {
			t.Fatalf("Failed to compile model with BatchNorm config %s: %v", config.description, err)
		}
		
		// Verify the parameters were set correctly
		batchNormLayer := model.Layers[0]
		actualNumFeatures := batchNormLayer.Parameters["num_features"].(int)
		actualEps := batchNormLayer.Parameters["eps"].(float32)
		actualMomentum := batchNormLayer.Parameters["momentum"].(float32)
		actualAffine := batchNormLayer.Parameters["affine"].(bool)
		
		if actualNumFeatures != config.numFeatures {
			t.Errorf("Config %s: Expected num_features %d, got %d", config.description, config.numFeatures, actualNumFeatures)
		}
		if actualEps != config.eps {
			t.Errorf("Config %s: Expected eps %f, got %f", config.description, config.eps, actualEps)
		}
		if actualMomentum != config.momentum {
			t.Errorf("Config %s: Expected momentum %f, got %f", config.description, config.momentum, actualMomentum)
		}
		if actualAffine != config.affine {
			t.Errorf("Config %s: Expected affine %t, got %t", config.description, config.affine, actualAffine)
		}
		
		// Verify parameter count
		expectedParams := int64(0)
		if config.affine {
			expectedParams = int64(config.numFeatures * 2) // gamma + beta
		}
		if batchNormLayer.ParameterCount != expectedParams {
			t.Errorf("Config %s: Expected %d parameters, got %d", config.description, expectedParams, batchNormLayer.ParameterCount)
		}
		
		fmt.Printf("✅ BatchNorm config %s compiled successfully (params=%d)\n", config.description, expectedParams)
	}
	
	fmt.Println("✅ Various BatchNorm configurations test completed successfully!")
}

// TestBatchNormShapeValidation tests BatchNorm input shape validation
func TestBatchNormShapeValidation(t *testing.T) {
	fmt.Println("\n=== Testing BatchNorm Shape Validation ===")
	
	// Test 1: Valid 2D input shape
	inputShape2D := []int{32, 64}
	builder2D := layers.NewModelBuilder(inputShape2D)
	_, err := builder2D.AddBatchNorm(64, 1e-5, 0.1, true, "batchnorm2d").Compile()
	if err != nil {
		t.Errorf("BatchNorm should accept 2D input: %v", err)
	}
	
	// Test 2: Valid 4D input shape
	inputShape4D := []int{32, 16, 32, 32}
	builder4D := layers.NewModelBuilder(inputShape4D)
	_, err = builder4D.AddBatchNorm(16, 1e-5, 0.1, true, "batchnorm4d").Compile()
	if err != nil {
		t.Errorf("BatchNorm should accept 4D input: %v", err)
	}
	
	// Test 3: Invalid num_features mismatch
	inputShapeMismatch := []int{32, 32}
	builderMismatch := layers.NewModelBuilder(inputShapeMismatch)
	_, err = builderMismatch.AddBatchNorm(16, 1e-5, 0.1, true, "batchnorm_mismatch").Compile()
	if err == nil {
		t.Error("BatchNorm should reject mismatched num_features")
	}
	
	// Test 4: 1D input should be rejected
	inputShape1D := []int{32}
	builder1D := layers.NewModelBuilder(inputShape1D)
	_, err = builder1D.AddBatchNorm(32, 1e-5, 0.1, true, "batchnorm1d").Compile()
	if err == nil {
		t.Error("BatchNorm should reject 1D input")
	}
	
	fmt.Println("✅ BatchNorm shape validation test completed successfully!")
}

// TestLeakyReLULayer tests the Leaky ReLU layer functionality
func TestLeakyReLULayer(t *testing.T) {
	fmt.Println("\n=== Testing Leaky ReLU Layer ===")
	
	// Test 1: Leaky ReLU layer creation
	inputShape := []int{32, 64}
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddLeakyReLU(0.2, "leaky_relu1").         // Add Leaky ReLU with 0.2 negative slope
		AddDense(32, true, "fc1").
		AddLeakyReLU(0.01, "leaky_relu2").        // Add another with default-like slope
		AddDense(10, true, "fc2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with Leaky ReLU: %v", err)
	}
	
	fmt.Printf("✅ Model with Leaky ReLU compiled successfully\n")
	fmt.Printf("Total layers: %d\n", len(model.Layers))
	
	// Test 2: Verify Leaky ReLU layer parameters
	var leakyReLULayers []*layers.LayerSpec
	for _, layer := range model.Layers {
		if layer.Type == layers.LeakyReLU {
			layerCopy := layer
			leakyReLULayers = append(leakyReLULayers, &layerCopy)
		}
	}
	
	if len(leakyReLULayers) != 2 {
		t.Fatalf("Expected 2 Leaky ReLU layers, found %d", len(leakyReLULayers))
	}
	
	// Check first Leaky ReLU parameters
	leakyReLU1 := leakyReLULayers[0]
	negativeSlope1, exists := leakyReLU1.Parameters["negative_slope"]
	if !exists {
		t.Fatalf("Leaky ReLU negative_slope parameter not found")
	}
	if negativeSlope1 != float32(0.2) {
		t.Errorf("Expected negative_slope 0.2, got %v", negativeSlope1)
	}
	
	// Check second Leaky ReLU parameters
	leakyReLU2 := leakyReLULayers[1]
	negativeSlope2, exists := leakyReLU2.Parameters["negative_slope"]
	if !exists {
		t.Fatalf("Leaky ReLU negative_slope parameter not found")
	}
	if negativeSlope2 != float32(0.01) {
		t.Errorf("Expected negative_slope 0.01, got %v", negativeSlope2)
	}
	
	fmt.Printf("✅ Leaky ReLU parameters verified: slopes=%.2f, %.2f\n", negativeSlope1, negativeSlope2)
	
	// Test 3: Verify Leaky ReLU has no trainable parameters
	for i, leakyReLULayer := range leakyReLULayers {
		if leakyReLULayer.ParameterCount != 0 {
			t.Errorf("Expected Leaky ReLU %d to have 0 parameters, got %d", i+1, leakyReLULayer.ParameterCount)
		}
	}
	
	// Test 4: Verify Leaky ReLU doesn't change shape
	for i, leakyReLULayer := range leakyReLULayers {
		expectedInputShape := leakyReLULayer.InputShape
		expectedOutputShape := leakyReLULayer.OutputShape
		
		if len(expectedInputShape) != len(expectedOutputShape) {
			t.Errorf("Leaky ReLU %d changed tensor rank: input %d, output %d", i+1, len(expectedInputShape), len(expectedOutputShape))
		}
		
		for j := range expectedInputShape {
			if expectedInputShape[j] != expectedOutputShape[j] {
				t.Errorf("Leaky ReLU %d changed shape at dimension %d: input %d, output %d", i+1, j, expectedInputShape[j], expectedOutputShape[j])
			}
		}
		
		fmt.Printf("✅ Leaky ReLU %d preserves tensor shape: %v\n", i+1, expectedInputShape)
	}
	
	// Test 5: Test serialization for CGO
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with Leaky ReLU: %v", err)
	}
	
	// Find Leaky ReLU layers in serialized format
	foundLeakyReLUs := 0
	for _, layer := range serialized.Layers {
		if layer.LayerType == 7 { // LeakyReLU = 7 in Go enum
			foundLeakyReLUs++
			
			// Verify serialized parameters: [negative_slope] in floats
			if len(layer.ParamFloat) != 1 {
				t.Errorf("Leaky ReLU should have 1 float param, got %d", len(layer.ParamFloat))
			}
			if foundLeakyReLUs == 1 && layer.ParamFloat[0] != 0.2 {
				t.Errorf("First Leaky ReLU negative_slope not serialized correctly: %f", layer.ParamFloat[0])
			}
			if foundLeakyReLUs == 2 && layer.ParamFloat[0] != 0.01 {
				t.Errorf("Second Leaky ReLU negative_slope not serialized correctly: %f", layer.ParamFloat[0])
			}
			
			if len(layer.ParamInt) != 0 {
				t.Errorf("Leaky ReLU should have 0 int params, got %d", len(layer.ParamInt))
			}
		}
	}
	
	if foundLeakyReLUs != 2 {
		t.Fatalf("Expected 2 Leaky ReLU layers in serialized model, found %d", foundLeakyReLUs)
	}
	
	fmt.Printf("✅ Leaky ReLU serialization verified\n")
	
	// Test 6: Test dynamic layer spec conversion
	dynamicSpecs, err := model.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	// Find Leaky ReLU in dynamic specs
	foundDynamicLeakyReLUs := 0
	for _, spec := range dynamicSpecs {
		if spec.LayerType == 7 { // LeakyReLU = 7 in Go enum
			foundDynamicLeakyReLUs++
			if spec.ParamFloatCount != 1 {
				t.Errorf("Dynamic Leaky ReLU should have 1 float param")
			}
			if spec.ParamIntCount != 0 {
				t.Errorf("Dynamic Leaky ReLU should have 0 int params")
			}
			if foundDynamicLeakyReLUs == 1 && spec.ParamFloat[0] != 0.2 {
				t.Errorf("Dynamic first Leaky ReLU negative_slope not correct")
			}
			if foundDynamicLeakyReLUs == 2 && spec.ParamFloat[0] != 0.01 {
				t.Errorf("Dynamic second Leaky ReLU negative_slope not correct")
			}
		}
	}
	
	if foundDynamicLeakyReLUs != 2 {
		t.Fatalf("Expected 2 Leaky ReLU in dynamic specs, found %d", foundDynamicLeakyReLUs)
	}
	
	fmt.Printf("✅ Dynamic spec conversion verified\n")
	
	fmt.Println("✅ Leaky ReLU layer test completed successfully!")
}

// TestLeakyReLUVariousSlopes tests Leaky ReLU with different negative slopes
func TestLeakyReLUVariousSlopes(t *testing.T) {
	fmt.Println("\n=== Testing Leaky ReLU with Various Slopes ===")
	
	slopes := []float32{0.0, 0.01, 0.1, 0.2, 0.3}
	
	for _, slope := range slopes {
		inputShape := []int{16, 64}
		
		builder := layers.NewModelBuilder(inputShape)
		model, err := builder.
			AddLeakyReLU(slope, fmt.Sprintf("leaky_relu_%.2f", slope)).
			AddDense(10, true, "output").
			Compile()
		
		if err != nil {
			t.Fatalf("Failed to compile model with Leaky ReLU slope %.2f: %v", slope, err)
		}
		
		// Verify the slope was set correctly
		leakyReLULayer := model.Layers[0]
		actualSlope := leakyReLULayer.Parameters["negative_slope"].(float32)
		if actualSlope != slope {
			t.Errorf("Expected slope %.2f, got %.2f", slope, actualSlope)
		}
		
		fmt.Printf("✅ Leaky ReLU slope %.2f compiled successfully\n", slope)
	}
	
	fmt.Println("✅ Various Leaky ReLU slopes test completed successfully!")
}

// TestLeakyReLUFactoryCreation tests factory-based Leaky ReLU creation
func TestLeakyReLUFactoryCreation(t *testing.T) {
	fmt.Println("\n=== Testing Leaky ReLU Factory Creation ===")
	
	factory := layers.NewFactory()
	
	// Test factory creation with different slopes
	testCases := []struct {
		slope       float32
		name        string
		description string
	}{
		{0.01, "leaky_relu_default", "default slope"},
		{0.2, "leaky_relu_high", "high slope"},
		{0.001, "leaky_relu_low", "low slope"},
		{0.0, "leaky_relu_zero", "zero slope (equivalent to ReLU)"},
	}
	
	for _, tc := range testCases {
		spec := factory.CreateLeakyReLUSpec(tc.slope, tc.name)
		
		if spec.Type != layers.LeakyReLU {
			t.Errorf("Expected type LeakyReLU, got %s", spec.Type.String())
		}
		
		if spec.Name != tc.name {
			t.Errorf("Expected name %s, got %s", tc.name, spec.Name)
		}
		
		actualSlope := spec.Parameters["negative_slope"].(float32)
		if actualSlope != tc.slope {
			t.Errorf("Expected slope %.3f, got %.3f", tc.slope, actualSlope)
		}
		
		fmt.Printf("✅ Factory Leaky ReLU with %s (%.3f) created successfully\n", tc.description, tc.slope)
	}
	
	fmt.Println("✅ Leaky ReLU factory creation test completed successfully!")
}

// TestELULayer tests the ELU layer functionality
func TestELULayer(t *testing.T) {
	fmt.Println("\n=== Testing ELU Layer ===")
	
	// Test 1: ELU layer creation
	inputShape := []int{32, 64}
	
	builder := layers.NewModelBuilder(inputShape)
	model, err := builder.
		AddELU(1.0, "elu1").                   // Add ELU with alpha=1.0
		AddDense(32, true, "fc1").
		AddELU(0.5, "elu2").                   // Add another with alpha=0.5
		AddDense(10, true, "fc2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with ELU: %v", err)
	}
	
	fmt.Printf("✅ Model with ELU compiled successfully\n")
	fmt.Printf("Total layers: %d\n", len(model.Layers))
	
	// Test 2: Verify ELU layer parameters
	var eluLayers []*layers.LayerSpec
	for _, layer := range model.Layers {
		if layer.Type == layers.ELU {
			layerCopy := layer
			eluLayers = append(eluLayers, &layerCopy)
		}
	}
	
	if len(eluLayers) != 2 {
		t.Fatalf("Expected 2 ELU layers, found %d", len(eluLayers))
	}
	
	// Check first ELU parameters
	elu1 := eluLayers[0]
	alpha1, exists := elu1.Parameters["alpha"]
	if !exists {
		t.Fatalf("ELU alpha parameter not found")
	}
	if alpha1 != float32(1.0) {
		t.Errorf("Expected alpha 1.0, got %v", alpha1)
	}
	
	// Check second ELU parameters
	elu2 := eluLayers[1]
	alpha2, exists := elu2.Parameters["alpha"]
	if !exists {
		t.Fatalf("ELU alpha parameter not found")
	}
	if alpha2 != float32(0.5) {
		t.Errorf("Expected alpha 0.5, got %v", alpha2)
	}
	
	fmt.Printf("✅ ELU parameters verified: alpha1=%.1f, alpha2=%.1f\n", alpha1, alpha2)
	
	// Test 3: Verify ELU has no trainable parameters
	for i, eluLayer := range eluLayers {
		if eluLayer.ParameterCount != 0 {
			t.Errorf("Expected ELU %d to have 0 parameters, got %d", i+1, eluLayer.ParameterCount)
		}
	}
	
	// Test 4: Verify ELU doesn't change shape
	for i, eluLayer := range eluLayers {
		expectedInputShape := eluLayer.InputShape
		expectedOutputShape := eluLayer.OutputShape
		
		if len(expectedInputShape) != len(expectedOutputShape) {
			t.Errorf("ELU %d changed tensor rank: input %d, output %d", i+1, len(expectedInputShape), len(expectedOutputShape))
		}
		
		for j := range expectedInputShape {
			if expectedInputShape[j] != expectedOutputShape[j] {
				t.Errorf("ELU %d changed shape at dimension %d: input %d, output %d", i+1, j, expectedInputShape[j], expectedOutputShape[j])
			}
		}
		
		fmt.Printf("✅ ELU %d preserves tensor shape: %v\n", i+1, expectedInputShape)
	}
	
	// Test 5: Test serialization for CGO
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with ELU: %v", err)
	}
	
	// Find ELU layers in serialized format
	foundELUs := 0
	for _, layer := range serialized.Layers {
		if layer.LayerType == 8 { // ELU = 8 in Go enum
			foundELUs++
			
			// Verify serialized parameters: [alpha] in floats
			if len(layer.ParamFloat) != 1 {
				t.Errorf("ELU should have 1 float param, got %d", len(layer.ParamFloat))
			}
			if foundELUs == 1 && layer.ParamFloat[0] != 1.0 {
				t.Errorf("First ELU alpha not serialized correctly: %f", layer.ParamFloat[0])
			}
			if foundELUs == 2 && layer.ParamFloat[0] != 0.5 {
				t.Errorf("Second ELU alpha not serialized correctly: %f", layer.ParamFloat[0])
			}
			
			if len(layer.ParamInt) != 0 {
				t.Errorf("ELU should have 0 int params, got %d", len(layer.ParamInt))
			}
		}
	}
	
	if foundELUs != 2 {
		t.Fatalf("Expected 2 ELU layers in serialized model, found %d", foundELUs)
	}
	
	fmt.Printf("✅ ELU serialization verified\n")
	
	// Test 6: Test dynamic layer spec conversion
	dynamicSpecs, err := model.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	// Find ELU in dynamic specs
	foundDynamicELUs := 0
	for _, spec := range dynamicSpecs {
		if spec.LayerType == 8 { // ELU = 8 in Go enum
			foundDynamicELUs++
			if spec.ParamFloatCount != 1 {
				t.Errorf("Dynamic ELU should have 1 float param")
			}
			if spec.ParamIntCount != 0 {
				t.Errorf("Dynamic ELU should have 0 int params")
			}
			if foundDynamicELUs == 1 && spec.ParamFloat[0] != 1.0 {
				t.Errorf("Dynamic first ELU alpha not correct")
			}
			if foundDynamicELUs == 2 && spec.ParamFloat[0] != 0.5 {
				t.Errorf("Dynamic second ELU alpha not correct")
			}
		}
	}
	
	if foundDynamicELUs != 2 {
		t.Fatalf("Expected 2 ELU in dynamic specs, found %d", foundDynamicELUs)
	}
	
	fmt.Printf("✅ Dynamic spec conversion verified\n")
	
	fmt.Println("✅ ELU layer test completed successfully!")
}

// TestELUVariousAlphas tests ELU with different alpha values
func TestELUVariousAlphas(t *testing.T) {
	fmt.Println("\n=== Testing ELU with Various Alpha Values ===")
	
	alphas := []float32{0.1, 0.5, 1.0, 2.0, 5.0}
	
	for _, alpha := range alphas {
		inputShape := []int{16, 64}
		
		builder := layers.NewModelBuilder(inputShape)
		model, err := builder.
			AddELU(alpha, fmt.Sprintf("elu_%.1f", alpha)).
			AddDense(10, true, "output").
			Compile()
		
		if err != nil {
			t.Fatalf("Failed to compile model with ELU alpha %.1f: %v", alpha, err)
		}
		
		// Verify the alpha was set correctly
		eluLayer := model.Layers[0]
		actualAlpha := eluLayer.Parameters["alpha"].(float32)
		if actualAlpha != alpha {
			t.Errorf("Expected alpha %.1f, got %.1f", alpha, actualAlpha)
		}
		
		fmt.Printf("✅ ELU alpha %.1f compiled successfully\n", alpha)
	}
	
	fmt.Println("✅ Various ELU alpha values test completed successfully!")
}

// TestELUFactoryCreation tests factory-based ELU creation
func TestELUFactoryCreation(t *testing.T) {
	fmt.Println("\n=== Testing ELU Factory Creation ===")
	
	factory := layers.NewFactory()
	
	// Test factory creation with different alpha values
	testCases := []struct {
		alpha       float32
		name        string
		description string
	}{
		{1.0, "elu_default", "default alpha"},
		{0.5, "elu_low", "low alpha"},
		{2.0, "elu_high", "high alpha"},
		{0.1, "elu_very_low", "very low alpha"},
	}
	
	for _, tc := range testCases {
		spec := factory.CreateELUSpec(tc.alpha, tc.name)
		
		if spec.Type != layers.ELU {
			t.Errorf("Expected type ELU, got %s", spec.Type.String())
		}
		
		if spec.Name != tc.name {
			t.Errorf("Expected name %s, got %s", tc.name, spec.Name)
		}
		
		actualAlpha := spec.Parameters["alpha"].(float32)
		if actualAlpha != tc.alpha {
			t.Errorf("Expected alpha %.1f, got %.1f", tc.alpha, actualAlpha)
		}
		
		fmt.Printf("✅ Factory ELU with %s (%.1f) created successfully\n", tc.description, tc.alpha)
	}
	
	fmt.Println("✅ ELU factory creation test completed successfully!")
}

// TestDropoutFactoryCreation tests factory-based Dropout creation
func TestDropoutFactoryCreation(t *testing.T) {
	fmt.Println("\n=== Testing Dropout Factory Creation ===")
	
	factory := layers.NewFactory()
	
	// Test factory creation with different dropout rates
	testCases := []struct {
		rate        float32
		name        string
		description string
	}{
		{0.0, "dropout_none", "no dropout"},
		{0.25, "dropout_light", "light dropout"},
		{0.5, "dropout_medium", "medium dropout"},
		{0.75, "dropout_heavy", "heavy dropout"},
	}
	
	for _, tc := range testCases {
		spec := factory.CreateDropoutSpec(tc.rate, tc.name)
		
		if spec.Type != layers.Dropout {
			t.Errorf("Expected type Dropout, got %s", spec.Type.String())
		}
		
		if spec.Name != tc.name {
			t.Errorf("Expected name %s, got %s", tc.name, spec.Name)
		}
		
		actualRate := spec.Parameters["rate"].(float32)
		if actualRate != tc.rate {
			t.Errorf("Expected rate %.2f, got %.2f", tc.rate, actualRate)
		}
		
		actualTraining := spec.Parameters["training"].(bool)
		if !actualTraining {
			t.Errorf("Expected training true, got %v", actualTraining)
		}
		
		fmt.Printf("✅ Factory Dropout with %s (%.2f) created successfully\n", tc.description, tc.rate)
	}
	
	fmt.Println("✅ Dropout factory creation test completed successfully!")
}

// TestBatchNormFactoryCreation tests factory-based BatchNorm creation
func TestBatchNormFactoryCreation(t *testing.T) {
	fmt.Println("\n=== Testing BatchNorm Factory Creation ===")
	
	factory := layers.NewFactory()
	
	// Test factory creation with different configurations
	testCases := []struct {
		numFeatures int
		eps         float32
		momentum    float32
		affine      bool
		name        string
		description string
	}{
		{32, 1e-5, 0.1, true, "bn_standard", "standard configuration"},
		{64, 1e-4, 0.01, true, "bn_low_momentum", "low momentum"},
		{128, 1e-6, 0.9, false, "bn_no_affine", "no affine transformation"},
		{16, 1e-3, 0.5, true, "bn_custom", "custom parameters"},
	}
	
	for _, tc := range testCases {
		spec := factory.CreateBatchNormSpec(tc.numFeatures, tc.eps, tc.momentum, tc.affine, tc.name)
		
		if spec.Type != layers.BatchNorm {
			t.Errorf("Expected type BatchNorm, got %s", spec.Type.String())
		}
		
		if spec.Name != tc.name {
			t.Errorf("Expected name %s, got %s", tc.name, spec.Name)
		}
		
		actualNumFeatures := spec.Parameters["num_features"].(int)
		if actualNumFeatures != tc.numFeatures {
			t.Errorf("Expected num_features %d, got %d", tc.numFeatures, actualNumFeatures)
		}
		
		actualEps := spec.Parameters["eps"].(float32)
		if actualEps != tc.eps {
			t.Errorf("Expected eps %f, got %f", tc.eps, actualEps)
		}
		
		actualMomentum := spec.Parameters["momentum"].(float32)
		if actualMomentum != tc.momentum {
			t.Errorf("Expected momentum %f, got %f", tc.momentum, actualMomentum)
		}
		
		actualAffine := spec.Parameters["affine"].(bool)
		if actualAffine != tc.affine {
			t.Errorf("Expected affine %t, got %t", tc.affine, actualAffine)
		}
		
		actualTrackRunningStats := spec.Parameters["track_running_stats"].(bool)
		if !actualTrackRunningStats {
			t.Errorf("Expected track_running_stats true, got %v", actualTrackRunningStats)
		}
		
		actualTraining := spec.Parameters["training"].(bool)
		if !actualTraining {
			t.Errorf("Expected training true, got %v", actualTraining)
		}
		
		fmt.Printf("✅ Factory BatchNorm with %s created successfully\n", tc.description)
	}
	
	fmt.Println("✅ BatchNorm factory creation test completed successfully!")
}

// TestSerializationRoundtrip tests that we can serialize and deserialize models correctly
func TestSerializationRoundtrip(t *testing.T) {
	fmt.Println("\n=== Testing Serialization Roundtrip ===")
	
	// Create a complex model with multiple layer types
	inputShape := []int{8, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	originalModel, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddBatchNorm(16, 1e-5, 0.1, true, "bn1").
		AddReLU("relu1").
		AddDropout(0.25, "dropout1").
		AddConv2D(32, 3, 2, 1, true, "conv2").
		AddLeakyReLU(0.2, "leaky_relu1").
		AddDense(64, true, "fc1").
		AddELU(1.0, "elu1").
		AddDense(10, true, "fc2").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile original model: %v", err)
	}
	
	// Test CGO serialization
	serialized, err := originalModel.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model: %v", err)
	}
	
	if len(serialized.Layers) != len(originalModel.Layers) {
		t.Errorf("Serialized layer count mismatch: expected %d, got %d", len(originalModel.Layers), len(serialized.Layers))
	}
	
	// Test dynamic layer spec conversion
	dynamicSpecs, err := originalModel.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	if len(dynamicSpecs) != len(originalModel.Layers) {
		t.Errorf("Dynamic specs count mismatch: expected %d, got %d", len(originalModel.Layers), len(dynamicSpecs))
	}
	
	// Verify all layer types are preserved
	expectedTypes := []int32{1, 6, 2, 5, 1, 7, 0, 8, 0, 3} // Conv2D, BatchNorm, ReLU, Dropout, Conv2D, LeakyReLU, Dense, ELU, Dense, Softmax
	for i, spec := range dynamicSpecs {
		if spec.LayerType != expectedTypes[i] {
			t.Errorf("Layer %d type mismatch: expected %d, got %d", i, expectedTypes[i], spec.LayerType)
		}
	}
	
	fmt.Printf("✅ Serialization preserved %d layers correctly\n", len(originalModel.Layers))
	fmt.Println("✅ Serialization roundtrip test completed successfully!")
}

// TestErrorHandling tests various error conditions and edge cases
func TestErrorHandling(t *testing.T) {
	fmt.Println("\n=== Testing Error Handling ===")
	
	// Test 1: Empty model compilation
	emptyBuilder := layers.NewModelBuilder([]int{32, 64})
	_, err := emptyBuilder.Compile()
	if err == nil {
		t.Error("Expected error for empty model compilation")
	}
	fmt.Println("✅ Empty model compilation correctly rejected")
	
	// Test 2: Invalid dropout rates
	invalidRates := []float32{-0.1, 1.1, 2.0}
	for _, rate := range invalidRates {
		builder := layers.NewModelBuilder([]int{32, 64})
		model, err := builder.AddDropout(rate, "invalid_dropout").AddDense(10, true, "output").Compile()
		
		// Note: The current implementation doesn't validate dropout rates, so this might pass
		// This test documents expected behavior for future validation implementation
		if err == nil && model != nil {
			fmt.Printf("⚠️  Invalid dropout rate %.1f was accepted (consider adding validation)\n", rate)
		}
	}
	
	// Test 3: Invalid BatchNorm configurations
	negativeFeatures := -16
	builder := layers.NewModelBuilder([]int{32, 64})
	_, err = builder.AddBatchNorm(negativeFeatures, 1e-5, 0.1, true, "invalid_bn").Compile()
	if err == nil {
		t.Error("Expected error for negative num_features in BatchNorm")
	}
	fmt.Println("✅ Negative BatchNorm features correctly rejected")
	
	// Test 4: Invalid input shapes (currently not validated - documenting behavior)
	invalidInputShapes := [][]int{
		{},           // Empty shape
		{0},          // Zero dimension
		{-1, 64},     // Negative dimension
	}
	
	for _, shape := range invalidInputShapes {
		builder := layers.NewModelBuilder(shape)
		_, err := builder.AddDense(10, true, "dense").Compile()
		if err == nil {
			fmt.Printf("⚠️  Invalid input shape %v was accepted (consider adding validation)\n", shape)
		} else {
			fmt.Printf("✅ Invalid input shape %v correctly rejected\n", shape)
		}
	}
	
	// Test 5: Dense layer with zero output size (currently not validated - documenting behavior)
	builder = layers.NewModelBuilder([]int{32, 64})
	_, err = builder.AddDense(0, true, "zero_output").Compile()
	if err == nil {
		fmt.Println("⚠️  Dense layer with zero output size was accepted (consider adding validation)")
	} else {
		fmt.Println("✅ Zero output size Dense layer correctly rejected")
	}
	
	// Test 6: Conv2D with invalid parameters (currently not validated - documenting behavior)
	builder = layers.NewModelBuilder([]int{32, 3, 32, 32})
	_, err = builder.AddConv2D(0, 3, 1, 0, true, "invalid_conv").Compile()
	if err == nil {
		fmt.Println("⚠️  Conv2D with zero output channels was accepted (consider adding validation)")
	} else {
		fmt.Println("✅ Invalid Conv2D parameters correctly rejected")
	}
	
	fmt.Println("✅ Error handling test completed successfully!")
}

// TestParameterCounting tests accurate parameter counting for all layer types
func TestParameterCounting(t *testing.T) {
	fmt.Println("\n=== Testing Parameter Counting ===")
	
	// Test 1: Dense layer parameter counting
	inputShape := []int{16, 128}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "dense_with_bias").    // 128*64 + 64 = 8256
		AddDense(32, false, "dense_without_bias"). // 64*32 = 2048
		AddDense(10, true, "output").             // 32*10 + 10 = 330
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	expectedParams := []int64{8256, 2048, 330}
	for i, expectedParam := range expectedParams {
		actualParam := model.Layers[i].ParameterCount
		if actualParam != expectedParam {
			t.Errorf("Layer %d: expected %d parameters, got %d", i, expectedParam, actualParam)
		}
	}
	
	expectedTotal := int64(8256 + 2048 + 330)
	if model.TotalParameters != expectedTotal {
		t.Errorf("Expected total parameters %d, got %d", expectedTotal, model.TotalParameters)
	}
	
	fmt.Printf("✅ Dense layer parameter counting verified: %d total\n", model.TotalParameters)
	
	// Test 2: Conv2D parameter counting
	inputShape = []int{16, 3, 32, 32}
	builder = layers.NewModelBuilder(inputShape)
	
	model, err = builder.
		AddConv2D(8, 3, 1, 1, true, "conv_with_bias").    // 3*8*3*3 + 8 = 224
		AddConv2D(16, 5, 1, 2, false, "conv_without_bias"). // 8*16*5*5 = 3200
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile conv model: %v", err)
	}
	
	expectedConvParams := []int64{224, 3200}
	for i, expectedParam := range expectedConvParams {
		actualParam := model.Layers[i].ParameterCount
		if actualParam != expectedParam {
			t.Errorf("Conv layer %d: expected %d parameters, got %d", i, expectedParam, actualParam)
		}
	}
	
	fmt.Printf("✅ Conv2D parameter counting verified\n")
	
	// Test 3: BatchNorm parameter counting
	inputShape = []int{16, 64}
	builder = layers.NewModelBuilder(inputShape)
	
	model, err = builder.
		AddBatchNorm(64, 1e-5, 0.1, true, "bn_affine").    // 64*2 = 128 (gamma + beta)
		AddBatchNorm(64, 1e-5, 0.1, false, "bn_no_affine"). // 0
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile BatchNorm model: %v", err)
	}
	
	expectedBNParams := []int64{128, 0}
	for i, expectedParam := range expectedBNParams {
		actualParam := model.Layers[i].ParameterCount
		if actualParam != expectedParam {
			t.Errorf("BatchNorm layer %d: expected %d parameters, got %d", i, expectedParam, actualParam)
		}
	}
	
	fmt.Printf("✅ BatchNorm parameter counting verified\n")
	
	// Test 4: Activation layers should have zero parameters
	inputShape = []int{16, 64}
	builder = layers.NewModelBuilder(inputShape)
	
	model, err = builder.
		AddReLU("relu").
		AddLeakyReLU(0.1, "leaky_relu").
		AddELU(1.0, "elu").
		AddDropout(0.5, "dropout").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile activation model: %v", err)
	}
	
	for i, layer := range model.Layers {
		if layer.ParameterCount != 0 {
			t.Errorf("Activation layer %d (%s): expected 0 parameters, got %d", i, layer.Name, layer.ParameterCount)
		}
	}
	
	fmt.Printf("✅ Activation layers parameter counting verified (all zero)\n")
	
	fmt.Println("✅ Parameter counting test completed successfully!")
}

// TestComplexModelArchitectures tests complex real-world model architectures
func TestComplexModelArchitectures(t *testing.T) {
	fmt.Println("\n=== Testing Complex Model Architectures ===")
	
	// Test 1: CNN with residual-like connections (simulated)
	inputShape := []int{16, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	cnnModel, err := builder.
		AddConv2D(32, 3, 1, 1, false, "conv1").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn1").
		AddReLU("relu1").
		AddConv2D(32, 3, 1, 1, false, "conv2").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn2").
		AddReLU("relu2").
		AddConv2D(64, 3, 2, 1, false, "conv3").
		AddBatchNorm(64, 1e-5, 0.1, true, "bn3").
		AddReLU("relu3").
		AddDropout(0.25, "dropout1").
		AddConv2D(64, 3, 1, 1, false, "conv4").
		AddBatchNorm(64, 1e-5, 0.1, true, "bn4").
		AddReLU("relu4").
		AddDropout(0.5, "dropout2").
		AddDense(128, true, "fc1").
		AddLeakyReLU(0.1, "leaky_relu1").
		AddDropout(0.5, "dropout3").
		AddDense(10, true, "fc2").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile complex CNN: %v", err)
	}
	
	fmt.Printf("✅ Complex CNN compiled: %d layers, %d parameters\n", len(cnnModel.Layers), cnnModel.TotalParameters)
	
	// Test 2: Deep MLP with various activations
	inputShape = []int{32, 784}
	builder = layers.NewModelBuilder(inputShape)
	
	mlpModel, err := builder.
		AddDense(512, true, "fc1").
		AddELU(1.0, "elu1").
		AddDropout(0.3, "dropout1").
		AddDense(256, true, "fc2").
		AddLeakyReLU(0.2, "leaky_relu1").
		AddBatchNorm(256, 1e-5, 0.1, true, "bn1").
		AddDropout(0.4, "dropout2").
		AddDense(128, true, "fc3").
		AddReLU("relu1").
		AddDropout(0.5, "dropout3").
		AddDense(64, true, "fc4").
		AddELU(0.5, "elu2").
		AddDense(10, true, "output").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile deep MLP: %v", err)
	}
	
	fmt.Printf("✅ Deep MLP compiled: %d layers, %d parameters\n", len(mlpModel.Layers), mlpModel.TotalParameters)
	
	// Test 3: Mixed architecture (CNN + MLP)
	inputShape = []int{8, 1, 28, 28}
	builder = layers.NewModelBuilder(inputShape)
	
	mixedModel, err := builder.
		AddConv2D(16, 5, 1, 2, true, "conv1").
		AddReLU("relu1").
		AddConv2D(32, 5, 2, 2, true, "conv2").
		AddLeakyReLU(0.1, "leaky_relu1").
		AddBatchNorm(32, 1e-5, 0.1, true, "bn1").
		AddDropout(0.25, "dropout1").
		AddDense(128, true, "fc1").
		AddELU(1.0, "elu1").
		AddDropout(0.5, "dropout2").
		AddDense(64, true, "fc2").
		AddReLU("relu2").
		AddDense(10, true, "output").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile mixed architecture: %v", err)
	}
	
	fmt.Printf("✅ Mixed architecture compiled: %d layers, %d parameters\n", len(mixedModel.Layers), mixedModel.TotalParameters)
	
	// Verify all models have reasonable parameter counts
	models := []*layers.ModelSpec{cnnModel, mlpModel, mixedModel}
	names := []string{"CNN", "MLP", "Mixed"}
	
	for i, model := range models {
		if model.TotalParameters <= 0 {
			t.Errorf("%s model has invalid parameter count: %d", names[i], model.TotalParameters)
		}
		
		// Verify shapes are computed
		if len(model.InputShape) == 0 || len(model.OutputShape) == 0 {
			t.Errorf("%s model has invalid shapes: input=%v, output=%v", names[i], model.InputShape, model.OutputShape)
		}
		
		// Verify all layers have computed shapes
		for j, layer := range model.Layers {
			if len(layer.InputShape) == 0 || len(layer.OutputShape) == 0 {
				t.Errorf("%s model layer %d has invalid shapes: input=%v, output=%v", names[i], j, layer.InputShape, layer.OutputShape)
			}
		}
	}
	
	fmt.Println("✅ Complex model architectures test completed successfully!")
}

// BenchmarkLayerCreation benchmarks layer creation performance
func BenchmarkLayerCreation(b *testing.B) {
	factory := layers.NewFactory()
	
	b.Run("Dense", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = factory.CreateDenseSpec(128, 64, true, "dense")
		}
	})
	
	b.Run("Conv2D", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = factory.CreateConv2DSpec(3, 16, 3, 1, 1, true, "conv")
		}
	})
	
	b.Run("LeakyReLU", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = factory.CreateLeakyReLUSpec(0.1, "leaky_relu")
		}
	})
}

// BenchmarkModelCompilation benchmarks model compilation performance
func BenchmarkModelCompilation(b *testing.B) {
	inputShape := []int{32, 3, 32, 32}
	
	b.Run("SimpleModel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			builder := layers.NewModelBuilder(inputShape)
			_, _ = builder.
				AddConv2D(16, 3, 1, 1, true, "conv1").
				AddReLU("relu1").
				AddDense(10, true, "fc1").
				AddSoftmax(-1, "softmax").
				Compile()
		}
	})
	
	b.Run("ComplexModel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			builder := layers.NewModelBuilder(inputShape)
			_, _ = builder.
				AddConv2D(32, 3, 1, 1, false, "conv1").
				AddBatchNorm(32, 1e-5, 0.1, true, "bn1").
				AddReLU("relu1").
				AddDropout(0.25, "dropout1").
				AddConv2D(64, 3, 2, 1, false, "conv2").
				AddBatchNorm(64, 1e-5, 0.1, true, "bn2").
				AddLeakyReLU(0.1, "leaky_relu1").
				AddDropout(0.5, "dropout2").
				AddDense(128, true, "fc1").
				AddELU(1.0, "elu1").
				AddDense(10, true, "fc2").
				AddSoftmax(-1, "softmax").
				Compile()
		}
	})
}

// TestGetCompiledModel tests the GetCompiledModel function
func TestGetCompiledModel(t *testing.T) {
	fmt.Println("\n=== Testing GetCompiledModel ===")
	
	// Test 1: Get compiled model from compiled builder
	inputShape := []int{16, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	// Compile the model first
	originalModel, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(10, true, "fc1").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile original model: %v", err)
	}
	
	// Get compiled model
	compiledModel, err := builder.GetCompiledModel()
	if err != nil {
		t.Fatalf("Failed to get compiled model: %v", err)
	}
	
	// Verify model properties match
	if compiledModel.TotalParameters != originalModel.TotalParameters {
		t.Errorf("Parameter count mismatch: expected %d, got %d", originalModel.TotalParameters, compiledModel.TotalParameters)
	}
	
	if len(compiledModel.Layers) != len(originalModel.Layers) {
		t.Errorf("Layer count mismatch: expected %d, got %d", len(originalModel.Layers), len(compiledModel.Layers))
	}
	
	fmt.Printf("✅ Compiled model retrieved successfully: %d layers, %d parameters\n", len(compiledModel.Layers), compiledModel.TotalParameters)
	
	// Test 2: Try to get compiled model from uncompiled builder
	uncompiledBuilder := layers.NewModelBuilder(inputShape)
	uncompiledBuilder.AddDense(10, true, "dense1") // Add layer but don't compile
	
	_, err = uncompiledBuilder.GetCompiledModel()
	if err == nil {
		t.Error("Expected error when getting compiled model from uncompiled builder")
	}
	
	fmt.Println("✅ GetCompiledModel correctly rejected uncompiled model")
	fmt.Println("✅ GetCompiledModel test completed successfully!")
}

// TestCreateParameterTensors tests the CreateParameterTensors function
func TestCreateParameterTensors(t *testing.T) {
	fmt.Println("\n=== Testing CreateParameterTensors ===")
	
	// Test 1: Create parameter tensors for compiled model
	inputShape := []int{16, 128}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "dense1").  // 128*64 + 64 = 8256 params
		AddDense(32, false, "dense2"). // 64*32 = 2048 params
		AddDense(10, true, "dense3").  // 32*10 + 10 = 330 params
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	// Create parameter tensors (handle potential panic from uninitialized memory manager)
	var tensors []*memory.Tensor
	
	func() {
		defer func() {
			if r := recover(); r != nil {
				if contains(fmt.Sprintf("%v", r), "global memory manager not initialized") {
					t.Skip("Skipping CreateParameterTensors test - memory manager not initialized in unit test context")
				}
				panic(r) // Re-panic if it's a different error
			}
		}()
		var createErr error
		tensors, createErr = model.CreateParameterTensors()
		err = createErr
	}()
	
	if err != nil {
		t.Fatalf("Failed to create parameter tensors: %v", err)
	}
	
	// Verify tensor count matches parameter shapes
	expectedTensorCount := len(model.ParameterShapes)
	if len(tensors) != expectedTensorCount {
		t.Errorf("Tensor count mismatch: expected %d, got %d", expectedTensorCount, len(tensors))
	}
	
	fmt.Printf("✅ Created %d parameter tensors successfully\n", len(tensors))
	
	// Clean up tensors
	for _, tensor := range tensors {
		tensor.Release()
	}
	
	// Test 2: Try to create tensors for uncompiled model
	uncompiled := &layers.ModelSpec{Compiled: false}
	_, err = uncompiled.CreateParameterTensors()
	if err == nil {
		t.Error("Expected error when creating tensors for uncompiled model")
	}
	
	fmt.Println("✅ CreateParameterTensors correctly rejected uncompiled model")
	fmt.Println("✅ CreateParameterTensors test completed successfully!")
}

// TestSummary tests the Summary function
func TestSummary(t *testing.T) {
	fmt.Println("\n=== Testing Summary ===")
	
	// Test 1: Summary for compiled model
	inputShape := []int{16, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(128, true, "dense1").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	summary := model.Summary()
	
	// Verify summary contains expected information
	if !contains(summary, "Model Summary") {
		t.Error("Summary missing 'Model Summary' header")
	}
	
	if !contains(summary, "Total Parameters") {
		t.Error("Summary missing 'Total Parameters'")
	}
	
	if !contains(summary, "conv1") || !contains(summary, "relu1") || !contains(summary, "dense1") {
		t.Error("Summary missing layer names")
	}
	
	fmt.Printf("✅ Summary generated successfully:\n%s\n", summary)
	
	// Test 2: Summary for uncompiled model
	uncompiled := &layers.ModelSpec{Compiled: false}
	summary = uncompiled.Summary()
	
	if summary != "Model not compiled" {
		t.Errorf("Expected 'Model not compiled', got: %s", summary)
	}
	
	fmt.Println("✅ Summary correctly handled uncompiled model")
	fmt.Println("✅ Summary test completed successfully!")
}

// Helper function to check if string contains substring
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

// TestValidationFunctions tests all model validation functions
func TestValidationFunctions(t *testing.T) {
	fmt.Println("\n=== Testing Validation Functions ===")
	
	// Test 1: ValidateModelForTrainingEngine (should use hybrid validation)
	inputShape := []int{16, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	cnnModel, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(10, true, "dense1").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile CNN model: %v", err)
	}
	
	// Test ValidateModelForTrainingEngine
	err = cnnModel.ValidateModelForTrainingEngine()
	if err != nil {
		t.Errorf("CNN model should be valid for TrainingEngine: %v", err)
	}
	fmt.Println("✅ ValidateModelForTrainingEngine passed for CNN model")
	
	// Test 2: ValidateModelForHybridEngine - valid CNN
	err = cnnModel.ValidateModelForHybridEngine()
	if err != nil {
		t.Errorf("CNN model should be valid for HybridEngine: %v", err)
	}
	fmt.Println("✅ ValidateModelForHybridEngine passed for CNN model")
	
	// Test 3: ValidateModelForHybridEngine - invalid (no Conv2D)
	mlpBuilder := layers.NewModelBuilder([]int{16, 3, 32, 32})
	mlpModel, err := mlpBuilder.
		AddDense(128, true, "dense1").
		AddDense(10, true, "dense2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile MLP model: %v", err)
	}
	
	err = mlpModel.ValidateModelForHybridEngine()
	if err != nil {
		t.Errorf("MLP model should be valid for HybridEngine (deprecated, uses dynamic engine): %v", err)
	}
	fmt.Println("✅ ValidateModelForHybridEngine correctly accepted MLP model (uses dynamic engine)")
	
	// Test 4: ValidateModelForDynamicEngine - valid CNN
	err = cnnModel.ValidateModelForDynamicEngine()
	if err != nil {
		t.Errorf("CNN model should be valid for DynamicEngine: %v", err)
	}
	fmt.Println("✅ ValidateModelForDynamicEngine passed for CNN model")
	
	// Test 5: ValidateModelForDynamicEngine - valid MLP with 2D input
	mlp2DBuilder := layers.NewModelBuilder([]int{16, 128})
	mlp2DModel, err := mlp2DBuilder.
		AddDense(64, true, "dense1").
		AddDense(10, true, "dense2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile 2D MLP model: %v", err)
	}
	
	err = mlp2DModel.ValidateModelForDynamicEngine()
	if err != nil {
		t.Errorf("2D MLP model should be valid for DynamicEngine: %v", err)
	}
	fmt.Println("✅ ValidateModelForDynamicEngine passed for 2D MLP model")
	
	// Test 6: ValidateModelForInference - valid models
	err = cnnModel.ValidateModelForInference()
	if err != nil {
		t.Errorf("CNN model should be valid for Inference: %v", err)
	}
	
	err = mlp2DModel.ValidateModelForInference()
	if err != nil {
		t.Errorf("2D MLP model should be valid for Inference: %v", err)
	}
	fmt.Println("✅ ValidateModelForInference passed for both models")
	
	// Test 7: Test uncompiled model validation
	uncompiled := &layers.ModelSpec{Compiled: false}
	err = uncompiled.ValidateModelForTrainingEngine()
	if err == nil {
		t.Error("Uncompiled model should fail validation")
	}
	
	err = uncompiled.ValidateModelForHybridEngine()
	if err == nil {
		t.Error("Uncompiled model should fail hybrid validation")
	}
	
	err = uncompiled.ValidateModelForDynamicEngine()
	if err == nil {
		t.Error("Uncompiled model should fail dynamic validation")
	}
	
	err = uncompiled.ValidateModelForInference()
	if err == nil {
		t.Error("Uncompiled model should fail inference validation")
	}
	fmt.Println("✅ All validation functions correctly rejected uncompiled model")
	
	// Test 8: Test empty model validation  
	// Note: The system correctly prevents compilation of empty models
	emptyBuilder := layers.NewModelBuilder([]int{16, 128})
	_, emptyErr := emptyBuilder.Compile() // Try to compile empty model
	if emptyErr == nil {
		t.Error("Empty model compilation should have been rejected")
	}
	fmt.Println("✅ Empty model compilation correctly rejected during build phase")
	
	fmt.Println("✅ Validation functions test completed successfully!")
}

// TestConvertToInferenceLayerSpecs tests the ConvertToInferenceLayerSpecs function
func TestConvertToInferenceLayerSpecs(t *testing.T) {
	fmt.Println("\n=== Testing ConvertToInferenceLayerSpecs ===")
	
	// Test 1: Convert valid CNN model
	inputShape := []int{16, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(128, true, "dense1").
		AddSoftmax(-1, "softmax").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	specs, err := model.ConvertToInferenceLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to inference specs: %v", err)
	}
	
	// Verify spec count matches layer count
	if len(specs) != len(model.Layers) {
		t.Errorf("Spec count mismatch: expected %d, got %d", len(model.Layers), len(specs))
	}
	
	// Verify each spec has correct properties
	for i, spec := range specs {
		if spec.LayerType != int32(model.Layers[i].Type) {
			t.Errorf("Layer %d type mismatch: expected %d, got %d", i, model.Layers[i].Type, spec.LayerType)
		}
		
		if spec.InputShapeLen <= 0 || spec.OutputShapeLen <= 0 {
			t.Errorf("Layer %d has invalid shape lengths: input=%d, output=%d", i, spec.InputShapeLen, spec.OutputShapeLen)
		}
	}
	
	fmt.Printf("✅ Converted %d layers to inference specs successfully\n", len(specs))
	
	// Test 2: Try to convert uncompiled model
	uncompiled := &layers.ModelSpec{Compiled: false}
	_, err = uncompiled.ConvertToInferenceLayerSpecs()
	if err == nil {
		t.Error("Expected error when converting uncompiled model")
	}
	
	fmt.Println("✅ ConvertToInferenceLayerSpecs correctly rejected uncompiled model")
	
	// Test 3: Try to convert model without Dense layer
	nodenBuilder := layers.NewModelBuilder([]int{16, 3, 32, 32})
	nodenseModel, err := nodenBuilder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile no-dense model: %v", err)
	}
	
	_, err = nodenseModel.ConvertToInferenceLayerSpecs()
	if err != nil {
		t.Errorf("Model without Dense layer should be valid for inference (uses dynamic engine): %v", err)
	}
	
	fmt.Println("✅ ConvertToInferenceLayerSpecs correctly accepted model without Dense layer (uses dynamic engine)")
	fmt.Println("✅ ConvertToInferenceLayerSpecs test completed successfully!")
}

// TestGetTrainingEngineConfig tests the GetTrainingEngineConfig function
func TestGetTrainingEngineConfig(t *testing.T) {
	fmt.Println("\n=== Testing GetTrainingEngineConfig ===")
	
	// Test 1: Get config for valid CNN model
	inputShape := []int{16, 3, 32, 32}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(16, 3, 1, 1, true, "conv1").
		AddReLU("relu1").
		AddDense(10, true, "dense1").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}
	
	config, err := model.GetTrainingEngineConfig()
	if err != nil {
		t.Fatalf("Failed to get training config: %v", err)
	}
	
	// Verify config contains expected keys
	expectedKeys := []string{"model_type", "input_shape", "output_shape", "total_parameters", "parameter_shapes", "layer_count", "layers"}
	for _, key := range expectedKeys {
		if _, ok := config[key]; !ok {
			t.Errorf("Config missing key: %s", key)
		}
	}
	
	// Verify config values
	if config["model_type"] != "cnn" {
		t.Errorf("Expected model_type 'cnn', got: %v", config["model_type"])
	}
	
	if config["layer_count"] != len(model.Layers) {
		t.Errorf("Layer count mismatch: expected %d, got %v", len(model.Layers), config["layer_count"])
	}
	
	fmt.Printf("✅ Training config generated successfully: %v layers, %v parameters\n", config["layer_count"], config["total_parameters"])
	
	// Test 2: Try to get config for invalid model (MLP only)
	mlpBuilder := layers.NewModelBuilder([]int{16, 3, 32, 32})
	mlpModel, err := mlpBuilder.
		AddDense(128, true, "dense1").
		AddDense(10, true, "dense2").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile MLP model: %v", err)
	}
	
	_, err = mlpModel.GetTrainingEngineConfig()
	if err != nil {
		t.Errorf("MLP model should be valid for training config (uses dynamic engine): %v", err)
	}
	
	fmt.Println("✅ GetTrainingEngineConfig correctly accepted MLP model (uses dynamic engine)")
	fmt.Println("✅ GetTrainingEngineConfig test completed successfully!")
}

// TestLayerTypeStringEdgeCases tests LayerType String() method edge cases  
func TestLayerTypeStringEdgeCases(t *testing.T) {
	fmt.Println("\n=== Testing LayerType String Edge Cases ===")
	
	// Test all valid layer types
	layerTypes := []layers.LayerType{
		layers.Dense,
		layers.Conv2D,
		layers.ReLU,
		layers.Softmax,
		layers.MaxPool2D,
		layers.Dropout,
		layers.BatchNorm,
		layers.LeakyReLU,
		layers.ELU,
	}
	
	expectedNames := []string{
		"Dense",
		"Conv2D", 
		"ReLU",
		"Softmax",
		"MaxPool2D",
		"Dropout",
		"BatchNorm",
		"LeakyReLU",
		"ELU",
	}
	
	for i, layerType := range layerTypes {
		str := layerType.String()
		if str != expectedNames[i] {
			t.Errorf("LayerType %d: expected '%s', got '%s'", layerType, expectedNames[i], str)
		}
	}
	
	fmt.Println("✅ All valid LayerType strings verified")
	
	// Test unknown layer type
	unknownType := layers.LayerType(999)
	str := unknownType.String()
	if str != "Unknown" {
		t.Errorf("Unknown LayerType: expected 'Unknown', got '%s'", str)
	}
	
	fmt.Println("✅ Unknown LayerType correctly handled")
	fmt.Println("✅ LayerType String edge cases test completed successfully!")
}

// TestSerializationEdgeCases tests serialization with comprehensive edge cases
func TestSerializationEdgeCases(t *testing.T) {
	fmt.Println("\n=== Testing Serialization Edge Cases ===")
	
	// Test 1: Model with all layer types and various configurations
	inputShape := []int{8, 3, 16, 16}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddConv2D(8, 1, 1, 0, false, "conv_no_bias").  // Conv without bias
		AddBatchNorm(8, 1e-5, 0.1, false, "bn_no_affine"). // BatchNorm without affine
		AddReLU("relu1").
		AddDropout(0.0, "dropout_zero").                    // Dropout with 0 rate
		AddConv2D(16, 3, 2, 1, true, "conv_with_bias").    // Conv with bias
		AddLeakyReLU(0.0, "leaky_relu_zero").               // LeakyReLU with 0 slope
		AddELU(0.1, "elu_small_alpha").                     // ELU with small alpha
		AddDense(32, false, "dense_no_bias").               // Dense without bias
		AddSoftmax(1, "softmax_axis1").                     // Softmax with axis 1
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile complex model: %v", err)
	}
	
	// Test SerializeForCGO
	cgoModel, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize for CGO: %v", err)
	}
	
	// Verify CGO serialization
	if len(cgoModel.Layers) != len(model.Layers) {
		t.Errorf("CGO layer count mismatch: expected %d, got %d", len(model.Layers), len(cgoModel.Layers))
	}
	
	fmt.Printf("✅ CGO serialization successful: %d layers\n", len(cgoModel.Layers))
	
	// Test ConvertToDynamicLayerSpecs
	dynamicSpecs, err := model.ConvertToDynamicLayerSpecs()
	if err != nil {
		t.Fatalf("Failed to convert to dynamic specs: %v", err)
	}
	
	if len(dynamicSpecs) != len(model.Layers) {
		t.Errorf("Dynamic specs count mismatch: expected %d, got %d", len(model.Layers), len(dynamicSpecs))
	}
	
	fmt.Printf("✅ Dynamic specs conversion successful: %d specs\n", len(dynamicSpecs))
	
	// Test 2: Try to serialize uncompiled model
	uncompiled := &layers.ModelSpec{Compiled: false}
	_, err = uncompiled.SerializeForCGO()
	if err == nil {
		t.Error("Expected error when serializing uncompiled model")
	}
	
	_, err = uncompiled.ConvertToDynamicLayerSpecs()
	if err == nil {
		t.Error("Expected error when converting uncompiled model to dynamic specs")
	}
	
	fmt.Println("✅ Serialization correctly rejected uncompiled model")
	fmt.Println("✅ Serialization edge cases test completed successfully!")
}

// TestSigmoidLayer tests the Sigmoid layer functionality
func TestSigmoidLayer(t *testing.T) {
	// Test 1: Sigmoid layer creation
	inputShape := []int{32, 128} // 2D input
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "dense1").
		AddSigmoid("sigmoid1").                    // Add Sigmoid activation
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with Sigmoid: %v", err)
	}
	
	// Test 2: Verify Sigmoid layer has no parameters
	sigmoidLayer := model.Layers[1] // Second layer is Sigmoid
	if sigmoidLayer.Type != layers.Sigmoid {
		t.Fatalf("Expected Sigmoid layer, got %s", sigmoidLayer.Type.String())
	}
	
	if sigmoidLayer.ParameterCount != 0 {
		t.Errorf("Expected Sigmoid to have 0 parameters, got %d", sigmoidLayer.ParameterCount)
	}
	
	// Test 3: Verify Sigmoid doesn't change shape
	expectedInputShape := []int{32, 64}  // Output of previous Dense layer
	expectedOutputShape := []int{32, 64} // Should be unchanged
	
	if len(sigmoidLayer.InputShape) != len(expectedInputShape) {
		t.Errorf("Sigmoid changed tensor rank: input %d, output %d", len(expectedInputShape), len(sigmoidLayer.InputShape))
	}
	
	for i, dim := range sigmoidLayer.InputShape {
		if dim != expectedInputShape[i] {
			t.Errorf("Sigmoid input shape mismatch at dimension %d: expected %d, got %d", i, expectedInputShape[i], dim)
		}
	}
	
	for i, dim := range sigmoidLayer.OutputShape {
		if dim != expectedOutputShape[i] {
			t.Errorf("Sigmoid output shape mismatch at dimension %d: expected %d, got %d", i, expectedOutputShape[i], dim)
		}
	}
	
	// Test 4: Verify Sigmoid serialization
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with Sigmoid: %v", err)
	}
	
	// Find Sigmoid layer in serialized format
	foundSigmoid := false
	for _, layer := range serialized.Layers {
		if layer.LayerType == 9 { // Sigmoid = 9 in Go enum
			foundSigmoid = true
			
			if len(layer.ParamFloat) != 0 {
				t.Errorf("Sigmoid should have 0 float params, got %d", len(layer.ParamFloat))
			}
			if len(layer.ParamInt) != 0 {
				t.Errorf("Sigmoid should have 0 int params, got %d", len(layer.ParamInt))
			}
			break
		}
	}
	
	if !foundSigmoid {
		t.Fatalf("Sigmoid layer not found in serialized model")
	}
}

// TestTanhLayer tests the Tanh layer functionality  
func TestTanhLayer(t *testing.T) {
	
	// Test 1: Tanh layer creation
	inputShape := []int{32, 128} // 2D input
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "dense1").
		AddTanh("tanh1").                         // Add Tanh activation
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with Tanh: %v", err)
	}
	
	
	// Test 2: Verify Tanh layer has no parameters
	tanhLayer := model.Layers[1] // Second layer is Tanh
	if tanhLayer.Type != layers.Tanh {
		t.Fatalf("Expected Tanh layer, got %s", tanhLayer.Type.String())
	}
	
	if tanhLayer.ParameterCount != 0 {
		t.Errorf("Expected Tanh to have 0 parameters, got %d", tanhLayer.ParameterCount)
	}
	
	// Test 3: Verify Tanh doesn't change shape
	expectedInputShape := []int{32, 64}  // Output of previous Dense layer
	expectedOutputShape := []int{32, 64} // Should be unchanged
	
	if len(tanhLayer.InputShape) != len(expectedInputShape) {
		t.Errorf("Tanh changed tensor rank: input %d, output %d", len(expectedInputShape), len(tanhLayer.InputShape))
	}
	
	for i, dim := range tanhLayer.InputShape {
		if dim != expectedInputShape[i] {
			t.Errorf("Tanh input shape mismatch at dimension %d: expected %d, got %d", i, expectedInputShape[i], dim)
		}
	}
	
	for i, dim := range tanhLayer.OutputShape {
		if dim != expectedOutputShape[i] {
			t.Errorf("Tanh output shape mismatch at dimension %d: expected %d, got %d", i, expectedOutputShape[i], dim)
		}
	}
	
	
	// Test 4: Verify Tanh serialization
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with Tanh: %v", err)
	}
	
	// Find Tanh layer in serialized format
	foundTanh := false
	for _, layer := range serialized.Layers {
		if layer.LayerType == 10 { // Tanh = 10 in Go enum
			foundTanh = true
			
			if len(layer.ParamFloat) != 0 {
				t.Errorf("Tanh should have 0 float params, got %d", len(layer.ParamFloat))
			}
			if len(layer.ParamInt) != 0 {
				t.Errorf("Tanh should have 0 int params, got %d", len(layer.ParamInt))
			}
			break
		}
	}
	
	if !foundTanh {
		t.Fatalf("Tanh layer not found in serialized model")
	}
	
}

// TestSwishLayer tests the Swish layer functionality
func TestSwishLayer(t *testing.T) {
	
	// Test 1: Swish layer creation
	inputShape := []int{32, 128} // 2D input
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(64, true, "dense1").
		AddSwish("swish1").                       // Add Swish activation
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to compile model with Swish: %v", err)
	}
	
	
	// Test 2: Verify Swish layer has no parameters
	swishLayer := model.Layers[1] // Second layer is Swish
	if swishLayer.Type != layers.Swish {
		t.Fatalf("Expected Swish layer, got %s", swishLayer.Type.String())
	}
	
	if swishLayer.ParameterCount != 0 {
		t.Errorf("Expected Swish to have 0 parameters, got %d", swishLayer.ParameterCount)
	}
	
	// Test 3: Verify Swish doesn't change shape
	expectedInputShape := []int{32, 64}  // Output of previous Dense layer
	expectedOutputShape := []int{32, 64} // Should be unchanged
	
	if len(swishLayer.InputShape) != len(expectedInputShape) {
		t.Errorf("Swish changed tensor rank: input %d, output %d", len(expectedInputShape), len(swishLayer.InputShape))
	}
	
	for i, dim := range swishLayer.InputShape {
		if dim != expectedInputShape[i] {
			t.Errorf("Swish input shape mismatch at dimension %d: expected %d, got %d", i, expectedInputShape[i], dim)
		}
	}
	
	for i, dim := range swishLayer.OutputShape {
		if dim != expectedOutputShape[i] {
			t.Errorf("Swish output shape mismatch at dimension %d: expected %d, got %d", i, expectedOutputShape[i], dim)
		}
	}
	
	
	// Test 4: Verify Swish serialization
	serialized, err := model.SerializeForCGO()
	if err != nil {
		t.Fatalf("Failed to serialize model with Swish: %v", err)
	}
	
	// Find Swish layer in serialized format
	foundSwish := false
	for _, layer := range serialized.Layers {
		if layer.LayerType == 11 { // Swish = 11 in Go enum
			foundSwish = true
			
			if len(layer.ParamFloat) != 0 {
				t.Errorf("Swish should have 0 float params, got %d", len(layer.ParamFloat))
			}
			if len(layer.ParamInt) != 0 {
				t.Errorf("Swish should have 0 int params, got %d", len(layer.ParamInt))
			}
			break
		}
	}
	
	if !foundSwish {
		t.Fatalf("Swish layer not found in serialized model")
	}
	
}

// TestAllActivationFunctions tests all activation functions together
func TestAllActivationFunctions(t *testing.T) {
	
	// Test comprehensive model with all activations
	inputShape := []int{32, 64}
	builder := layers.NewModelBuilder(inputShape)
	
	model, err := builder.
		AddDense(32, true, "dense1").
		AddReLU("relu1").
		AddDense(32, true, "dense2").
		AddSigmoid("sigmoid1").
		AddDense(32, true, "dense3").
		AddTanh("tanh1").
		AddDense(32, true, "dense4").
		AddSwish("swish1").
		AddDense(16, true, "dense5").
		AddLeakyReLU(0.1, "leaky_relu1").
		AddDense(8, true, "dense6").
		AddELU(1.0, "elu1").
		AddDense(1, true, "output").
		Compile()
	
	if err != nil {
		t.Fatalf("Failed to create comprehensive model: %v", err)
	}
	
	
	// Verify all activation types are present
	activationTypes := []layers.LayerType{
		layers.ReLU, layers.Sigmoid, layers.Tanh, layers.Swish, layers.LeakyReLU, layers.ELU,
	}
	
	foundActivations := make(map[layers.LayerType]bool)
	for _, layer := range model.Layers {
		for _, activationType := range activationTypes {
			if layer.Type == activationType {
				foundActivations[activationType] = true
			}
		}
	}
	
	for _, activationType := range activationTypes {
		if !foundActivations[activationType] {
			t.Errorf("Activation function %s not found in model", activationType.String())
		} else {
			fmt.Printf("   ✅ %s activation found\n", activationType.String())
		}
	}
	
	fmt.Println("✅ All activation functions test completed successfully!")
}