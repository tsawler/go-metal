package layers_test

import (
	"fmt"
	"testing"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/training"
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
	fmt.Println("\n=== Testing Trainer Factory Integration ===")
	
	factory := training.NewModelFactory()
	
	// Test CNN trainer creation
	inputShape := []int{16, 3, 32, 32}
	config := training.TrainerConfig{
		BatchSize:       16,
		LearningRate:    0.001,
		OptimizerType:   cgo_bridge.Adam,
		UseHybridEngine: true,
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		WeightDecay:     0.0001,
	}
	
	cnnTrainer, err := factory.CreateCNNTrainer(inputShape, 10, config)
	if err != nil {
		t.Fatalf("Failed to create CNN trainer: %v", err)
	}
	defer cnnTrainer.Cleanup()
	
	fmt.Println("✅ CNN trainer created successfully")
	fmt.Println(cnnTrainer.GetModelSummary())
	
	// Test MLP trainer creation
	mlpTrainer, err := factory.CreateMLPTrainer(784, []int{256, 128}, 10, config)
	if err != nil {
		t.Fatalf("Failed to create MLP trainer: %v", err)
	}
	defer mlpTrainer.Cleanup()
	
	fmt.Println("✅ MLP trainer created successfully")
	fmt.Println(mlpTrainer.GetModelSummary())
	
	fmt.Println("✅ Trainer factory integration test completed successfully!")
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
	
	fmt.Printf("✅ Model with Dropout compiled successfully\n")
	fmt.Printf("Total layers: %d\n", len(model.Layers))
	
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
	
	fmt.Printf("✅ Dropout parameters verified: rate=%.1f, training=%v\n", rate, training)
	
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
			if len(layer.ParamInt) == 0 || layer.ParamInt[0] != 1 {
				t.Errorf("Dropout training mode not serialized correctly: %v", layer.ParamInt)
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
			if spec.ParamIntCount != 1 || spec.ParamInt[0] != 1 {
				t.Errorf("Dynamic Dropout training mode not correct")
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
			if layer.ParamInt[3] != 1 { // training = true
				t.Errorf("BatchNorm training not serialized correctly: %d", layer.ParamInt[3])
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

// BenchmarkCompliantVsOriginal compares performance of compliant vs original architecture
func BenchmarkCompliantVsOriginal(b *testing.B) {
	// This benchmark would compare the compliant layer system performance
	// with the original SimpleTrainer to ensure no performance regression
	
	b.Skip("Performance benchmark - implement when ready for validation")
}