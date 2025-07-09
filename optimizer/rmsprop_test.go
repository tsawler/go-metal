package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/memory"
)

// MockDevice provides a mock Metal device for testing
type MockDevice struct{}

func (m *MockDevice) Pointer() unsafe.Pointer {
	return unsafe.Pointer(uintptr(0x1000)) // Mock device pointer
}

func TestDefaultRMSPropConfig(t *testing.T) {
	config := DefaultRMSPropConfig()
	
	expectedLR := float32(0.01)
	expectedAlpha := float32(0.99)
	expectedEpsilon := float32(1e-8)
	expectedWeightDecay := float32(0.0)
	expectedMomentum := float32(0.0)
	expectedCentered := false
	
	if config.LearningRate != expectedLR {
		t.Errorf("Expected LearningRate %f, got %f", expectedLR, config.LearningRate)
	}
	
	if config.Alpha != expectedAlpha {
		t.Errorf("Expected Alpha %f, got %f", expectedAlpha, config.Alpha)
	}
	
	if config.Epsilon != expectedEpsilon {
		t.Errorf("Expected Epsilon %f, got %f", expectedEpsilon, config.Epsilon)
	}
	
	if config.WeightDecay != expectedWeightDecay {
		t.Errorf("Expected WeightDecay %f, got %f", expectedWeightDecay, config.WeightDecay)
	}
	
	if config.Momentum != expectedMomentum {
		t.Errorf("Expected Momentum %f, got %f", expectedMomentum, config.Momentum)
	}
	
	if config.Centered != expectedCentered {
		t.Errorf("Expected Centered %t, got %t", expectedCentered, config.Centered)
	}
}

func TestNewRMSPropOptimizer(t *testing.T) {
	t.Skip("Skipping RMSProp optimizer test - requires Metal CGO integration")
	
	// Test configurations
	config := DefaultRMSPropConfig()
	
	// Test weight shapes
	weightShapes := [][]int{
		{100, 10},   // Weight matrix
		{10},        // Bias vector
		{64, 100},   // Another weight matrix
		{64},        // Another bias vector
	}
	
	t.Run("NilMemoryManager", func(t *testing.T) {
		device := &MockDevice{}
		_, err := NewRMSPropOptimizer(
			config,
			weightShapes,
			nil,
			device.Pointer(),
		)
		
		if err == nil {
			t.Fatal("Expected error for nil memory manager")
		}
	})
	
	t.Run("NilDevice", func(t *testing.T) {
		device := &MockDevice{}
		memoryManager := memory.NewMemoryManager(device.Pointer())
		_, err := NewRMSPropOptimizer(
			config,
			weightShapes,
			memoryManager,
			nil,
		)
		
		if err == nil {
			t.Fatal("Expected error for nil device")
		}
	})
	
	t.Run("EmptyWeightShapes", func(t *testing.T) {
		device := &MockDevice{}
		memoryManager := memory.NewMemoryManager(device.Pointer())
		_, err := NewRMSPropOptimizer(
			config,
			[][]int{},
			memoryManager,
			device.Pointer(),
		)
		
		if err == nil {
			t.Fatal("Expected error for empty weight shapes")
		}
	})
}

func TestRMSPropOptimizerMethods(t *testing.T) {
	// Test configurations
	config := DefaultRMSPropConfig()
	
	// Test weight shapes
	weightShapes := [][]int{
		{100, 10},   // Weight matrix
		{10},        // Bias vector
	}
	
	// Create a mock optimizer for testing methods
	optimizer := &RMSPropOptimizerState{
		LearningRate:          config.LearningRate,
		Alpha:                 config.Alpha,
		Epsilon:               config.Epsilon,
		WeightDecay:           config.WeightDecay,
		Momentum:              config.Momentum,
		Centered:              config.Centered,
		SquaredGradAvgBuffers: make([]unsafe.Pointer, len(weightShapes)),
		MomentumBuffers:       make([]unsafe.Pointer, len(weightShapes)),
		GradientAvgBuffers:    make([]unsafe.Pointer, len(weightShapes)),
		WeightBuffers:         make([]unsafe.Pointer, len(weightShapes)),
		bufferSizes:           make([]int, len(weightShapes)),
		StepCount:             0,
	}
	
	t.Run("UpdateLearningRate", func(t *testing.T) {
		newLR := float32(0.001)
		optimizer.UpdateLearningRate(newLR)
		
		if optimizer.LearningRate != newLR {
			t.Errorf("Expected LearningRate %f, got %f", newLR, optimizer.LearningRate)
		}
	})
	
	t.Run("GetStep", func(t *testing.T) {
		step := optimizer.GetStep()
		if step != 0 {
			t.Errorf("Expected step 0, got %d", step)
		}
	})
	
	t.Run("GetStats", func(t *testing.T) {
		stats := optimizer.GetStats()
		
		if stats.StepCount != 0 {
			t.Errorf("Expected StepCount 0, got %d", stats.StepCount)
		}
		
		if stats.LearningRate != optimizer.LearningRate {
			t.Errorf("Expected LearningRate %f, got %f", optimizer.LearningRate, stats.LearningRate)
		}
		
		if stats.Alpha != config.Alpha {
			t.Errorf("Expected Alpha %f, got %f", config.Alpha, stats.Alpha)
		}
		
		if stats.NumParameters != len(weightShapes) {
			t.Errorf("Expected NumParameters %d, got %d", len(weightShapes), stats.NumParameters)
		}
	})
	
	t.Run("SetCommandPool", func(t *testing.T) {
		mockPool := unsafe.Pointer(uintptr(0x2000))
		optimizer.SetCommandPool(mockPool)
		
		if optimizer.commandPool != mockPool {
			t.Errorf("Expected commandPool %p, got %p", mockPool, optimizer.commandPool)
		}
		
		if !optimizer.usePooling {
			t.Error("Expected usePooling to be true")
		}
		
		// Test setting nil pool
		optimizer.SetCommandPool(nil)
		if optimizer.usePooling {
			t.Error("Expected usePooling to be false after setting nil pool")
		}
	})
}

func TestRMSPropBufferCalculations(t *testing.T) {
	// Test buffer size calculations
	testCases := []struct {
		name     string
		shape    []int
		expected int
	}{
		{"Vector", []int{10}, 10},
		{"Matrix", []int{10, 20}, 200},
		{"3D Tensor", []int{10, 20, 30}, 6000},
		{"4D Tensor", []int{2, 3, 4, 5}, 120},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			size := calculateTensorSize(tc.shape)
			if size != tc.expected {
				t.Errorf("Expected size %d, got %d", tc.expected, size)
			}
		})
	}
}

func TestRMSPropTotalBufferSize(t *testing.T) {
	// Test weight shapes
	weightShapes := [][]int{
		{100, 10},   // 1000 elements = 4000 bytes
		{10},        // 10 elements = 40 bytes
	}
	
	bufferSizes := []int{4000, 40}
	
	t.Run("StandardRMSProp", func(t *testing.T) {
		config := DefaultRMSPropConfig()
		optimizer := &RMSPropOptimizerState{
			LearningRate:          config.LearningRate,
			Alpha:                 config.Alpha,
			Epsilon:               config.Epsilon,
			WeightDecay:           config.WeightDecay,
			Momentum:              config.Momentum,
			Centered:              config.Centered,
			SquaredGradAvgBuffers: make([]unsafe.Pointer, len(weightShapes)),
			MomentumBuffers:       make([]unsafe.Pointer, len(weightShapes)),
			GradientAvgBuffers:    make([]unsafe.Pointer, len(weightShapes)),
			WeightBuffers:         make([]unsafe.Pointer, len(weightShapes)),
			bufferSizes:           bufferSizes,
			StepCount:             0,
		}
		
		// Standard RMSProp only needs squared gradient average buffers
		expectedSize := 4000 + 40 // One buffer per weight tensor
		actualSize := optimizer.getTotalBufferSize()
		
		if actualSize != expectedSize {
			t.Errorf("Expected total buffer size %d, got %d", expectedSize, actualSize)
		}
	})
	
	t.Run("RMSPropWithMomentum", func(t *testing.T) {
		config := DefaultRMSPropConfig()
		config.Momentum = 0.9
		
		optimizer := &RMSPropOptimizerState{
			LearningRate:          config.LearningRate,
			Alpha:                 config.Alpha,
			Epsilon:               config.Epsilon,
			WeightDecay:           config.WeightDecay,
			Momentum:              config.Momentum,
			Centered:              config.Centered,
			SquaredGradAvgBuffers: make([]unsafe.Pointer, len(weightShapes)),
			MomentumBuffers:       make([]unsafe.Pointer, len(weightShapes)),
			GradientAvgBuffers:    make([]unsafe.Pointer, len(weightShapes)),
			WeightBuffers:         make([]unsafe.Pointer, len(weightShapes)),
			bufferSizes:           bufferSizes,
			StepCount:             0,
		}
		
		// RMSProp with momentum needs squared gradient average + momentum buffers
		expectedSize := (4000 + 40) * 2 // Two buffers per weight tensor
		actualSize := optimizer.getTotalBufferSize()
		
		if actualSize != expectedSize {
			t.Errorf("Expected total buffer size %d, got %d", expectedSize, actualSize)
		}
	})
	
	t.Run("CenteredRMSProp", func(t *testing.T) {
		config := DefaultRMSPropConfig()
		config.Centered = true
		
		optimizer := &RMSPropOptimizerState{
			LearningRate:          config.LearningRate,
			Alpha:                 config.Alpha,
			Epsilon:               config.Epsilon,
			WeightDecay:           config.WeightDecay,
			Momentum:              config.Momentum,
			Centered:              config.Centered,
			SquaredGradAvgBuffers: make([]unsafe.Pointer, len(weightShapes)),
			MomentumBuffers:       make([]unsafe.Pointer, len(weightShapes)),
			GradientAvgBuffers:    make([]unsafe.Pointer, len(weightShapes)),
			WeightBuffers:         make([]unsafe.Pointer, len(weightShapes)),
			bufferSizes:           bufferSizes,
			StepCount:             0,
		}
		
		// Centered RMSProp needs squared gradient average + gradient average buffers
		expectedSize := (4000 + 40) * 2 // Two buffers per weight tensor
		actualSize := optimizer.getTotalBufferSize()
		
		if actualSize != expectedSize {
			t.Errorf("Expected total buffer size %d, got %d", expectedSize, actualSize)
		}
	})
	
	t.Run("CenteredRMSPropWithMomentum", func(t *testing.T) {
		config := DefaultRMSPropConfig()
		config.Centered = true
		config.Momentum = 0.9
		
		optimizer := &RMSPropOptimizerState{
			LearningRate:          config.LearningRate,
			Alpha:                 config.Alpha,
			Epsilon:               config.Epsilon,
			WeightDecay:           config.WeightDecay,
			Momentum:              config.Momentum,
			Centered:              config.Centered,
			SquaredGradAvgBuffers: make([]unsafe.Pointer, len(weightShapes)),
			MomentumBuffers:       make([]unsafe.Pointer, len(weightShapes)),
			GradientAvgBuffers:    make([]unsafe.Pointer, len(weightShapes)),
			WeightBuffers:         make([]unsafe.Pointer, len(weightShapes)),
			bufferSizes:           bufferSizes,
			StepCount:             0,
		}
		
		// Centered RMSProp with momentum needs all three buffer types
		expectedSize := (4000 + 40) * 3 // Three buffers per weight tensor
		actualSize := optimizer.getTotalBufferSize()
		
		if actualSize != expectedSize {
			t.Errorf("Expected total buffer size %d, got %d", expectedSize, actualSize)
		}
	})
}