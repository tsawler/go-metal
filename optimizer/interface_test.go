package optimizer

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/checkpoints"
)

// MockOptimizer implements the Optimizer interface for testing
type MockOptimizer struct {
	stepCount     uint64
	learningRate  float32
	weightBuffers []unsafe.Pointer
	commandPool   unsafe.Pointer
	cleanupCalled bool
}

func (m *MockOptimizer) Step(gradientBuffers []unsafe.Pointer) error {
	if len(gradientBuffers) != len(m.weightBuffers) {
		return fmt.Errorf("gradient buffer count mismatch")
	}
	m.stepCount++
	return nil
}

func (m *MockOptimizer) GetState() (*OptimizerState, error) {
	return &OptimizerState{
		Type: "Mock",
		Parameters: map[string]interface{}{
			"learning_rate": float64(m.learningRate),
			"step_count":    float64(m.stepCount),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}, nil
}

func (m *MockOptimizer) LoadState(state *OptimizerState) error {
	if state.Type != "Mock" {
		return fmt.Errorf("state type mismatch: expected Mock, got %s", state.Type)
	}
	m.learningRate = extractFloat32Param(state.Parameters, "learning_rate", 0.001)
	m.stepCount = extractUint64Param(state.Parameters, "step_count", 0)
	return nil
}

func (m *MockOptimizer) GetStepCount() uint64 {
	return m.stepCount
}

func (m *MockOptimizer) UpdateLearningRate(lr float32) {
	m.learningRate = lr
}

func (m *MockOptimizer) SetWeightBuffers(weightBuffers []unsafe.Pointer) error {
	if len(weightBuffers) == 0 {
		return fmt.Errorf("no weight buffers provided")
	}
	m.weightBuffers = weightBuffers
	return nil
}

func (m *MockOptimizer) SetCommandPool(commandPool unsafe.Pointer) {
	m.commandPool = commandPool
}

func (m *MockOptimizer) Cleanup() {
	m.cleanupCalled = true
}

// TestOptimizerInterface tests the Optimizer interface using a mock implementation
func TestOptimizerInterface(t *testing.T) {
	mock := &MockOptimizer{
		stepCount:    0,
		learningRate: 0.001,
	}

	// Test that mock implements the interface
	var optimizer Optimizer = mock

	// Test SetWeightBuffers
	weightBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x1000)),
		unsafe.Pointer(uintptr(0x2000)),
	}
	err := optimizer.SetWeightBuffers(weightBuffers)
	if err != nil {
		t.Errorf("SetWeightBuffers() error = %v", err)
	}

	// Test SetWeightBuffers with empty buffers
	err = optimizer.SetWeightBuffers([]unsafe.Pointer{})
	if err == nil {
		t.Error("SetWeightBuffers() with empty buffers should return error")
	}

	// Reset weight buffers
	optimizer.SetWeightBuffers(weightBuffers)

	// Test Step
	gradientBuffers := []unsafe.Pointer{
		unsafe.Pointer(uintptr(0x3000)),
		unsafe.Pointer(uintptr(0x4000)),
	}
	err = optimizer.Step(gradientBuffers)
	if err != nil {
		t.Errorf("Step() error = %v", err)
	}

	// Test GetStepCount
	if optimizer.GetStepCount() != 1 {
		t.Errorf("GetStepCount() = %d, want 1", optimizer.GetStepCount())
	}

	// Test Step with mismatched gradient buffers
	err = optimizer.Step([]unsafe.Pointer{unsafe.Pointer(uintptr(0x3000))})
	if err == nil {
		t.Error("Step() with mismatched gradient buffers should return error")
	}

	// Test UpdateLearningRate
	newLR := float32(0.005)
	optimizer.UpdateLearningRate(newLR)
	if mock.learningRate != newLR {
		t.Errorf("UpdateLearningRate() learning rate = %f, want %f", mock.learningRate, newLR)
	}

	// Test SetCommandPool
	commandPool := unsafe.Pointer(uintptr(0x5000))
	optimizer.SetCommandPool(commandPool)
	if mock.commandPool != commandPool {
		t.Errorf("SetCommandPool() command pool = %p, want %p", mock.commandPool, commandPool)
	}

	// Test GetState
	state, err := optimizer.GetState()
	if err != nil {
		t.Errorf("GetState() error = %v", err)
	}
	if state.Type != "Mock" {
		t.Errorf("GetState() type = %s, want Mock", state.Type)
	}

	// Test LoadState
	newState := &OptimizerState{
		Type: "Mock",
		Parameters: map[string]interface{}{
			"learning_rate": float64(0.002),
			"step_count":    float64(5),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	err = optimizer.LoadState(newState)
	if err != nil {
		t.Errorf("LoadState() error = %v", err)
	}
	if mock.learningRate != 0.002 {
		t.Errorf("LoadState() learning rate = %f, want 0.002", mock.learningRate)
	}
	if mock.stepCount != 5 {
		t.Errorf("LoadState() step count = %d, want 5", mock.stepCount)
	}

	// Test LoadState with wrong type
	wrongState := &OptimizerState{
		Type: "WrongType",
		Parameters: map[string]interface{}{
			"learning_rate": float64(0.002),
		},
		StateData: []checkpoints.OptimizerTensor{},
	}
	err = optimizer.LoadState(wrongState)
	if err == nil {
		t.Error("LoadState() with wrong type should return error")
	}

	// Test Cleanup
	optimizer.Cleanup()
	if !mock.cleanupCalled {
		t.Error("Cleanup() was not called")
	}
}

// TestOptimizerInterfaceCompliance tests that all optimizer implementations comply with the interface
func TestOptimizerInterfaceCompliance(t *testing.T) {
	// Test that all optimizer state types implement the interface
	var optimizers []Optimizer

	// We can't create real optimizers without Metal device, but we can test the interface compliance
	// by checking that the types can be assigned to the interface variable

	// This would be tested in integration tests with real Metal device:
	// optimizers = append(optimizers, &AdamOptimizerState{})
	// optimizers = append(optimizers, &SGDOptimizerState{})
	// optimizers = append(optimizers, &RMSPropOptimizerState{})
	// optimizers = append(optimizers, &AdaGradOptimizerState{})
	// optimizers = append(optimizers, &AdaDeltaOptimizerState{})
	// optimizers = append(optimizers, &NadamOptimizerState{})
	// optimizers = append(optimizers, &LBFGSOptimizerState{})

	// For now, test with mock
	optimizers = append(optimizers, &MockOptimizer{})

	for i, optimizer := range optimizers {
		t.Run(fmt.Sprintf("Optimizer_%d", i), func(t *testing.T) {
			// Test that each optimizer implements all required methods
			if optimizer == nil {
				t.Error("Optimizer is nil")
			}

			// Test method signatures exist (compile-time check)
			_ = optimizer.Step
			_ = optimizer.GetState
			_ = optimizer.LoadState
			_ = optimizer.GetStepCount
			_ = optimizer.UpdateLearningRate
			_ = optimizer.SetWeightBuffers
			_ = optimizer.SetCommandPool
			_ = optimizer.Cleanup
		})
	}
}

// TestOptimizerStateStructure tests the OptimizerState struct
func TestOptimizerStateStructure(t *testing.T) {
	state := &OptimizerState{
		Type: "TestOptimizer",
		Parameters: map[string]interface{}{
			"learning_rate": float64(0.001),
			"beta1":         float64(0.9),
			"beta2":         float64(0.999),
			"epsilon":       float64(1e-8),
			"weight_decay":  float64(0.0),
			"step_count":    float64(100),
			"momentum":      float64(0.9),
			"nesterov":      true,
			"centered":      false,
		},
		StateData: []checkpoints.OptimizerTensor{
			{
				Name:      "momentum_0",
				Shape:     []int{10, 20},
				Data:      make([]float32, 200),
				StateType: "momentum",
			},
			{
				Name:      "variance_0",
				Shape:     []int{10, 20},
				Data:      make([]float32, 200),
				StateType: "variance",
			},
		},
	}

	// Test basic structure
	if state.Type != "TestOptimizer" {
		t.Errorf("Type = %s, want TestOptimizer", state.Type)
	}

	if len(state.Parameters) != 9 {
		t.Errorf("Parameters length = %d, want 9", len(state.Parameters))
	}

	if len(state.StateData) != 2 {
		t.Errorf("StateData length = %d, want 2", len(state.StateData))
	}

	// Test parameter extraction
	lr := extractFloat32Param(state.Parameters, "learning_rate", 0.0)
	if lr != 0.001 {
		t.Errorf("learning_rate = %f, want 0.001", lr)
	}

	beta1 := extractFloat32Param(state.Parameters, "beta1", 0.0)
	if beta1 != 0.9 {
		t.Errorf("beta1 = %f, want 0.9", beta1)
	}

	stepCount := extractUint64Param(state.Parameters, "step_count", 0)
	if stepCount != 100 {
		t.Errorf("step_count = %d, want 100", stepCount)
	}

	nesterov := extractBoolParam(state.Parameters, "nesterov", false)
	if !nesterov {
		t.Errorf("nesterov = %t, want true", nesterov)
	}

	centered := extractBoolParam(state.Parameters, "centered", true)
	if centered {
		t.Errorf("centered = %t, want false", centered)
	}

	// Test state data
	for i, tensor := range state.StateData {
		if len(tensor.Shape) != 2 {
			t.Errorf("Tensor %d shape dimensions = %d, want 2", i, len(tensor.Shape))
		}
		if tensor.Shape[0] != 10 || tensor.Shape[1] != 20 {
			t.Errorf("Tensor %d shape = %v, want [10, 20]", i, tensor.Shape)
		}
		if len(tensor.Data) != 200 {
			t.Errorf("Tensor %d data length = %d, want 200", i, len(tensor.Data))
		}
	}

	// Test buffer index extraction
	idx0 := extractBufferIndex(state.StateData[0].Name)
	if idx0 != 0 {
		t.Errorf("Buffer index for %s = %d, want 0", state.StateData[0].Name, idx0)
	}

	idx1 := extractBufferIndex(state.StateData[1].Name)
	if idx1 != 0 {
		t.Errorf("Buffer index for %s = %d, want 0", state.StateData[1].Name, idx1)
	}

	t.Log("OptimizerState structure test passed")
}

// TestOptimizerStateValidation tests state validation
func TestOptimizerStateValidation(t *testing.T) {
	tests := []struct {
		name          string
		optimizerType string
		state         *OptimizerState
		expectError   bool
	}{
		{
			name:          "valid_state",
			optimizerType: "Adam",
			state:         &OptimizerState{Type: "Adam"},
			expectError:   false,
		},
		{
			name:          "type_mismatch",
			optimizerType: "Adam",
			state:         &OptimizerState{Type: "SGD"},
			expectError:   true,
		},
		{
			name:          "empty_optimizer_type",
			optimizerType: "",
			state:         &OptimizerState{Type: "Adam"},
			expectError:   true,
		},
		{
			name:          "empty_state_type",
			optimizerType: "Adam",
			state:         &OptimizerState{Type: ""},
			expectError:   true,
		},
		{
			name:          "nil_state",
			optimizerType: "Adam",
			state:         nil,
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			if tt.state == nil {
				// Test panics for nil state
				defer func() {
					if r := recover(); r != nil {
						err = fmt.Errorf("panic: %v", r)
					}
				}()
				validateStateType(tt.optimizerType, tt.state)
			} else {
				err = validateStateType(tt.optimizerType, tt.state)
			}

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error but got: %v", err)
				}
			}
		})
	}
}

// TestOptimizerMethodSignatures tests that all required methods have correct signatures
func TestOptimizerMethodSignatures(t *testing.T) {
	// This is a compile-time test to ensure the interface is correctly defined
	var optimizer Optimizer = &MockOptimizer{}

	// Test Step method
	var stepFunc func([]unsafe.Pointer) error = optimizer.Step
	if stepFunc == nil {
		t.Error("Step method is nil")
	}

	// Test GetState method
	var getStateFunc func() (*OptimizerState, error) = optimizer.GetState
	if getStateFunc == nil {
		t.Error("GetState method is nil")
	}

	// Test LoadState method
	var loadStateFunc func(*OptimizerState) error = optimizer.LoadState
	if loadStateFunc == nil {
		t.Error("LoadState method is nil")
	}

	// Test GetStepCount method
	var getStepCountFunc func() uint64 = optimizer.GetStepCount
	if getStepCountFunc == nil {
		t.Error("GetStepCount method is nil")
	}

	// Test UpdateLearningRate method
	var updateLRFunc func(float32) = optimizer.UpdateLearningRate
	if updateLRFunc == nil {
		t.Error("UpdateLearningRate method is nil")
	}

	// Test SetWeightBuffers method
	var setWeightBuffersFunc func([]unsafe.Pointer) error = optimizer.SetWeightBuffers
	if setWeightBuffersFunc == nil {
		t.Error("SetWeightBuffers method is nil")
	}

	// Test SetCommandPool method
	var setCommandPoolFunc func(unsafe.Pointer) = optimizer.SetCommandPool
	if setCommandPoolFunc == nil {
		t.Error("SetCommandPool method is nil")
	}

	// Test Cleanup method
	var cleanupFunc func() = optimizer.Cleanup
	if cleanupFunc == nil {
		t.Error("Cleanup method is nil")
	}

	t.Log("All optimizer method signatures are correct")
}