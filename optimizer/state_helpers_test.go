package optimizer

import (
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/checkpoints"
)

// TestExtractFloat32Param tests the extractFloat32Param helper function
func TestExtractFloat32Param(t *testing.T) {
	tests := []struct {
		name         string
		params       map[string]interface{}
		key          string
		defaultValue float32
		expected     float32
	}{
		{
			name:         "existing_float64_param",
			params:       map[string]interface{}{"learning_rate": float64(0.01)},
			key:          "learning_rate",
			defaultValue: 0.001,
			expected:     0.01,
		},
		{
			name:         "missing_param",
			params:       map[string]interface{}{"beta1": float64(0.9)},
			key:          "learning_rate",
			defaultValue: 0.001,
			expected:     0.001,
		},
		{
			name:         "wrong_type_param",
			params:       map[string]interface{}{"learning_rate": "0.01"},
			key:          "learning_rate",
			defaultValue: 0.001,
			expected:     0.001,
		},
		{
			name:         "zero_value",
			params:       map[string]interface{}{"learning_rate": float64(0.0)},
			key:          "learning_rate",
			defaultValue: 0.001,
			expected:     0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractFloat32Param(tt.params, tt.key, tt.defaultValue)
			if result != tt.expected {
				t.Errorf("extractFloat32Param() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestExtractBoolParam tests the extractBoolParam helper function
func TestExtractBoolParam(t *testing.T) {
	tests := []struct {
		name         string
		params       map[string]interface{}
		key          string
		defaultValue bool
		expected     bool
	}{
		{
			name:         "existing_bool_param",
			params:       map[string]interface{}{"centered": true},
			key:          "centered",
			defaultValue: false,
			expected:     true,
		},
		{
			name:         "missing_param",
			params:       map[string]interface{}{"nesterov": true},
			key:          "centered",
			defaultValue: false,
			expected:     false,
		},
		{
			name:         "wrong_type_param",
			params:       map[string]interface{}{"centered": "true"},
			key:          "centered",
			defaultValue: false,
			expected:     false,
		},
		{
			name:         "false_value",
			params:       map[string]interface{}{"centered": false},
			key:          "centered",
			defaultValue: true,
			expected:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractBoolParam(tt.params, tt.key, tt.defaultValue)
			if result != tt.expected {
				t.Errorf("extractBoolParam() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestExtractUint64Param tests the extractUint64Param helper function
func TestExtractUint64Param(t *testing.T) {
	tests := []struct {
		name         string
		params       map[string]interface{}
		key          string
		defaultValue uint64
		expected     uint64
	}{
		{
			name:         "existing_float64_param",
			params:       map[string]interface{}{"step_count": float64(100)},
			key:          "step_count",
			defaultValue: 0,
			expected:     100,
		},
		{
			name:         "missing_param",
			params:       map[string]interface{}{"learning_rate": float64(0.01)},
			key:          "step_count",
			defaultValue: 0,
			expected:     0,
		},
		{
			name:         "wrong_type_param",
			params:       map[string]interface{}{"step_count": "100"},
			key:          "step_count",
			defaultValue: 0,
			expected:     0,
		},
		{
			name:         "zero_value",
			params:       map[string]interface{}{"step_count": float64(0)},
			key:          "step_count",
			defaultValue: 5,
			expected:     0,
		},
		{
			name:         "fractional_value",
			params:       map[string]interface{}{"step_count": float64(10.7)},
			key:          "step_count",
			defaultValue: 0,
			expected:     10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractUint64Param(tt.params, tt.key, tt.defaultValue)
			if result != tt.expected {
				t.Errorf("extractUint64Param() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestExtractBufferState tests the extractBufferState helper function
func TestExtractBufferState(t *testing.T) {
	// Test with nil buffer
	result, err := extractBufferState(nil, 64, "test_buffer", "momentum")
	if err != nil {
		t.Errorf("extractBufferState() with nil buffer should not return error, got: %v", err)
	}
	if result != nil {
		t.Errorf("extractBufferState() with nil buffer should return nil, got: %v", result)
	}

	// Skip CGO tests due to invalid pointer access
	t.Log("Skipping CGO tests for extractBufferState - requires real Metal buffer")
}

// TestRestoreBufferState tests the restoreBufferState helper function
func TestRestoreBufferState(t *testing.T) {
	// Test with nil buffer
	err := restoreBufferState(nil, []float32{1.0, 2.0}, 8, "test_buffer")
	if err == nil {
		t.Error("restoreBufferState() with nil buffer should return error")
	}
	if err.Error() != "test_buffer buffer is nil" {
		t.Errorf("restoreBufferState() error message = %v, want 'test_buffer buffer is nil'", err.Error())
	}

	// Test with size mismatch
	mockBuffer := unsafe.Pointer(uintptr(0x1000))
	err = restoreBufferState(mockBuffer, []float32{1.0, 2.0}, 16, "test_buffer")
	if err == nil {
		t.Error("restoreBufferState() with size mismatch should return error")
	}
	expectedErr := "data size mismatch for test_buffer: expected 4 elements, got 2"
	if err.Error() != expectedErr {
		t.Errorf("restoreBufferState() error message = %v, want %v", err.Error(), expectedErr)
	}

	// Skip CGO tests due to invalid pointer access
	t.Log("Skipping CGO tests for restoreBufferState - requires real Metal buffer")
}

// TestCalculateTensorSize tests the calculateTensorSize helper function
func TestCalculateTensorSize(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		expected int
	}{
		{
			name:     "scalar",
			shape:    []int{},
			expected: 1,
		},
		{
			name:     "vector",
			shape:    []int{10},
			expected: 10,
		},
		{
			name:     "matrix",
			shape:    []int{10, 20},
			expected: 200,
		},
		{
			name:     "3d_tensor",
			shape:    []int{5, 10, 20},
			expected: 1000,
		},
		{
			name:     "4d_tensor",
			shape:    []int{2, 3, 4, 5},
			expected: 120,
		},
		{
			name:     "single_element",
			shape:    []int{1},
			expected: 1,
		},
		{
			name:     "zero_dimension",
			shape:    []int{0},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateTensorSize(tt.shape)
			if result != tt.expected {
				t.Errorf("calculateTensorSize() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestValidateStateType tests the validateStateType helper function
func TestValidateStateType(t *testing.T) {
	tests := []struct {
		name           string
		optimizerType  string
		state          *OptimizerState
		expectError    bool
	}{
		{
			name:          "matching_type",
			optimizerType: "Adam",
			state:         &OptimizerState{Type: "Adam"},
			expectError:   false,
		},
		{
			name:          "mismatched_type",
			optimizerType: "Adam",
			state:         &OptimizerState{Type: "SGD"},
			expectError:   true,
		},
		{
			name:          "empty_state_type",
			optimizerType: "Adam",
			state:         &OptimizerState{Type: ""},
			expectError:   true,
		},
		{
			name:          "empty_optimizer_type",
			optimizerType: "",
			state:         &OptimizerState{Type: "Adam"},
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateStateType(tt.optimizerType, tt.state)
			if tt.expectError {
				if err == nil {
					t.Error("validateStateType() should have returned error")
				}
			} else {
				if err != nil {
					t.Errorf("validateStateType() should not have returned error: %v", err)
				}
			}
		})
	}
}

// TestExtractBufferIndex tests the extractBufferIndex helper function
func TestExtractBufferIndex(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int
	}{
		{
			name:     "momentum_buffer",
			input:    "momentum_0",
			expected: 0,
		},
		{
			name:     "variance_buffer",
			input:    "variance_5",
			expected: 5,
		},
		{
			name:     "squared_grad_avg_buffer",
			input:    "squared_grad_avg_10",
			expected: 10,
		},
		{
			name:     "no_underscore",
			input:    "momentum",
			expected: -1,
		},
		{
			name:     "non_numeric_suffix",
			input:    "momentum_abc",
			expected: -1,
		},
		{
			name:     "empty_string",
			input:    "",
			expected: -1,
		},
		{
			name:     "only_underscore",
			input:    "_",
			expected: -1,
		},
		{
			name:     "multiple_underscores",
			input:    "momentum_buffer_5",
			expected: 5,
		},
		{
			name:     "negative_index",
			input:    "momentum_-1",
			expected: -1,
		},
		{
			name:     "large_index",
			input:    "momentum_999",
			expected: 999,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractBufferIndex(tt.input)
			if result != tt.expected {
				t.Errorf("extractBufferIndex() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestOptimizerState tests the OptimizerState struct
func TestOptimizerState(t *testing.T) {
	state := &OptimizerState{
		Type: "Adam",
		Parameters: map[string]interface{}{
			"learning_rate": float64(0.001),
			"beta1":         float64(0.9),
			"beta2":         float64(0.999),
			"epsilon":       float64(1e-8),
			"weight_decay":  float64(0.0),
			"step_count":    float64(100),
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

	// Test state structure
	if state.Type != "Adam" {
		t.Errorf("Expected Type 'Adam', got '%s'", state.Type)
	}

	if len(state.Parameters) != 6 {
		t.Errorf("Expected 6 parameters, got %d", len(state.Parameters))
	}

	if len(state.StateData) != 2 {
		t.Errorf("Expected 2 state data entries, got %d", len(state.StateData))
	}

	// Test parameter extraction
	lr := extractFloat32Param(state.Parameters, "learning_rate", 0.0)
	if lr != 0.001 {
		t.Errorf("Expected learning_rate 0.001, got %f", lr)
	}

	stepCount := extractUint64Param(state.Parameters, "step_count", 0)
	if stepCount != 100 {
		t.Errorf("Expected step_count 100, got %d", stepCount)
	}

	// Test state data structure
	for i, tensor := range state.StateData {
		if len(tensor.Shape) != 2 {
			t.Errorf("Tensor %d: expected 2D shape, got %dD", i, len(tensor.Shape))
		}
		if len(tensor.Data) != 200 {
			t.Errorf("Tensor %d: expected 200 elements, got %d", i, len(tensor.Data))
		}
	}

	t.Log("OptimizerState structure test passed")
}
