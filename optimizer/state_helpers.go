package optimizer

import (
	"fmt"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
)

// Common helper functions for optimizer state management

// extractBufferState extracts a single buffer's state to CPU
func extractBufferState(buffer unsafe.Pointer, bufferSize int, name string, stateType string) (*checkpoints.OptimizerTensor, error) {
	if buffer == nil {
		return nil, nil
	}

	numElements := bufferSize / 4 // 4 bytes per float32
	data, err := cgo_bridge.CopyMetalBufferToFloat32Array(buffer, numElements)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %v", name, err)
	}

	return &checkpoints.OptimizerTensor{
		Name:      name,
		Shape:     []int{len(data)},
		Data:      data,
		StateType: stateType,
	}, nil
}

// restoreBufferState restores a single buffer's state from CPU to GPU
func restoreBufferState(buffer unsafe.Pointer, data []float32, bufferSize int, name string) error {
	if buffer == nil {
		return fmt.Errorf("%s buffer is nil", name)
	}

	expectedElements := bufferSize / 4
	if len(data) != expectedElements {
		return fmt.Errorf("data size mismatch for %s: expected %d elements, got %d",
			name, expectedElements, len(data))
	}

	if err := cgo_bridge.CopyFloat32ArrayToMetalBuffer(buffer, data); err != nil {
		return fmt.Errorf("failed to restore %s: %v", name, err)
	}

	return nil
}

// extractFloat32Param safely extracts a float32 parameter from the state map
func extractFloat32Param(params map[string]interface{}, key string, defaultValue float32) float32 {
	if val, ok := params[key].(float64); ok {
		return float32(val)
	}
	return defaultValue
}

// extractBoolParam safely extracts a bool parameter from the state map
func extractBoolParam(params map[string]interface{}, key string, defaultValue bool) bool {
	if val, ok := params[key].(bool); ok {
		return val
	}
	return defaultValue
}

// extractUint64Param safely extracts a uint64 parameter from the state map  
func extractUint64Param(params map[string]interface{}, key string, defaultValue uint64) uint64 {
	if val, ok := params[key].(float64); ok {
		return uint64(val)
	}
	return defaultValue
}

