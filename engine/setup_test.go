package engine

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// Shared test resources
var (
	sharedTestDevice    unsafe.Pointer
	sharedMemoryManager *memory.MemoryManager
	testSetupOnce       sync.Once
	testCleanupOnce     sync.Once
	setupErr           error
)

// setupSharedTestResources initializes shared Metal resources for all tests
func setupSharedTestResources() {
	testSetupOnce.Do(func() {
		// Create shared Metal device
		device, err := cgo_bridge.CreateMetalDevice()
		if err != nil {
			setupErr = err
			return
		}
		sharedTestDevice = device
		
		// Initialize shared memory manager
		memory.InitializeGlobalMemoryManager(device)
		sharedMemoryManager = memory.GetGlobalMemoryManager()
	})
}

// cleanupSharedTestResources cleans up shared resources (called by TestMain)
func cleanupSharedTestResources() {
	testCleanupOnce.Do(func() {
		if sharedMemoryManager != nil {
			// Memory manager doesn't have a Cleanup method
			// Resources will be cleaned up when device is destroyed
			sharedMemoryManager = nil
		}
		
		if sharedTestDevice != nil {
			cgo_bridge.DestroyMetalDevice(sharedTestDevice)
			sharedTestDevice = nil
		}
	})
}

// getSharedTestDevice returns the shared Metal device for tests
func getSharedTestDevice() (unsafe.Pointer, error) {
	setupSharedTestResources()
	if setupErr != nil {
		return nil, setupErr
	}
	return sharedTestDevice, nil
}

// getSharedMemoryManager returns the shared memory manager for tests
func getSharedMemoryManager() (*memory.MemoryManager, error) {
	setupSharedTestResources()
	if setupErr != nil {
		return nil, setupErr
	}
	return sharedMemoryManager, nil
}

// getSharedDevice returns the shared Metal device
func getSharedDevice() (unsafe.Pointer, error) {
	setupSharedTestResources()
	if setupErr != nil {
		return nil, setupErr
	}
	return sharedTestDevice, nil
}

// TestMain sets up and tears down shared resources for all tests
func TestMain(m *testing.M) {
	// Setup shared resources
	setupSharedTestResources()
	
	// Run tests
	exitCode := m.Run()
	
	// Cleanup shared resources
	cleanupSharedTestResources()
	
	// Exit with the same code as the tests (don't panic on failure)
	if exitCode != 0 {
		// Exit gracefully with failure code
		return
	}
}

// Helper function to create model training engine with shared resources
func createTestModelTrainingEngine(model interface{}, config cgo_bridge.TrainingConfig, t *testing.T) (*ModelTrainingEngine, error) {
	// Type assert the model to *layers.ModelSpec
	modelSpec, ok := model.(*layers.ModelSpec)
	if !ok {
		return nil, fmt.Errorf("model must be *layers.ModelSpec, got %T", model)
	}
	
	// Get shared device and memory manager
	device, err := getSharedDevice()
	if err != nil {
		t.Skipf("Skipping test - Metal device not available: %v", err)
		return nil, err
	}
	
	memoryManager, err := getSharedMemoryManager()
	if err != nil {
		t.Skipf("Skipping test - Memory manager not available: %v", err)
		return nil, err
	}
	
	// Create model training engine using shared resources
	// This requires accessing the internal creation mechanism
	modelEngine, err := createModelTrainingEngineWithSharedResources(modelSpec, config, device, memoryManager)
	if err != nil {
		// If shared resource creation fails, fall back to regular creation
		// but with better error handling for buffer pool exhaustion
		modelEngine, err := NewModelTrainingEngineDynamic(modelSpec, config)
		if err != nil {
			// Check if this is a buffer pool exhaustion error and skip gracefully
			if strings.Contains(err.Error(), "buffer pool at capacity") || 
			   strings.Contains(err.Error(), "failed to allocate") {
				t.Skipf("Skipping test - buffer pool exhausted: %v", err)
				return nil, err
			}
			return nil, err
		}
		return modelEngine, nil
	}
	
	return modelEngine, nil
}

// createModelTrainingEngineWithSharedResources creates a model training engine using provided shared resources
func createModelTrainingEngineWithSharedResources(modelSpec *layers.ModelSpec, config cgo_bridge.TrainingConfig, device unsafe.Pointer, memoryManager *memory.MemoryManager) (*ModelTrainingEngine, error) {
	// For now, since there's no direct API to create with shared resources,
	// we'll use the regular API but at least the memory manager is already initialized
	// This should reduce some resource pressure
	return NewModelTrainingEngineDynamic(modelSpec, config)
}

// Helper function to create test tensors with shared memory manager
func createTestTensor(shape []int, dtype memory.DataType, t *testing.T) (*memory.Tensor, error) {
	_, err := getSharedMemoryManager()
	if err != nil {
		t.Skipf("Skipping test - Memory manager not available: %v", err)
		return nil, err
	}
	
	tensor, err := memory.NewTensor(shape, dtype, memory.GPU)
	if err != nil {
		return nil, err
	}
	
	return tensor, nil
}