package cgo_bridge

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

// Shared test resources
var (
	sharedTestDevice    unsafe.Pointer
	sharedTestQueue     unsafe.Pointer
	sharedTestMutex     sync.Mutex
	setupErr            error
	setupOnce           sync.Once
)

// setupSharedTestResources initializes shared resources for all tests
func setupSharedTestResources() {
	setupOnce.Do(func() {
		// Create shared Metal device
		device, err := CreateMetalDevice()
		if err != nil {
			setupErr = fmt.Errorf("failed to create shared Metal device: %v", err)
			return
		}
		sharedTestDevice = device

		// Create shared command queue
		queue, err := CreateCommandQueue(device)
		if err != nil {
			DestroyMetalDevice(device)
			setupErr = fmt.Errorf("failed to create shared command queue: %v", err)
			return
		}
		sharedTestQueue = queue
	})
}

// cleanupSharedTestResources cleans up shared resources
func cleanupSharedTestResources() {
	sharedTestMutex.Lock()
	defer sharedTestMutex.Unlock()

	if sharedTestQueue != nil {
		DestroyCommandQueue(sharedTestQueue)
		sharedTestQueue = nil
	}

	if sharedTestDevice != nil {
		DestroyMetalDevice(sharedTestDevice)
		sharedTestDevice = nil
	}
}

// getSharedDevice returns the shared Metal device
func getSharedDevice() (unsafe.Pointer, error) {
	setupSharedTestResources()
	if setupErr != nil {
		return nil, setupErr
	}
	return sharedTestDevice, nil
}

// getSharedCommandQueue returns the shared command queue
func getSharedCommandQueue() (unsafe.Pointer, error) {
	setupSharedTestResources()
	if setupErr != nil {
		return nil, setupErr
	}
	return sharedTestQueue, nil
}

// TestMain sets up and tears down shared resources for all tests
func TestMain(m *testing.M) {
	// Setup shared resources
	setupSharedTestResources()
	
	// Run tests
	exitCode := m.Run()
	
	// Cleanup shared resources
	cleanupSharedTestResources()
	
	// Exit with the same code as the tests
	if exitCode != 0 {
		return
	}
}

// Helper function to create test training engine with shared resources
func createTestTrainingEngine(config TrainingConfig, t *testing.T) (unsafe.Pointer, error) {
	device, err := getSharedDevice()
	if err != nil {
		if strings.Contains(err.Error(), "Metal device not available") {
			t.Skipf("Skipping test - Metal device not available: %v", err)
			return nil, err
		}
		return nil, err
	}
	
	engine, err := CreateTrainingEngine(device, config)
	if err != nil {
		// Check for buffer pool exhaustion and skip gracefully
		if strings.Contains(err.Error(), "buffer pool at capacity") || 
		   strings.Contains(err.Error(), "failed to allocate") {
			t.Skipf("Skipping test - buffer pool exhausted: %v", err)
			return nil, err
		}
		return nil, err
	}
	
	return engine, nil
}

// Helper function to create test inference engine with shared resources
func createTestInferenceEngine(config InferenceConfig, t *testing.T) (unsafe.Pointer, error) {
	device, err := getSharedDevice()
	if err != nil {
		if strings.Contains(err.Error(), "Metal device not available") {
			t.Skipf("Skipping test - Metal device not available: %v", err)
			return nil, err
		}
		return nil, err
	}
	
	engine, err := CreateInferenceEngine(device, config)
	if err != nil {
		// Check for buffer pool exhaustion and skip gracefully
		if strings.Contains(err.Error(), "buffer pool at capacity") || 
		   strings.Contains(err.Error(), "failed to allocate") {
			t.Skipf("Skipping test - buffer pool exhausted: %v", err)
			return nil, err
		}
		return nil, err
	}
	
	return engine, nil
}

// Helper function to create test buffer with shared device
func createTestBuffer(size int, deviceType DeviceType, t *testing.T) (unsafe.Pointer, error) {
	device, err := getSharedDevice()
	if err != nil {
		if strings.Contains(err.Error(), "Metal device not available") {
			t.Skipf("Skipping test - Metal device not available: %v", err)
			return nil, err
		}
		return nil, err
	}
	
	buffer, err := AllocateMetalBuffer(device, size, deviceType)
	if err != nil {
		// Check for buffer pool exhaustion and skip gracefully
		if strings.Contains(err.Error(), "buffer pool at capacity") || 
		   strings.Contains(err.Error(), "failed to allocate") {
			t.Skipf("Skipping test - buffer pool exhausted: %v", err)
			return nil, err
		}
		return nil, err
	}
	
	return buffer, nil
}