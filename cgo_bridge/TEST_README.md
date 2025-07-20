# CGO Bridge Test Suite

This directory contains a comprehensive test suite for the `go-metal/cgo_bridge` package, which provides the critical bridge between Go and Metal/Objective-C code.

## Test Files

### `setup_test.go`
- **Shared test resource management** using the same pattern as `go-metal/engine`
- **TestMain** function for coordinated setup and cleanup
- **Helper functions** for creating test engines and buffers with shared resources
- **Buffer pool exhaustion handling** with graceful test skipping

### `bridge_test.go` 
- **Core functionality tests** for Metal device, command queue, and buffer lifecycle
- **Configuration validation** for `TrainingConfig` and `InferenceConfig` structs
- **Basic buffer operations** (allocation, data copying, zeroing)
- **Error handling tests** for invalid parameters
- **Device type constant validation**

### `advanced_test.go`
- **Memory operations** with int32 and float32 data types  
- **Buffer-to-buffer copying** operations
- **Command buffer lifecycle** management
- **Tensor type conversion** (with appropriate error handling)
- **Staging buffer operations** for CPU-GPU data transfer
- **Autorelease pool** integration with Metal operations
- **Configuration bounds testing** for edge cases
- **Device type usage patterns**

## Test Coverage

The test suite covers:

✅ **Metal Device Management**
- Device creation and destruction
- Shared device access
- Error handling for nil devices

✅ **Command Queue & Buffer Operations**  
- Command queue lifecycle
- Command buffer creation, commit, and completion
- Multiple command buffer management

✅ **Memory Buffer Operations**
- Buffer allocation and deallocation 
- Float32 and int32 data copying
- Buffer zeroing operations
- Buffer-to-buffer synchronous copying
- Staging buffer operations

✅ **Configuration Validation**
- TrainingConfig for SGD, Adam, and RMSProp optimizers
- InferenceConfig with dynamic engine settings
- Parameter bounds checking
- Invalid configuration handling

✅ **Resource Management**
- Shared resource patterns to prevent buffer pool exhaustion
- Graceful test skipping when resources are unavailable
- Proper cleanup and autorelease pool management

✅ **Error Handling**
- Invalid parameter detection
- Buffer pool exhaustion scenarios
- CGO bridge error codes

## Running Tests

```bash
# Run all tests
go test -v .

# Run specific test file
go test -v -run TestMetalDevice

# Run with timeout (useful for CGO operations)
go test -v -timeout 60s .
```

## Test Design Philosophy

### Shared Resources
- Uses the same `setup_test.go` pattern as `go-metal/engine`
- Prevents buffer pool exhaustion when running full test suite
- Centralizes Metal device and command queue management

### Graceful Degradation
- Tests skip rather than fail when Metal resources are unavailable
- Buffer pool exhaustion is handled gracefully with appropriate skipping

## Error Handling Examples

### Buffer Pool Exhaustion
```go
buffer, err := createTestBuffer(bufferSize, GPU, t)
if err != nil {
    return // Will skip if buffer pool exhausted
}
defer DeallocateMetalBuffer(buffer)
```

### Engine Creation with Resource Management
```go
engine, err := CreateTrainingEngine(device, config)
if err != nil {
    // Check for buffer pool exhaustion and skip gracefully
    if strings.Contains(err.Error(), "buffer pool at capacity") || 
       strings.Contains(err.Error(), "failed to allocate") {
        t.Skipf("Skipping test - buffer pool exhausted: %v", err)
        return
    }
    t.Fatalf("Failed to create training engine: %v", err)
}
defer DestroyTrainingEngine(engine)
```

### Metal Device Requirements
```go
device, err := getSharedDevice()
if err != nil {
    t.Skipf("Skipping test - Metal device not available: %v", err)
}
```
- Robust resource management ensures reliable test execution

### Comprehensive Coverage
- Tests both happy path and error conditions
- Validates configuration structures without expensive operations
- Covers memory operations, device management, and error handling

### Comprehensive Testing
- Tests actual training/inference engines with robust cleanup handling
- Covers both bridge functions and full engine lifecycle operations
- Uses appropriate buffer sizes for thorough testing while managing resources efficiently

## Key Features

1. **Robust Operation**: Designed with enhanced cleanup functionality for reliable CGO testing
2. **Resource Efficiency**: Uses shared resources and proper cleanup to prevent exhaustion
3. **Comprehensive Coverage**: Tests all major bridge functions and configuration types
4. **Realistic Usage**: Tests mirror real-world usage patterns of the bridge functions
5. **Error Resilience**: Proper handling of expected errors and edge cases
6. **Enhanced Reliability**: Benefits from improved cleanup methods that ensure stable test execution

This test suite ensures the reliability of the critical Go-Metal bridge layer while maintaining stability and performance.