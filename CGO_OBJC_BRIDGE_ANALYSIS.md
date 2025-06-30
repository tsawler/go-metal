# Go -> CGO -> Objective-C Bridge Analysis for go-metal

This document provides a comprehensive analysis of how the go-metal library implements the bridge between Go, CGO, and Objective-C to interface with Apple's Metal Performance Shaders (MPS) and Metal framework. It focuses on object counting, resource management, memory lifecycle, and the intricate details of cross-language interoperability.

## Architecture Overview

The go-metal library uses a three-layer architecture:

1. **Go Layer** (`tensor/`, `training/`): High-level tensor operations and ML primitives
2. **CGO Bridge Layer** (`metal_bridge/`): Go wrappers around C function calls
3. **Objective-C Implementation Layer** (`metal_bridge.h/.m`): Direct Metal/MPS API calls

```
┌─────────────────────────────────────────────────────┐
│                 Go Layer                            │
│  tensor.go, mps_ops.go, autograd.go, etc.          │
└─────────────────────┬───────────────────────────────┘
                      │ Go struct method calls
┌─────────────────────▼───────────────────────────────┐
│                CGO Bridge Layer                     │
│  metal.go (Go wrapper structs with CGO calls)      │
└─────────────────────┬───────────────────────────────┘
                      │ C.FunctionName() calls
┌─────────────────────▼───────────────────────────────┐
│            Objective-C Implementation               │
│  metal_bridge.h/.m (C function wrappers around     │
│                     Objective-C Metal/MPS APIs)    │
└─────────────────────┬───────────────────────────────┘
                      │ [obj method:param] calls
┌─────────────────────▼───────────────────────────────┐
│                Metal/MPS Framework                  │
│      Apple's native GPU computation APIs           │
└─────────────────────────────────────────────────────┘
```

**Major Architecture Update (2025)**: The library has transitioned from a custom compute engine to fully leverage MPSGraph for all GPU operations, including gradient computation. This provides better performance and compatibility with Apple's Metal ecosystem.

## Type System and Object Representation

### Header File Type Abstractions (`metal_bridge.h`)

The header file provides a crucial abstraction layer that allows the same code to compile in both C and Objective-C contexts:

```c
#ifdef __OBJC__
// When compiling Objective-C (.m files)
typedef id<MTLDevice> MTLDeviceRef;
typedef id<MTLCommandQueue> MTLCommandQueueRef;
typedef MPSGraph* MPSGraphRef;
// ... (actual Objective-C types)
#else
// When compiling C/CGO (.go files)
typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MPSGraphRef;
// ... (opaque pointers)
#endif
```

This design ensures:
- Go sees everything as `void*` (opaque pointers)
- Objective-C implementation uses proper typed protocols and classes
- Type safety is maintained at the Objective-C boundary
- No direct Objective-C syntax leaks into Go code

### Go Wrapper Structs

Each Metal/MPS object type has a corresponding Go wrapper struct that manages the object lifecycle:

```go
type Device struct {
    c_device C.MTLDeviceRef       // Opaque pointer to Objective-C object
    released int32                 // Atomic flag preventing double-release
}

type Buffer struct {
    c_buffer  C.MTLBufferRef      // Opaque pointer
    length    uintptr             // Cached length for efficiency
    inUse     bool                // Pool management flag
    refCount  int32               // Reference count for shared buffers
    allocator *BufferAllocator    // Back-reference for pool management
    released  int32               // Double-release prevention
}
```

## Memory Management and Reference Counting

### Object Lifecycle Management

The library implements a sophisticated multi-level reference counting system:

#### 1. **CGO Object Lifetime Management**

When objects cross from Objective-C to Go:

```go
func CreateSystemDefaultDevice() *Device {
    c_dev := C.CreateSystemDefaultDevice()  // Creates MTLDevice in ObjC
    if c_dev == nil {
        return nil
    }
    dev := &Device{c_device: c_dev}
    
    // Set Go finalizer to ensure cleanup
    runtime.SetFinalizer(dev, func(d *Device) {
        d.safeRelease()
    })
    return dev
}
```

```objc
// In metal_bridge.m
id<MTLDevice> CreateSystemDefaultDevice() {
    return MTLCreateSystemDefaultDevice();  // Returns +1 retained object
}
```

#### 2. **Explicit Retain/Release Pattern**

For objects that need explicit lifetime management:

```go
func (d *Device) CreateBufferWithBytes(data interface{}, options C.size_t) (*Buffer, error) {
    // ... create buffer via C call ...
    c_buf := C.CreateBufferWithBytes(d.c_device, unsafe.Pointer(&v[0]), byteLength, options)
    
    // CRITICAL: Explicitly retain for Go ownership
    C.CFRetain((C.CFTypeRef)(unsafe.Pointer(c_buf)))
    
    buf := &Buffer{
        c_buffer: c_buf,
        // ... other fields ...
    }
    runtime.SetFinalizer(buf, (*Buffer).finalize)
    return buf, nil
}
```

The `CFRetain` call is crucial because:
- Objective-C returns the object with +1 reference count
- Go takes ownership, so we explicitly retain to ensure it survives
- The finalizer will call `CFRelease` to balance the retain

#### 3. **Safe Release with Atomic Operations**

To prevent double-release bugs (common source of crashes in CGO):

```go
func (d *Device) safeRelease() {
    if atomic.CompareAndSwapInt32(&d.released, 0, 1) {
        if d.c_device != nil {
            C.ReleaseMetalObject(unsafe.Pointer(d.c_device))
            d.c_device = nil
        }
    }
}
```

```objc
void ReleaseMetalObject(void* obj) {
    if (obj) {
        @try {
            CFRelease((__bridge CFTypeRef)obj);  // Safe release
        } @catch (NSException *exception) {
            // Ignore exceptions - object may already be deallocated
            NSLog(@"Warning: Exception during ReleaseMetalObject: %@", exception);
        }
    }
}
```

### Buffer Pool Management

The library implements a sophisticated buffer pooling system to minimize allocation overhead:

#### Size-Based Pooling Strategy

```go
type BufferAllocator struct {
    pools map[int]*BufferPool    // Key is power-of-2 exponent
    // ... configuration and stats ...
}

func sizeToPoolKey(size uint64) int {
    roundedSize := roundUpToPowerOf2(size)
    // Find log2 of rounded size
    key := 0
    temp := roundedSize
    for temp > 1 {
        temp >>= 1
        key++
    }
    return key
}
```

This creates pools for sizes: 1KB, 2KB, 4KB, 8KB, 16KB, etc., minimizing fragmentation.

#### Pool Allocation Process

```go
func (a *BufferAllocator) Allocate(sizeInBytes uint64, options C.NSUInteger) (*Buffer, error) {
    // 1. Check if buffer exists in appropriate pool
    poolKey := sizeToPoolKey(sizeInBytes)
    pool := a.getOrCreatePool(poolKey)
    
    pool.mutex.Lock()
    if len(pool.buffers) > 0 {
        // 2. Reuse existing buffer
        buffer := pool.buffers[len(pool.buffers)-1]
        pool.buffers = pool.buffers[:len(pool.buffers)-1]
        pool.mutex.Unlock()
        
        // 3. Reset buffer state for reuse
        buffer.inUse = true
        buffer.refCount = 1
        return buffer, nil
    }
    pool.mutex.Unlock()
    
    // 4. Create new buffer if pool empty
    return a.allocateNew(actualSize, options)
}
```

#### Buffer Release and Pooling

```go
func (a *BufferAllocator) Release(buffer *Buffer) {
    // 1. Check pool capacity limits
    if len(pool.buffers) >= a.maxPoolSize {
        buffer.releaseNow()  // Pool full, release directly
        return
    }
    
    // 2. Check global memory limits
    if totalFree+uint64(buffer.length) > a.maxTotalMemory {
        buffer.releaseNow()  // Would exceed limit
        return
    }
    
    // 3. Return to pool for reuse
    buffer.inUse = false
    buffer.refCount = 0
    pool.buffers = append(pool.buffers, buffer)
}
```

## Data Transfer Mechanisms

### Go Slice to Metal Buffer Transfer

The most critical aspect is safely transferring Go slice data to Metal buffers:

```go
func (d *Device) CreateBufferWithBytes(data interface{}, options C.size_t) (*Buffer, error) {
    switch v := data.(type) {
    case []float32:
        byteLength := C.size_t(len(v) * int(unsafe.Sizeof(v[0])))
        
        // CRITICAL: Pass pointer to Go slice data
        c_buf := C.CreateBufferWithBytes(d.c_device, unsafe.Pointer(&v[0]), byteLength, options)
        
        // The Go slice MUST remain alive until GPU operations complete
        // This is ensured by:
        // 1. Tensor holding reference to Go slice
        // 2. Buffer operations being synchronous or properly sequenced
        // 3. MTLResourceStorageModeShared for direct memory access
```

```objc
id<MTLBuffer> CreateBufferWithBytes(id<MTLDevice> device, const void* data, size_t length, size_t resourceOptions) {
    // Creates Metal buffer that may directly reference Go memory
    return [device newBufferWithBytes:data length:length options:resourceOptions];
}
```

### Memory Safety Considerations

1. **Shared Memory Mode**: When using `MTLResourceStorageModeShared`, the Metal buffer directly references Go memory
2. **Lifetime Coordination**: The Go slice must remain alive until all GPU operations complete
3. **Synchronization**: Operations must be properly sequenced to avoid race conditions

## Asynchronous Operation Management

### Completion Handler Bridge

The library implements a sophisticated callback system to bridge Objective-C blocks with Go functions:

```go
// Global completion handler registry
var (
    completionHandlers = make(map[int]func(status int))
    handlerMutex       sync.Mutex
    nextHandlerID      int = 1  // Start from 1 to avoid nil pointer issues
)

func (cb *CommandBuffer) AddCompletedHandler(handler func(status int)) {
    // 1. Generate unique ID and store Go function
    handlerMutex.Lock()
    handlerID := nextHandlerID
    nextHandlerID++
    completionHandlers[handlerID] = handler
    handlerMutex.Unlock()
    
    // 2. Convert ID to userData pointer
    userData := unsafe.Pointer(uintptr(handlerID))
    
    // 3. Register with Objective-C completion handler
    C.AddCommandBufferCompletedHandler(cb.c_commandBuffer, userData, 
        (C.CompletionHandlerFunc)(C.goCommandBufferCompleted))
}
```

### C Function Bridge

The CGO `//export` directive creates a C function that Objective-C can call:

```go
//export goCommandBufferCompleted
func goCommandBufferCompleted(userData unsafe.Pointer, statusCode C.long) {
    if userData != nil {
        handlerID := int(uintptr(userData))  // Convert back to handler ID
        
        handlerMutex.Lock()
        handler, exists := completionHandlers[handlerID]
        if exists {
            delete(completionHandlers, handlerID)  // Cleanup
        }
        handlerMutex.Unlock()
        
        if exists {
            // Execute in goroutine to avoid blocking Metal callback thread
            go handler(int(statusCode))
        }
    }
}
```

### Objective-C Completion Handler

```objc
void AddCommandBufferCompletedHandler(id<MTLCommandBuffer> commandBuffer, void* userData, CompletionHandlerFunc handler) {
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buffer) {
        // Call the C function pointer with user data and status
        handler(userData, (long)[buffer status]);
    }];
}
```

## MPSGraph Integration

The library provides complete coverage of element-wise operations through MPSGraph:
- **AddMPS**: Addition using `MPSGraphAddition`
- **SubMPS**: Subtraction using `MPSGraphSubtraction`
- **MulMPS**: Multiplication using `MPSGraphMultiplication`
- **DivMPS**: Division using `MPSGraphDivision`

All operations follow the same memory management and lifecycle patterns described below.

### Graph Object Lifecycle

MPSGraph objects follow the same pattern but with additional complexity due to compilation and execution phases:

```go
type Graph struct {
    c_graph  C.MPSGraphRef
    released int32
}

func NewGraph() *Graph {
    c_graph := C.CreateMPSGraph()  // Creates MPSGraph instance
    graph := &Graph{c_graph: c_graph}
    runtime.SetFinalizer(graph, func(g *Graph) {
        g.safeRelease()
    })
    return graph
}
```

### Compilation and Execution Pipeline

1. **Graph Construction**: Operations are added to create computation graph
2. **Compilation**: Graph is compiled for specific device with input/output shapes
3. **Execution**: Compiled graph is executed with actual tensor data

```go
func (g *Graph) Compile(device *GraphDevice, inputTensors, targetTensors []*GraphTensor, compilationDescriptor *GraphCompilationDescriptor) *GraphExecutable {
    // Convert Go slices to C arrays
    inputCount := len(inputTensors)
    targetCount := len(targetTensors)
    
    cInputTensors := make([]C.MPSGraphTensorRef, inputCount)
    cTargetTensors := make([]C.MPSGraphTensorRef, targetCount)
    
    // Populate C arrays...
    
    c_executable := C.MPSGraphCompile(g.c_graph, device.c_device, 
        &cInputTensors[0], C.size_t(inputCount),
        &cTargetTensors[0], C.size_t(targetCount), 
        compilationDescriptor.c_descriptor)
    
    // Wrap and manage executable lifecycle...
}
```

## Tensor Reference Counting Integration

### Multi-Level Reference Management

Tensors implement their own reference counting that coordinates with buffer management:

```go
type Tensor struct {
    // ... tensor fields ...
    gpuBuffer    interface{}  // Points to *metal_bridge.Buffer
    refCount     int32        // Tensor-level reference count
}

func (t *Tensor) Retain() {
    if (t.Device == GPU || t.Device == PersistentGPU) && t.gpuBuffer != nil {
        atomic.AddInt32(&t.refCount, 1)
        
        // Also retain underlying buffer
        if buffer, ok := t.gpuBuffer.(interface{ Retain() }); ok {
            buffer.Retain()
        }
    }
}
```

### GPU Computation Graph Integration

The computation graph maintains references to ensure tensors remain alive during async operations:

```go
func (g *GPUComputationGraph) AddOperation(opType string, inputs []*Tensor, ...) {
    // Increment reference counts for input tensors
    for _, tensor := range inputs {
        if tensor != nil {
            g.tensorRefs[tensor]++
            tensor.Retain()
        }
    }
    
    // ... create and queue operation ...
    // Supports all element-wise operations: Add, Sub, Mul, Div, MatMul, ReLU, etc.
}

func (g *GPUComputationGraph) cleanupOperation(graphOp *GraphOperation) error {
    // Decrement reference counts after operation completes
    for _, tensor := range graphOp.InputTensors {
        if tensor != nil {
            g.tensorRefs[tensor]--
            tensor.Release()
            
            // Remove from tracking when count reaches zero
            if g.tensorRefs[tensor] <= 0 {
                delete(g.tensorRefs, tensor)
            }
        }
    }
}
```

## Error Handling and Safety

### Exception Safety at Objective-C Boundary

```objc
MPSGraph* CreateMPSGraph() {
    @try {
        MPSGraph* graph = [[MPSGraph alloc] init];
        if (graph == nil) {
            NSLog(@"Failed to allocate MPSGraph");
            return nil;
        }
        return graph;
    } @catch (NSException *exception) {
        NSLog(@"Exception creating MPSGraph: %@", exception);
        return nil;
    }
}
```

### Memory Safety Patterns

1. **Nil Checks**: All C pointers are checked before use
2. **Atomic Operations**: Reference counting uses atomic operations
3. **Finalizers**: Go finalizers ensure cleanup even if explicit cleanup is missed
4. **Exception Handling**: Objective-C exceptions are caught and converted to Go errors

## MPSGraph Integration for Machine Learning

### Gradient Computation Architecture

The library now implements automatic differentiation using MPSGraph's native gradient operations:

```go
// Conv2D gradient computation using MPSGraph
func (op *Conv2dOp) computeInputGradientMPSGraph(gradOutput *Tensor) (*Tensor, error) {
    engine, err := GetMPSGraphEngine()
    
    // Create cached graph for gradient computation
    createGraphFunc := func() *cachedGraph {
        graph := metal_bridge.NewGraph()
        
        // Create placeholder tensors
        gradOutputPlaceholder := graph.PlaceholderTensor(gradOutput.Shape, int(metal_bridge.MPSDataTypeFloat32))
        weightsPlaceholder := graph.PlaceholderTensor(op.weight.Shape, int(metal_bridge.MPSDataTypeFloat32))
        
        // Use native MPSGraph gradient operation
        inputGradTensor := graph.Convolution2DDataGradient(
            gradOutputPlaceholder,
            weightsPlaceholder,
            op.input.Shape,  // Original input shape
            op.stride, op.stride,
            op.dilation, op.dilation,
            op.padding, op.padding, op.padding, op.padding,
            op.groups,
        )
        
        // Compile the graph
        executable := graph.Compile(graphDevice, nil, nil, nil)
        
        return &cachedGraph{
            executable:    executable,
            inputTensors:  []*metal_bridge.GraphTensor{gradOutputPlaceholder, weightsPlaceholder},
            resultTensors: []*metal_bridge.GraphTensor{inputGradTensor},
        }
    }
    
    // Execute with caching for performance
    return engine.executeGraph(cachedGraph, inputs)
}
```

### Key MPSGraph Operations Added

1. **Convolution2DDataGradient**: Computes gradients with respect to input data
2. **Convolution2DWeightsGradient**: Computes gradients with respect to convolution weights
3. **ConvolutionTranspose2D**: Implements transposed convolution for gradient computation

### Graph Caching Strategy

The library implements a sophisticated caching mechanism for compiled MPSGraph operations:

```go
type MPSGraphEngine struct {
    device       *metal_bridge.Device
    graphDevice  *metal_bridge.GraphDevice
    commandQueue *metal_bridge.CommandQueue
    
    // Cache for compiled graph executables
    cacheMutex sync.Mutex
    graphCache map[string]*cachedGraph
}
```

This caching significantly improves performance by:
- Avoiding recompilation of identical graphs
- Reusing compiled executables across operations
- Maintaining thread-safe access to cached resources

## Performance Considerations

### Memory Pool Benefits

1. **Reduced Allocation Overhead**: Reusing buffers avoids Metal allocation costs
2. **Reduced Fragmentation**: Power-of-2 sizing reduces memory fragmentation
3. **Improved Cache Locality**: Buffer reuse improves memory access patterns

### CGO Call Overhead

1. **Batch Operations**: Multiple operations are batched to reduce CGO overhead
2. **Async Patterns**: Asynchronous operations allow overlapping computation and data transfer
3. **Buffer Reuse**: Pool management reduces frequent allocations

### MPSGraph Optimizations

1. **Graph Compilation Caching**: Compiled graphs are cached and reused
2. **Native Gradient Operations**: Using MPSGraph's built-in gradient functions avoids manual implementation overhead
3. **Persistent GPU Tensors**: Keeping intermediate results on GPU reduces data transfer

## Best Practices and Lessons Learned

### Memory Management
1. Always pair `CFRetain` with `CFRelease`
2. Use atomic operations for reference counting
3. Implement finalizers as safety nets
4. Check for nil pointers at every boundary

### Performance
1. Pool frequently allocated objects
2. Batch CGO calls when possible
3. Use asynchronous operations for GPU work
4. Minimize data copying between CPU and GPU

### Safety
1. Wrap all Objective-C calls in exception handlers
2. Use atomic flags to prevent double-release
3. Validate all parameters at boundaries
4. Keep Go data alive during GPU operations

### Debugging
1. Add extensive logging for resource lifecycle
2. Track reference counts for leak detection
3. Use meaningful names for completion handlers
4. Implement statistics for pool monitoring

## MPS Operations Standardization and Memory Management (2025 Update)

### Standardized Binary Operation Pattern

A major improvement to the MPSGraph integration was the implementation of a standardized helper function for binary operations, reducing code duplication and improving maintainability:

```go
// executeMPSBinaryOp executes a binary MPS operation with automatic buffer cleanup
func executeMPSBinaryOp(a, b *Tensor, opName string, graphBuilder func(*metal_bridge.Graph, *metal_bridge.GraphTensor, *metal_bridge.GraphTensor) *metal_bridge.GraphTensor) (*Tensor, error) {
    if !isCompatibleForOp(a, b) {
        return nil, fmt.Errorf("tensors are not compatible for %s", opName)
    }

    engine, err := GetMPSGraphEngine()
    if err != nil {
        return nil, fmt.Errorf("failed to get MPSGraph engine: %v", err)
    }

    // Standard graph caching and compilation pattern
    cacheKey := generateCacheKey(opName, a, b)
    cached, err := engine.getOrCreateGraph(cacheKey, func() *cachedGraph {
        graph := metal_bridge.NewGraph()
        dataType := convertDTypeToMPS(a.DType)
        placeholderA := graph.PlaceholderTensor(a.Shape, dataType)
        placeholderB := graph.PlaceholderTensor(b.Shape, dataType)
        resultTensor := graphBuilder(graph, placeholderA, placeholderB)
        compilationDescriptor := metal_bridge.NewGraphCompilationDescriptor()
        executable := graph.Compile(engine.graphDevice, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}, compilationDescriptor)
        return &cachedGraph{executable, []*metal_bridge.GraphTensor{placeholderA, placeholderB}, []*metal_bridge.GraphTensor{resultTensor}}
    })

    // ... buffer creation and execution with proper finalizer-based cleanup
}
```

### Finalizer-Based Buffer Management

All MPS operations now follow the documented finalizer-based cleanup pattern rather than attempting explicit cleanup:

```go
// Binary operations using standardized helper
func AddMPS(a, b *Tensor) (*Tensor, error) {
    return executeMPSBinaryOp(a, b, "add", func(graph *metal_bridge.Graph, placeholderA, placeholderB *metal_bridge.GraphTensor) *metal_bridge.GraphTensor {
        return graph.Addition(placeholderA, placeholderB)
    })
}

func SubMPS(a, b *Tensor) (*Tensor, error) {
    return executeMPSBinaryOp(a, b, "sub", func(graph *metal_bridge.Graph, placeholderA, placeholderB *metal_bridge.GraphTensor) *metal_bridge.GraphTensor {
        return graph.Subtraction(placeholderA, placeholderB)
    })
}

func MulMPS(a, b *Tensor) (*Tensor, error) {
    return executeMPSBinaryOp(a, b, "mul", func(graph *metal_bridge.Graph, placeholderA, placeholderB *metal_bridge.GraphTensor) *metal_bridge.GraphTensor {
        return graph.Multiplication(placeholderA, placeholderB)
    })
}

func DivMPS(a, b *Tensor) (*Tensor, error) {
    return executeMPSBinaryOp(a, b, "div", func(graph *metal_bridge.Graph, placeholderA, placeholderB *metal_bridge.GraphTensor) *metal_bridge.GraphTensor {
        return graph.Division(placeholderA, placeholderB)
    })
}
```

### Complex Operations with Documented Memory Management

For operations that cannot use the binary helper (unary operations, convolutions, pooling), explicit documentation of the finalizer-based approach was added:

```go
func ReLUMPS(a *Tensor) (*Tensor, error) {
    // ... graph creation and caching ...
    
    aBuffer, err := createMPSBufferFromTensor(a, engine.device)
    if err != nil {
        return nil, fmt.Errorf("failed to create buffer for tensor A: %v", err)
    }
    // Cleaned up by finalizers
    
    resultBuffer, err := createMPSResultBuffer(result.Shape, result.DType, engine.device)
    if err != nil {
        return nil, fmt.Errorf("failed to create result buffer: %v", err)
    }
    // Cleaned up by finalizers
    
    executionDescriptor := metal_bridge.NewGraphExecutionDescriptor()
    if executionDescriptor == nil {
        return nil, fmt.Errorf("failed to create execution descriptor")
    }
    // Cleaned up by finalizers
    
    // ... execution ...
}
```

### Memory Management Pattern Compliance

This refactoring ensures all MPS operations follow the documented memory management principles:

1. **Private `safeRelease()` Methods**: All Metal objects use private cleanup methods designed for finalizer use only
2. **No Explicit Cleanup**: Temporary buffers rely on Go's finalizer system for cleanup
3. **Atomic Reference Counting**: Buffer reference counting uses atomic operations for thread safety
4. **Exception Safety**: All cleanup is handled by finalizers, preventing memory leaks even if operations fail

### Performance Benefits

The standardization provides several performance improvements:

1. **Reduced Code Duplication**: Binary operations reduced from ~280 lines to ~20 lines of standardized code
2. **Consistent Graph Caching**: All operations use the same caching strategy for compiled graphs
3. **Unified Error Handling**: Standardized error patterns reduce debugging complexity
4. **Maintainability**: Single implementation point for binary operation patterns

### Memory Safety Validation

The refactoring was validated against all memory safety requirements:

- ✅ **CGO Object Lifetime**: Maintains existing `runtime.SetFinalizer()` patterns
- ✅ **Buffer Pool Integration**: Doesn't interfere with buffer pooling mechanisms  
- ✅ **Reference Counting**: Preserves atomic operations for shared resources
- ✅ **Data Transfer Safety**: Maintains synchronous execution ensuring Go slice lifetime
- ✅ **Exception Handling**: All Objective-C boundary exception handling preserved

### Operations Standardized

The following MPS operations were updated to use proper finalizer-based cleanup:

**Binary Operations (using helper):**
- `AddMPS`, `SubMPS`, `MulMPS`, `DivMPS`

**Complex Operations (documented cleanup):**
- `MatMulMPS`, `ReLUMPS`, `ReLUGradientMPS`, `SigmoidMPS`
- `Conv2DMPS`, `MaxPool2DMPS`, `AvgPool2DMPS`
- `ReshapeMPS`, `SumMPS`

This standardization ensures that all MPS operations follow the documented CGO bridge architecture while providing better code organization and maintainability.

---

This architecture demonstrates how to build a robust, high-performance bridge between Go and Apple's Metal framework while maintaining memory safety and providing good performance characteristics for machine learning workloads.