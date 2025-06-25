#import "metal_bridge.h"

// Resource options constants
const size_t MTLResourceStorageModeShared_Const = MTLResourceStorageModeShared;
const size_t MTLResourceStorageModeManaged_Const = MTLResourceStorageModeManaged;
const size_t MTLResourceStorageModePrivate_Const = MTLResourceStorageModePrivate;

id<MTLDevice> CreateSystemDefaultDevice() {
    return MTLCreateSystemDefaultDevice();
}

id<MTLCommandQueue> CreateCommandQueue(id<MTLDevice> device) {
    return [device newCommandQueue];
}

id<MTLBuffer> CreateBufferWithBytes(id<MTLDevice> device, const void* data, size_t length, size_t resourceOptions) {
    return [device newBufferWithBytes:data length:length options:resourceOptions];
}

id<MTLBuffer> CreateBufferWithLength(id<MTLDevice> device, size_t length, size_t resourceOptions) {
    return [device newBufferWithLength:length options:resourceOptions];
}

void* GetBufferContents(id<MTLBuffer> buffer) {
    return [buffer contents];
}

size_t GetBufferLength(id<MTLBuffer> buffer) {
    return [buffer length];
}

id<MTLLibrary> CreateLibraryWithSource(id<MTLDevice> device, const char* source) {
    NSString *sourceString = [NSString stringWithUTF8String:source];
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:sourceString options:nil error:&error];
    if (error) {
        NSLog(@"Error creating library: %@", error.localizedDescription);
        return nil;
    }
    return library;
}

id<MTLFunction> GetFunction(id<MTLLibrary> library, const char* functionName) {
    NSString *functionNameString = [NSString stringWithUTF8String:functionName];
    return [library newFunctionWithName:functionNameString];
}

id<MTLComputePipelineState> CreateComputePipelineStateWithFunction(id<MTLDevice> device, id<MTLFunction> function) {
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        NSLog(@"Error creating compute pipeline state: %@", error.localizedDescription);
        return nil;
    }
    return pipelineState;
}

id<MTLCommandBuffer> CreateCommandBuffer(id<MTLCommandQueue> queue) {
    return [queue commandBuffer];
}

id<MTLComputeCommandEncoder> CreateComputeCommandEncoder(id<MTLCommandBuffer> commandBuffer) {
    return [commandBuffer computeCommandEncoder];
}

void SetComputePipelineState(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipelineState) {
    [encoder setComputePipelineState:pipelineState];
}

void SetBuffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer, size_t offset, size_t index) {
    [encoder setBuffer:buffer offset:offset atIndex:index];
}

void DispatchThreads(id<MTLComputeCommandEncoder> encoder, size_t gridWidth, size_t gridHeight, size_t gridDepth, size_t threadgroupWidth, size_t threadgroupHeight, size_t threadgroupDepth) {
    MTLSize gridSize = MTLSizeMake(gridWidth, gridHeight, gridDepth);
    MTLSize threadgroupSize = MTLSizeMake(threadgroupWidth, threadgroupHeight, threadgroupDepth);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void EndEncoding(id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding];
}

void CommitCommandBuffer(id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer commit];
}

void WaitUntilCommandBufferCompleted(id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer waitUntilCompleted];
}

void AddCommandBufferCompletedHandler(id<MTLCommandBuffer> commandBuffer, void* userData, CompletionHandlerFunc handler) {
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buffer) {
        // Call the C function pointer with user data and status code
        // Note: MTLCommandBufferStatus is an enum, cast to long for C compatibility
        handler(userData, (long)[buffer status]);
    }];
}

void ReleaseMetalObject(void* obj) {
    // In an ARC environment, calling `release` directly on an `id` might not be correct
    // if the object is still managed by ARC elsewhere. For objects returned from C++
    // to C (and then to Go), if they are `__bridge_retained`, they need `CFRelease`.
    // For simplicity and safety with cgo, we usually return `void*` and let Go
    // manage the lifetime or ensure the Objective-C side uses `__autoreleasing` or similar
    // where explicit release isn't needed. However, when you 'own' the reference from Go,
    // you must release it. `CFRelease` is the safest way to release `id` from C.
    if (obj) {
        CFRelease((__bridge CFTypeRef)obj); // Requires CoreFoundation
    }
}