#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

// Use actual Objective-C types when compiling Objective-C
typedef id<MTLDevice> MTLDeviceRef;
typedef id<MTLCommandQueue> MTLCommandQueueRef;
typedef id<MTLBuffer> MTLBufferRef;
typedef id<MTLLibrary> MTLLibraryRef;
typedef id<MTLFunction> MTLFunctionRef;
typedef id<MTLComputePipelineState> MTLComputePipelineStateRef;
typedef id<MTLCommandBuffer> MTLCommandBufferRef;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoderRef;

#else
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Use void* to represent Objective-C objects in C
typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLBufferRef;
typedef void* MTLLibraryRef;
typedef void* MTLFunctionRef;
typedef void* MTLComputePipelineStateRef;
typedef void* MTLCommandBufferRef;
typedef void* MTLComputeCommandEncoderRef;

// C-compatible types for non-Objective-C compilation
typedef void* id;
typedef void* CFTypeRef;
typedef unsigned long NSUInteger;

// MTLSize struct in C
typedef struct {
    size_t width;
    size_t height;
    size_t depth;
} MTLSize;

// CoreFoundation functions
CFTypeRef CFRetain(CFTypeRef cf);
void CFRelease(CFTypeRef cf);
#endif

// Function declarations using C types
MTLDeviceRef CreateSystemDefaultDevice(void);
MTLCommandQueueRef CreateCommandQueue(MTLDeviceRef device);
MTLBufferRef CreateBufferWithBytes(MTLDeviceRef device, const void* data, size_t length, size_t resourceOptions);
MTLBufferRef CreateBufferWithLength(MTLDeviceRef device, size_t length, size_t resourceOptions);
void* GetBufferContents(MTLBufferRef buffer);
size_t GetBufferLength(MTLBufferRef buffer);
MTLLibraryRef CreateLibraryWithSource(MTLDeviceRef device, const char* source);
MTLFunctionRef GetFunction(MTLLibraryRef library, const char* functionName);
MTLComputePipelineStateRef CreateComputePipelineStateWithFunction(MTLDeviceRef device, MTLFunctionRef function);
MTLCommandBufferRef CreateCommandBuffer(MTLCommandQueueRef queue);
MTLComputeCommandEncoderRef CreateComputeCommandEncoder(MTLCommandBufferRef commandBuffer);
void SetComputePipelineState(MTLComputeCommandEncoderRef encoder, MTLComputePipelineStateRef pipelineState);
void SetBuffer(MTLComputeCommandEncoderRef encoder, MTLBufferRef buffer, size_t offset, size_t index);
void DispatchThreads(MTLComputeCommandEncoderRef encoder, size_t gridWidth, size_t gridHeight, size_t gridDepth, size_t threadgroupWidth, size_t threadgroupHeight, size_t threadgroupDepth);
void EndEncoding(MTLComputeCommandEncoderRef encoder);
void CommitCommandBuffer(MTLCommandBufferRef commandBuffer);
void WaitUntilCommandBufferCompleted(MTLCommandBufferRef commandBuffer);

// For asynchronous completion handling
typedef void (*CompletionHandlerFunc)(void* userData, long statusCode);
void AddCommandBufferCompletedHandler(MTLCommandBufferRef commandBuffer, void* userData, CompletionHandlerFunc handler);

// Function to release Metal objects
void ReleaseMetalObject(void* obj);

// Resource options constants
extern const size_t MTLResourceStorageModeShared_Const;
extern const size_t MTLResourceStorageModeManaged_Const;
extern const size_t MTLResourceStorageModePrivate_Const;

#endif // METAL_BRIDGE_H