#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
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

// MPSGraph types when compiling Objective-C
typedef MPSGraph* MPSGraphRef;
typedef MPSGraphTensor* MPSGraphTensorRef;
typedef MPSGraphExecutable* MPSGraphExecutableRef;
typedef MPSGraphExecutableExecutionDescriptor* MPSGraphExecutionDescriptorRef;
typedef MPSGraphCompilationDescriptor* MPSGraphCompilationDescriptorRef;
typedef MPSGraphDevice* MPSGraphDeviceRef;

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

// MPSGraph types for C compilation
typedef void* MPSGraphRef;
typedef void* MPSGraphTensorRef;
typedef void* MPSGraphExecutableRef;
typedef void* MPSGraphExecutionDescriptorRef;
typedef void* MPSGraphCompilationDescriptorRef;
typedef void* MPSGraphDeviceRef;

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

// MPSGraph function declarations
MPSGraphRef CreateMPSGraph(void);
MPSGraphDeviceRef CreateMPSGraphDevice(MTLDeviceRef metalDevice);
MPSGraphTensorRef MPSGraphPlaceholderTensor(MPSGraphRef graph, int* shape, size_t shapeCount, int dataType);
MPSGraphTensorRef MPSGraphConstantTensor(MPSGraphRef graph, double value, int* shape, size_t shapeCount, int dataType);

// MPSGraph operations
MPSGraphTensorRef MPSGraphAddition(MPSGraphRef graph, MPSGraphTensorRef primaryTensor, MPSGraphTensorRef secondaryTensor);
MPSGraphTensorRef MPSGraphSubtraction(MPSGraphRef graph, MPSGraphTensorRef primaryTensor, MPSGraphTensorRef secondaryTensor);
MPSGraphTensorRef MPSGraphMultiplication(MPSGraphRef graph, MPSGraphTensorRef primaryTensor, MPSGraphTensorRef secondaryTensor);
MPSGraphTensorRef MPSGraphDivision(MPSGraphRef graph, MPSGraphTensorRef primaryTensor, MPSGraphTensorRef secondaryTensor);
MPSGraphTensorRef MPSGraphMatrixMultiplication(MPSGraphRef graph, MPSGraphTensorRef primaryTensor, MPSGraphTensorRef secondaryTensor);
MPSGraphTensorRef MPSGraphReLU(MPSGraphRef graph, MPSGraphTensorRef tensor);
MPSGraphTensorRef MPSGraphSigmoid(MPSGraphRef graph, MPSGraphTensorRef tensor);
MPSGraphTensorRef MPSGraphSoftmax(MPSGraphRef graph, MPSGraphTensorRef tensor, size_t axis);
MPSGraphTensorRef MPSGraphTranspose(MPSGraphRef graph, MPSGraphTensorRef tensor, size_t dimension, size_t dimensionTwo);
MPSGraphTensorRef MPSGraphReshape(MPSGraphRef graph, MPSGraphTensorRef tensor, int* shape, size_t shapeCount);

// MPSGraph execution  
MPSGraphExecutableRef MPSGraphCompile(MPSGraphRef graph, MPSGraphDeviceRef device, MPSGraphTensorRef* inputTensors, size_t inputTensorsCount, MPSGraphTensorRef* targetTensors, size_t targetTensorsCount, MPSGraphCompilationDescriptorRef compilationDescriptor);
MPSGraphExecutionDescriptorRef CreateMPSGraphExecutionDescriptor(void);
MPSGraphCompilationDescriptorRef CreateMPSGraphCompilationDescriptor(void);
void MPSGraphExecuteExecutable(MPSGraphExecutableRef executable, MTLCommandQueueRef commandQueue, MPSGraphTensorRef* inputTensors, MTLBufferRef* inputBuffers, size_t inputCount, MPSGraphTensorRef* resultTensors, MTLBufferRef* resultBuffers, size_t resultCount, MPSGraphExecutionDescriptorRef executionDescriptor);

// Resource options constants
extern const size_t MTLResourceStorageModeShared_Const;
extern const size_t MTLResourceStorageModeManaged_Const;
extern const size_t MTLResourceStorageModePrivate_Const;

// MPSGraph data type constants
extern const int MPSDataTypeFloat32_Const;
extern const int MPSDataTypeFloat16_Const;
extern const int MPSDataTypeInt32_Const;

#endif // METAL_BRIDGE_H