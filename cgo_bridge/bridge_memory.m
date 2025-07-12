#import "bridge_memory.h"

// Allocate Metal buffer
uintptr_t allocate_metal_buffer(uintptr_t device_ptr, int size, int device_type) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        if (!device || size <= 0) {
            return 0;
        }
        
        MTLResourceOptions options;
        switch (device_type) {
            case 0: // CPU
                options = MTLResourceStorageModeShared;
                break;
            case 1: // GPU - Try managed mode for MPSGraph compatibility
                options = MTLResourceStorageModeManaged;
                break;
            case 2: // PersistentGPU
                options = MTLResourceStorageModeManaged;
                break;
            default:
                options = MTLResourceStorageModeManaged;
                break;
        }
        
        // For MPSGraph compatibility, ensure 16-byte alignment
        int alignedSize = ((size + 15) / 16) * 16;  // Round up to 16-byte boundary
        
        id<MTLBuffer> buffer = [device newBufferWithLength:alignedSize options:options];
        if (!buffer) {
            NSLog(@"Failed to allocate Metal buffer of size %d (aligned: %d)", size, alignedSize);
            return 0;
        }
        
        // NSLog(@"Allocated Metal buffer: requested=%d, aligned=%d, actual=%lu", 
        //       size, alignedSize, buffer.length);
        
        // Return buffer pointer (ARC will manage lifetime)
        return (uintptr_t)(__bridge_retained void*)buffer;
    }
}

// Deallocate Metal buffer
void deallocate_metal_buffer(uintptr_t buffer_ptr) {
    @autoreleasepool {
        if (buffer_ptr == 0) {
            return;
        }
        
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)(void*)buffer_ptr;
        // Buffer will be automatically released when it goes out of scope
        (void)buffer; // Suppress unused variable warning
    }
}

// Zero a Metal buffer using MPSGraph for GPU-only buffers
int zero_metal_buffer_mpsgraph(uintptr_t device_ptr, uintptr_t buffer_ptr, int size) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)buffer_ptr;
        
        if (!device || !buffer) {
            NSLog(@"Device or buffer is nil in zero_metal_buffer_mpsgraph");
            return -1;
        }
        
        if (size <= 0) {
            NSLog(@"Invalid buffer size: %d (must be positive)", size);
            return -2;
        }
        
        @try {
            // For MPSGraph buffer zeroing, we'll use a simple assignment approach
            // Create MPSGraph
            MPSGraph* graph = [[MPSGraph alloc] init];
            if (!graph) {
                NSLog(@"Failed to create MPSGraph for buffer zeroing");
                return -3;
            }
            
            // Create command queue
            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                NSLog(@"Failed to create command queue for buffer zeroing");
                return -4;
            }
            
            // Calculate number of elements (treat as float32)
            int num_elements = size / sizeof(float);
            if (size % sizeof(float) != 0) {
                num_elements = (size + sizeof(float) - 1) / sizeof(float);
            }
            
            NSArray<NSNumber*>* shape = @[@(num_elements)];
            
            // Create a zero-filled array
            float* zeroArray = (float*)calloc(num_elements, sizeof(float));
            if (!zeroArray) {
                NSLog(@"Failed to allocate zero array");
                return -5;
            }
            
            // Create a constant tensor from the zero array
            NSData* zeroData = [NSData dataWithBytesNoCopy:zeroArray
                                                     length:num_elements * sizeof(float)
                                               freeWhenDone:YES];
            
            MPSGraphTensor* zeroTensor = [graph constantWithData:zeroData
                                                            shape:shape
                                                         dataType:MPSDataTypeFloat32];
            
            // Create tensor data for our buffer
            MPSGraphTensorData* bufferData = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                                      shape:shape
                                                                                   dataType:MPSDataTypeFloat32];
            
            // Execute graph to write zeros to buffer
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = 
                [graph runWithMTLCommandQueue:commandQueue
                                        feeds:@{}
                               targetTensors:@[zeroTensor]
                            targetOperations:nil];
            
            MPSGraphTensorData* zeroResult = results[zeroTensor];
            if (!zeroResult) {
                NSLog(@"Failed to get zero tensor result");
                return -6;
            }
            
            // Use a blit encoder to copy the zeros to our buffer
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            
            // For simple zeroing, we can also use fillBuffer if available
            if ([buffer respondsToSelector:@selector(contents)] && [buffer contents] != nil) {
                // CPU-accessible buffer - use memset
                memset([buffer contents], 0, size);
            } else {
                // GPU-only buffer - use blit encoder
                id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
                [blitEncoder fillBuffer:buffer range:NSMakeRange(0, size) value:0];
                [blitEncoder endEncoding];
            }
            
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            if (commandBuffer.error) {
                NSLog(@"Command buffer error during buffer zeroing: %@", commandBuffer.error.localizedDescription);
                return -7;
            }
            
            return 0; // Success
            
        } @catch (NSException* exception) {
            NSLog(@"Zero buffer MPSGraph exception: %@", exception.reason);
            return -8;
        }
    }
}

// Legacy CPU-based buffer zeroing (keeping for CPU-accessible buffers)
int zero_metal_buffer(uintptr_t device_ptr, uintptr_t buffer_ptr, int size) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)buffer_ptr;
        
        if (!device || !buffer) {
            NSLog(@"Device or buffer is nil in zero_metal_buffer");
            return -1;
        }
        
        @try {
            // Try CPU-based zeroing first (fastest for CPU-accessible buffers)
            void* contents = [buffer contents];
            if (contents) {
                memset(contents, 0, size);
                return 0;
            } else {
                // Buffer is not CPU-accessible, use MPSGraph implementation
                return zero_metal_buffer_mpsgraph(device_ptr, buffer_ptr, size);
            }
        } @catch (NSException* exception) {
            NSLog(@"Zero buffer exception: %@", exception.reason);
            return -5;
        }
    }
}

// Copy data to Metal buffer
int copy_data_to_metal_buffer(uintptr_t buffer_ptr, void* data, int size) {
    @autoreleasepool {
        if (buffer_ptr == 0 || data == NULL || size <= 0) {
            NSLog(@"Invalid parameters for copy_data_to_metal_buffer");
            return -1;
        }
        
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)buffer_ptr;
        if (!buffer) {
            NSLog(@"Buffer is nil in copy_data_to_metal_buffer");
            return -2;
        }
        
        @try {
            void* contents = [buffer contents];
            if (contents) {
                memcpy(contents, data, size);
                return 0;
            } else {
                NSLog(@"Buffer contents not accessible for data copy");
                return -3;
            }
        } @catch (NSException* exception) {
            NSLog(@"Exception in copy_data_to_metal_buffer: %@", exception.reason);
            return -4;
        }
    }
}

// Copy float32 array to Metal buffer
int copy_float32_array_to_metal_buffer(uintptr_t buffer_ptr, float* data, int num_elements) {
    @autoreleasepool {
        if (buffer_ptr == 0 || data == NULL || num_elements <= 0) {
            NSLog(@"Invalid parameters for copy_float32_array_to_metal_buffer");
            return -1;
        }
        
        int size_bytes = num_elements * sizeof(float);
        int result = copy_data_to_metal_buffer(buffer_ptr, (void*)data, size_bytes);
        
        if (result == 0) {
            // NSLog(@"Successfully copied %d float32 elements (%d bytes) to Metal buffer", 
            //       num_elements, size_bytes);
        }
        
        return result;
    }
}

// Copy int32 array to Metal buffer
int copy_int32_array_to_metal_buffer(uintptr_t buffer_ptr, int* data, int num_elements) {
    @autoreleasepool {
        if (buffer_ptr == 0 || data == NULL || num_elements <= 0) {
            NSLog(@"Invalid parameters for copy_int32_array_to_metal_buffer");
            return -1;
        }
        
        int size_bytes = num_elements * sizeof(int);
        int result = copy_data_to_metal_buffer(buffer_ptr, (void*)data, size_bytes);
        
        if (result == 0) {
            // NSLog(@"Successfully copied %d int32 elements (%d bytes) to Metal buffer", 
            //       num_elements, size_bytes);
        }
        
        return result;
    }
}

// Copy Metal buffer to float32 array
int copy_metal_buffer_to_float32_array(uintptr_t buffer_ptr, float* data, int num_elements) {
    @autoreleasepool {
        if (buffer_ptr == 0 || data == NULL || num_elements <= 0) {
            NSLog(@"Invalid parameters for copy_metal_buffer_to_float32_array");
            return -1;
        }
        
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)buffer_ptr;
        if (!buffer) {
            NSLog(@"Buffer is nil in copy_metal_buffer_to_float32_array");
            return -2;
        }
        
        @try {
            void* contents = [buffer contents];
            if (contents) {
                int size_bytes = num_elements * sizeof(float);
                memcpy(data, contents, size_bytes);
                return 0;
            } else {
                NSLog(@"Buffer contents not accessible for reading in copy_metal_buffer_to_float32_array");
                return -3;
            }
        } @catch (NSException* exception) {
            NSLog(@"Exception in copy_metal_buffer_to_float32_array: %@", exception.reason);
            return -4;
        }
    }
}

// Copy Metal buffer to int32 array
int copy_metal_buffer_to_int32_array(uintptr_t buffer_ptr, int* data, int num_elements) {
    @autoreleasepool {
        if (buffer_ptr == 0 || data == NULL || num_elements <= 0) {
            NSLog(@"Invalid parameters for copy_metal_buffer_to_int32_array");
            return -1;
        }
        
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)buffer_ptr;
        if (!buffer) {
            NSLog(@"Buffer is nil in copy_metal_buffer_to_int32_array");
            return -2;
        }
        
        @try {
            void* contents = [buffer contents];
            if (contents) {
                int size_bytes = num_elements * sizeof(int);
                memcpy(data, contents, size_bytes);
                return 0;
            } else {
                NSLog(@"Buffer contents not accessible for reading in copy_metal_buffer_to_int32_array");
                return -3;
            }
        } @catch (NSException* exception) {
            NSLog(@"Exception in copy_metal_buffer_to_int32_array: %@", exception.reason);
            return -4;
        }
    }
}

// Convert tensor type on GPU using Metal compute shader
int convert_tensor_type(uintptr_t src_buffer_ptr, uintptr_t dst_buffer_ptr, 
                       int* shape, int num_dims, int src_type, int dst_type,
                       uintptr_t device_ptr) {
    @autoreleasepool {
        if (src_buffer_ptr == 0 || dst_buffer_ptr == 0 || shape == NULL || num_dims <= 0) {
            NSLog(@"Invalid parameters for convert_tensor_type");
            return -1;
        }
        
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)(void*)src_buffer_ptr;
        id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)(void*)dst_buffer_ptr;
        
        if (!device || !srcBuffer || !dstBuffer) {
            NSLog(@"Invalid device or buffers in convert_tensor_type");
            return -2;
        }
        
        // Calculate total number of elements
        int totalElements = 1;
        for (int i = 0; i < num_dims; i++) {
            totalElements *= shape[i];
        }
        
        // Create command queue and buffer
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Currently only support Float32 <-> Float16 conversion
        // Use Metal compute shader for type conversion
        if ((src_type == 0 && dst_type == 2) || (src_type == 2 && dst_type == 0)) {
            // Float32 <-> Float16 conversion using compute shader
            
            // Create compute encoder
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            // Create a simple Metal library with conversion kernels
            NSString* kernelSource;
            if (src_type == 0) { // Float32 to Float16
                kernelSource = @"#include <metal_stdlib>\n"
                              @"using namespace metal;\n"
                              @"kernel void convert_fp32_to_fp16(device const float* src [[buffer(0)]],\n"
                              @"                                 device half* dst [[buffer(1)]],\n"
                              @"                                 uint index [[thread_position_in_grid]]) {\n"
                              @"    dst[index] = half(src[index]);\n"
                              @"}\n";
            } else { // Float16 to Float32
                kernelSource = @"#include <metal_stdlib>\n"
                              @"using namespace metal;\n"
                              @"kernel void convert_fp16_to_fp32(device const half* src [[buffer(0)]],\n"
                              @"                                 device float* dst [[buffer(1)]],\n"
                              @"                                 uint index [[thread_position_in_grid]]) {\n"
                              @"    dst[index] = float(src[index]);\n"
                              @"}\n";
            }
            
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
            if (!library) {
                NSLog(@"Failed to create Metal library: %@", error.localizedDescription);
                return -4;
            }
            
            NSString* functionName = (src_type == 0) ? @"convert_fp32_to_fp16" : @"convert_fp16_to_fp32";
            id<MTLFunction> function = [library newFunctionWithName:functionName];
            if (!function) {
                NSLog(@"Failed to find kernel function: %@", functionName);
                return -5;
            }
            
            id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipelineState) {
                NSLog(@"Failed to create compute pipeline: %@", error.localizedDescription);
                return -6;
            }
            
            // Set up compute command encoder
            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:srcBuffer offset:0 atIndex:0];
            [encoder setBuffer:dstBuffer offset:0 atIndex:1];
            
            // Calculate thread group sizes
            NSUInteger threadsPerThreadgroup = MIN(pipelineState.maxTotalThreadsPerThreadgroup, 256);
            NSUInteger threadgroups = (totalElements + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
            
            MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
            MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        } else {
            NSLog(@"Unsupported type conversion: %d to %d", src_type, dst_type);
            return -3;
        }
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return 0;
    }
}

// MEMORY TRANSFER OPTIMIZATION: Direct Metal buffer operations for staging pool
// These functions provide efficient GPU-GPU and staging buffer operations

// Direct buffer-to-buffer copy using Metal blit command encoder
int copy_buffer_to_buffer_async(uintptr_t src_buffer_ptr, uintptr_t dst_buffer_ptr, 
                                int src_offset, int dst_offset, int size,
                                uintptr_t command_queue_ptr) {
    @autoreleasepool {
        if (src_buffer_ptr == 0 || dst_buffer_ptr == 0 || command_queue_ptr == 0) {
            NSLog(@"Invalid parameters for copy_buffer_to_buffer_async");
            return -1;
        }
        
        if (size <= 0 || src_offset < 0 || dst_offset < 0) {
            NSLog(@"Invalid size or offset parameters: size=%d, src_offset=%d, dst_offset=%d", 
                  size, src_offset, dst_offset);
            return -2;
        }
        
        id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)(void*)src_buffer_ptr;
        id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)(void*)dst_buffer_ptr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)(void*)command_queue_ptr;
        
        if (!srcBuffer || !dstBuffer || !commandQueue) {
            NSLog(@"Nil buffers or command queue in copy_buffer_to_buffer_async");
            return -3;
        }
        
        // Validate offsets and size don't exceed buffer bounds
        if ((NSUInteger)(src_offset + size) > srcBuffer.length) {
            NSLog(@"Source copy would exceed buffer bounds: offset=%d + size=%d > length=%lu", 
                  src_offset, size, (unsigned long)srcBuffer.length);
            return -4;
        }
        
        if ((NSUInteger)(dst_offset + size) > dstBuffer.length) {
            NSLog(@"Destination copy would exceed buffer bounds: offset=%d + size=%d > length=%lu", 
                  dst_offset, size, (unsigned long)dstBuffer.length);
            return -5;
        }
        
        @try {
            // Create command buffer from the queue
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            if (!commandBuffer) {
                NSLog(@"Failed to create command buffer");
                return -6;
            }
            
            // Create blit command encoder for efficient buffer copy
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            if (!blitEncoder) {
                NSLog(@"Failed to create blit command encoder");
                return -7;
            }
            
            // Perform the buffer copy operation
            [blitEncoder copyFromBuffer:srcBuffer
                           sourceOffset:src_offset
                               toBuffer:dstBuffer
                      destinationOffset:dst_offset
                                   size:size];
            
            [blitEncoder endEncoding];
            
            // Commit the command buffer for async execution
            [commandBuffer commit];
            
            return 0; // Success - operation is async
            
        } @catch (NSException* exception) {
            NSLog(@"Exception in copy_buffer_to_buffer_async: %@", exception.reason);
            return -8;
        }
    }
}

// Copy data from CPU to staging buffer (CPU-accessible)
int copy_data_to_staging_buffer(uintptr_t staging_buffer_ptr, void* data, int size) {
    @autoreleasepool {
        if (staging_buffer_ptr == 0 || data == NULL || size <= 0) {
            NSLog(@"Invalid parameters for copy_data_to_staging_buffer");
            return -1;
        }
        
        id<MTLBuffer> stagingBuffer = (__bridge id<MTLBuffer>)(void*)staging_buffer_ptr;
        if (!stagingBuffer) {
            NSLog(@"Staging buffer is nil in copy_data_to_staging_buffer");
            return -2;
        }
        
        // Validate size doesn't exceed buffer capacity
        if ((NSUInteger)size > stagingBuffer.length) {
            NSLog(@"Data size %d exceeds staging buffer capacity %lu", 
                  size, (unsigned long)stagingBuffer.length);
            return -3;
        }
        
        @try {
            // Staging buffers should be CPU-accessible (shared storage mode)
            void* contents = [stagingBuffer contents];
            if (!contents) {
                NSLog(@"Staging buffer is not CPU-accessible");
                return -4;
            }
            
            // Direct memory copy - fastest for CPU-accessible buffers
            memcpy(contents, data, size);
            
            // For managed buffers, synchronize with GPU
            if (stagingBuffer.storageMode == MTLStorageModeManaged) {
                [stagingBuffer didModifyRange:NSMakeRange(0, size)];
            }
            
            return 0; // Success
            
        } @catch (NSException* exception) {
            NSLog(@"Exception in copy_data_to_staging_buffer: %@", exception.reason);
            return -5;
        }
    }
}

// Copy from staging buffer to GPU buffer asynchronously
int copy_staging_to_gpu_buffer_async(uintptr_t staging_buffer_ptr, uintptr_t gpu_buffer_ptr,
                                     int staging_offset, int gpu_offset, int size,
                                     uintptr_t command_queue_ptr) {
    @autoreleasepool {
        if (staging_buffer_ptr == 0 || gpu_buffer_ptr == 0 || command_queue_ptr == 0) {
            NSLog(@"Invalid parameters for copy_staging_to_gpu_buffer_async");
            return -1;
        }
        
        if (size <= 0 || staging_offset < 0 || gpu_offset < 0) {
            NSLog(@"Invalid size or offset parameters: size=%d, staging_offset=%d, gpu_offset=%d", 
                  size, staging_offset, gpu_offset);
            return -2;
        }
        
        id<MTLBuffer> stagingBuffer = (__bridge id<MTLBuffer>)(void*)staging_buffer_ptr;
        id<MTLBuffer> gpuBuffer = (__bridge id<MTLBuffer>)(void*)gpu_buffer_ptr;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)(void*)command_queue_ptr;
        
        if (!stagingBuffer || !gpuBuffer || !commandQueue) {
            NSLog(@"Nil buffers or command queue in copy_staging_to_gpu_buffer_async");
            return -3;
        }
        
        // Validate buffer bounds
        if ((NSUInteger)(staging_offset + size) > stagingBuffer.length) {
            NSLog(@"Staging copy would exceed buffer bounds: offset=%d + size=%d > length=%lu", 
                  staging_offset, size, (unsigned long)stagingBuffer.length);
            return -4;
        }
        
        if ((NSUInteger)(gpu_offset + size) > gpuBuffer.length) {
            NSLog(@"GPU copy would exceed buffer bounds: offset=%d + size=%d > length=%lu", 
                  gpu_offset, size, (unsigned long)gpuBuffer.length);
            return -5;
        }
        
        @try {
            // Create command buffer
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            if (!commandBuffer) {
                NSLog(@"Failed to create command buffer for staging to GPU copy");
                return -6;
            }
            
            // Use blit encoder for optimal GPU transfer
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            if (!blitEncoder) {
                NSLog(@"Failed to create blit encoder for staging to GPU copy");
                return -7;
            }
            
            // Copy from staging (CPU-accessible) to GPU buffer
            [blitEncoder copyFromBuffer:stagingBuffer
                           sourceOffset:staging_offset
                               toBuffer:gpuBuffer
                      destinationOffset:gpu_offset
                                   size:size];
            
            [blitEncoder endEncoding];
            
            // Commit for async execution
            [commandBuffer commit];
            
            return 0; // Success - operation is async
            
        } @catch (NSException* exception) {
            NSLog(@"Exception in copy_staging_to_gpu_buffer_async: %@", exception.reason);
            return -8;
        }
    }
}

// Wait for buffer copy operations to complete
int wait_for_buffer_copy_completion(uintptr_t command_queue_ptr) {
    @autoreleasepool {
        if (command_queue_ptr == 0) {
            NSLog(@"Invalid command queue for wait_for_buffer_copy_completion");
            return -1;
        }
        
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)(void*)command_queue_ptr;
        if (!commandQueue) {
            NSLog(@"Command queue is nil in wait_for_buffer_copy_completion");
            return -2;
        }
        
        @try {
            // Create a synchronization command buffer
            id<MTLCommandBuffer> syncBuffer = [commandQueue commandBuffer];
            if (!syncBuffer) {
                NSLog(@"Failed to create sync command buffer");
                return -3;
            }
            
            // Commit and wait for completion
            [syncBuffer commit];
            [syncBuffer waitUntilCompleted];
            
            if (syncBuffer.error) {
                NSLog(@"Command buffer error during synchronization: %@", 
                      syncBuffer.error.localizedDescription);
                return -4;
            }
            
            return 0; // Success
            
        } @catch (NSException* exception) {
            NSLog(@"Exception in wait_for_buffer_copy_completion: %@", exception.reason);
            return -5;
        }
    }
}