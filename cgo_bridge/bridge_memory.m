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