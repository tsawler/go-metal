#import "metal_bridge.h"

// Resource options constants
const size_t MTLResourceStorageModeShared_Const = MTLResourceStorageModeShared;
const size_t MTLResourceStorageModeManaged_Const = MTLResourceStorageModeManaged;
const size_t MTLResourceStorageModePrivate_Const = MTLResourceStorageModePrivate;

// MPSGraph data type constants
const int MPSDataTypeFloat32_Const = MPSDataTypeFloat32;
const int MPSDataTypeFloat16_Const = MPSDataTypeFloat16;
const int MPSDataTypeInt32_Const = MPSDataTypeInt32;

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

// MPSGraph function implementations

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

MPSGraphDevice* CreateMPSGraphDevice(id<MTLDevice> metalDevice) {
    @try {
        if (metalDevice == nil) {
            NSLog(@"Metal device is nil in CreateMPSGraphDevice");
            return nil;
        }
        MPSGraphDevice* graphDevice = [MPSGraphDevice deviceWithMTLDevice:metalDevice];
        if (graphDevice == nil) {
            NSLog(@"Failed to create MPSGraphDevice");
            return nil;
        }
        return graphDevice;
    } @catch (NSException *exception) {
        NSLog(@"Exception creating MPSGraphDevice: %@", exception);
        return nil;
    }
}

MPSGraphTensor* MPSGraphPlaceholderTensor(MPSGraph* graph, int* shape, size_t shapeCount, int dataType) {
    NSMutableArray<NSNumber*>* nsShape = [[NSMutableArray alloc] initWithCapacity:shapeCount];
    for (size_t i = 0; i < shapeCount; i++) {
        [nsShape addObject:@(shape[i])];
    }
    
    return [graph placeholderWithShape:nsShape dataType:(MPSDataType)dataType name:nil];
}

MPSGraphTensor* MPSGraphConstantTensor(MPSGraph* graph, double value, int* shape, size_t shapeCount, int dataType) {
    NSMutableArray<NSNumber*>* nsShape = [[NSMutableArray alloc] initWithCapacity:shapeCount];
    for (size_t i = 0; i < shapeCount; i++) {
        [nsShape addObject:@(shape[i])];
    }
    
    return [graph constantWithScalar:value shape:nsShape dataType:(MPSDataType)dataType];
}

// MPSGraph operations
MPSGraphTensor* MPSGraphAddition(MPSGraph* graph, MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor) {
    return [graph additionWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:nil];
}

MPSGraphTensor* MPSGraphSubtraction(MPSGraph* graph, MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor) {
    return [graph subtractionWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:nil];
}

MPSGraphTensor* MPSGraphMultiplication(MPSGraph* graph, MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor) {
    return [graph multiplicationWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:nil];
}

MPSGraphTensor* MPSGraphDivision(MPSGraph* graph, MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor) {
    return [graph divisionWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:nil];
}

MPSGraphTensor* MPSGraphMatrixMultiplication(MPSGraph* graph, MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor) {
    return [graph matrixMultiplicationWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:nil];
}

MPSGraphTensor* MPSGraphReLU(MPSGraph* graph, MPSGraphTensor* tensor) {
    return [graph reLUWithTensor:tensor name:nil];
}

MPSGraphTensor* MPSGraphSigmoid(MPSGraph* graph, MPSGraphTensor* tensor) {
    return [graph sigmoidWithTensor:tensor name:nil];
}

MPSGraphTensor* MPSGraphSoftmax(MPSGraph* graph, MPSGraphTensor* tensor, size_t axis) {
    return [graph softMaxWithTensor:tensor axis:(NSInteger)axis name:nil];
}

MPSGraphTensor* MPSGraphTranspose(MPSGraph* graph, MPSGraphTensor* tensor, size_t dimension, size_t dimensionTwo) {
    return [graph transposeTensor:tensor dimension:(NSUInteger)dimension withDimension:(NSUInteger)dimensionTwo name:nil];
}

MPSGraphTensor* MPSGraphReshape(MPSGraph* graph, MPSGraphTensor* tensor, int* shape, size_t shapeCount) {
    NSMutableArray<NSNumber*>* nsShape = [[NSMutableArray alloc] initWithCapacity:shapeCount];
    for (size_t i = 0; i < shapeCount; i++) {
        [nsShape addObject:@(shape[i])];
    }
    
    return [graph reshapeTensor:tensor withShape:nsShape name:nil];
}

// MPSGraph Convolution and Pooling operations
MPSGraphTensor* MPSGraphConvolution2D(MPSGraph* graph, MPSGraphTensor* source, MPSGraphTensor* weights, MPSGraphTensor* bias, int strideInX, int strideInY, int dilationRateInX, int dilationRateInY, int paddingLeft, int paddingRight, int paddingTop, int paddingBottom, int groups) {
    MPSGraphConvolution2DOpDescriptor* descriptor = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    
    descriptor.strideInX = strideInX;
    descriptor.strideInY = strideInY;
    descriptor.dilationRateInX = dilationRateInX;
    descriptor.dilationRateInY = dilationRateInY;
    descriptor.paddingLeft = paddingLeft;
    descriptor.paddingRight = paddingRight;
    descriptor.paddingTop = paddingTop;
    descriptor.paddingBottom = paddingBottom;
    descriptor.groups = groups;
    descriptor.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    descriptor.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    
    if (bias != nil) {
        MPSGraphTensor* conv = [graph convolution2DWithSourceTensor:source
                                                     weightsTensor:weights
                                                        descriptor:descriptor
                                                              name:nil];
        return [graph additionWithPrimaryTensor:conv secondaryTensor:bias name:nil];
    } else {
        return [graph convolution2DWithSourceTensor:source
                                      weightsTensor:weights
                                         descriptor:descriptor
                                               name:nil];
    }
}

MPSGraphTensor* MPSGraphMaxPooling2D(MPSGraph* graph, MPSGraphTensor* source, int kernelWidth, int kernelHeight, int strideInX, int strideInY, int paddingLeft, int paddingRight, int paddingTop, int paddingBottom) {
    MPSGraphPooling2DOpDescriptor* descriptor = [[MPSGraphPooling2DOpDescriptor alloc] init];
    
    descriptor.kernelWidth = kernelWidth;
    descriptor.kernelHeight = kernelHeight;
    descriptor.strideInX = strideInX;
    descriptor.strideInY = strideInY;
    descriptor.paddingLeft = paddingLeft;
    descriptor.paddingRight = paddingRight;
    descriptor.paddingTop = paddingTop;
    descriptor.paddingBottom = paddingBottom;
    descriptor.dilationRateInX = 1; // Must be positive
    descriptor.dilationRateInY = 1; // Must be positive
    descriptor.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    
    return [graph maxPooling2DWithSourceTensor:source descriptor:descriptor name:nil];
}

MPSGraphTensor* MPSGraphAvgPooling2D(MPSGraph* graph, MPSGraphTensor* source, int kernelWidth, int kernelHeight, int strideInX, int strideInY, int paddingLeft, int paddingRight, int paddingTop, int paddingBottom) {
    MPSGraphPooling2DOpDescriptor* descriptor = [[MPSGraphPooling2DOpDescriptor alloc] init];
    
    descriptor.kernelWidth = kernelWidth;
    descriptor.kernelHeight = kernelHeight;
    descriptor.strideInX = strideInX;
    descriptor.strideInY = strideInY;
    descriptor.paddingLeft = paddingLeft;
    descriptor.paddingRight = paddingRight;
    descriptor.paddingTop = paddingTop;
    descriptor.paddingBottom = paddingBottom;
    descriptor.dilationRateInX = 1; // Must be positive
    descriptor.dilationRateInY = 1; // Must be positive
    descriptor.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    
    return [graph avgPooling2DWithSourceTensor:source descriptor:descriptor name:nil];
}

// MPSGraph execution
MPSGraphExecutable* MPSGraphCompile(MPSGraph* graph, MPSGraphDevice* device, MPSGraphTensor** inputTensors, size_t inputTensorsCount, MPSGraphTensor** targetTensors, size_t targetTensorsCount, MPSGraphCompilationDescriptor* compilationDescriptor) {
    NSMutableArray<MPSGraphTensor*>* targetTensorsArray = [[NSMutableArray alloc] init];
    NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds = [[NSMutableDictionary alloc] init];
    
    // Convert target tensors C array to NSArray
    for (size_t i = 0; i < targetTensorsCount; i++) {
        if (targetTensors[i] != nil) {
            [targetTensorsArray addObject:targetTensors[i]];
        }
    }
    
    // Create feeds dictionary for input placeholders
    for (size_t i = 0; i < inputTensorsCount; i++) {
        if (inputTensors[i] != nil) {
            // Create MPSGraphShapedType for the placeholder
            MPSGraphShapedType* shapedType = [[MPSGraphShapedType alloc] initWithShape:inputTensors[i].shape 
                                                                              dataType:inputTensors[i].dataType];
            feeds[inputTensors[i]] = shapedType;
        }
    }
    
    return [graph compileWithDevice:device
                              feeds:feeds
                      targetTensors:targetTensorsArray
                   targetOperations:nil
              compilationDescriptor:compilationDescriptor];
}

MPSGraphExecutableExecutionDescriptor* CreateMPSGraphExecutionDescriptor() {
    return [[MPSGraphExecutableExecutionDescriptor alloc] init];
}

MPSGraphCompilationDescriptor* CreateMPSGraphCompilationDescriptor() {
    return [[MPSGraphCompilationDescriptor alloc] init];
}

void MPSGraphExecuteExecutable(MPSGraphExecutable* executable, id<MTLCommandQueue> commandQueue, MPSGraphTensor** inputTensors, id<MTLBuffer>* inputBuffers, size_t inputCount, MPSGraphTensor** resultTensors, id<MTLBuffer>* resultBuffers, size_t resultCount, MPSGraphExecutableExecutionDescriptor* executionDescriptor) {
    NSMutableArray<MPSGraphTensorData*>* inputsArray = [[NSMutableArray alloc] init];
    NSMutableArray<MPSGraphTensorData*>* resultsArray = [[NSMutableArray alloc] init];
    
    // Setup input feeds
    for (size_t i = 0; i < inputCount; i++) {
        if (inputTensors[i] != nil && inputBuffers[i] != nil) {
            MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffers[i] 
                                                                                     shape:inputTensors[i].shape 
                                                                                  dataType:inputTensors[i].dataType];
            [inputsArray addObject:tensorData];
        }
    }
    
    // Setup result buffers  
    for (size_t i = 0; i < resultCount; i++) {
        if (resultTensors[i] != nil && resultBuffers[i] != nil) {
            MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:resultBuffers[i] 
                                                                                     shape:resultTensors[i].shape 
                                                                                  dataType:resultTensors[i].dataType];
            [resultsArray addObject:tensorData];
        }
    }
    
    // Execute the graph
    [executable runWithMTLCommandQueue:commandQueue
                           inputsArray:inputsArray
                          resultsArray:resultsArray.count > 0 ? resultsArray : nil
                   executionDescriptor:executionDescriptor];
}