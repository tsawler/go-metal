#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#import "bridge_types.h"
#import "bridge_graph.h"
#import "bridge_optimizer.h"

// MEMORY LEAK FIX: Helper function to update cached buffers if dimensions change
static void updateCachedBuffersIfNeeded(training_engine_t* engine, 
                                       int batchSize, 
                                       int inputChannels, 
                                       int outputChannels,
                                       int imageWidth,
                                       int imageHeight) {
    // Check if dimensions have changed
    BOOL dimensionsChanged = (engine->cachedBatchSize != batchSize ||
                             engine->cachedInputChannels != inputChannels ||
                             engine->cachedOutputChannels != outputChannels ||
                             engine->cachedImageWidth != imageWidth ||
                             engine->cachedImageHeight != imageHeight);
    
    if (dimensionsChanged) {
        // Release old buffers
        engine->cachedInputImage = nil;
        engine->cachedConvOutputImage = nil;
        engine->cachedConvOutputBuffer = nil;
        
        // Update cached dimensions
        engine->cachedBatchSize = batchSize;
        engine->cachedInputChannels = inputChannels;
        engine->cachedOutputChannels = outputChannels;
        engine->cachedImageWidth = imageWidth;
        engine->cachedImageHeight = imageHeight;
        
        // NSLog(@"Buffer dimensions changed to: batch=%d, in_ch=%d, out_ch=%d, w=%d, h=%d",
        //       batchSize, inputChannels, outputChannels, imageWidth, imageHeight);
    }
}

// MEMORY LEAK FIX: Helper function to get or create cached MPSImages
static void getCachedMPSImages(training_engine_t* engine, 
                              MPSImage** inputImage, 
                              MPSImage** convOutputImage) {
    // Create input image if not cached
    if (engine->cachedInputImage == nil) {
        MPSImageDescriptor* inputDesc = [MPSImageDescriptor 
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
            width:engine->cachedImageWidth
            height:engine->cachedImageHeight
            featureChannels:engine->cachedInputChannels
            numberOfImages:engine->cachedBatchSize
            usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
        
        engine->cachedInputImage = [[MPSImage alloc] initWithDevice:engine->device imageDescriptor:inputDesc];
    }
    *inputImage = engine->cachedInputImage;
    
    // Create output image if not cached
    if (engine->cachedConvOutputImage == nil) {
        MPSImageDescriptor* convOutputDesc = [MPSImageDescriptor 
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
            width:engine->cachedImageWidth
            height:engine->cachedImageHeight
            featureChannels:engine->cachedOutputChannels
            numberOfImages:engine->cachedBatchSize
            usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
        
        engine->cachedConvOutputImage = [[MPSImage alloc] initWithDevice:engine->device imageDescriptor:convOutputDesc];
    }
    *convOutputImage = engine->cachedConvOutputImage;
}

// MEMORY LEAK FIX: Helper function to get or create cached buffer
static id<MTLBuffer> getCachedConvOutputBuffer(training_engine_t* engine) {
    if (engine->cachedConvOutputBuffer == nil) {
        size_t bufferSize = engine->cachedBatchSize * engine->cachedOutputChannels * 
                           engine->cachedImageWidth * engine->cachedImageHeight * sizeof(float);
        engine->cachedConvOutputBuffer = [engine->device newBufferWithLength:bufferSize
                                                                     options:MTLResourceStorageModeShared];
    }
    return engine->cachedConvOutputBuffer;
}

// Create Metal device
uintptr_t create_metal_device() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Failed to create Metal device");
            return 0;
        }
        
        // Return device pointer (ARC will manage lifetime)
        return (uintptr_t)(__bridge_retained void*)device;
    }
}

// Destroy Metal device
void destroy_metal_device(uintptr_t device_ptr) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)(void*)device_ptr;
        device = nil;
    }
}

// Create training engine with basic CNN structure
uintptr_t create_training_engine(uintptr_t device_ptr, training_config_t* config) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        if (!device || !config) {
            return 0;
        }
        
        // Allocate training engine
        training_engine_t* engine = malloc(sizeof(training_engine_t));
        if (!engine) {
            return 0;
        }
        
        engine->device = device;
        engine->config = *config;
        engine->initialized = NO;
        
        // Create command queue
        engine->commandQueue = [device newCommandQueue];
        if (!engine->commandQueue) {
            free(engine);
            return 0;
        }
        
        // Create MPSGraph for CNN model
        engine->graph = [[MPSGraph alloc] init];
        if (!engine->graph) {
            free(engine);
            return 0;
        }
        
        // Build intermediate CNN graph - single conv layer first
        // Start simple: Conv1 → Global avg pool → FC  
        // This allows us to test 4D tensor operations incrementally
        
        MPSGraphTensor* inputTensor = [engine->graph placeholderWithShape:@[@32, @3, @32, @32]
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"input"];
        
        // Single convolution layer: 3->8 channels (smaller for testing)
        MPSGraphTensor* conv1Weights = [engine->graph placeholderWithShape:@[@8, @3, @3, @3]
                                                                  dataType:MPSDataTypeFloat32
                                                                      name:@"conv1_weights"];
        
        MPSGraphTensor* conv1Bias = [engine->graph placeholderWithShape:@[@8]
                                                               dataType:MPSDataTypeFloat32
                                                                   name:@"conv1_bias"];
        
        // FC layer weights: 8->2 (much smaller)
        MPSGraphTensor* fcWeights = [engine->graph placeholderWithShape:@[@8, @2]
                                                               dataType:MPSDataTypeFloat32
                                                                   name:@"fc_weights"];
        
        MPSGraphTensor* fcBias = [engine->graph placeholderWithShape:@[@2]
                                                            dataType:MPSDataTypeFloat32
                                                                name:@"fc_bias"];
        
        // Store references to placeholders for execution
        engine->inputTensor = inputTensor;
        engine->conv1Weights = conv1Weights;
        engine->conv1Bias = conv1Bias;
        engine->fcWeights = fcWeights;
        engine->fcBias = fcBias;
        
        // Forward pass - Conv1: input -> conv1
        MPSGraphConvolution2DOpDescriptor* conv1Desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
        conv1Desc.strideInX = 1;
        conv1Desc.strideInY = 1;
        conv1Desc.dilationRateInX = 1;
        conv1Desc.dilationRateInY = 1;
        conv1Desc.paddingLeft = 1;
        conv1Desc.paddingRight = 1; 
        conv1Desc.paddingTop = 1;
        conv1Desc.paddingBottom = 1;
        
        MPSGraphTensor* conv1 = [engine->graph convolution2DWithSourceTensor:inputTensor
                                                               weightsTensor:conv1Weights
                                                                  descriptor:conv1Desc
                                                                        name:@"conv1"];
        
        // Reshape bias to be broadcastable: [8] -> [1, 8, 1, 1] 
        MPSGraphTensor* conv1BiasReshaped = [engine->graph reshapeTensor:conv1Bias
                                                               withShape:@[@1, @8, @1, @1]
                                                                    name:@"conv1_bias_reshaped"];
        
        conv1 = [engine->graph additionWithPrimaryTensor:conv1
                                          secondaryTensor:conv1BiasReshaped
                                                     name:@"conv1_bias_add"];
        
        conv1 = [engine->graph reLUWithTensor:conv1 name:@"conv1_relu"];
        
        // Global average pooling: [batch, 8, H, W] -> [batch, 8, 1, 1]
        MPSGraphTensor* pooled = [engine->graph meanOfTensor:conv1
                                                        axes:@[@2, @3]
                                                        name:@"global_avg_pool"];
        
        // Flatten to [batch, 8] for FC layer
        MPSGraphTensor* flattened = [engine->graph reshapeTensor:pooled
                                                       withShape:@[@32, @8]
                                                            name:@"flatten"];
        
        // FC layer: [batch, 8] -> [batch, 2]
        MPSGraphTensor* logits = [engine->graph matrixMultiplicationWithPrimaryTensor:flattened
                                                                      secondaryTensor:fcWeights
                                                                                 name:@"fc"];
        
        logits = [engine->graph additionWithPrimaryTensor:logits
                                           secondaryTensor:fcBias
                                                      name:@"fc_bias_add"];
        
        // Simple loss as mean of logits (tests forward pass)
        MPSGraphTensor* loss = [engine->graph meanOfTensor:logits
                                                      axes:@[@0, @1]
                                                      name:@"simple_loss"];
        
        // Store reference to loss output
        engine->lossOutput = loss;
        
        // Skip compilation - use direct graph execution instead
        // This avoids the isStaticMPSType assertion that occurs during compilation
        // NSLog(@"Successfully created MPSGraph for intermediate CNN forward pass (direct execution)");
        engine->initialized = YES;
        engine->useConstantWeights = NO; // Default to placeholder approach
        
        return (uintptr_t)engine;
    }
}

// SOLUTION: Create training engine with constant weights to avoid MPSGraph convolution issues
uintptr_t create_training_engine_constant_weights(uintptr_t device_ptr, training_config_t* config) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        if (!device || !config) {
            return 0;
        }
        
        // Allocate training engine
        training_engine_t* engine = malloc(sizeof(training_engine_t));
        if (!engine) {
            return 0;
        }
        
        engine->device = device;
        engine->config = *config;
        engine->initialized = NO;
        engine->useConstantWeights = YES; // Use constant weights approach
        
        // Create command queue
        engine->commandQueue = [device newCommandQueue];
        if (!engine->commandQueue) {
            free(engine);
            return 0;
        }
        
        // Create MPSGraph for CNN model
        engine->graph = [[MPSGraph alloc] init];
        if (!engine->graph) {
            free(engine);
            return 0;
        }
        
        // Build CNN graph with CONSTANT WEIGHTS
        
        // Input placeholder (only input needs to be a placeholder)
        MPSGraphTensor* inputTensor = [engine->graph placeholderWithShape:@[@32, @3, @32, @32]
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"input"];
        engine->inputTensor = inputTensor;
        
        // Initialize weight data
        float* conv1WeightData = (float*)malloc(8*3*3*3*sizeof(float));
        float* conv1BiasData = (float*)malloc(8*sizeof(float));
        float* fcWeightData = (float*)malloc(8*2*sizeof(float));
        float* fcBiasData = (float*)malloc(2*sizeof(float));
        
        // Initialize with small random values
        for(int i = 0; i < 8*3*3*3; i++) conv1WeightData[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for(int i = 0; i < 8; i++) conv1BiasData[i] = 0.0f;
        for(int i = 0; i < 8*2; i++) fcWeightData[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for(int i = 0; i < 2; i++) fcBiasData[i] = 0.0f;
        
        // Create CONSTANT weight tensors (not placeholders)
        engine->conv1WeightsConst = [engine->graph constantWithData:[NSData dataWithBytes:conv1WeightData length:8*3*3*3*sizeof(float)]
                                                              shape:@[@8, @3, @3, @3]
                                                           dataType:MPSDataTypeFloat32];
        
        engine->conv1BiasConst = [engine->graph constantWithData:[NSData dataWithBytes:conv1BiasData length:8*sizeof(float)]
                                                           shape:@[@8]
                                                        dataType:MPSDataTypeFloat32];
        
        engine->fcWeightsConst = [engine->graph constantWithData:[NSData dataWithBytes:fcWeightData length:8*2*sizeof(float)]
                                                           shape:@[@8, @2]
                                                        dataType:MPSDataTypeFloat32];
        
        engine->fcBiasConst = [engine->graph constantWithData:[NSData dataWithBytes:fcBiasData length:2*sizeof(float)]
                                                        shape:@[@2]
                                                     dataType:MPSDataTypeFloat32];
        
        // Free temporary weight data
        free(conv1WeightData);
        free(conv1BiasData);
        free(fcWeightData);
        free(fcBiasData);
        
        // Build forward pass using CONSTANT weights
        MPSGraphConvolution2DOpDescriptor* conv1Desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
        conv1Desc.strideInX = 1;
        conv1Desc.strideInY = 1;
        conv1Desc.dilationRateInX = 1;
        conv1Desc.dilationRateInY = 1;
        conv1Desc.paddingLeft = 1;
        conv1Desc.paddingRight = 1; 
        conv1Desc.paddingTop = 1;
        conv1Desc.paddingBottom = 1;
        
        MPSGraphTensor* conv1 = [engine->graph convolution2DWithSourceTensor:inputTensor
                                                               weightsTensor:engine->conv1WeightsConst  // Use constant weights
                                                                  descriptor:conv1Desc
                                                                        name:@"conv1"];
        
        // Reshape bias to be broadcastable
        MPSGraphTensor* conv1BiasReshaped = [engine->graph reshapeTensor:engine->conv1BiasConst
                                                               withShape:@[@1, @8, @1, @1]
                                                                    name:@"conv1_bias_reshaped"];
        
        conv1 = [engine->graph additionWithPrimaryTensor:conv1
                                          secondaryTensor:conv1BiasReshaped
                                                     name:@"conv1_bias_add"];
        
        conv1 = [engine->graph reLUWithTensor:conv1 name:@"conv1_relu"];
        
        // Global average pooling
        MPSGraphTensor* pooled = [engine->graph meanOfTensor:conv1
                                                        axes:@[@2, @3]
                                                        name:@"global_avg_pool"];
        
        // Flatten
        MPSGraphTensor* flattened = [engine->graph reshapeTensor:pooled
                                                       withShape:@[@32, @8]
                                                            name:@"flatten"];
        
        // FC layer with constant weights
        MPSGraphTensor* logits = [engine->graph matrixMultiplicationWithPrimaryTensor:flattened
                                                                      secondaryTensor:engine->fcWeightsConst
                                                                                 name:@"fc"];
        
        logits = [engine->graph additionWithPrimaryTensor:logits
                                           secondaryTensor:engine->fcBiasConst
                                                      name:@"fc_bias_add"];
        
        // Simple loss
        MPSGraphTensor* loss = [engine->graph meanOfTensor:logits
                                                      axes:@[@0, @1]
                                                      name:@"simple_loss"];
        
        engine->lossOutput = loss;
        
        // NSLog(@"Successfully created MPSGraph with CONSTANT WEIGHTS for CNN");
        engine->initialized = YES;
        
        return (uintptr_t)engine;
    }
}

// NEW: Create hybrid training engine (MPS + MPSGraph)
uintptr_t create_training_engine_hybrid(uintptr_t device_ptr, training_config_t* config, model_config_t* model_config) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        if (!device || !config || !model_config) {
            return 0;
        }
        
        // Allocate training engine
        training_engine_t* engine = malloc(sizeof(training_engine_t));
        if (!engine) {
            return 0;
        }
        
        engine->device = device;
        engine->config = *config;
        engine->model_config = *model_config;  // Store model configuration
        engine->initialized = NO;
        engine->useConstantWeights = NO;
        engine->useHybridApproach = YES; // Enable hybrid approach
        
        // Create command queue
        engine->commandQueue = [device newCommandQueue];
        if (!engine->commandQueue) {
            free(engine);
            return 0;
        }
        
        // Create MPSGraph for post-convolution operations
        engine->graph = [[MPSGraph alloc] init];
        if (!engine->graph) {
            free(engine);
            return 0;
        }
        
        // === STEP 1: Create MPS Convolution Layer ===
        
        // Conv1 descriptor: using dynamic model configuration
        MPSCNNConvolutionDescriptor* conv1Desc = [MPSCNNConvolutionDescriptor 
            cnnConvolutionDescriptorWithKernelWidth:engine->model_config.conv1_kernel_size 
                                        kernelHeight:engine->model_config.conv1_kernel_size 
                                inputFeatureChannels:engine->model_config.input_channels 
                               outputFeatureChannels:engine->model_config.conv1_out_channels];
        conv1Desc.strideInPixelsX = engine->model_config.conv1_stride;
        conv1Desc.strideInPixelsY = engine->model_config.conv1_stride;
        
        // Create weight and bias buffers for MPS convolution
        size_t weightSize = engine->model_config.conv1_out_channels * engine->model_config.input_channels * 
                            engine->model_config.conv1_kernel_size * engine->model_config.conv1_kernel_size * sizeof(float);
        size_t biasSize = engine->model_config.conv1_out_channels * sizeof(float);
        
        engine->conv1WeightBuffer = [device newBufferWithLength:weightSize 
                                                       options:MTLResourceStorageModeShared];
        engine->conv1BiasBuffer = [device newBufferWithLength:biasSize 
                                                     options:MTLResourceStorageModeShared];
        
        if (!engine->conv1WeightBuffer || !engine->conv1BiasBuffer) {
            free(engine);
            return 0;
        }
        
        // Initialize weights with small random values
        float* weightData = (float*)engine->conv1WeightBuffer.contents;
        float* biasData = (float*)engine->conv1BiasBuffer.contents;
        
        int weightCount = engine->model_config.conv1_out_channels * engine->model_config.input_channels * 
                          engine->model_config.conv1_kernel_size * engine->model_config.conv1_kernel_size;
        for (int i = 0; i < weightCount; i++) {
            weightData[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = 0; i < engine->model_config.conv1_out_channels; i++) {
            biasData[i] = 0.0f;
        }
        
        // Create Conv1 layer (3->16 channels)
        engine->conv1Layer = [[MPSCNNConvolution alloc] 
            initWithDevice:device 
            convolutionDescriptor:conv1Desc 
            kernelWeights:weightData 
            biasTerms:biasData 
            flags:MPSCNNConvolutionFlagsNone];
        
        if (!engine->conv1Layer) {
            free(engine);
            return 0;
        }
        
        // === CREATE CONV2 LAYER: using dynamic model configuration ===
        MPSCNNConvolutionDescriptor* conv2Desc = [MPSCNNConvolutionDescriptor 
            cnnConvolutionDescriptorWithKernelWidth:engine->model_config.conv2_kernel_size 
                                      kernelHeight:engine->model_config.conv2_kernel_size 
                               inputFeatureChannels:engine->model_config.conv1_out_channels
                              outputFeatureChannels:engine->model_config.conv2_out_channels];
        conv2Desc.strideInPixelsX = engine->model_config.conv2_stride;
        conv2Desc.strideInPixelsY = engine->model_config.conv2_stride;
        
        // Create Conv2 buffers
        int conv2WeightSize = engine->model_config.conv2_out_channels * engine->model_config.conv1_out_channels * 
                              engine->model_config.conv2_kernel_size * engine->model_config.conv2_kernel_size * sizeof(float);
        int conv2BiasSize = engine->model_config.conv2_out_channels * sizeof(float);
        
        engine->conv2WeightBuffer = [device newBufferWithLength:conv2WeightSize 
                                                        options:MTLResourceStorageModeShared];
        engine->conv2BiasBuffer = [device newBufferWithLength:conv2BiasSize 
                                                      options:MTLResourceStorageModeShared];
        
        if (!engine->conv2WeightBuffer || !engine->conv2BiasBuffer) {
            free(engine);
            return 0;
        }
        
        // Initialize Conv2 weights
        float* conv2WeightData = (float*)engine->conv2WeightBuffer.contents;
        float* conv2BiasData = (float*)engine->conv2BiasBuffer.contents;
        
        int conv2WeightCount = engine->model_config.conv2_out_channels * engine->model_config.conv1_out_channels * 
                               engine->model_config.conv2_kernel_size * engine->model_config.conv2_kernel_size;
        for (int i = 0; i < conv2WeightCount; i++) {
            conv2WeightData[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = 0; i < engine->model_config.conv2_out_channels; i++) {
            conv2BiasData[i] = 0.0f;
        }
        
        // Create Conv2 layer
        engine->conv2Layer = [[MPSCNNConvolution alloc] 
            initWithDevice:device 
            convolutionDescriptor:conv2Desc 
            kernelWeights:conv2WeightData 
            biasTerms:conv2BiasData 
            flags:MPSCNNConvolutionFlagsNone];
        
        if (!engine->conv2Layer) {
            free(engine);
            return 0;
        }
        
        // === CREATE CONV3 LAYER: using dynamic model configuration ===
        MPSCNNConvolutionDescriptor* conv3Desc = [MPSCNNConvolutionDescriptor 
            cnnConvolutionDescriptorWithKernelWidth:engine->model_config.conv3_kernel_size 
                                      kernelHeight:engine->model_config.conv3_kernel_size 
                               inputFeatureChannels:engine->model_config.conv2_out_channels
                              outputFeatureChannels:engine->model_config.conv3_out_channels];
        conv3Desc.strideInPixelsX = engine->model_config.conv3_stride;
        conv3Desc.strideInPixelsY = engine->model_config.conv3_stride;
        
        // Create Conv3 buffers
        int conv3WeightSize = engine->model_config.conv3_out_channels * engine->model_config.conv2_out_channels * 
                              engine->model_config.conv3_kernel_size * engine->model_config.conv3_kernel_size * sizeof(float);
        int conv3BiasSize = engine->model_config.conv3_out_channels * sizeof(float);
        
        engine->conv3WeightBuffer = [device newBufferWithLength:conv3WeightSize 
                                                        options:MTLResourceStorageModeShared];
        engine->conv3BiasBuffer = [device newBufferWithLength:conv3BiasSize 
                                                      options:MTLResourceStorageModeShared];
        
        if (!engine->conv3WeightBuffer || !engine->conv3BiasBuffer) {
            free(engine);
            return 0;
        }
        
        // Initialize Conv3 weights
        float* conv3WeightData = (float*)engine->conv3WeightBuffer.contents;
        float* conv3BiasData = (float*)engine->conv3BiasBuffer.contents;
        
        int conv3WeightCount = engine->model_config.conv3_out_channels * engine->model_config.conv2_out_channels * 
                               engine->model_config.conv3_kernel_size * engine->model_config.conv3_kernel_size;
        for (int i = 0; i < conv3WeightCount; i++) {
            conv3WeightData[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = 0; i < engine->model_config.conv3_out_channels; i++) {
            conv3BiasData[i] = 0.0f;
        }
        
        // Create Conv3 layer
        engine->conv3Layer = [[MPSCNNConvolution alloc] 
            initWithDevice:device 
            convolutionDescriptor:conv3Desc 
            kernelWeights:conv3WeightData 
            biasTerms:conv3BiasData 
            flags:MPSCNNConvolutionFlagsNone];
        
        if (!engine->conv3Layer) {
            free(engine);
            return 0;
        }
        
        // === STEP 2: Create MPSGraph for post-convolution operations ===
        
        // Input to MPSGraph will be the flattened output of 3 conv layers
        NSNumber* batchSize = @(engine->model_config.batch_size);
        NSNumber* flattenedSize = @(engine->model_config.fc1_input_size);
        engine->hybridInputTensor = [engine->graph placeholderWithShape:@[batchSize, flattenedSize]
                                                               dataType:MPSDataTypeFloat32
                                                                   name:@"conv_output"];
        
        // FC weights and bias placeholders - using dynamic model dimensions
        engine->fcWeights = [engine->graph placeholderWithShape:@[@(engine->model_config.fc1_input_size), @(engine->model_config.fc1_output_size)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"fc_weights"];
        
        engine->fcBias = [engine->graph placeholderWithShape:@[@(engine->model_config.fc1_output_size)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:@"fc_bias"];
        
        // FC2 weights and bias placeholders - using dynamic model dimensions
        engine->fc2Weights = [engine->graph placeholderWithShape:@[@(engine->model_config.fc1_output_size), @(engine->model_config.fc2_output_size)]
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"fc2_weights"];
        
        engine->fc2Bias = [engine->graph placeholderWithShape:@[@(engine->model_config.fc2_output_size)]
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"fc2_bias"];
        
        // Build post-convolution graph: FC1 -> ReLU -> FC2 -> Softmax -> Loss
        // Input is already flattened [16, 16384], so no need for pooling or reshaping
        
        // FC1 layer: [16, 16384] -> [16, 128] 
        MPSGraphTensor* fc1Output = [engine->graph matrixMultiplicationWithPrimaryTensor:engine->hybridInputTensor
                                                                        secondaryTensor:engine->fcWeights
                                                                                   name:@"fc1"];
        
        fc1Output = [engine->graph additionWithPrimaryTensor:fc1Output
                                             secondaryTensor:engine->fcBias
                                                        name:@"fc1_bias_add"];
        
        // ReLU activation after FC1
        MPSGraphTensor* fc1Relu = [engine->graph reLUWithTensor:fc1Output name:@"fc1_relu"];
        
        // FC2 layer: [16, 128] -> [16, 2] (final output)
        MPSGraphTensor* logits = [engine->graph matrixMultiplicationWithPrimaryTensor:fc1Relu
                                                                      secondaryTensor:engine->fc2Weights
                                                                                 name:@"fc2"];
        
        logits = [engine->graph additionWithPrimaryTensor:logits
                                           secondaryTensor:engine->fc2Bias
                                                      name:@"fc2_bias_add"];
        
        // Simple loss as mean of logits
        MPSGraphTensor* loss = [engine->graph meanOfTensor:logits
                                                      axes:@[@0, @1]
                                                      name:@"simple_loss"];
        
        engine->lossOutput = loss;
        
        // === STEP 3: Create backward graph for gradient computation ===
        
        // Add label placeholder for proper loss computation (one-hot encoded float labels)
        engine->labelTensor = [engine->graph placeholderWithShape:@[@(engine->model_config.batch_size), @(engine->model_config.fc2_output_size)]
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"labels"];
        
        // Create proper softmax cross-entropy loss using softmax + logarithm
        MPSGraphTensor* softmaxLogits = [engine->graph softMaxWithTensor:logits
                                                                    axis:1
                                                                    name:@"softmax"];
        
        MPSGraphTensor* logSoftmax = [engine->graph logarithmWithTensor:softmaxLogits
                                                                   name:@"log_softmax"];
        
        // Element-wise multiplication with one-hot labels (both float32)
        MPSGraphTensor* crossEntropy = [engine->graph multiplicationWithPrimaryTensor:engine->labelTensor
                                                                      secondaryTensor:logSoftmax
                                                                                 name:@"cross_entropy"];
        
        // Sum across classes and negate
        MPSGraphTensor* sumCrossEntropy = [engine->graph reductionSumWithTensor:crossEntropy
                                                                           axes:@[@1]
                                                                           name:@"sum_cross_entropy"];
        
        MPSGraphTensor* negCrossEntropy = [engine->graph negativeWithTensor:sumCrossEntropy
                                                                       name:@"neg_cross_entropy"];
        
        // Mean loss across batch
        MPSGraphTensor* meanLoss = [engine->graph meanOfTensor:negCrossEntropy
                                                          axes:@[@0]
                                                          name:@"mean_loss"];
        
        // Update loss output to use proper loss
        engine->lossOutput = meanLoss;
        
        // Compute gradients using MPSGraph automatic differentiation (correct API)
        NSArray<MPSGraphTensor*>* inputTensors = @[engine->fcWeights, engine->fcBias, engine->fc2Weights, engine->fc2Bias];
        NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* gradientDict = 
            [engine->graph gradientForPrimaryTensor:meanLoss
                             withTensors:inputTensors
                                    name:@"gradients"];
        
        // Store gradient tensors  
        engine->fcWeightGrads = gradientDict[engine->fcWeights];      // FC1 weight gradients
        engine->fcBiasGrads = gradientDict[engine->fcBias];          // FC1 bias gradients
        engine->fc2WeightGrads = gradientDict[engine->fc2Weights];    // FC2 weight gradients
        engine->fc2BiasGrads = gradientDict[engine->fc2Bias];        // FC2 bias gradients
        
        // Store gradient tensors - shapes computed automatically by MPSGraph
        
        // NSLog(@"Successfully created HYBRID MPS/MPSGraph training engine");
        // NSLog(@"  - MPS Convolution: 3->8 channels, 3x3 kernel");
        // NSLog(@"  - MPSGraph: ReLU -> GlobalPool -> FC -> Loss + Gradients");
        
        // MEMORY LEAK FIX: Initialize cached buffers and dimensions
        engine->cachedConvOutputBuffer = nil;  // Will be created on first use
        engine->cachedInputImage = nil;        // Will be created on first use
        engine->cachedConvOutputImage = nil;   // Will be created on first use
        engine->cachedBatchSize = 0;
        engine->cachedInputChannels = 0;
        engine->cachedOutputChannels = 0;
        engine->cachedImageWidth = 0;
        engine->cachedImageHeight = 0;
        
        engine->initialized = YES;
        
        return (uintptr_t)engine;
    }
}

// Destroy training engine
void destroy_training_engine(uintptr_t engine_ptr) {
    @autoreleasepool {
        if (engine_ptr == 0) {
            return;
        }
        
        training_engine_t* engine = (training_engine_t*)engine_ptr;
        
        // Release Objective-C objects
        engine->graph = nil;
        engine->executable = nil;
        engine->commandQueue = nil;
        engine->device = nil;

        // Explicitly release objects for the hybrid approach that were stored directly in the C struct
        engine->conv1Layer = nil; // Release the MPSCNNConvolution object
        engine->conv2Layer = nil; // Release the MPSCNNConvolution object
        engine->conv3Layer = nil; // Release the MPSCNNConvolution object
        engine->conv1WeightBuffer = nil; // Release the MTLBuffer
        engine->conv1BiasBuffer = nil;   // Release the MTLBuffer
        engine->conv2WeightBuffer = nil; // Release the MTLBuffer
        engine->conv2BiasBuffer = nil;   // Release the MTLBuffer
        engine->conv3WeightBuffer = nil; // Release the MTLBuffer
        engine->conv3BiasBuffer = nil;   // Release the MTLBuffer
        
        // MEMORY LEAK FIX: Release cached buffers
        engine->cachedConvOutputBuffer = nil;
        engine->cachedInputImage = nil;
        engine->cachedConvOutputImage = nil;
        
        // Free the engine structure
        free(engine);
    }
}

// Create Metal command queue with proper resource management
uintptr_t create_command_queue(uintptr_t device_ptr) {
    @autoreleasepool {
        if (device_ptr == 0) {
            NSLog(@"❌ Cannot create command queue: device is null");
            return 0;
        }
        
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        if (!commandQueue) {
            NSLog(@"❌ Failed to create Metal command queue");
            return 0;
        }
        
        // Set a label for debugging
        commandQueue.label = @"go-metal-training-queue";
        
        // Return retained pointer to prevent ARC cleanup
        return (uintptr_t)CFBridgingRetain(commandQueue);
    }
}

// Create training engine with dynamic graph based on model specification
// This replaces the hardcoded hybrid CNN with dynamic architecture support
uintptr_t create_training_engine_dynamic(
    uintptr_t device_ptr,
    training_config_t* config,
    layer_spec_c_t* layers,
    int num_layers,
    int* input_shape,
    int input_shape_len
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_ptr;
        if (!device || !config || !layers || num_layers <= 0) {
            NSLog(@"Invalid parameters for dynamic training engine creation");
            return 0;
        }
        
        // Allocate and zero-initialize training engine to prevent garbage values
        // calloc is safe for structs with Objective-C objects since nil == 0
        training_engine_t* engine = calloc(1, sizeof(training_engine_t));
        if (!engine) {
            NSLog(@"Failed to allocate training engine");
            return 0;
        }
        
        engine->device = device;
        engine->config = *config;
        engine->initialized = NO;
        engine->useHybridApproach = YES; // Always use hybrid for dynamic models
        
        // Initialize placeholder tracking arrays
        engine->allWeightPlaceholders = [[NSMutableArray alloc] init];
        engine->allBiasPlaceholders = [[NSMutableArray alloc] init];
        
        // Initialize optimizer-specific graph compilation arrays
        engine->sgdGraphCompiled = NO;
        engine->sgdPrecompiledGradients = [[NSMutableArray alloc] init];
        engine->sgdPrecompiledUpdatedParams = [[NSMutableArray alloc] init]; 
        engine->sgdPrecompiledUpdatedMomentum = [[NSMutableArray alloc] init];
        engine->sgdScalarsCached = NO;
        engine->sgdCachedLrTensor = nil;
        engine->sgdCachedMomentumTensor = nil;
        
        engine->adamGraphCompiled = NO;
        engine->adamPrecompiledGradients = [[NSMutableArray alloc] init];
        engine->adamPrecompiledUpdatedParams = [[NSMutableArray alloc] init];
        engine->adamPrecompiledUpdatedMomentum = [[NSMutableArray alloc] init];
        engine->adamPrecompiledUpdatedVariance = [[NSMutableArray alloc] init];
        
        // Initialize RMSProp-specific graph compilation arrays
        engine->rmspropGraphCompiled = NO;
        engine->rmspropStateInitialized = NO;
        engine->rmspropPrecompiledGradients = [[NSMutableArray alloc] init];
        engine->rmspropPrecompiledUpdatedParams = [[NSMutableArray alloc] init];
        engine->rmspropPrecompiledUpdatedMomentum = [[NSMutableArray alloc] init];
        engine->rmspropPrecompiledUpdatedSquaredGrad = [[NSMutableArray alloc] init];
        engine->rmspropPrecompiledUpdatedGradAvg = [[NSMutableArray alloc] init];
        
        // Create command queue
        engine->commandQueue = [device newCommandQueue];
        if (!engine->commandQueue) {
            NSLog(@"Failed to create command queue");
            free(engine);
            return 0;
        }
        
        // Create MPSGraph for dynamic model
        engine->graph = [[MPSGraph alloc] init];
        if (!engine->graph) {
            NSLog(@"Failed to create MPSGraph");
            free(engine);
            return 0;
        }
        
        @try {
            // Build dynamic graph from layer specifications
            if (!buildDynamicGraphFromLayers(engine, 
                                             layers, 
                                             num_layers, 
                                             input_shape, 
                                             input_shape_len)) {
                NSLog(@"Failed to build dynamic graph");
                free(engine);
                return 0;
            }
            
            // Adam state initialization moved earlier - before pre-compilation
            
            engine->initialized = YES;
            // NSLog(@"✅ Dynamic training engine created successfully with %d layers", num_layers);
            
            return (uintptr_t)engine;
            
        } @catch (NSException* exception) {
            NSLog(@"Exception during dynamic graph creation: %@", exception.reason);
            free(engine);
            return 0;
        }
    }
}




// Create Metal command buffer
uintptr_t create_command_buffer(uintptr_t command_queue_ptr) {
    @autoreleasepool {
        if (command_queue_ptr == 0) {
            NSLog(@"Cannot create command buffer: command queue is null");
            return 0;
        }
        
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)(void*)command_queue_ptr;
        if (!commandQueue) {
            NSLog(@"Command queue is nil in create_command_buffer");
            return 0;
        }
        
        @try {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            if (!commandBuffer) {
                NSLog(@"Failed to create command buffer");
                return 0;
            }
            
            // Return retained reference
            return (uintptr_t)(__bridge_retained void*)commandBuffer;
        } @catch (NSException* exception) {
            NSLog(@"Exception creating command buffer: %@", exception.reason);
            return 0;
        }
    }
}

// Release Metal command buffer immediately
void release_command_buffer(uintptr_t command_buffer_ptr) {
    if (command_buffer_ptr != 0) {
        CFBridgingRelease((void*)command_buffer_ptr);
    }
}

// Release command queue
void release_command_queue(uintptr_t command_queue_ptr) {
    if (command_queue_ptr != 0) {
        CFBridgingRelease((void*)command_queue_ptr);
    }
}

// Commit command buffer for execution
int commit_command_buffer(uintptr_t command_buffer_ptr) {
    @autoreleasepool {
        if (command_buffer_ptr == 0) {
            NSLog(@"❌ Cannot commit command buffer: buffer is null");
            return -1;
        }
        
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)(void*)command_buffer_ptr;
        
        @try {
            [commandBuffer commit];
            return 0; // Success
        } @catch (NSException* exception) {
            NSLog(@"❌ Failed to commit command buffer: %@", exception.reason);
            return -2;
        }
    }
}

// Wait for command buffer completion with timeout and cleanup
int wait_command_buffer_completion(uintptr_t command_buffer_ptr) {
    @autoreleasepool {
        if (command_buffer_ptr == 0) {
            NSLog(@"❌ Cannot wait for command buffer: buffer is null");
            return -1;
        }
        
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)(void*)command_buffer_ptr;
        
        @try {
            [commandBuffer waitUntilCompleted];
            
            // Check for execution errors
            if (commandBuffer.status == MTLCommandBufferStatusError) {
                NSLog(@"❌ Command buffer execution failed: %@", commandBuffer.error.localizedDescription);
                return -2;
            }
            
            return 0; // Success
        } @catch (NSException* exception) {
            NSLog(@"❌ Command buffer wait failed: %@", exception.reason);
            return -3;
        }
    }
}

// Setup autorelease pool  
void setup_autorelease_pool() {
    // No-op in ARC - autorelease pools are managed automatically
}

// Drain autorelease pool
void drain_autorelease_pool() {
    // No-op in ARC - autorelease pools are managed automatically  
}