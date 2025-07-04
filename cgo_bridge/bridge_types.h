#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Layer specification structure for dynamic graph creation
typedef struct {
    int layer_type;          // 0=Dense, 1=Conv2D, 2=ReLU, 3=Softmax
    char name[64];           // Layer name
    int input_shape[4];      // Input dimensions [batch, channels, height, width]
    int input_shape_len;     // Number of valid dimensions
    int output_shape[4];     // Output dimensions
    int output_shape_len;    // Number of valid dimensions
    
    // Layer-specific parameters
    int param_int[8];        // Integer parameters (e.g., kernel_size, stride, padding)
    float param_float[8];    // Float parameters (e.g., dropout_rate)
    int param_int_count;     // Number of valid int parameters
    int param_float_count;   // Number of valid float parameters
} layer_spec_c_t;

// Training configuration struct from Go
typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float weight_decay;
    float epsilon;
    int optimizer_type; // 0 = SGD, 1 = Adam
} training_config_t;

// Training engine structure - Hybrid MPS/MPSGraph approach
typedef struct {
    id<MTLDevice> device;
    MPSGraph* graph;
    MPSGraphExecutable* executable;
    id<MTLCommandQueue> commandQueue;
    training_config_t config;
    BOOL initialized;
    BOOL useHybridApproach; // NEW: Use MPS for convolution, MPSGraph for rest
    
    // Store references to placeholders for execution
    __unsafe_unretained MPSGraphTensor* inputTensor;
    __unsafe_unretained MPSGraphTensor* conv1Weights;
    __unsafe_unretained MPSGraphTensor* conv1Bias;
    __unsafe_unretained MPSGraphTensor* fcWeights;      // FC1 weights: 262144 -> 128
    __unsafe_unretained MPSGraphTensor* fcBias;        // FC1 bias: 128
    __unsafe_unretained MPSGraphTensor* fc2Weights;    // FC2 weights: 128 -> 2  
    __unsafe_unretained MPSGraphTensor* fc2Bias;       // FC2 bias: 2
    __unsafe_unretained MPSGraphTensor* lossOutput;
    __unsafe_unretained MPSGraphTensor* predictionsTensor; // Softmax predictions for inference
    
    // SOLUTION: Add constant weight tensors to avoid external data issues
    __unsafe_unretained MPSGraphTensor* conv1WeightsConst;
    __unsafe_unretained MPSGraphTensor* conv1BiasConst;
    __unsafe_unretained MPSGraphTensor* fcWeightsConst;    // FC1 constant weights
    __unsafe_unretained MPSGraphTensor* fcBiasConst;      // FC1 constant bias
    __unsafe_unretained MPSGraphTensor* fc2WeightsConst;  // FC2 constant weights
    __unsafe_unretained MPSGraphTensor* fc2BiasConst;     // FC2 constant bias
    BOOL useConstantWeights;
    
    // NEW: MPS Convolution objects for hybrid approach - complete 3-layer CNN
    MPSCNNConvolution* conv1Layer;         // Conv1: 3->16 channels
    MPSCNNConvolution* conv2Layer;         // Conv2: 16->32 channels  
    MPSCNNConvolution* conv3Layer;         // Conv3: 32->64 channels
    id<MTLBuffer> conv1WeightBuffer;       // Conv1 weight buffer
    id<MTLBuffer> conv1BiasBuffer;         // Conv1 bias buffer
    id<MTLBuffer> conv2WeightBuffer;       // Conv2 weight buffer
    id<MTLBuffer> conv2BiasBuffer;         // Conv2 bias buffer
    id<MTLBuffer> conv3WeightBuffer;       // Conv3 weight buffer
    id<MTLBuffer> conv3BiasBuffer;         // Conv3 bias buffer
    __unsafe_unretained MPSGraphTensor* hybridInputTensor;  // Input to hybrid graph (post-convolution)
    
    // Backward pass support
    __unsafe_unretained MPSGraphTensor* labelTensor;        // Labels for loss computation
    __unsafe_unretained MPSGraphTensor* fcWeightGrads;      // FC1 weight gradients
    __unsafe_unretained MPSGraphTensor* fcBiasGrads;        // FC1 bias gradients
    __unsafe_unretained MPSGraphTensor* fc2WeightGrads;     // FC2 weight gradients  
    __unsafe_unretained MPSGraphTensor* fc2BiasGrads;       // FC2 bias gradients
    MPSGraph* backwardGraph;                                // Separate graph for gradients
    
    // Dynamic graph placeholders (for complex architectures)
    NSMutableArray* allWeightPlaceholders;                  // All weight placeholders in order
    NSMutableArray* allBiasPlaceholders;                    // All bias placeholders in order
    
    // MEMORY LEAK FIX: Cached buffers to avoid per-step allocations
    id<MTLBuffer> cachedConvOutputBuffer;                   // Reusable buffer for conv output
    MPSImage* cachedInputImage;                             // Reusable input image
    MPSImage* cachedConvOutputImage;                        // Reusable conv output image
    
    // Track dimensions for dynamic buffer resizing
    int cachedBatchSize;
    int cachedInputChannels;
    int cachedOutputChannels;
    int cachedImageWidth;
    int cachedImageHeight;
    
    // PRODUCTION OPTIMIZATION: Cached scalar tensors for Adam optimizer
    // This eliminates the primary source of performance degradation: scalar tensor creation
    MPSGraphTensor* cachedLrTensor;                         // Cached learning rate scalar
    MPSGraphTensor* cachedBeta1Tensor;                      // Cached beta1 scalar  
    MPSGraphTensor* cachedBeta2Tensor;                      // Cached beta2 scalar
    MPSGraphTensor* cachedEpsilonTensor;                    // Cached epsilon scalar
    MPSGraphTensor* cachedOneTensor;                        // Cached 1.0 scalar
    MPSGraphTensor* cachedOneMinusBeta1;                    // Cached (1 - beta1) scalar
    MPSGraphTensor* cachedOneMinusBeta2;                    // Cached (1 - beta2) scalar
    MPSGraphTensor* biasCorr1Placeholder;                   // Dynamic bias correction 1 placeholder
    MPSGraphTensor* biasCorr2Placeholder;                   // Dynamic bias correction 2 placeholder
    BOOL adamScalarsCached;                                 // Flag indicating scalars are cached
    
    // UNIFIED OPTIMIZER: Adam optimizer state arrays for parameter updates
    NSMutableArray* momentumPlaceholders;                   // Momentum state for each parameter  
    NSMutableArray* variancePlaceholders;                   // Variance state for each parameter
    NSMutableArray* momentumVariables;                      // MPSGraph variables for momentum state
    NSMutableArray* varianceVariables;                      // MPSGraph variables for variance state
    NSMutableArray* momentumBuffers;                        // Metal buffers for momentum state
    NSMutableArray* varianceBuffers;                        // Metal buffers for variance state
    int adamStepCount;                                      // Training step counter for bias correction
    BOOL adamStateInitialized;                             // Flag indicating Adam state is ready
    
    // PERFORMANCE: Cached bias correction buffers to avoid per-step allocations
    id<MTLBuffer> cachedBiasCorr1Buffer;                    // Reusable buffer for bias correction 1
    id<MTLBuffer> cachedBiasCorr2Buffer;                    // Reusable buffer for bias correction 2
    
    // TRUE PRE-COMPILATION: Pre-compiled gradient and optimizer operations
    NSMutableArray* precompiledGradientTensors;             // Pre-compiled gradient tensors
    NSMutableArray* precompiledUpdatedParams;               // Pre-compiled parameter updates
    NSMutableArray* precompiledUpdatedMomentum;             // Pre-compiled momentum updates  
    NSMutableArray* precompiledUpdatedVariance;             // Pre-compiled variance updates
} training_engine_t;