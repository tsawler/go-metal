#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Layer specification structure for dynamic graph creation
typedef struct {
    int layer_type;          // 0=Dense, 1=Conv2D, 2=ReLU, 3=Softmax, 4=MaxPool2D, 5=Dropout, 6=BatchNorm
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
    
    // Running statistics for layers like BatchNorm (non-learnable parameters)
    float* running_mean;     // Running mean data
    float* running_var;      // Running variance data
    int running_stats_size;  // Size of running statistics arrays
    int has_running_stats;   // Boolean flag indicating if running stats are available
} layer_spec_c_t;

// Model configuration structure for dynamic dimensions
typedef struct {
    // Input configuration
    int batch_size;
    int input_channels;
    int input_height;
    int input_width;
    
    // Convolution layer outputs (calculated or provided)
    int conv1_out_channels;
    int conv1_out_height;
    int conv1_out_width;
    
    int conv2_out_channels;
    int conv2_out_height;
    int conv2_out_width;
    
    int conv3_out_channels;
    int conv3_out_height;
    int conv3_out_width;
    
    // Fully connected layer dimensions
    int fc1_input_size;      // Flattened conv output size
    int fc1_output_size;     // Hidden layer size
    int fc2_output_size;     // Number of classes
    
    // Convolution parameters
    int conv1_kernel_size;
    int conv1_stride;
    int conv2_kernel_size;
    int conv2_stride;
    int conv3_kernel_size;
    int conv3_stride;
} model_config_t;

// Training configuration struct from Go
typedef struct {
    float learning_rate;
    float beta1;             // Adam momentum decay / RMSProp momentum (if > 0)
    float beta2;             // Adam variance decay (unused for RMSProp)
    float weight_decay;      // L2 regularization
    float epsilon;           // Numerical stability constant
    float alpha;             // RMSProp smoothing constant (typically 0.99)
    float momentum;          // RMSProp momentum coefficient (typically 0.0)
    int centered;            // RMSProp centered variant flag (0=false, 1=true)
    int optimizer_type;      // 0 = SGD, 1 = Adam, 2 = RMSProp
    int problem_type;        // 0 = Classification, 1 = Regression
    int loss_function;       // 0 = CrossEntropy, 1 = SparseCrossEntropy, 2 = MSE, 3 = MAE, 4 = Huber
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
    NSMutableArray* batchnormRunningStatsPlaceholders;      // BatchNorm running mean/variance placeholders
    
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
    
    // PRODUCTION OPTIMIZATION: Cached scalar tensors for optimizers
    // This eliminates the primary source of performance degradation: scalar tensor creation
    MPSGraphTensor* cachedLrTensor;                         // Cached learning rate scalar (shared)
    
    // Adam-specific cached scalars
    MPSGraphTensor* cachedBeta1Tensor;                      // Cached beta1 scalar  
    MPSGraphTensor* cachedBeta2Tensor;                      // Cached beta2 scalar
    MPSGraphTensor* cachedEpsilonTensor;                    // Cached epsilon scalar
    MPSGraphTensor* cachedOneTensor;                        // Cached 1.0 scalar
    MPSGraphTensor* cachedOneMinusBeta1;                    // Cached (1 - beta1) scalar
    MPSGraphTensor* cachedOneMinusBeta2;                    // Cached (1 - beta2) scalar
    MPSGraphTensor* biasCorr1Placeholder;                   // Dynamic bias correction 1 placeholder
    MPSGraphTensor* biasCorr2Placeholder;                   // Dynamic bias correction 2 placeholder
    BOOL adamScalarsCached;                                 // Flag indicating Adam scalars are cached
    
    // SGD-specific cached scalars
    MPSGraphTensor* cachedMomentumTensor;                   // Cached momentum factor scalar (for SGD with momentum)
    MPSGraphTensor* cachedWeightDecayTensor;                // Cached weight decay scalar
    BOOL sgdScalarsCached;                                  // Flag indicating SGD scalars are cached
    
    // UNIFIED OPTIMIZER: Optimizer state arrays for parameter updates
    NSMutableArray* momentumPlaceholders;                   // Momentum state for each parameter (shared: Adam/SGD)
    NSMutableArray* variancePlaceholders;                   // Variance state for each parameter (Adam only)
    NSMutableArray* momentumVariables;                      // MPSGraph variables for momentum state
    NSMutableArray* varianceVariables;                      // MPSGraph variables for variance state (Adam only)
    NSMutableArray* momentumBuffers;                        // Metal buffers for momentum state (shared: Adam/SGD)
    NSMutableArray* varianceBuffers;                        // Metal buffers for variance state (Adam only)
    int adamStepCount;                                      // Training step counter for bias correction (Adam only)
    BOOL adamStateInitialized;                             // Flag indicating Adam state is ready
    BOOL sgdStateInitialized;                              // Flag indicating SGD state is ready
    
    // PERFORMANCE: Cached bias correction buffers to avoid per-step allocations
    id<MTLBuffer> cachedBiasCorr1Buffer;                    // Reusable buffer for bias correction 1
    id<MTLBuffer> cachedBiasCorr2Buffer;                    // Reusable buffer for bias correction 2
    
    // TRUE PRE-COMPILATION: Pre-compiled gradient and optimizer operations
    NSMutableArray* precompiledGradientTensors;             // Pre-compiled gradient tensors
    NSMutableArray* precompiledUpdatedParams;               // Pre-compiled parameter updates
    NSMutableArray* precompiledUpdatedMomentum;             // Pre-compiled momentum updates (shared: Adam/SGD)
    NSMutableArray* precompiledUpdatedVariance;             // Pre-compiled variance updates (Adam only)
    
    // SGD-specific pre-compiled operations
    NSMutableArray* precompiledSGDUpdatedParams;            // Pre-compiled SGD parameter updates
    NSMutableArray* precompiledSGDUpdatedMomentum;          // Pre-compiled SGD momentum updates (if using momentum)
    
    // OPTIMIZER-SPECIFIC GRAPH COMPILATION: Separate pre-compilation per optimizer
    // This prevents placeholder conflicts and enables optimal performance for each optimizer
    
    // SGD-specific graph compilation
    BOOL sgdGraphCompiled;                                 // Flag indicating SGD graph is compiled
    NSMutableArray* sgdPrecompiledGradients;               // SGD pre-compiled gradient tensors  
    NSMutableArray* sgdPrecompiledUpdatedParams;           // SGD pre-compiled parameter updates
    NSMutableArray* sgdPrecompiledUpdatedMomentum;         // SGD pre-compiled momentum updates (if momentum enabled)
    MPSGraphTensor* sgdCachedLrTensor;                     // Cached SGD learning rate scalar
    MPSGraphTensor* sgdCachedMomentumTensor;               // Cached SGD momentum scalar (beta1)
    
    // Adam-specific graph compilation  
    BOOL adamGraphCompiled;                                // Flag indicating Adam graph is compiled
    NSMutableArray* adamPrecompiledGradients;              // Adam pre-compiled gradient tensors
    NSMutableArray* adamPrecompiledUpdatedParams;          // Adam pre-compiled parameter updates
    NSMutableArray* adamPrecompiledUpdatedMomentum;        // Adam pre-compiled momentum updates  
    NSMutableArray* adamPrecompiledUpdatedVariance;        // Adam pre-compiled variance updates
    
    // RMSProp-specific graph compilation
    BOOL rmspropGraphCompiled;                             // Flag indicating RMSProp graph is compiled
    BOOL rmspropStateInitialized;                          // Flag indicating RMSProp state is ready
    NSMutableArray* rmspropPrecompiledGradients;           // RMSProp pre-compiled gradient tensors
    NSMutableArray* rmspropPrecompiledUpdatedParams;       // RMSProp pre-compiled parameter updates
    NSMutableArray* rmspropPrecompiledUpdatedMomentum;     // RMSProp pre-compiled momentum updates (if momentum > 0)
    NSMutableArray* rmspropPrecompiledUpdatedSquaredGrad;  // RMSProp pre-compiled squared gradient averages
    NSMutableArray* rmspropPrecompiledUpdatedGradAvg;      // RMSProp pre-compiled gradient averages (if centered)
    
    // RMSProp-specific state arrays
    NSMutableArray* squaredGradPlaceholders;               // Squared gradient average state for each parameter
    NSMutableArray* squaredGradVariables;                  // MPSGraph variables for squared gradient state
    NSMutableArray* squaredGradBuffers;                    // Metal buffers for squared gradient state
    NSMutableArray* gradAvgPlaceholders;                   // Gradient average state for each parameter (centered RMSProp)
    NSMutableArray* gradAvgVariables;                      // MPSGraph variables for gradient average state
    NSMutableArray* gradAvgBuffers;                        // Metal buffers for gradient average state
    
    // RMSProp-specific cached scalars
    MPSGraphTensor* cachedAlphaTensor;                     // Cached alpha (smoothing constant) scalar
    MPSGraphTensor* cachedOneMinusAlphaTensor;             // Cached (1 - alpha) scalar
    MPSGraphTensor* rmspropCachedMomentumTensor;           // Cached RMSProp momentum scalar
    MPSGraphTensor* rmspropCachedEpsilonTensor;            // Cached RMSProp epsilon scalar
    BOOL rmspropScalarsCached;                             // Flag indicating RMSProp scalars are cached
    
    // Model configuration for dynamic dimensions
    model_config_t model_config;                             // Model architecture configuration
} training_engine_t;