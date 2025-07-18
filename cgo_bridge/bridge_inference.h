#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "bridge_types.h"

// Inference-specific configuration
typedef struct {
    float precision_threshold;    // Float16 conversion threshold
    int max_batch_size;          // Maximum supported batch size
    int optimization_level;      // 0=Conservative, 1=Balanced, 2=Aggressive
    int memory_strategy;         // 0=Minimal, 1=Balanced, 2=PreAllocated
    int enable_telemetry;        // 0=disabled, 1=enabled
    int cache_compiled_graphs;   // 0=disabled, 1=enabled
} inference_config_t;

// Inference buffer pool for GPU-resident memory management
typedef struct {
    void** input_buffers;        // Pre-allocated input buffers by size
    void** output_buffers;       // Pre-allocated output buffers by size
    void** intermediate_buffers; // Intermediate computation buffers
    void** parameter_buffers;    // Model parameter buffers (read-only)
    
    int* buffer_sizes;           // Buffer sizes for each category
    int* buffer_counts;          // Number of buffers in each category
    int* buffer_usage;           // Current usage count per buffer
    
    size_t total_allocated;      // Total GPU memory allocated
    size_t peak_usage;           // Peak memory usage recorded
    size_t current_usage;        // Current memory usage
} inference_buffer_pool_t;

// Performance telemetry for inference operations
typedef struct {
    uint64_t total_inferences;   // Total inference calls
    double total_time_ms;        // Total inference time in milliseconds
    double avg_latency_ms;       // Average inference latency
    double peak_throughput;      // Peak throughput (inferences/second)
    size_t peak_memory_usage;    // Peak GPU memory usage
    uint64_t cache_hits;         // Graph compilation cache hits
    uint64_t cache_misses;       // Graph compilation cache misses
} inference_telemetry_t;

// Core inference engine structure - GPU-resident with minimal CPU interaction
typedef struct MPSInferenceEngine {
    // Core Metal/MPSGraph components
    void* device;                    // id<MTLDevice> - GPU device
    void* command_queue;             // id<MTLCommandQueue> - command submission
    void* graph;                     // MPSGraph* - computation graph
    void* objc_engine;               // MPSInferenceEngineObjC* - Objective-C engine wrapper
    
    // Model specification
    layer_spec_c_t* layers;          // Layer specifications
    int layer_count;                 // Number of layers
    float** parameters;              // Model parameters
    int* parameter_sizes;            // Parameter tensor sizes
    int parameter_count;             // Number of parameter tensors
    
    // Inference-specific optimizations
    void* input_placeholders;       // NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>*
    void* output_tensors;            // NSMutableArray<MPSGraphTensor*>*
    void* tensor_name_map;           // NSMutableDictionary<NSString*, MPSGraphTensor*>*
    
    // GPU-resident buffer management
    inference_buffer_pool_t* buffer_pool;
    void* parameter_buffer;          // Single large GPU buffer for all parameters
    size_t* parameter_offsets;       // Byte offsets for each parameter in the buffer
    
    // Performance monitoring
    inference_telemetry_t* telemetry;
    inference_config_t config;
    
    // Reference counting for deterministic cleanup
    int reference_count;
    int is_compiled;                 // Graph compilation status
} MPSInferenceEngine;

// Inference result structure
typedef struct {
    float* predictions;              // Output predictions (GPU -> CPU copied)
    int* output_shape;               // Shape of output tensor
    int output_shape_len;            // Number of dimensions in output shape
    float confidence_score;          // Maximum confidence/probability
    int predicted_class;             // Predicted class index (for classification)
    double inference_time_ms;        // Time taken for this inference
    size_t memory_used_bytes;        // GPU memory used for this inference
} inference_result_t;

#ifdef __cplusplus
extern "C" {
#endif

// Core inference engine functions
void* create_inference_engine_optimized(
    void* device,
    inference_config_t config,
    layer_spec_c_t* layers,
    int layer_count,
    float** parameters,
    int* parameter_sizes,
    int parameter_count
);

// Batch inference with single CGO call
int execute_inference_batch_optimized(
    uintptr_t engine,
    float* input_data,
    int* input_shape,
    int input_shape_len,
    float* output_data,
    int* output_shape,
    int* output_shape_len,
    int batch_size,
    inference_result_t* results
);

// Single inference call
int execute_inference_single_optimized(
    uintptr_t engine,
    float* input_data,
    int* input_shape,
    int input_shape_len,
    inference_result_t* result
);

// Memory and performance management
int preallocate_inference_buffers(void* engine, int max_batch_size);
void get_inference_telemetry(void* engine, inference_telemetry_t* telemetry);
void reset_inference_telemetry(void* engine);

// Resource management
void retain_inference_engine(void* engine);
void release_inference_engine(void* engine);
void destroy_inference_engine_optimized(void* engine);

#ifdef __cplusplus
}
#endif