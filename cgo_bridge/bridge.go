package cgo_bridge

/*
#include <stdlib.h>
#include <stdint.h>

// Training configuration
typedef struct {
    float learning_rate;
    float beta1;             // Adam momentum decay / RMSProp momentum (if > 0)
    float beta2;             // Adam variance decay (unused for RMSProp)
    float weight_decay;      // L2 regularization
    float epsilon;           // Numerical stability constant
    float alpha;             // RMSProp smoothing constant (typically 0.99)
    float momentum;          // RMSProp momentum coefficient (typically 0.0)
    int centered;            // RMSProp centered variant flag (0=false, 1=true)
    int optimizer_type;      // 0 = SGD, 1 = Adam, 2 = RMSProp, 3 = L-BFGS
    int problem_type;        // 0 = Classification, 1 = Regression
    int loss_function;       // 0 = CrossEntropy, 1 = SparseCrossEntropy, 2 = MSE, 3 = MAE, 4 = Huber
} training_config_t;

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

// Forward declarations for CGO functions
uintptr_t create_metal_device();
void destroy_metal_device(uintptr_t device);
uintptr_t create_training_engine(uintptr_t device, training_config_t* config);
uintptr_t create_training_engine_constant_weights(uintptr_t device, training_config_t* config);
int execute_training_step(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* loss_out
);
uintptr_t allocate_metal_buffer(uintptr_t device, int size, int device_type);
void deallocate_metal_buffer(uintptr_t buffer);
void destroy_training_engine(uintptr_t engine);

// SGD optimizer functions
int execute_sgd_step_mpsgraph(
    uintptr_t device,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float momentum,
    float weight_decay,
    int nesterov,
    int step_count
);

// Adam optimizer functions

// MPSGraph-based Adam optimizer for optimal GPU performance
int execute_adam_step_mpsgraph(
    uintptr_t device,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* variance_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step_count
);
// RESOURCE LEAK FIX: Command buffer pooled version for Metal resource management
int execute_adam_step_mpsgraph_pooled(
    uintptr_t device,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* variance_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step_count,
    uintptr_t command_buffer
);


// Utility functions for buffer operations
int zero_metal_buffer(uintptr_t device, uintptr_t buffer, int size);

// MPSGraph-based buffer zeroing for GPU-only buffers
int zero_metal_buffer_mpsgraph(uintptr_t device, uintptr_t buffer, int size);

// Data transfer functions
int copy_data_to_metal_buffer(uintptr_t buffer, void* data, int size);
int copy_float32_array_to_metal_buffer(uintptr_t buffer, float* data, int num_elements);
int copy_int32_array_to_metal_buffer(uintptr_t buffer, int* data, int num_elements);
int copy_metal_buffer_to_float32_array(uintptr_t buffer, float* data, int num_elements);
int copy_metal_buffer_to_int32_array(uintptr_t buffer, int* data, int num_elements);
int convert_tensor_type(uintptr_t src_buffer, uintptr_t dst_buffer, int* shape, int num_dims, int src_type, int dst_type, uintptr_t device);

// MEMORY TRANSFER OPTIMIZATION: Direct Metal buffer operations for staging pool
int copy_buffer_to_buffer_async(uintptr_t src_buffer, uintptr_t dst_buffer, 
                                int src_offset, int dst_offset, int size,
                                uintptr_t command_queue);
int copy_buffer_to_buffer_sync(uintptr_t src_buffer, uintptr_t dst_buffer,
                               int src_offset, int dst_offset, int size);
int copy_data_to_staging_buffer(uintptr_t staging_buffer, void* data, int size);
int copy_staging_to_gpu_buffer_async(uintptr_t staging_buffer, uintptr_t gpu_buffer,
                                     int staging_offset, int gpu_offset, int size,
                                     uintptr_t command_queue);
int wait_for_buffer_copy_completion(uintptr_t command_queue);



// Dynamic inference using the same graph as training (forward pass only)
int execute_inference_dynamic(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* predictions_out,
    int batch_size,
    int num_classes
);

// Dynamic graph creation from model specification
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
    
    // Running statistics for layers like BatchNorm (non-learnable parameters)
    float* running_mean;     // Running mean data
    float* running_var;      // Running variance data
    int running_stats_size;  // Size of running statistics arrays
    int has_running_stats;   // Boolean flag indicating if running stats are available
} layer_spec_c_t;

uintptr_t create_training_engine_dynamic(
    uintptr_t device,
    training_config_t* config,
    layer_spec_c_t* layers,
    int num_layers,
    int* input_shape,
    int input_shape_len
);

int execute_training_step_dynamic(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float learning_rate,
    int batch_size,
    float* loss_out
);

int execute_training_step_dynamic_with_gradients(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    float learning_rate,
    int batch_size,
    float* loss_out
);

// RESOURCE LEAK FIX: Command buffer pooled version for Metal resource management
int execute_training_step_dynamic_with_gradients_pooled(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    int batch_size,
    uintptr_t command_buffer,
    float* loss_out
);

// SGD-specific pooled training function for optimal performance
int execute_training_step_sgd_pooled(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    float learning_rate,
    int batch_size,
    uintptr_t command_buffer,
    float* loss_out
);

// Command Buffer Management Functions for Resource Leak Prevention
uintptr_t create_command_queue(uintptr_t device);
void release_command_queue(uintptr_t command_queue);
uintptr_t create_command_buffer(uintptr_t command_queue);
void release_command_buffer(uintptr_t command_buffer);
int commit_command_buffer(uintptr_t command_buffer);
int wait_command_buffer_completion(uintptr_t command_buffer);
void setup_autorelease_pool();
void drain_autorelease_pool();

// Test function
void test_debug_output();

// Optimized inference execution with Metal buffers
int execute_inference_optimized_buffers(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* output_data,
    int* output_shape,
    int* output_shape_len,
    int batch_size,
    int num_classes
);

// RESOURCE LEAK FIX: Command buffer pool management functions for Metal level
uintptr_t get_command_buffer_from_pool(uintptr_t command_buffer);
void return_command_buffer_to_pool(uintptr_t command_buffer);

// RMSProp optimizer functions
int execute_rmsprop_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* squared_grad_avg_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* gradient_avg_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float alpha,
    float epsilon,
    float weight_decay,
    float momentum,
    _Bool centered,
    int step_count
);

// L-BFGS optimizer functions (using flattened arrays for CGO compatibility)
int execute_lbfgs_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* old_gradient_buffers,
    uintptr_t* search_dir_buffers,
    uintptr_t* s_vectors_flat,
    uintptr_t* y_vectors_flat,
    uintptr_t* rho_buffers,
    uintptr_t alpha_buffer,
    int num_weights,
    int* buffer_sizes,
    int history_size,
    int history_count,
    int history_index,
    float initial_step,
    float c1,
    float c2,
    int max_line_search,
    float current_loss,
    float prev_loss,
    float* step_size
);

int execute_lbfgs_step_mpsgraph_pooled(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* old_gradient_buffers,
    uintptr_t* search_dir_buffers,
    uintptr_t* s_vectors_flat,
    uintptr_t* y_vectors_flat,
    uintptr_t* rho_buffers,
    uintptr_t alpha_buffer,
    int num_weights,
    int* buffer_sizes,
    int history_size,
    int history_count,
    int history_index,
    float initial_step,
    float c1,
    float c2,
    int max_line_search,
    float current_loss,
    float prev_loss,
    uintptr_t command_buffer,
    float* step_size
);

// AdaGrad optimizer forward declarations
int execute_adagrad_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* squared_grad_avg_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float epsilon,
    float weight_decay
);

int execute_adagrad_step_mpsgraph_pooled(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* squared_grad_avg_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float epsilon,
    float weight_decay,
    uintptr_t command_buffer
);

// AdaDelta optimizer forward declarations
int execute_adadelta_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* squared_grad_avg_buffers,
    uintptr_t* squared_update_avg_buffers,
    int num_weights,
    int* buffer_sizes,
    float rho,
    float epsilon,
    float weight_decay
);

int execute_adadelta_step_mpsgraph_pooled(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* squared_grad_avg_buffers,
    uintptr_t* squared_update_avg_buffers,
    int num_weights,
    int* buffer_sizes,
    float rho,
    float epsilon,
    float weight_decay,
    uintptr_t command_buffer
);

int execute_nadam_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* variance_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step_count
);

// Dedicated inference engine configuration
typedef struct {
    float precision_threshold;    // Float16 conversion threshold
    int max_batch_size;          // Maximum supported batch size
    int optimization_level;      // 0=Conservative, 1=Balanced, 2=Aggressive
    int memory_strategy;         // 0=Minimal, 1=Balanced, 2=PreAllocated
    int enable_telemetry;        // 0=disabled, 1=enabled
    int cache_compiled_graphs;   // 0=disabled, 1=enabled
} inference_config_t;

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

// Core dedicated inference engine functions
uintptr_t create_inference_engine_optimized(
    uintptr_t device,
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
int preallocate_inference_buffers(uintptr_t engine, int max_batch_size);
void get_inference_telemetry(uintptr_t engine, inference_telemetry_t* telemetry);
void reset_inference_telemetry(uintptr_t engine);

// Resource management
void retain_inference_engine(uintptr_t engine);
void release_inference_engine(uintptr_t engine);
void destroy_inference_engine_optimized(uintptr_t engine);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// SetupMemoryBridge sets up bridge functions for memory package to avoid import cycles
// Call this from packages that need both cgo_bridge and memory functionality
func SetupMemoryBridge(setupFunc func(
	func(unsafe.Pointer, int) ([]float32, error),
	func(unsafe.Pointer, []float32) error,
	func(unsafe.Pointer, []int32) error,
)) {
	setupFunc(
		CopyMetalBufferToFloat32Array,
		CopyFloat32ArrayToMetalBuffer,
		CopyInt32ArrayToMetalBuffer,
	)
}

// SetupMemoryBridgeWithConvert sets up bridge functions including type conversion
func SetupMemoryBridgeWithConvert(setupFunc func(
	func(unsafe.Pointer, int) ([]float32, error),
	func(unsafe.Pointer, []float32) error,
	func(unsafe.Pointer, []int32) error,
	func(unsafe.Pointer, unsafe.Pointer, []int, int, int) error,
	func(unsafe.Pointer, unsafe.Pointer, int) error,
), getDeviceFunc func() unsafe.Pointer) {
	// Create wrapper for ConvertTensorType that matches the expected signature
	convertWrapper := func(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType int) error {
		device := getDeviceFunc()
		if device == nil {
			return fmt.Errorf("no device available for tensor conversion")
		}
		return ConvertTensorType(srcBuffer, dstBuffer, shape, srcType, dstType, device)
	}
	
	setupFunc(
		CopyMetalBufferToFloat32Array,
		CopyFloat32ArrayToMetalBuffer,
		CopyInt32ArrayToMetalBuffer,
		convertWrapper,
		CopyTensorBufferSync, // GPU-resident tensor copying
	)
}

// OptimizerType represents the type of optimizer
type OptimizerType int

const (
	SGD OptimizerType = iota
	Adam
	RMSProp
	LBFGS
	AdaGrad
	AdaDelta
	Nadam
)

// TrainingConfig holds training configuration
type TrainingConfig struct {
	LearningRate  float32
	Beta1         float32         // Adam momentum decay (or RMSProp momentum if > 0)
	Beta2         float32         // Adam variance decay (unused for RMSProp)
	WeightDecay   float32
	Epsilon       float32
	Alpha         float32         // RMSProp smoothing constant (typically 0.99)
	Momentum      float32         // RMSProp momentum (typically 0.0 or 0.9)
	Centered      bool            // RMSProp centered variant
	OptimizerType OptimizerType
	ProblemType   int             // 0 = Classification, 1 = Regression
	LossFunction  int             // 0 = CrossEntropy, 1 = SparseCrossEntropy, 2 = MSE, 3 = MAE, 4 = Huber
}

// ModelConfig holds model architecture configuration for dynamic dimensions
type ModelConfig struct {
	// Input configuration
	BatchSize      int
	InputChannels  int
	InputHeight    int
	InputWidth     int
	
	// Convolution layer outputs
	Conv1OutChannels int
	Conv1OutHeight   int
	Conv1OutWidth    int
	
	Conv2OutChannels int
	Conv2OutHeight   int
	Conv2OutWidth    int
	
	Conv3OutChannels int
	Conv3OutHeight   int
	Conv3OutWidth    int
	
	// Fully connected layer dimensions
	FC1InputSize  int // Flattened conv output size
	FC1OutputSize int // Hidden layer size
	FC2OutputSize int // Number of classes
	
	// Convolution parameters
	Conv1KernelSize int
	Conv1Stride     int
	Conv2KernelSize int
	Conv2Stride     int
	Conv3KernelSize int
	Conv3Stride     int
}

// InferenceConfig holds configuration for inference-only engines
type InferenceConfig struct {
	// Model configuration
	UseDynamicEngine bool              // Use dynamic graph engine
	BatchNormInferenceMode bool       // Use batch norm in inference mode
	
	// Input configuration
	InputShape      []int32           // Input tensor shape
	InputShapeLen   int32            // Length of input shape array
	
	// Layer specifications for dynamic models
	LayerSpecs      []LayerSpecC     // Layer specifications
	LayerSpecsLen   int32           // Number of layer specs
	
	// Problem type and loss function (CRITICAL FIX for regression inference)
	ProblemType     int              // 0 = Classification, 1 = Regression
	LossFunction    int              // 0 = CrossEntropy, 1 = SparseCrossEntropy, 2 = MSE, 3 = MAE, 4 = Huber
	
	// Performance settings
	UseCommandPooling bool           // Enable command buffer pooling
	OptimizeForSingleBatch bool     // Optimize for batch size 1
}

// DeviceType maps to our memory package
type DeviceType int

const (
	CPU DeviceType = iota
	GPU
	PersistentGPU
)

// CreateMetalDevice creates a Metal device
func CreateMetalDevice() (unsafe.Pointer, error) {
	device := C.create_metal_device()
	if device == 0 {
		return nil, fmt.Errorf("failed to create Metal device")
	}
	return unsafe.Pointer(uintptr(device)), nil
}

// DestroyMetalDevice destroys a Metal device
func DestroyMetalDevice(device unsafe.Pointer) {
	if device != nil {
		C.destroy_metal_device(C.uintptr_t(uintptr(device)))
	}
}

// CreateTrainingEngine creates a training engine
func CreateTrainingEngine(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error) {
	cConfig := C.training_config_t{
		learning_rate:  C.float(config.LearningRate),
		beta1:         C.float(config.Beta1),
		beta2:         C.float(config.Beta2),
		weight_decay:  C.float(config.WeightDecay),
		epsilon:       C.float(config.Epsilon),
		alpha:         C.float(config.Alpha),
		momentum:      C.float(config.Momentum),
		centered:      C.int(func() int { if config.Centered { return 1 } else { return 0 } }()),
		optimizer_type: C.int(config.OptimizerType),
		problem_type:   C.int(config.ProblemType),
		loss_function:  C.int(config.LossFunction),
	}
	
	engine := C.create_training_engine(C.uintptr_t(uintptr(device)), &cConfig)
	if engine == 0 {
		return nil, fmt.Errorf("failed to create training engine")
	}
	
	return unsafe.Pointer(uintptr(engine)), nil
}

// CreateTrainingEngineConstantWeights creates a training engine with constant weights to avoid MPSGraph assertion
func CreateTrainingEngineConstantWeights(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error) {
	cConfig := C.training_config_t{
		learning_rate:  C.float(config.LearningRate),
		beta1:         C.float(config.Beta1),
		beta2:         C.float(config.Beta2),
		weight_decay:  C.float(config.WeightDecay),
		epsilon:       C.float(config.Epsilon),
		alpha:         C.float(config.Alpha),
		momentum:      C.float(config.Momentum),
		centered:      C.int(func() int { if config.Centered { return 1 } else { return 0 } }()),
		optimizer_type: C.int(config.OptimizerType),
		problem_type:   C.int(config.ProblemType),
		loss_function:  C.int(config.LossFunction),
	}
	
	engine := C.create_training_engine_constant_weights(C.uintptr_t(uintptr(device)), &cConfig)
	if engine == 0 {
		return nil, fmt.Errorf("failed to create constant weights training engine")
	}
	
	return unsafe.Pointer(uintptr(engine)), nil
}


// ExecuteTrainingStep executes a complete training step
func ExecuteTrainingStep(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
) (float32, error) {
	// Convert weight buffers to C array
	cWeightBuffers := make([]C.uintptr_t, len(weightBuffers))
	for i, buf := range weightBuffers {
		cWeightBuffers[i] = C.uintptr_t(uintptr(buf))
	}
	
	var lossOut C.float
	result := C.execute_training_step(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		&cWeightBuffers[0],
		C.int(len(weightBuffers)),
		&lossOut,
	)
	
	if result != 0 {
		return 0, fmt.Errorf("training step failed with error code: %d", result)
	}
	
	return float32(lossOut), nil
}



// AllocateMetalBuffer allocates a Metal buffer
func AllocateMetalBuffer(device unsafe.Pointer, size int, deviceType DeviceType) (unsafe.Pointer, error) {
	buffer := C.allocate_metal_buffer(
		C.uintptr_t(uintptr(device)),
		C.int(size),
		C.int(deviceType),
	)
	
	if buffer == 0 {
		return nil, fmt.Errorf("failed to allocate Metal buffer of size %d", size)
	}
	
	return unsafe.Pointer(uintptr(buffer)), nil
}

// DeallocateMetalBuffer deallocates a Metal buffer
func DeallocateMetalBuffer(buffer unsafe.Pointer) {
	if buffer != nil {
		C.deallocate_metal_buffer(C.uintptr_t(uintptr(buffer)))
	}
}

// DestroyTrainingEngine destroys a training engine with comprehensive cleanup
// This function properly handles all optimizer states, cached resources,
// and pre-compiled operations with robust resource management
func DestroyTrainingEngine(engine unsafe.Pointer) {
	if engine != nil {
		C.destroy_training_engine(C.uintptr_t(uintptr(engine)))
	}
}

// CreateInferenceEngine creates an inference-only engine optimized for forward pass
func CreateInferenceEngine(device unsafe.Pointer, config InferenceConfig) (unsafe.Pointer, error) {
	// For now, use the existing training engine but configure it for inference only
	// In a full implementation, this would create a dedicated inference engine
	
	// Convert to training config for compatibility (will be optimized in C++ later)
	trainingConfig := TrainingConfig{
		LearningRate:  0.001, // Not used for inference
		Beta1:         0.9,   // Not used for inference
		Beta2:         0.999, // Not used for inference
		WeightDecay:   0.0,   // Not used for inference
		Epsilon:       1e-8,  // Not used for inference
		Alpha:         0.99,  // Not used for inference
		Momentum:      0.0,   // Not used for inference
		Centered:      false, // Not used for inference
		OptimizerType: SGD,   // Not used for inference
		
		// CRITICAL FIX: Set problem type and loss function for correct inference behavior
		ProblemType:   config.ProblemType,  // Pass through the problem type (0=Classification, 1=Regression)
		LossFunction:  config.LossFunction, // Pass through the loss function (0=CrossEntropy, 1=SparseCrossEntropy, 2=MSE, etc.)
	}
	
	if config.UseDynamicEngine {
		// Create dynamic training engine configured for inference
		// Convert input shape from int32 to int
		inputShape := make([]int, len(config.InputShape))
		for i, dim := range config.InputShape {
			inputShape[i] = int(dim)
		}
		
		engine, err := CreateTrainingEngineDynamic(
			device,
			trainingConfig,
			config.LayerSpecs,
			inputShape, // Use actual input shape, not zeros
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create dynamic inference engine: %v", err)
		}
		return engine, nil
	} else {
		return nil, fmt.Errorf("dynamic engine is required for inference - hybrid engine has been removed")
	}
}

// DestroyInferenceEngine destroys an inference engine
func DestroyInferenceEngine(engine unsafe.Pointer) {
	// For now, use the same cleanup as training engine
	DestroyTrainingEngine(engine)
}

// ExecuteInferenceOnly performs forward-only inference without loss computation
func ExecuteInferenceOnly(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	batchSize int,
	numClasses int,
	isDynamic bool,
	batchNormInferenceMode bool,
) (*InferenceResult, error) {
	// fmt.Printf("DEBUG: ExecuteInferenceOnly called with batchSize=%d, numClasses=%d, numWeights=%d\n", 
	// 	batchSize, numClasses, len(weightBuffers))
	
	// Test debug output
	C.test_debug_output()
	
	// Validate inputs
	if engine == nil || inputBuffer == nil {
		return nil, fmt.Errorf("engine or input buffer is nil")
	}
	
	if batchSize <= 0 || numClasses <= 0 {
		return nil, fmt.Errorf("invalid batch size (%d) or num classes (%d)", batchSize, numClasses)
	}

	// Allocate output buffer for predictions
	predictionsSize := batchSize * numClasses
	predictions := make([]float32, predictionsSize)

	// Convert Go slice to C array for weight buffers
	var cWeightBuffers *C.uintptr_t
	if len(weightBuffers) > 0 {
		cWeights := make([]C.uintptr_t, len(weightBuffers))
		for i, buf := range weightBuffers {
			cWeights[i] = C.uintptr_t(uintptr(buf))
		}
		cWeightBuffers = &cWeights[0]
	}
	
	// CRITICAL FIX: Since CreateInferenceEngine actually creates a training engine,
	// we need to use the training engine's inference function instead of the inference engine's function
	result := C.execute_inference_dynamic(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		cWeightBuffers,
		C.int(len(weightBuffers)),
		(*C.float)(unsafe.Pointer(&predictions[0])),
		C.int(batchSize),
		C.int(numClasses),
	)

	if result != 0 {
		return nil, fmt.Errorf("inference execution failed with error code: %d", result)
	}

	// Since we're using execute_inference_dynamic, we know the output shape is [batchSize, numClasses]
	goOutputShape := []int{batchSize, numClasses}

	// Build inference result
	inferenceResult := &InferenceResult{
		Predictions: predictions,
		BatchSize:   batchSize,
		OutputShape: goOutputShape,
		Success:     true,
	}

	return inferenceResult, nil
}

// BuildInferenceGraph builds an optimized inference graph
func BuildInferenceGraph(
	engine unsafe.Pointer,
	inputShape []int,
	inputShapeLen int32,
	batchNormInferenceMode bool,
) error {
	// For now, assume the graph is already built by the engine creation
	// In a full implementation, this would call a C function to build inference-optimized graph
	return nil
}

// ExecuteSGDStepMPSGraph executes a single SGD optimization step using MPSGraph for optimal GPU performance
func ExecuteSGDStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	momentum float32,
	weightDecay float32,
	nesterov bool,
	stepCount int,
) error {
	if len(weightBuffers) != len(gradientBuffers) ||
		len(weightBuffers) != len(bufferSizes) {
		return fmt.Errorf("weight, gradient, and buffer size arrays must have the same length")
	}

	// For SGD with momentum, check momentum buffers
	if momentum > 0 && len(weightBuffers) != len(momentumBuffers) {
		return fmt.Errorf("momentum buffers must have same length as weight buffers when momentum > 0")
	}

	numWeights := len(weightBuffers)

	// Convert Go slices to C arrays
	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cMomentumBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		if momentum > 0 && i < len(momentumBuffers) {
			cMomentumBuffers[i] = C.uintptr_t(uintptr(momentumBuffers[i]))
		} else {
			cMomentumBuffers[i] = 0 // NULL pointer for vanilla SGD
		}
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	// Get pointers to first elements or nil
	var cWeightPtr *C.uintptr_t
	var cGradientPtr *C.uintptr_t
	var cMomentumPtr *C.uintptr_t
	var cSizesPtr *C.int

	if numWeights > 0 {
		cWeightPtr = &cWeightBuffers[0]
		cGradientPtr = &cGradientBuffers[0]
		cMomentumPtr = &cMomentumBuffers[0]
		cSizesPtr = &cBufferSizes[0]
	}

	var nesterovInt int
	if nesterov {
		nesterovInt = 1
	} else {
		nesterovInt = 0
	}

	result := C.execute_sgd_step_mpsgraph(
		C.uintptr_t(uintptr(device)),
		cWeightPtr,
		cGradientPtr,
		cMomentumPtr,
		C.int(numWeights),
		cSizesPtr,
		C.float(learningRate),
		C.float(momentum),
		C.float(weightDecay),
		C.int(nesterovInt),
		C.int(stepCount),
	)

	if result != 0 {
		return fmt.Errorf("SGD step execution failed with error code: %d", result)
	}

	return nil
}

// ExecuteAdamStepMPSGraph executes a single Adam optimization step using MPSGraph for optimal GPU performance
func ExecuteAdamStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	varianceBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	beta1 float32,
	beta2 float32,
	epsilon float32,
	weightDecay float32,
	stepCount int,
) error {
	if len(weightBuffers) != len(gradientBuffers) ||
		len(weightBuffers) != len(momentumBuffers) ||
		len(weightBuffers) != len(varianceBuffers) ||
		len(weightBuffers) != len(bufferSizes) {
		return fmt.Errorf("all buffer arrays must have the same length")
	}

	numWeights := len(weightBuffers)

	// Convert Go slices to C arrays
	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cMomentumBuffers := make([]C.uintptr_t, numWeights)
	cVarianceBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cMomentumBuffers[i] = C.uintptr_t(uintptr(momentumBuffers[i]))
		cVarianceBuffers[i] = C.uintptr_t(uintptr(varianceBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	result := C.execute_adam_step_mpsgraph(
		C.uintptr_t(uintptr(device)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		&cMomentumBuffers[0],
		&cVarianceBuffers[0],
		C.int(numWeights),
		&cBufferSizes[0],
		C.float(learningRate),
		C.float(beta1),
		C.float(beta2),
		C.float(epsilon),
		C.float(weightDecay),
		C.int(stepCount),
	)

	if result != 0 {
		return fmt.Errorf("Adam MPSGraph step failed with error code: %d", result)
	}

	return nil
}



// ZeroMetalBuffer zeros a Metal buffer (uses CPU for accessible buffers, MPSGraph for GPU-only)
func ZeroMetalBuffer(device unsafe.Pointer, buffer unsafe.Pointer, size int) error {
	result := C.zero_metal_buffer(
		C.uintptr_t(uintptr(device)),
		C.uintptr_t(uintptr(buffer)),
		C.int(size),
	)

	if result != 0 {
		return fmt.Errorf("failed to zero buffer with error code: %d", result)
	}

	return nil
}

// ZeroMetalBufferMPSGraph zeros a Metal buffer using MPSGraph (works for all buffer types)
func ZeroMetalBufferMPSGraph(device unsafe.Pointer, buffer unsafe.Pointer, size int) error {
	result := C.zero_metal_buffer_mpsgraph(
		C.uintptr_t(uintptr(device)),
		C.uintptr_t(uintptr(buffer)),
		C.int(size),
	)

	if result != 0 {
		return fmt.Errorf("failed to zero buffer using MPSGraph with error code: %d", result)
	}

	return nil
}

// CopyFloat32ArrayToMetalBuffer copies float32 array data to a Metal buffer
func CopyFloat32ArrayToMetalBuffer(buffer unsafe.Pointer, data []float32) error {
	if len(data) == 0 {
		return nil // Nothing to copy
	}

	result := C.copy_float32_array_to_metal_buffer(
		C.uintptr_t(uintptr(buffer)),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(len(data)),
	)

	if result != 0 {
		return fmt.Errorf("failed to copy float32 array to Metal buffer with error code: %d", result)
	}

	return nil
}

// CopyInt32ArrayToMetalBuffer copies int32 array data to a Metal buffer
func CopyInt32ArrayToMetalBuffer(buffer unsafe.Pointer, data []int32) error {
	if len(data) == 0 {
		return nil // Nothing to copy
	}

	result := C.copy_int32_array_to_metal_buffer(
		C.uintptr_t(uintptr(buffer)),
		(*C.int)(unsafe.Pointer(&data[0])),
		C.int(len(data)),
	)

	if result != 0 {
		return fmt.Errorf("failed to copy int32 array to Metal buffer with error code: %d", result)
	}

	return nil
}

// CopyMetalBufferToFloat32Array copies data from a Metal buffer to a float32 array
func CopyMetalBufferToFloat32Array(buffer unsafe.Pointer, numElements int) ([]float32, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("invalid number of elements: %d", numElements)
	}

	// Allocate Go slice
	data := make([]float32, numElements)

	result := C.copy_metal_buffer_to_float32_array(
		C.uintptr_t(uintptr(buffer)),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(numElements),
	)

	if result != 0 {
		return nil, fmt.Errorf("failed to copy Metal buffer to float32 array with error code: %d", result)
	}

	return data, nil
}

// CopyMetalBufferToInt32Array copies data from a Metal buffer to an int32 array
func CopyMetalBufferToInt32Array(buffer unsafe.Pointer, numElements int) ([]int32, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("invalid number of elements: %d", numElements)
	}

	// Allocate Go slice
	data := make([]int32, numElements)

	result := C.copy_metal_buffer_to_int32_array(
		C.uintptr_t(uintptr(buffer)),
		(*C.int)(unsafe.Pointer(&data[0])),
		C.int(numElements),
	)

	if result != 0 {
		return nil, fmt.Errorf("failed to copy Metal buffer to int32 array with error code: %d", result)
	}

	return data, nil
}

// ConvertTensorType converts a tensor from one data type to another on GPU
func ConvertTensorType(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType int, device unsafe.Pointer) error {
	if len(shape) == 0 {
		return fmt.Errorf("shape cannot be empty")
	}
	
	// Convert shape to C array
	cShape := make([]C.int, len(shape))
	for i, dim := range shape {
		cShape[i] = C.int(dim)
	}
	
	result := C.convert_tensor_type(
		C.uintptr_t(uintptr(srcBuffer)),
		C.uintptr_t(uintptr(dstBuffer)),
		&cShape[0],
		C.int(len(shape)),
		C.int(srcType),
		C.int(dstType),
		C.uintptr_t(uintptr(device)),
	)
	
	if result != 0 {
		return fmt.Errorf("failed to convert tensor type with error code: %d", result)
	}
	
	return nil
}

// CopyDataToMetalBuffer copies raw byte data to a Metal buffer
func CopyDataToMetalBuffer(buffer unsafe.Pointer, data []byte) error {
	if len(data) == 0 {
		return nil // Nothing to copy
	}

	result := C.copy_data_to_metal_buffer(
		C.uintptr_t(uintptr(buffer)),
		unsafe.Pointer(&data[0]),
		C.int(len(data)),
	)

	if result != 0 {
		return fmt.Errorf("failed to copy data to Metal buffer with error code: %d", result)
	}

	return nil
}

// InferenceResult contains model predictions and metadata
type InferenceResult struct {
	Predictions []float32 // Model output logits/probabilities [batch_size * num_classes]
	BatchSize   int       // Actual batch size processed
	OutputShape []int     // Shape of prediction tensor [batch_size, num_classes]
	Success     bool      // Inference execution status
}

// Dedicated inference engine types

// OptimizationLevel controls the level of optimizations applied to the inference engine
type OptimizationLevel int

const (
	Conservative OptimizationLevel = iota // Safe optimizations only
	Balanced                              // Standard inference optimizations
	Aggressive                            // Maximum performance optimizations
)

// MemoryStrategy controls how the inference engine manages GPU memory
type MemoryStrategy int

const (
	Minimal      MemoryStrategy = iota // Minimal memory usage
	BalancedMem                       // Balanced memory vs performance
	PreAllocated                      // Pre-allocate buffers for maximum performance
)

// DedicatedInferenceConfig holds configuration for the dedicated inference engine
type DedicatedInferenceConfig struct {
	PrecisionThreshold   float32           // Float16 conversion threshold
	MaxBatchSize         int               // Maximum supported batch size
	OptimizationLevel    OptimizationLevel // Optimization aggressiveness
	MemoryStrategy       MemoryStrategy    // Memory management approach
	EnableTelemetry      bool              // Enable performance monitoring
	CacheCompiledGraphs  bool              // Cache compiled MPSGraph executables
}

// DedicatedInferenceResult contains comprehensive inference results and metadata
type DedicatedInferenceResult struct {
	Predictions      []float32 // Output predictions (GPU -> CPU copied)
	OutputShape      []int     // Shape of output tensor
	ConfidenceScore  float32   // Maximum confidence/probability
	PredictedClass   int       // Predicted class index (for classification)
	InferenceTimeMs  float64   // Time taken for this inference
	MemoryUsedBytes  uint64    // GPU memory used for this inference
}

// InferenceTelemetry provides performance metrics for the inference engine
type InferenceTelemetry struct {
	TotalInferences  uint64  // Total inference calls
	TotalTimeMs      float64 // Total inference time in milliseconds
	AvgLatencyMs     float64 // Average inference latency
	PeakThroughput   float64 // Peak throughput (inferences/second)
	PeakMemoryUsage  uint64  // Peak GPU memory usage
	CacheHits        uint64  // Graph compilation cache hits
	CacheMisses      uint64  // Graph compilation cache misses
}

// DedicatedInferenceEngine represents a GPU-resident inference engine optimized for forward pass only
type DedicatedInferenceEngine struct {
	engine    unsafe.Pointer           // Pointer to C inference engine
	config    DedicatedInferenceConfig // Engine configuration
	isDestroyed bool                   // Track destruction state
}

// ExecuteInference performs forward-only pass and returns predictions
// Conforms to design requirements: single CGO call, GPU-resident, shared resources
func ExecuteInference(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	batchSize int,
	numClasses int,
	isDynamic bool,
) (*InferenceResult, error) {
	
	// Validate inputs
	if engine == nil || inputBuffer == nil {
		return nil, fmt.Errorf("engine or input buffer is nil")
	}
	
	if batchSize <= 0 || numClasses <= 0 {
		return nil, fmt.Errorf("invalid batch size (%d) or num classes (%d)", batchSize, numClasses)
	}

	// Allocate output buffer for predictions (CPU-accessible for result extraction)
	predictionsSize := batchSize * numClasses
	predictions := make([]float32, predictionsSize)

	// Convert Go slice to C array for weight buffers
	var cWeightBuffers *C.uintptr_t
	if len(weightBuffers) > 0 {
		cWeights := make([]C.uintptr_t, len(weightBuffers))
		for i, buf := range weightBuffers {
			cWeights[i] = C.uintptr_t(uintptr(buf))
		}
		cWeightBuffers = &cWeights[0]
	}

	// Single CGO call for complete inference (design compliant)
	// Only dynamic inference is supported now
	var result C.int
	if isDynamic {
		result = C.execute_inference_dynamic(
			C.uintptr_t(uintptr(engine)),
			C.uintptr_t(uintptr(inputBuffer)),
			cWeightBuffers,
			C.int(len(weightBuffers)),
			(*C.float)(unsafe.Pointer(&predictions[0])),
			C.int(batchSize),
			C.int(numClasses),
		)
	} else {
		return nil, fmt.Errorf("dynamic engine is required for inference - hybrid engine has been removed")
	}

	if result != 0 {
		return nil, fmt.Errorf("inference execution failed with error code: %d", result)
	}

	return &InferenceResult{
		Predictions: predictions,
		BatchSize:   batchSize,
		OutputShape: []int{batchSize, numClasses},
		Success:     true,
	}, nil
}

// LayerSpecC represents a C-compatible layer specification
type LayerSpecC struct {
	LayerType       int32
	Name            [64]byte  // Fixed-size array for C compatibility
	InputShape      [4]int32
	InputShapeLen   int32
	OutputShape     [4]int32
	OutputShapeLen  int32
	ParamInt        [8]int32
	ParamFloat      [8]float32
	ParamIntCount   int32
	ParamFloatCount int32
	// Running statistics for layers like BatchNorm (non-learnable parameters)
	RunningMean     []float32
	RunningVar      []float32
	RunningStatsSize int32
	HasRunningStats  int32 // Boolean flag (0 or 1)
}

// CreateTrainingEngineDynamic creates a training engine with dynamic graph from model specification
func CreateTrainingEngineDynamic(
	device unsafe.Pointer,
	config TrainingConfig,
	layerSpecs []LayerSpecC,
	inputShape []int,
) (unsafe.Pointer, error) {
	if len(layerSpecs) == 0 {
		return nil, fmt.Errorf("no layer specifications provided")
	}
	
	if len(inputShape) == 0 {
		return nil, fmt.Errorf("no input shape provided")
	}

	// Convert Go training config to C
	cConfig := C.training_config_t{
		learning_rate:  C.float(config.LearningRate),
		beta1:         C.float(config.Beta1),
		beta2:         C.float(config.Beta2),
		weight_decay:  C.float(config.WeightDecay),
		epsilon:       C.float(config.Epsilon),
		alpha:         C.float(config.Alpha),
		momentum:      C.float(config.Momentum),
		centered:      C.int(func() int { if config.Centered { return 1 } else { return 0 } }()),
		optimizer_type: C.int(config.OptimizerType),
		problem_type:   C.int(config.ProblemType),
		loss_function:  C.int(config.LossFunction),
	}

	// Convert Go layer specs to C array
	cLayerSpecs := make([]C.layer_spec_c_t, len(layerSpecs))
	for i, spec := range layerSpecs {
		cLayerSpecs[i] = C.layer_spec_c_t{
			layer_type:        C.int(spec.LayerType),
			input_shape_len:   C.int(spec.InputShapeLen),
			output_shape_len:  C.int(spec.OutputShapeLen),
			param_int_count:   C.int(spec.ParamIntCount),
			param_float_count: C.int(spec.ParamFloatCount),
			running_stats_size: C.int(spec.RunningStatsSize),
			has_running_stats:  C.int(spec.HasRunningStats),
		}

		// Copy name (null-terminated string)
		nameLen := 0
		for j, b := range spec.Name {
			if b == 0 {
				break
			}
			cLayerSpecs[i].name[j] = C.char(b)
			nameLen = j + 1
		}
		if nameLen < 64 {
			cLayerSpecs[i].name[nameLen] = 0 // Ensure null termination
		}

		// Copy arrays
		for j := 0; j < int(spec.InputShapeLen) && j < 4; j++ {
			cLayerSpecs[i].input_shape[j] = C.int(spec.InputShape[j])
		}
		for j := 0; j < int(spec.OutputShapeLen) && j < 4; j++ {
			cLayerSpecs[i].output_shape[j] = C.int(spec.OutputShape[j])
		}
		for j := 0; j < int(spec.ParamIntCount) && j < 8; j++ {
			cLayerSpecs[i].param_int[j] = C.int(spec.ParamInt[j])
		}
		for j := 0; j < int(spec.ParamFloatCount) && j < 8; j++ {
			cLayerSpecs[i].param_float[j] = C.float(spec.ParamFloat[j])
		}
		
		// ARCHITECTURAL FIX: Copy running statistics if available
		if spec.HasRunningStats == 1 && len(spec.RunningMean) > 0 && len(spec.RunningVar) > 0 {
			// Allocate C arrays for running statistics
			cLayerSpecs[i].running_mean = (*C.float)(C.calloc(C.size_t(len(spec.RunningMean)), C.sizeof_float))
			cLayerSpecs[i].running_var = (*C.float)(C.calloc(C.size_t(len(spec.RunningVar)), C.sizeof_float))
			
			// Copy running mean data
			for j := 0; j < len(spec.RunningMean); j++ {
				*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cLayerSpecs[i].running_mean)) + uintptr(j)*unsafe.Sizeof(C.float(0)))) = C.float(spec.RunningMean[j])
			}
			
			// Copy running variance data
			for j := 0; j < len(spec.RunningVar); j++ {
				*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cLayerSpecs[i].running_var)) + uintptr(j)*unsafe.Sizeof(C.float(0)))) = C.float(spec.RunningVar[j])
			}
		} else {
			cLayerSpecs[i].running_mean = nil
			cLayerSpecs[i].running_var = nil
		}
	}

	// Convert input shape to C array
	cInputShape := make([]C.int, len(inputShape))
	for i, dim := range inputShape {
		cInputShape[i] = C.int(dim)
	}

	// Call C function to create dynamic engine
	engine := C.create_training_engine_dynamic(
		C.uintptr_t(uintptr(device)),
		&cConfig,
		&cLayerSpecs[0],
		C.int(len(layerSpecs)),
		&cInputShape[0],
		C.int(len(inputShape)),
	)

	if engine == 0 {
		return nil, fmt.Errorf("failed to create dynamic training engine")
	}

	return unsafe.Pointer(uintptr(engine)), nil
}

// ExecuteTrainingStepDynamic executes a training step using dynamic engine with real loss computation
func ExecuteTrainingStepDynamic(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	learningRate float32,
	batchSize int,
) (float32, error) {
	// Convert weight buffers to C array
	var cWeightBuffers *C.uintptr_t
	if len(weightBuffers) > 0 {
		cWeights := make([]C.uintptr_t, len(weightBuffers))
		for i, buf := range weightBuffers {
			cWeights[i] = C.uintptr_t(uintptr(buf))
		}
		cWeightBuffers = &cWeights[0]
	}

	var lossOut C.float
	result := C.execute_training_step_dynamic(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		cWeightBuffers,
		C.int(len(weightBuffers)),
		C.float(learningRate),
		C.int(batchSize),
		&lossOut,
	)

	if result != 0 {
		return 0, fmt.Errorf("dynamic training step failed with error code: %d", result)
	}

	return float32(lossOut), nil
}

// ExecuteTrainingStepDynamicWithGradients executes a dynamic training step with real gradient computation
func ExecuteTrainingStepDynamicWithGradients(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	learningRate float32,
	batchSize int,
) (float32, error) {
	// Validate input parameters
	if len(weightBuffers) != len(gradientBuffers) {
		return 0, fmt.Errorf("weight buffer count (%d) must match gradient buffer count (%d)", 
			len(weightBuffers), len(gradientBuffers))
	}
	
	// Convert weight buffers to C array
	var cWeightBuffers *C.uintptr_t
	if len(weightBuffers) > 0 {
		cWeights := make([]C.uintptr_t, len(weightBuffers))
		for i, buf := range weightBuffers {
			cWeights[i] = C.uintptr_t(uintptr(buf))
		}
		cWeightBuffers = &cWeights[0]
	}
	
	// Convert gradient buffers to C array
	var cGradientBuffers *C.uintptr_t
	if len(gradientBuffers) > 0 {
		cGradients := make([]C.uintptr_t, len(gradientBuffers))
		for i, buf := range gradientBuffers {
			cGradients[i] = C.uintptr_t(uintptr(buf))
		}
		cGradientBuffers = &cGradients[0]
	}

	var lossOut C.float
	result := C.execute_training_step_dynamic_with_gradients(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		cWeightBuffers,
		cGradientBuffers,
		C.int(len(weightBuffers)),
		C.float(learningRate),
		C.int(batchSize),
		&lossOut,
	)

	if result != 0 {
		return 0, fmt.Errorf("dynamic gradient training step failed with error code: %d", result)
	}

	return float32(lossOut), nil
}

// ExecuteTrainingStepDynamicWithGradientsPooled executes a dynamic training step with pooled command buffers
func ExecuteTrainingStepDynamicWithGradientsPooled(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	batchSize int,
	commandPool unsafe.Pointer,
) (float32, error) {
	// Validate input parameters
	if len(weightBuffers) != len(gradientBuffers) {
		return 0, fmt.Errorf("weight buffer count (%d) must match gradient buffer count (%d)", 
			len(weightBuffers), len(gradientBuffers))
	}
	
	// Convert weight buffers to C array
	var cWeightBuffers *C.uintptr_t
	if len(weightBuffers) > 0 {
		cWeights := make([]C.uintptr_t, len(weightBuffers))
		for i, buf := range weightBuffers {
			cWeights[i] = C.uintptr_t(uintptr(buf))
		}
		cWeightBuffers = &cWeights[0]
	}
	
	// Convert gradient buffers to C array
	var cGradientBuffers *C.uintptr_t
	if len(gradientBuffers) > 0 {
		cGradients := make([]C.uintptr_t, len(gradientBuffers))
		for i, buf := range gradientBuffers {
			cGradients[i] = C.uintptr_t(uintptr(buf))
		}
		cGradientBuffers = &cGradients[0]
	}

	var lossOut C.float
	result := C.execute_training_step_dynamic_with_gradients_pooled(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		cWeightBuffers,
		cGradientBuffers,
		C.int(len(weightBuffers)),
		C.int(batchSize),
		C.uintptr_t(uintptr(commandPool)),
		&lossOut,
	)

	if result != 0 {
		return 0, fmt.Errorf("pooled dynamic gradient training step failed with error code: %d", result)
	}

	return float32(lossOut), nil
}

// ExecuteTrainingStepSGDPooled executes SGD training step with pooled command buffers for optimal performance
func ExecuteTrainingStepSGDPooled(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	learningRate float32,
	batchSize int,
	commandPool unsafe.Pointer,
) (float32, error) {
	if engine == nil {
		return 0, fmt.Errorf("engine cannot be nil")
	}
	
	if inputBuffer == nil || labelBuffer == nil {
		return 0, fmt.Errorf("input and label buffers cannot be nil")
	}
	
	if len(weightBuffers) != len(gradientBuffers) {
		return 0, fmt.Errorf("weight and gradient buffer counts must match")
	}
	
	// Convert weight buffers to C array
	var cWeightBuffers *C.uintptr_t
	if len(weightBuffers) > 0 {
		cWeights := make([]C.uintptr_t, len(weightBuffers))
		for i, buf := range weightBuffers {
			cWeights[i] = C.uintptr_t(uintptr(buf))
		}
		cWeightBuffers = &cWeights[0]
	}
	
	// Convert gradient buffers to C array
	var cGradientBuffers *C.uintptr_t
	if len(gradientBuffers) > 0 {
		cGradients := make([]C.uintptr_t, len(gradientBuffers))
		for i, buf := range gradientBuffers {
			cGradients[i] = C.uintptr_t(uintptr(buf))
		}
		cGradientBuffers = &cGradients[0]
	}

	var lossOut C.float
	result := C.execute_training_step_dynamic_with_gradients_pooled(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		cWeightBuffers,
		cGradientBuffers,
		C.int(len(weightBuffers)),
		C.int(batchSize),
		C.uintptr_t(uintptr(commandPool)),
		&lossOut,
	)

	if result != 0 {
		return 0, fmt.Errorf("SGD pooled training step failed with error code: %d", result)
	}

	return float32(lossOut), nil
}

// Command Buffer Management Functions

// CreateCommandQueue creates a Metal command queue for the given device
func CreateCommandQueue(device unsafe.Pointer) (unsafe.Pointer, error) {
	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}
	
	commandQueue := C.create_command_queue(C.uintptr_t(uintptr(device)))
	if commandQueue == 0 {
		return nil, fmt.Errorf("failed to create command queue")
	}
	
	return unsafe.Pointer(uintptr(commandQueue)), nil
}

// ReleaseCommandQueue releases a Metal command queue
func ReleaseCommandQueue(commandQueue unsafe.Pointer) {
	if commandQueue != nil {
		C.release_command_queue(C.uintptr_t(uintptr(commandQueue)))
	}
}

// DestroyCommandQueue is an alias for ReleaseCommandQueue for consistency
func DestroyCommandQueue(commandQueue unsafe.Pointer) {
	ReleaseCommandQueue(commandQueue)
}

// CreateCommandBuffer creates a Metal command buffer from the given command queue
func CreateCommandBuffer(commandQueue unsafe.Pointer) (unsafe.Pointer, error) {
	if commandQueue == nil {
		return nil, fmt.Errorf("command queue cannot be nil")
	}
	
	commandBuffer := C.create_command_buffer(C.uintptr_t(uintptr(commandQueue)))
	if commandBuffer == 0 {
		return nil, fmt.Errorf("failed to create command buffer")
	}
	
	return unsafe.Pointer(uintptr(commandBuffer)), nil
}

// ReleaseCommandBuffer releases a Metal command buffer
func ReleaseCommandBuffer(commandBuffer unsafe.Pointer) {
	if commandBuffer != nil {
		C.release_command_buffer(C.uintptr_t(uintptr(commandBuffer)))
	}
}

// CommitCommandBuffer commits a command buffer for execution
func CommitCommandBuffer(commandBuffer unsafe.Pointer) error {
	if commandBuffer == nil {
		return fmt.Errorf("command buffer cannot be nil")
	}
	
	result := C.commit_command_buffer(C.uintptr_t(uintptr(commandBuffer)))
	if result != 0 {
		return fmt.Errorf("failed to commit command buffer with error code: %d", result)
	}
	
	return nil
}

// WaitCommandBufferCompletion waits for a command buffer to complete execution
func WaitCommandBufferCompletion(commandBuffer unsafe.Pointer) error {
	if commandBuffer == nil {
		return fmt.Errorf("command buffer cannot be nil")
	}
	
	result := C.wait_command_buffer_completion(C.uintptr_t(uintptr(commandBuffer)))
	if result != 0 {
		return fmt.Errorf("command buffer completion failed with error code: %d", result)
	}
	
	return nil
}

// SetupAutoreleasePool sets up an autorelease pool for Metal resource management
func SetupAutoreleasePool() {
	C.setup_autorelease_pool()
}

// DrainAutoreleasePool drains the autorelease pool to release Metal resources
func DrainAutoreleasePool() {
	C.drain_autorelease_pool()
}

// RESOURCE LEAK FIX: Command buffer pooled training functions


// GetCommandBufferFromPool gets a command buffer from the pool (Metal level interface)
func GetCommandBufferFromPool(commandPool unsafe.Pointer) (unsafe.Pointer, error) {
	if commandPool == nil {
		return nil, fmt.Errorf("command pool cannot be nil")
	}
	
	result := C.get_command_buffer_from_pool(C.uintptr_t(uintptr(commandPool)))
	if result == 0 {
		return nil, fmt.Errorf("failed to get command buffer from pool")
	}
	
	return unsafe.Pointer(uintptr(result)), nil
}

// ReturnCommandBufferToPool returns a command buffer to the pool (Metal level interface)
func ReturnCommandBufferToPool(commandBuffer unsafe.Pointer) {
	if commandBuffer == nil {
		return
	}
	
	C.return_command_buffer_to_pool(
		C.uintptr_t(uintptr(commandBuffer)),
	)
}


// ExecuteAdamStepMPSGraphPooled performs Adam optimization with pooled command buffers
// RESOURCE LEAK FIX: Uses command buffer pooling to prevent Metal resource accumulation
func ExecuteAdamStepMPSGraphPooled(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	varianceBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	beta1 float32,
	beta2 float32,
	epsilon float32,
	weightDecay float32,
	stepCount int,
	commandPool unsafe.Pointer,
) error {
	if device == nil || commandPool == nil {
		return fmt.Errorf("device or command pool is nil")
	}
	
	numWeights := len(weightBuffers)
	if len(gradientBuffers) != numWeights || len(momentumBuffers) != numWeights || 
	   len(varianceBuffers) != numWeights || len(bufferSizes) != numWeights {
		return fmt.Errorf("buffer count mismatch")
	}
	
	// Convert to C arrays
	weightBufPtrs := make([]C.uintptr_t, numWeights)
	gradBufPtrs := make([]C.uintptr_t, numWeights)
	momentumBufPtrs := make([]C.uintptr_t, numWeights)
	varianceBufPtrs := make([]C.uintptr_t, numWeights)
	bufSizes := make([]C.int, numWeights)
	
	for i := 0; i < numWeights; i++ {
		if weightBuffers[i] == nil || gradientBuffers[i] == nil ||
		   momentumBuffers[i] == nil || varianceBuffers[i] == nil {
			return fmt.Errorf("buffer %d is nil", i)
		}
		
		weightBufPtrs[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		gradBufPtrs[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		momentumBufPtrs[i] = C.uintptr_t(uintptr(momentumBuffers[i]))
		varianceBufPtrs[i] = C.uintptr_t(uintptr(varianceBuffers[i]))
		bufSizes[i] = C.int(bufferSizes[i])
	}
	
	result := C.execute_adam_step_mpsgraph_pooled(
		C.uintptr_t(uintptr(device)),
		&weightBufPtrs[0],
		&gradBufPtrs[0],
		&momentumBufPtrs[0],
		&varianceBufPtrs[0],
		C.int(numWeights),
		&bufSizes[0],
		C.float(learningRate),
		C.float(beta1),
		C.float(beta2),
		C.float(epsilon),
		C.float(weightDecay),
		C.int(stepCount),
		C.uintptr_t(uintptr(commandPool)),
	)
	
	if result != 0 {
		return fmt.Errorf("pooled Adam step failed with code: %d", result)
	}
	
	return nil
}

// ExecuteRMSPropStepMPSGraph executes a single RMSProp optimization step using MPSGraph for optimal GPU performance
func ExecuteRMSPropStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	gradientAvgBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	alpha float32,
	epsilon float32,
	weightDecay float32,
	momentum float32,
	centered bool,
	stepCount int,
) error {
	if len(weightBuffers) != len(gradientBuffers) ||
		len(weightBuffers) != len(squaredGradAvgBuffers) ||
		len(weightBuffers) != len(bufferSizes) {
		return fmt.Errorf("weight, gradient, squared gradient average, and buffer size arrays must have the same length")
	}

	numWeights := len(weightBuffers)

	// Convert Go slices to C arrays
	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cSquaredGradAvgBuffers := make([]C.uintptr_t, numWeights)
	cMomentumBuffers := make([]C.uintptr_t, numWeights)
	cGradientAvgBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cSquaredGradAvgBuffers[i] = C.uintptr_t(uintptr(squaredGradAvgBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
		
		// Optional buffers
		if momentum > 0.0 && i < len(momentumBuffers) && momentumBuffers[i] != nil {
			cMomentumBuffers[i] = C.uintptr_t(uintptr(momentumBuffers[i]))
		} else {
			cMomentumBuffers[i] = 0
		}
		
		if centered && i < len(gradientAvgBuffers) && gradientAvgBuffers[i] != nil {
			cGradientAvgBuffers[i] = C.uintptr_t(uintptr(gradientAvgBuffers[i]))
		} else {
			cGradientAvgBuffers[i] = 0
		}
	}

	result := C.execute_rmsprop_step_mpsgraph(
		C.uintptr_t(uintptr(device)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		&cSquaredGradAvgBuffers[0],
		&cMomentumBuffers[0],
		&cGradientAvgBuffers[0],
		C.int(numWeights),
		&cBufferSizes[0],
		C.float(learningRate),
		C.float(alpha),
		C.float(epsilon),
		C.float(weightDecay),
		C.float(momentum),
		C._Bool(centered),
		C.int(stepCount),
	)

	if result != 0 {
		return fmt.Errorf("RMSProp MPSGraph step failed with error code: %d", result)
	}

	return nil
}

// ExecuteLBFGSStepMPSGraph executes a single L-BFGS optimization step using MPSGraph for optimal GPU performance
func ExecuteLBFGSStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	oldGradientBuffers []unsafe.Pointer,
	searchDirBuffers []unsafe.Pointer,
	sVectors [][]unsafe.Pointer,
	yVectors [][]unsafe.Pointer,
	rhoBuffers []unsafe.Pointer,
	alphaBuffer unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	historySize int,
	historyCount int,
	historyIndex int,
	initialStep float32,
	c1 float32,
	c2 float32,
	maxLineSearch int,
	currentLoss float32,
	prevLoss float32,
	commandPool unsafe.Pointer,
	usePooling bool,
) (float32, error) {
	// Validate inputs
	if len(weightBuffers) != numWeights ||
		len(gradientBuffers) != numWeights ||
		len(oldGradientBuffers) != numWeights ||
		len(searchDirBuffers) != numWeights ||
		len(bufferSizes) != numWeights {
		return 0, fmt.Errorf("buffer arrays must have length %d", numWeights)
	}

	// Convert Go slices to C arrays
	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cOldGradientBuffers := make([]C.uintptr_t, numWeights)
	cSearchDirBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cOldGradientBuffers[i] = C.uintptr_t(uintptr(oldGradientBuffers[i]))
		cSearchDirBuffers[i] = C.uintptr_t(uintptr(searchDirBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	// Convert history vectors to flattened C arrays to avoid CGO pointer issues
	// Layout: [hist0_weight0, hist0_weight1, ..., hist1_weight0, hist1_weight1, ...]
	cSVectorsFlat := make([]C.uintptr_t, historySize*numWeights)
	cYVectorsFlat := make([]C.uintptr_t, historySize*numWeights)
	
	for h := 0; h < historySize; h++ {
		for w := 0; w < numWeights; w++ {
			idx := h*numWeights + w
			cSVectorsFlat[idx] = C.uintptr_t(uintptr(sVectors[h][w]))
			cYVectorsFlat[idx] = C.uintptr_t(uintptr(yVectors[h][w]))
		}
	}
	cRhoBuffers := make([]C.uintptr_t, historySize)

	for h := 0; h < historySize; h++ {
		cRhoBuffers[h] = C.uintptr_t(uintptr(rhoBuffers[h]))
	}

	var stepSize C.float = 0.0
	var result C.int

	if usePooling && commandPool != nil {
		result = C.execute_lbfgs_step_mpsgraph_pooled(
			C.uintptr_t(uintptr(device)),
			&cWeightBuffers[0],
			&cGradientBuffers[0],
			&cOldGradientBuffers[0],
			&cSearchDirBuffers[0],
			&cSVectorsFlat[0],
			&cYVectorsFlat[0],
			&cRhoBuffers[0],
			C.uintptr_t(uintptr(alphaBuffer)),
			C.int(numWeights),
			&cBufferSizes[0],
			C.int(historySize),
			C.int(historyCount),
			C.int(historyIndex),
			C.float(initialStep),
			C.float(c1),
			C.float(c2),
			C.int(maxLineSearch),
			C.float(currentLoss),
			C.float(prevLoss),
			C.uintptr_t(uintptr(commandPool)),
			&stepSize,
		)
	} else {
		result = C.execute_lbfgs_step_mpsgraph(
			C.uintptr_t(uintptr(device)),
			&cWeightBuffers[0],
			&cGradientBuffers[0],
			&cOldGradientBuffers[0],
			&cSearchDirBuffers[0],
			&cSVectorsFlat[0],
			&cYVectorsFlat[0],
			&cRhoBuffers[0],
			C.uintptr_t(uintptr(alphaBuffer)),
			C.int(numWeights),
			&cBufferSizes[0],
			C.int(historySize),
			C.int(historyCount),
			C.int(historyIndex),
			C.float(initialStep),
			C.float(c1),
			C.float(c2),
			C.int(maxLineSearch),
			C.float(currentLoss),
			C.float(prevLoss),
			&stepSize,
		)
	}

	if result != 0 {
		return 0, fmt.Errorf("L-BFGS MPSGraph step failed with error code: %d", result)
	}

	return float32(stepSize), nil
}

// ExecuteAdaGradStepMPSGraph executes a single AdaGrad optimization step using MPSGraph for optimal GPU performance
func ExecuteAdaGradStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	learningRate float32,
	epsilon float32,
	weightDecay float32,
) error {
	if device == nil {
		return fmt.Errorf("device cannot be nil")
	}

	if len(weightBuffers) != numWeights || len(gradientBuffers) != numWeights || 
	   len(squaredGradAvgBuffers) != numWeights || len(bufferSizes) != numWeights {
		return fmt.Errorf("buffer count mismatch: weights=%d, gradients=%d, squared_grad_avg=%d, sizes=%d, expected=%d",
			len(weightBuffers), len(gradientBuffers), len(squaredGradAvgBuffers), len(bufferSizes), numWeights)
	}

	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cSquaredGradAvgBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cSquaredGradAvgBuffers[i] = C.uintptr_t(uintptr(squaredGradAvgBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	result := C.execute_adagrad_step_mpsgraph(
		C.uintptr_t(uintptr(device)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		&cSquaredGradAvgBuffers[0],
		C.int(numWeights),
		&cBufferSizes[0],
		C.float(learningRate),
		C.float(epsilon),
		C.float(weightDecay),
	)

	if result != 0 {
		return fmt.Errorf("AdaGrad MPSGraph step failed with error code: %d", result)
	}

	return nil
}

// ExecuteAdaGradStepMPSGraphPooled executes a single AdaGrad optimization step using MPSGraph with command buffer pooling
func ExecuteAdaGradStepMPSGraphPooled(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	learningRate float32,
	epsilon float32,
	weightDecay float32,
	commandPool unsafe.Pointer,
) error {
	if device == nil {
		return fmt.Errorf("device cannot be nil")
	}

	if len(weightBuffers) != numWeights || len(gradientBuffers) != numWeights || 
	   len(squaredGradAvgBuffers) != numWeights || len(bufferSizes) != numWeights {
		return fmt.Errorf("buffer count mismatch: weights=%d, gradients=%d, squared_grad_avg=%d, sizes=%d, expected=%d",
			len(weightBuffers), len(gradientBuffers), len(squaredGradAvgBuffers), len(bufferSizes), numWeights)
	}

	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cSquaredGradAvgBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cSquaredGradAvgBuffers[i] = C.uintptr_t(uintptr(squaredGradAvgBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	result := C.execute_adagrad_step_mpsgraph_pooled(
		C.uintptr_t(uintptr(device)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		&cSquaredGradAvgBuffers[0],
		C.int(numWeights),
		&cBufferSizes[0],
		C.float(learningRate),
		C.float(epsilon),
		C.float(weightDecay),
		C.uintptr_t(uintptr(commandPool)),
	)

	if result != 0 {
		return fmt.Errorf("AdaGrad MPSGraph pooled step failed with error code: %d", result)
	}

	return nil
}

// ExecuteAdaDeltaStepMPSGraph executes a single AdaDelta optimization step using MPSGraph for optimal GPU performance
func ExecuteAdaDeltaStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	squaredUpdateAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	rho float32,
	epsilon float32,
	weightDecay float32,
) error {
	if device == nil {
		return fmt.Errorf("device cannot be nil")
	}

	if len(weightBuffers) != numWeights || len(gradientBuffers) != numWeights || 
	   len(squaredGradAvgBuffers) != numWeights || len(squaredUpdateAvgBuffers) != numWeights || 
	   len(bufferSizes) != numWeights {
		return fmt.Errorf("buffer count mismatch: weights=%d, gradients=%d, squared_grad_avg=%d, squared_update_avg=%d, sizes=%d, expected=%d",
			len(weightBuffers), len(gradientBuffers), len(squaredGradAvgBuffers), len(squaredUpdateAvgBuffers), len(bufferSizes), numWeights)
	}

	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cSquaredGradAvgBuffers := make([]C.uintptr_t, numWeights)
	cSquaredUpdateAvgBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cSquaredGradAvgBuffers[i] = C.uintptr_t(uintptr(squaredGradAvgBuffers[i]))
		cSquaredUpdateAvgBuffers[i] = C.uintptr_t(uintptr(squaredUpdateAvgBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	result := C.execute_adadelta_step_mpsgraph(
		C.uintptr_t(uintptr(device)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		&cSquaredGradAvgBuffers[0],
		&cSquaredUpdateAvgBuffers[0],
		C.int(numWeights),
		&cBufferSizes[0],
		C.float(rho),
		C.float(epsilon),
		C.float(weightDecay),
	)

	if result != 0 {
		return fmt.Errorf("AdaDelta MPSGraph step failed with error code: %d", result)
	}

	return nil
}

// ExecuteAdaDeltaStepMPSGraphPooled executes a single AdaDelta optimization step using MPSGraph with command buffer pooling
func ExecuteAdaDeltaStepMPSGraphPooled(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	squaredGradAvgBuffers []unsafe.Pointer,
	squaredUpdateAvgBuffers []unsafe.Pointer,
	numWeights int,
	bufferSizes []int,
	rho float32,
	epsilon float32,
	weightDecay float32,
	commandPool unsafe.Pointer,
) error {
	if device == nil {
		return fmt.Errorf("device cannot be nil")
	}

	if len(weightBuffers) != numWeights || len(gradientBuffers) != numWeights || 
	   len(squaredGradAvgBuffers) != numWeights || len(squaredUpdateAvgBuffers) != numWeights || 
	   len(bufferSizes) != numWeights {
		return fmt.Errorf("buffer count mismatch: weights=%d, gradients=%d, squared_grad_avg=%d, squared_update_avg=%d, sizes=%d, expected=%d",
			len(weightBuffers), len(gradientBuffers), len(squaredGradAvgBuffers), len(squaredUpdateAvgBuffers), len(bufferSizes), numWeights)
	}

	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cSquaredGradAvgBuffers := make([]C.uintptr_t, numWeights)
	cSquaredUpdateAvgBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cSquaredGradAvgBuffers[i] = C.uintptr_t(uintptr(squaredGradAvgBuffers[i]))
		cSquaredUpdateAvgBuffers[i] = C.uintptr_t(uintptr(squaredUpdateAvgBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	result := C.execute_adadelta_step_mpsgraph_pooled(
		C.uintptr_t(uintptr(device)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		&cSquaredGradAvgBuffers[0],
		&cSquaredUpdateAvgBuffers[0],
		C.int(numWeights),
		&cBufferSizes[0],
		C.float(rho),
		C.float(epsilon),
		C.float(weightDecay),
		C.uintptr_t(uintptr(commandPool)),
	)

	if result != 0 {
		return fmt.Errorf("AdaDelta MPSGraph pooled step failed with error code: %d", result)
	}

	return nil
}

// ExecuteNadamStepMPSGraph executes a single Nadam optimization step using MPSGraph for optimal GPU performance
// Nadam combines Adam's adaptive learning rates with Nesterov momentum
func ExecuteNadamStepMPSGraph(
	device unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
	momentumBuffers []unsafe.Pointer,
	varianceBuffers []unsafe.Pointer,
	bufferSizes []int,
	learningRate float32,
	beta1 float32,
	beta2 float32,
	epsilon float32,
	weightDecay float32,
	stepCount int,
) error {
	if len(weightBuffers) != len(gradientBuffers) ||
		len(weightBuffers) != len(momentumBuffers) ||
		len(weightBuffers) != len(varianceBuffers) ||
		len(weightBuffers) != len(bufferSizes) {
		return fmt.Errorf("all buffer arrays must have the same length")
	}

	numWeights := len(weightBuffers)

	// Convert Go slices to C arrays
	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)
	cMomentumBuffers := make([]C.uintptr_t, numWeights)
	cVarianceBuffers := make([]C.uintptr_t, numWeights)
	cBufferSizes := make([]C.int, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
		cMomentumBuffers[i] = C.uintptr_t(uintptr(momentumBuffers[i]))
		cVarianceBuffers[i] = C.uintptr_t(uintptr(varianceBuffers[i]))
		cBufferSizes[i] = C.int(bufferSizes[i])
	}

	// Call C function
	result := C.execute_nadam_step_mpsgraph(
		C.uintptr_t(uintptr(device)),
		(*C.uintptr_t)(unsafe.Pointer(&cWeightBuffers[0])),
		(*C.uintptr_t)(unsafe.Pointer(&cGradientBuffers[0])),
		(*C.uintptr_t)(unsafe.Pointer(&cMomentumBuffers[0])),
		(*C.uintptr_t)(unsafe.Pointer(&cVarianceBuffers[0])),
		C.int(numWeights),
		(*C.int)(unsafe.Pointer(&cBufferSizes[0])),
		C.float(learningRate),
		C.float(beta1),
		C.float(beta2),
		C.float(epsilon),
		C.float(weightDecay),
		C.int(stepCount),
	)

	if result != 0 {
		return fmt.Errorf("Nadam MPSGraph step failed with error code: %d", result)
	}

	return nil
}

// CopyTensorBufferSync copies data from one tensor buffer to another synchronously
// This implements GPU-resident tensor copying with minimal CGO calls using the optimized buffer copy
func CopyTensorBufferSync(srcBuffer, dstBuffer unsafe.Pointer, size int) error {
	if srcBuffer == nil || dstBuffer == nil {
		return fmt.Errorf("source or destination buffer is nil")
	}
	
	if size <= 0 {
		return fmt.Errorf("invalid copy size: %d", size)
	}
	
	// Use the synchronous buffer copy using existing C bridge function
	// This is efficient and reuses the existing blit encoder implementation
	result := C.copy_buffer_to_buffer_sync(
		C.uintptr_t(uintptr(srcBuffer)),
		C.uintptr_t(uintptr(dstBuffer)),
		C.int(0),    // Source offset
		C.int(0),    // Destination offset  
		C.int(size), // Copy size
	)
	
	if result != 0 {
		return fmt.Errorf("tensor buffer copy failed with error code: %d", result)
	}
	
	return nil
}

// MEMORY TRANSFER OPTIMIZATION: Go wrapper functions for direct Metal buffer operations

// CopyBufferToBufferAsync performs asynchronous buffer-to-buffer copy using Metal blit encoder
func CopyBufferToBufferAsync(srcBuffer, dstBuffer unsafe.Pointer, 
                            srcOffset, dstOffset, size int, 
                            commandQueue unsafe.Pointer) error {
	if srcBuffer == nil || dstBuffer == nil || commandQueue == nil {
		return fmt.Errorf("invalid buffer or command queue pointers")
	}
	
	if size <= 0 || srcOffset < 0 || dstOffset < 0 {
		return fmt.Errorf("invalid size or offset parameters")
	}
	
	result := C.copy_buffer_to_buffer_async(
		C.uintptr_t(uintptr(srcBuffer)),
		C.uintptr_t(uintptr(dstBuffer)),
		C.int(srcOffset),
		C.int(dstOffset),
		C.int(size),
		C.uintptr_t(uintptr(commandQueue)),
	)
	
	if result != 0 {
		return fmt.Errorf("buffer copy failed with error code: %d", result)
	}
	
	return nil
}

// CopyDataToStagingBuffer copies data from CPU memory to staging buffer
func CopyDataToStagingBuffer(stagingBuffer unsafe.Pointer, data []byte) error {
	if stagingBuffer == nil {
		return fmt.Errorf("staging buffer is nil")
	}
	
	if len(data) == 0 {
		return fmt.Errorf("data is empty")
	}
	
	result := C.copy_data_to_staging_buffer(
		C.uintptr_t(uintptr(stagingBuffer)),
		unsafe.Pointer(&data[0]),
		C.int(len(data)),
	)
	
	if result != 0 {
		return fmt.Errorf("data copy to staging buffer failed with error code: %d", result)
	}
	
	return nil
}

// CopyStagingToGPUBufferAsync copies from staging buffer to GPU buffer asynchronously
func CopyStagingToGPUBufferAsync(stagingBuffer, gpuBuffer unsafe.Pointer,
                                stagingOffset, gpuOffset, size int,
                                commandQueue unsafe.Pointer) error {
	if stagingBuffer == nil || gpuBuffer == nil || commandQueue == nil {
		return fmt.Errorf("invalid buffer or command queue pointers")
	}
	
	if size <= 0 || stagingOffset < 0 || gpuOffset < 0 {
		return fmt.Errorf("invalid size or offset parameters")
	}
	
	result := C.copy_staging_to_gpu_buffer_async(
		C.uintptr_t(uintptr(stagingBuffer)),
		C.uintptr_t(uintptr(gpuBuffer)),
		C.int(stagingOffset),
		C.int(gpuOffset),
		C.int(size),
		C.uintptr_t(uintptr(commandQueue)),
	)
	
	if result != 0 {
		return fmt.Errorf("staging to GPU copy failed with error code: %d", result)
	}
	
	return nil
}

// WaitForBufferCopyCompletion waits for all pending buffer copy operations to complete
func WaitForBufferCopyCompletion(commandQueue unsafe.Pointer) error {
	if commandQueue == nil {
		return fmt.Errorf("command queue is nil")
	}
	
	result := C.wait_for_buffer_copy_completion(
		C.uintptr_t(uintptr(commandQueue)),
	)
	
	if result != 0 {
		return fmt.Errorf("wait for buffer copy completion failed with error code: %d", result)
	}
	
	return nil
}

// Dedicated Inference Engine Implementation

// NewDedicatedInferenceEngine creates a new dedicated inference engine optimized for forward pass only
func NewDedicatedInferenceEngine(
	device unsafe.Pointer,
	config DedicatedInferenceConfig,
	layers []LayerSpecC,
	parameters [][]float32,
) (*DedicatedInferenceEngine, error) {
	if device == nil {
		return nil, fmt.Errorf("device cannot be nil")
	}
	
	if len(layers) == 0 {
		return nil, fmt.Errorf("at least one layer must be provided")
	}
	
	// Convert Go config to C config
	cConfig := C.inference_config_t{
		precision_threshold:    C.float(config.PrecisionThreshold),
		max_batch_size:        C.int(config.MaxBatchSize),
		optimization_level:    C.int(config.OptimizationLevel),
		memory_strategy:       C.int(config.MemoryStrategy),
		enable_telemetry:      boolToInt(config.EnableTelemetry),
		cache_compiled_graphs: boolToInt(config.CacheCompiledGraphs),
	}
	
	// Convert Go layers to C layer specifications
	cLayers := make([]C.layer_spec_c_t, len(layers))
	for i, layer := range layers {
		cLayers[i] = convertLayerSpecToC(layer)
	}
	
	// Prepare parameters for C interface - copy data to C-allocated memory to avoid CGO pointer issues
	cParameters := make([]*C.float, len(parameters))
	cParameterSizes := make([]C.int, len(parameters))
	
	for i, param := range parameters {
		if len(param) > 0 {
			// Allocate C memory and copy data
			cParam := (*C.float)(C.malloc(C.size_t(len(param) * 4))) // 4 bytes per float32
			if cParam == nil {
				return nil, fmt.Errorf("failed to allocate C memory for parameter %d", i)
			}
			
			// Copy Go slice data to C memory
			paramSlice := (*[1 << 30]C.float)(unsafe.Pointer(cParam))[:len(param):len(param)]
			for j, val := range param {
				paramSlice[j] = C.float(val)
			}
			
			cParameters[i] = cParam
			cParameterSizes[i] = C.int(len(param))
		}
	}
	
	// Note: We need to free this memory after the C call, but the engine takes ownership
	
	// Create dedicated inference engine
	enginePtr := C.create_inference_engine_optimized(
		C.uintptr_t(uintptr(device)),
		cConfig,
		&cLayers[0],
		C.int(len(layers)),
		(**C.float)(&cParameters[0]),
		&cParameterSizes[0],
		C.int(len(parameters)),
	)
	
	if enginePtr == 0 {
		return nil, fmt.Errorf("failed to create dedicated inference engine")
	}
	
	return &DedicatedInferenceEngine{
		engine:      unsafe.Pointer(uintptr(enginePtr)),
		config:      config,
		isDestroyed: false,
	}, nil
}

// InferBatch performs batch inference with optimized GPU execution
func (e *DedicatedInferenceEngine) InferBatch(
	inputData []float32,
	inputShape []int,
	batchSize int,
) (*DedicatedInferenceResult, error) {
	fmt.Printf("DEBUG: DedicatedInferenceEngine.InferBatch called with batchSize=%d\n", batchSize)
	if e.isDestroyed {
		return nil, fmt.Errorf("inference engine has been destroyed")
	}
	
	if len(inputData) == 0 {
		return nil, fmt.Errorf("input data cannot be empty")
	}
	
	if batchSize <= 0 || batchSize > e.config.MaxBatchSize {
		return nil, fmt.Errorf("batch size %d must be between 1 and %d", batchSize, e.config.MaxBatchSize)
	}
	
	// Prepare input shape for C interface
	cInputShape := make([]C.int, len(inputShape))
	for i, dim := range inputShape {
		cInputShape[i] = C.int(dim)
	}
	
	// Allocate output buffers
	maxOutputSize := batchSize * 1000 // Reasonable default, will be adjusted by C function
	outputData := make([]float32, maxOutputSize)
	outputShape := make([]C.int, 4) // Max 4 dimensions
	outputShapeLen := C.int(0)
	
	// Prepare result structure for C interface
	var cResult C.inference_result_t
	
	// Execute batch inference with single CGO call
	fmt.Printf("DEBUG: About to call C.execute_inference_batch_optimized\n")
	result := C.execute_inference_batch_optimized(
		C.uintptr_t(uintptr(e.engine)),
		(*C.float)(&inputData[0]),
		&cInputShape[0],
		C.int(len(inputShape)),
		(*C.float)(&outputData[0]),
		&outputShape[0],
		&outputShapeLen,
		C.int(batchSize),
		&cResult,
	)
	
	if result != 0 {
		return nil, fmt.Errorf("batch inference failed with error code: %d", result)
	}
	
	// Convert C result to Go result
	goOutputShape := make([]int, int(outputShapeLen))
	for i := 0; i < int(outputShapeLen); i++ {
		goOutputShape[i] = int(outputShape[i])
	}
	
	// Calculate actual output size
	actualOutputSize := 1
	for _, dim := range goOutputShape {
		actualOutputSize *= dim
	}
	
	return &DedicatedInferenceResult{
		Predictions:     outputData[:actualOutputSize],
		OutputShape:     goOutputShape,
		ConfidenceScore: float32(cResult.confidence_score),
		PredictedClass:  int(cResult.predicted_class),
		InferenceTimeMs: float64(cResult.inference_time_ms),
		MemoryUsedBytes: uint64(cResult.memory_used_bytes),
	}, nil
}

// InferSingle performs single sample inference
func (e *DedicatedInferenceEngine) InferSingle(
	inputData []float32,
	inputShape []int,
) (*DedicatedInferenceResult, error) {
	fmt.Printf("DEBUG: DedicatedInferenceEngine.InferSingle called\n")
	return e.InferBatch(inputData, inputShape, 1)
}

// PreallocateBuffers pre-allocates GPU buffers for optimal performance
func (e *DedicatedInferenceEngine) PreallocateBuffers(maxBatchSize int) error {
	if e.isDestroyed {
		return fmt.Errorf("inference engine has been destroyed")
	}
	
	result := C.preallocate_inference_buffers(
		C.uintptr_t(uintptr(e.engine)),
		C.int(maxBatchSize),
	)
	
	if result != 0 {
		return fmt.Errorf("buffer preallocation failed with error code: %d", result)
	}
	
	return nil
}

// GetTelemetry returns performance telemetry data
func (e *DedicatedInferenceEngine) GetTelemetry() (*InferenceTelemetry, error) {
	if e.isDestroyed {
		return nil, fmt.Errorf("inference engine has been destroyed")
	}
	
	var cTelemetry C.inference_telemetry_t
	C.get_inference_telemetry(
		C.uintptr_t(uintptr(e.engine)),
		&cTelemetry,
	)
	
	return &InferenceTelemetry{
		TotalInferences: uint64(cTelemetry.total_inferences),
		TotalTimeMs:     float64(cTelemetry.total_time_ms),
		AvgLatencyMs:    float64(cTelemetry.avg_latency_ms),
		PeakThroughput:  float64(cTelemetry.peak_throughput),
		PeakMemoryUsage: uint64(cTelemetry.peak_memory_usage),
		CacheHits:       uint64(cTelemetry.cache_hits),
		CacheMisses:     uint64(cTelemetry.cache_misses),
	}, nil
}

// ResetTelemetry clears all telemetry counters
func (e *DedicatedInferenceEngine) ResetTelemetry() error {
	if e.isDestroyed {
		return fmt.Errorf("inference engine has been destroyed")
	}
	
	C.reset_inference_telemetry(C.uintptr_t(uintptr(e.engine)))
	return nil
}

// Destroy properly cleans up the inference engine resources
func (e *DedicatedInferenceEngine) Destroy() error {
	if e.isDestroyed {
		return nil // Already destroyed
	}
	
	C.destroy_inference_engine_optimized(C.uintptr_t(uintptr(e.engine)))
	e.isDestroyed = true
	e.engine = nil
	
	return nil
}

// Helper function to convert Go LayerSpecC to C layer_spec_c_t
func convertLayerSpecToC(layer LayerSpecC) C.layer_spec_c_t {
	var cLayer C.layer_spec_c_t
	
	// Basic layer information
	cLayer.layer_type = C.int(layer.LayerType)
	
	// Copy layer name (truncate if necessary to fit buffer)
	nameBytes := layer.Name[:]
	nameLen := len(nameBytes)
	if nameLen >= 64 {
		nameLen = 63
	}
	copy((*[64]C.char)(unsafe.Pointer(&cLayer.name[0]))[:nameLen], (*[64]C.char)(unsafe.Pointer(&nameBytes[0]))[:nameLen])
	
	// Input shape
	for i := 0; i < len(layer.InputShape) && i < 4; i++ {
		cLayer.input_shape[i] = C.int(layer.InputShape[i])
	}
	cLayer.input_shape_len = C.int(layer.InputShapeLen)
	
	// Output shape
	for i := 0; i < len(layer.OutputShape) && i < 4; i++ {
		cLayer.output_shape[i] = C.int(layer.OutputShape[i])
	}
	cLayer.output_shape_len = C.int(layer.OutputShapeLen)
	
	// Integer parameters
	for i := 0; i < len(layer.ParamInt) && i < 8; i++ {
		cLayer.param_int[i] = C.int(layer.ParamInt[i])
	}
	cLayer.param_int_count = C.int(len(layer.ParamInt))
	
	// Float parameters
	for i := 0; i < len(layer.ParamFloat) && i < 8; i++ {
		cLayer.param_float[i] = C.float(layer.ParamFloat[i])
	}
	cLayer.param_float_count = C.int(len(layer.ParamFloat))
	
	// Running statistics (for BatchNorm layers) - handle CGO pointer issues
	if len(layer.RunningMean) > 0 {
		// Allocate C memory for running mean
		cRunningMean := (*C.float)(C.malloc(C.size_t(len(layer.RunningMean) * 4)))
		if cRunningMean != nil {
			// Copy data to C memory
			meanSlice := (*[1 << 30]C.float)(unsafe.Pointer(cRunningMean))[:len(layer.RunningMean):len(layer.RunningMean)]
			for i, val := range layer.RunningMean {
				meanSlice[i] = C.float(val)
			}
			cLayer.running_mean = cRunningMean
			cLayer.running_stats_size = C.int(len(layer.RunningMean))
			cLayer.has_running_stats = 1
		} else {
			cLayer.running_mean = nil
			cLayer.running_stats_size = 0
			cLayer.has_running_stats = 0
		}
	} else {
		cLayer.running_mean = nil
		cLayer.running_stats_size = 0
		cLayer.has_running_stats = 0
	}
	
	if len(layer.RunningVar) > 0 {
		// Allocate C memory for running variance
		cRunningVar := (*C.float)(C.malloc(C.size_t(len(layer.RunningVar) * 4)))
		if cRunningVar != nil {
			// Copy data to C memory
			varSlice := (*[1 << 30]C.float)(unsafe.Pointer(cRunningVar))[:len(layer.RunningVar):len(layer.RunningVar)]
			for i, val := range layer.RunningVar {
				varSlice[i] = C.float(val)
			}
			cLayer.running_var = cRunningVar
		} else {
			cLayer.running_var = nil
		}
	} else {
		cLayer.running_var = nil
	}
	
	return cLayer
}

// Helper function to convert bool to int for C interface
func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}