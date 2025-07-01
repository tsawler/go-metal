package cgo_bridge

/*
#include <stdlib.h>
#include <stdint.h>

// Training configuration
typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float weight_decay;
    float epsilon;
    int optimizer_type; // 0 = SGD, 1 = Adam
} training_config_t;

// Forward declarations for CGO functions
uintptr_t create_metal_device();
uintptr_t create_training_engine(uintptr_t device, training_config_t* config);
uintptr_t create_training_engine_constant_weights(uintptr_t device, training_config_t* config);
uintptr_t create_training_engine_hybrid(uintptr_t device, training_config_t* config);
int execute_training_step(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* loss_out
);
int execute_training_step_hybrid(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* loss_out
);
int execute_training_step_hybrid_full(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float learning_rate,
    float* loss_out
);
uintptr_t allocate_metal_buffer(uintptr_t device, int size, int device_type);
void deallocate_metal_buffer(uintptr_t buffer);
void destroy_training_engine(uintptr_t engine);

// Adam optimizer functions
int execute_adam_step(
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

// Utility functions for buffer operations
int zero_metal_buffer(uintptr_t device, uintptr_t buffer, int size);

// MPSGraph-based buffer zeroing for GPU-only buffers
int zero_metal_buffer_mpsgraph(uintptr_t device, uintptr_t buffer, int size);

// Data transfer functions
int copy_data_to_metal_buffer(uintptr_t buffer, void* data, int size);
int copy_float32_array_to_metal_buffer(uintptr_t buffer, float* data, int num_elements);
int copy_int32_array_to_metal_buffer(uintptr_t buffer, int* data, int num_elements);

// Forward+backward pass that returns gradients for Adam optimizer
int execute_training_step_hybrid_with_gradients(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    float* loss_out
);

// Forward-only inference that returns predictions
int execute_inference_hybrid(
    uintptr_t engine,
    uintptr_t input_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* predictions_out,
    int batch_size,
    int num_classes
);

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
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// OptimizerType represents the type of optimizer
type OptimizerType int

const (
	SGD OptimizerType = iota
	Adam
)

// TrainingConfig holds training configuration
type TrainingConfig struct {
	LearningRate  float32
	Beta1         float32
	Beta2         float32
	WeightDecay   float32
	Epsilon       float32
	OptimizerType OptimizerType
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

// CreateTrainingEngine creates a training engine
func CreateTrainingEngine(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error) {
	cConfig := C.training_config_t{
		learning_rate:  C.float(config.LearningRate),
		beta1:         C.float(config.Beta1),
		beta2:         C.float(config.Beta2),
		weight_decay:  C.float(config.WeightDecay),
		epsilon:       C.float(config.Epsilon),
		optimizer_type: C.int(config.OptimizerType),
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
		optimizer_type: C.int(config.OptimizerType),
	}
	
	engine := C.create_training_engine_constant_weights(C.uintptr_t(uintptr(device)), &cConfig)
	if engine == 0 {
		return nil, fmt.Errorf("failed to create constant weights training engine")
	}
	
	return unsafe.Pointer(uintptr(engine)), nil
}

// CreateTrainingEngineHybrid creates a hybrid MPS/MPSGraph training engine
func CreateTrainingEngineHybrid(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error) {
	cConfig := C.training_config_t{
		learning_rate:  C.float(config.LearningRate),
		beta1:         C.float(config.Beta1),
		beta2:         C.float(config.Beta2),
		weight_decay:  C.float(config.WeightDecay),
		epsilon:       C.float(config.Epsilon),
		optimizer_type: C.int(config.OptimizerType),
	}
	
	engine := C.create_training_engine_hybrid(C.uintptr_t(uintptr(device)), &cConfig)
	if engine == 0 {
		return nil, fmt.Errorf("failed to create hybrid training engine")
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

// ExecuteTrainingStepHybrid executes a training step using hybrid MPS/MPSGraph approach
func ExecuteTrainingStepHybrid(
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
	result := C.execute_training_step_hybrid(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		&cWeightBuffers[0],
		C.int(len(weightBuffers)),
		&lossOut,
	)
	
	if result != 0 {
		return 0, fmt.Errorf("hybrid training step failed with error code: %d", result)
	}
	
	return float32(lossOut), nil
}

// ExecuteTrainingStepHybridFull executes a complete training step with backward pass using hybrid MPS/MPSGraph approach
func ExecuteTrainingStepHybridFull(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	learningRate float32,
) (float32, error) {
	// Convert weight buffers to C array
	cWeightBuffers := make([]C.uintptr_t, len(weightBuffers))
	for i, buf := range weightBuffers {
		cWeightBuffers[i] = C.uintptr_t(uintptr(buf))
	}
	
	var lossOut C.float
	result := C.execute_training_step_hybrid_full(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		&cWeightBuffers[0],
		C.int(len(weightBuffers)),
		C.float(learningRate),
		&lossOut,
	)
	
	if result != 0 {
		return 0, fmt.Errorf("hybrid full training step failed with error code: %d", result)
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

// DestroyTrainingEngine destroys a training engine
func DestroyTrainingEngine(engine unsafe.Pointer) {
	if engine != nil {
		C.destroy_training_engine(C.uintptr_t(uintptr(engine)))
	}
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

// ExecuteAdamStep executes a single Adam optimization step on GPU (legacy CPU-based implementation)
func ExecuteAdamStep(
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

	result := C.execute_adam_step(
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
		return fmt.Errorf("Adam step failed with error code: %d", result)
	}

	return nil
}

// ExecuteTrainingStepHybridWithGradients executes forward+backward pass and returns gradients
func ExecuteTrainingStepHybridWithGradients(
	engine unsafe.Pointer,
	inputBuffer unsafe.Pointer,
	labelBuffer unsafe.Pointer,
	weightBuffers []unsafe.Pointer,
	gradientBuffers []unsafe.Pointer,
) (float32, error) {
	if len(weightBuffers) != len(gradientBuffers) {
		return 0, fmt.Errorf("weight buffers (%d) and gradient buffers (%d) count mismatch",
			len(weightBuffers), len(gradientBuffers))
	}

	numWeights := len(weightBuffers)

	// Convert Go slices to C arrays
	cWeightBuffers := make([]C.uintptr_t, numWeights)
	cGradientBuffers := make([]C.uintptr_t, numWeights)

	for i := 0; i < numWeights; i++ {
		cWeightBuffers[i] = C.uintptr_t(uintptr(weightBuffers[i]))
		cGradientBuffers[i] = C.uintptr_t(uintptr(gradientBuffers[i]))
	}

	var lossOut C.float
	result := C.execute_training_step_hybrid_with_gradients(
		C.uintptr_t(uintptr(engine)),
		C.uintptr_t(uintptr(inputBuffer)),
		C.uintptr_t(uintptr(labelBuffer)),
		&cWeightBuffers[0],
		&cGradientBuffers[0],
		C.int(numWeights),
		&lossOut,
	)

	if result != 0 {
		return 0, fmt.Errorf("hybrid training step with gradients failed with error code: %d", result)
	}

	return float32(lossOut), nil
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
	// Choose between hybrid and dynamic inference based on engine type
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
		result = C.execute_inference_hybrid(
			C.uintptr_t(uintptr(engine)),
			C.uintptr_t(uintptr(inputBuffer)),
			cWeightBuffers,
			C.int(len(weightBuffers)),
			(*C.float)(unsafe.Pointer(&predictions[0])),
			C.int(batchSize),
			C.int(numClasses),
		)
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
		optimizer_type: C.int(config.OptimizerType),
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