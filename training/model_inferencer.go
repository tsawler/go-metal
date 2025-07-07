package training

import (
	"fmt"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/engine"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// ModelInferencer provides inference-only functionality using dedicated InferenceEngine
// Optimized for forward-pass only without training overhead
type ModelInferencer struct {
	inferenceEngine *engine.ModelInferenceEngine
	modelSpec       *layers.ModelSpec
	batchSize       int
	config          InferencerConfig
}

// InferencerConfig holds configuration for inference-only operations
type InferencerConfig struct {
	// Performance settings
	BatchSize            int  `json:"batch_size"`             // Target batch size for inference
	UseDynamicEngine     bool `json:"use_dynamic_engine"`     // Use dynamic graph (recommended: true)
	OptimizeForSingleBatch bool `json:"optimize_single_batch"` // Optimize for batch size 1
	UseCommandPooling    bool `json:"use_command_pooling"`    // Enable command buffer pooling
	
	// Batch normalization mode
	BatchNormInferenceMode bool `json:"batchnorm_inference_mode"` // Use running stats for batch norm
}

// NewModelInferencer creates a new inference-only engine
func NewModelInferencer(
	modelSpec *layers.ModelSpec,
	config InferencerConfig,
) (*ModelInferencer, error) {
	// Validate configuration
	if err := validateInferencerConfig(config); err != nil {
		return nil, fmt.Errorf("invalid inferencer configuration: %v", err)
	}
	
	// Convert to inference engine config
	inferenceConfig := cgo_bridge.InferenceConfig{
		UseDynamicEngine:       config.UseDynamicEngine,
		BatchNormInferenceMode: config.BatchNormInferenceMode,
		UseCommandPooling:      config.UseCommandPooling,
		OptimizeForSingleBatch: config.OptimizeForSingleBatch,
	}
	
	// Set input shape
	if len(modelSpec.InputShape) > 0 {
		inferenceConfig.InputShape = make([]int32, len(modelSpec.InputShape))
		for i, dim := range modelSpec.InputShape {
			inferenceConfig.InputShape[i] = int32(dim)
		}
		inferenceConfig.InputShapeLen = int32(len(modelSpec.InputShape))
	}
	
	// Create inference engine
	inferenceEngine, err := engine.NewModelInferenceEngine(modelSpec, inferenceConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create inference engine: %v", err)
	}
	
	return &ModelInferencer{
		inferenceEngine: inferenceEngine,
		modelSpec:       modelSpec,
		batchSize:       config.BatchSize,
		config:          config,
	}, nil
}

// Cleanup performs deterministic resource cleanup
func (mi *ModelInferencer) Cleanup() {
	if mi.inferenceEngine != nil {
		mi.inferenceEngine.Cleanup()
		mi.inferenceEngine = nil
	}
}

// LoadWeights loads pre-trained weights into the inference engine
func (mi *ModelInferencer) LoadWeights(weights []checkpoints.WeightTensor) error {
	if mi.inferenceEngine == nil {
		return fmt.Errorf("inference engine not initialized")
	}
	
	return mi.inferenceEngine.LoadWeights(weights)
}

// LoadWeightsFromCheckpoint loads weights from a checkpoint
func (mi *ModelInferencer) LoadWeightsFromCheckpoint(weights []checkpoints.WeightTensor) error {
	if mi.inferenceEngine == nil {
		return fmt.Errorf("inference engine not initialized")
	}
	
	// Convert checkpoint weights to checkpoints.WeightTensor format
	// (already in the correct format)
	memoryWeights := weights
	
	return mi.inferenceEngine.LoadWeights(memoryWeights)
}

// Predict performs single forward pass for inference
// This is the lightweight method optimized for single-image or small batch inference
func (mi *ModelInferencer) Predict(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error) {
	if mi.inferenceEngine == nil {
		return nil, fmt.Errorf("inference engine not initialized")
	}
	
	// Use the inference engine's optimized predict method
	return mi.inferenceEngine.Predict(inputData, inputShape)
}

// PredictBatch performs inference on a batch of data
// Optimized for larger batches with efficient GPU utilization
func (mi *ModelInferencer) PredictBatch(
	inputData []float32,
	inputShape []int,
) (*cgo_bridge.InferenceResult, error) {
	// For now, use the same predict method
	// In a full implementation, this could have batch-specific optimizations
	return mi.Predict(inputData, inputShape)
}

// GetModelSpec returns the model specification
func (mi *ModelInferencer) GetModelSpec() *layers.ModelSpec {
	return mi.modelSpec
}

// GetParameterTensors returns the GPU-resident parameter tensors
func (mi *ModelInferencer) GetParameterTensors() []*memory.Tensor {
	if mi.inferenceEngine == nil {
		return nil
	}
	return mi.inferenceEngine.GetParameterTensors()
}

// validateInferencerConfig validates the inferencer configuration
func validateInferencerConfig(config InferencerConfig) error {
	if config.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive, got %d", config.BatchSize)
	}
	
	if config.BatchSize > 1024 {
		return fmt.Errorf("batch size too large, got %d (max: 1024)", config.BatchSize)
	}
	
	return nil
}

// DefaultInferencerConfig returns a sensible default configuration for inference
func DefaultInferencerConfig() InferencerConfig {
	return InferencerConfig{
		BatchSize:              1,    // Single image inference by default
		UseDynamicEngine:       true, // Use dynamic engine for flexibility
		OptimizeForSingleBatch: true, // Optimize for single image inference
		UseCommandPooling:      true, // Enable command buffer pooling for performance
		BatchNormInferenceMode: true, // Use running stats for batch normalization
	}
}

// Note: Using checkpoints.WeightTensor directly to avoid type conflicts