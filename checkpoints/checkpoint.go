package checkpoints

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/tsawler/go-metal/cgo_bridge"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// Initialize memory bridge for checkpoint functionality
func init() {
	cgo_bridge.SetupMemoryBridge(memory.SetupBridge)
}

// CheckpointFormat defines the serialization format
type CheckpointFormat int

const (
	FormatJSON CheckpointFormat = iota
	FormatONNX
)

func (cf CheckpointFormat) String() string {
	switch cf {
	case FormatJSON:
		return "JSON"
	case FormatONNX:
		return "ONNX"
	default:
		return "Unknown"
	}
}

// Checkpoint represents a complete model state including weights, optimizer state, and training metadata
type Checkpoint struct {
	// Model architecture and weights
	ModelSpec   *layers.ModelSpec `json:"model_spec"`
	Weights     []WeightTensor    `json:"weights"`
	
	// Training state
	TrainingState TrainingState `json:"training_state"`
	
	// Optimizer state (if available)
	OptimizerState *OptimizerState `json:"optimizer_state,omitempty"`
	
	// Metadata
	Metadata CheckpointMetadata `json:"metadata"`
}

// WeightTensor represents a model parameter tensor with its data
type WeightTensor struct {
	Name   string    `json:"name"`
	Shape  []int     `json:"shape"`
	Data   []float32 `json:"data"`
	Layer  string    `json:"layer"`
	Type   string    `json:"type"` // "weight", "bias", "gamma", "beta", etc.
}

// TrainingState captures the current training progress
type TrainingState struct {
	Epoch         int     `json:"epoch"`
	Step          int     `json:"step"`
	LearningRate  float32 `json:"learning_rate"`
	BestLoss      float32 `json:"best_loss"`
	BestAccuracy  float32 `json:"best_accuracy"`
	TotalSteps    int     `json:"total_steps"`
}

// OptimizerState captures optimizer-specific state (momentum, variance, etc.)
type OptimizerState struct {
	Type        string                 `json:"type"` // "SGD", "Adam", etc.
	Parameters  map[string]interface{} `json:"parameters"`
	StateData   []OptimizerTensor      `json:"state_data"`
}

// OptimizerTensor represents optimizer state tensors (momentum, variance, etc.)
type OptimizerTensor struct {
	Name      string    `json:"name"`
	Shape     []int     `json:"shape"`
	Data      []float32 `json:"data"`
	StateType string    `json:"state_type"` // "momentum", "variance", "m", "v", etc.
}

// CheckpointMetadata contains checkpoint metadata
type CheckpointMetadata struct {
	Version     string    `json:"version"`
	Framework   string    `json:"framework"`
	CreatedAt   time.Time `json:"created_at"`
	Description string    `json:"description,omitempty"`
	Tags        []string  `json:"tags,omitempty"`
}

// CheckpointSaver handles saving model checkpoints in various formats
type CheckpointSaver struct {
	format CheckpointFormat
}

// NewCheckpointSaver creates a new checkpoint saver for the specified format
func NewCheckpointSaver(format CheckpointFormat) *CheckpointSaver {
	return &CheckpointSaver{
		format: format,
	}
}

// SaveCheckpoint saves a complete model checkpoint
func (cs *CheckpointSaver) SaveCheckpoint(checkpoint *Checkpoint, path string) error {
	switch cs.format {
	case FormatJSON:
		return cs.saveJSON(checkpoint, path)
	case FormatONNX:
		return cs.saveONNX(checkpoint, path)
	default:
		return fmt.Errorf("unsupported checkpoint format: %s", cs.format.String())
	}
}

// LoadCheckpoint loads a model checkpoint
func (cs *CheckpointSaver) LoadCheckpoint(path string) (*Checkpoint, error) {
	switch cs.format {
	case FormatJSON:
		return cs.loadJSON(path)
	case FormatONNX:
		return cs.loadONNX(path)
	default:
		return nil, fmt.Errorf("unsupported checkpoint format: %s", cs.format.String())
	}
}

// saveJSON saves checkpoint in JSON format
func (cs *CheckpointSaver) saveJSON(checkpoint *Checkpoint, path string) error {
	// Ensure metadata is set
	if checkpoint.Metadata.Framework == "" {
		checkpoint.Metadata.Framework = "go-metal"
		checkpoint.Metadata.Version = "1.0.0"
		checkpoint.Metadata.CreatedAt = time.Now()
	}
	
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint file: %v", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Pretty print JSON
	
	if err := encoder.Encode(checkpoint); err != nil {
		return fmt.Errorf("failed to encode checkpoint: %v", err)
	}
	
	return nil
}

// loadJSON loads checkpoint from JSON format
func (cs *CheckpointSaver) loadJSON(path string) (*Checkpoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open checkpoint file: %v", err)
	}
	defer file.Close()
	
	var checkpoint Checkpoint
	decoder := json.NewDecoder(file)
	
	if err := decoder.Decode(&checkpoint); err != nil {
		return nil, fmt.Errorf("failed to decode checkpoint: %v", err)
	}
	
	return &checkpoint, nil
}

// saveONNX saves checkpoint in ONNX format
func (cs *CheckpointSaver) saveONNX(checkpoint *Checkpoint, path string) error {
	exporter := NewONNXExporter()
	return exporter.ExportToONNX(checkpoint, path)
}

// loadONNX loads checkpoint from ONNX format
func (cs *CheckpointSaver) loadONNX(path string) (*Checkpoint, error) {
	importer := NewONNXImporter()
	return importer.ImportFromONNX(path)
}

// ExtractWeightsFromTensors extracts weight data from GPU tensors while maintaining GPU-resident design
func ExtractWeightsFromTensors(tensors []*memory.Tensor, modelSpec *layers.ModelSpec) ([]WeightTensor, error) {
	var weights []WeightTensor
	
	// Map parameter tensors to their corresponding layers and types
	paramIndex := 0
	for layerIdx, layerSpec := range modelSpec.Layers {
		layerName := layerSpec.Name
		
		switch layerSpec.Type {
		case layers.Dense:
			// Dense layer: weight + optional bias
			if paramIndex >= len(tensors) {
				return nil, fmt.Errorf("insufficient tensors for dense layer %s", layerName)
			}
			
			// Weight tensor
			weightTensor := tensors[paramIndex]
			weightData, err := weightTensor.ToFloat32Slice()
			if err != nil {
				return nil, fmt.Errorf("failed to extract weight data for layer %s: %v", layerName, err)
			}
			
			weights = append(weights, WeightTensor{
				Name:  fmt.Sprintf("%s.weight", layerName),
				Shape: weightTensor.Shape(),
				Data:  weightData,
				Layer: layerName,
				Type:  "weight",
			})
			paramIndex++
			
			// Bias tensor (if present)
			useBias := layerSpec.Parameters["use_bias"].(bool)
			if useBias {
				if paramIndex >= len(tensors) {
					return nil, fmt.Errorf("insufficient tensors for dense layer bias %s", layerName)
				}
				
				biasTensor := tensors[paramIndex]
				biasData, err := biasTensor.ToFloat32Slice()
				if err != nil {
					return nil, fmt.Errorf("failed to extract bias data for layer %s: %v", layerName, err)
				}
				
				weights = append(weights, WeightTensor{
					Name:  fmt.Sprintf("%s.bias", layerName),
					Shape: biasTensor.Shape(),
					Data:  biasData,
					Layer: layerName,
					Type:  "bias",
				})
				paramIndex++
			}
			
		case layers.Conv2D:
			// Conv2D layer: weight + optional bias
			if paramIndex >= len(tensors) {
				return nil, fmt.Errorf("insufficient tensors for conv2d layer %s", layerName)
			}
			
			// Weight tensor
			weightTensor := tensors[paramIndex]
			weightData, err := weightTensor.ToFloat32Slice()
			if err != nil {
				return nil, fmt.Errorf("failed to extract weight data for layer %s: %v", layerName, err)
			}
			
			weights = append(weights, WeightTensor{
				Name:  fmt.Sprintf("%s.weight", layerName),
				Shape: weightTensor.Shape(),
				Data:  weightData,
				Layer: layerName,
				Type:  "weight",
			})
			paramIndex++
			
			// Bias tensor (if present)
			useBias := layerSpec.Parameters["use_bias"].(bool)
			if useBias {
				if paramIndex >= len(tensors) {
					return nil, fmt.Errorf("insufficient tensors for conv2d layer bias %s", layerName)
				}
				
				biasTensor := tensors[paramIndex]
				biasData, err := biasTensor.ToFloat32Slice()
				if err != nil {
					return nil, fmt.Errorf("failed to extract bias data for layer %s: %v", layerName, err)
				}
				
				weights = append(weights, WeightTensor{
					Name:  fmt.Sprintf("%s.bias", layerName),
					Shape: biasTensor.Shape(),
					Data:  biasData,
					Layer: layerName,
					Type:  "bias",
				})
				paramIndex++
			}
			
		case layers.BatchNorm:
			// BatchNorm layer: gamma + beta (if affine=true)
			affine := layerSpec.Parameters["affine"].(bool)
			if affine {
				if paramIndex+1 >= len(tensors) {
					return nil, fmt.Errorf("insufficient tensors for batchnorm layer %s", layerName)
				}
				
				// Gamma (scale) tensor
				gammaTensor := tensors[paramIndex]
				gammaData, err := gammaTensor.ToFloat32Slice()
				if err != nil {
					return nil, fmt.Errorf("failed to extract gamma data for layer %s: %v", layerName, err)
				}
				
				weights = append(weights, WeightTensor{
					Name:  fmt.Sprintf("%s.weight", layerName), // ONNX uses "weight" for gamma
					Shape: gammaTensor.Shape(),
					Data:  gammaData,
					Layer: layerName,
					Type:  "gamma",
				})
				paramIndex++
				
				// Beta (shift) tensor
				betaTensor := tensors[paramIndex]
				betaData, err := betaTensor.ToFloat32Slice()
				if err != nil {
					return nil, fmt.Errorf("failed to extract beta data for layer %s: %v", layerName, err)
				}
				
				weights = append(weights, WeightTensor{
					Name:  fmt.Sprintf("%s.bias", layerName), // ONNX uses "bias" for beta
					Shape: betaTensor.Shape(),
					Data:  betaData,
					Layer: layerName,
					Type:  "beta",
				})
				paramIndex++
			}
			
		case layers.ReLU, layers.LeakyReLU, layers.Softmax, layers.Dropout:
			// Activation layers have no parameters
			continue
			
		default:
			return nil, fmt.Errorf("unsupported layer type for weight extraction: %s", layerSpec.Type.String())
		}
		
		_ = layerIdx // Suppress unused variable warning
	}
	
	return weights, nil
}

// LoadWeightsIntoTensors loads weight data back into GPU tensors
func LoadWeightsIntoTensors(weights []WeightTensor, tensors []*memory.Tensor) error {
	// Create a map for quick weight lookup
	weightMap := make(map[string]WeightTensor)
	for _, weight := range weights {
		weightMap[weight.Name] = weight
	}
	
	// We need to match tensors with weights based on the layer naming convention
	// This is a simplified implementation that assumes tensors are in the same order
	// as they were extracted
	if len(weights) != len(tensors) {
		return fmt.Errorf("weight count mismatch: %d weights, %d tensors", len(weights), len(tensors))
	}
	
	for i, tensor := range tensors {
		if i >= len(weights) {
			break
		}
		
		weight := weights[i]
		
		// Verify tensor and weight compatibility
		tensorShape := tensor.Shape()
		if len(tensorShape) != len(weight.Shape) {
			return fmt.Errorf("shape mismatch for weight %s: tensor %v vs weight %v", 
				weight.Name, tensorShape, weight.Shape)
		}
		
		for j, dim := range tensorShape {
			if dim != weight.Shape[j] {
				return fmt.Errorf("dimension mismatch for weight %s at index %d: tensor %d vs weight %d", 
					weight.Name, j, dim, weight.Shape[j])
			}
		}
		
		// Copy weight data into tensor
		if err := tensor.CopyFloat32Data(weight.Data); err != nil {
			return fmt.Errorf("failed to copy weight data for %s: %v", weight.Name, err)
		}
	}
	
	return nil
}