package training

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tsawler/go-metal/checkpoints"
	"github.com/tsawler/go-metal/layers"
	"github.com/tsawler/go-metal/memory"
)

// CheckpointConfig configures checkpoint saving behavior
type CheckpointConfig struct {
	SaveDirectory    string                        // Directory to save checkpoints
	SaveFrequency    int                          // Save every N epochs (0 = disabled)
	SaveBest         bool                         // Save checkpoint when validation improves
	MaxCheckpoints   int                          // Maximum number of checkpoints to keep (0 = unlimited)
	Format           checkpoints.CheckpointFormat // JSON or ONNX
	FilenamePattern  string                       // Pattern for checkpoint filenames
}

// DefaultCheckpointConfig returns a sensible default configuration
func DefaultCheckpointConfig() CheckpointConfig {
	return CheckpointConfig{
		SaveDirectory:   "./checkpoints",
		SaveFrequency:   5, // Save every 5 epochs
		SaveBest:        true,
		MaxCheckpoints:  10,
		Format:          checkpoints.FormatJSON,
		FilenamePattern: "checkpoint_epoch_%d_step_%d",
	}
}

// CheckpointManager handles checkpoint saving and loading for ModelTrainer
type CheckpointManager struct {
	config      CheckpointConfig
	trainer     *ModelTrainer
	saver       *checkpoints.CheckpointSaver
	bestLoss    float32
	bestAccuracy float32
	savedFiles  []string // Track saved checkpoint files for cleanup
}

// NewCheckpointManager creates a new checkpoint manager
func NewCheckpointManager(trainer *ModelTrainer, config CheckpointConfig) *CheckpointManager {
	return &CheckpointManager{
		config:       config,
		trainer:      trainer,
		saver:        checkpoints.NewCheckpointSaver(config.Format),
		bestLoss:     float32(1e9), // Initialize with high loss
		bestAccuracy: 0.0,
		savedFiles:   make([]string, 0),
	}
}

// SaveCheckpoint saves the current model state
func (cm *CheckpointManager) SaveCheckpoint(epoch int, step int, loss float32, accuracy float32, description string) error {
	// Create checkpoint from current trainer state
	checkpoint, err := cm.createCheckpointFromTrainer(epoch, step, loss, accuracy, description)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint: %v", err)
	}
	
	// Generate filename
	filename := cm.generateFilename(epoch, step)
	filepath := filepath.Join(cm.config.SaveDirectory, filename)
	
	// Ensure directory exists
	if err := cm.ensureDirectory(); err != nil {
		return fmt.Errorf("failed to create checkpoint directory: %v", err)
	}
	
	// Save checkpoint
	if err := cm.saver.SaveCheckpoint(checkpoint, filepath); err != nil {
		return fmt.Errorf("failed to save checkpoint: %v", err)
	}
	
	// Track saved file
	cm.savedFiles = append(cm.savedFiles, filepath)
	
	// Cleanup old checkpoints if needed
	if err := cm.cleanupOldCheckpoints(); err != nil {
		// Log warning but don't fail the save operation
		fmt.Printf("Warning: failed to cleanup old checkpoints: %v\n", err)
	}
	
	return nil
}

// SaveBestCheckpoint saves a checkpoint if it's better than previous best
func (cm *CheckpointManager) SaveBestCheckpoint(epoch int, step int, loss float32, accuracy float32) (bool, error) {
	if !cm.config.SaveBest {
		return false, nil
	}
	
	// Check if this is better than previous best
	isBetterLoss := loss < cm.bestLoss
	isBetterAccuracy := accuracy > cm.bestAccuracy
	
	if isBetterLoss || isBetterAccuracy {
		// Update best metrics
		if isBetterLoss {
			cm.bestLoss = loss
		}
		if isBetterAccuracy {
			cm.bestAccuracy = accuracy
		}
		
		// Save as best checkpoint
		description := fmt.Sprintf("Best checkpoint - Loss: %.6f, Accuracy: %.2f%%", loss, accuracy*100)
		filename := fmt.Sprintf("best_checkpoint.%s", cm.getFileExtension())
		filepath := filepath.Join(cm.config.SaveDirectory, filename)
		
		checkpoint, err := cm.createCheckpointFromTrainer(epoch, step, loss, accuracy, description)
		if err != nil {
			return false, fmt.Errorf("failed to create best checkpoint: %v", err)
		}
		
		if err := cm.saver.SaveCheckpoint(checkpoint, filepath); err != nil {
			return false, fmt.Errorf("failed to save best checkpoint: %v", err)
		}
		
		return true, nil
	}
	
	return false, nil
}

// SavePeriodicCheckpoint saves a checkpoint if it's time based on frequency
func (cm *CheckpointManager) SavePeriodicCheckpoint(epoch int, step int, loss float32, accuracy float32) (bool, error) {
	if cm.config.SaveFrequency <= 0 {
		return false, nil
	}
	
	if epoch%cm.config.SaveFrequency == 0 {
		description := fmt.Sprintf("Periodic checkpoint - Epoch %d", epoch)
		if err := cm.SaveCheckpoint(epoch, step, loss, accuracy, description); err != nil {
			return false, err
		}
		return true, nil
	}
	
	return false, nil
}

// LoadCheckpoint loads a checkpoint and restores trainer state
func (cm *CheckpointManager) LoadCheckpoint(filepath string) error {
	// Load checkpoint
	checkpoint, err := cm.saver.LoadCheckpoint(filepath)
	if err != nil {
		return fmt.Errorf("failed to load checkpoint: %v", err)
	}
	
	// Restore trainer state from checkpoint
	if err := cm.restoreTrainerFromCheckpoint(checkpoint); err != nil {
		return fmt.Errorf("failed to restore trainer state: %v", err)
	}
	
	return nil
}

// createCheckpointFromTrainer creates a checkpoint from current trainer state
func (cm *CheckpointManager) createCheckpointFromTrainer(epoch int, step int, loss float32, accuracy float32, description string) (*checkpoints.Checkpoint, error) {
	// Get model specification
	modelSpec := cm.trainer.GetModelSpec()
	if modelSpec == nil {
		return nil, fmt.Errorf("trainer has no model specification")
	}
	
	// Extract weights from GPU tensors
	parameterTensors := cm.trainer.GetParameterTensors()
	weights, err := checkpoints.ExtractWeightsFromTensors(parameterTensors, modelSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to extract weights: %v", err)
	}
	
	// Create training state
	trainingState := checkpoints.TrainingState{
		Epoch:        epoch,
		Step:         step,
		LearningRate: cm.trainer.GetCurrentLearningRate(),
		BestLoss:     cm.bestLoss,
		BestAccuracy: cm.bestAccuracy,
		TotalSteps:   step,
	}
	
	// Create optimizer state (if available)
	var optimizerState *checkpoints.OptimizerState
	if optimizerData := cm.trainer.GetOptimizerState(); optimizerData != nil {
		optimizerState = &checkpoints.OptimizerState{
			Type:       optimizerData.Type,
			Parameters: optimizerData.Parameters,
			StateData:  optimizerData.StateData,
		}
	}
	
	// Create checkpoint
	checkpoint := &checkpoints.Checkpoint{
		ModelSpec:      modelSpec,
		Weights:        weights,
		TrainingState:  trainingState,
		OptimizerState: optimizerState,
		Metadata: checkpoints.CheckpointMetadata{
			Version:     "1.0.0",
			Framework:   "go-metal",
			Description: description,
			Tags:        []string{fmt.Sprintf("epoch_%d", epoch)},
		},
	}
	
	return checkpoint, nil
}

// restoreTrainerFromCheckpoint restores trainer state from checkpoint
func (cm *CheckpointManager) restoreTrainerFromCheckpoint(checkpoint *checkpoints.Checkpoint) error {
	// Validate model compatibility
	currentModelSpec := cm.trainer.GetModelSpec()
	if currentModelSpec == nil {
		return fmt.Errorf("trainer has no model specification")
	}
	
	// Check if model architectures match
	if !cm.modelsCompatible(currentModelSpec, checkpoint.ModelSpec) {
		return fmt.Errorf("checkpoint model architecture incompatible with current trainer")
	}
	
	// Load weights into GPU tensors
	parameterTensors := cm.trainer.GetParameterTensors()
	if err := checkpoints.LoadWeightsIntoTensors(checkpoint.Weights, parameterTensors); err != nil {
		return fmt.Errorf("failed to load weights: %v", err)
	}
	
	// Restore training state
	cm.bestLoss = checkpoint.TrainingState.BestLoss
	cm.bestAccuracy = checkpoint.TrainingState.BestAccuracy
	
	// Set learning rate if scheduler is available
	if scheduler := cm.trainer.GetLRScheduler(); scheduler != nil {
		cm.trainer.SetLearningRate(checkpoint.TrainingState.LearningRate)
	}
	
	// Restore optimizer state if available
	if checkpoint.OptimizerState != nil {
		if err := cm.trainer.SetOptimizerState(checkpoint.OptimizerState); err != nil {
			return fmt.Errorf("failed to restore optimizer state: %v", err)
		}
	}
	
	return nil
}

// Helper methods

func (cm *CheckpointManager) generateFilename(epoch int, step int) string {
	pattern := cm.config.FilenamePattern
	if pattern == "" {
		pattern = "checkpoint_epoch_%d_step_%d"
	}
	
	// Generate the base filename using the pattern
	baseFilename := fmt.Sprintf(pattern, epoch, step)
	
	// Add the file extension
	filename := fmt.Sprintf("%s.%s", baseFilename, cm.getFileExtension())
	
	return filename
}

func (cm *CheckpointManager) getFileExtension() string {
	switch cm.config.Format {
	case checkpoints.FormatJSON:
		return "json"
	case checkpoints.FormatONNX:
		return "onnx"
	default:
		return "json"
	}
}

func (cm *CheckpointManager) ensureDirectory() error {
	return os.MkdirAll(cm.config.SaveDirectory, 0755)
}

func (cm *CheckpointManager) cleanupOldCheckpoints() error {
	if cm.config.MaxCheckpoints <= 0 {
		return nil // No limit
	}
	
	if len(cm.savedFiles) <= cm.config.MaxCheckpoints {
		return nil // Under limit
	}
	
	// Remove oldest checkpoints
	toRemove := len(cm.savedFiles) - cm.config.MaxCheckpoints
	for i := 0; i < toRemove; i++ {
		if err := os.Remove(cm.savedFiles[i]); err != nil {
			return fmt.Errorf("failed to remove old checkpoint %s: %v", cm.savedFiles[i], err)
		}
	}
	
	// Update tracked files
	cm.savedFiles = cm.savedFiles[toRemove:]
	
	return nil
}

func (cm *CheckpointManager) modelsCompatible(model1, model2 *layers.ModelSpec) bool {
	// Check if models have same number of layers
	if len(model1.Layers) != len(model2.Layers) {
		return false
	}
	
	// Check if each layer is compatible
	for i, layer1 := range model1.Layers {
		layer2 := model2.Layers[i]
		
		// Check layer type
		if layer1.Type != layer2.Type {
			return false
		}
		
		// Check parameter shapes (this ensures weight tensors are compatible)
		if len(layer1.ParameterShapes) != len(layer2.ParameterShapes) {
			return false
		}
		
		for j, shape1 := range layer1.ParameterShapes {
			shape2 := layer2.ParameterShapes[j]
			if len(shape1) != len(shape2) {
				return false
			}
			for k, dim1 := range shape1 {
				if dim1 != shape2[k] {
					return false
				}
			}
		}
	}
	
	return true
}

// Add checkpoint methods to ModelTrainer interface
type CheckpointCapable interface {
	GetModelSpec() *layers.ModelSpec
	GetParameterTensors() []*memory.Tensor
	GetCurrentLearningRate() float32
	GetLRScheduler() interface{} // Returns the scheduler if available
	SetLearningRate(lr float32)
	GetOptimizerState() *OptimizerStateData
	SetOptimizerState(state *checkpoints.OptimizerState) error
}

// OptimizerStateData represents internal optimizer state
type OptimizerStateData struct {
	Type       string
	Parameters map[string]interface{}
	StateData  []checkpoints.OptimizerTensor
}