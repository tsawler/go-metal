package training

import (
	"fmt"
	"strings"
	"time"

	"github.com/tsawler/go-metal/layers"
)

// ProgressBar provides PyTorch-style training progress visualization
type ProgressBar struct {
	description string
	total       int
	current     int
	startTime   time.Time
	width       int
	showRate    bool
	showETA     bool
	metrics     map[string]float64
}

// NewProgressBar creates a new progress bar
func NewProgressBar(description string, total int) *ProgressBar {
	return &ProgressBar{
		description: description,
		total:       total,
		current:     0,
		startTime:   time.Now(),
		width:       70, // Character width of progress bar
		showRate:    true,
		showETA:     true,
		metrics:     make(map[string]float64),
	}
}

// Update advances the progress bar
func (pb *ProgressBar) Update(step int, metrics map[string]float64) {
	pb.current = step
	pb.metrics = metrics
	pb.render()
}

// UpdateMetrics updates metrics without advancing progress
func (pb *ProgressBar) UpdateMetrics(metrics map[string]float64) {
	for k, v := range metrics {
		pb.metrics[k] = v
	}
	pb.render()
}

// Finish completes the progress bar
func (pb *ProgressBar) Finish() {
	pb.current = pb.total
	pb.render()
	fmt.Println() // New line after completion
}

// render draws the progress bar
func (pb *ProgressBar) render() {
	// Calculate progress percentage
	percentage := float64(pb.current) / float64(pb.total)
	if percentage > 1.0 {
		percentage = 1.0
	}

	// Calculate filled width
	filled := int(percentage * float64(pb.width))
	if filled > pb.width {
		filled = pb.width
	}

	// Build progress bar string
	bar := strings.Repeat("█", filled) + strings.Repeat(" ", pb.width-filled)

	// Calculate timing information
	elapsed := time.Since(pb.startTime)
	var eta time.Duration
	var rate float64

	if pb.current > 0 {
		rate = float64(pb.current) / elapsed.Seconds()
		if percentage > 0 {
			totalTime := time.Duration(float64(elapsed) / percentage)
			eta = totalTime - elapsed
		}
	}

	// Format the progress line
	line := fmt.Sprintf("\r%s: %3.0f%%|%s| %d/%d",
		pb.description,
		percentage*100,
		bar,
		pb.current,
		pb.total,
	)

	// Add timing information
	if pb.showETA && eta > 0 {
		line += fmt.Sprintf(" [%s<%s",
			formatDuration(elapsed),
			formatDuration(eta),
		)
	} else {
		line += fmt.Sprintf(" [%s<00:00",
			formatDuration(elapsed),
		)
	}

	// Add rate information
	if pb.showRate && rate > 0 {
		line += fmt.Sprintf(", %.2fbatch/s", rate)
	}

	// Add metrics
	for key, value := range pb.metrics {
		if strings.Contains(key, "accuracy") || strings.Contains(key, "acc") {
			line += fmt.Sprintf(", %s=%.2f%%", key, value*100)
		} else {
			line += fmt.Sprintf(", %s=%.3f", key, value)
		}
	}

	line += "]"

	// Print the line (carriage return overwrites previous line)
	fmt.Print(line)
}

// formatDuration formats duration as MM:SS
func formatDuration(d time.Duration) string {
	minutes := int(d.Minutes())
	seconds := int(d.Seconds()) % 60
	return fmt.Sprintf("%02d:%02d", minutes, seconds)
}

// ModelArchitecturePrinter prints PyTorch-style model architecture
type ModelArchitecturePrinter struct {
	modelName string
}

// NewModelArchitecturePrinter creates a new model architecture printer
func NewModelArchitecturePrinter(modelName string) *ModelArchitecturePrinter {
	return &ModelArchitecturePrinter{
		modelName: modelName,
	}
}

// PrintArchitecture prints the model architecture in PyTorch style
func (p *ModelArchitecturePrinter) PrintArchitecture(modelSpec *layers.ModelSpec) {
	fmt.Printf("Model Architecture:\n")
	fmt.Printf("%s(\n", p.modelName)

	for i, layer := range modelSpec.Layers {
		layerStr := p.formatLayer(layer, i)
		fmt.Printf("  %s\n", layerStr)
	}

	fmt.Printf(")\n\n")

	// Print parameter summary
	fmt.Printf("Total parameters: %s\n", formatParameterCount(modelSpec.TotalParameters))
	fmt.Printf("Trainable parameters: %s\n", formatParameterCount(modelSpec.TotalParameters)) // Assuming all are trainable
	fmt.Printf("Non-trainable parameters: 0\n")
	fmt.Printf("Input size (MB): %.3f\n", calculateInputSize(modelSpec.InputShape))
	fmt.Printf("Forward/backward pass size (MB): %.3f\n", estimateForwardBackwardSize(modelSpec))
	fmt.Printf("Params size (MB): %.3f\n", float64(modelSpec.TotalParameters*4)/1024/1024) // 4 bytes per float32
	fmt.Printf("Estimated Total Size (MB): %.3f\n\n", estimateTotalSize(modelSpec))
}

// formatLayer formats a single layer for display
func (p *ModelArchitecturePrinter) formatLayer(layer layers.LayerSpec, index int) string {
	switch layer.Type {
	case layers.Conv2D:
		return p.formatConv2D(layer)
	case layers.Dense:
		return p.formatDense(layer)
	case layers.ReLU:
		return fmt.Sprintf("(%s): ReLU()", layer.Name)
	case layers.Softmax:
		axis := layer.Parameters["axis"].(int)
		return fmt.Sprintf("(%s): Softmax(dim=%d)", layer.Name, axis)
	default:
		return fmt.Sprintf("(%s): %s()", layer.Name, layer.Type.String())
	}
}

// formatConv2D formats a Conv2D layer
func (p *ModelArchitecturePrinter) formatConv2D(layer layers.LayerSpec) string {
	inChannels := layer.Parameters["input_channels"].(int)
	outChannels := layer.Parameters["output_channels"].(int)
	kernelSize := layer.Parameters["kernel_size"].(int)
	stride := layer.Parameters["stride"].(int)
	padding := layer.Parameters["padding"].(int)
	useBias := layer.Parameters["use_bias"].(bool)

	return fmt.Sprintf("(%s): Conv2d(%d, %d, kernel_size=(%d, %d), stride=(%d, %d), padding=(%d, %d), bias=%t)",
		layer.Name, inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, useBias)
}

// formatDense formats a Dense/Linear layer
func (p *ModelArchitecturePrinter) formatDense(layer layers.LayerSpec) string {
	inFeatures := layer.Parameters["input_size"].(int)
	outFeatures := layer.Parameters["output_size"].(int)
	useBias := layer.Parameters["use_bias"].(bool)

	return fmt.Sprintf("(%s): Linear(in_features=%d, out_features=%d, bias=%t)",
		layer.Name, inFeatures, outFeatures, useBias)
}

// formatParameterCount formats parameter count with K/M suffixes
func formatParameterCount(count int64) string {
	if count >= 1000000 {
		return fmt.Sprintf("%.1fM", float64(count)/1000000.0)
	} else if count >= 1000 {
		return fmt.Sprintf("%.1fK", float64(count)/1000.0)
	}
	return fmt.Sprintf("%d", count)
}

// calculateInputSize estimates input tensor size in MB
func calculateInputSize(inputShape []int) float64 {
	size := 1
	for _, dim := range inputShape {
		size *= dim
	}
	return float64(size*4) / 1024 / 1024 // 4 bytes per float32
}

// estimateForwardBackwardSize estimates forward/backward pass memory usage
func estimateForwardBackwardSize(modelSpec *layers.ModelSpec) float64 {
	// Rough estimate: 2x forward pass memory for activations + gradients
	inputSize := calculateInputSize(modelSpec.InputShape)
	outputSize := calculateInputSize(modelSpec.OutputShape)
	
	// Estimate intermediate activations (rough heuristic)
	maxIntermediateSize := inputSize
	for _, layer := range modelSpec.Layers {
		if len(layer.OutputShape) > 0 {
			layerSize := calculateInputSize(layer.OutputShape)
			if layerSize > maxIntermediateSize {
				maxIntermediateSize = layerSize
			}
		}
	}
	
	return (inputSize + outputSize + maxIntermediateSize) * 2 // 2x for forward + backward
}

// estimateTotalSize estimates total model memory usage
func estimateTotalSize(modelSpec *layers.ModelSpec) float64 {
	inputSize := calculateInputSize(modelSpec.InputShape)
	paramsSize := float64(modelSpec.TotalParameters*4) / 1024 / 1024
	forwardBackwardSize := estimateForwardBackwardSize(modelSpec)
	
	return inputSize + paramsSize + forwardBackwardSize
}

// TrainingSession manages a complete training session with progress visualization
type TrainingSession struct {
	trainer           *ModelTrainer
	modelName         string
	epochs            int
	stepsPerEpoch     int
	validationSteps   int
	currentEpoch      int
	architecturePrinter *ModelArchitecturePrinter
	
	// Progress tracking
	trainProgress     *ProgressBar
	validationProgress *ProgressBar
	
	// Metrics tracking
	trainLoss         float64
	trainAccuracy     float64
	validationLoss    float64
	validationAccuracy float64
}

// NewTrainingSession creates a new training session with progress visualization
func NewTrainingSession(
	trainer *ModelTrainer,
	modelName string,
	epochs int,
	stepsPerEpoch int,
	validationSteps int,
) *TrainingSession {
	return &TrainingSession{
		trainer:           trainer,
		modelName:         modelName,
		epochs:            epochs,
		stepsPerEpoch:     stepsPerEpoch,
		validationSteps:   validationSteps,
		currentEpoch:      0,
		architecturePrinter: NewModelArchitecturePrinter(modelName),
	}
}

// StartTraining begins the training session with model architecture display
func (ts *TrainingSession) StartTraining() {
	ts.architecturePrinter.PrintArchitecture(ts.trainer.GetModelSpec())
	fmt.Println("Starting training...")
}

// StartEpoch begins a new epoch
func (ts *TrainingSession) StartEpoch(epoch int) {
	ts.currentEpoch = epoch
	
	// Create training progress bar
	description := fmt.Sprintf("Epoch %d/%d (Training)", epoch, ts.epochs)
	ts.trainProgress = NewProgressBar(description, ts.stepsPerEpoch)
}

// UpdateTrainingProgress updates training progress
func (ts *TrainingSession) UpdateTrainingProgress(step int, loss float64, accuracy float64) {
	ts.trainLoss = loss
	ts.trainAccuracy = accuracy
	
	metrics := map[string]float64{
		"loss": loss,
	}
	
	if accuracy >= 0 {
		metrics["accuracy"] = accuracy
	}
	
	ts.trainProgress.Update(step, metrics)
}

// FinishTrainingEpoch completes the training phase of an epoch
func (ts *TrainingSession) FinishTrainingEpoch() {
	ts.trainProgress.Finish()
}

// StartValidation begins the validation phase
func (ts *TrainingSession) StartValidation() {
	if ts.validationSteps <= 0 {
		return
	}
	
	description := fmt.Sprintf("Epoch %d/%d (Validation)", ts.currentEpoch, ts.epochs)
	ts.validationProgress = NewProgressBar(description, ts.validationSteps)
}

// UpdateValidationProgress updates validation progress
func (ts *TrainingSession) UpdateValidationProgress(step int, loss float64, accuracy float64) {
	ts.validationLoss = loss
	ts.validationAccuracy = accuracy
	
	metrics := map[string]float64{
		"loss": loss,
	}
	
	if accuracy >= 0 {
		metrics["accuracy"] = accuracy
	}
	
	ts.validationProgress.Update(step, metrics)
}

// FinishValidationEpoch completes the validation phase of an epoch
func (ts *TrainingSession) FinishValidationEpoch() {
	if ts.validationProgress != nil {
		ts.validationProgress.Finish()
	}
}

// PrintEpochSummary prints a summary of the completed epoch
func (ts *TrainingSession) PrintEpochSummary() {
	fmt.Printf("Epoch %d/%d Summary:\n", ts.currentEpoch, ts.epochs)
	fmt.Printf("  Training   - Loss: %.4f", ts.trainLoss)
	
	if ts.trainAccuracy >= 0 {
		fmt.Printf(", Accuracy: %.2f%%", ts.trainAccuracy*100)
	}
	fmt.Println()
	
	if ts.validationSteps > 0 {
		fmt.Printf("  Validation - Loss: %.4f", ts.validationLoss)
		if ts.validationAccuracy >= 0 {
			fmt.Printf(", Accuracy: %.2f%%", ts.validationAccuracy*100)
		}
		fmt.Println()
	}
	
	fmt.Println()
}