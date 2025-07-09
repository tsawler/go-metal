package training

import (
	"encoding/json"
	"fmt"
	"time"
)

// PlotType represents different types of plots that can be generated
type PlotType string

const (
	// Training plots
	TrainingCurves     PlotType = "training_curves"
	LearningRateSchedule PlotType = "learning_rate_schedule"
	
	// Evaluation plots
	ROCCurve           PlotType = "roc_curve"
	PrecisionRecall    PlotType = "precision_recall"
	ConfusionMatrixPlot PlotType = "confusion_matrix"
	
	// Model analysis plots
	ParameterDistribution PlotType = "parameter_distribution"
	GradientHistogram     PlotType = "gradient_histogram"
	ActivationPattern     PlotType = "activation_pattern"
	
	// Regression plots
	RegressionScatter     PlotType = "regression_scatter"
	ResidualPlot          PlotType = "residual_plot"
)

// PlotData represents the universal JSON format for the sidecar plotting service
type PlotData struct {
	// Metadata
	PlotType    PlotType  `json:"plot_type"`
	Title       string    `json:"title"`
	Timestamp   time.Time `json:"timestamp"`
	ModelName   string    `json:"model_name"`
	
	// Data series - flexible structure for different plot types
	Series []SeriesData `json:"series"`
	
	// Plot configuration
	Config PlotConfig `json:"config"`
	
	// Metrics metadata
	Metrics map[string]interface{} `json:"metrics,omitempty"`
}

// SeriesData represents a single data series in a plot
type SeriesData struct {
	Name   string                 `json:"name"`
	Type   string                 `json:"type"`   // "line", "scatter", "histogram", "heatmap", "bar"
	Data   []DataPoint            `json:"data"`
	Style  map[string]interface{} `json:"style,omitempty"`
}

// DataPoint represents a single data point - flexible for different plot types
type DataPoint struct {
	X     interface{} `json:"x"`
	Y     interface{} `json:"y"`
	Z     interface{} `json:"z,omitempty"`     // For heatmaps, 3D plots
	Label string      `json:"label,omitempty"` // For categorical data
	Color string      `json:"color,omitempty"` // For custom coloring
}

// PlotConfig contains plot-specific configuration
type PlotConfig struct {
	XAxisLabel    string                 `json:"x_axis_label"`
	YAxisLabel    string                 `json:"y_axis_label"`
	ZAxisLabel    string                 `json:"z_axis_label,omitempty"`
	XAxisScale    string                 `json:"x_axis_scale"`    // "linear", "log"
	YAxisScale    string                 `json:"y_axis_scale"`    // "linear", "log"
	ShowLegend    bool                   `json:"show_legend"`
	ShowGrid      bool                   `json:"show_grid"`
	Width         int                    `json:"width"`
	Height        int                    `json:"height"`
	Interactive   bool                   `json:"interactive"`
	CustomOptions map[string]interface{} `json:"custom_options,omitempty"`
}

// VisualizationCollector handles data collection for plotting
type VisualizationCollector struct {
	modelName string
	enabled   bool
	
	// Training data
	trainingLoss     []float64
	trainingAccuracy []float64
	validationLoss   []float64
	validationAccuracy []float64
	epochs           []int
	steps            []int
	learningRates    []float64
	
	// Evaluation data
	rocPoints        []ROCPointViz
	prPoints         []PRPoint
	confusionMatrix  [][]int
	classNames       []string
	
	// Regression data
	predictions      []float64
	trueValues       []float64
	residuals        []float64
	
	// Model analysis data
	parameterStats   map[string]ParameterStats
	gradientStats    map[string]GradientStats
	activationStats  map[string]ActivationStats
}

// ROCPointViz represents a point on the ROC curve for visualization
type ROCPointViz struct {
	FPR       float64 `json:"fpr"`
	TPR       float64 `json:"tpr"`
	Threshold float64 `json:"threshold"`
}

// PRPoint represents a point on the Precision-Recall curve
type PRPoint struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	Threshold float64 `json:"threshold"`
}

// ParameterStats represents parameter distribution statistics
type ParameterStats struct {
	LayerName string    `json:"layer_name"`
	ParamType string    `json:"param_type"` // "weight", "bias"
	Mean      float64   `json:"mean"`
	Std       float64   `json:"std"`
	Min       float64   `json:"min"`
	Max       float64   `json:"max"`
	Histogram []float64 `json:"histogram"`
	Bins      []float64 `json:"bins"`
}

// GradientStats represents gradient statistics
type GradientStats struct {
	LayerName     string    `json:"layer_name"`
	ParamType     string    `json:"param_type"`
	GradientNorm  float64   `json:"gradient_norm"`
	Mean          float64   `json:"mean"`
	Std           float64   `json:"std"`
	Histogram     []float64 `json:"histogram"`
	Bins          []float64 `json:"bins"`
}

// ActivationStats represents activation pattern statistics
type ActivationStats struct {
	LayerName     string    `json:"layer_name"`
	ActivationType string   `json:"activation_type"`
	Mean          float64   `json:"mean"`
	Std           float64   `json:"std"`
	SparsityRatio float64   `json:"sparsity_ratio"`
	Histogram     []float64 `json:"histogram"`
	Bins          []float64 `json:"bins"`
}

// NewVisualizationCollector creates a new visualization collector
func NewVisualizationCollector(modelName string) *VisualizationCollector {
	return &VisualizationCollector{
		modelName:       modelName,
		enabled:         false,
		trainingLoss:    make([]float64, 0),
		trainingAccuracy: make([]float64, 0),
		validationLoss:  make([]float64, 0),
		validationAccuracy: make([]float64, 0),
		epochs:          make([]int, 0),
		steps:           make([]int, 0),
		learningRates:   make([]float64, 0),
		rocPoints:       make([]ROCPointViz, 0),
		prPoints:        make([]PRPoint, 0),
		predictions:     make([]float64, 0),
		trueValues:      make([]float64, 0),
		residuals:       make([]float64, 0),
		parameterStats:  make(map[string]ParameterStats),
		gradientStats:   make(map[string]GradientStats),
		activationStats: make(map[string]ActivationStats),
	}
}

// Enable enables visualization data collection
func (vc *VisualizationCollector) Enable() {
	vc.enabled = true
}

// Disable disables visualization data collection
func (vc *VisualizationCollector) Disable() {
	vc.enabled = false
}

// IsEnabled returns whether visualization is enabled
func (vc *VisualizationCollector) IsEnabled() bool {
	return vc.enabled
}

// RecordTrainingStep records training metrics for a single step
func (vc *VisualizationCollector) RecordTrainingStep(step int, loss, accuracy, learningRate float64) {
	if !vc.enabled {
		return
	}
	
	vc.steps = append(vc.steps, step)
	vc.trainingLoss = append(vc.trainingLoss, loss)
	vc.trainingAccuracy = append(vc.trainingAccuracy, accuracy)
	vc.learningRates = append(vc.learningRates, learningRate)
}

// RecordValidationStep records validation metrics for a single step
func (vc *VisualizationCollector) RecordValidationStep(step int, loss, accuracy float64) {
	if !vc.enabled {
		return
	}
	
	vc.validationLoss = append(vc.validationLoss, loss)
	vc.validationAccuracy = append(vc.validationAccuracy, accuracy)
}

// RecordEpoch records epoch-level metrics
func (vc *VisualizationCollector) RecordEpoch(epoch int, trainLoss, trainAcc, valLoss, valAcc float64) {
	if !vc.enabled {
		return
	}
	
	vc.epochs = append(vc.epochs, epoch)
	// Note: These will be used for epoch-level plots, separate from step-level data
}

// RecordROCData records ROC curve data points
func (vc *VisualizationCollector) RecordROCData(rocPoints []ROCPointViz) {
	if !vc.enabled {
		return
	}
	
	vc.rocPoints = rocPoints
}

// RecordPRData records Precision-Recall curve data points
func (vc *VisualizationCollector) RecordPRData(prPoints []PRPoint) {
	if !vc.enabled {
		return
	}
	
	vc.prPoints = prPoints
}

// RecordConfusionMatrix records confusion matrix data
func (vc *VisualizationCollector) RecordConfusionMatrix(matrix [][]int, classNames []string) {
	if !vc.enabled {
		return
	}
	
	vc.confusionMatrix = matrix
	vc.classNames = classNames
}

// RecordRegressionData records regression predictions and true values
func (vc *VisualizationCollector) RecordRegressionData(predictions, trueValues []float64) {
	if !vc.enabled {
		return
	}
	
	vc.predictions = predictions
	vc.trueValues = trueValues
	
	// Calculate residuals
	vc.residuals = make([]float64, len(predictions))
	for i := range predictions {
		vc.residuals[i] = predictions[i] - trueValues[i]
	}
}

// RecordParameterStats records parameter distribution statistics
func (vc *VisualizationCollector) RecordParameterStats(layerName, paramType string, stats ParameterStats) {
	if !vc.enabled {
		return
	}
	
	key := fmt.Sprintf("%s_%s", layerName, paramType)
	vc.parameterStats[key] = stats
}

// RecordGradientStats records gradient statistics
func (vc *VisualizationCollector) RecordGradientStats(layerName, paramType string, stats GradientStats) {
	if !vc.enabled {
		return
	}
	
	key := fmt.Sprintf("%s_%s", layerName, paramType)
	vc.gradientStats[key] = stats
}

// RecordActivationStats records activation pattern statistics
func (vc *VisualizationCollector) RecordActivationStats(layerName, activationType string, stats ActivationStats) {
	if !vc.enabled {
		return
	}
	
	key := fmt.Sprintf("%s_%s", layerName, activationType)
	vc.activationStats[key] = stats
}

// GenerateTrainingCurvesPlot generates training curves plot data
func (vc *VisualizationCollector) GenerateTrainingCurvesPlot() PlotData {
	series := []SeriesData{
		{
			Name: "Training Loss",
			Type: "line",
			Data: make([]DataPoint, len(vc.trainingLoss)),
			Style: map[string]interface{}{
				"color": "#FF6B6B",
				"line_width": 2,
			},
		},
		{
			Name: "Training Accuracy",
			Type: "line",
			Data: make([]DataPoint, len(vc.trainingAccuracy)),
			Style: map[string]interface{}{
				"color": "#4ECDC4",
				"line_width": 2,
			},
		},
	}
	
	// Add training loss data
	for i, loss := range vc.trainingLoss {
		series[0].Data[i] = DataPoint{
			X: vc.steps[i],
			Y: loss,
		}
	}
	
	// Add training accuracy data
	for i, acc := range vc.trainingAccuracy {
		series[1].Data[i] = DataPoint{
			X: vc.steps[i],
			Y: acc,
		}
	}
	
	// Add validation data if available
	if len(vc.validationLoss) > 0 {
		valLossSeries := SeriesData{
			Name: "Validation Loss",
			Type: "line",
			Data: make([]DataPoint, len(vc.validationLoss)),
			Style: map[string]interface{}{
				"color": "#FF9F43",
				"line_width": 2,
				"line_style": "dashed",
			},
		}
		
		valAccSeries := SeriesData{
			Name: "Validation Accuracy",
			Type: "line",
			Data: make([]DataPoint, len(vc.validationAccuracy)),
			Style: map[string]interface{}{
				"color": "#5F27CD",
				"line_width": 2,
				"line_style": "dashed",
			},
		}
		
		for i, loss := range vc.validationLoss {
			valLossSeries.Data[i] = DataPoint{
				X: i + 1, // Validation step numbers
				Y: loss,
			}
		}
		
		for i, acc := range vc.validationAccuracy {
			valAccSeries.Data[i] = DataPoint{
				X: i + 1, // Validation step numbers
				Y: acc,
			}
		}
		
		series = append(series, valLossSeries, valAccSeries)
	}
	
	return PlotData{
		PlotType:  TrainingCurves,
		Title:     fmt.Sprintf("Training Curves - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Step",
			YAxisLabel:  "Loss / Accuracy",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       800,
			Height:      600,
			Interactive: true,
		},
	}
}

// GenerateLearningRateSchedulePlot generates learning rate schedule plot data
func (vc *VisualizationCollector) GenerateLearningRateSchedulePlot() PlotData {
	series := []SeriesData{
		{
			Name: "Learning Rate",
			Type: "line",
			Data: make([]DataPoint, len(vc.learningRates)),
			Style: map[string]interface{}{
				"color": "#6C5CE7",
				"line_width": 2,
			},
		},
	}
	
	for i, lr := range vc.learningRates {
		series[0].Data[i] = DataPoint{
			X: vc.steps[i],
			Y: lr,
		}
	}
	
	return PlotData{
		PlotType:  LearningRateSchedule,
		Title:     fmt.Sprintf("Learning Rate Schedule - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Step",
			YAxisLabel:  "Learning Rate",
			XAxisScale:  "linear",
			YAxisScale:  "log",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       800,
			Height:      400,
			Interactive: true,
		},
	}
}

// GenerateROCCurvePlot generates ROC curve plot data
func (vc *VisualizationCollector) GenerateROCCurvePlot() PlotData {
	series := []SeriesData{
		{
			Name: "ROC Curve",
			Type: "line",
			Data: make([]DataPoint, len(vc.rocPoints)),
			Style: map[string]interface{}{
				"color": "#FF6B6B",
				"line_width": 2,
			},
		},
		{
			Name: "Random Classifier",
			Type: "line",
			Data: []DataPoint{
				{X: 0.0, Y: 0.0},
				{X: 1.0, Y: 1.0},
			},
			Style: map[string]interface{}{
				"color": "#95A5A6",
				"line_width": 1,
				"line_style": "dashed",
			},
		},
	}
	
	for i, point := range vc.rocPoints {
		series[0].Data[i] = DataPoint{
			X: point.FPR,
			Y: point.TPR,
		}
	}
	
	return PlotData{
		PlotType:  ROCCurve,
		Title:     fmt.Sprintf("ROC Curve - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "False Positive Rate",
			YAxisLabel:  "True Positive Rate",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       600,
			Height:      600,
			Interactive: true,
		},
	}
}

// GeneratePrecisionRecallPlot generates Precision-Recall curve plot data
func (vc *VisualizationCollector) GeneratePrecisionRecallPlot() PlotData {
	series := []SeriesData{
		{
			Name: "Precision-Recall Curve",
			Type: "line",
			Data: make([]DataPoint, len(vc.prPoints)),
			Style: map[string]interface{}{
				"color": "#4ECDC4",
				"line_width": 2,
			},
		},
	}
	
	for i, point := range vc.prPoints {
		series[0].Data[i] = DataPoint{
			X: point.Recall,
			Y: point.Precision,
		}
	}
	
	return PlotData{
		PlotType:  PrecisionRecall,
		Title:     fmt.Sprintf("Precision-Recall Curve - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Recall",
			YAxisLabel:  "Precision",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       600,
			Height:      600,
			Interactive: true,
		},
	}
}

// GenerateConfusionMatrixPlot generates confusion matrix plot data
func (vc *VisualizationCollector) GenerateConfusionMatrixPlot() PlotData {
	if len(vc.confusionMatrix) == 0 {
		return PlotData{}
	}
	
	var data []DataPoint
	for i, row := range vc.confusionMatrix {
		for j, value := range row {
			data = append(data, DataPoint{
				X: j,
				Y: i,
				Z: value,
				Label: fmt.Sprintf("True: %s, Pred: %s", vc.classNames[i], vc.classNames[j]),
			})
		}
	}
	
	series := []SeriesData{
		{
			Name: "Confusion Matrix",
			Type: "heatmap",
			Data: data,
			Style: map[string]interface{}{
				"colorscale": "Blues",
			},
		},
	}
	
	return PlotData{
		PlotType:  ConfusionMatrixPlot,
		Title:     fmt.Sprintf("Confusion Matrix - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Predicted Class",
			YAxisLabel:  "True Class",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  false,
			ShowGrid:    false,
			Width:       600,
			Height:      600,
			Interactive: true,
			CustomOptions: map[string]interface{}{
				"class_names": vc.classNames,
			},
		},
	}
}

// GenerateRegressionScatterPlot generates regression scatter plot data
func (vc *VisualizationCollector) GenerateRegressionScatterPlot() PlotData {
	if len(vc.predictions) == 0 {
		return PlotData{}
	}
	
	// Scatter plot data
	scatterData := make([]DataPoint, len(vc.predictions))
	for i := range vc.predictions {
		scatterData[i] = DataPoint{
			X: vc.trueValues[i],
			Y: vc.predictions[i],
		}
	}
	
	// Perfect prediction line
	minVal := vc.trueValues[0]
	maxVal := vc.trueValues[0]
	for _, val := range vc.trueValues {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}
	
	perfectLine := []DataPoint{
		{X: minVal, Y: minVal},
		{X: maxVal, Y: maxVal},
	}
	
	series := []SeriesData{
		{
			Name: "Predictions",
			Type: "scatter",
			Data: scatterData,
			Style: map[string]interface{}{
				"color": "#4ECDC4",
				"alpha": 0.6,
			},
		},
		{
			Name: "Perfect Prediction",
			Type: "line",
			Data: perfectLine,
			Style: map[string]interface{}{
				"color": "#FF6B6B",
				"line_width": 2,
				"line_style": "dashed",
			},
		},
	}
	
	return PlotData{
		PlotType:  RegressionScatter,
		Title:     fmt.Sprintf("Regression Scatter Plot - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "True Values",
			YAxisLabel:  "Predicted Values",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       600,
			Height:      600,
			Interactive: true,
		},
	}
}

// GenerateResidualPlot generates residual plot data
func (vc *VisualizationCollector) GenerateResidualPlot() PlotData {
	if len(vc.residuals) == 0 {
		return PlotData{}
	}
	
	residualData := make([]DataPoint, len(vc.residuals))
	for i := range vc.residuals {
		residualData[i] = DataPoint{
			X: vc.predictions[i],
			Y: vc.residuals[i],
		}
	}
	
	// Zero line
	minPred := vc.predictions[0]
	maxPred := vc.predictions[0]
	for _, pred := range vc.predictions {
		if pred < minPred {
			minPred = pred
		}
		if pred > maxPred {
			maxPred = pred
		}
	}
	
	zeroLine := []DataPoint{
		{X: minPred, Y: 0.0},
		{X: maxPred, Y: 0.0},
	}
	
	series := []SeriesData{
		{
			Name: "Residuals",
			Type: "scatter",
			Data: residualData,
			Style: map[string]interface{}{
				"color": "#FF9F43",
				"alpha": 0.6,
			},
		},
		{
			Name: "Zero Line",
			Type: "line",
			Data: zeroLine,
			Style: map[string]interface{}{
				"color": "#95A5A6",
				"line_width": 1,
				"line_style": "dashed",
			},
		},
	}
	
	return PlotData{
		PlotType:  ResidualPlot,
		Title:     fmt.Sprintf("Residual Plot - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Predicted Values",
			YAxisLabel:  "Residuals",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       600,
			Height:      600,
			Interactive: true,
		},
	}
}

// ToJSON converts plot data to JSON string
func (pd PlotData) ToJSON() (string, error) {
	jsonData, err := json.MarshalIndent(pd, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal plot data to JSON: %w", err)
	}
	return string(jsonData), nil
}

// Clear resets all collected data
func (vc *VisualizationCollector) Clear() {
	vc.trainingLoss = vc.trainingLoss[:0]
	vc.trainingAccuracy = vc.trainingAccuracy[:0]
	vc.validationLoss = vc.validationLoss[:0]
	vc.validationAccuracy = vc.validationAccuracy[:0]
	vc.epochs = vc.epochs[:0]
	vc.steps = vc.steps[:0]
	vc.learningRates = vc.learningRates[:0]
	vc.rocPoints = vc.rocPoints[:0]
	vc.prPoints = vc.prPoints[:0]
	vc.confusionMatrix = nil
	vc.classNames = nil
	vc.predictions = vc.predictions[:0]
	vc.trueValues = vc.trueValues[:0]
	vc.residuals = vc.residuals[:0]
	vc.parameterStats = make(map[string]ParameterStats)
	vc.gradientStats = make(map[string]GradientStats)
	vc.activationStats = make(map[string]ActivationStats)
}