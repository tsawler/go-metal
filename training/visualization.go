package training

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
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
	QQPlot               PlotType = "qq_plot"
	FeatureImportancePlot PlotType = "feature_importance"
	LearningCurvePlot     PlotType = "learning_curve"
	ValidationCurvePlot   PlotType = "validation_curve"
	PredictionIntervalPlot PlotType = "prediction_interval"
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
	featureNames     []string
	coefficients     []float64
	featureStdErrors []float64
	
	// Learning curve data
	trainingSizes     []int
	trainingScores    []float64
	validationScores  []float64
	trainingStdErrors []float64
	validationStdErrors []float64
	
	// Validation curve data
	parameterName         string
	parameterValues       []float64
	validationCurveTraining   []float64
	validationCurveValidation []float64
	validationCurveTrainingStd []float64
	validationCurveValidationStd []float64
	
	// Prediction interval data
	predictionIntervalX          []float64
	predictionIntervalY          []float64
	confidenceIntervalLower      []float64
	confidenceIntervalUpper      []float64
	predictionIntervalLower      []float64
	predictionIntervalUpper      []float64
	predictionStandardErrors     []float64
	
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

// RecordFeatureImportance records feature names and their coefficients for regression models
func (vc *VisualizationCollector) RecordFeatureImportance(featureNames []string, coefficients []float64, stdErrors []float64) {
	if !vc.enabled {
		return
	}
	
	vc.featureNames = featureNames
	vc.coefficients = coefficients
	vc.featureStdErrors = stdErrors
}

// RecordLearningCurve records learning curve data showing performance vs training set size
func (vc *VisualizationCollector) RecordLearningCurve(trainingSizes []int, trainingScores, validationScores []float64, trainingStdErrors, validationStdErrors []float64) {
	if !vc.enabled {
		return
	}
	
	vc.trainingSizes = trainingSizes
	vc.trainingScores = trainingScores
	vc.validationScores = validationScores
	vc.trainingStdErrors = trainingStdErrors
	vc.validationStdErrors = validationStdErrors
}

// RecordValidationCurve records validation curve data showing performance vs hyperparameter values
func (vc *VisualizationCollector) RecordValidationCurve(parameterName string, parameterValues []float64, trainingScores, validationScores []float64, trainingStdErrors, validationStdErrors []float64) {
	if !vc.enabled {
		return
	}
	
	vc.parameterName = parameterName
	vc.parameterValues = parameterValues
	vc.validationCurveTraining = trainingScores
	vc.validationCurveValidation = validationScores
	vc.validationCurveTrainingStd = trainingStdErrors
	vc.validationCurveValidationStd = validationStdErrors
}

// RecordPredictionInterval records prediction interval data for regression uncertainty visualization
func (vc *VisualizationCollector) RecordPredictionInterval(x, y []float64, confidenceLower, confidenceUpper, predictionLower, predictionUpper, standardErrors []float64) {
	if !vc.enabled {
		return
	}
	
	vc.predictionIntervalX = x
	vc.predictionIntervalY = y
	vc.confidenceIntervalLower = confidenceLower
	vc.confidenceIntervalUpper = confidenceUpper
	vc.predictionIntervalLower = predictionLower
	vc.predictionIntervalUpper = predictionUpper
	vc.predictionStandardErrors = standardErrors
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

// GenerateQQPlot generates Q-Q plot data for validating normal distribution of residuals
func (vc *VisualizationCollector) GenerateQQPlot() PlotData {
	if len(vc.residuals) == 0 {
		return PlotData{}
	}
	
	// Sort residuals
	sortedResiduals := make([]float64, len(vc.residuals))
	copy(sortedResiduals, vc.residuals)
	sort.Float64s(sortedResiduals)
	
	n := len(sortedResiduals)
	qqData := make([]DataPoint, n)
	
	// Calculate sample quantiles and theoretical normal quantiles
	for i := 0; i < n; i++ {
		// Sample quantile (sorted residuals)
		sampleQuantile := sortedResiduals[i]
		
		// Theoretical quantile for standard normal distribution
		// Using approximation for normal quantiles
		p := float64(i+1) / float64(n+1) // Plotting position
		theoreticalQuantile := normalQuantile(p)
		
		qqData[i] = DataPoint{
			X: theoreticalQuantile,
			Y: sampleQuantile,
		}
	}
	
	// Create reference line (perfect normal distribution)
	// Line goes from min to max of theoretical quantiles
	minTheoreticalQ := normalQuantile(1.0 / float64(n+1))
	maxTheoreticalQ := normalQuantile(float64(n) / float64(n+1))
	
	// Calculate slope and intercept for reference line
	// Use sample standard deviation and mean for scaling
	mean, stddev := calculateMeanAndStd(vc.residuals)
	
	referenceLine := []DataPoint{
		{X: minTheoreticalQ, Y: mean + stddev*minTheoreticalQ},
		{X: maxTheoreticalQ, Y: mean + stddev*maxTheoreticalQ},
	}
	
	series := []SeriesData{
		{
			Name: "Sample Quantiles",
			Type: "scatter",
			Data: qqData,
			Style: map[string]interface{}{
				"color": "#4ECDC4",
				"alpha": 0.7,
				"size":  6,
			},
		},
		{
			Name: "Normal Reference Line",
			Type: "line",
			Data: referenceLine,
			Style: map[string]interface{}{
				"color": "#FF6B6B",
				"line_width": 2,
				"line_style": "dashed",
			},
		},
	}
	
	return PlotData{
		PlotType:  QQPlot,
		Title:     fmt.Sprintf("Q-Q Plot - %s", vc.modelName),
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Theoretical Quantiles",
			YAxisLabel:  "Sample Quantiles",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       600,
			Height:      600,
			Interactive: true,
			CustomOptions: map[string]interface{}{
				"subtitle": "Normal Q-Q Plot for Residuals - Tests normality assumption",
			},
		},
		Metrics: map[string]interface{}{
			"sample_mean": mean,
			"sample_std":  stddev,
			"sample_size": n,
		},
	}
}

// GenerateFeatureImportancePlot generates feature importance plot data
func (vc *VisualizationCollector) GenerateFeatureImportancePlot() PlotData {
	if len(vc.coefficients) == 0 || len(vc.featureNames) == 0 {
		return PlotData{}
	}
	
	// Calculate absolute importance (abs coefficient values)
	n := len(vc.coefficients)
	importanceData := make([]DataPoint, n)
	
	// Create pairs of (feature, importance) for sorting
	type featureImportance struct {
		name       string
		coefficient float64
		absValue   float64
		stdError   float64
	}
	
	features := make([]featureImportance, n)
	for i := 0; i < n; i++ {
		stdErr := float64(0)
		if len(vc.featureStdErrors) > i {
			stdErr = vc.featureStdErrors[i]
		}
		
		features[i] = featureImportance{
			name:       vc.featureNames[i],
			coefficient: vc.coefficients[i],
			absValue:   math.Abs(vc.coefficients[i]),
			stdError:   stdErr,
		}
	}
	
	// Sort by absolute importance (descending)
	sort.Slice(features, func(i, j int) bool {
		return features[i].absValue > features[j].absValue
	})
	
	// Create data points for horizontal bar chart
	for i, feat := range features {
		importanceData[i] = DataPoint{
			X:     feat.coefficient,
			Y:     float64(i), // Y position for horizontal bars
			Label: feat.name,
		}
	}
	
	// Calculate confidence intervals if standard errors are available
	var errorBars []DataPoint
	if len(vc.featureStdErrors) > 0 {
		errorBars = make([]DataPoint, n)
		for i, feat := range features {
			// 95% confidence interval (1.96 * standard error)
			margin := 1.96 * feat.stdError
			errorBars[i] = DataPoint{
				X: margin,
				Y: float64(i),
			}
		}
	}
	
	// Create colors based on coefficient sign
	colors := make([]string, n)
	for i, feat := range features {
		if feat.coefficient > 0 {
			colors[i] = "#2ECC71" // Green for positive
		} else {
			colors[i] = "#E74C3C" // Red for negative
		}
	}
	
	series := []SeriesData{
		{
			Name: "Feature Coefficients",
			Type: "bar",
			Data: importanceData,
			Style: map[string]interface{}{
				"orientation": "horizontal",
				"colors": colors, // Pass the pre-calculated colors array
				"alpha": 0.8,
			},
		},
	}
	
	// Add error bars if available
	if len(errorBars) > 0 {
		series = append(series, SeriesData{
			Name: "95% Confidence Interval",
			Type: "errorbar",
			Data: errorBars,
			Style: map[string]interface{}{
				"color": "#34495E",
				"width": 2,
				"capsize": 5,
			},
		})
	}
	
	// Get feature labels in sorted order
	featureLabels := make([]string, n)
	for i, feat := range features {
		featureLabels[i] = feat.name
	}
	
	// Create descriptive title using model name if available
	title := "Feature Importance Analysis"
	if vc.modelName != "" && vc.modelName != "Model" {
		title = vc.modelName + " - Feature Importance"
	}
	
	return PlotData{
		PlotType:  FeatureImportancePlot,
		Title:     title,
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Coefficient Value",
			YAxisLabel:  "Features",
			XAxisScale:  "linear",
			ShowLegend:  len(errorBars) > 0,
			ShowGrid:    true,
			Width:       800,
			Height:      max(400, n*40), // Dynamic height based on feature count
			Interactive: true,
			CustomOptions: map[string]interface{}{
				"yTickLabels": featureLabels,
				"zeroline":    true,
				"subtitle":    "Regression coefficients showing feature contribution",
			},
		},
		Metrics: map[string]interface{}{
			"num_features": n,
			"top_feature":  features[0].name,
			"top_coeff":    features[0].coefficient,
		},
	}
}

// GenerateLearningCurvePlot generates learning curve plot data showing performance vs training set size
func (vc *VisualizationCollector) GenerateLearningCurvePlot() PlotData {
	if len(vc.trainingSizes) == 0 || len(vc.trainingScores) == 0 || len(vc.validationScores) == 0 {
		return PlotData{}
	}
	
	n := len(vc.trainingSizes)
	
	// Create training score data points
	trainingData := make([]DataPoint, n)
	for i := 0; i < n; i++ {
		trainingData[i] = DataPoint{
			X: float64(vc.trainingSizes[i]),
			Y: vc.trainingScores[i],
		}
	}
	
	// Create validation score data points
	validationData := make([]DataPoint, n)
	for i := 0; i < n; i++ {
		validationData[i] = DataPoint{
			X: float64(vc.trainingSizes[i]),
			Y: vc.validationScores[i],
		}
	}
	
	series := []SeriesData{
		{
			Name: "Training Score",
			Type: "line",
			Data: trainingData,
			Style: map[string]interface{}{
				"color":      "#2ECC71",
				"line_width": 2,
				"marker":     "circle",
				"alpha":      0.8,
			},
		},
		{
			Name: "Validation Score",
			Type: "line",
			Data: validationData,
			Style: map[string]interface{}{
				"color":      "#E74C3C",
				"line_width": 2,
				"marker":     "circle",
				"alpha":      0.8,
			},
		},
	}
	
	// Add error bands if standard errors are available
	if len(vc.trainingStdErrors) > 0 && len(vc.validationStdErrors) > 0 {
		// Training error band (upper and lower bounds)
		trainingUpperBand := make([]DataPoint, n)
		trainingLowerBand := make([]DataPoint, n)
		for i := 0; i < n; i++ {
			trainingUpperBand[i] = DataPoint{
				X: float64(vc.trainingSizes[i]),
				Y: vc.trainingScores[i] + vc.trainingStdErrors[i],
			}
			trainingLowerBand[i] = DataPoint{
				X: float64(vc.trainingSizes[i]),
				Y: vc.trainingScores[i] - vc.trainingStdErrors[i],
			}
		}
		
		// Validation error band
		validationUpperBand := make([]DataPoint, n)
		validationLowerBand := make([]DataPoint, n)
		for i := 0; i < n; i++ {
			validationUpperBand[i] = DataPoint{
				X: float64(vc.trainingSizes[i]),
				Y: vc.validationScores[i] + vc.validationStdErrors[i],
			}
			validationLowerBand[i] = DataPoint{
				X: float64(vc.trainingSizes[i]),
				Y: vc.validationScores[i] - vc.validationStdErrors[i],
			}
		}
		
		// Add error band series
		series = append(series, SeriesData{
			Name: "Training Error Band",
			Type: "fill",
			Data: append(trainingUpperBand, reverse(trainingLowerBand)...),
			Style: map[string]interface{}{
				"color": "#2ECC71",
				"alpha": 0.2,
				"fill":  "tonexty",
			},
		})
		
		series = append(series, SeriesData{
			Name: "Validation Error Band",
			Type: "fill",
			Data: append(validationUpperBand, reverse(validationLowerBand)...),
			Style: map[string]interface{}{
				"color": "#E74C3C",
				"alpha": 0.2,
				"fill":  "tonexty",
			},
		})
	}
	
	// Calculate analysis metrics
	finalTrainingScore := vc.trainingScores[n-1]
	finalValidationScore := vc.validationScores[n-1]
	gap := math.Abs(finalTrainingScore - finalValidationScore)
	
	// Determine learning curve characteristics
	var diagnosis string
	if gap > 0.1 && finalTrainingScore > finalValidationScore {
		diagnosis = "Overfitting detected"
	} else if finalTrainingScore < 0.7 && finalValidationScore < 0.7 {
		diagnosis = "Underfitting detected"
	} else if gap < 0.05 {
		diagnosis = "Good fit"
	} else {
		diagnosis = "Slight overfitting"
	}
	
	return PlotData{
		PlotType:  LearningCurvePlot,
		Title:     "Learning Curve Analysis",
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Training Set Size",
			YAxisLabel:  "Score",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       800,
			Height:      600,
			Interactive: true,
			CustomOptions: map[string]interface{}{
				"subtitle": "Performance vs Training Set Size - Diagnose overfitting/underfitting",
				"diagnosis": diagnosis,
			},
		},
		Metrics: map[string]interface{}{
			"final_training_score":   finalTrainingScore,
			"final_validation_score": finalValidationScore,
			"training_val_gap":       gap,
			"num_training_sizes":     n,
			"diagnosis":             diagnosis,
		},
	}
}

// GenerateValidationCurvePlot generates validation curve plot data showing performance vs hyperparameter values
func (vc *VisualizationCollector) GenerateValidationCurvePlot() PlotData {
	// Create descriptive title using model name if available
	title := "Validation Curve Analysis"
	if vc.modelName != "" && vc.modelName != "Model" {
		title = vc.modelName + " - Validation Curve"
	}
	
	if len(vc.parameterValues) == 0 || len(vc.validationCurveTraining) == 0 {
		return PlotData{
			PlotType:  ValidationCurvePlot,
			Title:     title,
			Timestamp: time.Now(),
			ModelName: vc.modelName,
			Series:    []SeriesData{},
		}
	}
	
	n := len(vc.parameterValues)
	
	// Helper function to reverse slice for error band plotting
	reverse := func(data []DataPoint) []DataPoint {
		reversed := make([]DataPoint, len(data))
		for i, point := range data {
			reversed[len(data)-1-i] = point
		}
		return reversed
	}
	
	// Create training score series
	trainingSeries := SeriesData{
		Name: "Training Score",
		Type: "line",
		Data: make([]DataPoint, n),
		Style: map[string]interface{}{
			"color": "#2ECC71",
			"width": 2,
		},
	}
	
	for i := 0; i < n; i++ {
		trainingSeries.Data[i] = DataPoint{
			X: vc.parameterValues[i],
			Y: vc.validationCurveTraining[i],
		}
	}
	
	// Create validation score series
	validationSeries := SeriesData{
		Name: "Validation Score",
		Type: "line",
		Data: make([]DataPoint, n),
		Style: map[string]interface{}{
			"color": "#E74C3C",
			"width": 2,
		},
	}
	
	for i := 0; i < n; i++ {
		validationSeries.Data[i] = DataPoint{
			X: vc.parameterValues[i],
			Y: vc.validationCurveValidation[i],
		}
	}
	
	series := []SeriesData{trainingSeries, validationSeries}
	
	// Add error bands if standard errors are available
	if len(vc.validationCurveTrainingStd) == n && len(vc.validationCurveValidationStd) == n {
		// Training error band
		trainingUpperBand := make([]DataPoint, n)
		trainingLowerBand := make([]DataPoint, n)
		
		for i := 0; i < n; i++ {
			upper := vc.validationCurveTraining[i] + vc.validationCurveTrainingStd[i]
			lower := vc.validationCurveTraining[i] - vc.validationCurveTrainingStd[i]
			
			trainingUpperBand[i] = DataPoint{X: vc.parameterValues[i], Y: upper}
			trainingLowerBand[i] = DataPoint{X: vc.parameterValues[i], Y: lower}
		}
		
		series = append(series, SeriesData{
			Name: "Training Error Band",
			Type: "fill",
			Data: append(trainingUpperBand, reverse(trainingLowerBand)...),
			Style: map[string]interface{}{
				"color": "#2ECC71",
				"alpha": 0.2,
				"fill":  "tonexty",
			},
		})
		
		// Validation error band
		validationUpperBand := make([]DataPoint, n)
		validationLowerBand := make([]DataPoint, n)
		
		for i := 0; i < n; i++ {
			upper := vc.validationCurveValidation[i] + vc.validationCurveValidationStd[i]
			lower := vc.validationCurveValidation[i] - vc.validationCurveValidationStd[i]
			
			validationUpperBand[i] = DataPoint{X: vc.parameterValues[i], Y: upper}
			validationLowerBand[i] = DataPoint{X: vc.parameterValues[i], Y: lower}
		}
		
		series = append(series, SeriesData{
			Name: "Validation Error Band",
			Type: "fill",
			Data: append(validationUpperBand, reverse(validationLowerBand)...),
			Style: map[string]interface{}{
				"color": "#E74C3C",
				"alpha": 0.2,
				"fill":  "tonexty",
			},
		})
	}
	
	// Find optimal parameter value (highest validation score)
	optimalIndex := 0
	maxValidationScore := vc.validationCurveValidation[0]
	for i := 1; i < n; i++ {
		if vc.validationCurveValidation[i] > maxValidationScore {
			maxValidationScore = vc.validationCurveValidation[i]
			optimalIndex = i
		}
	}
	
	optimalValue := vc.parameterValues[optimalIndex]
	
	// Calculate analysis metrics
	finalTrainingScore := vc.validationCurveTraining[n-1]
	finalValidationScore := vc.validationCurveValidation[n-1]
	gap := math.Abs(finalTrainingScore - finalValidationScore)
	
	// Determine validation curve characteristics
	var diagnosis string
	if maxValidationScore == vc.validationCurveValidation[0] || maxValidationScore == vc.validationCurveValidation[n-1] {
		diagnosis = "Optimal value at boundary - consider expanding search range"
	} else if gap > 0.1 {
		diagnosis = "Large training-validation gap - possible overfitting"
	} else if maxValidationScore < 0.7 {
		diagnosis = "Low performance - consider different hyperparameter ranges"
	} else {
		diagnosis = "Good hyperparameter range found"
	}
	
	return PlotData{
		PlotType:  ValidationCurvePlot,
		Title:     title,
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  vc.parameterName,
			YAxisLabel:  "Score",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       800,
			Height:      600,
			Interactive: true,
			CustomOptions: map[string]interface{}{
				"subtitle": fmt.Sprintf("Performance vs %s - Hyperparameter tuning", vc.parameterName),
				"diagnosis": diagnosis,
				"optimal_value": optimalValue,
			},
		},
		Metrics: map[string]interface{}{
			"parameter_name":        vc.parameterName,
			"optimal_value":         optimalValue,
			"optimal_validation_score": maxValidationScore,
			"final_training_score":  finalTrainingScore,
			"final_validation_score": finalValidationScore,
			"training_val_gap":      gap,
			"num_parameter_values":  n,
			"diagnosis":            diagnosis,
		},
	}
}

// GeneratePredictionIntervalPlot generates prediction interval plot data showing prediction uncertainty
func (vc *VisualizationCollector) GeneratePredictionIntervalPlot() PlotData {
	// Create descriptive title using model name if available
	title := "Prediction Interval Analysis"
	if vc.modelName != "" && vc.modelName != "Model" {
		title = vc.modelName + " - Prediction Intervals"
	}
	
	if len(vc.predictionIntervalX) == 0 || len(vc.predictionIntervalY) == 0 {
		return PlotData{
			PlotType:  PredictionIntervalPlot,
			Title:     title,
			Timestamp: time.Now(),
			ModelName: vc.modelName,
			Series:    []SeriesData{},
		}
	}
	
	n := len(vc.predictionIntervalX)
	
	// Create the main prediction line
	predictionData := make([]DataPoint, n)
	for i := 0; i < n; i++ {
		predictionData[i] = DataPoint{
			X: vc.predictionIntervalX[i],
			Y: vc.predictionIntervalY[i],
		}
	}
	
	series := []SeriesData{
		{
			Name: "Predictions",
			Type: "line",
			Data: predictionData,
			Style: map[string]interface{}{
				"color": "#2E86AB",
				"width": 2,
			},
		},
	}
	
	// Add confidence interval if available
	if len(vc.confidenceIntervalLower) == n && len(vc.confidenceIntervalUpper) == n {
		// Create confidence interval band
		confidenceData := make([]DataPoint, 2*n)
		
		// Upper band
		for i := 0; i < n; i++ {
			confidenceData[i] = DataPoint{
				X: vc.predictionIntervalX[i],
				Y: vc.confidenceIntervalUpper[i],
			}
		}
		
		// Lower band (reversed order for fill)
		for i := 0; i < n; i++ {
			confidenceData[n+i] = DataPoint{
				X: vc.predictionIntervalX[n-1-i],
				Y: vc.confidenceIntervalLower[n-1-i],
			}
		}
		
		series = append(series, SeriesData{
			Name: "Confidence Interval (95%)",
			Type: "fill",
			Data: confidenceData,
			Style: map[string]interface{}{
				"color": "#2E86AB",
				"alpha": 0.3,
				"fill":  "tonexty",
			},
		})
	}
	
	// Add prediction interval if available
	if len(vc.predictionIntervalLower) == n && len(vc.predictionIntervalUpper) == n {
		// Create prediction interval band
		predictionBandData := make([]DataPoint, 2*n)
		
		// Upper band
		for i := 0; i < n; i++ {
			predictionBandData[i] = DataPoint{
				X: vc.predictionIntervalX[i],
				Y: vc.predictionIntervalUpper[i],
			}
		}
		
		// Lower band (reversed order for fill)
		for i := 0; i < n; i++ {
			predictionBandData[n+i] = DataPoint{
				X: vc.predictionIntervalX[n-1-i],
				Y: vc.predictionIntervalLower[n-1-i],
			}
		}
		
		series = append(series, SeriesData{
			Name: "Prediction Interval (95%)",
			Type: "fill",
			Data: predictionBandData,
			Style: map[string]interface{}{
				"color": "#F24236",
				"alpha": 0.2,
				"fill":  "tonexty",
			},
		})
	}
	
	// Calculate diagnostic metrics
	var meanStdError float64
	if len(vc.predictionStandardErrors) > 0 {
		sum := 0.0
		for _, se := range vc.predictionStandardErrors {
			sum += se
		}
		meanStdError = sum / float64(len(vc.predictionStandardErrors))
	}
	
	// Determine prediction reliability
	var reliability string
	if meanStdError < 0.05 {
		reliability = "High confidence predictions"
	} else if meanStdError < 0.15 {
		reliability = "Moderate prediction uncertainty"
	} else {
		reliability = "High prediction uncertainty"
	}
	
	// Calculate interval widths for analysis
	var avgConfidenceWidth, avgPredictionWidth float64
	if len(vc.confidenceIntervalLower) == n && len(vc.confidenceIntervalUpper) == n {
		for i := 0; i < n; i++ {
			avgConfidenceWidth += vc.confidenceIntervalUpper[i] - vc.confidenceIntervalLower[i]
		}
		avgConfidenceWidth /= float64(n)
	}
	
	if len(vc.predictionIntervalLower) == n && len(vc.predictionIntervalUpper) == n {
		for i := 0; i < n; i++ {
			avgPredictionWidth += vc.predictionIntervalUpper[i] - vc.predictionIntervalLower[i]
		}
		avgPredictionWidth /= float64(n)
	}
	
	return PlotData{
		PlotType:  PredictionIntervalPlot,
		Title:     title,
		Timestamp: time.Now(),
		ModelName: vc.modelName,
		Series:    series,
		Config: PlotConfig{
			XAxisLabel:  "Input Values",
			YAxisLabel:  "Predictions",
			XAxisScale:  "linear",
			YAxisScale:  "linear",
			ShowLegend:  true,
			ShowGrid:    true,
			Width:       800,
			Height:      600,
			Interactive: true,
			CustomOptions: map[string]interface{}{
				"subtitle": "Prediction uncertainty with confidence and prediction intervals",
				"reliability": reliability,
				"mean_std_error": meanStdError,
			},
		},
		Metrics: map[string]interface{}{
			"num_predictions":         n,
			"mean_standard_error":     meanStdError,
			"avg_confidence_width":    avgConfidenceWidth,
			"avg_prediction_width":    avgPredictionWidth,
			"reliability":            reliability,
		},
	}
}

// reverse reverses a slice of DataPoint for fill_between functionality
func reverse(data []DataPoint) []DataPoint {
	reversed := make([]DataPoint, len(data))
	for i, j := 0, len(data)-1; i < len(data); i, j = i+1, j-1 {
		reversed[i] = data[j]
	}
	return reversed
}

// normalQuantile calculates approximate normal quantile for probability p
// Uses Beasley-Springer-Moro algorithm approximation
func normalQuantile(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	
	// For p around 0.5, use direct calculation
	if p == 0.5 {
		return 0.0
	}
	
	// Use inverse error function approximation
	// This is a simplified approximation for the normal quantile function
	if p < 0.5 {
		// For lower tail, use symmetry
		return -normalQuantile(1.0 - p)
	}
	
	// Rational approximation for upper tail
	t := math.Sqrt(-2.0 * math.Log(1.0-p))
	
	// Coefficients for the approximation
	c0 := 2.515517
	c1 := 0.802853
	c2 := 0.010328
	d1 := 1.432788
	d2 := 0.189269
	d3 := 0.001308
	
	numerator := c0 + c1*t + c2*t*t
	denominator := 1.0 + d1*t + d2*t*t + d3*t*t*t
	
	return t - numerator/denominator
}

// calculateMeanAndStd calculates mean and standard deviation of a slice
func calculateMeanAndStd(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	
	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	
	// Calculate standard deviation
	sumSquaredDiffs := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiffs += diff * diff
	}
	variance := sumSquaredDiffs / float64(len(values)-1) // Sample standard deviation
	stddev := math.Sqrt(variance)
	
	return mean, stddev
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