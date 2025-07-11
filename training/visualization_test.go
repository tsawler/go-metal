package training

import (
	"encoding/json"
	"math"
	"testing"
	"time"
)

// TestPlotType tests PlotType constants and usage
func TestPlotType(t *testing.T) {
	expectedTypes := map[PlotType]string{
		TrainingCurves:         "training_curves",
		LearningRateSchedule:   "learning_rate_schedule",
		ROCCurve:              "roc_curve",
		PrecisionRecall:       "precision_recall",
		ConfusionMatrixPlot:   "confusion_matrix",
		ParameterDistribution: "parameter_distribution",
		GradientHistogram:     "gradient_histogram",
		ActivationPattern:     "activation_pattern",
		RegressionScatter:     "regression_scatter",
		ResidualPlot:          "residual_plot",
		QQPlot:               "qq_plot",
		FeatureImportancePlot: "feature_importance",
		LearningCurvePlot:     "learning_curve",
		ValidationCurvePlot:   "validation_curve",
		PredictionIntervalPlot: "prediction_interval",
		FeatureCorrelationPlot: "feature_correlation",
		PartialDependencePlot:  "partial_dependence",
	}

	for plotType, expectedString := range expectedTypes {
		if string(plotType) != expectedString {
			t.Errorf("PlotType %v should equal %s, got %s", plotType, expectedString, string(plotType))
		}
	}
}

// TestNewVisualizationCollector tests collector creation
func TestNewVisualizationCollector(t *testing.T) {
	modelName := "TestModel"
	vc := NewVisualizationCollector(modelName)

	if vc.modelName != modelName {
		t.Errorf("Expected model name %s, got %s", modelName, vc.modelName)
	}

	if vc.enabled {
		t.Error("Expected collector to be disabled by default")
	}

	// Check that all slices are initialized
	if vc.trainingLoss == nil {
		t.Error("Training loss slice should be initialized")
	}
	if vc.trainingAccuracy == nil {
		t.Error("Training accuracy slice should be initialized")
	}
	if vc.parameterStats == nil {
		t.Error("Parameter stats map should be initialized")
	}
	if vc.gradientStats == nil {
		t.Error("Gradient stats map should be initialized")
	}
	if vc.activationStats == nil {
		t.Error("Activation stats map should be initialized")
	}
}

// TestVisualizationCollectorEnableDisable tests enable/disable functionality
func TestVisualizationCollectorEnableDisable(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")

	// Initially disabled
	if vc.IsEnabled() {
		t.Error("Collector should be disabled initially")
	}

	// Enable
	vc.Enable()
	if !vc.IsEnabled() {
		t.Error("Collector should be enabled after Enable()")
	}

	// Disable
	vc.Disable()
	if vc.IsEnabled() {
		t.Error("Collector should be disabled after Disable()")
	}
}

// TestRecordTrainingStep tests training step recording
func TestRecordTrainingStep(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")

	// Should not record when disabled
	vc.RecordTrainingStep(1, 0.5, 0.8, 0.01)
	if len(vc.steps) != 0 {
		t.Error("Should not record data when disabled")
	}

	// Enable and record
	vc.Enable()
	vc.RecordTrainingStep(1, 0.5, 0.8, 0.01)
	vc.RecordTrainingStep(2, 0.4, 0.85, 0.009)

	if len(vc.steps) != 2 {
		t.Errorf("Expected 2 steps, got %d", len(vc.steps))
	}
	if len(vc.trainingLoss) != 2 {
		t.Errorf("Expected 2 loss values, got %d", len(vc.trainingLoss))
	}
	if len(vc.trainingAccuracy) != 2 {
		t.Errorf("Expected 2 accuracy values, got %d", len(vc.trainingAccuracy))
	}
	if len(vc.learningRates) != 2 {
		t.Errorf("Expected 2 learning rate values, got %d", len(vc.learningRates))
	}

	// Check values
	if vc.steps[0] != 1 || vc.steps[1] != 2 {
		t.Errorf("Steps values incorrect: %v", vc.steps)
	}
	if vc.trainingLoss[0] != 0.5 || vc.trainingLoss[1] != 0.4 {
		t.Errorf("Training loss values incorrect: %v", vc.trainingLoss)
	}
	if vc.trainingAccuracy[0] != 0.8 || vc.trainingAccuracy[1] != 0.85 {
		t.Errorf("Training accuracy values incorrect: %v", vc.trainingAccuracy)
	}
	if vc.learningRates[0] != 0.01 || vc.learningRates[1] != 0.009 {
		t.Errorf("Learning rate values incorrect: %v", vc.learningRates)
	}
}

// TestRecordValidationStep tests validation step recording
func TestRecordValidationStep(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	vc.RecordValidationStep(1, 0.6, 0.75)
	vc.RecordValidationStep(2, 0.55, 0.78)

	if len(vc.validationLoss) != 2 {
		t.Errorf("Expected 2 validation loss values, got %d", len(vc.validationLoss))
	}
	if len(vc.validationAccuracy) != 2 {
		t.Errorf("Expected 2 validation accuracy values, got %d", len(vc.validationAccuracy))
	}

	if vc.validationLoss[0] != 0.6 || vc.validationLoss[1] != 0.55 {
		t.Errorf("Validation loss values incorrect: %v", vc.validationLoss)
	}
	if vc.validationAccuracy[0] != 0.75 || vc.validationAccuracy[1] != 0.78 {
		t.Errorf("Validation accuracy values incorrect: %v", vc.validationAccuracy)
	}
}

// TestRecordROCData tests ROC data recording
func TestRecordROCData(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	rocPoints := []ROCPointViz{
		{FPR: 0.0, TPR: 0.0, Threshold: 1.0},
		{FPR: 0.2, TPR: 0.8, Threshold: 0.5},
		{FPR: 1.0, TPR: 1.0, Threshold: 0.0},
	}

	vc.RecordROCData(rocPoints)

	if len(vc.rocPoints) != 3 {
		t.Errorf("Expected 3 ROC points, got %d", len(vc.rocPoints))
	}
	if vc.rocPoints[1].FPR != 0.2 || vc.rocPoints[1].TPR != 0.8 {
		t.Errorf("ROC point values incorrect: %+v", vc.rocPoints[1])
	}
}

// TestRecordConfusionMatrix tests confusion matrix recording
func TestRecordConfusionMatrix(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	matrix := [][]int{
		{50, 5},
		{10, 35},
	}
	classNames := []string{"Class0", "Class1"}

	vc.RecordConfusionMatrix(matrix, classNames)

	if len(vc.confusionMatrix) != 2 {
		t.Errorf("Expected 2x2 matrix, got %d rows", len(vc.confusionMatrix))
	}
	if len(vc.classNames) != 2 {
		t.Errorf("Expected 2 class names, got %d", len(vc.classNames))
	}
	if vc.confusionMatrix[0][0] != 50 {
		t.Errorf("Matrix value incorrect: expected 50, got %d", vc.confusionMatrix[0][0])
	}
	if vc.classNames[0] != "Class0" {
		t.Errorf("Class name incorrect: expected Class0, got %s", vc.classNames[0])
	}
}

// TestRecordRegressionData tests regression data recording
func TestRecordRegressionData(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	predictions := []float64{1.1, 2.2, 3.3}
	trueValues := []float64{1.0, 2.0, 3.0}

	vc.RecordRegressionData(predictions, trueValues)

	if len(vc.predictions) != 3 {
		t.Errorf("Expected 3 predictions, got %d", len(vc.predictions))
	}
	if len(vc.trueValues) != 3 {
		t.Errorf("Expected 3 true values, got %d", len(vc.trueValues))
	}
	if len(vc.residuals) != 3 {
		t.Errorf("Expected 3 residuals, got %d", len(vc.residuals))
	}

	// Check residual calculation
	expectedResiduals := []float64{0.1, 0.2, 0.3}
	for i, expected := range expectedResiduals {
		if math.Abs(vc.residuals[i]-expected) > 1e-9 {
			t.Errorf("Residual %d: expected %f, got %f", i, expected, vc.residuals[i])
		}
	}
}

// TestRecordFeatureImportance tests feature importance recording
func TestRecordFeatureImportance(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	features := []string{"feature1", "feature2", "feature3"}
	coeffs := []float64{0.5, -0.3, 0.8}
	stdErrors := []float64{0.1, 0.05, 0.15}

	vc.RecordFeatureImportance(features, coeffs, stdErrors)

	if len(vc.featureNames) != 3 {
		t.Errorf("Expected 3 feature names, got %d", len(vc.featureNames))
	}
	if len(vc.coefficients) != 3 {
		t.Errorf("Expected 3 coefficients, got %d", len(vc.coefficients))
	}
	if len(vc.featureStdErrors) != 3 {
		t.Errorf("Expected 3 standard errors, got %d", len(vc.featureStdErrors))
	}

	if vc.featureNames[1] != "feature2" {
		t.Errorf("Feature name incorrect: expected feature2, got %s", vc.featureNames[1])
	}
	if vc.coefficients[2] != 0.8 {
		t.Errorf("Coefficient incorrect: expected 0.8, got %f", vc.coefficients[2])
	}
}

// TestGenerateTrainingCurvesPlot tests training curves plot generation
func TestGenerateTrainingCurvesPlot(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	// Add some training data
	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
	vc.RecordTrainingStep(2, 0.6, 0.8, 0.009)
	vc.RecordValidationStep(1, 0.7, 0.7)
	vc.RecordValidationStep(2, 0.65, 0.75)

	plot := vc.GenerateTrainingCurvesPlot()

	if plot.PlotType != TrainingCurves {
		t.Errorf("Expected plot type %s, got %s", TrainingCurves, plot.PlotType)
	}
	if plot.ModelName != "TestModel" {
		t.Errorf("Expected model name TestModel, got %s", plot.ModelName)
	}

	// Should have at least training loss and accuracy series
	if len(plot.Series) < 2 {
		t.Errorf("Expected at least 2 series, got %d", len(plot.Series))
	}

	// Check training loss series
	trainingLossSeries := plot.Series[0]
	if trainingLossSeries.Name != "Training Loss" {
		t.Errorf("Expected Training Loss series, got %s", trainingLossSeries.Name)
	}
	if len(trainingLossSeries.Data) != 2 {
		t.Errorf("Expected 2 data points, got %d", len(trainingLossSeries.Data))
	}

	// Check validation series are included
	if len(plot.Series) >= 4 {
		valLossSeries := plot.Series[2]
		if valLossSeries.Name != "Validation Loss" {
			t.Errorf("Expected Validation Loss series, got %s", valLossSeries.Name)
		}
	}
}

// TestGenerateLearningRateSchedulePlot tests learning rate schedule plot generation
func TestGenerateLearningRateSchedulePlot(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
	vc.RecordTrainingStep(2, 0.6, 0.8, 0.005)

	plot := vc.GenerateLearningRateSchedulePlot()

	if plot.PlotType != LearningRateSchedule {
		t.Errorf("Expected plot type %s, got %s", LearningRateSchedule, plot.PlotType)
	}

	if len(plot.Series) != 1 {
		t.Errorf("Expected 1 series, got %d", len(plot.Series))
	}

	lrSeries := plot.Series[0]
	if lrSeries.Name != "Learning Rate" {
		t.Errorf("Expected Learning Rate series, got %s", lrSeries.Name)
	}
	if len(lrSeries.Data) != 2 {
		t.Errorf("Expected 2 data points, got %d", len(lrSeries.Data))
	}

	// Check Y-axis is log scale
	if plot.Config.YAxisScale != "log" {
		t.Errorf("Expected log Y-axis scale, got %s", plot.Config.YAxisScale)
	}
}

// TestGenerateROCCurvePlot tests ROC curve plot generation
func TestGenerateROCCurvePlot(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	rocPoints := []ROCPointViz{
		{FPR: 0.0, TPR: 0.0, Threshold: 1.0},
		{FPR: 0.2, TPR: 0.8, Threshold: 0.5},
		{FPR: 1.0, TPR: 1.0, Threshold: 0.0},
	}
	vc.RecordROCData(rocPoints)

	plot := vc.GenerateROCCurvePlot()

	if plot.PlotType != ROCCurve {
		t.Errorf("Expected plot type %s, got %s", ROCCurve, plot.PlotType)
	}

	if len(plot.Series) != 2 {
		t.Errorf("Expected 2 series (ROC + Random), got %d", len(plot.Series))
	}

	rocSeries := plot.Series[0]
	if rocSeries.Name != "ROC Curve" {
		t.Errorf("Expected ROC Curve series, got %s", rocSeries.Name)
	}
	if len(rocSeries.Data) != 3 {
		t.Errorf("Expected 3 data points, got %d", len(rocSeries.Data))
	}

	randomSeries := plot.Series[1]
	if randomSeries.Name != "Random Classifier" {
		t.Errorf("Expected Random Classifier series, got %s", randomSeries.Name)
	}
}

// TestGenerateConfusionMatrixPlot tests confusion matrix plot generation
func TestGenerateConfusionMatrixPlot(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	matrix := [][]int{
		{50, 5},
		{10, 35},
	}
	classNames := []string{"Class0", "Class1"}
	vc.RecordConfusionMatrix(matrix, classNames)

	plot := vc.GenerateConfusionMatrixPlot()

	if plot.PlotType != ConfusionMatrixPlot {
		t.Errorf("Expected plot type %s, got %s", ConfusionMatrixPlot, plot.PlotType)
	}

	if len(plot.Series) != 1 {
		t.Errorf("Expected 1 series, got %d", len(plot.Series))
	}

	heatmapSeries := plot.Series[0]
	if heatmapSeries.Type != "heatmap" {
		t.Errorf("Expected heatmap series type, got %s", heatmapSeries.Type)
	}
	if len(heatmapSeries.Data) != 4 { // 2x2 matrix = 4 points
		t.Errorf("Expected 4 data points, got %d", len(heatmapSeries.Data))
	}

	// Check one data point
	point := heatmapSeries.Data[0]
	if point.Z != 50 {
		t.Errorf("Expected Z value 50, got %v", point.Z)
	}
}

// TestGenerateRegressionScatterPlot tests regression scatter plot generation
func TestGenerateRegressionScatterPlot(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	predictions := []float64{1.1, 2.2, 3.3}
	trueValues := []float64{1.0, 2.0, 3.0}
	vc.RecordRegressionData(predictions, trueValues)

	plot := vc.GenerateRegressionScatterPlot()

	if plot.PlotType != RegressionScatter {
		t.Errorf("Expected plot type %s, got %s", RegressionScatter, plot.PlotType)
	}

	if len(plot.Series) != 2 {
		t.Errorf("Expected 2 series (scatter + perfect line), got %d", len(plot.Series))
	}

	scatterSeries := plot.Series[0]
	if scatterSeries.Type != "scatter" {
		t.Errorf("Expected scatter series type, got %s", scatterSeries.Type)
	}
	if len(scatterSeries.Data) != 3 {
		t.Errorf("Expected 3 data points, got %d", len(scatterSeries.Data))
	}

	lineSeries := plot.Series[1]
	if lineSeries.Name != "Perfect Prediction" {
		t.Errorf("Expected Perfect Prediction series, got %s", lineSeries.Name)
	}
}

// TestGenerateFeatureImportancePlot tests feature importance plot generation
func TestGenerateFeatureImportancePlot(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	features := []string{"feature1", "feature2", "feature3"}
	coeffs := []float64{0.5, -0.8, 0.3} // feature2 should be most important
	stdErrors := []float64{0.1, 0.05, 0.15}
	vc.RecordFeatureImportance(features, coeffs, stdErrors)

	plot := vc.GenerateFeatureImportancePlot()

	if plot.PlotType != FeatureImportancePlot {
		t.Errorf("Expected plot type %s, got %s", FeatureImportancePlot, plot.PlotType)
	}

	// Should have bar chart and error bars
	if len(plot.Series) < 1 {
		t.Errorf("Expected at least 1 series, got %d", len(plot.Series))
	}

	barSeries := plot.Series[0]
	if barSeries.Type != "bar" {
		t.Errorf("Expected bar series type, got %s", barSeries.Type)
	}
	if len(barSeries.Data) != 3 {
		t.Errorf("Expected 3 data points, got %d", len(barSeries.Data))
	}

	// Check sorting (feature2 with coeff -0.8 should be first due to abs value)
	firstPoint := barSeries.Data[0]
	if firstPoint.X != -0.8 {
		t.Errorf("Expected most important feature (-0.8), got %v", firstPoint.X)
	}
}

// TestGenerateEmptyPlots tests plot generation with empty data
func TestGenerateEmptyPlots(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	// Test empty confusion matrix
	confusionPlot := vc.GenerateConfusionMatrixPlot()
	if len(confusionPlot.Series) != 0 {
		t.Error("Empty confusion matrix should return empty plot")
	}

	// Test empty regression scatter
	regressionPlot := vc.GenerateRegressionScatterPlot()
	if len(regressionPlot.Series) != 0 {
		t.Error("Empty regression data should return empty plot")
	}

	// Test empty feature importance
	featurePlot := vc.GenerateFeatureImportancePlot()
	if len(featurePlot.Series) != 0 {
		t.Error("Empty feature data should return empty plot")
	}
}

// TestPlotDataToJSON tests JSON serialization
func TestPlotDataToJSON(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
	plot := vc.GenerateTrainingCurvesPlot()

	jsonStr, err := plot.ToJSON()
	if err != nil {
		t.Fatalf("Failed to convert to JSON: %v", err)
	}

	// Test that we can parse it back
	var parsedPlot PlotData
	err = json.Unmarshal([]byte(jsonStr), &parsedPlot)
	if err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}

	if parsedPlot.PlotType != plot.PlotType {
		t.Errorf("PlotType mismatch after JSON round-trip")
	}
	if parsedPlot.ModelName != plot.ModelName {
		t.Errorf("ModelName mismatch after JSON round-trip")
	}
}

// TestVisualizationCollectorClear tests data clearing
func TestVisualizationCollectorClear(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	// Add some data
	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
	vc.RecordValidationStep(1, 0.7, 0.7)
	vc.RecordConfusionMatrix([][]int{{1, 2}, {3, 4}}, []string{"A", "B"})
	vc.RecordRegressionData([]float64{1.0}, []float64{1.1})

	// Clear
	vc.Clear()

	// Check all data is cleared
	if len(vc.trainingLoss) != 0 {
		t.Error("Training loss should be cleared")
	}
	if len(vc.validationLoss) != 0 {
		t.Error("Validation loss should be cleared")
	}
	if len(vc.steps) != 0 {
		t.Error("Steps should be cleared")
	}
	if vc.confusionMatrix != nil {
		t.Error("Confusion matrix should be cleared")
	}
	if len(vc.predictions) != 0 {
		t.Error("Predictions should be cleared")
	}
	if len(vc.parameterStats) != 0 {
		t.Error("Parameter stats should be cleared")
	}
}

// TestRecordParameterStats tests parameter statistics recording
func TestRecordParameterStats(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	stats := ParameterStats{
		LayerName: "layer1",
		ParamType: "weight",
		Mean:      0.1,
		Std:       0.2,
		Min:       -0.5,
		Max:       0.8,
		Histogram: []float64{1, 2, 3, 4, 5},
		Bins:      []float64{-0.5, -0.25, 0, 0.25, 0.5, 0.8},
	}

	vc.RecordParameterStats("layer1", "weight", stats)

	key := "layer1_weight"
	if _, exists := vc.parameterStats[key]; !exists {
		t.Error("Parameter stats should be recorded")
	}

	stored := vc.parameterStats[key]
	if stored.Mean != 0.1 {
		t.Errorf("Expected mean 0.1, got %f", stored.Mean)
	}
	if len(stored.Histogram) != 5 {
		t.Errorf("Expected histogram length 5, got %d", len(stored.Histogram))
	}
}

// TestRecordGradientStats tests gradient statistics recording
func TestRecordGradientStats(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	stats := GradientStats{
		LayerName:    "layer1",
		ParamType:    "bias",
		GradientNorm: 0.05,
		Mean:         0.001,
		Std:          0.01,
		Histogram:    []float64{10, 20, 30},
		Bins:         []float64{-0.1, 0, 0.1, 0.2},
	}

	vc.RecordGradientStats("layer1", "bias", stats)

	key := "layer1_bias"
	if _, exists := vc.gradientStats[key]; !exists {
		t.Error("Gradient stats should be recorded")
	}

	stored := vc.gradientStats[key]
	if stored.GradientNorm != 0.05 {
		t.Errorf("Expected gradient norm 0.05, got %f", stored.GradientNorm)
	}
}

// TestRecordActivationStats tests activation statistics recording
func TestRecordActivationStats(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	stats := ActivationStats{
		LayerName:     "layer1",
		ActivationType: "relu",
		Mean:          0.5,
		Std:           0.3,
		SparsityRatio: 0.2,
		Histogram:     []float64{5, 10, 15, 20},
		Bins:          []float64{0, 0.25, 0.5, 0.75, 1.0},
	}

	vc.RecordActivationStats("layer1", "relu", stats)

	key := "layer1_relu"
	if _, exists := vc.activationStats[key]; !exists {
		t.Error("Activation stats should be recorded")
	}

	stored := vc.activationStats[key]
	if stored.SparsityRatio != 0.2 {
		t.Errorf("Expected sparsity ratio 0.2, got %f", stored.SparsityRatio)
	}
}

// TestDisabledRecording tests that recording is ignored when disabled
func TestDisabledRecording(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	// Don't enable

	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
	vc.RecordValidationStep(1, 0.7, 0.7)
	vc.RecordEpoch(1, 0.8, 0.6, 0.7, 0.7)
	vc.RecordROCData([]ROCPointViz{{FPR: 0.1, TPR: 0.9, Threshold: 0.5}})
	vc.RecordConfusionMatrix([][]int{{1}}, []string{"A"})
	vc.RecordRegressionData([]float64{1.0}, []float64{1.1})
	vc.RecordFeatureImportance([]string{"f1"}, []float64{0.5}, []float64{0.1})

	// All should be empty
	if len(vc.steps) != 0 {
		t.Error("Steps should be empty when disabled")
	}
	if len(vc.validationLoss) != 0 {
		t.Error("Validation loss should be empty when disabled")
	}
	if len(vc.epochs) != 0 {
		t.Error("Epochs should be empty when disabled")
	}
	if len(vc.rocPoints) != 0 {
		t.Error("ROC points should be empty when disabled")
	}
	if vc.confusionMatrix != nil {
		t.Error("Confusion matrix should be empty when disabled")
	}
	if len(vc.predictions) != 0 {
		t.Error("Predictions should be empty when disabled")
	}
}

// TestNormalQuantile tests the normal quantile approximation function
func TestNormalQuantile(t *testing.T) {
	tests := []struct {
		p        float64
		expected float64
		tolerance float64
	}{
		{0.5, 0.0, 1e-10},      // Median should be 0
		{0.0, math.Inf(-1), 0}, // Lower bound
		{1.0, math.Inf(1), 0},  // Upper bound
		{0.16, -1.0, 0.1},      // Approximately -1 standard deviation
		{0.84, 1.0, 0.1},       // Approximately +1 standard deviation
	}

	for _, test := range tests {
		result := normalQuantile(test.p)
		if math.IsInf(test.expected, 0) {
			// Check if both are infinite with the same sign
			expectedSign := 1
			if math.Signbit(test.expected) {
				expectedSign = -1
			}
			if !math.IsInf(result, expectedSign) {
				t.Errorf("normalQuantile(%f): expected %f, got %f", test.p, test.expected, result)
			}
		} else if math.Abs(result-test.expected) > test.tolerance {
			t.Errorf("normalQuantile(%f): expected %f±%f, got %f", test.p, test.expected, test.tolerance, result)
		}
	}
}

// TestCalculateMeanAndStd tests the mean and standard deviation calculation
func TestCalculateMeanAndStd(t *testing.T) {
	t.Run("EmptySlice", func(t *testing.T) {
		mean, std := calculateMeanAndStd([]float64{})
		if mean != 0 || std != 0 {
			t.Errorf("Empty slice should return 0, 0, got %f, %f", mean, std)
		}
	})

	t.Run("SingleValue", func(t *testing.T) {
		mean, std := calculateMeanAndStd([]float64{5.0})
		if mean != 5.0 {
			t.Errorf("Single value mean should return 5.0, got %f", mean)
		}
		// For a single value, sample standard deviation is undefined (NaN)
		// This is mathematically correct behavior
		if !math.IsNaN(std) {
			t.Errorf("Single value std should be NaN (undefined), got %f", std)
		}
	})

	t.Run("KnownValues", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		mean, std := calculateMeanAndStd(values)
		
		expectedMean := 3.0
		expectedStd := math.Sqrt(2.5) // Sample std dev
		
		if math.Abs(mean-expectedMean) > 1e-10 {
			t.Errorf("Expected mean %f, got %f", expectedMean, mean)
		}
		if math.Abs(std-expectedStd) > 1e-10 {
			t.Errorf("Expected std %f, got %f", expectedStd, std)
		}
	})
}

// TestMaxFunction tests the max utility function
func TestMaxFunction(t *testing.T) {
	tests := []struct {
		a, b, expected int
	}{
		{1, 2, 2},
		{5, 3, 5},
		{0, 0, 0},
		{-1, -2, -1},
	}

	for _, test := range tests {
		result := max(test.a, test.b)
		if result != test.expected {
			t.Errorf("max(%d, %d): expected %d, got %d", test.a, test.b, test.expected, result)
		}
	}
}

// TestDataPointCreation tests DataPoint struct creation and usage
func TestDataPointCreation(t *testing.T) {
	dp := DataPoint{
		X:     1.0,
		Y:     2.0,
		Z:     3.0,
		Label: "test point",
		Color: "#FF0000",
	}

	if dp.X != 1.0 {
		t.Errorf("Expected X=1.0, got %v", dp.X)
	}
	if dp.Y != 2.0 {
		t.Errorf("Expected Y=2.0, got %v", dp.Y)
	}
	if dp.Z != 3.0 {
		t.Errorf("Expected Z=3.0, got %v", dp.Z)
	}
	if dp.Label != "test point" {
		t.Errorf("Expected Label='test point', got %s", dp.Label)
	}
	if dp.Color != "#FF0000" {
		t.Errorf("Expected Color='#FF0000', got %s", dp.Color)
	}
}

// TestPlotConfigCreation tests PlotConfig struct creation
func TestPlotConfigCreation(t *testing.T) {
	config := PlotConfig{
		XAxisLabel:  "X Axis",
		YAxisLabel:  "Y Axis",
		XAxisScale:  "linear",
		YAxisScale:  "log",
		ShowLegend:  true,
		ShowGrid:    false,
		Width:       800,
		Height:      600,
		Interactive: true,
		CustomOptions: map[string]interface{}{
			"test": "value",
		},
	}

	if config.XAxisLabel != "X Axis" {
		t.Errorf("Expected XAxisLabel='X Axis', got %s", config.XAxisLabel)
	}
	if config.YAxisScale != "log" {
		t.Errorf("Expected YAxisScale='log', got %s", config.YAxisScale)
	}
	if !config.Interactive {
		t.Error("Expected Interactive=true")
	}
	if config.CustomOptions["test"] != "value" {
		t.Errorf("Expected custom option 'test'='value', got %v", config.CustomOptions["test"])
	}
}

// TestSeriesDataCreation tests SeriesData struct creation
func TestSeriesDataCreation(t *testing.T) {
	data := []DataPoint{
		{X: 1, Y: 2},
		{X: 3, Y: 4},
	}

	series := SeriesData{
		Name: "Test Series",
		Type: "line",
		Data: data,
		Style: map[string]interface{}{
			"color": "#FF0000",
			"width": 2,
		},
	}

	if series.Name != "Test Series" {
		t.Errorf("Expected Name='Test Series', got %s", series.Name)
	}
	if series.Type != "line" {
		t.Errorf("Expected Type='line', got %s", series.Type)
	}
	if len(series.Data) != 2 {
		t.Errorf("Expected 2 data points, got %d", len(series.Data))
	}
	if series.Style["color"] != "#FF0000" {
		t.Errorf("Expected color='#FF0000', got %v", series.Style["color"])
	}
}

// TestTimestampInPlots tests that timestamps are set in generated plots
func TestTimestampInPlots(t *testing.T) {
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()

	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)

	before := time.Now()
	plot := vc.GenerateTrainingCurvesPlot()
	after := time.Now()

	if plot.Timestamp.Before(before) || plot.Timestamp.After(after) {
		t.Error("Plot timestamp should be set to current time")
	}
}

// BenchmarkVisualizationDataRecording benchmarks data recording performance
func BenchmarkVisualizationDataRecording(b *testing.B) {
	vc := NewVisualizationCollector("BenchmarkModel")
	vc.Enable()

	b.Run("TrainingStep", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vc.RecordTrainingStep(i, float64(i)*0.01, float64(i)*0.02, 0.01)
		}
	})

	b.Run("ValidationStep", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vc.RecordValidationStep(i, float64(i)*0.01, float64(i)*0.02)
		}
	})

	b.Run("RegressionData", func(b *testing.B) {
		predictions := make([]float64, 100)
		trueValues := make([]float64, 100)
		for i := range predictions {
			predictions[i] = float64(i)
			trueValues[i] = float64(i) + 0.1
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vc.RecordRegressionData(predictions, trueValues)
		}
	})
}

// BenchmarkPlotGeneration benchmarks plot generation performance
func BenchmarkPlotGeneration(b *testing.B) {
	vc := NewVisualizationCollector("BenchmarkModel")
	vc.Enable()

	// Setup data
	for i := 0; i < 1000; i++ {
		vc.RecordTrainingStep(i, float64(i)*0.001, float64(i)*0.002, 0.01)
		if i%10 == 0 {
			vc.RecordValidationStep(i/10, float64(i)*0.0015, float64(i)*0.0025)
		}
	}

	b.Run("TrainingCurves", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = vc.GenerateTrainingCurvesPlot()
		}
	})

	b.Run("LearningRateSchedule", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = vc.GenerateLearningRateSchedulePlot()
		}
	})
}

// BenchmarkJSONSerialization benchmarks JSON conversion performance
func BenchmarkJSONSerialization(b *testing.B) {
	vc := NewVisualizationCollector("BenchmarkModel")
	vc.Enable()

	// Setup data
	for i := 0; i < 100; i++ {
		vc.RecordTrainingStep(i, float64(i)*0.01, float64(i)*0.02, 0.01)
	}

	plot := vc.GenerateTrainingCurvesPlot()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := plot.ToJSON()
		if err != nil {
			b.Fatal(err)
		}
	}
}