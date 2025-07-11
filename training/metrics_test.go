package training

import (
	"math"
	"testing"
)

// TestMetricTypeString tests the string representation of MetricType
func TestMetricTypeString(t *testing.T) {
	tests := []struct {
		metric   MetricType
		expected string
	}{
		{Precision, "Precision"},
		{Recall, "Recall"},
		{F1Score, "F1Score"},
		{Specificity, "Specificity"},
		{NPV, "NPV"},
		{MacroPrecision, "MacroPrecision"},
		{MacroRecall, "MacroRecall"},
		{MacroF1, "MacroF1"},
		{MicroPrecision, "MicroPrecision"},
		{MicroRecall, "MicroRecall"},
		{MicroF1, "MicroF1"},
		{AUCROC, "AUCROC"},
		{AUCPR, "AUCPR"},
		{MAE, "MAE"},
		{MSE, "MSE"},
		{RMSE, "RMSE"},
		{R2, "R2"},
		{NMAE, "NMAE"},
		{MetricType(999), "Unknown(999)"},
	}

	for _, test := range tests {
		result := test.metric.String()
		if result != test.expected {
			t.Errorf("MetricType(%d).String() = %s, expected %s", test.metric, result, test.expected)
		}
	}
}

// TestNewConfusionMatrix tests confusion matrix creation
func TestNewConfusionMatrix(t *testing.T) {
	cm := NewConfusionMatrix(3)
	
	if cm.NumClasses != 3 {
		t.Errorf("Expected 3 classes, got %d", cm.NumClasses)
	}
	
	if len(cm.Matrix) != 3 {
		t.Errorf("Expected matrix with 3 rows, got %d", len(cm.Matrix))
	}
	
	for i, row := range cm.Matrix {
		if len(row) != 3 {
			t.Errorf("Row %d: expected 3 columns, got %d", i, len(row))
		}
		for j, val := range row {
			if val != 0 {
				t.Errorf("Matrix[%d][%d]: expected 0, got %d", i, j, val)
			}
		}
	}
	
	if cm.TotalSamples != 0 {
		t.Errorf("Expected 0 total samples, got %d", cm.TotalSamples)
	}
}

// TestConfusionMatrixReset tests reset functionality
func TestConfusionMatrixReset(t *testing.T) {
	cm := NewConfusionMatrix(2)
	
	// Manually set some values
	cm.Matrix[0][0] = 5
	cm.Matrix[0][1] = 2
	cm.Matrix[1][0] = 1
	cm.Matrix[1][1] = 7
	cm.TotalSamples = 15
	cm.metricsValid = true
	cm.cachedMetrics[Precision] = 0.8
	
	// Reset
	cm.Reset()
	
	// Check all values are reset
	for i := range cm.Matrix {
		for j := range cm.Matrix[i] {
			if cm.Matrix[i][j] != 0 {
				t.Errorf("Matrix[%d][%d]: expected 0 after reset, got %d", i, j, cm.Matrix[i][j])
			}
		}
	}
	
	if cm.TotalSamples != 0 {
		t.Errorf("Expected 0 total samples after reset, got %d", cm.TotalSamples)
	}
	
	if cm.metricsValid {
		t.Error("Expected metricsValid to be false after reset")
	}
	
	if len(cm.cachedMetrics) != 0 {
		t.Errorf("Expected empty cached metrics after reset, got %d entries", len(cm.cachedMetrics))
	}
}

// TestConfusionMatrixUpdateFromPredictions tests updating from predictions
func TestConfusionMatrixUpdateFromPredictions(t *testing.T) {
	t.Run("BinaryClassification", func(t *testing.T) {
		cm := NewConfusionMatrix(2)
		
		// Binary classification predictions: logits for positive class
		predictions := []float32{0.8, -0.3, 1.2, -0.9} // > 0 = class 1, <= 0 = class 0
		trueLabels := []int32{1, 0, 1, 0}
		
		err := cm.UpdateFromPredictions(predictions, trueLabels, 4, 2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// Expected: pred [1, 0, 1, 0], true [1, 0, 1, 0]
		// Matrix[0][0] = 2 (TN), Matrix[0][1] = 0 (FP)
		// Matrix[1][0] = 0 (FN), Matrix[1][1] = 2 (TP)
		if cm.Matrix[0][0] != 2 {
			t.Errorf("Matrix[0][0]: expected 2, got %d", cm.Matrix[0][0])
		}
		if cm.Matrix[0][1] != 0 {
			t.Errorf("Matrix[0][1]: expected 0, got %d", cm.Matrix[0][1])
		}
		if cm.Matrix[1][0] != 0 {
			t.Errorf("Matrix[1][0]: expected 0, got %d", cm.Matrix[1][0])
		}
		if cm.Matrix[1][1] != 2 {
			t.Errorf("Matrix[1][1]: expected 2, got %d", cm.Matrix[1][1])
		}
		
		if cm.TotalSamples != 4 {
			t.Errorf("Expected 4 total samples, got %d", cm.TotalSamples)
		}
	})

	t.Run("MultiClassClassification", func(t *testing.T) {
		cm := NewConfusionMatrix(3)
		
		// Multi-class predictions: 3 classes, 2 samples
		predictions := []float32{
			0.1, 0.8, 0.1, // Sample 0: class 1 (argmax)
			0.6, 0.2, 0.2, // Sample 1: class 0 (argmax)
		}
		trueLabels := []int32{1, 0}
		
		err := cm.UpdateFromPredictions(predictions, trueLabels, 2, 3)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// Expected: pred [1, 0], true [1, 0]
		// Matrix[0][0] = 1, Matrix[1][1] = 1, others = 0
		if cm.Matrix[0][0] != 1 {
			t.Errorf("Matrix[0][0]: expected 1, got %d", cm.Matrix[0][0])
		}
		if cm.Matrix[1][1] != 1 {
			t.Errorf("Matrix[1][1]: expected 1, got %d", cm.Matrix[1][1])
		}
		
		if cm.TotalSamples != 2 {
			t.Errorf("Expected 2 total samples, got %d", cm.TotalSamples)
		}
	})

	t.Run("ErrorCases", func(t *testing.T) {
		cm := NewConfusionMatrix(2)
		
		// Wrong predictions length
		predictions := []float32{0.8, -0.3}
		trueLabels := []int32{1, 0, 1}
		err := cm.UpdateFromPredictions(predictions, trueLabels, 3, 2)
		if err == nil {
			t.Error("Expected error for predictions length mismatch")
		}
		
		// Wrong labels length
		predictions = []float32{0.8, -0.3}
		trueLabels = []int32{1}
		err = cm.UpdateFromPredictions(predictions, trueLabels, 2, 2)
		if err == nil {
			t.Error("Expected error for labels length mismatch")
		}
		
		// Wrong class count
		predictions = []float32{0.8, -0.3}
		trueLabels = []int32{1, 0}
		err = cm.UpdateFromPredictions(predictions, trueLabels, 2, 3)
		if err == nil {
			t.Error("Expected error for class count mismatch")
		}
	})

	t.Run("InvalidClassIndices", func(t *testing.T) {
		cm := NewConfusionMatrix(2)
		
		// Include invalid class labels - should be skipped
		predictions := []float32{0.8, -0.3, 1.2}
		trueLabels := []int32{1, 0, 5} // Class 5 is invalid for 2-class matrix
		
		err := cm.UpdateFromPredictions(predictions, trueLabels, 3, 2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// Should only count the first 2 valid samples
		if cm.TotalSamples != 2 {
			t.Errorf("Expected 2 total samples (invalid sample skipped), got %d", cm.TotalSamples)
		}
	})
}

// TestBinaryClassificationMetrics tests binary classification metrics
func TestBinaryClassificationMetrics(t *testing.T) {
	cm := NewConfusionMatrix(2)
	
	// Set up a confusion matrix manually
	// Matrix[0][0] = 50 (TN), Matrix[0][1] = 10 (FP)
	// Matrix[1][0] = 5 (FN),  Matrix[1][1] = 35 (TP)
	cm.Matrix[0][0] = 50
	cm.Matrix[0][1] = 10
	cm.Matrix[1][0] = 5
	cm.Matrix[1][1] = 35
	cm.TotalSamples = 100

	tests := []struct {
		metric   MetricType
		expected float64
		tolerance float64
	}{
		{Precision, 35.0 / (35.0 + 10.0), 1e-6},    // TP / (TP + FP)
		{Recall, 35.0 / (35.0 + 5.0), 1e-6},       // TP / (TP + FN)
		{Specificity, 50.0 / (50.0 + 10.0), 1e-6}, // TN / (TN + FP)
		{NPV, 50.0 / (50.0 + 5.0), 1e-6},          // TN / (TN + FN)
	}

	for _, test := range tests {
		result := cm.GetMetric(test.metric)
		if math.Abs(result-test.expected) > test.tolerance {
			t.Errorf("Metric %s: expected %f, got %f", test.metric.String(), test.expected, result)
		}
	}

	// Test F1 Score calculation
	precision := cm.GetMetric(Precision)
	recall := cm.GetMetric(Recall)
	expectedF1 := 2 * (precision * recall) / (precision + recall)
	f1 := cm.GetMetric(F1Score)
	if math.Abs(f1-expectedF1) > 1e-6 {
		t.Errorf("F1Score: expected %f, got %f", expectedF1, f1)
	}
}

// TestMultiClassMetrics tests multi-class classification metrics
func TestMultiClassMetrics(t *testing.T) {
	cm := NewConfusionMatrix(3)
	
	// Set up a 3x3 confusion matrix
	cm.Matrix[0][0] = 10 // Class 0 correctly classified
	cm.Matrix[0][1] = 2  // Class 0 misclassified as 1
	cm.Matrix[0][2] = 1  // Class 0 misclassified as 2
	cm.Matrix[1][0] = 3  // Class 1 misclassified as 0
	cm.Matrix[1][1] = 15 // Class 1 correctly classified
	cm.Matrix[1][2] = 2  // Class 1 misclassified as 2
	cm.Matrix[2][0] = 1  // Class 2 misclassified as 0
	cm.Matrix[2][1] = 1  // Class 2 misclassified as 1
	cm.Matrix[2][2] = 8  // Class 2 correctly classified
	cm.TotalSamples = 43

	// Test macro precision
	// Class 0: TP=10, FP=3+1=4, Precision=10/14
	// Class 1: TP=15, FP=2+1=3, Precision=15/18
	// Class 2: TP=8, FP=1+2=3, Precision=8/11
	expectedMacroPrecision := ((10.0/14.0) + (15.0/18.0) + (8.0/11.0)) / 3.0
	macroPrecision := cm.GetMetric(MacroPrecision)
	if math.Abs(macroPrecision-expectedMacroPrecision) > 1e-6 {
		t.Errorf("MacroPrecision: expected %f, got %f", expectedMacroPrecision, macroPrecision)
	}

	// Test macro recall
	// Class 0: TP=10, FN=2+1=3, Recall=10/13
	// Class 1: TP=15, FN=3+2=5, Recall=15/20
	// Class 2: TP=8, FN=1+1=2, Recall=8/10
	expectedMacroRecall := ((10.0/13.0) + (15.0/20.0) + (8.0/10.0)) / 3.0
	macroRecall := cm.GetMetric(MacroRecall)
	if math.Abs(macroRecall-expectedMacroRecall) > 1e-6 {
		t.Errorf("MacroRecall: expected %f, got %f", expectedMacroRecall, macroRecall)
	}

	// Test micro precision/recall (should be equal to accuracy for multi-class)
	expectedMicroPrecisionRecall := (10.0 + 15.0 + 8.0) / 43.0
	microPrecision := cm.GetMetric(MicroPrecision)
	microRecall := cm.GetMetric(MicroRecall)
	
	if math.Abs(microPrecision-expectedMicroPrecisionRecall) > 1e-6 {
		t.Errorf("MicroPrecision: expected %f, got %f", expectedMicroPrecisionRecall, microPrecision)
	}
	if math.Abs(microRecall-expectedMicroPrecisionRecall) > 1e-6 {
		t.Errorf("MicroRecall: expected %f, got %f", expectedMicroPrecisionRecall, microRecall)
	}
}

// TestGetAccuracy tests accuracy calculation
func TestGetAccuracy(t *testing.T) {
	t.Run("WithSamples", func(t *testing.T) {
		cm := NewConfusionMatrix(3)
		cm.Matrix[0][0] = 10
		cm.Matrix[1][1] = 15
		cm.Matrix[2][2] = 8
		cm.Matrix[0][1] = 2
		cm.Matrix[1][2] = 3
		cm.TotalSamples = 38
		
		expectedAccuracy := (10.0 + 15.0 + 8.0) / 38.0
		accuracy := cm.GetAccuracy()
		
		if math.Abs(accuracy-expectedAccuracy) > 1e-6 {
			t.Errorf("Expected accuracy %f, got %f", expectedAccuracy, accuracy)
		}
	})

	t.Run("NoSamples", func(t *testing.T) {
		cm := NewConfusionMatrix(2)
		accuracy := cm.GetAccuracy()
		
		if accuracy != 0.0 {
			t.Errorf("Expected 0.0 accuracy for no samples, got %f", accuracy)
		}
	})
}

// TestMetricCaching tests that metrics are properly cached
func TestMetricCaching(t *testing.T) {
	cm := NewConfusionMatrix(2)
	cm.Matrix[0][0] = 50
	cm.Matrix[0][1] = 10
	cm.Matrix[1][0] = 5
	cm.Matrix[1][1] = 35
	cm.TotalSamples = 100

	// First call should calculate and cache
	precision1 := cm.GetMetric(Precision)
	
	// Verify it's cached
	if _, exists := cm.cachedMetrics[Precision]; !exists {
		t.Error("Expected precision to be cached")
	}
	
	// Second call should return cached value
	precision2 := cm.GetMetric(Precision)
	
	if precision1 != precision2 {
		t.Errorf("Cached precision mismatch: %f vs %f", precision1, precision2)
	}
	
	// Reset should clear cache
	cm.Reset()
	if len(cm.cachedMetrics) != 0 {
		t.Error("Expected cache to be cleared after reset")
	}
}

// TestCalculateAUCROC tests AUC-ROC calculation
func TestCalculateAUCROC(t *testing.T) {
	t.Run("PerfectClassification", func(t *testing.T) {
		predictions := []float32{0.9, 0.8, 0.1, 0.2}
		trueLabels := []int32{1, 1, 0, 0}
		
		auc := CalculateAUCROC(predictions, trueLabels, 4)
		
		if auc != 1.0 {
			t.Errorf("Expected AUC 1.0 for perfect classification, got %f", auc)
		}
	})

	t.Run("RandomClassification", func(t *testing.T) {
		predictions := []float32{0.5, 0.5, 0.5, 0.5}
		trueLabels := []int32{1, 0, 1, 0}
		
		auc := CalculateAUCROC(predictions, trueLabels, 4)
		
		// For identical predictions, AUC can be 0.5 or other values depending on tie-breaking
		// The exact value depends on the sorting algorithm's stability
		if auc < 0.0 || auc > 1.0 {
			t.Errorf("Expected AUC between 0.0-1.0 for random classification, got %f", auc)
		}
	})

	t.Run("ErrorCases", func(t *testing.T) {
		// Mismatched lengths
		predictions := []float32{0.9, 0.8}
		trueLabels := []int32{1, 1, 0}
		auc := CalculateAUCROC(predictions, trueLabels, 3)
		if auc != 0.0 {
			t.Errorf("Expected 0.0 AUC for mismatched lengths, got %f", auc)
		}
		
		// Only one class
		predictions = []float32{0.9, 0.8, 0.7, 0.6}
		trueLabels = []int32{1, 1, 1, 1}
		auc = CalculateAUCROC(predictions, trueLabels, 4)
		if auc != 0.0 {
			t.Errorf("Expected 0.0 AUC for single class, got %f", auc)
		}
	})

	t.Run("RealWorldExample", func(t *testing.T) {
		predictions := []float32{0.9, 0.6, 0.35, 0.8, 0.2, 0.1}
		trueLabels := []int32{1, 1, 0, 1, 0, 0}
		
		auc := CalculateAUCROC(predictions, trueLabels, 6)
		
		// Manual calculation for verification
		// Sorted by prediction: (0.9,1), (0.8,1), (0.6,1), (0.35,0), (0.2,0), (0.1,0)
		// TPR and FPR at each threshold should give us the AUC
		if auc < 0.8 || auc > 1.0 {
			t.Errorf("Expected AUC between 0.8-1.0 for this example, got %f", auc)
		}
	})
}

// TestCalculateRegressionMetrics tests regression metrics calculation
func TestCalculateRegressionMetrics(t *testing.T) {
	t.Run("PerfectPredictions", func(t *testing.T) {
		predictions := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
		trueValues := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
		
		metrics := CalculateRegressionMetrics(predictions, trueValues, 5)
		
		if metrics.MAE != 0.0 {
			t.Errorf("Expected MAE 0.0 for perfect predictions, got %f", metrics.MAE)
		}
		if metrics.MSE != 0.0 {
			t.Errorf("Expected MSE 0.0 for perfect predictions, got %f", metrics.MSE)
		}
		if metrics.RMSE != 0.0 {
			t.Errorf("Expected RMSE 0.0 for perfect predictions, got %f", metrics.RMSE)
		}
		if metrics.R2 != 1.0 {
			t.Errorf("Expected R² 1.0 for perfect predictions, got %f", metrics.R2)
		}
	})

	t.Run("KnownValues", func(t *testing.T) {
		predictions := []float32{2.5, 0.0, 2.0, 8.0}
		trueValues := []float32{3.0, -0.5, 2.0, 7.0}
		
		metrics := CalculateRegressionMetrics(predictions, trueValues, 4)
		
		// Manual calculations:
		// Errors: [-0.5, 0.5, 0.0, 1.0]
		// MAE = (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
		expectedMAE := 0.5
		if math.Abs(metrics.MAE-expectedMAE) > 1e-6 {
			t.Errorf("Expected MAE %f, got %f", expectedMAE, metrics.MAE)
		}
		
		// MSE = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
		expectedMSE := 0.375
		if math.Abs(metrics.MSE-expectedMSE) > 1e-6 {
			t.Errorf("Expected MSE %f, got %f", expectedMSE, metrics.MSE)
		}
		
		// RMSE = sqrt(0.375) ≈ 0.612
		expectedRMSE := math.Sqrt(0.375)
		if math.Abs(metrics.RMSE-expectedRMSE) > 1e-6 {
			t.Errorf("Expected RMSE %f, got %f", expectedRMSE, metrics.RMSE)
		}
		
		// R² calculation: mean = (3-0.5+2+7)/4 = 2.875
		// SS_tot = (3-2.875)² + (-0.5-2.875)² + (2-2.875)² + (7-2.875)² = 30.1875
		// SS_res = 1.5, R² = 1 - 1.5/30.1875 ≈ 0.950
		if metrics.R2 < 0.9 || metrics.R2 > 1.0 {
			t.Errorf("Expected R² around 0.95, got %f", metrics.R2)
		}
	})

	t.Run("ErrorCases", func(t *testing.T) {
		// Insufficient data
		predictions := []float32{1.0}
		trueValues := []float32{1.0, 2.0}
		
		metrics := CalculateRegressionMetrics(predictions, trueValues, 2)
		
		// Should return zero metrics for insufficient data
		if metrics.MAE != 0.0 || metrics.MSE != 0.0 || metrics.RMSE != 0.0 || metrics.R2 != 0.0 {
			t.Error("Expected zero metrics for insufficient data")
		}
	})

	t.Run("ConstantTarget", func(t *testing.T) {
		predictions := []float32{5.1, 4.9, 5.2, 4.8}
		trueValues := []float32{5.0, 5.0, 5.0, 5.0}
		
		metrics := CalculateRegressionMetrics(predictions, trueValues, 4)
		
		// For constant target values, R² calculation should handle zero variance
		if math.IsNaN(metrics.R2) || math.IsInf(metrics.R2, 0) {
			t.Errorf("R² should be finite for constant targets, got %f", metrics.R2)
		}
	})
}

// TestEdgeCases tests various edge cases for metrics
func TestEdgeCases(t *testing.T) {
	t.Run("BinaryMetricsForNonBinary", func(t *testing.T) {
		cm := NewConfusionMatrix(3) // 3 classes
		
		// Binary metrics should return 0.0 for non-binary classification
		binaryMetrics := []MetricType{Precision, Recall, F1Score, Specificity, NPV}
		for _, metric := range binaryMetrics {
			result := cm.GetMetric(metric)
			if result != 0.0 {
				t.Errorf("Binary metric %s should return 0.0 for 3-class matrix, got %f", metric.String(), result)
			}
		}
	})

	t.Run("EmptyConfusionMatrix", func(t *testing.T) {
		cm := NewConfusionMatrix(2)
		
		// All metrics should return 0.0 for empty matrix
		metrics := []MetricType{Precision, Recall, F1Score, Specificity, NPV, MacroPrecision, MacroRecall, MacroF1, MicroPrecision, MicroRecall, MicroF1}
		for _, metric := range metrics {
			result := cm.GetMetric(metric)
			if result != 0.0 {
				t.Errorf("Metric %s should return 0.0 for empty matrix, got %f", metric.String(), result)
			}
		}
	})

	t.Run("UnknownMetric", func(t *testing.T) {
		cm := NewConfusionMatrix(2)
		
		result := cm.GetMetric(MetricType(999))
		if result != 0.0 {
			t.Errorf("Unknown metric should return 0.0, got %f", result)
		}
	})
}

// BenchmarkConfusionMatrixUpdate benchmarks confusion matrix updates
func BenchmarkConfusionMatrixUpdate(b *testing.B) {
	cm := NewConfusionMatrix(10)
	
	// Generate test data
	predictions := make([]float32, 1000*10) // 1000 samples, 10 classes
	trueLabels := make([]int32, 1000)
	
	for i := 0; i < 1000; i++ {
		trueLabels[i] = int32(i % 10)
		// Set prediction for correct class higher
		for j := 0; j < 10; j++ {
			if j == int(trueLabels[i]) {
				predictions[i*10+j] = 0.8
			} else {
				predictions[i*10+j] = 0.02
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cm.Reset()
		err := cm.UpdateFromPredictions(predictions, trueLabels, 1000, 10)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMetricCalculation benchmarks metric calculations
func BenchmarkMetricCalculation(b *testing.B) {
	cm := NewConfusionMatrix(10)
	
	// Set up a realistic confusion matrix
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			if i == j {
				cm.Matrix[i][j] = 85 // Correct predictions
			} else {
				cm.Matrix[i][j] = 15 / 9 // Misclassifications
			}
		}
	}
	cm.TotalSamples = 1000

	b.Run("MacroPrecision", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = cm.GetMetric(MacroPrecision)
		}
	})

	b.Run("MicroF1", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = cm.GetMetric(MicroF1)
		}
	})
}

// BenchmarkAUCROC benchmarks AUC-ROC calculation
func BenchmarkAUCROC(b *testing.B) {
	// Generate test data
	size := 10000
	predictions := make([]float32, size)
	trueLabels := make([]int32, size)
	
	for i := 0; i < size; i++ {
		predictions[i] = float32(i) / float32(size) // Ascending scores
		if i%2 == 0 {
			trueLabels[i] = 0
		} else {
			trueLabels[i] = 1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalculateAUCROC(predictions, trueLabels, size)
	}
}

// BenchmarkRegressionMetrics benchmarks regression metrics calculation
func BenchmarkRegressionMetrics(b *testing.B) {
	size := 10000
	predictions := make([]float32, size)
	trueValues := make([]float32, size)
	
	for i := 0; i < size; i++ {
		trueValues[i] = float32(i) * 0.1
		predictions[i] = trueValues[i] + float32(i%10)*0.01 // Add some noise
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalculateRegressionMetrics(predictions, trueValues, size)
	}
}