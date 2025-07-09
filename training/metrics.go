package training

import (
	"fmt"
	"math"
	"sort"
)

// MetricType represents different evaluation metrics
type MetricType int

const (
	// Binary Classification Metrics
	Precision MetricType = iota
	Recall
	F1Score
	Specificity
	NPV // Negative Predictive Value
	
	// Multi-class Metrics  
	MacroPrecision
	MacroRecall
	MacroF1
	MicroPrecision
	MicroRecall
	MicroF1
	
	// Ranking Metrics
	AUCROC
	AUCPR // Area Under Precision-Recall Curve
	
	// Regression Metrics
	MAE  // Mean Absolute Error
	MSE  // Mean Squared Error
	RMSE // Root Mean Squared Error
	R2   // R-squared
	NMAE // Normalized Mean Absolute Error
)

func (mt MetricType) String() string {
	switch mt {
	case Precision:
		return "Precision"
	case Recall:
		return "Recall"
	case F1Score:
		return "F1Score"
	case Specificity:
		return "Specificity"
	case NPV:
		return "NPV"
	case MacroPrecision:
		return "MacroPrecision"
	case MacroRecall:
		return "MacroRecall"
	case MacroF1:
		return "MacroF1"
	case MicroPrecision:
		return "MicroPrecision"
	case MicroRecall:
		return "MicroRecall"
	case MicroF1:
		return "MicroF1"
	case AUCROC:
		return "AUCROC"
	case AUCPR:
		return "AUCPR"
	case MAE:
		return "MAE"
	case MSE:
		return "MSE"
	case RMSE:
		return "RMSE"
	case R2:
		return "R2"
	case NMAE:
		return "NMAE"
	default:
		return fmt.Sprintf("Unknown(%d)", int(mt))
	}
}

// ConfusionMatrix represents a confusion matrix for classification tasks
// Adheres to GPU-resident architecture: all tensors on GPU, only scalar results on CPU
type ConfusionMatrix struct {
	NumClasses int
	Matrix     [][]int // [true_class][predicted_class]
	TotalSamples int
	
	// Cached metrics to avoid recomputation
	cachedMetrics map[MetricType]float64
	metricsValid  bool
}

// NewConfusionMatrix creates a new confusion matrix
func NewConfusionMatrix(numClasses int) *ConfusionMatrix {
	matrix := make([][]int, numClasses)
	for i := range matrix {
		matrix[i] = make([]int, numClasses)
	}
	
	return &ConfusionMatrix{
		NumClasses:    numClasses,
		Matrix:        matrix,
		cachedMetrics: make(map[MetricType]float64),
	}
}

// Reset clears the confusion matrix
func (cm *ConfusionMatrix) Reset() {
	for i := range cm.Matrix {
		for j := range cm.Matrix[i] {
			cm.Matrix[i][j] = 0
		}
	}
	cm.TotalSamples = 0
	cm.metricsValid = false
	cm.cachedMetrics = make(map[MetricType]float64)
}

// UpdateFromPredictions updates confusion matrix from GPU-resident predictions
// Maintains design compliance: predictions/labels come from GPU, only scalar updates on CPU
func (cm *ConfusionMatrix) UpdateFromPredictions(
	predictions []float32, // GPU tensor data copied to CPU for scalar extraction
	trueLabels []int32,    // True class labels
	batchSize int,
	numClasses int,
) error {
	// Handle single output binary classification (BCEWithLogits)
	expectedPredictions := batchSize * numClasses
	if cm.NumClasses == 2 && len(predictions) == batchSize {
		// Single output binary classification - convert to 2-class format
		expectedPredictions = batchSize
	}
	
	if len(predictions) != expectedPredictions {
		return fmt.Errorf("predictions length mismatch: expected %d, got %d", expectedPredictions, len(predictions))
	}
	
	if len(trueLabels) != batchSize {
		return fmt.Errorf("labels length mismatch: expected %d, got %d", batchSize, len(trueLabels))
	}
	
	if numClasses != cm.NumClasses {
		return fmt.Errorf("class count mismatch: expected %d, got %d", cm.NumClasses, numClasses)
	}
	
	// CPU-only processing for final scalar metrics (design compliant)
	for i := 0; i < batchSize; i++ {
		var predClass int
		
		if cm.NumClasses == 2 && len(predictions) == batchSize {
			// Single output binary classification (BCEWithLogits)
			// prediction > 0 means class 1, prediction <= 0 means class 0
			if predictions[i] > 0 {
				predClass = 1
			} else {
				predClass = 0
			}
		} else {
			// Multi-class classification - find predicted class (argmax)
			maxIdx := 0
			maxVal := predictions[i*numClasses]
			
			for j := 1; j < numClasses; j++ {
				if predictions[i*numClasses+j] > maxVal {
					maxVal = predictions[i*numClasses+j]
					maxIdx = j
				}
			}
			predClass = maxIdx
		}
		
		trueClass := int(trueLabels[i])
		
		// Validate class indices
		if trueClass < 0 || trueClass >= cm.NumClasses || predClass < 0 || predClass >= cm.NumClasses {
			continue // Skip invalid samples
		}
		
		// Update confusion matrix
		cm.Matrix[trueClass][predClass]++
		cm.TotalSamples++
	}
	
	// Invalidate cached metrics
	cm.metricsValid = false
	return nil
}

// GetMetric calculates and caches evaluation metrics
// Only CPU access for final scalar metrics (GPU-resident architecture compliance)
func (cm *ConfusionMatrix) GetMetric(metric MetricType) float64 {
	if cm.metricsValid {
		if value, exists := cm.cachedMetrics[metric]; exists {
			return value
		}
	}
	
	var result float64
	
	switch metric {
	case Precision:
		result = cm.calculateBinaryPrecision()
	case Recall:
		result = cm.calculateBinaryRecall()
	case F1Score:
		result = cm.calculateBinaryF1()
	case Specificity:
		result = cm.calculateSpecificity()
	case NPV:
		result = cm.calculateNPV()
	case MacroPrecision:
		result = cm.calculateMacroPrecision()
	case MacroRecall:
		result = cm.calculateMacroRecall()
	case MacroF1:
		result = cm.calculateMacroF1()
	case MicroPrecision:
		result = cm.calculateMicroPrecision()
	case MicroRecall:
		result = cm.calculateMicroRecall()
	case MicroF1:
		result = cm.calculateMicroF1()
	default:
		return 0.0
	}
	
	// Cache the result
	cm.cachedMetrics[metric] = result
	return result
}

// Binary classification metrics (assuming class 1 is positive)
func (cm *ConfusionMatrix) calculateBinaryPrecision() float64 {
	if cm.NumClasses != 2 {
		return 0.0 // Only valid for binary classification
	}
	
	tp := float64(cm.Matrix[1][1]) // True positives
	fp := float64(cm.Matrix[0][1]) // False positives
	
	if tp+fp == 0 {
		return 0.0 // No positive predictions
	}
	
	return tp / (tp + fp)
}

func (cm *ConfusionMatrix) calculateBinaryRecall() float64 {
	if cm.NumClasses != 2 {
		return 0.0
	}
	
	tp := float64(cm.Matrix[1][1]) // True positives
	fn := float64(cm.Matrix[1][0]) // False negatives
	
	if tp+fn == 0 {
		return 0.0 // No actual positives
	}
	
	return tp / (tp + fn)
}

func (cm *ConfusionMatrix) calculateBinaryF1() float64 {
	precision := cm.calculateBinaryPrecision()
	recall := cm.calculateBinaryRecall()
	
	if precision+recall == 0 {
		return 0.0
	}
	
	return 2 * (precision * recall) / (precision + recall)
}

func (cm *ConfusionMatrix) calculateSpecificity() float64 {
	if cm.NumClasses != 2 {
		return 0.0
	}
	
	tn := float64(cm.Matrix[0][0]) // True negatives
	fp := float64(cm.Matrix[0][1]) // False positives
	
	if tn+fp == 0 {
		return 0.0 // No actual negatives
	}
	
	return tn / (tn + fp)
}

func (cm *ConfusionMatrix) calculateNPV() float64 {
	if cm.NumClasses != 2 {
		return 0.0
	}
	
	tn := float64(cm.Matrix[0][0]) // True negatives
	fn := float64(cm.Matrix[1][0]) // False negatives
	
	if tn+fn == 0 {
		return 0.0 // No negative predictions
	}
	
	return tn / (tn + fn)
}

// Multi-class metrics
func (cm *ConfusionMatrix) calculateMacroPrecision() float64 {
	if cm.NumClasses < 2 {
		return 0.0
	}
	
	sum := 0.0
	validClasses := 0
	
	for class := 0; class < cm.NumClasses; class++ {
		tp := float64(cm.Matrix[class][class])
		fp := 0.0
		
		// Sum false positives for this class
		for otherClass := 0; otherClass < cm.NumClasses; otherClass++ {
			if otherClass != class {
				fp += float64(cm.Matrix[otherClass][class])
			}
		}
		
		if tp+fp > 0 {
			sum += tp / (tp + fp)
			validClasses++
		}
	}
	
	if validClasses == 0 {
		return 0.0
	}
	
	return sum / float64(validClasses)
}

func (cm *ConfusionMatrix) calculateMacroRecall() float64 {
	if cm.NumClasses < 2 {
		return 0.0
	}
	
	sum := 0.0
	validClasses := 0
	
	for class := 0; class < cm.NumClasses; class++ {
		tp := float64(cm.Matrix[class][class])
		fn := 0.0
		
		// Sum false negatives for this class
		for otherClass := 0; otherClass < cm.NumClasses; otherClass++ {
			if otherClass != class {
				fn += float64(cm.Matrix[class][otherClass])
			}
		}
		
		if tp+fn > 0 {
			sum += tp / (tp + fn)
			validClasses++
		}
	}
	
	if validClasses == 0 {
		return 0.0
	}
	
	return sum / float64(validClasses)
}

func (cm *ConfusionMatrix) calculateMacroF1() float64 {
	precision := cm.calculateMacroPrecision()
	recall := cm.calculateMacroRecall()
	
	if precision+recall == 0 {
		return 0.0
	}
	
	return 2 * (precision * recall) / (precision + recall)
}

func (cm *ConfusionMatrix) calculateMicroPrecision() float64 {
	totalTP := 0.0
	totalFP := 0.0
	
	for class := 0; class < cm.NumClasses; class++ {
		totalTP += float64(cm.Matrix[class][class])
		
		for otherClass := 0; otherClass < cm.NumClasses; otherClass++ {
			if otherClass != class {
				totalFP += float64(cm.Matrix[otherClass][class])
			}
		}
	}
	
	if totalTP+totalFP == 0 {
		return 0.0
	}
	
	return totalTP / (totalTP + totalFP)
}

func (cm *ConfusionMatrix) calculateMicroRecall() float64 {
	totalTP := 0.0
	totalFN := 0.0
	
	for class := 0; class < cm.NumClasses; class++ {
		totalTP += float64(cm.Matrix[class][class])
		
		for otherClass := 0; otherClass < cm.NumClasses; otherClass++ {
			if otherClass != class {
				totalFN += float64(cm.Matrix[class][otherClass])
			}
		}
	}
	
	if totalTP+totalFN == 0 {
		return 0.0
	}
	
	return totalTP / (totalTP + totalFN)
}

func (cm *ConfusionMatrix) calculateMicroF1() float64 {
	precision := cm.calculateMicroPrecision()
	recall := cm.calculateMicroRecall()
	
	if precision+recall == 0 {
		return 0.0
	}
	
	return 2 * (precision * recall) / (precision + recall)
}

// GetAccuracy returns overall classification accuracy
func (cm *ConfusionMatrix) GetAccuracy() float64 {
	if cm.TotalSamples == 0 {
		return 0.0
	}
	
	correct := 0
	for i := 0; i < cm.NumClasses; i++ {
		correct += cm.Matrix[i][i]
	}
	
	return float64(correct) / float64(cm.TotalSamples)
}

// ROCPoint represents a point on the ROC curve
type ROCPoint struct {
	Threshold float32
	TPR       float64 // True Positive Rate (Recall)
	FPR       float64 // False Positive Rate (1 - Specificity)
}

// CalculateAUCROC calculates Area Under ROC Curve for binary classification
// GPU-resident architecture: operates on GPU tensor data, returns CPU scalar
func CalculateAUCROC(
	predictions []float32, // Raw prediction scores from GPU tensor
	trueLabels []int32,    // Binary labels (0 or 1)
	batchSize int,
) float64 {
	if len(predictions) != batchSize || len(trueLabels) != batchSize {
		return 0.0
	}
	
	// Create prediction-label pairs for sorting
	type predLabel struct {
		score float32
		label int32
	}
	
	pairs := make([]predLabel, batchSize)
	for i := 0; i < batchSize; i++ {
		pairs[i] = predLabel{score: predictions[i], label: trueLabels[i]}
	}
	
	// Sort by prediction score (descending)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})
	
	// Count total positives and negatives
	totalPos := 0
	totalNeg := 0
	for _, pair := range pairs {
		if pair.label == 1 {
			totalPos++
		} else {
			totalNeg++
		}
	}
	
	if totalPos == 0 || totalNeg == 0 {
		return 0.0 // Cannot calculate AUC without both classes
	}
	
	// Calculate AUC using trapezoidal rule
	auc := 0.0
	tp := 0
	fp := 0
	prevTPR := 0.0
	prevFPR := 0.0
	
	for _, pair := range pairs {
		if pair.label == 1 {
			tp++
		} else {
			fp++
		}
		
		tpr := float64(tp) / float64(totalPos)
		fpr := float64(fp) / float64(totalNeg)
		
		// Add trapezoid area
		auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0
		
		prevTPR = tpr
		prevFPR = fpr
	}
	
	return auc
}

// RegressionMetrics holds comprehensive regression evaluation metrics
type RegressionMetrics struct {
	MAE  float64 // Mean Absolute Error
	MSE  float64 // Mean Squared Error
	RMSE float64 // Root Mean Squared Error
	R2   float64 // R-squared
	NMAE float64 // Normalized Mean Absolute Error
}

// CalculateRegressionMetrics computes comprehensive regression metrics
// GPU-resident architecture: operates on GPU tensor data, returns CPU scalars
func CalculateRegressionMetrics(
	predictions []float32,
	trueValues []float32,
	batchSize int,
) *RegressionMetrics {
	if len(predictions) < batchSize || len(trueValues) < batchSize {
		return &RegressionMetrics{}
	}
	
	// Calculate mean of true values for R²
	meanTrue := 0.0
	for i := 0; i < batchSize; i++ {
		meanTrue += float64(trueValues[i])
	}
	meanTrue /= float64(batchSize)
	
	// Calculate metrics
	sumAbsErr := 0.0
	sumSqErr := 0.0
	sumSqTotal := 0.0
	minTrue := math.Inf(1)
	maxTrue := math.Inf(-1)
	
	for i := 0; i < batchSize; i++ {
		pred := float64(predictions[i])
		true := float64(trueValues[i])
		
		absErr := math.Abs(pred - true)
		sqErr := (pred - true) * (pred - true)
		
		sumAbsErr += absErr
		sumSqErr += sqErr
		sumSqTotal += (true - meanTrue) * (true - meanTrue)
		
		if true < minTrue {
			minTrue = true
		}
		if true > maxTrue {
			maxTrue = true
		}
	}
	
	mae := sumAbsErr / float64(batchSize)
	mse := sumSqErr / float64(batchSize)
	rmse := math.Sqrt(mse)
	
	// R² calculation
	r2 := 0.0
	if sumSqTotal > 0 {
		r2 = 1.0 - (sumSqErr / sumSqTotal)
	}
	
	// Normalized MAE (scale by range)
	nmae := 0.0
	if maxTrue > minTrue {
		nmae = mae / (maxTrue - minTrue)
	}
	
	return &RegressionMetrics{
		MAE:  mae,
		MSE:  mse,
		RMSE: rmse,
		R2:   r2,
		NMAE: nmae,
	}
}