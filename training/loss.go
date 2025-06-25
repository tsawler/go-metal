package training

import (
	"fmt"
	"math"

	"github.com/tsawler/go-metal/tensor"
)

// Loss interface defines methods that all loss functions must implement
type Loss interface {
	Forward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
	Backward(predicted, target *tensor.Tensor) (*tensor.Tensor, error)
}

// MSELoss implements Mean Squared Error loss function
type MSELoss struct {
	reduction string // "mean" or "sum"
}

// NewMSELoss creates a new Mean Squared Error loss function
func NewMSELoss(reduction string) *MSELoss {
	if reduction == "" {
		reduction = "mean"
	}
	return &MSELoss{reduction: reduction}
}

// Forward computes the MSE loss: L = (1/N) * sum((y_pred - y_true)^2)
func (mse *MSELoss) Forward(predicted, target *tensor.Tensor) (*tensor.Tensor, error) {
	if predicted.DType != target.DType {
		return nil, fmt.Errorf("predicted and target tensors must have the same dtype")
	}
	
	if len(predicted.Shape) != len(target.Shape) {
		return nil, fmt.Errorf("predicted and target tensors must have the same shape")
	}
	
	for i, dim := range predicted.Shape {
		if dim != target.Shape[i] {
			return nil, fmt.Errorf("predicted and target tensors must have the same shape")
		}
	}
	
	// Use GPU operations if available, otherwise fall back to CPU
	var diff *tensor.Tensor
	var err error
	
	// Use CPU operations for now - we'll implement SubMPS later if needed
	diff, err = tensor.Sub(predicted, target)
	if err != nil {
		return nil, fmt.Errorf("subtraction failed: %v", err)
	}
	
	// Compute squared differences
	squared, err := tensor.Mul(diff, diff)
	if err != nil {
		return nil, fmt.Errorf("multiplication failed: %v", err)
	}
	
	// Sum all elements
	loss, err := mse.sumAllElements(squared)
	if err != nil {
		return nil, fmt.Errorf("sum computation failed: %v", err)
	}
	
	// Apply reduction
	if mse.reduction == "mean" {
		n := float64(predicted.NumElems)
		meanScale := tensor.FromScalar(1.0/n, loss.DType, loss.Device)
		
		loss, err = tensor.Mul(loss, meanScale)
		
		if err != nil {
			return nil, fmt.Errorf("mean computation failed: %v", err)
		}
	}
	
	return loss, nil
}

// Backward computes the gradient of MSE loss
func (mse *MSELoss) Backward(predicted, target *tensor.Tensor) (*tensor.Tensor, error) {
	// MSE gradient: d/d(pred) = 2 * (predicted - target) / N
	var diff *tensor.Tensor
	var err error
	
	diff, err = tensor.Sub(predicted, target)
	
	if err != nil {
		return nil, fmt.Errorf("gradient subtraction failed: %v", err)
	}
	
	// Scale by 2
	scale := tensor.FromScalar(2.0, diff.DType, diff.Device)
	
	grad, err := tensor.Mul(diff, scale)
	
	if err != nil {
		return nil, fmt.Errorf("gradient scaling failed: %v", err)
	}
	
	// Apply reduction scaling
	if mse.reduction == "mean" {
		n := float64(predicted.NumElems)
		meanScale := tensor.FromScalar(1.0/n, grad.DType, grad.Device)
		
		grad, err = tensor.Mul(grad, meanScale)
		
		if err != nil {
			return nil, fmt.Errorf("gradient mean computation failed: %v", err)
		}
	}
	
	return grad, nil
}

// sumAllElements sums all elements in a tensor to produce a scalar
func (mse *MSELoss) sumAllElements(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t.DType != tensor.Float32 {
		return nil, fmt.Errorf("sumAllElements only supports Float32 tensors")
	}
	
	data := t.Data.([]float32)
	var sum float32
	for _, val := range data {
		sum += val
	}
	
	return tensor.NewTensor([]int{1}, t.DType, t.Device, []float32{sum})
}

// CrossEntropyLoss implements Cross Entropy loss function for classification
type CrossEntropyLoss struct {
	reduction string // "mean" or "sum"
}

// NewCrossEntropyLoss creates a new Cross Entropy loss function
func NewCrossEntropyLoss(reduction string) *CrossEntropyLoss {
	if reduction == "" {
		reduction = "mean"
	}
	return &CrossEntropyLoss{reduction: reduction}
}

// Forward computes the Cross Entropy loss
// predicted: [batch_size, num_classes] logits
// target: [batch_size] class indices
func (ce *CrossEntropyLoss) Forward(predicted, target *tensor.Tensor) (*tensor.Tensor, error) {
	if predicted.DType != tensor.Float32 || target.DType != tensor.Int32 {
		return nil, fmt.Errorf("predicted must be Float32 and target must be Int32")
	}
	
	if len(predicted.Shape) != 2 {
		return nil, fmt.Errorf("predicted must be 2D tensor [batch_size, num_classes], got shape %v", predicted.Shape)
	}
	
	if len(target.Shape) != 1 {
		return nil, fmt.Errorf("target must be 1D tensor [batch_size], got shape %v", target.Shape)
	}
	
	batchSize := predicted.Shape[0]
	numClasses := predicted.Shape[1]
	
	if target.Shape[0] != batchSize {
		return nil, fmt.Errorf("batch size mismatch: predicted %d, target %d", batchSize, target.Shape[0])
	}
	
	// Apply softmax to get probabilities
	softmaxProbs, err := ce.softmax(predicted)
	if err != nil {
		return nil, fmt.Errorf("softmax computation failed: %v", err)
	}
	
	// Compute negative log likelihood
	loss, err := ce.negativeLogLikelihood(softmaxProbs, target, batchSize, numClasses)
	if err != nil {
		return nil, fmt.Errorf("negative log likelihood computation failed: %v", err)
	}
	
	// Apply reduction
	if ce.reduction == "mean" {
		n := float64(batchSize)
		meanScale := tensor.FromScalar(1.0/n, loss.DType, loss.Device)
		
		meanLoss, err := tensor.Mul(loss, meanScale)
		
		if err != nil {
			return nil, fmt.Errorf("mean computation failed: %v", err)
		}
		loss = meanLoss
	}
	
	return loss, nil
}

// Backward computes the gradient of Cross Entropy loss
func (ce *CrossEntropyLoss) Backward(predicted, target *tensor.Tensor) (*tensor.Tensor, error) {
	if predicted.DType != tensor.Float32 || target.DType != tensor.Int32 {
		return nil, fmt.Errorf("predicted must be Float32 and target must be Int32")
	}
	
	batchSize := predicted.Shape[0]
	numClasses := predicted.Shape[1]
	
	// Apply softmax to get probabilities
	softmaxProbs, err := ce.softmax(predicted)
	if err != nil {
		return nil, fmt.Errorf("softmax computation failed: %v", err)
	}
	
	// Create gradient tensor initialized with softmax probabilities
	grad, err := softmaxProbs.Clone()
	if err != nil {
		return nil, fmt.Errorf("gradient initialization failed: %v", err)
	}
	
	// Subtract 1 from the true class probabilities
	gradData := grad.Data.([]float32)
	targetData := target.Data.([]int32)
	
	for i := 0; i < batchSize; i++ {
		targetClass := targetData[i]
		if targetClass < 0 || int(targetClass) >= numClasses {
			return nil, fmt.Errorf("target class %d out of range [0, %d)", targetClass, numClasses)
		}
		
		gradIdx := i*numClasses + int(targetClass)
		gradData[gradIdx] -= 1.0
	}
	
	// Apply reduction scaling
	if ce.reduction == "mean" {
		n := float64(batchSize)
		meanScale := tensor.FromScalar(1.0/n, grad.DType, grad.Device)
		
		grad, err = tensor.Mul(grad, meanScale)
		
		if err != nil {
			return nil, fmt.Errorf("gradient mean computation failed: %v", err)
		}
	}
	
	return grad, nil
}

// softmax applies softmax activation to predicted logits
func (ce *CrossEntropyLoss) softmax(logits *tensor.Tensor) (*tensor.Tensor, error) {
	if logits.DType != tensor.Float32 {
		return nil, fmt.Errorf("softmax only supports Float32 tensors")
	}
	
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]
	
	data := logits.Data.([]float32)
	result := make([]float32, len(data))
	
	// Apply softmax row by row (for each sample in the batch)
	for i := 0; i < batchSize; i++ {
		offset := i * numClasses
		
		// Find max for numerical stability
		maxVal := data[offset]
		for j := 1; j < numClasses; j++ {
			if data[offset+j] > maxVal {
				maxVal = data[offset+j]
			}
		}
		
		// Compute exp(x - max) and sum
		var sum float32
		for j := 0; j < numClasses; j++ {
			exp := float32(math.Exp(float64(data[offset+j] - maxVal)))
			result[offset+j] = exp
			sum += exp
		}
		
		// Normalize
		for j := 0; j < numClasses; j++ {
			result[offset+j] /= sum
		}
	}
	
	return tensor.NewTensor(logits.Shape, logits.DType, logits.Device, result)
}

// negativeLogLikelihood computes the negative log likelihood
func (ce *CrossEntropyLoss) negativeLogLikelihood(probs *tensor.Tensor, target *tensor.Tensor, batchSize, numClasses int) (*tensor.Tensor, error) {
	probsData := probs.Data.([]float32)
	targetData := target.Data.([]int32)
	
	var totalLoss float32
	
	for i := 0; i < batchSize; i++ {
		targetClass := targetData[i]
		if targetClass < 0 || int(targetClass) >= numClasses {
			return nil, fmt.Errorf("target class %d out of range [0, %d)", targetClass, numClasses)
		}
		
		probIdx := i*numClasses + int(targetClass)
		prob := probsData[probIdx]
		
		// Add small epsilon to prevent log(0)
		if prob < 1e-10 {
			prob = 1e-10
		}
		
		totalLoss += -float32(math.Log(float64(prob)))
	}
	
	return tensor.NewTensor([]int{1}, tensor.Float32, probs.Device, []float32{totalLoss})
}

