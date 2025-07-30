# Regression Tutorial

Master building regression models with go-metal for predicting continuous values.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- Build regression models for continuous value prediction
- Use different loss functions (MSE, MAE, Huber) effectively
- Handle different types of regression problems
- Evaluate regression model performance
- Apply feature scaling and normalization
- Build real-world regression applications

## 📊 Regression Fundamentals

### What is Regression?

Regression predicts continuous numerical values rather than discrete classes:
- **Input**: Features (measurements, attributes)
- **Output**: Continuous values (prices, temperatures, scores)
- **Goal**: Minimize prediction error on unseen data

### Types of Regression Problems

```
Linear Regression:     y = mx + b
Polynomial Regression: y = ax² + bx + c
Multiple Regression:   y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
Non-linear Regression: y = f(x) (neural networks)
```

## 🚀 Tutorial 1: Simple Linear Regression

Let's start with a basic single-variable regression problem.

### Step 1: Setup and Data Generation

```go
package main

import (
    "fmt"
    "log"
    "math"
    "math/rand"

    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

func main() {
    fmt.Println("📈 Regression Tutorial: Linear Prediction")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Set random seed for reproducibility
    rand.Seed(42)
    
    // Problem setup
    batchSize := 64
    inputSize := 1    // Single feature
    outputSize := 1   // Single continuous output
    
    // Continue with model building...
}
```

### Step 2: Generate Synthetic Linear Data

```go
func generateLinearData(batchSize int) ([]float32, []float32, []int, []int) {
    fmt.Println("📊 Generating Synthetic Linear Dataset")
    
    // True relationship: y = 2x + 1 + noise
    trueSlope := float32(2.0)
    trueIntercept := float32(1.0)
    noiseLevel := float32(0.1)
    
    inputData := make([]float32, batchSize)
    targetData := make([]float32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        // Generate x values between -2 and 2
        x := rand.Float32()*4.0 - 2.0
        
        // Generate y = 2x + 1 + noise
        noise := (rand.Float32() - 0.5) * 2.0 * noiseLevel
        y := trueSlope*x + trueIntercept + noise
        
        inputData[i] = x
        targetData[i] = y
    }
    
    inputShape := []int{batchSize, 1}   // Single feature
    targetShape := []int{batchSize, 1}  // Single target
    
    fmt.Printf("   ✅ Generated %d samples\n", batchSize)
    fmt.Printf("   📐 True relationship: y = %.1fx + %.1f + noise\n", 
               trueSlope, trueIntercept)
    
    return inputData, targetData, inputShape, targetShape
}
```

### Step 3: Build Simple Linear Model

```go
func buildLinearRegressionModel(batchSize, inputSize, outputSize int) (*layers.ModelSpec, error) {
    fmt.Println("🏗️ Building Linear Regression Model")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Single linear layer: y = wx + b
        AddDense(outputSize, true, "linear").
        // No activation - raw output for regression
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("linear model compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: %d → %d (single linear layer)\n", 
               inputSize, outputSize)
    fmt.Printf("   📊 Parameters: weight + bias\n")
    
    return model, nil
}
```

### Step 4: Configure Regression Training

```go
func setupRegressionTrainer(model *layers.ModelSpec, batchSize int) (*training.ModelTrainer, error) {
    fmt.Println("⚙️ Configuring Regression Training")
    
    config := training.TrainerConfig{
        // Basic parameters
        BatchSize:    batchSize,
        LearningRate: 0.01,  // Higher LR often works for simple regression
        
        // Optimizer
        OptimizerType: cgo_bridge.Adam,
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
        
        // Problem configuration
        EngineType:   training.Dynamic,
        ProblemType:  training.Regression,        // Key: Regression, not Classification
        LossFunction: training.MeanSquaredError,  // MSE for standard regression
    }
    
    trainer, err := training.NewModelTrainer(model, config)
    if err != nil {
        return nil, fmt.Errorf("regression trainer creation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Problem Type: Regression\n")
    fmt.Printf("   ✅ Loss Function: Mean Squared Error\n")
    fmt.Printf("   ✅ Optimizer: Adam (lr=%.3f)\n", config.LearningRate)
    
    return trainer, nil
}
```

### Step 5: Training Loop with Metrics

```go
func trainRegression(trainer *training.ModelTrainer, inputData []float32, inputShape []int,
                    targetData []float32, targetShape []int, epochs int) error {
    fmt.Printf("🚀 Training Regression Model for %d epochs\n", epochs)
    fmt.Println("Epoch | MSE Loss | Trend   | RMSE   | Status")
    fmt.Println("------|----------|---------|--------|----------")
    
    var prevLoss float32 = 999.0
    
    for epoch := 1; epoch <= epochs; epoch++ {
        // Execute training step
        result, err := trainer.TrainBatch(inputData, inputShape, targetData, targetShape)
        if err != nil {
            return fmt.Errorf("regression training epoch %d failed: %v", epoch, err)
        }
        
        // Calculate additional metrics
        rmse := float32(math.Sqrt(float64(result.Loss)))
        
        // Loss trend analysis
        var trend string
        if result.Loss < prevLoss {
            trend = "↓ Good"
        } else {
            trend = "↑ Check"
        }
        
        // Training status
        var status string
        if result.Loss < 0.01 {
            status = "Excellent"
        } else if result.Loss < 0.1 {
            status = "Good"
        } else if result.Loss < 1.0 {
            status = "Learning"
        } else {
            status = "Starting"
        }
        
        fmt.Printf("%5d | %.6f | %-7s | %.4f | %s\n", 
                   epoch, result.Loss, trend, rmse, status)
        
        prevLoss = result.Loss
        
        // Early stopping
        if result.Loss < 0.001 {
            fmt.Printf("🎉 Regression converged! (MSE < 0.001)\n")
            break
        }
    }
    
    return nil
}
```

### Step 6: Complete Linear Regression Program

```go
func main() {
    fmt.Println("📈 Regression Tutorial: Linear Prediction")
    
    // Initialize Metal device and memory manager
    device, err := cgo_bridge.CreateMetalDevice()
    if err != nil {
        log.Fatalf("Failed to create Metal device: %v", err)
    }
    defer cgo_bridge.DestroyMetalDevice(device)
    
    memory.InitializeGlobalMemoryManager(device)
    
    // Set random seed for reproducibility
    rand.Seed(42)
    
    // Problem setup
    batchSize := 64
    inputSize := 1
    outputSize := 1
    epochs := 100
    
    // Step 1: Build model
    model, err := buildLinearRegressionModel(batchSize, inputSize, outputSize)
    if err != nil {
        log.Fatalf("Model building failed: %v", err)
    }
    
    // Step 2: Setup trainer
    trainer, err := setupRegressionTrainer(model, batchSize)
    if err != nil {
        log.Fatalf("Trainer setup failed: %v", err)
    }
    defer trainer.Cleanup()
    
    // Step 3: Generate data
    inputData, targetData, inputShape, targetShape := generateLinearData(batchSize)
    
    // Step 4: Train model
    err = trainRegression(trainer, inputData, inputShape, targetData, targetShape, epochs)
    if err != nil {
        log.Fatalf("Training failed: %v", err)
    }
    
    fmt.Println("\n🎓 Linear Regression Complete!")
    fmt.Println("   ✅ Successfully learned linear relationship")
    fmt.Println("   ✅ Used MSE loss for regression")
    fmt.Println("   ✅ Demonstrated continuous value prediction")
}
```

## 🧠 Tutorial 2: Multi-Variable Regression

Now let's tackle a more realistic problem with multiple input features.

### House Price Prediction Model

```go
func generateHousePriceData(batchSize int) ([]float32, []float32, []int, []int) {
    fmt.Println("🏠 Generating House Price Dataset")
    
    inputSize := 5  // 5 features
    inputData := make([]float32, batchSize*inputSize)
    targetData := make([]float32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        baseIdx := i * inputSize
        
        // Feature 1: Square footage (1000-4000 sq ft)
        sqft := rand.Float32()*3000 + 1000
        inputData[baseIdx] = sqft / 4000.0  // Normalize to [0.25, 1.0]
        
        // Feature 2: Number of bedrooms (1-5)
        bedrooms := rand.Float32()*4 + 1
        inputData[baseIdx+1] = bedrooms / 5.0  // Normalize to [0.2, 1.0]
        
        // Feature 3: Number of bathrooms (1-4)
        bathrooms := rand.Float32()*3 + 1
        inputData[baseIdx+2] = bathrooms / 4.0  // Normalize to [0.25, 1.0]
        
        // Feature 4: Age of house (0-50 years)
        age := rand.Float32() * 50
        inputData[baseIdx+3] = 1.0 - (age / 50.0)  // Newer = higher value
        
        // Feature 5: Location score (0.1-1.0)
        location := rand.Float32()*0.9 + 0.1
        inputData[baseIdx+4] = location
        
        // Price calculation with realistic weights
        price := sqft*150 +                    // $150 per sq ft
                bedrooms*10000 +               // $10k per bedroom
                bathrooms*8000 +               // $8k per bathroom
                (50-age)*1000 +                // $1k per year newer
                location*50000 +               // Location bonus
                rand.Float32()*20000 - 10000   // Random noise ±$10k
        
        // Normalize price to reasonable range
        targetData[i] = price / 1000000.0  // Convert to millions
    }
    
    inputShape := []int{batchSize, inputSize}
    targetShape := []int{batchSize, 1}
    
    fmt.Printf("   ✅ Generated %d house samples\n", batchSize)
    fmt.Printf("   🏠 Features: sqft, bedrooms, bathrooms, age, location\n")
    fmt.Printf("   💰 Target: Price (normalized to millions)\n")
    
    return inputData, targetData, inputShape, targetShape
}
```

### Advanced Regression Architecture

```go
func buildHousePriceModel(batchSize, inputSize, outputSize int) (*layers.ModelSpec, error) {
    fmt.Println("🏗️ Building House Price Regression Model")
    
    inputShape := []int{batchSize, inputSize}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Feature extraction layers
        AddDense(64, true, "features1").
        AddReLU("relu1").
        AddDense(32, true, "features2").
        AddReLU("relu2").
        
        // Price prediction layer
        AddDense(16, true, "price_features").
        AddReLU("relu3").
        AddDense(outputSize, true, "price_output").
        // No final activation - raw continuous output
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("house price model compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: %d → 64 → 32 → 16 → %d\n", 
               inputSize, outputSize)
    fmt.Printf("   🏠 Features: Multi-layer feature extraction\n")
    fmt.Printf("   📊 Output: Single price prediction\n")
    
    return model, nil
}
```

## 🎯 Tutorial 3: Robust Regression with Different Loss Functions

Let's explore how different loss functions handle outliers and noise.

### Comparing MSE, MAE, and Huber Loss

Choosing the right loss function is crucial for regression performance. Here's a comparison of the three main loss functions available in go-metal:

| Loss Function | Description | Best Use Case | Formula |
|---------------|-------------|---------------|----------|
| **Mean Squared Error (MSE)** | Quadratic penalty, smooth gradients | Standard regression with normally distributed errors | `(y_true - y_pred)²` |
| **Mean Absolute Error (MAE)** | Linear penalty, robust to outliers | Data with outliers or heavy-tailed distributions | `|y_true - y_pred|` |
| **Huber Loss** | Combines MSE + MAE benefits | Best of both worlds - smooth near zero, robust to outliers | Quadratic for small errors, linear for large |

#### When to Use Each Loss Function

**Mean Squared Error (MSE)**
- ✅ Default choice for most regression problems
- ✅ Penalizes large errors more heavily (quadratic)
- ✅ Smooth gradients lead to stable training
- ❌ Sensitive to outliers
- Use when: Your data is clean and errors are normally distributed

**Mean Absolute Error (MAE)**
- ✅ Robust to outliers
- ✅ All errors weighted equally (linear penalty)
- ✅ More interpretable (same units as target)
- ❌ Non-smooth at zero (can cause training instability)
- Use when: Your data contains outliers or you want equal treatment of all errors

**Huber Loss**
- ✅ Combines benefits of MSE and MAE
- ✅ Smooth gradients near zero (like MSE)
- ✅ Robust to outliers (like MAE)
- ✅ Tunable delta parameter for transition point
- Use when: You want robustness without sacrificing training stability

### Outlier-Resistant Regression Example

```go
func generateNoisyData(batchSize int) ([]float32, []float32, []int, []int) {
    fmt.Println("📊 Generating Dataset with Outliers")
    
    inputData := make([]float32, batchSize)
    targetData := make([]float32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        x := rand.Float32()*4.0 - 2.0  // -2 to 2
        
        // True relationship: y = x² + x + noise
        y := x*x + x
        
        // Add outliers to 10% of data
        if rand.Float32() < 0.1 {
            // Outlier: add large random error
            outlierNoise := (rand.Float32() - 0.5) * 10.0
            y += outlierNoise
            fmt.Printf("   🚨 Outlier at x=%.2f: y=%.2f (noise=%.2f)\n", x, y, outlierNoise)
        } else {
            // Normal noise
            normalNoise := (rand.Float32() - 0.5) * 0.2
            y += normalNoise
        }
        
        inputData[i] = x
        targetData[i] = y
    }
    
    inputShape := []int{batchSize, 1}
    targetShape := []int{batchSize, 1}
    
    fmt.Printf("   ✅ Generated %d samples with ~10%% outliers\n", batchSize)
    
    return inputData, targetData, inputShape, targetShape
}
```

### Training with Different Loss Functions

```go
func trainWithDifferentLosses(model *layers.ModelSpec, inputData []float32, inputShape []int,
                             targetData []float32, targetShape []int, batchSize int) {
    
    lossFunctions := []training.LossFunction{
        training.MeanSquaredError,
        training.MeanAbsoluteError,
        training.Huber,
    }
    
    lossNames := []string{"MSE", "MAE", "Huber"}
    
    for i, lossFunc := range lossFunctions {
        fmt.Printf("\n🔍 Training with %s Loss\n", lossNames[i])
        
        config := training.TrainerConfig{
            BatchSize:     batchSize,
            LearningRate:  0.01,
            OptimizerType: cgo_bridge.Adam,
            EngineType:    training.Dynamic,
            ProblemType:   training.Regression,
            LossFunction:  lossFunc,  // Different loss function
            Beta1:         0.9,
            Beta2:         0.999,
            Epsilon:       1e-8,
        }
        
        trainer, err := training.NewModelTrainer(model, config)
        if err != nil {
            fmt.Printf("❌ Failed to create trainer with %s: %v\n", lossNames[i], err)
            continue
        }
        
        // Train for 20 epochs
        for epoch := 1; epoch <= 20; epoch++ {
            result, err := trainer.TrainBatch(inputData, inputShape, targetData, targetShape)
            if err != nil {
                fmt.Printf("❌ Training failed at epoch %d: %v\n", epoch, err)
                break
            }
            
            if epoch%5 == 0 {
                fmt.Printf("   Epoch %d: Loss = %.4f\n", epoch, result.Loss)
            }
        }
        
        trainer.Cleanup()
        
        fmt.Printf("   ✅ %s training completed\n", lossNames[i])
    }
}
```

## 📊 Tutorial 4: Time Series Regression

Apply regression to sequential data prediction.

### Time Series Data Generation

```go
func generateTimeSeriesData(batchSize, sequenceLength int) ([]float32, []float32, []int, []int) {
    fmt.Println("📈 Generating Time Series Dataset")
    
    inputData := make([]float32, batchSize*sequenceLength)
    targetData := make([]float32, batchSize)
    
    for i := 0; i < batchSize; i++ {
        baseIdx := i * sequenceLength
        
        // Generate sequence: sine wave + trend + noise
        startTime := rand.Float32() * 100.0
        
        var sequence []float32
        for t := 0; t < sequenceLength; t++ {
            time := startTime + float32(t)*0.1
            
            // Sine wave + linear trend + noise
            value := float32(math.Sin(float64(time))) +  // Sine component
                    time*0.01 +                          // Trend component
                    (rand.Float32()-0.5)*0.1             // Noise
            
            sequence = append(sequence, value)
            inputData[baseIdx+t] = value
        }
        
        // Target: predict next value in sequence
        nextTime := startTime + float32(sequenceLength)*0.1
        nextValue := float32(math.Sin(float64(nextTime))) + nextTime*0.01
        targetData[i] = nextValue
    }
    
    inputShape := []int{batchSize, sequenceLength}
    targetShape := []int{batchSize, 1}
    
    fmt.Printf("   ✅ Generated %d sequences of length %d\n", batchSize, sequenceLength)
    fmt.Printf("   📊 Pattern: sine wave + trend + noise\n")
    fmt.Printf("   🎯 Task: Predict next value in sequence\n")
    
    return inputData, targetData, inputShape, targetShape
}
```

### Time Series Regression Model

```go
func buildTimeSeriesModel(batchSize, sequenceLength, outputSize int) (*layers.ModelSpec, error) {
    fmt.Println("🏗️ Building Time Series Regression Model")
    
    inputShape := []int{batchSize, sequenceLength}
    builder := layers.NewModelBuilder(inputShape)
    
    model, err := builder.
        // Sequence processing layers
        AddDense(128, true, "sequence1").
        AddReLU("relu1").
        AddDense(64, true, "sequence2").
        AddReLU("relu2").
        
        // Temporal feature extraction
        AddDense(32, true, "temporal").
        AddReLU("relu3").
        
        // Final prediction
        AddDense(outputSize, true, "prediction").
        Compile()
    
    if err != nil {
        return nil, fmt.Errorf("time series model compilation failed: %v", err)
    }
    
    fmt.Printf("   ✅ Architecture: %d → 128 → 64 → 32 → %d\n", 
               sequenceLength, outputSize)
    fmt.Printf("   📈 Purpose: Sequential pattern learning\n")
    
    return model, nil
}
```

## 🎓 Advanced Regression Techniques

### Feature Scaling and Normalization

Feature scaling is crucial for regression models to ensure all features contribute equally to the learning process and to improve numerical stability during optimization.

#### Why Scale Features?

Features often have different units and ranges which can cause problems:
- **Different units**: Age (0-100 years) vs Income ($0-$1,000,000) vs Square footage (500-5000 sq ft)
- **Different ranges**: Small features get overshadowed by large ones
- **Optimization efficiency**: Gradient descent converges faster with scaled features
- **Numerical stability**: Prevents overflow/underflow in calculations

#### Common Scaling Methods

**1. Min-Max Normalization**
Scales features to a fixed range, typically [0, 1]:
```go
x_scaled = (x - min) / (max - min)
```
- **Pros**: Bounded range, preserves zero values
- **Cons**: Sensitive to outliers
- **Use when**: You know the approximate min/max values

**2. Z-Score Standardization**
Transforms features to have zero mean and unit variance:
```go
x_scaled = (x - mean) / std
```
- **Pros**: Handles outliers better than min-max
- **Cons**: No bounded range
- **Use when**: Features are normally distributed

**3. Robust Scaling**
Uses median and interquartile range (IQR) instead of mean and standard deviation:
```go
x_scaled = (x - median) / IQR
```
- **Pros**: Very robust to outliers
- **Cons**: Less common, may need explanation
- **Use when**: Dataset contains many outliers

#### Practical Implementation Example

```go
// Min-max scaling implementation
func minMaxScale(data []float32) []float32 {
    min := data[0]
    max := data[0]
    
    // Find min and max
    for _, val := range data {
        if val < min {
            min = val
        }
        if val > max {
            max = val
        }
    }
    
    // Scale data
    scaled := make([]float32, len(data))
    rangeVal := max - min
    if rangeVal > 0 {
        for i, val := range data {
            scaled[i] = (val - min) / rangeVal
        }
    }
    
    return scaled
}

// Z-score standardization implementation
func standardize(data []float32) []float32 {
    // Calculate mean
    var sum float32
    for _, val := range data {
        sum += val
    }
    mean := sum / float32(len(data))
    
    // Calculate standard deviation
    var variance float32
    for _, val := range data {
        diff := val - mean
        variance += diff * diff
    }
    std := float32(math.Sqrt(float64(variance / float32(len(data)))))
    
    // Standardize data
    scaled := make([]float32, len(data))
    if std > 0 {
        for i, val := range data {
            scaled[i] = (val - mean) / std
        }
    }
    
    return scaled
}
```

### Regression Evaluation Metrics

Evaluating regression models requires understanding different metrics, each providing unique insights into model performance.

#### Key Regression Metrics

**1. Mean Squared Error (MSE)**
- **Formula**: `MSE = (1/n) * Σ(y_true - y_pred)²`
- **Interpretation**: Average of squared differences; lower is better
- **Units**: Squared units of the target variable
- **Use Case**: Standard metric for model comparison
- **Pros**: Penalizes large errors heavily, smooth for optimization
- **Cons**: Not interpretable in original units, sensitive to outliers

**2. Root Mean Squared Error (RMSE)**
- **Formula**: `RMSE = √MSE`
- **Interpretation**: Square root of MSE; lower is better
- **Units**: Same as target variable
- **Use Case**: When you need interpretable error magnitude
- **Pros**: Interpretable units, maintains MSE properties
- **Cons**: Still sensitive to outliers

**3. Mean Absolute Error (MAE)**
- **Formula**: `MAE = (1/n) * Σ|y_true - y_pred|`
- **Interpretation**: Average absolute difference; lower is better
- **Units**: Same as target variable
- **Use Case**: When robustness to outliers is important
- **Pros**: Robust to outliers, easy to interpret
- **Cons**: Less smooth for optimization

**4. R² Score (Coefficient of Determination)**
- **Formula**: `R² = 1 - (SS_residual / SS_total)`
- **Interpretation**: Proportion of variance explained; 0-1, higher is better
- **Units**: Dimensionless (percentage when multiplied by 100)
- **Use Case**: Understanding model explanatory power
- **Pros**: Normalized metric, easy comparison across datasets
- **Cons**: Can be misleading for non-linear relationships

#### Practical Implementation

```go
// Calculate regression metrics
func calculateMetrics(predictions, targets []float32) map[string]float32 {
    n := float32(len(predictions))
    var mse, mae, targetMean float32
    
    // Calculate target mean for R²
    for _, target := range targets {
        targetMean += target
    }
    targetMean /= n
    
    // Calculate errors and variances
    var ssRes, ssTot float32
    for i := range predictions {
        error := targets[i] - predictions[i]
        mse += error * error
        mae += float32(math.Abs(float64(error)))
        
        ssRes += error * error
        ssTot += (targets[i] - targetMean) * (targets[i] - targetMean)
    }
    
    mse /= n
    mae /= n
    rmse := float32(math.Sqrt(float64(mse)))
    r2 := 1.0 - (ssRes / ssTot)
    
    return map[string]float32{
        "MSE":  mse,
        "RMSE": rmse,
        "MAE":  mae,
        "R2":   r2,
    }
}
```

### Model Validation Techniques

Proper validation is essential to ensure your regression model generalizes well to unseen data. Here are the key techniques:

#### Train-Validation-Test Split

Divide your dataset into three parts:
- **Training Set (60-70%)**: Used for model learning and parameter updates
- **Validation Set (15-20%)**: Used for hyperparameter tuning and model selection
- **Test Set (15-20%)**: Used for final, unbiased evaluation

```go
// Example data split implementation
func splitData(data []float32, labels []float32, trainRatio, valRatio float32) (
    trainData, trainLabels,
    valData, valLabels,
    testData, testLabels []float32) {
    
    n := len(data)
    trainEnd := int(float32(n) * trainRatio)
    valEnd := trainEnd + int(float32(n) * valRatio)
    
    trainData = data[:trainEnd]
    trainLabels = labels[:trainEnd]
    valData = data[trainEnd:valEnd]
    valLabels = labels[trainEnd:valEnd]
    testData = data[valEnd:]
    testLabels = labels[valEnd:]
    
    return
}
```

#### Cross-Validation

**K-Fold Cross-Validation** provides more robust evaluation:
1. Split data into K equal parts (folds)
2. Train on K-1 folds, validate on the remaining fold
3. Repeat K times, using each fold as validation once
4. Average the K results for final metric

**Benefits**:
- Uses all data for both training and validation
- Reduces variance in performance estimates
- Especially useful for smaller datasets

#### Overfitting Detection and Prevention

**Signs of Overfitting**:
- Training loss continues decreasing while validation loss increases
- Large gap between training and validation performance
- Model performs poorly on new, unseen data
- Extremely high R² on training data but low on validation

**Prevention Techniques**:

1. **Regularization**
   - L1 (Lasso): Encourages sparsity, feature selection
   - L2 (Ridge): Prevents large weights, smooths model
   - Elastic Net: Combines L1 and L2

2. **Early Stopping**
   - Monitor validation loss during training
   - Stop when validation loss stops improving
   - Save best model checkpoint

3. **Data Augmentation**
   - Generate synthetic training examples
   - Add controlled noise to inputs
   - Apply domain-specific transformations

4. **Model Simplification**
   - Reduce number of layers/parameters
   - Use dropout for neural networks
   - Feature selection to remove irrelevant inputs

## 🛠️ Practical Applications

### Complete House Price Prediction

```go
func completeHousePriceExample() {
    fmt.Println("🏠 Complete House Price Prediction Example")
    
    // ... (device setup)
    
    batchSize := 128
    inputSize := 5    // sqft, bedrooms, bathrooms, age, location
    outputSize := 1   // price
    epochs := 200
    
    // Build advanced model
    model, _ := buildHousePriceModel(batchSize, inputSize, outputSize)
    
    // Use Huber loss for robustness
    config := training.TrainerConfig{
        BatchSize:     batchSize,
        LearningRate:  0.001,
        OptimizerType: cgo_bridge.Adam,
        EngineType:    training.Dynamic,
        ProblemType:   training.Regression,
        LossFunction:  training.Huber,  // Robust to outlier prices
        Beta1:         0.9,
        Beta2:         0.999,
        Epsilon:       1e-8,
    }
    
    trainer, _ := training.NewModelTrainer(model, config)
    defer trainer.Cleanup()
    
    // Generate realistic house data
    inputData, targetData, inputShape, targetShape := generateHousePriceData(batchSize)
    
    // Training with progress monitoring
    fmt.Println("🚀 Training house price model...")
    for epoch := 1; epoch <= epochs; epoch++ {
        result, err := trainer.TrainBatch(inputData, inputShape, targetData, targetShape)
        if err != nil {
            fmt.Printf("❌ Training failed: %v\n", err)
            break
        }
        
        if epoch%20 == 0 {
            rmse := math.Sqrt(float64(result.Loss))
            fmt.Printf("Epoch %d: Huber Loss = %.6f, RMSE ≈ %.4f\n", 
                       epoch, result.Loss, rmse)
        }
    }
    
    fmt.Println("✅ House price model training completed!")
}
```

### Stock Price Prediction

#### Problem Overview

Stock price prediction demonstrates time series regression with unique challenges:

**Problem Setup**:
- **Input**: Historical price sequences (open, high, low, close, volume)
- **Output**: Next period's price or price movement
- **Challenges**: High volatility, market noise, external factors

#### Model Architecture Considerations

**Input Features**:
- Raw price data (OHLCV)
- Technical indicators (moving averages, RSI, MACD)
- Price transformations (returns, log returns)
- Volume indicators
- Time-based features (day of week, month)

**Loss Function Selection**:
- **MAE or Huber Loss**: More robust to price spikes and outliers
- **Custom Loss**: Weight recent errors more heavily
- **Directional Loss**: Penalize wrong direction predictions more

**Architecture Design**:
```go
// Example stock prediction model structure
model, err := builder.
    // Process sequential data
    AddDense(256, true, "sequence_processing").
    AddReLU("relu1").
    AddDropout(0.2, "dropout1").  // Prevent overfitting
    
    // Extract temporal patterns
    AddDense(128, true, "temporal_features").
    AddReLU("relu2").
    AddDropout(0.2, "dropout2").
    
    // Final prediction
    AddDense(64, true, "prediction_layer").
    AddReLU("relu3").
    AddDense(1, true, "price_output").
    Compile()
```

#### Important Considerations

⚠️ **Disclaimer**: Stock market prediction is extremely challenging due to:
- Market efficiency hypothesis
- External factors (news, economics, sentiment)
- Non-stationary data distributions
- High noise-to-signal ratio

**Best Practices**:
1. Treat as a learning exercise, not investment advice
2. Use proper backtesting with realistic constraints
3. Account for transaction costs and slippage
4. Implement risk management strategies
5. Consider ensemble methods for robustness

## 🎯 Best Practices Summary

### Regression Model Design

#### Architecture Design Best Practices

**1. Start Simple, Add Complexity Gradually**
- Begin with a linear model (single dense layer)
- Add hidden layers only if linear model underperforms
- Each layer should have a clear purpose
- Monitor validation performance with each addition

**2. Layer Configuration**
- **Hidden Layers**: Use ReLU activation for non-linearity
- **Output Layer**: No activation function (raw continuous output)
- **Layer Sizes**: Gradually decrease neurons (funnel architecture)
- **Initialization**: Use appropriate weight initialization (Xavier/He)

**Example Progressive Architecture**:
```go
// Start simple
model := builder.AddDense(1, true, "linear").Compile()

// Add complexity if needed
model := builder.
    AddDense(64, true, "hidden1").AddReLU("relu1").
    AddDense(32, true, "hidden2").AddReLU("relu2").
    AddDense(1, true, "output").Compile()
```

#### Data Preparation Guidelines

**Essential Steps**:
1. **Feature Scaling**: Normalize or standardize all inputs
2. **Outlier Handling**: Identify and handle appropriately
3. **Missing Values**: Impute or remove
4. **Feature Engineering**: Create meaningful derived features
5. **Data Splitting**: Ensure no data leakage between sets

**Data Leakage Prevention**:
- Scale using only training set statistics
- Don't use future information in features
- Maintain temporal order for time series
- Validate feature engineering on separate data

#### Loss Function Selection Strategy

| Data Characteristics | Recommended Loss | Reason |
|---------------------|------------------|--------|
| Clean, normal distribution | MSE | Smooth optimization, fast convergence |
| Contains outliers | MAE or Huber | Robust to extreme values |
| Mixed characteristics | Huber | Balanced approach |
| Financial data | MAE | Equal treatment of all errors |
| Scientific measurements | MSE | Assumes Gaussian noise |

#### Training Configuration

**Hyperparameter Guidelines**:
- **Learning Rate**: Start with 0.001-0.01 (lower than classification)
- **Batch Size**: 32-128 (larger for stable gradients)
- **Optimizer**: Adam (good default), SGD with momentum (simpler)
- **Epochs**: Use early stopping based on validation loss

**Monitoring and Evaluation**:
1. Track both training and validation metrics
2. Plot learning curves to detect overfitting
3. Use multiple metrics (MSE, MAE, R²)
4. Implement early stopping with patience
5. Save best model based on validation performance

**Early Stopping Implementation**:
```go
bestLoss := float32(math.Inf(1))
patience := 10
noImprovement := 0

for epoch := 1; epoch <= maxEpochs; epoch++ {
    // Training step
    result, _ := trainer.TrainBatch(...)
    
    // Validation step
    valLoss := evaluateValidation(...)
    
    if valLoss < bestLoss {
        bestLoss = valLoss
        noImprovement = 0
        // Save model checkpoint
    } else {
        noImprovement++
        if noImprovement >= patience {
            fmt.Println("Early stopping triggered")
            break
        }
    }
}
```

### Common Pitfalls and Solutions

#### 1. Unscaled Features

**Problem**: Different feature scales cause optimization issues
**Symptoms**:
- Extremely slow convergence
- Loss jumping erratically
- Gradient explosion or vanishing
- Some features dominating others

**Solutions**:
```go
// Always scale your features
scaledData := make([]float32, len(data))
for i := range data {
    scaledData[i] = (data[i] - min) / (max - min)  // Min-max scaling
    // OR
    scaledData[i] = (data[i] - mean) / std  // Standardization
}
```

#### 2. Wrong Loss Function Choice

**Problem**: Loss function doesn't match data characteristics
**Symptoms**:
- Model sensitive to outliers (using MSE)
- Unstable training (using MAE)
- Poor convergence

**Solutions**:
- Clean data → MSE
- Outliers present → MAE or Huber
- Unknown distribution → Start with Huber
- Monitor performance with multiple metrics

#### 3. Overfitting

**Problem**: Model memorizes training data
**Symptoms**:
- Training loss << validation loss
- Perfect training performance, poor test performance
- Loss curves diverge after initial epochs

**Solutions**:
1. **Regularization**:
   ```go
   // Add L2 regularization
   config.L2Penalty = 0.001
   ```
2. **Dropout layers** (for neural networks)
3. **Reduce model complexity**
4. **Collect more training data**
5. **Data augmentation**

#### 4. Underfitting

**Problem**: Model too simple for the data
**Symptoms**:
- High training AND validation loss
- Loss plateaus early
- Poor performance across all metrics

**Solutions**:
1. **Increase model capacity**:
   ```go
   // Add more layers or neurons
   builder.AddDense(128, true, "layer1").
          AddReLU("relu1").
          AddDense(64, true, "layer2")
   ```
2. **Feature engineering** - create polynomial or interaction features
3. **Reduce regularization**
4. **Train for more epochs**

#### 5. Data Leakage

**Problem**: Test information leaks into training
**Symptoms**:
- Unrealistically high performance
- Model fails in production
- Large train-test performance gap

**Common Causes and Solutions**:

| Leakage Type | Example | Solution |
|--------------|---------|----------|
| Temporal | Using future data in features | Maintain strict time ordering |
| Preprocessing | Scaling with full dataset stats | Scale using only training data |
| Duplicate data | Same samples in train and test | Remove duplicates before splitting |
| Target leakage | Features contain target information | Careful feature selection |

**Prevention Checklist**:
```go
// Correct approach
trainData, testData := splitData(data)
scaler := fitScaler(trainData)  // Fit only on training
trainScaled := scaler.transform(trainData)
testScaled := scaler.transform(testData)  // Apply same scaling
```

#### Quick Diagnostic Guide

| Symptom | Likely Cause | First Action |
|---------|--------------|--------------||
| Loss = NaN | Unscaled features, high LR | Check data scaling, reduce LR |
| Loss not decreasing | Underfitting, wrong loss | Increase capacity, check loss function |
| Validation loss increases | Overfitting | Add regularization, early stopping |
| Erratic loss values | Batch too small, high LR | Increase batch size, reduce LR |
| Good metrics, bad predictions | Data leakage | Review preprocessing pipeline |

## 🚀 Next Steps

You've mastered regression with go-metal! Continue your journey:

- **[CNN Tutorial](cnn-tutorial.md)** - Apply regression to computer vision
- **[Performance Guide](../guides/performance.md)** - Optimize regression training
- **[Mixed Precision Tutorial](mixed-precision.md)** - Faster regression with FP16
- **[Advanced Examples](../examples/)** - Real-world regression applications

**Ready for production?** Check out the [House Price Regression Example](../examples/house-price-regression.md) for a complete project.

---

## 🧠 Key Takeaways

- **Regression predicts continuous values**: Unlike classification's discrete classes
- **Loss function choice matters**: MSE for standard, MAE/Huber for robustness
- **Feature scaling is crucial**: Normalize inputs for stable training
- **No final activation**: Regression outputs raw continuous values
- **Evaluation requires multiple metrics**: MSE, RMSE, MAE, R² each tell different stories
- **Go-metal advantages**: GPU-resident computation, optimized loss functions, numerical stability

With these skills, you can tackle any regression problem using go-metal on Apple Silicon!