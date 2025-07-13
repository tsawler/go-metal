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

```go
func compareRegressionLosses() {
    fmt.Println("🔍 Comparing Regression Loss Functions")
    
    lossFunctions := []struct {
        name string
        lossType training.LossFunction
        description string
        useCase string
    }{
        {
            "Mean Squared Error",
            training.MeanSquaredError,
            "Quadratic penalty, smooth gradients",
            "Standard regression, normal data",
        },
        {
            "Mean Absolute Error", 
            training.MeanAbsoluteError,
            "Linear penalty, robust to outliers",
            "Data with outliers, robust regression",
        },
        {
            "Huber Loss",
            training.Huber,
            "Combines MSE + MAE benefits",
            "Best of both worlds",
        },
    }
    
    fmt.Printf("%-20s | %-35s | %-30s\n", "Loss Function", "Description", "Best Use Case")
    fmt.Println("---------------------|-------------------------------------|------------------------------")
    
    for _, loss := range lossFunctions {
        fmt.Printf("%-20s | %-35s | %-30s\n", 
                   loss.name, loss.description, loss.useCase)
    }
}
```

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

```go
func demonstrateFeatureScaling() {
    fmt.Println("📊 Feature Scaling Techniques")
    
    fmt.Println("\n🎯 Why Scale Features?")
    fmt.Println("   • Different units (age vs income vs sqft)")
    fmt.Println("   • Different ranges (0-100 vs 0-1000000)")
    fmt.Println("   • Optimization efficiency")
    fmt.Println("   • Numerical stability")
    
    fmt.Println("\n📏 Scaling Methods:")
    
    fmt.Println("\n1. Min-Max Normalization:")
    fmt.Println("   x_scaled = (x - min) / (max - min)")
    fmt.Println("   Result: [0, 1] range")
    
    fmt.Println("\n2. Z-Score Standardization:")
    fmt.Println("   x_scaled = (x - mean) / std")
    fmt.Println("   Result: mean=0, std=1")
    
    fmt.Println("\n3. Robust Scaling:")
    fmt.Println("   x_scaled = (x - median) / IQR")
    fmt.Println("   Result: Less sensitive to outliers")
    
    fmt.Println("\n💡 Implementation:")
    fmt.Println("   // Min-max example")
    fmt.Println("   min, max := 0.0, 1000.0")
    fmt.Println("   scaled := (value - min) / (max - min)")
}
```

### Regression Evaluation Metrics

```go
func regressionEvaluationMetrics() {
    fmt.Println("📈 Regression Evaluation Metrics")
    
    metrics := []struct {
        name string
        formula string
        interpretation string
        use_case string
    }{
        {
            "Mean Squared Error (MSE)",
            "MSE = (1/n) * Σ(y_true - y_pred)²",
            "Lower is better, units²",
            "Standard regression metric",
        },
        {
            "Root Mean Squared Error (RMSE)",
            "RMSE = √MSE",
            "Lower is better, same units as target",
            "Interpretable error magnitude",
        },
        {
            "Mean Absolute Error (MAE)",
            "MAE = (1/n) * Σ|y_true - y_pred|",
            "Lower is better, same units as target",
            "Robust to outliers",
        },
        {
            "R² Score (Coefficient of Determination)",
            "R² = 1 - SS_res/SS_tot",
            "0-1, higher is better",
            "Proportion of variance explained",
        },
    }
    
    fmt.Printf("%-35s | %-30s | %-25s | %-25s\n",
               "Metric", "Formula", "Interpretation", "Use Case")
    fmt.Println("-------------------------------------|--------------------------------|---------------------------|-------------------------")
    
    for _, metric := range metrics {
        fmt.Printf("%-35s | %-30s | %-25s | %-25s\n",
                   metric.name, metric.formula, metric.interpretation, metric.use_case)
    }
}
```

### Model Validation Techniques

```go
func demonstrateModelValidation() {
    fmt.Println("✅ Model Validation Techniques")
    
    fmt.Println("\n🎯 Train-Validation-Test Split:")
    fmt.Println("   • Training: 60-70% (model learning)")
    fmt.Println("   • Validation: 15-20% (hyperparameter tuning)")
    fmt.Println("   • Test: 15-20% (final evaluation)")
    
    fmt.Println("\n🔄 Cross-Validation:")
    fmt.Println("   • K-fold: Split data into K parts")
    fmt.Println("   • Train on K-1 parts, validate on 1")
    fmt.Println("   • Repeat K times, average results")
    
    fmt.Println("\n📊 Overfitting Detection:")
    fmt.Println("   • Training loss decreases, validation increases")
    fmt.Println("   • Large gap between training and validation")
    fmt.Println("   • Model memorizes rather than generalizes")
    
    fmt.Println("\n🔧 Prevention Techniques:")
    fmt.Println("   • Regularization (L1, L2, dropout)")
    fmt.Println("   • Early stopping")
    fmt.Println("   • More training data")
    fmt.Println("   • Simpler model architecture")
}
```

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

```go
func stockPricePredictionExample() {
    fmt.Println("📈 Stock Price Prediction Example")
    
    fmt.Println("\n🎯 Problem Setup:")
    fmt.Println("   • Input: Historical prices (sequence)")
    fmt.Println("   • Output: Next day's price")
    fmt.Println("   • Challenge: High volatility, noise")
    
    fmt.Println("\n🔧 Model Considerations:")
    fmt.Println("   • Use MAE or Huber (robust to outliers)")
    fmt.Println("   • Sequence-to-one prediction")
    fmt.Println("   • Feature engineering (moving averages, etc.)")
    
    fmt.Println("\n⚠️ Important Notes:")
    fmt.Println("   • Stock prediction is inherently difficult")
    fmt.Println("   • Past performance ≠ future results")
    fmt.Println("   • Consider this as a technical exercise")
}
```

## 🎯 Best Practices Summary

### Regression Model Design

```go
func regressionBestPractices() {
    fmt.Println("🎓 Regression Best Practices")
    
    fmt.Println("\n🏗️ Architecture Design:")
    fmt.Println("   ✅ Start with simple linear model")
    fmt.Println("   ✅ Add complexity gradually (more layers)")
    fmt.Println("   ✅ No activation on final layer (raw output)")
    fmt.Println("   ✅ Use ReLU for hidden layers")
    
    fmt.Println("\n📊 Data Preparation:")
    fmt.Println("   ✅ Scale/normalize input features")
    fmt.Println("   ✅ Handle outliers appropriately")
    fmt.Println("   ✅ Split data properly (train/val/test)")
    fmt.Println("   ✅ Check for data leakage")
    
    fmt.Println("\n🎯 Loss Function Selection:")
    fmt.Println("   ✅ MSE: Standard choice, smooth gradients")
    fmt.Println("   ✅ MAE: Robust to outliers")
    fmt.Println("   ✅ Huber: Best of both worlds")
    
    fmt.Println("\n⚙️ Training Configuration:")
    fmt.Println("   ✅ Lower learning rates than classification")
    fmt.Println("   ✅ Monitor validation loss for overfitting")
    fmt.Println("   ✅ Use early stopping")
    fmt.Println("   ✅ Evaluate with multiple metrics")
}
```

### Common Pitfalls and Solutions

```go
func commonRegressionPitfalls() {
    fmt.Println("⚠️ Common Regression Pitfalls")
    
    pitfalls := []struct {
        problem string
        symptom string
        solution string
    }{
        {
            "Unscaled features",
            "Slow convergence, numerical issues",
            "Normalize inputs to [0,1] or standardize",
        },
        {
            "Wrong loss function",
            "Poor convergence, outlier sensitivity",
            "Use MAE/Huber for outliers, MSE for clean data",
        },
        {
            "Overfitting",
            "Training loss << validation loss",
            "Regularization, more data, simpler model",
        },
        {
            "Underfitting",
            "High training and validation loss",
            "More complex model, better features",
        },
        {
            "Data leakage",
            "Unrealistically good performance",
            "Careful feature engineering, proper splits",
        },
    }
    
    fmt.Printf("%-20s | %-30s | %-35s\n", "Problem", "Symptom", "Solution")
    fmt.Println("---------------------|--------------------------------|------------------------------------")
    
    for _, pitfall := range pitfalls {
        fmt.Printf("%-20s | %-30s | %-35s\n",
                   pitfall.problem, pitfall.symptom, pitfall.solution)
    }
}
```

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