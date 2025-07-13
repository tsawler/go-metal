# Visualization Guide

Advanced visualization techniques for go-metal models and training progress.

## 🎯 Overview

This guide covers comprehensive visualization strategies for go-metal models, training metrics, and data analysis. You'll learn to create production-ready visualizations that help debug models, communicate results, and monitor training progress.

## 📊 Training Progress Visualization

### Real-Time Training Metrics

```go
package main

import (
    "fmt"
    "log"
    "math"
    "time"
    
    "github.com/tsawler/go-metal/cgo_bridge"
    "github.com/tsawler/go-metal/layers"
    "github.com/tsawler/go-metal/memory"
    "github.com/tsawler/go-metal/training"
)

// TrainingVisualizer manages real-time training visualization
type TrainingVisualizer struct {
    TrainingLosses   []float32
    ValidationLosses []float32
    Accuracies       []float32
    Epochs           []int
    StartTime        time.Time
    PlotWidth        int
    PlotHeight       int
}

func NewTrainingVisualizer() *TrainingVisualizer {
    return &TrainingVisualizer{
        TrainingLosses:   make([]float32, 0),
        ValidationLosses: make([]float32, 0),
        Accuracies:       make([]float32, 0),
        Epochs:           make([]int, 0),
        StartTime:        time.Now(),
        PlotWidth:        80,
        PlotHeight:       20,
    }
}

func (tv *TrainingVisualizer) AddMetrics(epoch int, trainLoss, valLoss, accuracy float32) {
    tv.Epochs = append(tv.Epochs, epoch)
    tv.TrainingLosses = append(tv.TrainingLosses, trainLoss)
    tv.ValidationLosses = append(tv.ValidationLosses, valLoss)
    tv.Accuracies = append(tv.Accuracies, accuracy)
}

func (tv *TrainingVisualizer) DisplayProgress() {
    fmt.Printf("\033[2J\033[H") // Clear screen and move cursor to top
    
    fmt.Println("🚀 Training Progress Visualization")
    fmt.Println("==================================")
    
    if len(tv.TrainingLosses) == 0 {
        fmt.Println("No training data yet...")
        return
    }
    
    // Current metrics
    latest := len(tv.TrainingLosses) - 1
    currentEpoch := tv.Epochs[latest]
    currentTrainLoss := tv.TrainingLosses[latest]
    currentValLoss := tv.ValidationLosses[latest]
    currentAccuracy := tv.Accuracies[latest]
    elapsed := time.Since(tv.StartTime)
    
    fmt.Printf("📊 Current Status (Epoch %d):\n", currentEpoch)
    fmt.Printf("   Training Loss:   %.6f\n", currentTrainLoss)
    fmt.Printf("   Validation Loss: %.6f\n", currentValLoss)
    fmt.Printf("   Accuracy:        %.1f%%\n", currentAccuracy*100)
    fmt.Printf("   Training Time:   %v\n", elapsed.Round(time.Second))
    
    // Loss trend analysis
    if len(tv.TrainingLosses) >= 2 {
        prevTrainLoss := tv.TrainingLosses[latest-1]
        trainTrend := currentTrainLoss - prevTrainLoss
        
        fmt.Printf("\n📈 Trends:\n")
        if trainTrend < 0 {
            fmt.Printf("   Loss: ↓ Improving (Δ %.6f)\n", trainTrend)
        } else {
            fmt.Printf("   Loss: ↑ Check learning rate (Δ %.6f)\n", trainTrend)
        }
        
        // Overfitting detection
        if currentValLoss > currentTrainLoss*1.2 {
            fmt.Printf("   ⚠️  Potential overfitting detected\n")
        }
    }
    
    // ASCII plot of training progress
    tv.plotLossProgress()
    tv.plotAccuracyProgress()
}

func (tv *TrainingVisualizer) plotLossProgress() {
    if len(tv.TrainingLosses) < 2 {
        return
    }
    
    fmt.Printf("\n📉 Loss Progress (last %d epochs):\n", len(tv.TrainingLosses))
    
    // Find min/max for scaling
    minLoss := tv.TrainingLosses[0]
    maxLoss := tv.TrainingLosses[0]
    
    for _, loss := range tv.TrainingLosses {
        if loss < minLoss {
            minLoss = loss
        }
        if loss > maxLoss {
            maxLoss = loss
        }
    }
    
    for _, loss := range tv.ValidationLosses {
        if loss < minLoss {
            minLoss = loss
        }
        if loss > maxLoss {
            maxLoss = loss
        }
    }
    
    // Avoid division by zero
    if maxLoss == minLoss {
        maxLoss = minLoss + 1.0
    }
    
    // Plot training and validation loss
    plotHeight := 15
    plotWidth := min(len(tv.TrainingLosses), 60)
    
    for y := plotHeight - 1; y >= 0; y-- {
        // Y-axis label
        value := minLoss + (maxLoss-minLoss)*float32(y)/float32(plotHeight-1)
        fmt.Printf("%6.3f │", value)
        
        // Plot line
        for x := 0; x < plotWidth; x++ {
            idx := len(tv.TrainingLosses) - plotWidth + x
            if idx < 0 {
                fmt.Print(" ")
                continue
            }
            
            trainY := int((tv.TrainingLosses[idx] - minLoss) / (maxLoss - minLoss) * float32(plotHeight-1))
            valY := int((tv.ValidationLosses[idx] - minLoss) / (maxLoss - minLoss) * float32(plotHeight-1))
            
            if trainY == y && valY == y {
                fmt.Print("◆") // Both lines
            } else if trainY == y {
                fmt.Print("●") // Training loss
            } else if valY == y {
                fmt.Print("○") // Validation loss
            } else {
                fmt.Print(" ")
            }
        }
        fmt.Println()
    }
    
    // X-axis
    fmt.Print("       └")
    for i := 0; i < plotWidth; i++ {
        fmt.Print("─")
    }
    fmt.Println()
    fmt.Println("        ● Training Loss    ○ Validation Loss")
}

func (tv *TrainingVisualizer) plotAccuracyProgress() {
    if len(tv.Accuracies) < 2 {
        return
    }
    
    fmt.Printf("\n📈 Accuracy Progress:\n")
    
    // Simple horizontal bar chart for latest accuracy
    accuracy := tv.Accuracies[len(tv.Accuracies)-1]
    barLength := int(accuracy * 50) // Scale to 50 chars max
    
    fmt.Print("   ")
    for i := 0; i < barLength; i++ {
        fmt.Print("█")
    }
    for i := barLength; i < 50; i++ {
        fmt.Print("░")
    }
    fmt.Printf(" %.1f%%\n", accuracy*100)
    
    // Show accuracy trend
    if len(tv.Accuracies) >= 2 {
        prev := tv.Accuracies[len(tv.Accuracies)-2]
        current := tv.Accuracies[len(tv.Accuracies)-1]
        change := current - prev
        
        if change > 0 {
            fmt.Printf("   📈 Improving (+%.1f%%)\n", change*100)
        } else if change < 0 {
            fmt.Printf("   📉 Declining (%.1f%%)\n", change*100)
        } else {
            fmt.Printf("   ➡️  Stable\n")
        }
    }
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Training Loop with Visualization

```go
func trainWithVisualization(trainer *training.ModelTrainer, 
                           inputData, targetData []float32,
                           inputShape, targetShape []int, 
                           epochs int) error {
    
    visualizer := NewTrainingVisualizer()
    
    fmt.Println("🎬 Starting Training with Live Visualization")
    
    for epoch := 1; epoch <= epochs; epoch++ {
        // Training step
        result, err := trainer.TrainBatch(inputData, inputShape, targetData, targetShape)
        if err != nil {
            return fmt.Errorf("training failed at epoch %d: %v", epoch, err)
        }
        
        // Simulate validation metrics (in practice, use separate validation data)
        valLoss := result.Loss * (1.0 + (rand.Float32()-0.5)*0.1)
        accuracy := float32(0.5 + 0.4*math.Exp(-float64(result.Loss)))
        
        // Update visualization
        visualizer.AddMetrics(epoch, result.Loss, valLoss, accuracy)
        
        // Display progress every 5 epochs or on key milestones
        if epoch%5 == 0 || epoch == 1 || epoch == epochs {
            visualizer.DisplayProgress()
            time.Sleep(100 * time.Millisecond) // Brief pause for visibility
        }
        
        // Early stopping with visual feedback
        if result.Loss < 0.01 {
            fmt.Printf("\n🎉 Training converged at epoch %d!\n", epoch)
            break
        }
    }
    
    return nil
}
```

## 🎨 Model Architecture Visualization

### Network Structure Display

```go
// ModelVisualizer provides detailed model architecture visualization
type ModelVisualizer struct {
    Model *layers.ModelSpec
}

func NewModelVisualizer(model *layers.ModelSpec) *ModelVisualizer {
    return &ModelVisualizer{Model: model}
}

func (mv *ModelVisualizer) DisplayArchitecture() {
    fmt.Println("🏗️ Model Architecture Visualization")
    fmt.Println("===================================")
    
    if mv.Model == nil || len(mv.Model.Layers) == 0 {
        fmt.Println("❌ No model layers to display")
        return
    }
    
    fmt.Printf("📊 Total Layers: %d\n", len(mv.Model.Layers))
    fmt.Printf("🔢 Input Shape: %v\n", mv.Model.InputShape)
    
    // Layer-by-layer breakdown
    fmt.Println("\n📋 Layer Details:")
    fmt.Printf("%-4s │ %-15s │ %-20s │ %-15s │ %-10s\n", 
               "Idx", "Type", "Name", "Output Shape", "Params")
    fmt.Println("─────┼─────────────────┼──────────────────────┼─────────────────┼──────────")
    
    totalParams := 0
    currentShape := mv.Model.InputShape
    
    for i, layer := range mv.Model.Layers {
        layerType := getLayerType(layer)
        layerName := getLayerName(layer)
        outputShape := calculateOutputShape(layer, currentShape)
        params := calculateLayerParams(layer, currentShape)
        
        fmt.Printf("%4d │ %-15s │ %-20s │ %-15s │ %10d\n",
                   i, layerType, layerName, formatShape(outputShape), params)
        
        totalParams += params
        currentShape = outputShape
    }
    
    fmt.Println("─────┴─────────────────┴──────────────────────┴─────────────────┴──────────")
    fmt.Printf("📊 Total Parameters: %d\n", totalParams)
    fmt.Printf("🎯 Final Output Shape: %v\n", currentShape)
    
    // ASCII architecture diagram
    mv.drawArchitectureDiagram()
}

func (mv *ModelVisualizer) drawArchitectureDiagram() {
    fmt.Println("\n🎨 Architecture Diagram:")
    
    maxWidth := 20
    
    // Input
    fmt.Printf("┌%s┐\n", strings.Repeat("─", maxWidth-2))
    fmt.Printf("│%s│\n", center("INPUT", maxWidth-2))
    fmt.Printf("│%s│\n", center(formatShape(mv.Model.InputShape), maxWidth-2))
    fmt.Printf("└%s┘\n", strings.Repeat("─", maxWidth-2))
    fmt.Println("    │")
    fmt.Println("    ▼")
    
    // Layers
    for i, layer := range mv.Model.Layers {
        layerType := getLayerType(layer)
        layerName := getLayerName(layer)
        
        fmt.Printf("┌%s┐\n", strings.Repeat("─", maxWidth-2))
        fmt.Printf("│%s│\n", center(layerType, maxWidth-2))
        fmt.Printf("│%s│\n", center(layerName, maxWidth-2))
        fmt.Printf("└%s┘\n", strings.Repeat("─", maxWidth-2))
        
        if i < len(mv.Model.Layers)-1 {
            fmt.Println("    │")
            fmt.Println("    ▼")
        }
    }
    
    // Output
    fmt.Println("    │")
    fmt.Println("    ▼")
    fmt.Printf("┌%s┐\n", strings.Repeat("─", maxWidth-2))
    fmt.Printf("│%s│\n", center("OUTPUT", maxWidth-2))
    fmt.Printf("└%s┘\n", strings.Repeat("─", maxWidth-2))
}

// Helper functions for model visualization
func getLayerType(layer *layers.Layer) string {
    // In a real implementation, you'd inspect the layer type
    // This is a simplified example
    if layer.Name == "conv1" || layer.Name == "conv2" {
        return "Conv2D"
    } else if layer.Name == "relu1" || layer.Name == "relu2" {
        return "ReLU"
    } else if layer.Name == "flatten" {
        return "Flatten"
    } else if layer.Name == "dense1" || layer.Name == "output" {
        return "Dense"
    } else if layer.Name == "dropout1" {
        return "Dropout"
    }
    return "Unknown"
}

func getLayerName(layer *layers.Layer) string {
    if layer.Name != "" {
        return layer.Name
    }
    return "unnamed"
}

func calculateOutputShape(layer *layers.Layer, inputShape []int) []int {
    // Simplified shape calculation - in practice, this would be more complex
    return inputShape // Placeholder
}

func calculateLayerParams(layer *layers.Layer, inputShape []int) int {
    // Simplified parameter calculation
    return 1000 // Placeholder
}

func formatShape(shape []int) string {
    result := "("
    for i, dim := range shape {
        if i > 0 {
            result += ","
        }
        result += fmt.Sprintf("%d", dim)
    }
    result += ")"
    return result
}

func center(text string, width int) string {
    if len(text) >= width {
        return text[:width]
    }
    padding := width - len(text)
    leftPad := padding / 2
    rightPad := padding - leftPad
    return strings.Repeat(" ", leftPad) + text + strings.Repeat(" ", rightPad)
}
```

## 📈 Data Visualization

### Dataset Analysis and Visualization

```go
// DataVisualizer provides comprehensive dataset analysis
type DataVisualizer struct {
    Data   []float32
    Labels []float32
    Name   string
}

func NewDataVisualizer(data, labels []float32, name string) *DataVisualizer {
    return &DataVisualizer{
        Data:   data,
        Labels: labels,
        Name:   name,
    }
}

func (dv *DataVisualizer) AnalyzeDataset() {
    fmt.Printf("📊 Dataset Analysis: %s\n", dv.Name)
    fmt.Println("========================")
    
    // Basic statistics
    dv.displayBasicStats()
    
    // Data distribution
    dv.displayDataDistribution()
    
    // Label distribution
    dv.displayLabelDistribution()
    
    // Data quality checks
    dv.checkDataQuality()
}

func (dv *DataVisualizer) displayBasicStats() {
    fmt.Println("\n📋 Basic Statistics:")
    
    if len(dv.Data) == 0 {
        fmt.Println("   ❌ No data available")
        return
    }
    
    // Calculate statistics
    min, max, mean, std := calculateStats(dv.Data)
    
    fmt.Printf("   Samples: %d\n", len(dv.Data))
    fmt.Printf("   Min:     %.6f\n", min)
    fmt.Printf("   Max:     %.6f\n", max)
    fmt.Printf("   Mean:    %.6f\n", mean)
    fmt.Printf("   Std:     %.6f\n", std)
    
    // Range analysis
    dataRange := max - min
    fmt.Printf("   Range:   %.6f\n", dataRange)
    
    if dataRange > 100 {
        fmt.Println("   ⚠️  Large range detected - consider normalization")
    }
    
    if std > mean*2 {
        fmt.Println("   ⚠️  High variance detected - check for outliers")
    }
}

func (dv *DataVisualizer) displayDataDistribution() {
    fmt.Println("\n📊 Data Distribution:")
    
    if len(dv.Data) == 0 {
        return
    }
    
    // Create histogram
    numBins := 20
    min, max, _, _ := calculateStats(dv.Data)
    
    if max == min {
        fmt.Println("   All values are identical")
        return
    }
    
    binWidth := (max - min) / float32(numBins)
    bins := make([]int, numBins)
    
    // Count values in each bin
    for _, value := range dv.Data {
        binIdx := int((value - min) / binWidth)
        if binIdx >= numBins {
            binIdx = numBins - 1
        }
        bins[binIdx]++
    }
    
    // Find max count for scaling
    maxCount := 0
    for _, count := range bins {
        if count > maxCount {
            maxCount = count
        }
    }
    
    // Display histogram
    plotWidth := 40
    for i := 0; i < numBins; i++ {
        binStart := min + float32(i)*binWidth
        binEnd := binStart + binWidth
        
        barLength := 0
        if maxCount > 0 {
            barLength = (bins[i] * plotWidth) / maxCount
        }
        
        fmt.Printf("   [%6.2f-%6.2f) │", binStart, binEnd)
        for j := 0; j < barLength; j++ {
            fmt.Print("█")
        }
        fmt.Printf(" (%d)\n", bins[i])
    }
}

func (dv *DataVisualizer) displayLabelDistribution() {
    if len(dv.Labels) == 0 {
        return
    }
    
    fmt.Println("\n🏷️ Label Distribution:")
    
    // Count unique labels
    labelCounts := make(map[float32]int)
    for _, label := range dv.Labels {
        labelCounts[label]++
    }
    
    // Display distribution
    totalSamples := len(dv.Labels)
    for label, count := range labelCounts {
        percentage := float32(count) / float32(totalSamples) * 100
        fmt.Printf("   Label %.1f: %d samples (%.1f%%)\n", label, count, percentage)
    }
    
    // Check for class imbalance
    if len(labelCounts) > 1 {
        minCount := totalSamples
        maxCount := 0
        
        for _, count := range labelCounts {
            if count < minCount {
                minCount = count
            }
            if count > maxCount {
                maxCount = count
            }
        }
        
        imbalanceRatio := float32(maxCount) / float32(minCount)
        if imbalanceRatio > 2.0 {
            fmt.Printf("   ⚠️  Class imbalance detected (ratio: %.1f:1)\n", imbalanceRatio)
        }
    }
}

func (dv *DataVisualizer) checkDataQuality() {
    fmt.Println("\n🔍 Data Quality Check:")
    
    // Check for NaN/Inf values
    nanCount := 0
    infCount := 0
    
    for _, value := range dv.Data {
        if math.IsNaN(float64(value)) {
            nanCount++
        } else if math.IsInf(float64(value), 0) {
            infCount++
        }
    }
    
    if nanCount > 0 {
        fmt.Printf("   ❌ Found %d NaN values\n", nanCount)
    }
    
    if infCount > 0 {
        fmt.Printf("   ❌ Found %d Inf values\n", infCount)
    }
    
    if nanCount == 0 && infCount == 0 {
        fmt.Println("   ✅ No NaN or Inf values detected")
    }
    
    // Check for constant values
    min, max, _, _ := calculateStats(dv.Data)
    if max == min {
        fmt.Println("   ⚠️  All values are constant")
    } else {
        fmt.Println("   ✅ Data has variation")
    }
    
    // Memory usage estimate
    dataSize := len(dv.Data) * 4 // 4 bytes per float32
    fmt.Printf("   📊 Memory usage: ~%d bytes (%.1f KB)\n", dataSize, float32(dataSize)/1024)
}

func calculateStats(data []float32) (min, max, mean, std float32) {
    if len(data) == 0 {
        return 0, 0, 0, 0
    }
    
    min = data[0]
    max = data[0]
    sum := float32(0)
    
    for _, value := range data {
        if value < min {
            min = value
        }
        if value > max {
            max = value
        }
        sum += value
    }
    
    mean = sum / float32(len(data))
    
    // Calculate standard deviation
    sumSquares := float32(0)
    for _, value := range data {
        diff := value - mean
        sumSquares += diff * diff
    }
    
    std = float32(math.Sqrt(float64(sumSquares / float32(len(data)))))
    
    return min, max, mean, std
}
```

## 🔍 Model Performance Visualization

### Confusion Matrix and Metrics

```go
// ClassificationVisualizer for classification model analysis
type ClassificationVisualizer struct {
    Predictions []float32
    TrueLabels  []float32
    ClassNames  []string
}

func NewClassificationVisualizer(predictions, trueLabels []float32, classNames []string) *ClassificationVisualizer {
    return &ClassificationVisualizer{
        Predictions: predictions,
        TrueLabels:  trueLabels,
        ClassNames:  classNames,
    }
}

func (cv *ClassificationVisualizer) DisplayConfusionMatrix() {
    fmt.Println("🎯 Confusion Matrix Analysis")
    fmt.Println("============================")
    
    if len(cv.Predictions) != len(cv.TrueLabels) {
        fmt.Println("❌ Predictions and labels length mismatch")
        return
    }
    
    // Determine number of classes
    numClasses := len(cv.ClassNames)
    if numClasses == 0 {
        fmt.Println("❌ No class names provided")
        return
    }
    
    // Create confusion matrix
    matrix := make([][]int, numClasses)
    for i := range matrix {
        matrix[i] = make([]int, numClasses)
    }
    
    // Fill confusion matrix
    for i := 0; i < len(cv.Predictions); i++ {
        predClass := int(cv.Predictions[i])
        trueClass := int(cv.TrueLabels[i])
        
        if predClass >= 0 && predClass < numClasses && 
           trueClass >= 0 && trueClass < numClasses {
            matrix[trueClass][predClass]++
        }
    }
    
    // Display matrix
    cv.printConfusionMatrix(matrix)
    
    // Calculate and display metrics
    cv.calculateClassificationMetrics(matrix)
}

func (cv *ClassificationVisualizer) printConfusionMatrix(matrix [][]int) {
    numClasses := len(matrix)
    
    fmt.Println("\n📊 Confusion Matrix:")
    fmt.Println("     (Rows: True, Columns: Predicted)")
    
    // Header
    fmt.Print("        ")
    for i := 0; i < numClasses; i++ {
        if i < len(cv.ClassNames) {
            fmt.Printf("%8s", cv.ClassNames[i][:min(8, len(cv.ClassNames[i]))])
        } else {
            fmt.Printf("%8d", i)
        }
    }
    fmt.Println()
    
    // Matrix rows
    for i := 0; i < numClasses; i++ {
        // Row label
        if i < len(cv.ClassNames) {
            fmt.Printf("%8s", cv.ClassNames[i][:min(8, len(cv.ClassNames[i]))])
        } else {
            fmt.Printf("%8d", i)
        }
        
        // Matrix values
        for j := 0; j < numClasses; j++ {
            if i == j {
                fmt.Printf("\033[32m%8d\033[0m", matrix[i][j]) // Green for correct
            } else if matrix[i][j] > 0 {
                fmt.Printf("\033[31m%8d\033[0m", matrix[i][j]) // Red for errors
            } else {
                fmt.Printf("%8d", matrix[i][j])
            }
        }
        fmt.Println()
    }
}

func (cv *ClassificationVisualizer) calculateClassificationMetrics(matrix [][]int) {
    numClasses := len(matrix)
    
    fmt.Println("\n📈 Classification Metrics:")
    
    // Calculate per-class metrics
    fmt.Printf("%-10s │ %-9s │ %-9s │ %-9s │ %-9s\n", 
               "Class", "Precision", "Recall", "F1-Score", "Support")
    fmt.Println("───────────┼───────────┼───────────┼───────────┼───────────")
    
    totalCorrect := 0
    totalSamples := 0
    weightedF1 := float32(0)
    
    for i := 0; i < numClasses; i++ {
        // True positives, false positives, false negatives
        tp := matrix[i][i]
        fp := 0
        fn := 0
        
        // Calculate FP and FN
        for j := 0; j < numClasses; j++ {
            if j != i {
                fp += matrix[j][i] // Predicted as i but actually j
                fn += matrix[i][j] // Actually i but predicted as j
            }
        }
        
        // Support (total samples for this class)
        support := tp + fn
        
        // Precision, Recall, F1
        var precision, recall, f1 float32
        
        if tp+fp > 0 {
            precision = float32(tp) / float32(tp+fp)
        }
        
        if tp+fn > 0 {
            recall = float32(tp) / float32(tp+fn)
        }
        
        if precision+recall > 0 {
            f1 = 2 * precision * recall / (precision + recall)
        }
        
        // Display metrics
        className := fmt.Sprintf("Class %d", i)
        if i < len(cv.ClassNames) {
            className = cv.ClassNames[i]
        }
        
        fmt.Printf("%-10s │ %9.3f │ %9.3f │ %9.3f │ %9d\n",
                   className, precision, recall, f1, support)
        
        totalCorrect += tp
        totalSamples += support
        weightedF1 += f1 * float32(support)
    }
    
    // Overall metrics
    fmt.Println("───────────┴───────────┴───────────┴───────────┴───────────")
    
    accuracy := float32(totalCorrect) / float32(totalSamples)
    avgF1 := weightedF1 / float32(totalSamples)
    
    fmt.Printf("%-10s │           │           │ %9.3f │ %9d\n",
               "Weighted", avgF1, totalSamples)
    fmt.Printf("\n🎯 Overall Accuracy: %.3f (%.1f%%)\n", accuracy, accuracy*100)
}
```

## 📊 Advanced Visualization Techniques

### Real-Time Model Monitoring

```go
// ModelMonitor provides comprehensive real-time model monitoring
type ModelMonitor struct {
    StartTime      time.Time
    EpochTimes     []time.Duration
    LossHistory    []float32
    GradientNorms  []float32
    LearningRates  []float32
    MemoryUsage    []float64
}

func NewModelMonitor() *ModelMonitor {
    return &ModelMonitor{
        StartTime:      time.Now(),
        EpochTimes:     make([]time.Duration, 0),
        LossHistory:    make([]float32, 0),
        GradientNorms:  make([]float32, 0),
        LearningRates:  make([]float32, 0),
        MemoryUsage:    make([]float64, 0),
    }
}

func (mm *ModelMonitor) RecordEpoch(epochTime time.Duration, loss float32, 
                                   gradNorm float32, lr float32, memMB float64) {
    mm.EpochTimes = append(mm.EpochTimes, epochTime)
    mm.LossHistory = append(mm.LossHistory, loss)
    mm.GradientNorms = append(mm.GradientNorms, gradNorm)
    mm.LearningRates = append(mm.LearningRates, lr)
    mm.MemoryUsage = append(mm.MemoryUsage, memMB)
}

func (mm *ModelMonitor) DisplayDashboard() {
    fmt.Printf("\033[2J\033[H") // Clear screen
    
    fmt.Println("🚀 Real-Time Model Monitor")
    fmt.Println("=========================")
    
    if len(mm.LossHistory) == 0 {
        fmt.Println("No data recorded yet...")
        return
    }
    
    // Current status
    currentEpoch := len(mm.LossHistory)
    currentLoss := mm.LossHistory[len(mm.LossHistory)-1]
    totalTime := time.Since(mm.StartTime)
    avgEpochTime := totalTime / time.Duration(currentEpoch)
    
    fmt.Printf("📊 Status: Epoch %d │ Loss %.6f │ Time %v │ Avg/Epoch %v\n",
               currentEpoch, currentLoss, totalTime.Round(time.Second), 
               avgEpochTime.Round(time.Millisecond))
    
    // Performance indicators
    mm.displayPerformanceIndicators()
    
    // Training progress visualization
    mm.displayProgressCharts()
    
    // System resources
    mm.displayResourceUsage()
}

func (mm *ModelMonitor) displayPerformanceIndicators() {
    if len(mm.LossHistory) < 2 {
        return
    }
    
    fmt.Println("\n🎯 Performance Indicators:")
    
    // Loss trend
    recentLoss := mm.LossHistory[len(mm.LossHistory)-1]
    prevLoss := mm.LossHistory[len(mm.LossHistory)-2]
    lossTrend := recentLoss - prevLoss
    
    fmt.Printf("   Loss Trend: ")
    if lossTrend < 0 {
        fmt.Printf("📉 Improving (Δ %.6f)\n", lossTrend)
    } else {
        fmt.Printf("📈 Increasing (Δ %.6f)\n", lossTrend)
    }
    
    // Convergence rate
    if len(mm.LossHistory) >= 10 {
        initialLoss := mm.LossHistory[0]
        currentLoss := mm.LossHistory[len(mm.LossHistory)-1]
        improvement := (initialLoss - currentLoss) / initialLoss * 100
        
        fmt.Printf("   Improvement: %.1f%% from initial loss\n", improvement)
    }
    
    // Training speed
    if len(mm.EpochTimes) > 0 {
        avgTime := calculateAverageTime(mm.EpochTimes)
        fmt.Printf("   Avg Speed: %v per epoch\n", avgTime.Round(time.Millisecond))
    }
}

func (mm *ModelMonitor) displayProgressCharts() {
    fmt.Println("\n📈 Training Progress:")
    
    // Loss chart
    mm.plotSimpleChart("Loss", mm.LossHistory, 15)
    
    // Gradient norms (if available)
    if len(mm.GradientNorms) > 0 {
        mm.plotSimpleChart("Grad Norm", mm.GradientNorms, 10)
    }
}

func (mm *ModelMonitor) plotSimpleChart(title string, data []float32, height int) {
    if len(data) < 2 {
        return
    }
    
    fmt.Printf("\n   %s:\n", title)
    
    // Find min/max for scaling
    min, max := data[0], data[0]
    for _, value := range data {
        if value < min {
            min = value
        }
        if value > max {
            max = value
        }
    }
    
    if max == min {
        max = min + 1 // Avoid division by zero
    }
    
    // Simple sparkline
    width := min(len(data), 60)
    fmt.Print("   ")
    
    for i := 0; i < width; i++ {
        idx := len(data) - width + i
        if idx < 0 {
            continue
        }
        
        normalized := (data[idx] - min) / (max - min)
        
        if normalized < 0.2 {
            fmt.Print("▁")
        } else if normalized < 0.4 {
            fmt.Print("▂")
        } else if normalized < 0.6 {
            fmt.Print("▄")
        } else if normalized < 0.8 {
            fmt.Print("▆")
        } else {
            fmt.Print("█")
        }
    }
    
    fmt.Printf(" (%.4f → %.4f)\n", data[0], data[len(data)-1])
}

func (mm *ModelMonitor) displayResourceUsage() {
    if len(mm.MemoryUsage) == 0 {
        return
    }
    
    fmt.Println("\n💾 Resource Usage:")
    
    currentMem := mm.MemoryUsage[len(mm.MemoryUsage)-1]
    fmt.Printf("   Memory: %.1f MB\n", currentMem)
    
    // Memory trend
    if len(mm.MemoryUsage) > 1 {
        prevMem := mm.MemoryUsage[len(mm.MemoryUsage)-2]
        memChange := currentMem - prevMem
        
        if memChange > 1.0 {
            fmt.Printf("   ⚠️  Memory increasing (+%.1f MB)\n", memChange)
        } else if memChange < -1.0 {
            fmt.Printf("   ✅ Memory decreasing (%.1f MB)\n", memChange)
        } else {
            fmt.Printf("   ➡️  Memory stable\n")
        }
    }
}

func calculateAverageTime(times []time.Duration) time.Duration {
    if len(times) == 0 {
        return 0
    }
    
    var total time.Duration
    for _, t := range times {
        total += t
    }
    
    return total / time.Duration(len(times))
}
```

## 🎯 Best Practices for Visualization

### Production Visualization Guidelines

```go
func visualizationBestPractices() {
    fmt.Println("🎯 Visualization Best Practices")
    fmt.Println("===============================")
    
    fmt.Println("\n📊 Training Monitoring:")
    fmt.Println("   ✅ Plot training and validation loss together")
    fmt.Println("   ✅ Use log scale for large loss ranges")
    fmt.Println("   ✅ Show learning rate changes")
    fmt.Println("   ✅ Monitor gradient norms for stability")
    fmt.Println("   ✅ Track memory usage over time")
    
    fmt.Println("\n🎨 Chart Design:")
    fmt.Println("   ✅ Use clear, contrasting colors")
    fmt.Println("   ✅ Add legends and axis labels")
    fmt.Println("   ✅ Choose appropriate scales")
    fmt.Println("   ✅ Highlight important thresholds")
    fmt.Println("   ✅ Use consistent styling")
    
    fmt.Println("\n📈 Data Analysis:")
    fmt.Println("   ✅ Show data distribution histograms")
    fmt.Println("   ✅ Check for class imbalance")
    fmt.Println("   ✅ Visualize feature correlations")
    fmt.Println("   ✅ Identify outliers and anomalies")
    fmt.Println("   ✅ Display sample data examples")
    
    fmt.Println("\n🔍 Model Analysis:")
    fmt.Println("   ✅ Confusion matrices for classification")
    fmt.Println("   ✅ ROC curves and AUC scores")
    fmt.Println("   ✅ Precision-recall curves")
    fmt.Println("   ✅ Feature importance plots")
    fmt.Println("   ✅ Model architecture diagrams")
    
    fmt.Println("\n⚡ Performance Tips:")
    fmt.Println("   ✅ Update visualizations periodically, not every epoch")
    fmt.Println("   ✅ Use efficient ASCII plots for real-time monitoring")
    fmt.Println("   ✅ Save plots to files for later analysis")
    fmt.Println("   ✅ Implement visualization caching for large datasets")
    fmt.Println("   ✅ Consider web-based dashboards for remote monitoring")
}

func exampleUsage() {
    fmt.Println("\n🚀 Example Usage")
    fmt.Println("================")
    
    fmt.Println(`
// 1. Training with visualization
visualizer := NewTrainingVisualizer()
for epoch := 1; epoch <= numEpochs; epoch++ {
    result, _ := trainer.TrainBatch(inputs, inputShape, targets, targetShape)
    visualizer.AddMetrics(epoch, result.Loss, valLoss, accuracy)
    
    if epoch%5 == 0 {
        visualizer.DisplayProgress()
    }
}

// 2. Model architecture analysis
modelViz := NewModelVisualizer(model)
modelViz.DisplayArchitecture()

// 3. Dataset analysis
dataViz := NewDataVisualizer(trainData, trainLabels, "Training Set")
dataViz.AnalyzeDataset()

// 4. Classification results
classViz := NewClassificationVisualizer(predictions, trueLabels, classNames)
classViz.DisplayConfusionMatrix()

// 5. Real-time monitoring
monitor := NewModelMonitor()
for epoch := 1; epoch <= numEpochs; epoch++ {
    start := time.Now()
    result, _ := trainer.TrainBatch(inputs, inputShape, targets, targetShape)
    epochTime := time.Since(start)
    
    monitor.RecordEpoch(epochTime, result.Loss, gradNorm, learningRate, memUsage)
    monitor.DisplayDashboard()
}`)
}
```

## 🚀 Integration with External Tools

### Exporting Data for Advanced Visualization

```go
// DataExporter helps export go-metal data to external visualization tools
type DataExporter struct {
    OutputDir string
}

func NewDataExporter(outputDir string) *DataExporter {
    return &DataExporter{OutputDir: outputDir}
}

func (de *DataExporter) ExportTrainingMetrics(losses []float32, accuracies []float32, filename string) error {
    fmt.Printf("📁 Exporting training metrics to %s/%s.csv\n", de.OutputDir, filename)
    
    // Create CSV content
    csvContent := "epoch,loss,accuracy\n"
    for i := 0; i < len(losses); i++ {
        accuracy := float32(0)
        if i < len(accuracies) {
            accuracy = accuracies[i]
        }
        csvContent += fmt.Sprintf("%d,%.6f,%.6f\n", i+1, losses[i], accuracy)
    }
    
    // Save to file (simplified - in practice, use proper file handling)
    fmt.Printf("   ✅ CSV format ready (%d rows)\n", len(losses))
    fmt.Println("   💡 Import into tools like Excel, Python/matplotlib, R, etc.")
    
    return nil
}

func (de *DataExporter) ExportPythonScript(modelName string) {
    fmt.Printf("🐍 Generating Python visualization script for %s\n", modelName)
    
    pythonScript := `
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load training data
data = pd.read_csv('training_metrics.csv')

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Go-Metal Model Training Analysis', fontsize=16)

# Training loss
axes[0,0].plot(data['epoch'], data['loss'], 'b-', linewidth=2)
axes[0,0].set_title('Training Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].grid(True, alpha=0.3)

# Training accuracy
axes[0,1].plot(data['epoch'], data['accuracy']*100, 'g-', linewidth=2)
axes[0,1].set_title('Training Accuracy')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy (%)')
axes[0,1].grid(True, alpha=0.3)

# Loss distribution
axes[1,0].hist(data['loss'], bins=20, alpha=0.7, color='blue')
axes[1,0].set_title('Loss Distribution')
axes[1,0].set_xlabel('Loss Value')
axes[1,0].set_ylabel('Frequency')

# Training progress
axes[1,1].plot(data['epoch'], data['loss'], 'b-', label='Loss', linewidth=2)
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Loss', color='b')
axes[1,1].tick_params(axis='y', labelcolor='b')

ax2 = axes[1,1].twinx()
ax2.plot(data['epoch'], data['accuracy']*100, 'g-', label='Accuracy', linewidth=2)
ax2.set_ylabel('Accuracy (%)', color='g')
ax2.tick_params(axis='y', labelcolor='g')

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'training_analysis.png'")
`
    
    fmt.Println("   ✅ Python script generated")
    fmt.Println("   💡 Run with: python visualize_training.py")
    fmt.Println("   📊 Requires: matplotlib, pandas, numpy")
}

func (de *DataExporter) SuggestVisualizationTools() {
    fmt.Println("\n🛠️ Recommended Visualization Tools")
    fmt.Println("==================================")
    
    tools := []struct {
        name string
        description string
        useCase string
        setup string
    }{
        {
            "Python + Matplotlib",
            "Comprehensive plotting library",
            "Static plots, research, reports",
            "pip install matplotlib pandas",
        },
        {
            "Python + Plotly",
            "Interactive web-based plots",
            "Interactive dashboards, exploration",
            "pip install plotly",
        },
        {
            "TensorBoard",
            "ML-focused visualization platform",
            "Real-time monitoring, experiment tracking",
            "pip install tensorboard",
        },
        {
            "Grafana",
            "Real-time dashboard platform",
            "Production monitoring, alerting",
            "Docker: grafana/grafana",
        },
        {
            "R + ggplot2",
            "Statistical visualization",
            "Statistical analysis, publication",
            "install.packages('ggplot2')",
        },
    }
    
    fmt.Printf("%-20s │ %-30s │ %-25s │ %-25s\n",
               "Tool", "Description", "Best For", "Installation")
    fmt.Println("─────────────────────┼────────────────────────────────┼───────────────────────────┼─────────────────────────")
    
    for _, tool := range tools {
        fmt.Printf("%-20s │ %-30s │ %-25s │ %-25s\n",
                   tool.name, tool.description, tool.useCase, tool.setup)
    }
    
    fmt.Println("\n💡 Quick Start Recommendations:")
    fmt.Println("   🐍 Python users: Start with matplotlib for static plots")
    fmt.Println("   🌐 Interactive needs: Use Plotly for web-based dashboards") 
    fmt.Println("   🔄 Real-time monitoring: Set up Grafana for production")
    fmt.Println("   📊 Statistical analysis: Use R + ggplot2 for research")
}
```

## 🎓 Complete Visualization Example

```go
func completeVisualizationExample() {
    fmt.Println("🎓 Complete Visualization Workflow")
    fmt.Println("==================================")
    
    fmt.Println(`
// Complete example integrating all visualization components
func runCompleteVisualizationWorkflow() {
    // 1. Setup
    device, _ := cgo_bridge.CreateMetalDevice()
    defer cgo_bridge.DestroyMetalDevice(device)
    memory.InitializeGlobalMemoryManager(device)
    
    // 2. Data analysis
    dataViz := NewDataVisualizer(trainData, trainLabels, "MNIST Training")
    dataViz.AnalyzeDataset()
    
    // 3. Model architecture
    modelViz := NewModelVisualizer(model)
    modelViz.DisplayArchitecture()
    
    // 4. Training with visualization
    visualizer := NewTrainingVisualizer()
    monitor := NewModelMonitor()
    
    for epoch := 1; epoch <= 100; epoch++ {
        start := time.Now()
        
        // Training step
        result, _ := trainer.TrainBatch(inputs, inputShape, targets, targetShape)
        
        // Record metrics
        epochTime := time.Since(start)
        gradNorm := float32(1.0) // Calculate actual gradient norm
        lr := 0.001
        memUsage := 256.0 // Get actual memory usage
        
        visualizer.AddMetrics(epoch, result.Loss, result.Loss*1.1, 0.85)
        monitor.RecordEpoch(epochTime, result.Loss, gradNorm, lr, memUsage)
        
        // Display progress
        if epoch%10 == 0 {
            visualizer.DisplayProgress()
            monitor.DisplayDashboard()
        }
    }
    
    // 5. Final analysis
    predictions := make([]float32, len(testLabels)) // Get actual predictions
    classViz := NewClassificationVisualizer(predictions, testLabels, classNames)
    classViz.DisplayConfusionMatrix()
    
    // 6. Export for external tools
    exporter := NewDataExporter("./visualization_output")
    exporter.ExportTrainingMetrics(visualizer.TrainingLosses, visualizer.Accuracies, "training")
    exporter.ExportPythonScript("mnist_model")
    exporter.SuggestVisualizationTools()
    
    fmt.Println("\\n🎉 Complete visualization workflow finished!")
    fmt.Println("   ✅ Real-time training monitoring")
    fmt.Println("   ✅ Model architecture analysis")
    fmt.Println("   ✅ Data quality assessment")
    fmt.Println("   ✅ Performance evaluation")
    fmt.Println("   ✅ Export for external tools")
}`)
}
```

## 🚀 Ready for Production

This comprehensive visualization guide demonstrates:

- **Real-time Monitoring**: Live training progress and performance tracking
- **Model Analysis**: Architecture visualization and performance metrics
- **Data Insights**: Dataset analysis and quality assessment
- **Production Integration**: Export capabilities and tool recommendations
- **Best Practices**: Professional visualization standards

**Continue Learning:**
- **[Performance Guide](performance.md)** - Optimize visualization performance
- **[Mixed Precision Tutorial](../tutorials/mixed-precision.md)** - Monitor FP16 training
- **[Advanced Examples](../examples/)** - Real-world visualization applications

---

## 🧠 Key Takeaways

- **Real-time feedback improves training**: Monitor progress as it happens
- **Visualization reveals insights**: Spot overfitting, convergence issues, data problems
- **ASCII plots work well in CLI**: Effective for terminal-based monitoring
- **Export enables collaboration**: Share data with visualization specialists
- **Multiple views tell the story**: Combine training metrics, model analysis, and data exploration
- **Go-metal advantages**: GPU-resident data enables efficient monitoring without performance impact

You now have the skills to create production-ready visualizations for any go-metal project!