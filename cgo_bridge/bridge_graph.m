#import "bridge_graph.h"
#import "bridge_optimizer.h"

// Build dynamic MPSGraph from layer specifications
BOOL buildDynamicGraphFromLayers(training_engine_t* engine,
                                  layer_spec_c_t* layers,
                                  int numLayers,
                                  int* inputShape,
                                  int inputShapeLen) {
    
    if (!engine || !layers || numLayers <= 0) {
        return NO;
    }
    
    @try {
        // Create input placeholder with dynamic batch size support
        NSMutableArray<NSNumber*>* inputShapeNS = [[NSMutableArray alloc] init];
        for (int i = 0; i < inputShapeLen; i++) {
            if (i == 0) {
                // Use -1 for batch dimension to support dynamic batch sizes
                [inputShapeNS addObject:@(-1)];
            } else {
                // Keep other dimensions fixed (channels, height, width)
                [inputShapeNS addObject:@(inputShape[i])];
            }
        }
        
        MPSGraphTensor* currentTensor = [engine->graph placeholderWithShape:inputShapeNS
                                                                   dataType:MPSDataTypeFloat32
                                                                       name:@"input"];
        engine->inputTensor = currentTensor;
        
        // Create a single array to track ALL parameters in the correct order (weight, bias, weight, bias...)
        // This matches the Go parameter tensor creation order exactly
        NSMutableArray* allParameterPlaceholders = [[NSMutableArray alloc] init];
        
        // Process each layer sequentially
        for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            layer_spec_c_t* layer = &layers[layerIdx];
            
            switch (layer->layer_type) {
                case 0: // Dense
                    // Check if we need to flatten the input (from 4D conv output to 2D dense input)
                    if (currentTensor.shape.count == 4) {
                        // Flatten [batch, channels, height, width] to [batch, channels*height*width]
                        // Support dynamic batch size by preserving the batch dimension
                        NSArray<NSNumber*>* currentShape = currentTensor.shape;
                        int channels = [currentShape[1] intValue];
                        int height = [currentShape[2] intValue];
                        int width = [currentShape[3] intValue];
                        int flattenedSize = channels * height * width;
                        
                        // Use -1 for batch dimension to support dynamic batch sizes
                        NSArray<NSNumber*>* flattenShape = @[@(-1), @(flattenedSize)];
                        currentTensor = [engine->graph reshapeTensor:currentTensor
                                                           withShape:flattenShape
                                                                name:[NSString stringWithFormat:@"flatten_before_dense_%d", layerIdx]];
                        // NSLog(@"✅ Flattened tensor from %@ to %@ for Dense layer %d", 
                        //       currentShape, flattenShape, layerIdx);
                    }
                    
                    currentTensor = addDenseLayerToGraph(engine->graph,
                                                        currentTensor,
                                                        layer,
                                                        layerIdx,
                                                        allParameterPlaceholders);
                    break;
                    
                case 1: // Conv2D
                    currentTensor = addConv2DLayerToGraph(engine->graph,
                                                         currentTensor,
                                                         layer,
                                                         layerIdx,
                                                         allParameterPlaceholders);
                    break;
                    
                case 2: // ReLU
                    currentTensor = [engine->graph reLUWithTensor:currentTensor
                                                             name:[NSString stringWithFormat:@"relu_%d", layerIdx]];
                    break;
                    
                case 3: // Softmax
                    {
                        int axis = layer->param_int_count > 0 ? layer->param_int[0] : -1;
                        currentTensor = [engine->graph softMaxWithTensor:currentTensor
                                                                    axis:axis
                                                                    name:[NSString stringWithFormat:@"softmax_%d", layerIdx]];
                    }
                    break;
                    
                case 5: // Dropout
                    {
                        // Dropout parameters: rate (float), training (int)
                        float rate = layer->param_float_count > 0 ? layer->param_float[0] : 0.5f;
                        int training = layer->param_int_count > 0 ? layer->param_int[0] : 1;
                        
                        if (training) {
                            // Training mode: Apply dropout with MPSGraph
                            currentTensor = [engine->graph dropoutTensor:currentTensor
                                                                    rate:rate
                                                                    name:[NSString stringWithFormat:@"dropout_%d", layerIdx]];
                        }
                        // Inference mode: Pass through unchanged (MPSGraph handles this automatically)
                    }
                    break;
                    
                case 6: // BatchNorm
                    {
                        currentTensor = addBatchNormLayerToGraph(engine->graph,
                                                               currentTensor,
                                                               layer,
                                                               layerIdx,
                                                               allParameterPlaceholders,
                                                               engine);
                    }
                    break;
                    
                case 7: // LeakyReLU
                    {
                        // Leaky ReLU parameters: negative_slope (float)
                        float negativeSlope = layer->param_float_count > 0 ? layer->param_float[0] : 0.01f;
                        
                        currentTensor = [engine->graph leakyReLUWithTensor:currentTensor
                                                                     alpha:negativeSlope
                                                                      name:[NSString stringWithFormat:@"leaky_relu_%d", layerIdx]];
                    }
                    break;
                    
                case 8: // ELU
                    {
                        // ELU parameters: alpha (float)
                        float alpha = layer->param_float_count > 0 ? layer->param_float[0] : 1.0f;
                        
                        // ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
                        // Implementation using MPSGraph operations since there's no built-in ELU
                        
                        MPSGraphTensor* zeroTensor = [engine->graph constantWithScalar:0.0f 
                                                                              dataType:MPSDataTypeFloat32];
                        MPSGraphTensor* alphaTensor = [engine->graph constantWithScalar:alpha 
                                                                               dataType:MPSDataTypeFloat32];
                        MPSGraphTensor* oneTensor = [engine->graph constantWithScalar:1.0f 
                                                                             dataType:MPSDataTypeFloat32];
                        
                        
                        // Use differentiable approach: ELU(x) = max(0, x) + alpha * (exp(min(0, x)) - 1)
                        // This avoids selectWithPredicateTensor which causes gradient computation issues
                        
                        // Positive part: max(0, x)
                        MPSGraphTensor* positivePart = [engine->graph maximumWithPrimaryTensor:currentTensor
                                                                              secondaryTensor:zeroTensor
                                                                                         name:[NSString stringWithFormat:@"elu_positive_%d", layerIdx]];
                        
                        // Negative part: min(0, x) 
                        MPSGraphTensor* negativePart = [engine->graph minimumWithPrimaryTensor:currentTensor
                                                                              secondaryTensor:zeroTensor
                                                                                         name:[NSString stringWithFormat:@"elu_negative_input_%d", layerIdx]];
                        
                        // For negative inputs: alpha * (exp(min(0, x)) - 1)
                        MPSGraphTensor* expNegative = [engine->graph exponentWithTensor:negativePart
                                                                                   name:[NSString stringWithFormat:@"elu_exp_negative_%d", layerIdx]];
                        MPSGraphTensor* expMinusOneNeg = [engine->graph subtractionWithPrimaryTensor:expNegative
                                                                                     secondaryTensor:oneTensor
                                                                                                name:[NSString stringWithFormat:@"elu_exp_minus_one_%d", layerIdx]];
                        MPSGraphTensor* scaledNegative = [engine->graph multiplicationWithPrimaryTensor:alphaTensor
                                                                                       secondaryTensor:expMinusOneNeg
                                                                                                  name:[NSString stringWithFormat:@"elu_scaled_negative_%d", layerIdx]];
                        
                        // Combine: max(0, x) + alpha * (exp(min(0, x)) - 1)
                        currentTensor = [engine->graph additionWithPrimaryTensor:positivePart
                                                                  secondaryTensor:scaledNegative
                                                                             name:[NSString stringWithFormat:@"elu_%d", layerIdx]];
                    }
                    break;
                    
                default:
                    NSLog(@"Unsupported layer type: %d", layer->layer_type);
                    return NO;
            }
            
            if (!currentTensor) {
                NSLog(@"Failed to create layer %d (type: %d)", layerIdx, layer->layer_type);
                return NO;
            }
        }
        
        // Create label placeholder for loss computation with dynamic batch size
        // Determine output classes from the last Dense layer in the model
        int numClasses = 2; // Default for binary classification
        
        // Find the last Dense layer to get the output size
        for (int i = numLayers - 1; i >= 0; i--) {
            layer_spec_c_t* layer = &layers[i];
            if (layer->layer_type == 0 && layer->param_int_count >= 2) { // Dense layer
                numClasses = layer->param_int[1]; // output_size
                // NSLog(@"🔍 Found output classes from Dense layer %d: %d classes", i, numClasses);
                break;
            }
        }
        
        // Labels placeholder [batch_size, num_classes] with dynamic batch size support
        // Using -1 for the batch dimension allows MPSGraph to handle variable batch sizes
        NSArray<NSNumber*>* labelShape = @[@(-1), @(numClasses)];
        // NSLog(@"🔍 Creating label placeholder with shape: %@", labelShape);
        MPSGraphTensor* labelTensor = [engine->graph placeholderWithShape:labelShape
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"labels"];
        engine->labelTensor = labelTensor;
        
        // Handle predictions based on problem type
        MPSGraphTensor* predictionsTensor;
        
        if (engine->config.problem_type == 0) { // Classification
            // Apply softmax to get predictions (multi-class classification)
            predictionsTensor = [engine->graph softMaxWithTensor:currentTensor
                                                            axis:-1
                                                            name:@"predictions"];
        } else { // Regression
            // For regression, predictions are raw outputs (no activation)
            predictionsTensor = currentTensor;
        }
        
        // Store predictions tensor for inference use
        engine->predictionsTensor = predictionsTensor;
        
        // Calculate loss based on problem type and loss function
        MPSGraphTensor* actualLoss;
        
        switch (engine->config.problem_type) {
            case 0: // Classification
                switch (engine->config.loss_function) {
                    case 0: // CrossEntropy
                        actualLoss = [engine->graph softMaxCrossEntropyWithSourceTensor:currentTensor
                                                                           labelsTensor:labelTensor
                                                                                   axis:-1
                                                                         reductionType:MPSGraphLossReductionTypeMean
                                                                                  name:@"cross_entropy_loss"];
                        break;
                    case 1: // SparseCrossEntropy
                        {
                            // For sparse cross-entropy, labels are class indices (e.g., [0, 1, 2])
                            // Use MPSGraph's built-in sparse cross-entropy functionality
                            
                            // Apply softmax to get probabilities
                            MPSGraphTensor* probabilities = [engine->graph softMaxWithTensor:currentTensor
                                                                                        axis:-1
                                                                                        name:@"softmax_probs"];
                            
                            // Apply log to get log probabilities for numerical stability
                            MPSGraphTensor* logProbs = [engine->graph logarithmWithTensor:probabilities
                                                                                     name:@"log_probs"];
                            
                            // Convert labels to int32 indices if needed
                            MPSGraphTensor* labelIndices = [engine->graph castTensor:labelTensor
                                                                              toType:MPSDataTypeInt32
                                                                                name:@"label_indices_int32"];
                            
                            // Use MPSGraph's gatherAlongAxis to select the correct log probabilities
                            // This is the correct way to implement sparse cross-entropy in MPSGraph
                            MPSGraphTensor* selectedLogProbs = [engine->graph gatherAlongAxis:1
                                                                              withUpdatesTensor:logProbs
                                                                                 indicesTensor:labelIndices
                                                                                          name:@"selected_log_probs"];
                            
                            // Compute negative log likelihood: -selected_log_probs
                            MPSGraphTensor* negLogLikelihood = [engine->graph negativeWithTensor:selectedLogProbs
                                                                                           name:@"neg_log_likelihood"];
                            
                            // Mean over batch using MPSGraph reduction (supports dynamic batch size)
                            // Sum over batch dimension, then MPSGraph will handle the division automatically
                            actualLoss = [engine->graph reductionSumWithTensor:negLogLikelihood
                                                                          axes:@[@0]  // Reduce over batch dimension
                                                                          name:@"sparse_cross_entropy_sum"];
                            
                            // Get the shape tensor to compute batch size dynamically
                            MPSGraphTensor* shapeTensor = [engine->graph shapeOfTensor:labelTensor name:@"label_shape"];
                            MPSGraphTensor* batchSizeTensor = [engine->graph sliceTensor:shapeTensor
                                                                               dimension:0
                                                                                   start:0
                                                                                  length:1
                                                                                    name:@"batch_size"];
                            MPSGraphTensor* batchSizeFloat = [engine->graph castTensor:batchSizeTensor
                                                                                toType:MPSDataTypeFloat32
                                                                                  name:@"batch_size_float"];
                            
                            actualLoss = [engine->graph divisionWithPrimaryTensor:actualLoss
                                                                  secondaryTensor:batchSizeFloat
                                                                             name:@"sparse_cross_entropy_loss"];
                        }
                        break;
                    case 2: // BinaryCrossEntropy
                        {
                            // Binary cross-entropy for binary classification
                            // Assumes labels are probabilities [0.0, 1.0] and predictions are probabilities (sigmoid output)
                            
                            // Ensure predictions are in valid range [epsilon, 1-epsilon] for numerical stability
                            float epsilon = 1e-7f;
                            MPSGraphTensor* epsilonTensor = [engine->graph constantWithScalar:epsilon
                                                                                    dataType:MPSDataTypeFloat32];
                            MPSGraphTensor* oneMinusEpsilon = [engine->graph constantWithScalar:(1.0f - epsilon)
                                                                                        dataType:MPSDataTypeFloat32];
                            
                            // Clamp predictions to [epsilon, 1-epsilon]
                            MPSGraphTensor* clampedPreds = [engine->graph clampWithTensor:currentTensor
                                                                            minValueTensor:epsilonTensor
                                                                            maxValueTensor:oneMinusEpsilon
                                                                                      name:@"clamped_predictions"];
                            
                            // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
                            MPSGraphTensor* logPreds = [engine->graph logarithmWithTensor:clampedPreds
                                                                                     name:@"log_predictions"];
                            
                            MPSGraphTensor* oneMinusPreds = [engine->graph subtractionWithPrimaryTensor:oneMinusEpsilon
                                                                                        secondaryTensor:clampedPreds
                                                                                                   name:@"one_minus_predictions"];
                            MPSGraphTensor* logOneMinusPreds = [engine->graph logarithmWithTensor:oneMinusPreds
                                                                                             name:@"log_one_minus_predictions"];
                            
                            MPSGraphTensor* oneMinusLabels = [engine->graph subtractionWithPrimaryTensor:oneMinusEpsilon
                                                                                         secondaryTensor:labelTensor
                                                                                                    name:@"one_minus_labels"];
                            
                            // y * log(p)
                            MPSGraphTensor* firstTerm = [engine->graph multiplicationWithPrimaryTensor:labelTensor
                                                                                       secondaryTensor:logPreds
                                                                                                  name:@"first_term"];
                            
                            // (1-y) * log(1-p)
                            MPSGraphTensor* secondTerm = [engine->graph multiplicationWithPrimaryTensor:oneMinusLabels
                                                                                        secondaryTensor:logOneMinusPreds
                                                                                                   name:@"second_term"];
                            
                            // BCE = -[y*log(p) + (1-y)*log(1-p)]
                            MPSGraphTensor* bceSum = [engine->graph additionWithPrimaryTensor:firstTerm
                                                                             secondaryTensor:secondTerm
                                                                                        name:@"bce_sum"];
                            MPSGraphTensor* bceNegative = [engine->graph negativeWithTensor:bceSum
                                                                                       name:@"bce_negative"];
                            
                            // Mean over batch using MPSGraph reduction (supports dynamic batch size)
                            // Sum over batch dimension, then divide by dynamic batch size
                            MPSGraphTensor* sumLoss = [engine->graph reductionSumWithTensor:bceNegative
                                                                                       axes:@[@0]  // Reduce over batch dimension
                                                                                       name:@"bce_sum"];
                            
                            // Get the shape tensor to compute batch size dynamically
                            MPSGraphTensor* shapeTensor = [engine->graph shapeOfTensor:labelTensor name:@"label_shape_bce"];
                            MPSGraphTensor* batchSizeTensor = [engine->graph sliceTensor:shapeTensor
                                                                               dimension:0
                                                                                   start:0
                                                                                  length:1
                                                                                    name:@"batch_size_bce"];
                            MPSGraphTensor* batchSizeFloat = [engine->graph castTensor:batchSizeTensor
                                                                                toType:MPSDataTypeFloat32
                                                                                  name:@"batch_size_float_bce"];
                            
                            actualLoss = [engine->graph divisionWithPrimaryTensor:sumLoss
                                                                  secondaryTensor:batchSizeFloat
                                                                             name:@"binary_cross_entropy_loss"];
                        }
                        break;
                    case 3: // BCEWithLogits
                        {
                            // Binary cross-entropy with logits (more numerically stable)
                            // Assumes labels are probabilities [0.0, 1.0] and predictions are raw logits
                            
                            // Use log-sum-exp trick for numerical stability
                            // BCE with logits: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                            // where x = logits, z = labels
                            
                            MPSGraphTensor* zeroTensor = [engine->graph constantWithScalar:0.0f
                                                                                 dataType:MPSDataTypeFloat32];
                            
                            // max(x, 0)
                            MPSGraphTensor* maxTerm = [engine->graph maximumWithPrimaryTensor:currentTensor
                                                                               secondaryTensor:zeroTensor
                                                                                          name:@"max_logits_zero"];
                            
                            // x * z
                            MPSGraphTensor* logitsLabels = [engine->graph multiplicationWithPrimaryTensor:currentTensor
                                                                                          secondaryTensor:labelTensor
                                                                                                     name:@"logits_times_labels"];
                            
                            // abs(x)
                            MPSGraphTensor* absLogits = [engine->graph absoluteWithTensor:currentTensor
                                                                                    name:@"abs_logits"];
                            
                            // -abs(x)
                            MPSGraphTensor* negAbsLogits = [engine->graph negativeWithTensor:absLogits
                                                                                       name:@"neg_abs_logits"];
                            
                            // exp(-abs(x))
                            MPSGraphTensor* expNegAbs = [engine->graph exponentWithTensor:negAbsLogits
                                                                                     name:@"exp_neg_abs"];
                            
                            // 1 + exp(-abs(x))
                            MPSGraphTensor* oneTensor = [engine->graph constantWithScalar:1.0f
                                                                                dataType:MPSDataTypeFloat32];
                            MPSGraphTensor* onePlusExp = [engine->graph additionWithPrimaryTensor:oneTensor
                                                                                  secondaryTensor:expNegAbs
                                                                                             name:@"one_plus_exp"];
                            
                            // log(1 + exp(-abs(x)))
                            MPSGraphTensor* logOnePlusExp = [engine->graph logarithmWithTensor:onePlusExp
                                                                                         name:@"log_one_plus_exp"];
                            
                            // BCE = max(x, 0) - x * z + log(1 + exp(-abs(x)))
                            MPSGraphTensor* firstPart = [engine->graph subtractionWithPrimaryTensor:maxTerm
                                                                                   secondaryTensor:logitsLabels
                                                                                              name:@"first_part"];
                            MPSGraphTensor* bceLogits = [engine->graph additionWithPrimaryTensor:firstPart
                                                                                secondaryTensor:logOnePlusExp
                                                                                           name:@"bce_logits"];
                            
                            // Mean over batch
                            // Mean over batch using MPSGraph reduction (supports dynamic batch size)
                            // Sum over batch dimension, then divide by dynamic batch size
                            MPSGraphTensor* sumLoss = [engine->graph reductionSumWithTensor:bceLogits
                                                                                       axes:@[@0]  // Reduce over batch dimension
                                                                                       name:@"bce_logits_sum"];
                            
                            // Get the shape tensor to compute batch size dynamically
                            MPSGraphTensor* shapeTensor = [engine->graph shapeOfTensor:labelTensor name:@"label_shape_bce_logits"];
                            MPSGraphTensor* batchSizeTensor = [engine->graph sliceTensor:shapeTensor
                                                                               dimension:0
                                                                                   start:0
                                                                                  length:1
                                                                                    name:@"batch_size_bce_logits"];
                            MPSGraphTensor* batchSizeFloat = [engine->graph castTensor:batchSizeTensor
                                                                                toType:MPSDataTypeFloat32
                                                                                  name:@"batch_size_float_bce_logits"];
                            
                            actualLoss = [engine->graph divisionWithPrimaryTensor:sumLoss
                                                                  secondaryTensor:batchSizeFloat
                                                                             name:@"bce_with_logits_loss"];
                        }
                        break;
                    case 4: // CategoricalCrossEntropy
                        {
                            // Categorical cross-entropy without softmax
                            // Assumes labels are probabilities (one-hot or soft labels) and predictions are probabilities
                            
                            // Ensure predictions are in valid range [epsilon, 1-epsilon] for numerical stability
                            float epsilon = 1e-7f;
                            MPSGraphTensor* epsilonTensor = [engine->graph constantWithScalar:epsilon
                                                                                    dataType:MPSDataTypeFloat32];
                            MPSGraphTensor* oneMinusEpsilon = [engine->graph constantWithScalar:(1.0f - epsilon)
                                                                                        dataType:MPSDataTypeFloat32];
                            
                            // Clamp predictions to [epsilon, 1-epsilon]
                            MPSGraphTensor* clampedPreds = [engine->graph clampWithTensor:currentTensor
                                                                            minValueTensor:epsilonTensor
                                                                            maxValueTensor:oneMinusEpsilon
                                                                                      name:@"clamped_predictions"];
                            
                            // Categorical cross-entropy: -sum(y_i * log(p_i))
                            MPSGraphTensor* logPreds = [engine->graph logarithmWithTensor:clampedPreds
                                                                                     name:@"log_predictions"];
                            
                            // y_i * log(p_i)
                            MPSGraphTensor* labelLogPreds = [engine->graph multiplicationWithPrimaryTensor:labelTensor
                                                                                           secondaryTensor:logPreds
                                                                                                      name:@"label_log_predictions"];
                            
                            // Sum over classes (axis=1)
                            MPSGraphTensor* sumOverClasses = [engine->graph reductionSumWithTensor:labelLogPreds
                                                                                             axes:@[@1]
                                                                                             name:@"sum_over_classes"];
                            
                            // Negative sum
                            MPSGraphTensor* negativeSum = [engine->graph negativeWithTensor:sumOverClasses
                                                                                       name:@"negative_sum"];
                            
                            // Mean over batch using MPSGraph reduction (supports dynamic batch size)
                            // Sum over batch dimension, then divide by dynamic batch size
                            MPSGraphTensor* sumLoss = [engine->graph reductionSumWithTensor:negativeSum
                                                                                       axes:@[@0]  // Reduce over batch dimension
                                                                                       name:@"categorical_ce_sum"];
                            
                            // Get the shape tensor to compute batch size dynamically
                            MPSGraphTensor* shapeTensor = [engine->graph shapeOfTensor:labelTensor name:@"label_shape_cat_ce"];
                            MPSGraphTensor* batchSizeTensor = [engine->graph sliceTensor:shapeTensor
                                                                               dimension:0
                                                                                   start:0
                                                                                  length:1
                                                                                    name:@"batch_size_cat_ce"];
                            MPSGraphTensor* batchSizeFloat = [engine->graph castTensor:batchSizeTensor
                                                                                toType:MPSDataTypeFloat32
                                                                                  name:@"batch_size_float_cat_ce"];
                            
                            actualLoss = [engine->graph divisionWithPrimaryTensor:sumLoss
                                                                  secondaryTensor:batchSizeFloat
                                                                             name:@"categorical_cross_entropy_loss"];
                        }
                        break;
                    default:
                        NSLog(@"❌ Unsupported classification loss function: %d", engine->config.loss_function);
                        return NO;
                }
                break;
                
            case 1: // Regression
                switch (engine->config.loss_function) {
                    case 5: // MSE (Mean Squared Error)
                        {
                            // For regression, currentTensor contains raw predictions (no softmax)
                            // Calculate (predictions - labels)^2
                            MPSGraphTensor* diff = [engine->graph subtractionWithPrimaryTensor:currentTensor
                                                                               secondaryTensor:labelTensor
                                                                                          name:@"mse_diff"];
                            MPSGraphTensor* squared = [engine->graph squareWithTensor:diff name:@"mse_squared"];
                            
                            // Mean over all elements
                            MPSGraphTensor* sumLoss = [engine->graph reductionSumWithTensor:squared
                                                                                       axes:nil
                                                                                       name:@"mse_sum"];
                            NSArray<NSNumber*>* shape = squared.shape;
                            int totalElements = 1;
                            for (NSNumber* dim in shape) {
                                totalElements *= [dim intValue];
                            }
                            MPSGraphTensor* countTensor = [engine->graph constantWithScalar:(float)totalElements
                                                                                   dataType:MPSDataTypeFloat32];
                            actualLoss = [engine->graph divisionWithPrimaryTensor:sumLoss
                                                                  secondaryTensor:countTensor
                                                                             name:@"mse_loss"];
                        }
                        break;
                        
                    case 6: // MAE (Mean Absolute Error)
                        {
                            MPSGraphTensor* diff = [engine->graph subtractionWithPrimaryTensor:currentTensor
                                                                               secondaryTensor:labelTensor
                                                                                          name:@"mae_diff"];
                            MPSGraphTensor* abs = [engine->graph absoluteWithTensor:diff name:@"mae_abs"];
                            
                            MPSGraphTensor* sumLoss = [engine->graph reductionSumWithTensor:abs
                                                                                       axes:nil
                                                                                       name:@"mae_sum"];
                            NSArray<NSNumber*>* shape = abs.shape;
                            int totalElements = 1;
                            for (NSNumber* dim in shape) {
                                totalElements *= [dim intValue];
                            }
                            MPSGraphTensor* countTensor = [engine->graph constantWithScalar:(float)totalElements
                                                                                   dataType:MPSDataTypeFloat32];
                            actualLoss = [engine->graph divisionWithPrimaryTensor:sumLoss
                                                                  secondaryTensor:countTensor
                                                                             name:@"mae_loss"];
                        }
                        break;
                        
                    case 7: // Huber
                        {
                            // Huber loss with delta = 1.0 (can be made configurable)
                            float delta = 1.0f;
                            MPSGraphTensor* deltaTensor = [engine->graph constantWithScalar:delta dataType:MPSDataTypeFloat32];
                            
                            MPSGraphTensor* diff = [engine->graph subtractionWithPrimaryTensor:currentTensor
                                                                               secondaryTensor:labelTensor
                                                                                          name:@"huber_diff"];
                            MPSGraphTensor* absDiff = [engine->graph absoluteWithTensor:diff name:@"huber_abs_diff"];
                            
                            // Create condition: |diff| <= delta
                            MPSGraphTensor* condition = [engine->graph lessThanOrEqualToWithPrimaryTensor:absDiff
                                                                                         secondaryTensor:deltaTensor
                                                                                                    name:@"huber_condition"];
                            
                            // Quadratic part: 0.5 * diff^2
                            MPSGraphTensor* squared = [engine->graph squareWithTensor:diff name:@"huber_squared"];
                            MPSGraphTensor* halfTensor = [engine->graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32];
                            MPSGraphTensor* quadratic = [engine->graph multiplicationWithPrimaryTensor:halfTensor
                                                                                      secondaryTensor:squared
                                                                                                 name:@"huber_quadratic"];
                            
                            // Linear part: delta * (|diff| - 0.5 * delta)
                            MPSGraphTensor* halfDelta = [engine->graph multiplicationWithPrimaryTensor:halfTensor
                                                                                      secondaryTensor:deltaTensor
                                                                                                 name:@"half_delta"];
                            MPSGraphTensor* linearDiff = [engine->graph subtractionWithPrimaryTensor:absDiff
                                                                                     secondaryTensor:halfDelta
                                                                                                name:@"linear_diff"];
                            MPSGraphTensor* linear = [engine->graph multiplicationWithPrimaryTensor:deltaTensor
                                                                                   secondaryTensor:linearDiff
                                                                                              name:@"huber_linear"];
                            
                            // Select based on condition
                            MPSGraphTensor* huberPerElement = [engine->graph selectWithPredicateTensor:condition
                                                                                  truePredicateTensor:quadratic
                                                                                 falsePredicateTensor:linear
                                                                                                 name:@"huber_per_element"];
                            
                            MPSGraphTensor* sumLoss = [engine->graph reductionSumWithTensor:huberPerElement
                                                                                       axes:nil
                                                                                       name:@"huber_sum"];
                            NSArray<NSNumber*>* shape = huberPerElement.shape;
                            int totalElements = 1;
                            for (NSNumber* dim in shape) {
                                totalElements *= [dim intValue];
                            }
                            MPSGraphTensor* countTensor = [engine->graph constantWithScalar:(float)totalElements
                                                                                   dataType:MPSDataTypeFloat32];
                            actualLoss = [engine->graph divisionWithPrimaryTensor:sumLoss
                                                                  secondaryTensor:countTensor
                                                                             name:@"huber_loss"];
                        }
                        break;
                        
                    default:
                        NSLog(@"❌ Unsupported regression loss function: %d", engine->config.loss_function);
                        return NO;
                }
                break;
                
            default:
                NSLog(@"❌ Unsupported problem type: %d", engine->config.problem_type);
                return NO;
        }
        
        // Store ACTUAL LOSS for gradient computation (not predictions)
        engine->lossOutput = actualLoss;
        
        // Store the ordered parameter placeholders in the engine
        // Clear the old arrays and copy from our correctly ordered array
        [engine->allWeightPlaceholders removeAllObjects];
        [engine->allBiasPlaceholders removeAllObjects];
        
        // Initialize BatchNorm running stats placeholders array
        if (!engine->batchnormRunningStatsPlaceholders) {
            engine->batchnormRunningStatsPlaceholders = [[NSMutableArray alloc] init];
        }
        [engine->batchnormRunningStatsPlaceholders removeAllObjects];
        
        // Store all parameters in the correct order (weight, bias, weight, bias...)
        // This matches exactly how Go CreateParameterTensors works
        for (int i = 0; i < allParameterPlaceholders.count; i++) {
            MPSGraphTensor* placeholder = allParameterPlaceholders[i];
            [engine->allWeightPlaceholders addObject:placeholder];
        }
        
        // Get loss function name from configuration
        NSString* lossFunctionName = @"Unknown";
        switch (engine->config.loss_function) {
            case 0: lossFunctionName = @"CrossEntropy"; break;
            case 1: lossFunctionName = @"SparseCrossEntropy"; break;
            case 2: lossFunctionName = @"BinaryCrossEntropy"; break;
            case 3: lossFunctionName = @"BCEWithLogits"; break;
            case 4: lossFunctionName = @"CategoricalCrossEntropy"; break;
            case 5: lossFunctionName = @"MSE"; break;
            case 6: lossFunctionName = @"MAE"; break;
            case 7: lossFunctionName = @"Huber"; break;
        }
        
        NSLog(@"✅ Dynamic graph built successfully with %d layers and proper loss computation", numLayers);
        NSLog(@"   - Parameters: %lu placeholders", (unsigned long)allParameterPlaceholders.count);
        NSLog(@"   - Output classes: %d", numClasses);
        NSLog(@"   - Loss: %@ with automatic differentiation", lossFunctionName);
        
        // PRODUCTION OPTIMIZATION: Cache Adam scalar tensors if using Adam optimizer
        if (engine->config.optimizer_type == 1) { // Adam optimizer
            cacheAdamScalarTensors(engine);
            
            // UNIFIED OPTIMIZER: Initialize Adam state arrays BEFORE pre-compilation
            engine->momentumPlaceholders = [[NSMutableArray alloc] init];
            engine->variancePlaceholders = [[NSMutableArray alloc] init];
            engine->momentumVariables = [[NSMutableArray alloc] init];
            engine->varianceVariables = [[NSMutableArray alloc] init];
            engine->momentumBuffers = [[NSMutableArray alloc] init];
            engine->varianceBuffers = [[NSMutableArray alloc] init];
            
            // Initialize momentum and variance state for each parameter
            for (int i = 0; i < [engine->allWeightPlaceholders count]; i++) {
                MPSGraphTensor* paramTensor = [engine->allWeightPlaceholders objectAtIndex:i];
                NSArray<NSNumber*>* paramShape = [paramTensor shape];
                
                // Calculate total elements and buffer size
                NSUInteger elementCount = 1;
                for (NSNumber* dim in paramShape) {
                    elementCount *= [dim unsignedIntegerValue];
                }
                NSUInteger bufferSize = elementCount * sizeof(float);
                
                // Create Metal buffers for momentum and variance (initialized to zero)
                id<MTLBuffer> momentumBuffer = [engine->device newBufferWithLength:bufferSize 
                                                                           options:MTLResourceStorageModeShared];
                id<MTLBuffer> varianceBuffer = [engine->device newBufferWithLength:bufferSize 
                                                                           options:MTLResourceStorageModeShared];
                
                // Zero-initialize the buffers
                memset([momentumBuffer contents], 0, bufferSize);
                memset([varianceBuffer contents], 0, bufferSize);
                
                [engine->momentumBuffers addObject:momentumBuffer];
                [engine->varianceBuffers addObject:varianceBuffer];
                
                // Create placeholders for momentum and variance state
                MPSGraphTensor* momentumPlaceholder = [engine->graph placeholderWithShape:paramShape
                                                                                 dataType:MPSDataTypeFloat32
                                                                                     name:[NSString stringWithFormat:@"momentum_%d", i]];
                MPSGraphTensor* variancePlaceholder = [engine->graph placeholderWithShape:paramShape
                                                                                 dataType:MPSDataTypeFloat32
                                                                                     name:[NSString stringWithFormat:@"variance_%d", i]];
                
                [engine->momentumPlaceholders addObject:momentumPlaceholder];
                [engine->variancePlaceholders addObject:variancePlaceholder];
            }
            
            // Initialize Adam step counter
            engine->adamStepCount = 0;
            
            engine->adamStateInitialized = YES;
            NSLog(@"✅ Adam state initialized EARLY for %lu parameters", [engine->allWeightPlaceholders count]);
            NSLog(@"🔍 ADAM DEBUG: State flag set to %d", engine->adamStateInitialized);
        }
        
        // UNIFIED OPTIMIZER: Initialize SGD state for momentum (shares momentum arrays with Adam)
        if (engine->config.optimizer_type == 0) { // SGD optimizer
            
            // Cache SGD scalar tensors during graph building phase
            cacheSGDScalarTensors(engine);
            
            // SGD can optionally use momentum (through config.beta1 for compatibility)
            if (engine->config.beta1 > 0.0f) {
                // Initialize momentum state using the same pattern as Adam (placeholders + buffers)
                if (!engine->momentumPlaceholders) {
                    engine->momentumPlaceholders = [[NSMutableArray alloc] init];
                    engine->momentumBuffers = [[NSMutableArray alloc] init];
                    
                    for (int i = 0; i < [engine->allWeightPlaceholders count]; i++) {
                        MPSGraphTensor* paramTensor = [engine->allWeightPlaceholders objectAtIndex:i];
                        NSArray<NSNumber*>* paramShape = [paramTensor shape];
                        
                        // Calculate buffer size for momentum state
                        NSUInteger elementCount = 1;
                        for (NSNumber* dim in paramShape) {
                            elementCount *= [dim unsignedIntegerValue];
                        }
                        NSUInteger bufferSize = elementCount * sizeof(float);
                        
                        // Create momentum buffer (initialized to zero)
                        id<MTLBuffer> momentumBuffer = [engine->device newBufferWithLength:bufferSize 
                                                                                   options:MTLResourceStorageModeShared];
                        memset([momentumBuffer contents], 0, bufferSize); // Zero-initialize
                        [engine->momentumBuffers addObject:momentumBuffer];
                        
                        // Create momentum placeholder for feeding data to graph
                        MPSGraphTensor* momentumPlaceholder = [engine->graph placeholderWithShape:paramShape
                                                                                         dataType:MPSDataTypeFloat32
                                                                                             name:[NSString stringWithFormat:@"sgd_momentum_%d", i]];
                        [engine->momentumPlaceholders addObject:momentumPlaceholder];
                    }
                }
            }
            
            engine->sgdStateInitialized = YES;
            NSLog(@"✅ SGD state initialized for %lu parameters (momentum: %s)", 
                  [engine->allWeightPlaceholders count], 
                  engine->config.beta1 > 0.0f ? "enabled" : "disabled");
        }
        
        // RMSProp OPTIMIZER: Initialize RMSProp state arrays
        if (engine->config.optimizer_type == 2) { // RMSProp optimizer
            
            // Cache RMSProp scalar tensors during graph building phase
            cacheRMSPropScalarTensors(engine);
            
            // Initialize RMSProp state arrays for squared gradient averages
            engine->squaredGradPlaceholders = [[NSMutableArray alloc] init];
            engine->squaredGradVariables = [[NSMutableArray alloc] init];
            engine->squaredGradBuffers = [[NSMutableArray alloc] init];
            
            // Initialize optional momentum state (if momentum > 0)
            if (engine->config.momentum > 0.0f) {
                if (!engine->momentumPlaceholders) {
                    engine->momentumPlaceholders = [[NSMutableArray alloc] init];
                    engine->momentumBuffers = [[NSMutableArray alloc] init];
                }
            }
            
            // Initialize optional gradient averages for centered RMSProp (if centered == 1)
            if (engine->config.centered) {
                engine->gradAvgPlaceholders = [[NSMutableArray alloc] init];
                engine->gradAvgVariables = [[NSMutableArray alloc] init];
                engine->gradAvgBuffers = [[NSMutableArray alloc] init];
            }
            
            // Initialize state buffers for each parameter
            for (int i = 0; i < [engine->allWeightPlaceholders count]; i++) {
                MPSGraphTensor* paramTensor = [engine->allWeightPlaceholders objectAtIndex:i];
                NSArray<NSNumber*>* paramShape = [paramTensor shape];
                
                // Calculate total elements and buffer size
                NSUInteger elementCount = 1;
                for (NSNumber* dim in paramShape) {
                    elementCount *= [dim unsignedIntegerValue];
                }
                NSUInteger bufferSize = elementCount * sizeof(float);
                
                // Create Metal buffer for squared gradient averages (initialized to zero)
                id<MTLBuffer> squaredGradBuffer = [engine->device newBufferWithLength:bufferSize 
                                                                               options:MTLResourceStorageModeShared];
                memset([squaredGradBuffer contents], 0, bufferSize);
                [engine->squaredGradBuffers addObject:squaredGradBuffer];
                
                // Create placeholder for squared gradient average state
                MPSGraphTensor* squaredGradPlaceholder = [engine->graph placeholderWithShape:paramShape
                                                                                     dataType:MPSDataTypeFloat32
                                                                                         name:[NSString stringWithFormat:@"squared_grad_%d", i]];
                [engine->squaredGradPlaceholders addObject:squaredGradPlaceholder];
                
                // Initialize momentum state if momentum > 0
                if (engine->config.momentum > 0.0f && !engine->momentumBuffers) {
                    id<MTLBuffer> momentumBuffer = [engine->device newBufferWithLength:bufferSize 
                                                                               options:MTLResourceStorageModeShared];
                    memset([momentumBuffer contents], 0, bufferSize);
                    [engine->momentumBuffers addObject:momentumBuffer];
                    
                    MPSGraphTensor* momentumPlaceholder = [engine->graph placeholderWithShape:paramShape
                                                                                     dataType:MPSDataTypeFloat32
                                                                                         name:[NSString stringWithFormat:@"rmsprop_momentum_%d", i]];
                    [engine->momentumPlaceholders addObject:momentumPlaceholder];
                }
                
                // Initialize gradient averages for centered RMSProp
                if (engine->config.centered) {
                    id<MTLBuffer> gradAvgBuffer = [engine->device newBufferWithLength:bufferSize 
                                                                               options:MTLResourceStorageModeShared];
                    memset([gradAvgBuffer contents], 0, bufferSize);
                    [engine->gradAvgBuffers addObject:gradAvgBuffer];
                    
                    MPSGraphTensor* gradAvgPlaceholder = [engine->graph placeholderWithShape:paramShape
                                                                                     dataType:MPSDataTypeFloat32
                                                                                         name:[NSString stringWithFormat:@"grad_avg_%d", i]];
                    [engine->gradAvgPlaceholders addObject:gradAvgPlaceholder];
                }
            }
            
            engine->rmspropStateInitialized = YES;
            NSLog(@"✅ RMSProp state initialized for %lu parameters (momentum=%s, centered=%s)", 
                  [engine->allWeightPlaceholders count], 
                  engine->config.momentum > 0.0f ? "enabled" : "disabled",
                  engine->config.centered ? "enabled" : "disabled");
        }
        
        // OPTIMIZER-SPECIFIC PRE-COMPILATION: Build optimizer-specific graphs to avoid conflicts
        // This eliminates runtime operation creation and prevents placeholder issues
        // NSLog(@"🔍 PRE-COMPILATION DEBUG: optimizer=%d, params=%lu, adam_state=%d, sgd_state=%d", 
        //       engine->config.optimizer_type, [engine->allWeightPlaceholders count], engine->adamStateInitialized, engine->sgdStateInitialized);
        
        // SGD-SPECIFIC GRAPH COMPILATION: Build SGD graph without Adam dependencies
        if (engine->config.optimizer_type == 0 && engine->allWeightPlaceholders.count > 0 && engine->sgdStateInitialized && !engine->sgdGraphCompiled) { // SGD optimizer
            
            NSLog(@"🚀 SGD PRE-COMPILATION: Building gradient and SGD operations in graph...");
            
            // SGD scalar tensors should already be cached by cacheSGDScalarTensors
            // Just verify they exist and use the correct field names
            if (!engine->sgdScalarsCached) {
                NSLog(@"⚠️ SGD scalars not cached - this should not happen");
                engine->sgdScalarsCached = YES;
            }
            
            // Verify SGD-specific learning rate tensor is available
            if (!engine->sgdCachedLrTensor) {
                NSLog(@"❌ CRITICAL: SGD learning rate tensor not cached! This should have been created in cacheSGDScalarTensors.");
                return NO;
            }
            // NSLog(@"🔧 DEBUG: SGD using sgdCachedLrTensor with config value: %.6f", engine->config.learning_rate);
            
            // Pre-compile gradient computation using automatic differentiation ONCE during graph building (same as Adam)
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* precompiledGradients = 
                [engine->graph gradientForPrimaryTensor:engine->lossOutput
                                            withTensors:engine->allWeightPlaceholders
                                                   name:@"sgd_precompiled_gradients"];
            
            // Store gradient tensors for execution (these are pre-compiled, not runtime-created) - same as Adam
            NSMutableArray<MPSGraphTensor*>* precompiledGradientTensors = [[NSMutableArray alloc] init];
            for (MPSGraphTensor* paramPlaceholder in engine->allWeightPlaceholders) {
                MPSGraphTensor* gradTensor = precompiledGradients[paramPlaceholder];
                if (gradTensor) {
                    [precompiledGradientTensors addObject:gradTensor];
                } else {
                    NSLog(@"❌ Failed to pre-compile gradient for parameter");
                    return NO;
                }
            }
            
            // Store pre-compiled gradients in SGD-specific arrays (same pattern as Adam)
            engine->sgdPrecompiledGradients = precompiledGradientTensors;
            
            // Also store in legacy array for backward compatibility (same as Adam)
            engine->precompiledGradientTensors = precompiledGradientTensors;
            // NSLog(@"🔧 DEBUG: Set precompiledGradientTensors to %lu items", [precompiledGradientTensors count]);
            
            NSLog(@"✅ PRE-COMPILATION: Successfully built gradient operations for %lu parameters", [engine->allWeightPlaceholders count]);
            NSLog(@"🚀 PRE-COMPILATION: Building SGD parameter update operations...");
            
            // Pre-compile SGD parameter updates only if properly initialized (same pattern as Adam)
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedParams = [[NSMutableArray alloc] init];
            // Note: This array can contain both MPSGraphTensor and NSNull objects for SGD without momentum
            NSMutableArray* precompiledUpdatedMomentum = [[NSMutableArray alloc] init];
            
            for (int i = 0; i < [engine->allWeightPlaceholders count]; i++) {
                MPSGraphTensor* paramTensor = [engine->allWeightPlaceholders objectAtIndex:i];
                MPSGraphTensor* gradTensor = [precompiledGradientTensors objectAtIndex:i];
                
                if (gradTensor) {
                    if (engine->config.beta1 > 0.0f && engine->momentumPlaceholders && i < engine->momentumPlaceholders.count) {
                        // CORRECTED SGD with momentum: m = beta1 * m + grad; param = param - lr * m
                        MPSGraphTensor* momentumPlaceholder = [engine->momentumPlaceholders objectAtIndex:i];
                        
                        // Updated momentum: new_momentum = beta1 * old_momentum + gradient (NO learning rate scaling)
                        MPSGraphTensor* scaledMomentum = [engine->graph multiplicationWithPrimaryTensor:momentumPlaceholder
                                                                                      secondaryTensor:engine->cachedMomentumTensor
                                                                                                 name:[NSString stringWithFormat:@"sgd_scaled_momentum_%d", i]];
                        MPSGraphTensor* updatedMomentum = [engine->graph additionWithPrimaryTensor:scaledMomentum
                                                                                   secondaryTensor:gradTensor
                                                                                              name:[NSString stringWithFormat:@"sgd_updated_momentum_%d", i]];
                        
                        // Apply learning rate to final momentum: lr * updated_momentum
                        MPSGraphTensor* scaledMomentumUpdate = [engine->graph multiplicationWithPrimaryTensor:updatedMomentum
                                                                                           secondaryTensor:engine->sgdCachedLrTensor
                                                                                                      name:[NSString stringWithFormat:@"sgd_lr_momentum_%d", i]];
                        
                        // Updated parameter: param = param - lr * momentum
                        MPSGraphTensor* updatedParam = [engine->graph subtractionWithPrimaryTensor:paramTensor
                                                                                   secondaryTensor:scaledMomentumUpdate
                                                                                              name:[NSString stringWithFormat:@"sgd_updated_param_%d", i]];
                        
                        [precompiledUpdatedMomentum addObject:updatedMomentum];
                        [precompiledUpdatedParams addObject:updatedParam];
                    } else {
                        // Standard SGD: param = param - learning_rate * gradient
                        MPSGraphTensor* scaledGradient = [engine->graph multiplicationWithPrimaryTensor:gradTensor
                                                                                      secondaryTensor:engine->sgdCachedLrTensor
                                                                                                 name:[NSString stringWithFormat:@"sgd_scaled_grad_%d", i]];
                        
                        // CORRECTED: Apply weight decay by adding to gradient (no double learning rate scaling)
                        if (engine->config.weight_decay > 0.0f) {
                            MPSGraphTensor* weightDecayTerm = [engine->graph multiplicationWithPrimaryTensor:engine->cachedWeightDecayTensor
                                                                                             secondaryTensor:paramTensor
                                                                                                        name:[NSString stringWithFormat:@"sgd_weight_decay_%d", i]];
                            // Add weight decay directly to gradient (learning rate applied once below)
                            MPSGraphTensor* gradientWithDecay = [engine->graph additionWithPrimaryTensor:gradTensor
                                                                                         secondaryTensor:weightDecayTerm
                                                                                                    name:[NSString stringWithFormat:@"sgd_grad_plus_decay_%d", i]];
                            // Apply learning rate once to combined gradient + weight decay
                            scaledGradient = [engine->graph multiplicationWithPrimaryTensor:gradientWithDecay
                                                                              secondaryTensor:engine->sgdCachedLrTensor
                                                                                         name:[NSString stringWithFormat:@"sgd_scaled_grad_%d", i]];
                        } else {
                            // No weight decay: just apply learning rate to gradient
                            scaledGradient = [engine->graph multiplicationWithPrimaryTensor:gradTensor
                                                                              secondaryTensor:engine->sgdCachedLrTensor
                                                                                         name:[NSString stringWithFormat:@"sgd_scaled_grad_%d", i]];
                        }
                        
                        MPSGraphTensor* updatedParam = [engine->graph subtractionWithPrimaryTensor:paramTensor
                                                                                   secondaryTensor:scaledGradient
                                                                                              name:[NSString stringWithFormat:@"sgd_updated_param_%d", i]];
                        
                        [precompiledUpdatedParams addObject:updatedParam];
                        // CRITICAL FIX: For standard SGD without momentum, we don't need momentum updates
                        // Adding NSNull placeholder to maintain array alignment with parameters
                        // Note: The training code checks for NSNull and skips momentum updates
                        [precompiledUpdatedMomentum addObject:[NSNull null]];
                    }
                } else {
                    NSLog(@"❌ Failed to pre-compile parameter update for parameter %d", i);
                    return NO;
                }
            }
            
            // Store pre-compiled SGD operations in SGD-specific arrays (same pattern as Adam)
            engine->sgdPrecompiledUpdatedParams = precompiledUpdatedParams;
            engine->sgdPrecompiledUpdatedMomentum = precompiledUpdatedMomentum;
            
            // Also store in legacy arrays for backward compatibility (same as Adam)
            engine->precompiledUpdatedParams = precompiledUpdatedParams;
            // NSLog(@"🔧 DEBUG: Set precompiledUpdatedParams to %lu items", [precompiledUpdatedParams count]);
            engine->precompiledUpdatedMomentum = precompiledUpdatedMomentum;
            
            engine->sgdGraphCompiled = YES;
            NSLog(@"✅ SGD PRE-COMPILATION: Successfully built SGD update operations for %lu parameters", [engine->allWeightPlaceholders count]);
        }
        
        // ADAM-SPECIFIC GRAPH COMPILATION: Build Adam graph without SGD dependencies  
        else if (engine->config.optimizer_type == 1 && engine->allWeightPlaceholders.count > 0 && engine->adamStateInitialized && !engine->adamGraphCompiled) { // Adam optimizer
            
            NSLog(@"🚀 PRE-COMPILATION: Building gradient and Adam operations in graph...");
            
            // Pre-compile gradient computation using automatic differentiation ONCE during graph building
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* precompiledGradients = 
                [engine->graph gradientForPrimaryTensor:engine->lossOutput
                                            withTensors:engine->allWeightPlaceholders
                                                   name:@"precompiled_gradients"];
            
            // Store gradient tensors for execution (these are pre-compiled, not runtime-created)
            NSMutableArray<MPSGraphTensor*>* precompiledGradientTensors = [[NSMutableArray alloc] init];
            for (MPSGraphTensor* paramPlaceholder in engine->allWeightPlaceholders) {
                MPSGraphTensor* gradTensor = precompiledGradients[paramPlaceholder];
                if (gradTensor) {
                    [precompiledGradientTensors addObject:gradTensor];
                } else {
                    NSLog(@"❌ Failed to pre-compile gradient for parameter");
                    return NO;
                }
            }
            
            // Store pre-compiled gradients in Adam-specific arrays
            engine->adamPrecompiledGradients = precompiledGradientTensors;
            
            // Also store in legacy array for backward compatibility
            engine->precompiledGradientTensors = precompiledGradientTensors;
            
            NSLog(@"✅ PRE-COMPILATION: Successfully built gradient operations for %lu parameters", [engine->allWeightPlaceholders count]);
            NSLog(@"🚀 PRE-COMPILATION: Building Adam parameter update operations...");
            
            
            // Pre-compile Adam parameter updates only if properly initialized
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedParams = [[NSMutableArray alloc] init];
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedMomentum = [[NSMutableArray alloc] init];
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedVariance = [[NSMutableArray alloc] init];
            
            for (int i = 0; i < [engine->allWeightPlaceholders count]; i++) {
                MPSGraphTensor* paramTensor = [engine->allWeightPlaceholders objectAtIndex:i];
                MPSGraphTensor* gradTensor = [precompiledGradientTensors objectAtIndex:i];
                MPSGraphTensor* momentumTensor = [engine->momentumPlaceholders objectAtIndex:i];
                MPSGraphTensor* varianceTensor = [engine->variancePlaceholders objectAtIndex:i];
                
                
                if (gradTensor && momentumTensor && varianceTensor) {
                    // Adam update equations:
                    // m = beta1 * m + (1 - beta1) * g
                    // v = beta2 * v + (1 - beta2) * g^2
                    // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
                    
                    
                    MPSGraphTensor* scaledMomentum = [engine->graph multiplicationWithPrimaryTensor:engine->cachedBeta1Tensor
                                                                                   secondaryTensor:momentumTensor
                                                                                              name:nil];
                    
                    MPSGraphTensor* scaledGradient = [engine->graph multiplicationWithPrimaryTensor:engine->cachedOneMinusBeta1
                                                                                   secondaryTensor:gradTensor
                                                                                              name:nil];
                    
                    MPSGraphTensor* updatedMomentum = [engine->graph additionWithPrimaryTensor:scaledMomentum
                                                                               secondaryTensor:scaledGradient
                                                                                          name:nil];
                    
                    MPSGraphTensor* scaledVariance = [engine->graph multiplicationWithPrimaryTensor:engine->cachedBeta2Tensor
                                                                                   secondaryTensor:varianceTensor
                                                                                              name:nil];
                    
                    MPSGraphTensor* squaredGradient = [engine->graph squareWithTensor:gradTensor
                                                                                 name:nil];
                    
                    MPSGraphTensor* scaledSquaredGradient = [engine->graph multiplicationWithPrimaryTensor:engine->cachedOneMinusBeta2
                                                                                           secondaryTensor:squaredGradient
                                                                                                      name:nil];
                    
                    MPSGraphTensor* updatedVariance = [engine->graph additionWithPrimaryTensor:scaledVariance
                                                                               secondaryTensor:scaledSquaredGradient
                                                                                          name:nil];
                    
                    // Bias correction
                    MPSGraphTensor* biasCorrection1 = [engine->graph divisionWithPrimaryTensor:updatedMomentum
                                                                               secondaryTensor:engine->biasCorr1Placeholder
                                                                                          name:nil];
                    
                    MPSGraphTensor* biasCorrection2 = [engine->graph divisionWithPrimaryTensor:updatedVariance
                                                                               secondaryTensor:engine->biasCorr2Placeholder
                                                                                          name:nil];
                    
                    // Parameter update
                    MPSGraphTensor* sqrtVariance = [engine->graph squareRootWithTensor:biasCorrection2
                                                                                  name:nil];
                    
                    MPSGraphTensor* denominator = [engine->graph additionWithPrimaryTensor:sqrtVariance
                                                                           secondaryTensor:engine->cachedEpsilonTensor
                                                                                      name:nil];
                    
                    MPSGraphTensor* updateDirection = [engine->graph divisionWithPrimaryTensor:biasCorrection1
                                                                               secondaryTensor:denominator
                                                                                          name:nil];
                    
                    MPSGraphTensor* scaledUpdate = [engine->graph multiplicationWithPrimaryTensor:engine->cachedLrTensor
                                                                                  secondaryTensor:updateDirection
                                                                                             name:nil];
                    
                    // Apply weight decay if specified
                    if (engine->config.weight_decay > 0.0f) {
                        MPSGraphTensor* weightDecayTerm = [engine->graph multiplicationWithPrimaryTensor:engine->cachedWeightDecayTensor
                                                                                         secondaryTensor:paramTensor
                                                                                                    name:nil];
                        scaledUpdate = [engine->graph additionWithPrimaryTensor:scaledUpdate
                                                                secondaryTensor:weightDecayTerm
                                                                           name:nil];
                    }
                    
                    MPSGraphTensor* updatedParam = [engine->graph subtractionWithPrimaryTensor:paramTensor
                                                                              secondaryTensor:scaledUpdate
                                                                                         name:nil];
                    
                    [precompiledUpdatedParams addObject:updatedParam];
                    [precompiledUpdatedMomentum addObject:updatedMomentum];
                    [precompiledUpdatedVariance addObject:updatedVariance];
                } else {
                    NSLog(@"❌ Failed to pre-compile Adam parameter update for parameter %d", i);
                    return NO;
                }
            }
            
            // Store pre-compiled Adam operations in Adam-specific arrays
            engine->adamPrecompiledUpdatedParams = precompiledUpdatedParams;
            engine->adamPrecompiledUpdatedMomentum = precompiledUpdatedMomentum;
            engine->adamPrecompiledUpdatedVariance = precompiledUpdatedVariance;
            
            // Also store in legacy arrays for backward compatibility
            engine->precompiledUpdatedParams = precompiledUpdatedParams;
            engine->precompiledUpdatedMomentum = precompiledUpdatedMomentum;
            engine->precompiledUpdatedVariance = precompiledUpdatedVariance;
            
            engine->adamGraphCompiled = YES;
            NSLog(@"✅ PRE-COMPILATION: Successfully built Adam operations for %lu parameters", [engine->allWeightPlaceholders count]);
        }
        
        // RMSPROP-SPECIFIC GRAPH COMPILATION: Build RMSProp graph without Adam/SGD dependencies
        else if (engine->config.optimizer_type == 2 && engine->allWeightPlaceholders.count > 0 && engine->rmspropStateInitialized && !engine->rmspropGraphCompiled) { // RMSProp optimizer
            
            NSLog(@"🚀 PRE-COMPILATION: Building gradient and RMSProp operations in graph...");
            
            // Pre-compile gradient computation using automatic differentiation ONCE during graph building
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* precompiledGradients = 
                [engine->graph gradientForPrimaryTensor:engine->lossOutput
                                            withTensors:engine->allWeightPlaceholders
                                                   name:@"precompiled_gradients"];
            
            // Store gradient tensors for execution (these are pre-compiled, not runtime-created)
            NSMutableArray<MPSGraphTensor*>* precompiledGradientTensors = [[NSMutableArray alloc] init];
            for (MPSGraphTensor* paramPlaceholder in engine->allWeightPlaceholders) {
                MPSGraphTensor* gradTensor = precompiledGradients[paramPlaceholder];
                if (gradTensor) {
                    [precompiledGradientTensors addObject:gradTensor];
                } else {
                    NSLog(@"❌ Failed to pre-compile gradient for parameter");
                    return NO;
                }
            }
            
            // Store pre-compiled gradients in RMSProp-specific arrays
            engine->rmspropPrecompiledGradients = precompiledGradientTensors;
            
            // Also store in legacy array for backward compatibility
            engine->precompiledGradientTensors = precompiledGradientTensors;
            
            NSLog(@"✅ PRE-COMPILATION: Successfully built gradient operations for %lu parameters", [engine->allWeightPlaceholders count]);
            NSLog(@"🚀 PRE-COMPILATION: Building RMSProp parameter update operations...");
            
            // CRITICAL: Verify array integrity before use to prevent string/array confusion crashes
            if (![engine->squaredGradPlaceholders isKindOfClass:[NSMutableArray class]]) {
                NSLog(@"❌ CRITICAL: squaredGradPlaceholders is not NSMutableArray (type: %@) - aborting", [engine->squaredGradPlaceholders class]);
                return NO;
            }
            
            NSLog(@"🔍 Array verification: squaredGradPlaceholders has %lu items", [engine->squaredGradPlaceholders count]);
            
            // Pre-compile RMSProp parameter updates
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedParams = [[NSMutableArray alloc] init];
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedSquaredGrads = [[NSMutableArray alloc] init];
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedMomentum = [[NSMutableArray alloc] init]; // For momentum variant
            NSMutableArray<MPSGraphTensor*>* precompiledUpdatedGradientAvg = [[NSMutableArray alloc] init]; // For centered variant
            
            for (int i = 0; i < [engine->allWeightPlaceholders count]; i++) {
                MPSGraphTensor* paramTensor = [engine->allWeightPlaceholders objectAtIndex:i];
                MPSGraphTensor* gradTensor = [precompiledGradientTensors objectAtIndex:i];
                
                // SAFE: Verify index bounds and type before accessing
                if (i >= [engine->squaredGradPlaceholders count]) {
                    NSLog(@"❌ CRITICAL: Index %d out of bounds for squaredGradPlaceholders (count: %lu)", i, [engine->squaredGradPlaceholders count]);
                    return NO;
                }
                
                MPSGraphTensor* squaredGradTensor = [engine->squaredGradPlaceholders objectAtIndex:i];
                
                if (gradTensor && squaredGradTensor) {
                    // RMSProp update equations:
                    // squared_grad_avg = alpha * squared_grad_avg + (1 - alpha) * grad^2
                    
                    MPSGraphTensor* gradientSquared = [engine->graph squareWithTensor:gradTensor name:nil];
                    
                    MPSGraphTensor* scaledSquaredGradAvg = [engine->graph multiplicationWithPrimaryTensor:engine->cachedAlphaTensor
                                                                                         secondaryTensor:squaredGradTensor
                                                                                                    name:nil];
                    
                    MPSGraphTensor* scaledGradientSquared = [engine->graph multiplicationWithPrimaryTensor:engine->cachedOneMinusAlphaTensor
                                                                                           secondaryTensor:gradientSquared
                                                                                                      name:nil];
                    
                    MPSGraphTensor* updatedSquaredGradAvg = [engine->graph additionWithPrimaryTensor:scaledSquaredGradAvg
                                                                                   secondaryTensor:scaledGradientSquared
                                                                                              name:nil];
                    
                    // Calculate denominator based on whether centered or not
                    MPSGraphTensor* denominator;
                    if (engine->config.centered) {
                        // For centered RMSProp: denominator = sqrt(squared_grad_avg - grad_avg^2 + ε)
                        // Need to update gradient average: grad_avg = alpha * grad_avg + (1 - alpha) * grad
                        if (i < [engine->gradAvgPlaceholders count]) {
                            MPSGraphTensor* gradientAvgTensor = [engine->gradAvgPlaceholders objectAtIndex:i];
                            
                            MPSGraphTensor* scaledGradAvg = [engine->graph multiplicationWithPrimaryTensor:engine->cachedAlphaTensor
                                                                                           secondaryTensor:gradientAvgTensor
                                                                                                      name:nil];
                            
                            MPSGraphTensor* scaledGradient = [engine->graph multiplicationWithPrimaryTensor:engine->cachedOneMinusAlphaTensor
                                                                                           secondaryTensor:gradTensor
                                                                                                      name:nil];
                            
                            MPSGraphTensor* updatedGradientAvg = [engine->graph additionWithPrimaryTensor:scaledGradAvg
                                                                                         secondaryTensor:scaledGradient
                                                                                                    name:nil];
                            
                            // Calculate variance: squared_grad_avg - grad_avg^2
                            MPSGraphTensor* gradientAvgSquared = [engine->graph squareWithTensor:updatedGradientAvg name:nil];
                            MPSGraphTensor* variance = [engine->graph subtractionWithPrimaryTensor:updatedSquaredGradAvg
                                                                                  secondaryTensor:gradientAvgSquared
                                                                                             name:nil];
                            
                            // Ensure variance is always positive (avoid numerical issues)
                            MPSGraphTensor* zeroTensor = [engine->graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
                            MPSGraphTensor* positiveVariance = [engine->graph maximumWithPrimaryTensor:variance
                                                                                      secondaryTensor:zeroTensor
                                                                                                 name:nil];
                            
                            MPSGraphTensor* varianceWithEpsilon = [engine->graph additionWithPrimaryTensor:positiveVariance
                                                                                          secondaryTensor:engine->cachedEpsilonTensor
                                                                                                     name:nil];
                            denominator = [engine->graph squareRootWithTensor:varianceWithEpsilon name:nil];
                            
                            // Store the updated gradient average for centered variant
                            [precompiledUpdatedGradientAvg addObject:updatedGradientAvg];
                        } else {
                            NSLog(@"❌ Missing gradient average placeholder for centered RMSProp parameter %d", i);
                            return NO;
                        }
                    } else {
                        // For standard RMSProp: denominator = sqrt(squared_grad_avg + ε)
                        MPSGraphTensor* squaredGradAvgWithEpsilon = [engine->graph additionWithPrimaryTensor:updatedSquaredGradAvg
                                                                                            secondaryTensor:engine->cachedEpsilonTensor
                                                                                                       name:nil];
                        denominator = [engine->graph squareRootWithTensor:squaredGradAvgWithEpsilon name:nil];
                        
                        // Non-centered variant: no gradient average needed
                    }
                    
                    // Basic update: grad / denominator
                    MPSGraphTensor* updateDirection = [engine->graph divisionWithPrimaryTensor:gradTensor
                                                                               secondaryTensor:denominator
                                                                                          name:nil];
                    
                    // Apply momentum if specified (momentum > 0)
                    if (engine->config.momentum > 0.0f && engine->momentumPlaceholders && i < [engine->momentumPlaceholders count]) {
                        MPSGraphTensor* momentumTensor = [engine->momentumPlaceholders objectAtIndex:i];
                        if (momentumTensor) {
                            // momentum = momentum * old_momentum + update
                            MPSGraphTensor* scaledMomentum = [engine->graph multiplicationWithPrimaryTensor:engine->cachedMomentumTensor
                                                                                           secondaryTensor:momentumTensor
                                                                                                      name:nil];
                            
                            MPSGraphTensor* updatedMomentum = [engine->graph additionWithPrimaryTensor:scaledMomentum
                                                                                      secondaryTensor:updateDirection
                                                                                                 name:nil];
                            
                            updateDirection = updatedMomentum;
                            [precompiledUpdatedMomentum addObject:updatedMomentum];
                        }
                    }
                    
                    // Apply weight decay if specified
                    if (engine->config.weight_decay > 0.0f) {
                        MPSGraphTensor* weightDecayTerm = [engine->graph multiplicationWithPrimaryTensor:paramTensor
                                                                                        secondaryTensor:engine->cachedWeightDecayTensor
                                                                                                   name:nil];
                        updateDirection = [engine->graph additionWithPrimaryTensor:updateDirection
                                                                  secondaryTensor:weightDecayTerm
                                                                             name:nil];
                    }
                    
                    // Scale by learning rate
                    MPSGraphTensor* scaledUpdate = [engine->graph multiplicationWithPrimaryTensor:engine->cachedLrTensor
                                                                                  secondaryTensor:updateDirection
                                                                                             name:nil];
                    
                    // Final parameter update: param = param - lr * update
                    MPSGraphTensor* updatedParam = [engine->graph subtractionWithPrimaryTensor:paramTensor
                                                                               secondaryTensor:scaledUpdate
                                                                                          name:nil];
                    
                    [precompiledUpdatedParams addObject:updatedParam];
                    [precompiledUpdatedSquaredGrads addObject:updatedSquaredGradAvg];
                } else {
                    NSLog(@"❌ Failed to pre-compile RMSProp update for parameter %d", i);
                    return NO;
                }
            }
            
            // Store pre-compiled RMSProp operations in RMSProp-specific arrays
            engine->rmspropPrecompiledUpdatedParams = precompiledUpdatedParams;
            engine->rmspropPrecompiledUpdatedSquaredGrad = precompiledUpdatedSquaredGrads;
            engine->rmspropPrecompiledUpdatedMomentum = precompiledUpdatedMomentum;
            engine->rmspropPrecompiledUpdatedGradAvg = precompiledUpdatedGradientAvg;
            
            // Also store in legacy arrays for backward compatibility
            engine->precompiledUpdatedParams = precompiledUpdatedParams;
            
            engine->rmspropGraphCompiled = YES;
            NSLog(@"✅ PRE-COMPILATION: Successfully built RMSProp operations for %lu parameters", [engine->allWeightPlaceholders count]);
        } else {
            NSLog(@"⚠️ PRE-COMPILATION: Skipping pre-compilation (optimizer: %d, params: %lu, adam_state: %d, sgd_state: %d)", 
                  engine->config.optimizer_type, [engine->allWeightPlaceholders count], 
                  engine->adamStateInitialized, engine->sgdStateInitialized);
        }
        
        // Note: We no longer need to create separate convolution output placeholders
        // since we're using actual MPSGraph convolution operations now
        
        return YES;
        
    } @catch (NSException* exception) {
        NSLog(@"❌ Exception building dynamic graph: %@", exception.reason);
        NSLog(@"❌ Exception name: %@", exception.name);
        NSLog(@"❌ Exception stack trace: %@", [exception callStackSymbols]);
        return NO;
    }
}

// Add Dense layer to dynamic graph
MPSGraphTensor* addDenseLayerToGraph(MPSGraph* graph,
                                     MPSGraphTensor* input,
                                     layer_spec_c_t* layerSpec,
                                     int layerIdx,
                                     NSMutableArray* allParameterPlaceholders) {
    
    if (layerSpec->param_int_count < 2) {
        NSLog(@"Dense layer missing required parameters (input_size, output_size)");
        return nil;
    }
    
    int inputSize = layerSpec->param_int[0];
    int outputSize = layerSpec->param_int[1];
    BOOL useBias = layerSpec->param_int_count > 2 ? (layerSpec->param_int[2] != 0) : YES;
    
    // Create weight placeholder and add to ordered array
    NSArray<NSNumber*>* weightShape = @[@(inputSize), @(outputSize)];
    MPSGraphTensor* weightTensor = [graph placeholderWithShape:weightShape
                                                      dataType:MPSDataTypeFloat32
                                                          name:[NSString stringWithFormat:@"dense_%d_weight", layerIdx]];
    [allParameterPlaceholders addObject:weightTensor];
    
    // Matrix multiplication
    MPSGraphTensor* output = [graph matrixMultiplicationWithPrimaryTensor:input
                                                          secondaryTensor:weightTensor
                                                                     name:[NSString stringWithFormat:@"dense_%d_matmul", layerIdx]];
    
    // Add bias if enabled - reshape bias for broadcasting compatibility
    if (useBias) {
        NSArray<NSNumber*>* biasShape = @[@(outputSize)];
        MPSGraphTensor* biasTensor = [graph placeholderWithShape:biasShape
                                                        dataType:MPSDataTypeFloat32
                                                            name:[NSString stringWithFormat:@"dense_%d_bias", layerIdx]];
        [allParameterPlaceholders addObject:biasTensor];
        
        // Reshape bias from [output_size] to [1, output_size] for broadcasting compatibility
        // This ensures compatibility with output tensor shape [batch_size, output_size]
        NSArray<NSNumber*>* broadcastBiasShape = @[@1, @(outputSize)];
        MPSGraphTensor* reshapedBias = [graph reshapeTensor:biasTensor
                                                  withShape:broadcastBiasShape
                                                       name:[NSString stringWithFormat:@"dense_%d_bias_reshaped", layerIdx]];
        
        output = [graph additionWithPrimaryTensor:output
                                  secondaryTensor:reshapedBias
                                             name:[NSString stringWithFormat:@"dense_%d_add_bias", layerIdx]];
        // NSLog(@"✅ Dense bias addition created successfully with broadcasting shape %@", broadcastBiasShape);
    }
    
    return output;
}

// Add Conv2D layer to dynamic graph
MPSGraphTensor* addConv2DLayerToGraph(MPSGraph* graph,
                                      MPSGraphTensor* input,
                                      layer_spec_c_t* layerSpec,
                                      int layerIdx,
                                      NSMutableArray* allParameterPlaceholders) {
    
    if (layerSpec->param_int_count < 5) {
        NSLog(@"Conv2D layer missing required parameters");
        return nil;
    }
    
    int inputChannels = layerSpec->param_int[0];
    int outputChannels = layerSpec->param_int[1];
    int kernelSize = layerSpec->param_int[2];
    int stride = layerSpec->param_int[3];
    int padding = layerSpec->param_int[4];
    BOOL useBias = layerSpec->param_int_count > 5 ? (layerSpec->param_int[5] != 0) : YES;
    
    // Debug: DYNAMIC ENGINE: Using MPSGraph convolution with corrected layout for Conv2D layer
    // Parameters: in_ch, out_ch, kernel, stride, padding, bias
    
    // Create weight placeholder [outputChannels, inputChannels, kernelSize, kernelSize] to match Go tensor creation
    NSArray<NSNumber*>* weightShape = @[@(outputChannels), @(inputChannels), @(kernelSize), @(kernelSize)];
    MPSGraphTensor* weightTensor = [graph placeholderWithShape:weightShape
                                                      dataType:MPSDataTypeFloat32
                                                          name:[NSString stringWithFormat:@"conv_%d_weight", layerIdx]];
    [allParameterPlaceholders addObject:weightTensor];
    
    // Add bias placeholder if enabled - IMPORTANT: add bias immediately after weight to match Go parameter order
    MPSGraphTensor* biasTensor = nil;
    if (useBias) {
        NSArray<NSNumber*>* biasShape = @[@(outputChannels)];
        biasTensor = [graph placeholderWithShape:biasShape
                                        dataType:MPSDataTypeFloat32
                                            name:[NSString stringWithFormat:@"conv_%d_bias", layerIdx]];
        [allParameterPlaceholders addObject:biasTensor];
    }
    
    // Log input details - note that MPSGraph tensors may have null shapes during graph construction
    NSArray<NSNumber*>* inputShape = input.shape;
    // Debug: Input tensor shape: (may be null during graph construction)
    // Debug: Weight tensor shape
    
    // For intermediate tensors in graph construction, shape may be null
    // We'll proceed with the convolution operation regardless
    
    // Use MPSGraph convolution with explicit data layout specification
    MPSGraphConvolution2DOpDescriptor* convDesc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    convDesc.strideInX = stride;
    convDesc.strideInY = stride;
    convDesc.paddingLeft = padding;
    convDesc.paddingRight = padding;
    convDesc.paddingTop = padding;
    convDesc.paddingBottom = padding;
    convDesc.dilationRateInX = 1;
    convDesc.dilationRateInY = 1;
    convDesc.groups = 1;
    
    // Explicitly set data layout to ensure compatibility
    convDesc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;     // Input: [N, C, H, W]
    convDesc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;  // Weights: [O, I, H, W]
    
    // Debug: Convolution descriptor: stride, padding, dataLayout=NCHW, weightsLayout=OIHW
    
    // Perform MPSGraph convolution operation
    MPSGraphTensor* convResult;
    @try {
        convResult = [graph convolution2DWithSourceTensor:input
                                             weightsTensor:weightTensor
                                                descriptor:convDesc
                                                      name:[NSString stringWithFormat:@"conv_%d", layerIdx]];
        
        // Debug: Conv2D operation created successfully
        
    } @catch (NSException* exception) {
        NSLog(@"❌ Conv2D operation failed: %@", exception.reason);
        return nil;
    }
    
    // Add bias if enabled - reshape bias for broadcasting compatibility
    if (useBias && biasTensor) {
        // Reshape bias from [output_channels] to [1, output_channels, 1, 1] for NCHW broadcasting
        NSArray<NSNumber*>* broadcastBiasShape = @[@1, @(outputChannels), @1, @1];
        MPSGraphTensor* reshapedBias = [graph reshapeTensor:biasTensor
                                                  withShape:broadcastBiasShape
                                                       name:[NSString stringWithFormat:@"conv_%d_bias_reshaped", layerIdx]];
        
        convResult = [graph additionWithPrimaryTensor:convResult
                                      secondaryTensor:reshapedBias
                                                 name:[NSString stringWithFormat:@"conv_%d_add_bias", layerIdx]];
        // Debug: Bias addition created successfully with broadcasting shape
    }
    
    // Debug: Conv2D layer created with MPSGraph convolution and output shape
    
    return convResult;
}

// Add Batch Normalization layer to dynamic graph
MPSGraphTensor* addBatchNormLayerToGraph(MPSGraph* graph,
                                       MPSGraphTensor* input,
                                       layer_spec_c_t* layerSpec,
                                       int layerIdx,
                                       NSMutableArray* allParameterPlaceholders,
                                       training_engine_t* engine) {
    
    // Extract BatchNorm parameters: [eps, momentum] in floats, [num_features, affine, track_running_stats, training] in ints
    if (layerSpec->param_float_count < 2 || layerSpec->param_int_count < 4) {
        NSLog(@"BatchNorm layer missing required parameters (need: float_count>=2, int_count>=4)");
        return nil;
    }
    
    float eps = layerSpec->param_float[0];
    float momentum = layerSpec->param_float[1];  // Not used in MPSGraph directly - managed by trainer
    int numFeatures = layerSpec->param_int[0];
    BOOL affine = layerSpec->param_int[1] != 0;
    BOOL trackRunningStats = layerSpec->param_int[2] != 0;  // Always true in our implementation
    BOOL training = layerSpec->param_int[3] != 0;
    int inputShapeLen = layerSpec->input_shape_len;  // Declare early for use throughout function
    
    MPSGraphTensor* normalizedTensor = input;
    
    if (affine) {
        // Create learnable scale (gamma) and shift (beta) parameters
        // Both have shape [num_features] matching the feature dimension
        NSArray<NSNumber*>* paramShape = @[@(numFeatures)];
        
        // Scale parameter (gamma) - initialized to 1.0
        MPSGraphTensor* scaleTensor = [graph placeholderWithShape:paramShape
                                                         dataType:MPSDataTypeFloat32
                                                             name:[NSString stringWithFormat:@"batchnorm_%d_scale", layerIdx]];
        [allParameterPlaceholders addObject:scaleTensor];
        
        // Shift parameter (beta) - initialized to 0.0
        MPSGraphTensor* shiftTensor = [graph placeholderWithShape:paramShape
                                                         dataType:MPSDataTypeFloat32
                                                             name:[NSString stringWithFormat:@"batchnorm_%d_shift", layerIdx]];
        [allParameterPlaceholders addObject:shiftTensor];
        
        // Reshape gamma and beta for proper broadcasting with 4D inputs
        // For 4D input [N, C, H, W], gamma and beta should be [1, C, 1, 1]
        // For 2D input [N, C], gamma and beta should be [1, C]
        MPSGraphTensor* reshapedScaleTensor = scaleTensor;
        MPSGraphTensor* reshapedShiftTensor = shiftTensor;
        
        if (inputShapeLen == 4) {
            // Reshape from [C] to [1, C, 1, 1] for 4D broadcasting
            NSArray<NSNumber*>* broadcastShape = @[@1, @(numFeatures), @1, @1];
            reshapedScaleTensor = [graph reshapeTensor:scaleTensor
                                             withShape:broadcastShape
                                                  name:[NSString stringWithFormat:@"batchnorm_%d_scale_reshaped", layerIdx]];
            reshapedShiftTensor = [graph reshapeTensor:shiftTensor
                                             withShape:broadcastShape
                                                  name:[NSString stringWithFormat:@"batchnorm_%d_shift_reshaped", layerIdx]];
        } else if (inputShapeLen == 2) {
            // Reshape from [C] to [1, C] for 2D broadcasting
            NSArray<NSNumber*>* broadcastShape = @[@1, @(numFeatures)];
            reshapedScaleTensor = [graph reshapeTensor:scaleTensor
                                             withShape:broadcastShape
                                                  name:[NSString stringWithFormat:@"batchnorm_%d_scale_reshaped", layerIdx]];
            reshapedShiftTensor = [graph reshapeTensor:shiftTensor
                                             withShape:broadcastShape
                                                  name:[NSString stringWithFormat:@"batchnorm_%d_shift_reshaped", layerIdx]];
        }
        
        if (training) {
            // Training mode: Use batch statistics
            // MPSGraph will compute mean and variance from the current batch
            
            // Determine normalization axes based on layer input shape specification
            // Use the layer spec input shape since MPSGraph tensor shape may not be available at build time
            NSMutableArray<NSNumber*>* axesArray = [NSMutableArray array];
            
            if (inputShapeLen == 4) {
                // Conv layer output [N, C, H, W] - normalize over N, H, W dimensions
                [axesArray addObject:@0];
                [axesArray addObject:@2];
                [axesArray addObject:@3];
            } else if (inputShapeLen == 2) {
                // Dense layer output [N, C] - normalize over N dimension
                [axesArray addObject:@0];
            } else {
                NSLog(@"BatchNorm: Unsupported input shape rank: %d", inputShapeLen);
                return nil;
            }
            
            
            // Use MPSGraph's built-in batch normalization for proper gradient computation
            // For training mode, we need to compute mean and variance from the batch
            // Then use normalization with those computed statistics
            
            // Compute mean and variance from batch using the correct axes
            MPSGraphTensor* meanTensor = [graph meanOfTensor:input
                                                        axes:axesArray
                                                        name:[NSString stringWithFormat:@"batchnorm_%d_mean", layerIdx]];
            MPSGraphTensor* varianceTensor = [graph varianceOfTensor:input
                                                          meanTensor:meanTensor
                                                                axes:axesArray
                                                                name:[NSString stringWithFormat:@"batchnorm_%d_variance", layerIdx]];
            
            // Apply normalization with computed statistics and learnable parameters
            normalizedTensor = [graph normalizationWithTensor:input
                                                   meanTensor:meanTensor          // Use computed batch mean
                                               varianceTensor:varianceTensor      // Use computed batch variance
                                                  gammaTensor:reshapedScaleTensor // Use properly shaped scale
                                                   betaTensor:reshapedShiftTensor // Use properly shaped shift
                                                      epsilon:eps
                                                         name:[NSString stringWithFormat:@"batchnorm_%d", layerIdx]];
        } else {
            // Inference mode: Use running statistics as constants (standard for inference)
            // For inference, running statistics are fixed values, not variables needing feeding
            NSArray<NSNumber*>* statsShape = @[@(numFeatures)];
            
            // UNIFIED SOLUTION: Use constants with default values for inference mode
            // This avoids MPSGraph placeholder issues during inference execution
            
            // ARCHITECTURAL FIX: Use actual running statistics from layer specification
            // This fixes the limitation where inference mode used ONNX defaults instead of trained values
            
            // Create NSData from float arrays for MPSGraph constants
            float* meanData = (float*)malloc(numFeatures * sizeof(float));
            float* varData = (float*)malloc(numFeatures * sizeof(float));
            
            if (layerSpec->has_running_stats && layerSpec->running_mean && layerSpec->running_var) {
                // Use actual trained running statistics from the model
                for (int i = 0; i < numFeatures && i < layerSpec->running_stats_size; i++) {
                    meanData[i] = layerSpec->running_mean[i];
                    varData[i] = layerSpec->running_var[i];
                }
            } else {
                // Fallback to ONNX standard defaults if running stats not available
                for (int i = 0; i < numFeatures; i++) {
                    meanData[i] = 0.0f;  // ONNX standard default: Running mean = 0
                    varData[i] = 1.0f;   // ONNX standard default: Running variance = 1
                }
            }
            
            NSData* meanNSData = [NSData dataWithBytes:meanData length:numFeatures * sizeof(float)];
            NSData* varNSData = [NSData dataWithBytes:varData length:numFeatures * sizeof(float)];
            
            free(meanData);
            free(varData);
            
            MPSGraphTensor* runningMeanTensor = [graph constantWithData:meanNSData
                                                                  shape:statsShape
                                                               dataType:MPSDataTypeFloat32];
            
            MPSGraphTensor* runningVarTensor = [graph constantWithData:varNSData
                                                                 shape:statsShape
                                                              dataType:MPSDataTypeFloat32];
            
            // Reshape running mean and variance for proper broadcasting with 4D inputs
            // Same reshaping logic as gamma/beta tensors
            MPSGraphTensor* reshapedRunningMeanTensor = runningMeanTensor;
            MPSGraphTensor* reshapedRunningVarTensor = runningVarTensor;
            
            if (inputShapeLen == 4) {
                // Reshape from [C] to [1, C, 1, 1] for 4D broadcasting
                NSArray<NSNumber*>* broadcastShape = @[@1, @(numFeatures), @1, @1];
                reshapedRunningMeanTensor = [graph reshapeTensor:runningMeanTensor
                                                       withShape:broadcastShape
                                                            name:[NSString stringWithFormat:@"batchnorm_%d_running_mean_reshaped", layerIdx]];
                reshapedRunningVarTensor = [graph reshapeTensor:runningVarTensor
                                                      withShape:broadcastShape
                                                           name:[NSString stringWithFormat:@"batchnorm_%d_running_var_reshaped", layerIdx]];
            } else if (inputShapeLen == 2) {
                // Reshape from [C] to [1, C] for 2D broadcasting
                NSArray<NSNumber*>* broadcastShape = @[@1, @(numFeatures)];
                reshapedRunningMeanTensor = [graph reshapeTensor:runningMeanTensor
                                                       withShape:broadcastShape
                                                            name:[NSString stringWithFormat:@"batchnorm_%d_running_mean_reshaped", layerIdx]];
                reshapedRunningVarTensor = [graph reshapeTensor:runningVarTensor
                                                      withShape:broadcastShape
                                                           name:[NSString stringWithFormat:@"batchnorm_%d_running_var_reshaped", layerIdx]];
            }
            
            // Use MPSGraph's built-in batch normalization with running statistics
            normalizedTensor = [graph normalizationWithTensor:input
                                                   meanTensor:reshapedRunningMeanTensor  // Use properly shaped running mean
                                               varianceTensor:reshapedRunningVarTensor   // Use properly shaped running variance
                                                  gammaTensor:reshapedScaleTensor        // Use properly shaped scale
                                                   betaTensor:reshapedShiftTensor        // Use properly shaped shift
                                                      epsilon:eps
                                                         name:[NSString stringWithFormat:@"batchnorm_%d", layerIdx]];
        }
    } else {
        // No affine transformation - just normalize without learnable parameters
        if (training) {
            // Training mode: compute batch statistics
            // Use layer spec input shape
            NSMutableArray<NSNumber*>* axesArray = [NSMutableArray array];
            
            if (inputShapeLen == 4) {
                [axesArray addObject:@0];
                [axesArray addObject:@2];
                [axesArray addObject:@3];
            } else if (inputShapeLen == 2) {
                [axesArray addObject:@0];
            } else {
                NSLog(@"BatchNorm: Unsupported input shape rank: %d", inputShapeLen);
                return nil;
            }
            
            // For non-affine BatchNorm, compute statistics and normalize without learnable parameters
            // Compute mean and variance from batch
            MPSGraphTensor* meanTensor = [graph meanOfTensor:input
                                                        axes:axesArray
                                                        name:[NSString stringWithFormat:@"batchnorm_%d_mean", layerIdx]];
            MPSGraphTensor* varianceTensor = [graph varianceOfTensor:input
                                                          meanTensor:meanTensor
                                                                axes:axesArray
                                                                name:[NSString stringWithFormat:@"batchnorm_%d_variance", layerIdx]];
            
            // Create identity scale and zero shift tensors for no affine transformation
            // Shape them correctly for broadcasting
            NSArray<NSNumber*>* broadcastShape;
            if (inputShapeLen == 4) {
                broadcastShape = @[@1, @(numFeatures), @1, @1]; // [1, C, 1, 1] for 4D
            } else {
                broadcastShape = @[@1, @(numFeatures)]; // [1, C] for 2D
            }
            
            MPSGraphTensor* identityScale = [graph constantWithScalar:1.0f
                                                                shape:broadcastShape
                                                             dataType:MPSDataTypeFloat32];
            MPSGraphTensor* zeroShift = [graph constantWithScalar:0.0f
                                                            shape:broadcastShape
                                                         dataType:MPSDataTypeFloat32];
            
            normalizedTensor = [graph normalizationWithTensor:input
                                                   meanTensor:meanTensor
                                               varianceTensor:varianceTensor
                                                  gammaTensor:identityScale  // Identity scale
                                                   betaTensor:zeroShift      // Zero shift
                                                      epsilon:eps
                                                         name:[NSString stringWithFormat:@"batchnorm_%d_no_affine", layerIdx]];
        } else {
            // Inference mode without affine parameters - use constant running statistics
            NSArray<NSNumber*>* statsShape = @[@(numFeatures)];
            
            // UNIFIED SOLUTION: Use constants for non-affine inference mode as well
            
            // ARCHITECTURAL FIX: Use actual running statistics for non-affine BatchNorm too
            // This fixes the same limitation as the affine case above
            
            // Create NSData from float arrays for MPSGraph constants (non-affine case)
            float* meanData = (float*)malloc(numFeatures * sizeof(float));
            float* varData = (float*)malloc(numFeatures * sizeof(float));
            
            if (layerSpec->has_running_stats && layerSpec->running_mean && layerSpec->running_var) {
                // Use actual trained running statistics from the model
                for (int i = 0; i < numFeatures && i < layerSpec->running_stats_size; i++) {
                    meanData[i] = layerSpec->running_mean[i];
                    varData[i] = layerSpec->running_var[i];
                }
            } else {
                // Fallback to ONNX standard defaults if running stats not available
                for (int i = 0; i < numFeatures; i++) {
                    meanData[i] = 0.0f;  // ONNX standard default: Running mean = 0
                    varData[i] = 1.0f;   // ONNX standard default: Running variance = 1
                }
            }
            
            NSData* meanNSData = [NSData dataWithBytes:meanData length:numFeatures * sizeof(float)];
            NSData* varNSData = [NSData dataWithBytes:varData length:numFeatures * sizeof(float)];
            
            free(meanData);
            free(varData);
            
            MPSGraphTensor* runningMeanTensor = [graph constantWithData:meanNSData
                                                                  shape:statsShape
                                                               dataType:MPSDataTypeFloat32];
            
            MPSGraphTensor* runningVarTensor = [graph constantWithData:varNSData
                                                                 shape:statsShape
                                                              dataType:MPSDataTypeFloat32];
            
            // Create identity scale and zero shift tensors for no affine transformation
            // Shape them correctly for broadcasting
            NSArray<NSNumber*>* broadcastShape;
            if (inputShapeLen == 4) {
                broadcastShape = @[@1, @(numFeatures), @1, @1]; // [1, C, 1, 1] for 4D
            } else {
                broadcastShape = @[@1, @(numFeatures)]; // [1, C] for 2D
            }
            
            MPSGraphTensor* identityScale = [graph constantWithScalar:1.0f
                                                                shape:broadcastShape
                                                             dataType:MPSDataTypeFloat32];
            MPSGraphTensor* zeroShift = [graph constantWithScalar:0.0f
                                                            shape:broadcastShape
                                                         dataType:MPSDataTypeFloat32];
            
            normalizedTensor = [graph normalizationWithTensor:input
                                                   meanTensor:runningMeanTensor
                                               varianceTensor:runningVarTensor
                                                  gammaTensor:identityScale     // Identity scale
                                                   betaTensor:zeroShift         // Zero shift
                                                      epsilon:eps
                                                         name:[NSString stringWithFormat:@"batchnorm_%d_no_affine", layerIdx]];
        }
    }
    
    return normalizedTensor;
}