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
        // Create input placeholder with fixed batch size (temporarily to debug the channel mismatch)
        NSMutableArray<NSNumber*>* inputShapeNS = [[NSMutableArray alloc] init];
        for (int i = 0; i < inputShapeLen; i++) {
            [inputShapeNS addObject:@(inputShape[i])]; // Use the original shape as-is for now
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
                        NSArray<NSNumber*>* currentShape = currentTensor.shape;
                        int batchSize = [currentShape[0] intValue];
                        int channels = [currentShape[1] intValue];
                        int height = [currentShape[2] intValue];
                        int width = [currentShape[3] intValue];
                        int flattenedSize = channels * height * width;
                        
                        NSArray<NSNumber*>* flattenShape = @[@(batchSize), @(flattenedSize)];
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
        
        // Labels placeholder [batch_size, num_classes] with fixed batch size for now
        NSArray<NSNumber*>* labelShape = @[@(inputShape[0]), @(numClasses)];
        // NSLog(@"🔍 Creating label placeholder with shape: %@", labelShape);
        MPSGraphTensor* labelTensor = [engine->graph placeholderWithShape:labelShape
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"labels"];
        engine->labelTensor = labelTensor;
        
        // CRITICAL FIX: Compute actual cross-entropy loss in the graph for proper gradient flow
        // Store predictions tensor separately for inference
        MPSGraphTensor* predictionsTensor = [engine->graph softMaxWithTensor:currentTensor
                                                                        axis:-1
                                                                        name:@"predictions"];
        
        // Store predictions tensor for inference use
        engine->predictionsTensor = predictionsTensor;
        
        // FIXED: Use MPSGraph's built-in softmax cross-entropy for numerical stability and proper gradients
        // This computes: -mean(sum(labels * log_softmax(logits))) in a numerically stable way
        MPSGraphTensor* actualLoss = [engine->graph softMaxCrossEntropyWithSourceTensor:currentTensor
                                                                           labelsTensor:labelTensor
                                                                                   axis:-1
                                                                         reductionType:MPSGraphLossReductionTypeMean
                                                                                  name:@"cross_entropy_loss"];
        
        // Store ACTUAL LOSS for gradient computation (not predictions)
        engine->lossOutput = actualLoss;
        
        // Store the ordered parameter placeholders in the engine
        // Clear the old arrays and copy from our correctly ordered array
        [engine->allWeightPlaceholders removeAllObjects];
        [engine->allBiasPlaceholders removeAllObjects];
        
        // Store all parameters in the correct order (weight, bias, weight, bias...)
        // This matches exactly how Go CreateParameterTensors works
        for (int i = 0; i < allParameterPlaceholders.count; i++) {
            MPSGraphTensor* placeholder = allParameterPlaceholders[i];
            [engine->allWeightPlaceholders addObject:placeholder];
        }
        
        NSLog(@"✅ Dynamic graph built successfully with %d layers and proper loss computation", numLayers);
        NSLog(@"   - Parameters: %lu placeholders", (unsigned long)allParameterPlaceholders.count);
        NSLog(@"   - Output classes: %d", numClasses);
        NSLog(@"   - Loss: Cross-entropy with automatic differentiation");
        
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
        
        // OPTIMIZER-SPECIFIC PRE-COMPILATION: Build optimizer-specific graphs to avoid conflicts
        // This eliminates runtime operation creation and prevents placeholder issues
        NSLog(@"🔍 PRE-COMPILATION DEBUG: optimizer=%d, params=%lu, adam_state=%d, sgd_state=%d", 
              engine->config.optimizer_type, [engine->allWeightPlaceholders count], engine->adamStateInitialized, engine->sgdStateInitialized);
        
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
            NSLog(@"🔧 DEBUG: SGD using sgdCachedLrTensor with config value: %.6f", engine->config.learning_rate);
            
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
            NSLog(@"🔧 DEBUG: Set precompiledGradientTensors to %lu items", [precompiledGradientTensors count]);
            
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
            NSLog(@"🔧 DEBUG: Set precompiledUpdatedParams to %lu items", [precompiledUpdatedParams count]);
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
        } else {
            NSLog(@"⚠️ PRE-COMPILATION: Skipping pre-compilation (optimizer: %d, params: %lu, adam_state: %d, sgd_state: %d)", 
                  engine->config.optimizer_type, [engine->allWeightPlaceholders count], 
                  engine->adamStateInitialized, engine->sgdStateInitialized);
        }
        
        // Note: We no longer need to create separate convolution output placeholders
        // since we're using actual MPSGraph convolution operations now
        
        return YES;
        
    } @catch (NSException* exception) {
        NSLog(@"Exception building dynamic graph: %@", exception.reason);
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