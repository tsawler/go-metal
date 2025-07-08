#pragma once

#import "bridge_types.h"

// Dynamic graph building functions
BOOL buildDynamicGraphFromLayers(training_engine_t* engine,
                                layer_spec_c_t* layers,
                                int numLayers,
                                int* inputShape,
                                int inputShapeLen);

// Layer creation functions
MPSGraphTensor* addDenseLayerToGraph(MPSGraph* graph,
                                    MPSGraphTensor* input,
                                    layer_spec_c_t* layerSpec,
                                    int layerIdx,
                                    NSMutableArray* allParameterPlaceholders);

MPSGraphTensor* addConv2DLayerToGraph(MPSGraph* graph,
                                     MPSGraphTensor* input,
                                     layer_spec_c_t* layerSpec,
                                     int layerIdx,
                                     NSMutableArray* allParameterPlaceholders);

MPSGraphTensor* addBatchNormLayerToGraph(MPSGraph* graph,
                                       MPSGraphTensor* input,
                                       layer_spec_c_t* layerSpec,
                                       int layerIdx,
                                       NSMutableArray* allParameterPlaceholders,
                                       training_engine_t* engine);