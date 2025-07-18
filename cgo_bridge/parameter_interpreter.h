#ifndef PARAMETER_INTERPRETER_H
#define PARAMETER_INTERPRETER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "bridge_types.h"

// Parameter format types
typedef enum {
    PARAM_FORMAT_STANDARD,      // Standard [out_ch, in_ch, kh, kw] format
    PARAM_FORMAT_COMPACT,       // Compact channel-wise encoding
    PARAM_FORMAT_COMPRESSED,    // Compressed/quantized format
    PARAM_FORMAT_FACTORIZED,    // Factorized/decomposed format
    PARAM_FORMAT_UNKNOWN        // Unable to determine format
} ParameterFormat;

// Parameter interpretation result
typedef struct {
    ParameterFormat format;
    int channels_in;
    int channels_out;
    int kernel_height;
    int kernel_width;
    float confidence;           // 0.0 to 1.0, higher = more confident
    const char* reasoning;      // Human-readable explanation
} ParameterInterpretation;

// Layer architecture information
typedef struct {
    int layer_type;            // From layer_spec_c_t
    int input_channels;
    int output_channels;
    int kernel_size;
    bool channels_verified;    // Has this been validated against actual data?
    bool is_dynamic;          // Can this layer adapt to different input sizes?
    ParameterFormat param_format;
    float interpretation_confidence;
} LayerArchInfo;

// Model architecture analysis
typedef struct {
    LayerArchInfo* layers;
    int layer_count;
    int input_channels;        // Model input channels (e.g., 3 for RGB)
    bool architecture_valid;   // Is the architecture consistent?
    float overall_confidence;  // Overall confidence in the interpretation
} ModelArchitecture;

// Core parameter interpretation functions
ParameterInterpretation interpretConvParameter(float* data, int size, 
                                             int expected_in, int expected_out);
ParameterInterpretation tryStandardFormat(float* data, int size, 
                                        int expected_in, int expected_out);
ParameterInterpretation tryCompactFormat(float* data, int size);
ParameterInterpretation tryCompressedFormat(float* data, int size);
ParameterInterpretation tryFactorizedFormat(float* data, int size, 
                                          int expected_in, int expected_out);

// Model architecture analysis
ModelArchitecture analyzeModelArchitecture(layer_spec_c_t* layers, int layerCount,
                                          float** parameters, int* paramSizes, int paramCount);
int getActualInputChannels(ModelArchitecture* arch, int layerIdx);
int getActualOutputChannels(ModelArchitecture* arch, int layerIdx);
bool isCompatibleWithDedicatedEngine(ModelArchitecture* arch);

// Parameter format conversion
bool convertParameterFormat(float* src_data, int src_size, ParameterFormat src_format,
                          float** dst_data, int* dst_size, ParameterFormat dst_format,
                          int channels_in, int channels_out, int kernel_h, int kernel_w);

// Utility functions
float calculateInterpretationConfidence(ParameterInterpretation* interp, 
                                      float* data, int size);
void printModelArchitecture(ModelArchitecture* arch);
void freeModelArchitecture(ModelArchitecture* arch);

// Input channel detection
int detectInputChannels(layer_spec_c_t* firstLayer, float* firstParam, int firstParamSize);

#endif // PARAMETER_INTERPRETER_H