#include "parameter_interpreter.h"
#include <string.h>

// Helper macros
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// Helper function to check if a number is a perfect square
static bool isPerfectSquare(int n, int* root) {
    if (n < 0) return false;
    if (n == 0) {
        *root = 0;
        return true;
    }
    
    int r = (int)sqrt(n);
    if (r * r == n) {
        *root = r;
        return true;
    }
    return false;
}

// Helper function to find all divisors of a number
static void findDivisors(int n, int* divisors, int* count) {
    *count = 0;
    for (int i = 1; i <= n && *count < 20; i++) {
        if (n % i == 0) {
            divisors[(*count)++] = i;
        }
    }
}

// Try to interpret parameter as standard [out_ch, in_ch, kh, kw] format
ParameterInterpretation tryStandardFormat(float* data, int size, 
                                        int expected_in, int expected_out) {
    ParameterInterpretation result = {
        .format = PARAM_FORMAT_STANDARD,
        .channels_in = expected_in,
        .channels_out = expected_out,
        .kernel_height = 1,
        .kernel_width = 1,
        .confidence = 0.0,
        .reasoning = "Standard format analysis"
    };
    
    printf("  → Trying standard format: %d floats for [%d, %d, ?, ?]\n", 
           size, expected_out, expected_in);
    
    if (expected_in <= 0 || expected_out <= 0) {
        printf("    ✗ Invalid expected channels\n");
        return result;
    }
    
    // Calculate kernel size: size = out_ch * in_ch * kh * kw
    int kernel_area = size / (expected_out * expected_in);
    
    if (kernel_area <= 0 || size % (expected_out * expected_in) != 0) {
        printf("    ✗ Size %d not divisible by (%d * %d)\n", 
               size, expected_out, expected_in);
        return result;
    }
    
    // Try square kernels first
    int kernel_size;
    if (isPerfectSquare(kernel_area, &kernel_size)) {
        result.kernel_height = kernel_size;
        result.kernel_width = kernel_size;
        result.confidence = 0.9;
        printf("    ✓ Square kernel: %dx%d, confidence=%.2f\n", 
               kernel_size, kernel_size, result.confidence);
        return result;
    }
    
    // Try rectangular kernels
    int divisors[20];
    int divisor_count;
    findDivisors(kernel_area, divisors, &divisor_count);
    
    if (divisor_count > 0) {
        // Pick the most balanced rectangle
        int best_h = 1, best_w = kernel_area;
        int best_diff = abs(best_h - best_w);
        
        for (int i = 0; i < divisor_count; i++) {
            int h = divisors[i];
            int w = kernel_area / h;
            int diff = abs(h - w);
            if (diff < best_diff) {
                best_h = h;
                best_w = w;
                best_diff = diff;
            }
        }
        
        result.kernel_height = best_h;
        result.kernel_width = best_w;
        result.confidence = 0.7;
        printf("    ✓ Rectangular kernel: %dx%d, confidence=%.2f\n", 
               best_h, best_w, result.confidence);
        return result;
    }
    
    printf("    ✗ Cannot determine kernel size from area %d\n", kernel_area);
    return result;
}

// Try to interpret parameter as compact format
ParameterInterpretation tryCompactFormat(float* data, int size) {
    ParameterInterpretation result = {
        .format = PARAM_FORMAT_COMPACT,
        .channels_in = 0,
        .channels_out = 0,
        .kernel_height = 1,
        .kernel_width = 1,
        .confidence = 0.0,
        .reasoning = "Compact format analysis"
    };
    
    printf("  → Trying compact format: %d floats\n", size);
    
    // Common compact patterns
    
    // Pattern 1: Per-output-channel weights (size = output_channels)
    if (size >= 1 && size <= 1024) {
        result.channels_out = size;
        result.channels_in = -1;  // Will be determined from context
        result.confidence = 0.6;
        printf("    ✓ Per-output-channel pattern: %d output channels\n", size);
        return result;
    }
    
    // Pattern 2: Channel-pair weights (size = output_channels * input_channels)
    // Try common input channel counts
    int common_inputs[] = {1, 3, 16, 32, 64, 128, 256, 512};
    int num_common = sizeof(common_inputs) / sizeof(int);
    
    for (int i = 0; i < num_common; i++) {
        int in_ch = common_inputs[i];
        if (size % in_ch == 0) {
            int out_ch = size / in_ch;
            if (out_ch > 0 && out_ch <= 1024) {
                result.channels_in = in_ch;
                result.channels_out = out_ch;
                result.confidence = 0.5 + (0.2 * (i < 3 ? 1.0 : 0.5));  // Prefer 1, 3, 16
                printf("    ✓ Channel-pair pattern: %d→%d channels\n", in_ch, out_ch);
                return result;
            }
        }
    }
    
    printf("    ✗ No compact pattern found\n");
    return result;
}

// Try to interpret parameter as compressed format
ParameterInterpretation tryCompressedFormat(float* data, int size) {
    ParameterInterpretation result = {
        .format = PARAM_FORMAT_COMPRESSED,
        .confidence = 0.0,
        .reasoning = "Compressed format analysis"
    };
    
    printf("  → Trying compressed format: %d floats\n", size);
    
    // Look for compression indicators
    if (size < 100) {
        // Small size suggests compression
        result.confidence = 0.4;
        printf("    ⚠ Small size suggests compression, confidence=%.2f\n", result.confidence);
    }
    
    // Analyze value distribution for compression patterns
    float min_val = data[0], max_val = data[0];
    int zero_count = 0, unique_count = 0;
    
    for (int i = 0; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        if (data[i] == 0.0f) zero_count++;
    }
    
    float range = max_val - min_val;
    float sparsity = (float)zero_count / size;
    
    if (sparsity > 0.5) {
        result.confidence += 0.3;
        printf("    ✓ High sparsity (%.1f%%) suggests compression\n", sparsity * 100);
    }
    
    if (range < 2.0 && range > 0.01) {
        result.confidence += 0.2;
        printf("    ✓ Limited range (%.3f) suggests quantization\n", range);
    }
    
    printf("    → Compressed format confidence: %.2f\n", result.confidence);
    return result;
}

// Try to interpret parameter as factorized format
ParameterInterpretation tryFactorizedFormat(float* data, int size, 
                                          int expected_in, int expected_out) {
    ParameterInterpretation result = {
        .format = PARAM_FORMAT_FACTORIZED,
        .confidence = 0.0,
        .reasoning = "Factorized format analysis"
    };
    
    printf("  → Trying factorized format: %d floats for %d→%d\n", 
           size, expected_in, expected_out);
    
    if (expected_in <= 0 || expected_out <= 0) {
        return result;
    }
    
    // Common factorization patterns
    
    // Pattern 1: Separable convolution (depthwise + pointwise)
    int depthwise_size = expected_in * 9;  // 3x3 depthwise
    int pointwise_size = expected_in * expected_out;  // 1x1 pointwise
    
    if (size == depthwise_size + pointwise_size) {
        result.channels_in = expected_in;
        result.channels_out = expected_out;
        result.kernel_height = 3;
        result.kernel_width = 3;
        result.confidence = 0.8;
        printf("    ✓ Separable convolution pattern (3x3 depthwise + 1x1 pointwise)\n");
        return result;
    }
    
    // Pattern 2: Low-rank factorization
    // A = UV where U is out_ch x rank, V is rank x in_ch
    for (int rank = 1; rank <= min(expected_in, expected_out); rank++) {
        int factorized_size = expected_out * rank + rank * expected_in;
        if (size == factorized_size) {
            result.channels_in = expected_in;
            result.channels_out = expected_out;
            result.kernel_height = 1;
            result.kernel_width = 1;
            result.confidence = 0.7;
            printf("    ✓ Low-rank factorization pattern (rank=%d)\n", rank);
            return result;
        }
    }
    
    printf("    ✗ No factorization pattern found\n");
    return result;
}

// Main parameter interpretation function
ParameterInterpretation interpretConvParameter(float* data, int size, 
                                             int expected_in, int expected_out) {
    printf("🔍 Interpreting conv parameter: %d floats, expecting %d→%d channels\n", 
           size, expected_in, expected_out);
    
    ParameterInterpretation candidates[4];
    int candidate_count = 0;
    
    // Try all interpretation methods
    candidates[candidate_count++] = tryStandardFormat(data, size, expected_in, expected_out);
    candidates[candidate_count++] = tryCompactFormat(data, size);
    candidates[candidate_count++] = tryCompressedFormat(data, size);
    candidates[candidate_count++] = tryFactorizedFormat(data, size, expected_in, expected_out);
    
    // Find the best interpretation
    ParameterInterpretation best = candidates[0];
    for (int i = 1; i < candidate_count; i++) {
        if (candidates[i].confidence > best.confidence) {
            best = candidates[i];
        }
    }
    
    printf("📊 Best interpretation: %s (confidence=%.2f)\n", 
           best.reasoning, best.confidence);
    
    return best;
}

// Detect input channels from first layer
int detectInputChannels(layer_spec_c_t* firstLayer, float* firstParam, int firstParamSize) {
    printf("🔍 Detecting input channels from first layer\n");
    
    // Try common input channel counts
    int common_channels[] = {1, 3, 4};  // Grayscale, RGB, RGBA
    int num_common = sizeof(common_channels) / sizeof(int);
    
    for (int i = 0; i < num_common; i++) {
        int test_channels = common_channels[i];
        int expected_out = firstLayer->output_shape[1];
        
        if (expected_out > 0) {
            ParameterInterpretation interp = interpretConvParameter(
                firstParam, firstParamSize, test_channels, expected_out);
            
            if (interp.confidence > 0.7) {
                printf("✓ Detected %d input channels (confidence=%.2f)\n", 
                       test_channels, interp.confidence);
                return test_channels;
            }
        }
    }
    
    // Fallback to RGB
    printf("⚠ Unable to detect input channels, defaulting to 3 (RGB)\n");
    return 3;
}

// Get actual input channels for a layer
int getActualInputChannels(ModelArchitecture* arch, int layerIdx) {
    if (layerIdx == 0) {
        return arch->input_channels;
    }
    
    // Find the most recent layer that produces channels
    for (int i = layerIdx - 1; i >= 0; i--) {
        if (arch->layers[i].layer_type == 1 && arch->layers[i].channels_verified) {  // Conv2D
            return arch->layers[i].output_channels;
        }
    }
    
    // Fallback to declared channels
    return arch->layers[layerIdx].input_channels;
}

// Get actual output channels for a layer
int getActualOutputChannels(ModelArchitecture* arch, int layerIdx) {
    if (layerIdx >= 0 && layerIdx < arch->layer_count) {
        return arch->layers[layerIdx].output_channels;
    }
    return 0;
}

// Check if model is compatible with dedicated engine
bool isCompatibleWithDedicatedEngine(ModelArchitecture* arch) {
    if (!arch->architecture_valid) {
        return false;
    }
    
    // Check if all critical layers have high confidence interpretations
    for (int i = 0; i < arch->layer_count; i++) {
        if (arch->layers[i].layer_type == 1) {  // Conv2D
            if (arch->layers[i].interpretation_confidence < 0.6) {
                printf("⚠ Layer %d has low confidence (%.2f), not compatible with DedicatedEngine\n", 
                       i, arch->layers[i].interpretation_confidence);
                return false;
            }
        }
    }
    
    return true;
}

// Print model architecture for debugging
void printModelArchitecture(ModelArchitecture* arch) {
    printf("\n📋 Model Architecture Analysis:\n");
    printf("   Input channels: %d\n", arch->input_channels);
    printf("   Architecture valid: %s\n", arch->architecture_valid ? "Yes" : "No");
    printf("   Overall confidence: %.2f\n", arch->overall_confidence);
    printf("   Layers:\n");
    
    for (int i = 0; i < arch->layer_count; i++) {
        LayerArchInfo* layer = &arch->layers[i];
        printf("     %2d: Type=%d, %d→%d channels, %dx%d kernel, conf=%.2f %s\n", 
               i, layer->layer_type, layer->input_channels, layer->output_channels,
               layer->kernel_size, layer->kernel_size, layer->interpretation_confidence,
               layer->channels_verified ? "✓" : "?");
    }
    printf("\n");
}

// Analyze complete model architecture
ModelArchitecture analyzeModelArchitecture(layer_spec_c_t* layers, int layerCount,
                                          float** parameters, int* paramSizes, int paramCount) {
    printf("🏗️  Analyzing model architecture: %d layers, %d parameters\n", 
           layerCount, paramCount);
    
    ModelArchitecture arch = {0};
    arch.layers = (LayerArchInfo*)calloc(layerCount, sizeof(LayerArchInfo));
    arch.layer_count = layerCount;
    arch.architecture_valid = true;
    arch.overall_confidence = 0.0;
    
    // First pass: Extract declared architecture
    for (int i = 0; i < layerCount; i++) {
        arch.layers[i].layer_type = layers[i].layer_type;
        arch.layers[i].input_channels = layers[i].input_shape[1];
        arch.layers[i].output_channels = layers[i].output_shape[1];
        arch.layers[i].kernel_size = layers[i].param_int_count > 2 ? layers[i].param_int[2] : 1;
        arch.layers[i].channels_verified = false;
        arch.layers[i].is_dynamic = false;
        arch.layers[i].param_format = PARAM_FORMAT_UNKNOWN;
        arch.layers[i].interpretation_confidence = 0.0;
    }
    
    // Detect input channels from first layer
    arch.input_channels = 3;  // Default to RGB
    if (paramCount > 0) {
        arch.input_channels = detectInputChannels(&layers[0], parameters[0], paramSizes[0]);
    }
    
    // Second pass: Validate against actual parameter data
    int paramIndex = 0;
    float total_confidence = 0.0;
    int confident_layers = 0;
    
    for (int i = 0; i < layerCount && paramIndex < paramCount; i++) {
        LayerArchInfo* layer = &arch.layers[i];
        
        if (layer->layer_type == 1) {  // Conv2D
            int actual_in_channels = getActualInputChannels(&arch, i);
            int expected_out_channels = layer->output_channels;
            
            ParameterInterpretation interp = interpretConvParameter(
                parameters[paramIndex], paramSizes[paramIndex] / sizeof(float),
                actual_in_channels, expected_out_channels);
            
            // Update layer information based on interpretation
            if (interp.confidence > 0.5) {
                if (interp.channels_in > 0) {
                    layer->input_channels = interp.channels_in;
                }
                if (interp.channels_out > 0) {
                    layer->output_channels = interp.channels_out;
                }
                layer->kernel_size = interp.kernel_height;  // Assume square kernels
                layer->channels_verified = true;
                layer->param_format = interp.format;
                layer->interpretation_confidence = interp.confidence;
                
                total_confidence += interp.confidence;
                confident_layers++;
                
                printf("✓ Layer %d verified: %d→%d channels, %dx%d kernel, format=%d\n", 
                       i, layer->input_channels, layer->output_channels, 
                       layer->kernel_size, layer->kernel_size, layer->param_format);
            } else {
                printf("⚠ Layer %d has low confidence interpretation (%.2f)\n", 
                       i, interp.confidence);
                arch.architecture_valid = false;
            }
            
            paramIndex++;  // Conv layers consume one parameter for weights
            
            // Skip bias parameter if present
            if (paramIndex < paramCount && paramSizes[paramIndex] == layer->output_channels * sizeof(float)) {
                paramIndex++;
            }
        } else if (layer->layer_type == 6) {  // BatchNorm
            // BatchNorm layers consume 4 parameters: scale, bias, mean, variance
            int expected_features = getActualInputChannels(&arch, i);
            layer->input_channels = expected_features;
            layer->output_channels = expected_features;
            layer->channels_verified = true;
            layer->interpretation_confidence = 0.9;  // BatchNorm is straightforward
            
            total_confidence += 0.9;
            confident_layers++;
            
            printf("✓ Layer %d (BatchNorm): %d features\n", i, expected_features);
            
            // Skip BatchNorm parameters
            for (int j = 0; j < 4 && paramIndex < paramCount; j++) {
                if (paramSizes[paramIndex] == expected_features * sizeof(float)) {
                    paramIndex++;
                } else {
                    break;
                }
            }
        } else {
            // Other layer types (ReLU, Dropout, etc.) don't change channels
            layer->input_channels = getActualInputChannels(&arch, i);
            layer->output_channels = layer->input_channels;
            layer->channels_verified = true;
            layer->interpretation_confidence = 1.0;  // Pass-through layers are certain
            
            printf("✓ Layer %d (pass-through): %d channels\n", i, layer->input_channels);
        }
    }
    
    // Calculate overall confidence
    arch.overall_confidence = confident_layers > 0 ? total_confidence / confident_layers : 0.0;
    
    printf("📊 Architecture analysis complete: %.2f overall confidence\n", 
           arch.overall_confidence);
    
    return arch;
}

// Convert parameter format
bool convertParameterFormat(float* src_data, int src_size, ParameterFormat src_format,
                          float** dst_data, int* dst_size, ParameterFormat dst_format,
                          int channels_in, int channels_out, int kernel_h, int kernel_w) {
    
    printf("🔄 Converting parameter format: %d → %d (%d floats)\n", 
           src_format, dst_format, src_size);
    
    if (src_format == dst_format) {
        // No conversion needed
        *dst_data = (float*)malloc(src_size * sizeof(float));
        memcpy(*dst_data, src_data, src_size * sizeof(float));
        *dst_size = src_size;
        return true;
    }
    
    if (dst_format == PARAM_FORMAT_STANDARD) {
        // Convert to standard format
        int expected_size = channels_out * channels_in * kernel_h * kernel_w;
        *dst_data = (float*)calloc(expected_size, sizeof(float));
        *dst_size = expected_size;
        
        switch (src_format) {
            case PARAM_FORMAT_COMPACT:
                printf("  Converting compact → standard\n");
                
                if (src_size == channels_out) {
                    // Per-output-channel format
                    for (int oc = 0; oc < channels_out; oc++) {
                        float channel_weight = src_data[oc];
                        for (int ic = 0; ic < channels_in; ic++) {
                            for (int kh = 0; kh < kernel_h; kh++) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    int idx = ((oc * channels_in + ic) * kernel_h + kh) * kernel_w + kw;
                                    (*dst_data)[idx] = channel_weight / (channels_in * kernel_h * kernel_w);
                                }
                            }
                        }
                    }
                } else if (src_size == channels_out * channels_in) {
                    // Channel-pair format
                    for (int oc = 0; oc < channels_out; oc++) {
                        for (int ic = 0; ic < channels_in; ic++) {
                            float pair_weight = src_data[oc * channels_in + ic];
                            for (int kh = 0; kh < kernel_h; kh++) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    int idx = ((oc * channels_in + ic) * kernel_h + kh) * kernel_w + kw;
                                    (*dst_data)[idx] = pair_weight / (kernel_h * kernel_w);
                                }
                            }
                        }
                    }
                } else {
                    // Generic proportional scaling
                    for (int i = 0; i < expected_size; i++) {
                        int src_idx = (i * src_size) / expected_size;
                        (*dst_data)[i] = src_data[src_idx];
                    }
                }
                break;
                
            case PARAM_FORMAT_COMPRESSED:
                printf("  Converting compressed → standard\n");
                // Simple expansion - repeat values
                for (int i = 0; i < expected_size; i++) {
                    int src_idx = i % src_size;
                    (*dst_data)[i] = src_data[src_idx];
                }
                break;
                
            case PARAM_FORMAT_FACTORIZED:
                printf("  Converting factorized → standard\n");
                // This would need more complex logic based on specific factorization
                // For now, use simple expansion
                for (int i = 0; i < expected_size; i++) {
                    int src_idx = i % src_size;
                    (*dst_data)[i] = src_data[src_idx];
                }
                break;
                
            default:
                printf("  ❌ Unsupported source format\n");
                free(*dst_data);
                return false;
        }
        
        printf("  ✓ Converted %d → %d floats\n", src_size, expected_size);
        return true;
    }
    
    printf("  ❌ Unsupported destination format\n");
    return false;
}

// Calculate interpretation confidence based on data analysis
float calculateInterpretationConfidence(ParameterInterpretation* interp, 
                                      float* data, int size) {
    float confidence = interp->confidence;
    
    // Analyze data characteristics
    if (size > 0) {
        float mean = 0.0, variance = 0.0;
        for (int i = 0; i < size; i++) {
            mean += data[i];
        }
        mean /= size;
        
        for (int i = 0; i < size; i++) {
            variance += (data[i] - mean) * (data[i] - mean);
        }
        variance /= size;
        
        // Reasonable weight statistics suggest valid parameters
        if (variance > 0.0001 && variance < 1.0) {
            confidence += 0.1;
        }
        
        // Values in typical neural network weight range
        if (mean > -1.0 && mean < 1.0) {
            confidence += 0.05;
        }
    }
    
    return min(confidence, 1.0);
}

// Free model architecture
void freeModelArchitecture(ModelArchitecture* arch) {
    if (arch && arch->layers) {
        free(arch->layers);
        arch->layers = NULL;
    }
}