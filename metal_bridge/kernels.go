package metal_bridge

// MSL source code for Metal compute kernels
const MetalKernelSource = `
#include <metal_stdlib>
using namespace metal;

// Simple element-wise addition kernel for float32 tensors
kernel void add_arrays_float32(device const float* inputA [[buffer(0)]],
                              device const float* inputB [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] + inputB[index];
}

// Simple element-wise addition kernel for int32 tensors
kernel void add_arrays_int32(device const int* inputA [[buffer(0)]],
                            device const int* inputB [[buffer(1)]],
                            device int* result [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] + inputB[index];
}

// Element-wise multiplication kernel for float32 tensors
kernel void mul_arrays_float32(device const float* inputA [[buffer(0)]],
                              device const float* inputB [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] * inputB[index];
}

// Element-wise multiplication kernel for int32 tensors
kernel void mul_arrays_int32(device const int* inputA [[buffer(0)]],
                            device const int* inputB [[buffer(1)]],
                            device int* result [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    result[index] = inputA[index] * inputB[index];
}

// ReLU activation kernel for float32 tensors
kernel void relu_float32(device const float* input [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
    result[index] = max(0.0f, input[index]);
}

// ReLU activation kernel for int32 tensors
kernel void relu_int32(device const int* input [[buffer(0)]],
                      device int* result [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = max(0, input[index]);
}

// Sigmoid activation kernel for float32 tensors
kernel void sigmoid_float32(device const float* input [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
    result[index] = 1.0f / (1.0f + exp(-input[index]));
}

// Matrix multiplication kernel for float32 tensors
// Simple version for 2D matrices
kernel void matmul_float32(device const float* matrixA [[buffer(0)]],
                          device const float* matrixB [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          constant uint& M [[buffer(3)]],     // rows of A
                          constant uint& N [[buffer(4)]],     // cols of A / rows of B
                          constant uint& P [[buffer(5)]],     // cols of B
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= P) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; k++) {
        sum += matrixA[row * N + k] * matrixB[k * P + col];
    }
    result[row * P + col] = sum;
}

// ===== FUSED OPERATION KERNELS =====
// These kernels combine multiple operations to reduce GPU kernel launch overhead

// Fused Linear layer kernel: MatMul + Bias addition in one GPU call
kernel void linear_forward_float32(device const float* input [[buffer(0)]],      // [batch_size, input_features]
                                  device const float* weight [[buffer(1)]],     // [output_features, input_features]
                                  device const float* bias [[buffer(2)]],       // [output_features]
                                  device float* output [[buffer(3)]],            // [batch_size, output_features]
                                  constant uint& batch_size [[buffer(4)]],
                                  constant uint& input_features [[buffer(5)]],
                                  constant uint& output_features [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    if (batch_idx >= batch_size || out_idx >= output_features) return;
    
    // Compute matrix multiplication for this output element
    float sum = 0.0f;
    for (uint in_idx = 0; in_idx < input_features; in_idx++) {
        sum += input[batch_idx * input_features + in_idx] * weight[in_idx * output_features + out_idx];
    }
    
    // Add bias
    sum += bias[out_idx];
    
    // Store result
    output[batch_idx * output_features + out_idx] = sum;
}

// Fused Linear + ReLU kernel: MatMul + Bias + ReLU activation in one GPU call
kernel void linear_relu_float32(device const float* input [[buffer(0)]],
                               device const float* weight [[buffer(1)]],
                               device const float* bias [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& batch_size [[buffer(4)]],
                               constant uint& input_features [[buffer(5)]],
                               constant uint& output_features [[buffer(6)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    if (batch_idx >= batch_size || out_idx >= output_features) return;
    
    // Compute matrix multiplication
    float sum = 0.0f;
    for (uint in_idx = 0; in_idx < input_features; in_idx++) {
        sum += input[batch_idx * input_features + in_idx] * weight[in_idx * output_features + out_idx];
    }
    
    // Add bias and apply ReLU activation
    sum = max(0.0f, sum + bias[out_idx]);
    
    // Store result
    output[batch_idx * output_features + out_idx] = sum;
}

// Fused Linear + Sigmoid kernel: MatMul + Bias + Sigmoid activation in one GPU call
kernel void linear_sigmoid_float32(device const float* input [[buffer(0)]],
                                  device const float* weight [[buffer(1)]],
                                  device const float* bias [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& batch_size [[buffer(4)]],
                                  constant uint& input_features [[buffer(5)]],
                                  constant uint& output_features [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    if (batch_idx >= batch_size || out_idx >= output_features) return;
    
    // Compute matrix multiplication
    float sum = 0.0f;
    for (uint in_idx = 0; in_idx < input_features; in_idx++) {
        sum += input[batch_idx * input_features + in_idx] * weight[in_idx * output_features + out_idx];
    }
    
    // Add bias and apply Sigmoid activation
    sum = 1.0f / (1.0f + exp(-(sum + bias[out_idx])));
    
    // Store result
    output[batch_idx * output_features + out_idx] = sum;
}

// Fused Batch MatMul kernel for processing multiple matrix multiplications in one call
kernel void batch_matmul_float32(device const float* batchA [[buffer(0)]],
                                device const float* batchB [[buffer(1)]],
                                device float* batchResult [[buffer(2)]],
                                constant uint& batch_size [[buffer(3)]],
                                constant uint& M [[buffer(4)]],
                                constant uint& N [[buffer(5)]],
                                constant uint& P [[buffer(6)]],
                                uint3 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    if (batch_idx >= batch_size || row >= M || col >= P) return;
    
    uint batch_offset_a = batch_idx * M * N;
    uint batch_offset_b = batch_idx * N * P;
    uint batch_offset_result = batch_idx * M * P;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; k++) {
        sum += batchA[batch_offset_a + row * N + k] * batchB[batch_offset_b + k * P + col];
    }
    
    batchResult[batch_offset_result + row * P + col] = sum;
}

// Fused gradient accumulation kernel for optimizer updates
kernel void adam_update_float32(device float* params [[buffer(0)]],
                               device const float* gradients [[buffer(1)]],
                               device float* m [[buffer(2)]],           // First moment estimate
                               device float* v [[buffer(3)]],           // Second moment estimate
                               constant float& lr [[buffer(4)]],        // Learning rate
                               constant float& beta1 [[buffer(5)]],     // Beta1
                               constant float& beta2 [[buffer(6)]],     // Beta2
                               constant float& eps [[buffer(7)]],       // Epsilon
                               constant uint& t [[buffer(8)]],          // Time step
                               uint index [[thread_position_in_grid]]) {
    // Update biased first moment estimate
    m[index] = beta1 * m[index] + (1.0f - beta1) * gradients[index];
    
    // Update biased second moment estimate
    v[index] = beta2 * v[index] + (1.0f - beta2) * gradients[index] * gradients[index];
    
    // Compute bias correction
    float m_hat = m[index] / (1.0f - pow(beta1, float(t)));
    float v_hat = v[index] / (1.0f - pow(beta2, float(t)));
    
    // Update parameters
    params[index] -= lr * m_hat / (sqrt(v_hat) + eps);
}

// Fused SGD with momentum update kernel
kernel void sgd_momentum_update_float32(device float* params [[buffer(0)]],
                                       device const float* gradients [[buffer(1)]],
                                       device float* velocity [[buffer(2)]],
                                       constant float& lr [[buffer(3)]],
                                       constant float& momentum [[buffer(4)]],
                                       constant float& weight_decay [[buffer(5)]],
                                       uint index [[thread_position_in_grid]]) {
    // Apply weight decay
    float grad = gradients[index] + weight_decay * params[index];
    
    // Update velocity with momentum
    velocity[index] = momentum * velocity[index] + grad;
    
    // Update parameters
    params[index] -= lr * velocity[index];
}

// Fused layer norm forward kernel
kernel void layer_norm_float32(device const float* input [[buffer(0)]],
                              device const float* gamma [[buffer(1)]],
                              device const float* beta [[buffer(2)]],
                              device float* output [[buffer(3)]],
                              device float* mean_out [[buffer(4)]],
                              device float* var_out [[buffer(5)]],
                              constant uint& batch_size [[buffer(6)]],
                              constant uint& features [[buffer(7)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint batch_idx = gid.y;
    
    if (batch_idx >= batch_size) return;
    
    uint offset = batch_idx * features;
    
    // Calculate mean
    float mean = 0.0f;
    for (uint i = 0; i < features; i++) {
        mean += input[offset + i];
    }
    mean /= float(features);
    mean_out[batch_idx] = mean;
    
    // Calculate variance
    float var = 0.0f;
    for (uint i = 0; i < features; i++) {
        float diff = input[offset + i] - mean;
        var += diff * diff;
    }
    var /= float(features);
    var_out[batch_idx] = var;
    
    // Normalize and scale
    float std_inv = rsqrt(var + 1e-5f);
    
    uint feature_idx = gid.x;
    if (feature_idx < features) {
        float normalized = (input[offset + feature_idx] - mean) * std_inv;
        output[offset + feature_idx] = gamma[feature_idx] * normalized + beta[feature_idx];
    }
}
`