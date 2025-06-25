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
`