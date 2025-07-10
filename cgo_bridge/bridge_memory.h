#pragma once

#import "bridge_types.h"

// Memory management functions
uintptr_t allocate_metal_buffer(uintptr_t device_ptr, int size, int device_type);
void deallocate_metal_buffer(uintptr_t buffer_ptr);

// Buffer utility functions
int zero_metal_buffer_mpsgraph(uintptr_t device_ptr, uintptr_t buffer_ptr, int size);
int zero_metal_buffer(uintptr_t device_ptr, uintptr_t buffer_ptr, int size);

// Buffer copy functions
int copy_data_to_metal_buffer(uintptr_t buffer_ptr, void* data, int size);
int copy_float32_array_to_metal_buffer(uintptr_t buffer_ptr, float* data, int num_elements);
int copy_int32_array_to_metal_buffer(uintptr_t buffer_ptr, int* data, int num_elements);
int copy_metal_buffer_to_float32_array(uintptr_t buffer_ptr, float* data, int num_elements);
int copy_metal_buffer_to_int32_array(uintptr_t buffer_ptr, int* data, int num_elements);

// Tensor type conversion
int convert_tensor_type(uintptr_t src_buffer_ptr, uintptr_t dst_buffer_ptr, 
                       int* shape, int num_dims, int src_type, int dst_type,
                       uintptr_t device_ptr);