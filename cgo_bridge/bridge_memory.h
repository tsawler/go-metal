#pragma once

#import "bridge_types.h"

// Memory management functions
uintptr_t allocate_metal_buffer(uintptr_t device_ptr, int size, int device_type);
void deallocate_metal_buffer(uintptr_t buffer_ptr);

// Buffer utility functions
int zero_metal_buffer_mpsgraph(uintptr_t device_ptr, uintptr_t buffer_ptr, int size);
int zero_metal_buffer(uintptr_t device_ptr, uintptr_t buffer_ptr, int size);