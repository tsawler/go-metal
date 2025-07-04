#pragma once

#import "bridge_types.h"

// Device management functions
uintptr_t create_metal_device(void);
void destroy_metal_device(uintptr_t device_ptr);

// Training engine creation functions
uintptr_t create_training_engine(uintptr_t device_ptr, training_config_t* config);
uintptr_t create_training_engine_constant_weights(uintptr_t device_ptr, training_config_t* config);
uintptr_t create_training_engine_hybrid(uintptr_t device_ptr, training_config_t* config);
void destroy_training_engine(uintptr_t engine_ptr);