#pragma once

#import "bridge_types.h"

// Device management functions
uintptr_t create_metal_device(void);
void destroy_metal_device(uintptr_t device_ptr);

// Training engine creation functions
uintptr_t create_training_engine(uintptr_t device_ptr, training_config_t* config);
uintptr_t create_training_engine_constant_weights(uintptr_t device_ptr, training_config_t* config);
uintptr_t create_training_engine_hybrid(uintptr_t device_ptr, training_config_t* config, model_config_t* model_config);
uintptr_t create_training_engine_dynamic(uintptr_t device_ptr, training_config_t* config, 
                                       layer_spec_c_t* layers, int numLayers, int* inputShape, int inputShapeLen);
uintptr_t create_command_queue(uintptr_t device_ptr);
void destroy_training_engine(uintptr_t engine_ptr);

// Command buffer management functions
uintptr_t create_command_buffer(uintptr_t command_queue_ptr);
void release_command_buffer(uintptr_t command_buffer_ptr);
void release_command_queue(uintptr_t command_queue_ptr);
int commit_command_buffer(uintptr_t command_buffer_ptr);
int wait_command_buffer_completion(uintptr_t command_buffer_ptr);

// Autorelease pool management
void setup_autorelease_pool(void);
void drain_autorelease_pool(void);