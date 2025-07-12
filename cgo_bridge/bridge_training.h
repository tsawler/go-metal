#pragma once

#import "bridge_types.h"

// Training execution functions
// Implementations are in bridge_training.m

// Helper functions
uintptr_t get_command_buffer_from_pool(uintptr_t command_buffer);
void return_command_buffer_to_pool(uintptr_t command_buffer);

// Basic training step functions
int execute_training_step(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* loss_out
);

// Hybrid training step functions (MPS + MPSGraph)
int execute_training_step_hybrid(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* loss_out
);

int execute_training_step_hybrid_full(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float learning_rate,
    float* loss_out
);

int execute_training_step_hybrid_full_pooled(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float learning_rate,
    uintptr_t command_buffer,
    float* loss_out
);

// Dynamic training step functions
int execute_training_step_dynamic_with_gradients(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    float learning_rate,
    int batch_size,
    float* loss_out
);

int execute_training_step_dynamic_with_gradients_pooled(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    int batch_size,
    uintptr_t command_buffer,
    float* loss_out
);

// Hybrid training step with gradients (for Adam optimizer)
int execute_training_step_hybrid_with_gradients(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    float* loss_out
);

int execute_training_step_hybrid_with_gradients_pooled(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    uintptr_t command_buffer,
    float* loss_out
);

// SGD-specific optimized training step (critical for SGD performance)
int execute_training_step_sgd_pooled(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    int num_weights,
    float learning_rate,
    int batch_size,
    uintptr_t command_buffer,
    float* loss_out
);

// Additional training step functions
int execute_training_step_dynamic(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t label_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float learning_rate,
    int batch_size,
    float* loss_out
);

// Inference functions  
int execute_inference_hybrid(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* predictions_out,
    int batch_size,
    int num_classes
);

int execute_inference_dynamic(
    uintptr_t engine_ptr,
    uintptr_t input_buffer,
    uintptr_t* weight_buffers,
    int num_weights,
    float* predictions_out,
    int batch_size,
    int num_classes
);