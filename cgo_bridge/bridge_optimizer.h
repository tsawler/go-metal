#pragma once

#import "bridge_types.h"

// Optimizer scalar tensor caching functions
void cacheAdamScalarTensors(training_engine_t* engine);
void cacheRMSPropScalarTensors(training_engine_t* engine);
void cacheSGDScalarTensors(training_engine_t* engine);

// Adam optimizer functions
int execute_adam_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* variance_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step_count
);

int execute_adam_step(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* variance_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step_count
);

int execute_adam_step_mpsgraph_pooled(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* variance_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step_count,
    uintptr_t command_pool
);

// RMSProp optimizer functions
int execute_rmsprop_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* squared_grad_avg_buffers,
    uintptr_t* momentum_buffers,
    uintptr_t* gradient_avg_buffers,
    int num_weights,
    int* buffer_sizes,
    float learning_rate,
    float alpha,
    float epsilon,
    float weight_decay,
    float momentum,
    bool centered,
    int step_count
);

// L-BFGS optimizer functions
void cacheLBFGSScalarTensors(training_engine_t* engine);

int execute_lbfgs_step_mpsgraph(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* old_gradient_buffers,
    uintptr_t* search_dir_buffers,
    uintptr_t* s_vectors_flat,
    uintptr_t* y_vectors_flat,
    uintptr_t* rho_buffers,
    uintptr_t alpha_buffer,
    int num_weights,
    int* buffer_sizes,
    int history_size,
    int history_count,
    int history_index,
    float initial_step,
    float c1,
    float c2,
    int max_line_search,
    float current_loss,
    float prev_loss,
    float* step_size
);

int execute_lbfgs_step_mpsgraph_pooled(
    uintptr_t device_ptr,
    uintptr_t* weight_buffers,
    uintptr_t* gradient_buffers,
    uintptr_t* old_gradient_buffers,
    uintptr_t* search_dir_buffers,
    uintptr_t* s_vectors_flat,
    uintptr_t* y_vectors_flat,
    uintptr_t* rho_buffers,
    uintptr_t alpha_buffer,
    int num_weights,
    int* buffer_sizes,
    int history_size,
    int history_count,
    int history_index,
    float initial_step,
    float c1,
    float c2,
    int max_line_search,
    float current_loss,
    float prev_loss,
    uintptr_t command_pool,
    float* step_size
);