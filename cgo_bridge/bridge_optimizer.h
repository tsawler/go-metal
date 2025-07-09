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