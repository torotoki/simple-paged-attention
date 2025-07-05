#pragma once

#include "common/host_utils.h"

Matrix<float> launch_attention_kernels(
    int context_size,
    int d_model,
    int d_k,
    Matrix<float> h_W_Q,
    Matrix<float> h_W_K,
    Matrix<float> h_W_V,
    Matrix<float> h_X
);

