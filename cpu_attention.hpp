#pragma once
#include "common/host_utils.h"

void simple_gemm(
  int N,
  int M,
  int L,
  float* A,
  float* B,
  float* out
);

void transpose_gemm(
  int N,
  int M,
  float* A,
  float* B,
  float* out
);

void softmax_norm(
  int N,
  int M,
  float* A,
  int d_k = 1.0
);

Matrix<float> compute_attention_on_cpu(
    int context_size,
    int d_model,
    int d_k,
    Matrix<float> W_Q,
    Matrix<float> W_K,
    Matrix<float> W_V,
    Matrix<float> X
);

