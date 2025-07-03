#include "common/host_utils.h"
#include "common/cuda_utility.hpp"
#include <cuda_runtime.h>

void launch_simple_gemm_kernel(
  int N,
  int M,
  int L,
  float* d_A,
  float* d_B,
  float* d_out
) {
}

void transpose_gemm(
  int N,
  int M,
  float* A,
  float* B,
  float* out
) {
}

void softmax_norm(
  int N,
  int M,
  float* A,
  int d_k = 1.0
) {
  assert(N > 0 && M > 0);
}

Matrix<float> launch_attention_kernels(
    int context_size,
    int d_model,
    int d_k,
    Matrix<float> h_W_Q,
    Matrix<float> h_W_K,
    Matrix<float> h_W_V,
    Matrix<float> h_X
) {
  // write code here
  //
  // Input
  float *d_W_Q, *d_W_K, *d_W_V, *d_X;
  size_t input_size = context_size * d_model * sizeof(float);
  size_t weight_size = d_model * d_k * sizeof(float);
  checkCudaErrors(cudaMalloc(&d_W_Q, weight_size));
  checkCudaErrors(cudaMalloc(&d_W_K, weight_size));
  checkCudaErrors(cudaMalloc(&d_W_V, weight_size));
  checkCudaErrors(cudaMalloc(&d_X, input_size));
  checkCudaErrors(cudaMemcpy(d_W_Q, h_W_Q.get(), weight_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(d_W_K, h_W_K.get(), weight_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(d_W_V, h_W_V.get(), weight_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(d_X, h_X.get(), input_size, cudaMemcpyDefault));

  // Intermediate output for projection
  float *d_Q, *d_K, *d_V;
  checkCudaErrors(cudaMalloc(&d_Q, context_size * d_k * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_K, context_size * d_k * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_V, context_size * d_k * sizeof(float)));

  // Compute attention matrix
  launch_simple_gemm_kernel(context_size, d_model, d_k, d_X, d_W_Q, d_Q);
  launch_simple_gemm_kernel(context_size, d_model, d_k, d_X, d_W_K, d_K);
  launch_simple_gemm_kernel(context_size, d_model, d_k, d_X, d_W_V, d_V);

  // Intermediate output for attention, and output
  Matrix<float> h_out(context_size, d_k);
  float *d_QKT, *d_out;
  size_t qkt_size = context_size * context_size * sizeof(float);
  size_t output_size = context_size * d_k * sizeof(float);
  checkCudaErrors(cudaMalloc(&d_QKT, qkt_size));
  checkCudaErrors(cudaMalloc(&d_out, output_size));

  // Compute softmax(QKT / norm) @ V
  launch_transpose_gemm_kernel(context_size, d_k, d_Q, d_K, d_QKT);
  launch_softmax_norm_kernel(context_size, context_size, d_QKT, d_k);
  launch_simple_gemm_kernel(context_size, context_size, d_k, d_QKT, d_V, d_out);
  checkCudaErrors(cudaMemcpy(h_out.get(), d_out, output_size, cudaMemcpyDefault));

  return h_out;
};

