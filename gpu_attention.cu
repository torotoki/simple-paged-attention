#include "common/host_utils.h"
#include "common/cuda_utility.hpp"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void simple_gemm_kernel(
    int N, int M, int L,
    float* A, float* B, float* C
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_idx >= N || global_idy >= L) return;
    
    float sum = 0.0f;
    for (int j = 0; j < M; ++j) {
      sum += A[global_idx * M + j] * B[j * L + global_idy];
    }
    C[global_idx * L + global_idy] = sum;
}

void launch_simple_gemm_kernel(
  int N,
  int M,
  int L,
  float* d_A,
  float* d_B,
  float* d_out
) {
  dim3 block_size(16, 16);
  dim3 grid_size(
      (N + block_size.x - 1) / block_size.x,
      (L + block_size.y - 1) / block_size.y
  );
  simple_gemm_kernel<<<grid_size, block_size>>>(
      N, M, L,
      d_A, d_B, d_out
  );
}

__global__ void transpose_gemm_kernel(
    int N, int M,
    float* A, float* B, float* out
) {
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int global_idy = blockDim.y * blockIdx.y + threadIdx.y;
  if (global_idx >= N || global_idy >= N) return;

  float sum = 0.0f;
  for (int j = 0; j < M; ++j) {
    // Q (N x d_k),  K (N x d_k)
    // Q (N x d_k) \dot K^T (d_k x N)
    // out (N x N)
    sum += A[global_idx * M + j] * B[global_idy * M + j];
  }
  out[global_idx * N + global_idy] = sum;
}

void launch_transpose_gemm_kernel(
  int N,
  int M,
  float* A,
  float* B,
  float* out
) {
  dim3 block_size(16, 16);
  dim3 grid_size(
      (N + block_size.x - 1) / block_size.x,
      (N + block_size.y - 1) / block_size.y
  );
  transpose_gemm_kernel<<<grid_size, block_size>>>(
      N, M, A, B, out
  );
}

__global__ void softmax_norm_kernel(
    int N, int M,
    float* A, float norm
) {
  /**
   * Very naive implementation of softmax
   * 1) 行ごとにsum_exp, max を集計する
   * 2) exp((A[i, j] - max) / d_k) / sum_exp
   *   sum_exp = Sigma_j exp((A[i, j] - max) / d_k)
   *
   **/
  cg::thread_block cta = cg::this_thread_block();
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_idy = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float max_value, sum;
  max_value = A[global_idy * M + 0] / norm;
  sum = 0.0f;
  if (threadIdx.x == 0) {
    for (int j = 0; j < M; ++j) {
      max_value = max(max_value, A[global_idy * M + j] / norm);
    }
  }
  cg::sync(cta);
  if (threadIdx.x == 0) {
    for (int j = 0; j < M; ++j) {
      sum += exp(A[global_idy * M + j] / norm - max_value);
    }
  }
  cg::sync(cta);

  if (global_idx < N && global_idy < M) {
    A[global_idy * M + global_idx] =
      exp(A[global_idy * M + global_idx] / norm - max_value) / sum;
  }
}

void launch_softmax_norm_kernel(
  int N,
  int M,
  float* A,
  int d_k = 1
) {
  assert(N > 0 && M > 0);
  float norm = std::sqrt(d_k);

  dim3 block_size(16, 1);
  dim3 grid_size(
      (N + block_size.x - 1) / block_size.x,
      (M + block_size.y - 1) / block_size.y
  );
  softmax_norm_kernel<<<grid_size, block_size>>>(
      N, M, A, norm
  );
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
  cudaDeviceSynchronize();

  // Free memory
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_QKT));
  checkCudaErrors(cudaFree(d_Q));
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaFree(d_V));
  checkCudaErrors(cudaFree(d_W_Q));
  checkCudaErrors(cudaFree(d_W_K));
  checkCudaErrors(cudaFree(d_W_V));
  checkCudaErrors(cudaFree(d_X));

  return h_out;
};

