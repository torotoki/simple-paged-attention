#include <cmath>
#include <cassert>
#include "common/host_utils.h"
#include "common/cuda_utility.hpp"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace GPUAutoregressiveAttention {
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

__global__ void transpose_gemm_imbalance_kernel(
    int N, int r, int M,
    float* A, float* B, float* out
) {
  /**
   * Compute A ・B^T
   * A: N x M
   * B: r x M
   * out: N x r
   */
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int global_idy = blockDim.y * blockIdx.y + threadIdx.y;
  if (global_idx >= r || global_idy >= N) return;

  float sum = 0.0f;
  for (int j = 0; j < M; ++j) {
    // Q (N x d_k),  K (N x d_k)
    // Q (N x d_k) \dot K^T (d_k x N)
    // out (N x N)
    sum += A[global_idx * M + j] * B[global_idy * M + j];
  }
  out[global_idx * r + global_idy] = sum;
}

void launch_transpose_gemm_imbalance_kernel(
  int N,
  int r,
  int M,
  float* A,
  float* B,
  float* out
) {
  dim3 block_size(16, 16);
  dim3 grid_size(
      (r + block_size.x - 1) / block_size.x,
      (N + block_size.y - 1) / block_size.y
  );
  transpose_gemm_imbalance_kernel<<<grid_size, block_size>>>(
      N, r, M, A, B, out
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


Matrix<float> launch_autoregressive_attention_kernels(
    int context_size,
    int d_model,
    int d_k,
    Matrix<float>& h_W_Q,
    Matrix<float>& h_W_K,
    Matrix<float>& h_W_V,
    Matrix<float>& h_X,
    bool enable_kv_cache = false,
    bool verbose = false
) {
  // write code here

  float* d_W_Q, *d_W_K, *d_W_V, *d_X;
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
  float *d_K_cache, *d_V_cache;
  checkCudaErrors(cudaMalloc(&d_K_cache, context_size * d_k * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_V_cache, context_size * d_k * sizeof(float)));

  // Reuse d_Q over the time
  float* d_Q;
  checkCudaErrors(cudaMalloc(&d_Q, 1 * d_k * sizeof(float)));

  // Variables for stacking outputs
  Matrix<float> h_out(context_size, d_k);
  float *d_out;
  size_t output_size = context_size * d_k * sizeof(float);
  checkCudaErrors(cudaMalloc(&d_out, output_size));

  // Compute the multiplicative attention iteratively
  for (int t = 1; t <= context_size; ++t) {
    // Q: 1 x d_k
    
    // h_last_word_embed: 1 x d_model
    float* d_last_word_embed = &d_X[h_X.num_cols * (t - 1)];

    // Q (1 x d_k) = last_word_embed (1 x d_model) ・W_Q (d_model x d_k)
    launch_simple_gemm_kernel(1, d_model, d_k, d_last_word_embed, d_W_Q, d_Q);
    
    if (!enable_kv_cache) {
      launch_simple_gemm_kernel(t, d_model, d_k, d_X, d_W_K, d_K_cache);
      launch_simple_gemm_kernel(t, d_model, d_k, d_X, d_W_V, d_V_cache);
    } else {
      assert(false);
    }

    // QKT (1 x t) = Q (1 x d_k) ・K^T (t x d_k)^T
    float* d_QKT;
    checkCudaErrors(cudaMalloc(&d_QKT, 1 * t * sizeof(float)));
    launch_transpose_gemm_imbalance_kernel(1, t, d_k, d_Q, d_K_cache, d_QKT);
    
    launch_softmax_norm_kernel(1, t, d_QKT, d_k);
    
    // Extract t-th output embedding
    float* d_out_single_word = &d_out[h_out.num_cols * (t - 1)];
    launch_simple_gemm_kernel(1, t, d_k, d_QKT, d_V_cache, d_out_single_word);

    checkCudaErrors(cudaFree(d_QKT));
  }

  checkCudaErrors(cudaMemcpy(h_out.get(), d_out, output_size, cudaMemcpyDefault));
  cudaDeviceSynchronize();

  // Free memory
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_Q));
  checkCudaErrors(cudaFree(d_K_cache));
  checkCudaErrors(cudaFree(d_V_cache));
  checkCudaErrors(cudaFree(d_W_Q));
  checkCudaErrors(cudaFree(d_W_K));
  checkCudaErrors(cudaFree(d_W_V));
  checkCudaErrors(cudaFree(d_X));

  return h_out;
}
}
