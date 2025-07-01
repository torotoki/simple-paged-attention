#include "common/host_utils.cpp"
#include <cuda_runtime.h>

__global__ simple_gemm_kernel(
  int N,
  int M,
  int L,
  float* A,
  float* B,
  float* out
) {
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;
  int res = 0.0;
  for (int j = 0; j < M; ++j) {
    res += A[threadIdx.y * blockDim.x + j] * B[j * blockDim.x + threadidx.x];
  }
  out[global_idx] = res;
}


Matrix<float> launch_attention_kernel(
    Matrix<float> W_Q,
    Matrix<float> W_K,
    Matrix<float> W_V,
    Matrix<float> X
) {
  // write code here
  //
  simple_gemm_kernel
};

