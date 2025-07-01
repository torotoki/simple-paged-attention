#include "common/host_utils.cpp"
#include <cuda_runtime.h>

void simple_gemm(
  int N,
  int M,
  int L,
  float* A,
  float* B,
  float* out
) {
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < L; ++k) {
      float res = 0.0f;
      for (int j = 0; j < M; ++j) {
        res += A[i * N + j] * B[j * M + k];
      }
      out[i * N + k] = res;
    }
  }
}

void transpose_gemm(
  int N,
  int M,
  float* A,
  float* B,
  float* out
) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float res = 0.0f;
      for (int k = 0; k < M; ++k) {
        res += A[i * N + k] * B[j * N + k];
      }
      out[i * N + j] = res;
    }
  }
}

void softmax_norm(
  int N,
  int M,
  float* A,
  int d_k
) {
  assert(N > 0 && M > 0);
  float max_element = A[0];
  float sum = 0.0f;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      max_element = max(max_element, A[i * N + j]);
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      sum += std::exp(A[i * N + j] - max_element);
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i * N + j] = std::exp(A[i * N + j] - max_element) / sum;
    }
  }
}

Matrix<float> launch_attention_kernel(
    int context_size,
    int d_model,
    int d_k,
    Matrix<float> W_Q,
    Matrix<float> W_K,
    Matrix<float> W_V,
    Matrix<float> X
) {
  // write code here
  //
  Matrix<float> Q(context_size, d_k);
  Matrix<float> K(context_size, d_k);
  Matrix<float> V(context_size, d_k);
  simple_gemm(context_size, d_model, d_k, X.get(), W_Q.get(), Q.get());
  simple_gemm(context_size, d_model, d_k, X.get(), W_K.get(), K.get());
  simple_gemm(context_size, d_model, d_k, X.get(), W_V.get(), V.get());

  Matrix<float> QKT(context_size, context_size);
  transpose_gemm(context_size, d_k, Q.get(), K.get(), QKT.get());
  softmax_norm(context_size, context_size, QKT.get(), d_k);
  Matrix<float> out(context_size, d_k);
  simple_gemm(context_size, context_size, d_k, QKT.get(), V.get(), out.get());
  return out;
};

