#include <cmath>
#include "common/host_utils.h"

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
        res += A[i * M + j] * B[j * L + k];
      }
      out[i * L + k] = res;
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
  /**
   * A: N x M
   * B: N x M
   * out: N x N
   */

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float res = 0.0f;
      for (int k = 0; k < M; ++k) {
        res += A[i * M + k] * B[k * N + j];
      }
      out[i * N + j] = res;
    }
  }
}

void softmax_norm(
  int N,
  int M,
  float* A,
  int d_k = 1
) {
  /**
   * Assumption:
   *  A: context_size (N) x context_size (M)
   *
   **/

  assert(N > 0 && M > 0);
  float norm = std::sqrt(d_k);
  for (int i = 0; i < N; ++i) {
    float max_value = A[i * M];
    for (int j = 0; j < M; ++j) {
      max_value = max(max_value, A[i * M + j] / norm);
    }

    float sum = 0.0f;
    for (int j = 0; j < M; ++j) {
      A[i * M + j] = std::exp(A[i * M + j] / norm - max_value);
      sum += A[i * M + j];
    }

    for (int j = 0; j < M; ++j) {
      A[i * M + j] /= sum;
    }
  }
}

Matrix<float> compute_attention_on_cpu(
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
  cout << "X: " << X.num_rows << " " << X.num_cols << endl;
  cout << "W_Q: " << W_Q.num_rows << " " << W_Q.num_cols << endl;
  cout << "Q: " << Q.num_rows << " " << Q.num_cols << endl;
  simple_gemm(context_size, d_model, d_k, X.get(), W_Q.get(), Q.get());
  simple_gemm(context_size, d_model, d_k, X.get(), W_K.get(), K.get());
  simple_gemm(context_size, d_model, d_k, X.get(), W_V.get(), V.get());

  Matrix<float> QKT(context_size, context_size);
  Matrix<float> out(context_size, d_k);
  cout << "QKT: " << QKT.num_rows << " " << QKT.num_cols << endl;
  cout << "out: " << out.num_rows << " " << out.num_cols << endl;
  transpose_gemm(context_size, d_k, Q.get(), K.get(), QKT.get());
  softmax_norm(context_size, context_size, QKT.get(), d_k);
  simple_gemm(context_size, context_size, d_k, QKT.get(), V.get(), out.get());
  return out;
};

