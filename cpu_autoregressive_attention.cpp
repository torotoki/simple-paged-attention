#include <cmath>
#include <algorithm>
#include <cassert>
#include "common/host_utils.h"

namespace CPUAutoregressiveAttention {
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

void simple_gemm_with_kv_cache(
  int N,
  int M,
  int L,
  float* A,
  float* B,
  float* out
) {
  /**
   * A: t x d_model
   * B: d_model x d_k
   */
  for (int l = 0; l < L; ++l) {
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
      acc += A[(N - 1) * M + m] * B[m * L + l];
    }
    out[(N - 1) * L + l] = acc;
  }
}

void transpose_gemm_imbalance(
  int N,
  int r,
  int M,
  float* A,
  float* B,
  float* out
) {
  /**
   * Compute A ・B^T
   * A: N x M
   * B: r x M
   * out: N x r
   */

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < r; ++j) {
      float res = 0.0f;
      for (int k = 0; k < M; ++k) {
        res += A[i * M + k] * B[j * M + k];
      }
      out[i * r + j] = res;
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
    float max_value = A[i * M] / norm;
    for (int j = 0; j < M; ++j) {
      max_value = std::max(max_value, A[i * M + j] / norm);
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

Matrix<float> compute_autoregressive_attention_on_cpu(
    int context_size,
    int d_model,
    int d_k,
    Matrix<float>& W_Q,
    Matrix<float>& W_K,
    Matrix<float>& W_V,
    Matrix<float>& X,
    bool enable_kv_cache = true,
    bool verbose = false
) {
  // write code here

  Matrix<float> K_cache(context_size, d_k);
  Matrix<float> V_cache(context_size, d_k);
  Matrix<float> out(context_size, d_k);

  for (int t = 1; t <= context_size; ++t) {
    Matrix<float> Q(1, d_k);

    // Extract t-th word's embedding.
    // last_word_embed: 1 x d_model
    float* last_word_embed = &X.get()[X.num_cols * (t - 1)];

    // Q: 1 x d_k
    simple_gemm(1, d_model, d_k, last_word_embed, W_Q.get(), Q.get());

    // K_cache (t x d_k) = X (t x d_model) ・W_K (d_model x d_k)
    // V_cache (t x d_k) = X (t x d_model) ・W_V (d_model x d_k)
    // TODO: Need to use KV-cache
    // Replace simple_gemm(...) to gemm_with_kv_cache(...)!
    if (!enable_kv_cache) {
      simple_gemm(t, d_model, d_k, X.get(), W_K.get(), K_cache.get());
      simple_gemm(t, d_model, d_k, X.get(), W_V.get(), V_cache.get());
    } else {
      simple_gemm_with_kv_cache(t, d_model, d_k, X.get(), W_K.get(), K_cache.get());
      simple_gemm_with_kv_cache(t, d_model, d_k, X.get(), W_V.get(), V_cache.get());
    }

    Matrix<float> QKT(1, t);
    // QKT (1 x t) = Q (1 x d_k) ・K^T (t, d_k)^T
    transpose_gemm_imbalance(1, t, d_k, Q.get(), K_cache.get(), QKT.get());

    softmax_norm(1, t, QKT.get(), d_k);

    // Extract t-th output embedding
    float* out_single_word = &out.get()[out.num_cols * (t - 1)];
    simple_gemm(1, t, d_k, QKT.get(), V_cache.get(), out_single_word);
  }

  return out;
};
}
