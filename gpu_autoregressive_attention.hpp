#include <cmath>
#include "common/host_utils.h"


namespace GPUAutoregressiveAttention {
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
);
}
