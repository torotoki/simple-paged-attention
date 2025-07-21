# simple-paged-attention

This is an implementation of PagedAttention with CUDA and C++.

It contains five types of implementations:

- ✅ Standard causal attention on CPU
- ✅ Standard causal attention on GPU
- ✅ Attention with autoregressive output and KV-cache (common in inference) on CPU
- ✅ Attention with autoregressive output and KV-cache (common in inference) on GPU
- 🚧 PagedAttention on GPU


## Benchmark Results:

```
Command: attention_cpu
Averaged Time (msec): 3.42877

Command: attention_gpu
Averaged Time (msec): 1.26602

Command: attention_cpu_autoregressive
Enable KV cache: 0
Averaged Time (msec): 18.6311

Command: attention_cpu_autoregressive
Enable KV cache: 1
Averaged Time (msec): 3.65721

Command: attention_gpu_autoregressive
Enable KV cache: 0
Averaged Time (msec): 3.11079

Command: attention_gpu_autoregressive
Enable KV cache: 1
Averaged Time (msec): 2.88444
```

