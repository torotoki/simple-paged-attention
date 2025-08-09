#include <stdio.h>
#include <cuda_runtime.h>
#include "common/cuda_utility.hpp"

#define MAX_NUM_SEQUENCES 32
#define TOKENS_PER_BLOCK 4
#define MAX_NUM_BLOCKS_PER_SEQ 64  // context_size = 64 * 4 = 256
#define MAX_BLOCKS (MAX_NUM_SEQUENCES * MAX_NUM_BLOCKS_PER_SEQ)

// NOTE: CUDA does not support the global variables.
// Instead, we use malloc and pass-by-reference.
// unsigned int* page_table[MAX_BLOCKS];

__host__ void init_page_table(int* page_table) {
  /**
   * Given empty page table, initialize it.
   */
  checkCudaErrors(cudaMalloc(&page_table, MAX_BLOCKS * sizeof(int)));
  for (int i = 0; i < MAX_BLOCKS; ++i) {
    page_table[i] = -1;  // initialized as the invalid address
  }
}

template<typename scalar_t>
__host__ void init_blocks(scalar_t* blocks, int d_k) {
  checkCudaErrors(cudaMalloc(&blocks, MAX_BLOCKS * d_k * sizeof(scalar_t)));
}

template<typename scalar_t>
__host__ void allocate_block(
    unsigned int physical_address,
    scalar_t* block,
    int d_k,
    int* page_table,
    scalar_t* cache_blocks
) {
  // TODO: allocate block naively on the page table
  checkCudaErrors(cudaMalloc(&block, TOKENS_PER_BLOCK * d_k * sizeof(scalar_t)));
}

__device__ unsigned int translate_address(
    unsigned int seq_idx,
    unsigned int block_idx,
    unsigned int* page_table
) {
  /**
   * seq_idx and block_idx represents a virtual address.
   */
  unsigned int virtual_address = seq_idx * MAX_NUM_BLOCKS_PER_SEQ + block_idx;
  if (virtual_address >= MAX_BLOCKS || block_idx >= MAX_NUM_BLOCKS_PER_SEQ) {
    printf("Error: the virtual address is out of range.\n");
  }
  
  unsigned int physical_address = page_table[virtual_address];
  return physical_address;
}

template<typename scalar_t>
__device__ __inline__ scalar_t* fetch_block(
  scalar_t* cache_blocks,
  int physical_address
) {
  return &cache_blocks[physical_address];
}

