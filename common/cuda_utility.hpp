#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

