/**
 * @file cuda_utils.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief CUDA error checking and device setup helpers
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <sstream>

#include "benchmarks/cuda_utils.hpp"

namespace benchmarks {

void CheckCuda(cudaError_t result, const char* expression, const char* file,
               int line) {
  if (result == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": CUDA call " << expression
          << " failed with " << cudaGetErrorName(result) << " ("
          << static_cast<int>(result) << "): " << cudaGetErrorString(result);
  throw CudaError(message.str());
}

cudaDeviceProp InitializeCudaDevice(int device) {
  int device_count = 0;
  BENCHMARKS_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device >= device_count) {
    std::ostringstream message;
    message << "requested CUDA device " << device << ", but only "
            << device_count << " device(s) are available";
    throw std::invalid_argument(message.str());
  }
  BENCHMARKS_CUDA_CHECK(cudaSetDevice(device));
  BENCHMARKS_CUDA_CHECK(
      cudaFree(nullptr));  // Materialize the primary context before timing.

  cudaDeviceProp properties{};
  BENCHMARKS_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  return properties;
}

}  // namespace benchmarks
