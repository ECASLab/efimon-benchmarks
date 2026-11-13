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

Status CheckCuda(cudaError_t result, const char* expression, const char* file,
                 int line) {
  if (result == cudaSuccess) {
    return Status();
  }
  std::ostringstream message;
  message << file << ':' << line << ": CUDA call " << expression
          << " failed with " << cudaGetErrorName(result) << " ("
          << static_cast<int>(result) << "): " << cudaGetErrorString(result);
  return Status::Error(message.str());
}

Status InitializeCudaDevice(int device, cudaDeviceProp* properties) {
  int device_count = 0;
  BENCHMARKS_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device >= device_count) {
    std::ostringstream message;
    message << "requested CUDA device " << device << ", but only "
            << device_count << " device(s) are available";
    return Status::Error(message.str());
  }
  BENCHMARKS_CUDA_CHECK(cudaSetDevice(device));
  // Create the primary context now so that it is not part of the timing.
  BENCHMARKS_CUDA_CHECK(cudaFree(nullptr));

  *properties = cudaDeviceProp{};
  BENCHMARKS_CUDA_CHECK(cudaGetDeviceProperties(properties, device));
  return Status();
}

}  // namespace benchmarks
