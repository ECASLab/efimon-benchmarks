/**
 * @file cuda_utils.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief CUDA error checking and device setup helpers
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_CUDA_UTILS_HPP_
#define INCLUDE_BENCHMARKS_CUDA_UTILS_HPP_

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace benchmarks {

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void CheckCuda(cudaError_t result, const char* expression, const char* file,
               int line);
cudaDeviceProp InitializeCudaDevice(int device);

}  // namespace benchmarks

#define BENCHMARKS_CUDA_CHECK(expression) \
  ::benchmarks::CheckCuda((expression), #expression, __FILE__, __LINE__)

#endif  // INCLUDE_BENCHMARKS_CUDA_UTILS_HPP_
