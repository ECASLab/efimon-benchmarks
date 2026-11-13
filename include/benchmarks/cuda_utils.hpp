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

#include <memory>

#include "benchmarks/status.hpp"

namespace benchmarks {

/**
 * @brief Converts the result of a CUDA runtime call into a Status
 *
 * Use it through BENCHMARKS_CUDA_CHECK, which fills the expression, file and
 * line.
 *
 * @param result Value returned by the CUDA call
 * @param expression Text of the CUDA call
 * @param file Source file of the call
 * @param line Source line of the call
 * @return Error with the CUDA error name, code and description if the call
 * failed
 */
Status CheckCuda(cudaError_t result, const char* expression, const char* file,
                 int line);

/**
 * @brief Selects the CUDA device and creates its context
 *
 * The context is created here so that its cost is not part of the measured
 * phase.
 *
 * @param device CUDA device index
 * @param properties Output with the device properties. Must not be null
 * @return Error if the device does not exist or a CUDA call fails
 */
Status InitializeCudaDevice(int device, cudaDeviceProp* properties);

/**
 * @brief Deleter that releases device memory with cudaFree()
 */
struct CudaFreeDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};

/**
 * @brief Deleter that releases pinned host memory with cudaFreeHost()
 */
struct CudaFreeHostDeleter {
  void operator()(void* pointer) const { cudaFreeHost(pointer); }
};

/**
 * @brief Owner of device memory, released when it goes out of scope
 */
template <typename T>
using DeviceBuffer = std::unique_ptr<T, CudaFreeDeleter>;

/**
 * @brief Owner of pinned host memory, released when it goes out of scope
 */
template <typename T>
using PinnedBuffer = std::unique_ptr<T, CudaFreeHostDeleter>;

}  // namespace benchmarks

/**
 * @brief Returns an error Status from the current function if the CUDA call
 * fails
 */
#define BENCHMARKS_CUDA_CHECK(expression) \
  BENCHMARKS_RETURN_IF_ERROR(             \
      ::benchmarks::CheckCuda((expression), #expression, __FILE__, __LINE__))

#endif  // INCLUDE_BENCHMARKS_CUDA_UTILS_HPP_
