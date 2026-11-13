/**
 * @file bench_memory.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Global memory streaming copy benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "benchmarks/benchmark_config.hpp"
#include "benchmarks/benchmark_utils.hpp"
#include "benchmarks/cuda_utils.hpp"
#include "benchmarks/status.hpp"
#include "benchmarks/timing.hpp"

namespace {

/**
 * @brief Copies source into destination with a grid-stride loop
 *
 * @param destination Output buffer with at least elements floats
 * @param source Input buffer with at least elements floats
 * @param elements Number of floats to copy
 */
__global__ void StreamingCopyKernel(float* destination, const float* source,
                                    std::size_t elements) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = first; index < elements; index += step) {
    destination[index] = source[index];
  }
}

/**
 * @brief Prints the usage and the options of the benchmark
 *
 * @param program Name of the executable, from argv[0]
 */
void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Global-memory streaming-copy CUDA microbenchmark.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device (default: 0)\n"
      << "  --duration <seconds>   Stable phase duration (default: 10)\n"
      << "  --warmup <iterations>  Warm-up kernel launches (default: 5)\n"
      << "  --block-size <threads> Threads per block (default: 256)\n"
      << "  --size <bytes>         Bytes per buffer (default: 268435456)\n"
      << "  --mode <copy>          Memory operation (default: copy)\n"
      << "  --help                  Show this help\n";
}

/**
 * @brief Parses the options, launches the copy kernel for the measured
 * phase and prints the results
 *
 * @param argc Number of arguments, as received by main()
 * @param argv Arguments, as received by main()
 * @return Error if an option is invalid or a CUDA call fails
 */
benchmarks::Status Run(int argc, char** argv) {
  benchmarks::ArgumentParser arguments;
  BENCHMARKS_RETURN_IF_ERROR(arguments.Parse(argc, argv));
  if (arguments.HelpRequested()) {
    PrintHelp(argv[0]);
    return benchmarks::Status();
  }
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.ValidateOptions({"--device", "--duration", "--warmup",
                                 "--block-size", "--size", "--mode"}));
  benchmarks::CommonConfig config;
  BENCHMARKS_RETURN_IF_ERROR(benchmarks::ParseCommonConfig(arguments, &config));
  std::int64_t size_value = 0;
  BENCHMARKS_RETURN_IF_ERROR(arguments.Int64Value(
      "--size", benchmarks::kDefaultBufferBytes, &size_value));
  if (size_value <= 0) {
    return benchmarks::Status::Error(
        "--size is out of range for this platform");
  }
  const auto bytes = static_cast<std::size_t>(size_value);
  if (bytes % sizeof(float) != 0) {
    return benchmarks::Status::Error("--size must be a multiple of 4 bytes");
  }
  const std::string mode = arguments.StringValue("--mode", "copy");
  if (mode != "copy") {
    return benchmarks::Status::Error("--mode currently supports only 'copy'");
  }

  std::cout << "BENCHMARK=memory\nVARIANT=streaming_copy\n";
  benchmarks::PrintCommonConfig(config);
  std::cout << "BYTES=" << bytes << "\nMODE=" << mode << '\n';

  cudaDeviceProp properties;
  BENCHMARKS_RETURN_IF_ERROR(
      benchmarks::InitializeCudaDevice(config.device, &properties));
  if (config.block_size > properties.maxThreadsPerBlock) {
    return benchmarks::Status::Error("--block-size exceeds the device limit");
  }
  const std::size_t elements = bytes / sizeof(float);
  const std::size_t required_blocks =
      (elements + static_cast<std::size_t>(config.block_size) - 1) /
      static_cast<std::size_t>(config.block_size);
  const auto grid_size = static_cast<int>(std::min<std::size_t>(
      required_blocks,
      static_cast<std::size_t>(properties.multiProcessorCount) * 32));

  float* source_memory = nullptr;
  BENCHMARKS_CUDA_CHECK(cudaMalloc(&source_memory, bytes));
  benchmarks::DeviceBuffer<float> source(source_memory);
  float* destination_memory = nullptr;
  BENCHMARKS_CUDA_CHECK(cudaMalloc(&destination_memory, bytes));
  benchmarks::DeviceBuffer<float> destination(destination_memory);
  BENCHMARKS_CUDA_CHECK(cudaMemset(source.get(), 0x3f, bytes));
  BENCHMARKS_CUDA_CHECK(cudaMemset(destination.get(), 0, bytes));

  for (std::int64_t launch = 0; launch < config.warmup_iterations; ++launch) {
    StreamingCopyKernel<<<grid_size, config.block_size>>>(
        destination.get(), source.get(), elements);
    BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  }
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());

  std::int64_t launches = 0;
  const benchmarks::SteadyTimer timer;
  do {
    for (int batch = 0; batch < benchmarks::kLaunchesBetweenSynchronizations;
         ++batch) {
      StreamingCopyKernel<<<grid_size, config.block_size>>>(
          destination.get(), source.get(), elements);
      BENCHMARKS_CUDA_CHECK(cudaGetLastError());
      ++launches;
    }
    BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  } while (timer.ElapsedSeconds() < config.duration_seconds);
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  const double elapsed = timer.ElapsedSeconds();

  BENCHMARKS_CUDA_CHECK(cudaFree(destination.release()));
  BENCHMARKS_CUDA_CHECK(cudaFree(source.release()));
  benchmarks::PrintExecutionSummary(elapsed, launches);
  return benchmarks::Status();
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
