/**
 * @file bench_compute.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief FP32 FMA compute-bound benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "benchmarks/benchmark_config.hpp"
#include "benchmarks/benchmark_utils.hpp"
#include "benchmarks/cuda_utils.hpp"
#include "benchmarks/status.hpp"
#include "benchmarks/timing.hpp"

namespace {

/**
 * @brief Runs eight independent FP32 FMA chains per thread in registers
 *
 * Each thread writes one value so that the compiler keeps the loop.
 *
 * @param output One float per thread
 * @param iterations FMA steps per chain
 */
__global__ void FmaFp32Kernel(float* output, int iterations) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float seed = 1.0F + static_cast<float>(index & 0xffU) * 0.0001F;
  float a0 = seed;
  float a1 = seed + 0.01F;
  float a2 = seed + 0.02F;
  float a3 = seed + 0.03F;
  float a4 = seed + 0.04F;
  float a5 = seed + 0.05F;
  float a6 = seed + 0.06F;
  float a7 = seed + 0.07F;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    a0 = fmaf(a0, 1.0000001F, 0.0000001F);
    a1 = fmaf(a1, 0.9999999F, 0.0000002F);
    a2 = fmaf(a2, 1.0000002F, -0.0000001F);
    a3 = fmaf(a3, 0.9999998F, 0.0000003F);
    a4 = fmaf(a4, 1.0000001F, -0.0000002F);
    a5 = fmaf(a5, 0.9999999F, 0.0000004F);
    a6 = fmaf(a6, 1.0000002F, -0.0000003F);
    a7 = fmaf(a7, 0.9999998F, 0.0000005F);
  }
  output[index] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

/**
 * @brief Prints the usage and the options of the benchmark
 *
 * @param program Name of the executable, from argv[0]
 */
void PrintHelp(const char* program) {
  std::cout << "Usage: " << program << " [options]\n\n"
            << "FP32 FMA compute-bound CUDA microbenchmark.\n\n"
            << "Options:\n"
            << "  --device <id>          CUDA device (default: 0)\n"
            << "  --duration <seconds>   Stable phase duration (default: 10)\n"
            << "  --warmup <iterations>  Warm-up kernel launches (default: 5)\n"
            << "  --block-size <threads> Threads per block (default: 256)\n"
            << "  --iterations <N>       FMA loop iterations (default: 10000)\n"
            << "  --help                  Show this help\n";
}

/**
 * @brief Parses the options, launches the FMA kernel for the measured
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
  BENCHMARKS_RETURN_IF_ERROR(arguments.ValidateOptions(
      {"--device", "--duration", "--warmup", "--block-size", "--iterations"}));
  benchmarks::CommonConfig config;
  BENCHMARKS_RETURN_IF_ERROR(benchmarks::ParseCommonConfig(arguments, &config));
  int iterations = 0;
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--iterations", 10000, &iterations));
  if (iterations <= 0) {
    return benchmarks::Status::Error(
        "--iterations must be in the range [1, 2^31-1]");
  }

  std::cout << "BENCHMARK=compute\nVARIANT=fma_fp32\n";
  benchmarks::PrintCommonConfig(config);
  std::cout << "ITERATIONS=" << iterations << '\n';

  cudaDeviceProp properties;
  BENCHMARKS_RETURN_IF_ERROR(
      benchmarks::InitializeCudaDevice(config.device, &properties));
  if (config.block_size > properties.maxThreadsPerBlock) {
    return benchmarks::Status::Error("--block-size exceeds the device limit");
  }
  const int grid_size = properties.multiProcessorCount * 8;
  const std::size_t output_elements =
      static_cast<std::size_t>(grid_size) * config.block_size;
  float* output_memory = nullptr;
  BENCHMARKS_CUDA_CHECK(
      cudaMalloc(&output_memory, output_elements * sizeof(float)));
  benchmarks::DeviceBuffer<float> output(output_memory);

  for (std::int64_t launch = 0; launch < config.warmup_iterations; ++launch) {
    FmaFp32Kernel<<<grid_size, config.block_size>>>(output.get(), iterations);
    BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  }
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());

  std::int64_t launches = 0;
  const benchmarks::SteadyTimer timer;
  do {
    for (int batch = 0; batch < benchmarks::kLaunchesBetweenSynchronizations;
         ++batch) {
      FmaFp32Kernel<<<grid_size, config.block_size>>>(output.get(), iterations);
      BENCHMARKS_CUDA_CHECK(cudaGetLastError());
      ++launches;
    }
    BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  } while (timer.ElapsedSeconds() < config.duration_seconds);
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  const double elapsed = timer.ElapsedSeconds();

  BENCHMARKS_CUDA_CHECK(cudaFree(output.release()));
  benchmarks::PrintExecutionSummary(elapsed, launches);
  return benchmarks::Status();
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
