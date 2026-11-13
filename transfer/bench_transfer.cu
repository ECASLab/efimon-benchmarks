/**
 * @file bench_transfer.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Host-device copy benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <string>

#include "benchmarks/benchmark_config.hpp"
#include "benchmarks/benchmark_utils.hpp"
#include "benchmarks/cuda_utils.hpp"
#include "benchmarks/status.hpp"
#include "benchmarks/timing.hpp"

namespace {

/**
 * @brief Prints the usage and the options of the benchmark
 *
 * @param program Name of the executable, from argv[0]
 */
void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Repeated host/device CUDA transfer microbenchmark.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device (default: 0)\n"
      << "  --duration <seconds>   Stable phase duration (default: 10)\n"
      << "  --warmup <iterations>  Warm-up transfers (default: 5)\n"
      << "  --block-size <threads> Accepted for CLI consistency "
         "(default: 256)\n"
      << "  --size <bytes>         Bytes per transfer (default: 268435456)\n"
      << "  --direction <h2d|d2h>  Transfer direction (default: h2d)\n"
      << "  --memory <type>        pinned or pageable (default: pinned)\n"
      << "  --help                  Show this help\n";
}

/**
 * @brief Converts the --direction value into a cudaMemcpyKind
 *
 * @param direction h2d or d2h
 * @param kind Output. Must not be null
 * @return Error if the direction is not supported
 */
benchmarks::Status CopyKind(const std::string& direction,
                            cudaMemcpyKind* kind) {
  if (direction == "h2d") {
    *kind = cudaMemcpyHostToDevice;
    return benchmarks::Status();
  }
  if (direction == "d2h") {
    *kind = cudaMemcpyDeviceToHost;
    return benchmarks::Status();
  }
  return benchmarks::Status::Error("--direction must be 'h2d' or 'd2h'");
}

/**
 * @brief Parses the options, copies between host and device for the
 * measured phase and prints the results
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
      {"--device", "--duration", "--warmup", "--block-size", "--size",
       "--direction", "--memory"}));
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
  const std::string direction = arguments.StringValue("--direction", "h2d");
  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  BENCHMARKS_RETURN_IF_ERROR(CopyKind(direction, &kind));
  const std::string memory = arguments.StringValue("--memory", "pinned");
  if (memory != "pinned" && memory != "pageable") {
    return benchmarks::Status::Error("--memory must be 'pinned' or 'pageable'");
  }

  std::cout << "BENCHMARK=transfer\nVARIANT=cuda_memcpy\n";
  benchmarks::PrintCommonConfig(config);
  std::cout << "BYTES=" << bytes << "\nDIRECTION=" << direction
            << "\nMEMORY=" << memory << '\n';

  cudaDeviceProp properties;
  BENCHMARKS_RETURN_IF_ERROR(
      benchmarks::InitializeCudaDevice(config.device, &properties));
  void* device_memory = nullptr;
  BENCHMARKS_CUDA_CHECK(cudaMalloc(&device_memory, bytes));
  benchmarks::DeviceBuffer<void> device_buffer(device_memory);

  // Host bytes are raw bit patterns, so unsigned char is the right type.
  benchmarks::PinnedBuffer<void> pinned_buffer;
  std::unique_ptr<unsigned char[]> pageable_buffer;
  unsigned char* host_buffer = nullptr;
  if (memory == "pinned") {
    void* pinned_memory = nullptr;
    BENCHMARKS_CUDA_CHECK(cudaMallocHost(&pinned_memory, bytes));
    pinned_buffer.reset(pinned_memory);
    host_buffer = static_cast<unsigned char*>(pinned_memory);
  } else {
    pageable_buffer.reset(new (std::nothrow) unsigned char[bytes]);
    if (pageable_buffer == nullptr) {
      return benchmarks::Status::Error(
          "failed to allocate pageable host memory");
    }
    host_buffer = pageable_buffer.get();
  }
  std::fill_n(host_buffer, bytes, static_cast<unsigned char>(0x5a));
  BENCHMARKS_CUDA_CHECK(cudaMemset(device_buffer.get(), 0x3c, bytes));

  void* destination = direction == "h2d" ? device_buffer.get() : host_buffer;
  const void* source = direction == "h2d" ? host_buffer : device_buffer.get();
  for (std::int64_t copy = 0; copy < config.warmup_iterations; ++copy) {
    BENCHMARKS_CUDA_CHECK(cudaMemcpy(destination, source, bytes, kind));
  }
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());

  std::int64_t copies = 0;
  const benchmarks::SteadyTimer timer;
  do {
    BENCHMARKS_CUDA_CHECK(cudaMemcpy(destination, source, bytes, kind));
    ++copies;
  } while (timer.ElapsedSeconds() < config.duration_seconds);
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  const double elapsed = timer.ElapsedSeconds();

  if (pinned_buffer != nullptr) {
    BENCHMARKS_CUDA_CHECK(cudaFreeHost(pinned_buffer.release()));
  }
  pageable_buffer.reset();
  BENCHMARKS_CUDA_CHECK(cudaFree(device_buffer.release()));
  benchmarks::PrintExecutionSummary(elapsed, copies);
  return benchmarks::Status();
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
