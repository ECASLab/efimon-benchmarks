/**
 * @file bench_idle.cpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Idle benchmark with or without a CUDA context
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

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
      << "Idle baseline with an optional pre-created CUDA context.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device for cuda-context mode "
         "(default: 0)\n"
      << "  --duration <seconds>   Idle phase duration (default: 10)\n"
      << "  --warmup <iterations>  Accepted for CLI consistency (default: 5)\n"
      << "  --block-size <threads> Accepted for CLI consistency "
         "(default: 256)\n"
      << "  --mode <mode>           system or cuda-context (default: system)\n"
      << "  --help                  Show this help\n";
}

/**
 * @brief Parses the options, sleeps for the measured phase and prints
 * the results
 *
 * @param argc Number of arguments, as received by main()
 * @param argv Arguments, as received by main()
 * @return Error if an option is invalid or the CUDA context cannot be
 * created
 */
benchmarks::Status Run(int argc, char** argv) {
  benchmarks::ArgumentParser arguments;
  BENCHMARKS_RETURN_IF_ERROR(arguments.Parse(argc, argv));
  if (arguments.HelpRequested()) {
    PrintHelp(argv[0]);
    return benchmarks::Status();
  }
  BENCHMARKS_RETURN_IF_ERROR(arguments.ValidateOptions(
      {"--device", "--duration", "--warmup", "--block-size", "--mode"}));
  benchmarks::CommonConfig config;
  BENCHMARKS_RETURN_IF_ERROR(benchmarks::ParseCommonConfig(arguments, &config));
  const std::string mode = arguments.StringValue("--mode", "system");
  if (mode != "system" && mode != "cuda-context") {
    return benchmarks::Status::Error(
        "--mode must be 'system' or 'cuda-context'");
  }

  std::cout << "BENCHMARK=idle\nVARIANT=" << mode << '\n';
  benchmarks::PrintCommonConfig(config);
  std::cout << "MODE=" << mode << '\n';

  if (mode == "cuda-context") {
    cudaDeviceProp properties;
    BENCHMARKS_RETURN_IF_ERROR(
        benchmarks::InitializeCudaDevice(config.device, &properties));
  }

  const benchmarks::SteadyTimer timer;
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.duration_seconds));
  const double elapsed = timer.ElapsedSeconds();

  benchmarks::PrintExecutionSummary(elapsed, 0);
  return benchmarks::Status();
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
