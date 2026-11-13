/**
 * @file bench_power_cycle.cu
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Phased, duty-cycle-controlled power benchmark
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "benchmarks/benchmark_config.hpp"
#include "benchmarks/benchmark_utils.hpp"
#include "benchmarks/cuda_utils.hpp"
#include "benchmarks/status.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultWarmupSeconds = 15.0;
constexpr double kDefaultMeasurementSeconds = 45.0;
constexpr double kDefaultCooldownSeconds = 15.0;
constexpr double kDefaultDutyCyclePct = 100.0;
constexpr double kDefaultPeriodMs = 100.0;
constexpr int kDefaultComputeIterations = 10000;
constexpr int kDefaultMixedComputePct = 50;

/** Kind of GPU work launched in the active part of each period */
enum class Workload { kIdle, kCompute, kMemory, kMixed };

/**
 * @brief Options of one benchmark condition
 */
struct Config {
  /** CUDA device index */
  int device = benchmarks::kDefaultDevice;
  /** Threads per CUDA block */
  int block_size = benchmarks::kDefaultBlockSize;
  /** Duration of the paced warm-up phase */
  double warmup_seconds = kDefaultWarmupSeconds;
  /** Duration of the paced measurement phase */
  double measurement_seconds = kDefaultMeasurementSeconds;
  /** Duration of the context-idle cooldown phase */
  double cooldown_seconds = kDefaultCooldownSeconds;
  /** Active share of each period, in [0, 100] */
  double duty_cycle_pct = kDefaultDutyCyclePct;
  /** Length of one duty-control period */
  double period_ms = kDefaultPeriodMs;
  /** Bytes per memory buffer */
  std::size_t buffer_bytes = benchmarks::kDefaultBufferBytes;
  /** FMA steps per chain in the compute kernel */
  int compute_iterations = kDefaultComputeIterations;
  /** Share of compute launches in the mixed workload, in [0, 100] */
  int mixed_compute_pct = kDefaultMixedComputePct;
  /** Kind of GPU work */
  Workload workload = Workload::kCompute;
};

/**
 * @brief Device buffers and launch sizes of the selected workload
 */
struct DeviceResources {
  /** Output of the compute kernel, one float per thread */
  benchmarks::DeviceBuffer<float> compute_output;
  /** Source buffer of the memory kernel */
  benchmarks::DeviceBuffer<float> memory_source;
  /** Destination buffer of the memory kernel */
  benchmarks::DeviceBuffer<float> memory_destination;
  /** Blocks per compute launch */
  int compute_grid_size = 0;
  /** Blocks per memory launch */
  int memory_grid_size = 0;
  /** Floats per memory buffer */
  std::size_t memory_elements = 0;
};

/**
 * @brief Counters of one paced phase
 */
struct PhaseStats {
  /** Wall time of the phase */
  double elapsed_seconds = 0.0;
  /** Time spent launching and waiting for GPU work */
  double active_seconds = 0.0;
  /** Kernel launches in the phase */
  std::int64_t operations = 0;
  /** Periods whose work did not finish before the period ended */
  std::int64_t period_overruns = 0;
};

// The marker kernels keep their snake_case names on purpose: the analysis
// scripts find the phase boundaries by these names in the CUPTI kernel trace,
// and other workloads emit the same names. Renaming them breaks that format.

/** Marks the start of the warm-up phase in the kernel trace */
__global__ void benchmark_warmup_start_marker() {}
/** Marks the start of the measurement phase in the kernel trace */
__global__ void benchmark_measurement_start_marker() {}
/** Marks the start of the cooldown phase in the kernel trace */
__global__ void benchmark_cooldown_start_marker() {}
/** Marks the end of the benchmark in the kernel trace */
__global__ void benchmark_complete_marker() {}

/**
 * @brief Runs eight independent FP32 FMA chains per thread in registers
 *
 * Each thread writes one value so that the compiler keeps the loop.
 *
 * @param output One float per thread
 * @param iterations FMA steps per chain
 */
__global__ void PowerCycleFmaKernel(float* output, int iterations) {
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
 * @brief Copies source into destination with a grid-stride loop
 *
 * @param destination Output buffer with at least elements floats
 * @param source Input buffer with at least elements floats
 * @param elements Number of floats to copy
 */
__global__ void PowerCycleMemoryKernel(float* destination, const float* source,
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
      << "Phased, duty-cycle-controlled CUDA power benchmark.\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --workload <name>         idle, compute, memory, or mixed\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 15)\n"
      << "  --duration <seconds>      Stable measurement phase (default: 45)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 15)\n"
      << "  --duty-cycle <percent>    Active part of each period [0, 100]\n"
      << "  --period-ms <value>       Duty-control period (default: 100)\n"
      << "  --block-size <threads>    CUDA block size (default: 256)\n"
      << "  --iterations <N>          Compute-kernel FMA iterations\n"
      << "  --size <bytes>            Bytes per memory buffer\n"
      << "  --mixed-compute-pct <N>   Compute operation share [0, 100]\n"
      << "  --help                     Show this help\n";
}

/**
 * @brief Converts the --workload value into a Workload
 *
 * @param value idle, compute, memory or mixed
 * @param workload Output. Must not be null
 * @return Error if the value is not a known workload
 */
benchmarks::Status ParseWorkload(const std::string& value, Workload* workload) {
  if (value == "idle") {
    *workload = Workload::kIdle;
  } else if (value == "compute") {
    *workload = Workload::kCompute;
  } else if (value == "memory") {
    *workload = Workload::kMemory;
  } else if (value == "mixed") {
    *workload = Workload::kMixed;
  } else {
    return benchmarks::Status::Error(
        "--workload must be idle, compute, memory, or mixed");
  }
  return benchmarks::Status();
}

/**
 * @brief Returns the --workload value of a Workload
 */
const char* WorkloadName(Workload workload) {
  switch (workload) {
    case Workload::kIdle:
      return "idle";
    case Workload::kCompute:
      return "compute";
    case Workload::kMemory:
      return "memory";
    case Workload::kMixed:
      return "mixed";
  }
  return "unknown";
}

/**
 * @brief Reads and validates the options of the benchmark
 *
 * @param arguments Parsed command line
 * @param config Output with the given values or the defaults. Must not be
 * null
 * @return Error if an option is unknown, invalid or out of range
 */
benchmarks::Status ParseConfig(const benchmarks::ArgumentParser& arguments,
                               Config* config) {
  BENCHMARKS_RETURN_IF_ERROR(arguments.ValidateOptions(
      {"--device", "--workload", "--warmup-seconds", "--duration",
       "--cooldown-seconds", "--duty-cycle", "--period-ms", "--block-size",
       "--iterations", "--size", "--mixed-compute-pct"}));

  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--device", config->device, &config->device));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--block-size", config->block_size, &config->block_size));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--warmup-seconds", config->warmup_seconds, &config->warmup_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--duration", config->measurement_seconds, &config->measurement_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue("--cooldown-seconds",
                                                   config->cooldown_seconds,
                                                   &config->cooldown_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--duty-cycle", config->duty_cycle_pct, &config->duty_cycle_pct));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--period-ms", config->period_ms, &config->period_ms));
  BENCHMARKS_RETURN_IF_ERROR(ParseWorkload(
      arguments.StringValue("--workload", "compute"), &config->workload));

  std::int64_t size_value = 0;
  BENCHMARKS_RETURN_IF_ERROR(arguments.Int64Value(
      "--size", benchmarks::kDefaultBufferBytes, &size_value));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--iterations", kDefaultComputeIterations, &config->compute_iterations));
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--mixed-compute-pct", kDefaultMixedComputePct,
                             &config->mixed_compute_pct));

  if (config->device < 0) {
    return benchmarks::Status::Error("--device must be non-negative");
  }
  if (config->block_size <= 0 || config->block_size > 1024) {
    return benchmarks::Status::Error(
        "--block-size must be in the range [1, 1024]");
  }
  if (!(config->warmup_seconds >= 0.0) ||
      !(config->measurement_seconds > 0.0) ||
      !(config->cooldown_seconds >= 0.0)) {
    return benchmarks::Status::Error(
        "phase durations must be finite; measurement must be positive");
  }
  if (config->duty_cycle_pct < 0.0 || config->duty_cycle_pct > 100.0) {
    return benchmarks::Status::Error(
        "--duty-cycle must be in the range [0, 100]");
  }
  if (!(config->period_ms > 0.0)) {
    return benchmarks::Status::Error("--period-ms must be greater than zero");
  }
  if (size_value <= 0 ||
      size_value % static_cast<std::int64_t>(sizeof(float)) != 0) {
    return benchmarks::Status::Error(
        "--size must be a positive multiple of 4 representable by size_t");
  }
  if (config->compute_iterations <= 0) {
    return benchmarks::Status::Error(
        "--iterations must be in the range [1, 2^31-1]");
  }
  if (config->mixed_compute_pct < 0 || config->mixed_compute_pct > 100) {
    return benchmarks::Status::Error("--mixed-compute-pct must be in [0, 100]");
  }
  if (config->workload == Workload::kIdle && config->duty_cycle_pct != 0.0) {
    return benchmarks::Status::Error("idle workload requires --duty-cycle 0");
  }

  config->buffer_bytes = static_cast<std::size_t>(size_value);
  return benchmarks::Status();
}

/**
 * @brief Prints the configuration as KEY=VALUE lines on stdout
 */
void PrintConfig(const Config& config) {
  std::cout << std::setprecision(12) << "BENCHMARK=power_cycle\n"
            << "WORKLOAD=" << WorkloadName(config.workload) << '\n'
            << "DEVICE=" << config.device << '\n'
            << "WARMUP_SECONDS=" << config.warmup_seconds << '\n'
            << "MEASUREMENT_SECONDS=" << config.measurement_seconds << '\n'
            << "COOLDOWN_SECONDS=" << config.cooldown_seconds << '\n'
            << "DUTY_CYCLE_PCT=" << config.duty_cycle_pct << '\n'
            << "PERIOD_MS=" << config.period_ms << '\n'
            << "BLOCK_SIZE=" << config.block_size << '\n'
            << "ITERATIONS=" << config.compute_iterations << '\n'
            << "BYTES=" << config.buffer_bytes << '\n'
            << "MIXED_COMPUTE_PCT=" << config.mixed_compute_pct << '\n';
}

/** Phase boundary recorded with a marker kernel */
enum class PhaseMarker { kWarmup, kMeasurement, kCooldown, kComplete };

/**
 * @brief Launches the marker kernel of a phase and prints PHASE=<phase>
 *
 * @param phase Phase name printed on stdout
 * @param marker Marker kernel to launch
 * @return Error if the launch or the synchronization fails
 */
benchmarks::Status EmitPhase(const char* phase, PhaseMarker marker) {
  switch (marker) {
    case PhaseMarker::kWarmup:
      benchmark_warmup_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kMeasurement:
      benchmark_measurement_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kCooldown:
      benchmark_cooldown_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kComplete:
      benchmark_complete_marker<<<1, 1>>>();
      break;
  }
  BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "PHASE=" << phase << '\n' << std::flush;
  return benchmarks::Status();
}

/**
 * @brief Allocates the buffers needed by the selected workload
 *
 * @param config Benchmark options
 * @param properties Properties of the selected device
 * @param resources Output buffers and launch sizes. Must not be null
 * @return Error if a CUDA allocation fails
 */
benchmarks::Status AllocateResources(const Config& config,
                                     const cudaDeviceProp& properties,
                                     DeviceResources* resources) {
  const bool needs_compute = config.workload == Workload::kCompute ||
                             config.workload == Workload::kMixed;
  const bool needs_memory = config.workload == Workload::kMemory ||
                            config.workload == Workload::kMixed;
  if (needs_compute) {
    resources->compute_grid_size = properties.multiProcessorCount * 8;
    const std::size_t output_elements =
        static_cast<std::size_t>(resources->compute_grid_size) *
        static_cast<std::size_t>(config.block_size);
    float* output = nullptr;
    BENCHMARKS_CUDA_CHECK(cudaMalloc(&output, output_elements * sizeof(float)));
    resources->compute_output.reset(output);
  }
  if (needs_memory) {
    resources->memory_elements = config.buffer_bytes / sizeof(float);
    const std::size_t required_blocks =
        (resources->memory_elements + config.block_size - 1) /
        static_cast<std::size_t>(config.block_size);
    resources->memory_grid_size = static_cast<int>(std::min<std::size_t>(
        required_blocks,
        static_cast<std::size_t>(properties.multiProcessorCount) * 32));
    float* source = nullptr;
    BENCHMARKS_CUDA_CHECK(cudaMalloc(&source, config.buffer_bytes));
    resources->memory_source.reset(source);
    float* destination = nullptr;
    BENCHMARKS_CUDA_CHECK(cudaMalloc(&destination, config.buffer_bytes));
    resources->memory_destination.reset(destination);
    BENCHMARKS_CUDA_CHECK(cudaMemset(source, 0x3f, config.buffer_bytes));
    BENCHMARKS_CUDA_CHECK(cudaMemset(destination, 0, config.buffer_bytes));
  }
  return benchmarks::Status();
}

/**
 * @brief Runs one compute kernel and waits for it
 */
benchmarks::Status LaunchCompute(const Config& config,
                                 const DeviceResources& resources) {
  PowerCycleFmaKernel<<<resources.compute_grid_size, config.block_size>>>(
      resources.compute_output.get(), config.compute_iterations);
  BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  return benchmarks::Status();
}

/**
 * @brief Runs one memory kernel and waits for it
 */
benchmarks::Status LaunchMemory(const Config& config,
                                const DeviceResources& resources) {
  PowerCycleMemoryKernel<<<resources.memory_grid_size, config.block_size>>>(
      resources.memory_destination.get(), resources.memory_source.get(),
      resources.memory_elements);
  BENCHMARKS_CUDA_CHECK(cudaGetLastError());
  BENCHMARKS_CUDA_CHECK(cudaDeviceSynchronize());
  return benchmarks::Status();
}

/**
 * @brief Runs one operation of the selected workload
 *
 * The mixed workload adds mixed_compute_pct to an accumulator on each call
 * and runs a compute kernel every time it reaches 100, so the compute share
 * is exact over time.
 *
 * @param config Benchmark options
 * @param resources Allocated buffers
 * @param mixed_accumulator State of the mixed workload. Must not be null
 * @return Error if a kernel launch fails
 */
benchmarks::Status LaunchOperation(const Config& config,
                                   const DeviceResources& resources,
                                   int* mixed_accumulator) {
  if (config.workload == Workload::kCompute) {
    return LaunchCompute(config, resources);
  }
  if (config.workload == Workload::kMemory) {
    return LaunchMemory(config, resources);
  }
  if (config.workload == Workload::kMixed) {
    *mixed_accumulator += config.mixed_compute_pct;
    if (*mixed_accumulator >= 100) {
      *mixed_accumulator -= 100;
      return LaunchCompute(config, resources);
    }
    return LaunchMemory(config, resources);
  }
  return benchmarks::Status();
}

/**
 * @brief Converts seconds into a steady-clock duration
 */
Clock::duration SecondsToDuration(double seconds) {
  return std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(seconds));
}

/**
 * @brief Alternates GPU work and idle time for the length of a phase
 *
 * Each period launches work for duty_cycle_pct of period_ms and sleeps for
 * the rest. A period whose work takes longer counts as an overrun.
 *
 * @param config Benchmark options
 * @param resources Allocated buffers
 * @param phase_seconds Length of the phase
 * @param stats Output counters of the phase. Must not be null
 * @return Error if a kernel launch fails
 */
benchmarks::Status RunPacedPhase(const Config& config,
                                 const DeviceResources& resources,
                                 double phase_seconds, PhaseStats* stats) {
  *stats = PhaseStats();
  if (phase_seconds <= 0.0) {
    return benchmarks::Status();
  }

  const auto phase_start = Clock::now();
  const auto phase_end = phase_start + SecondsToDuration(phase_seconds);
  const auto period = SecondsToDuration(config.period_ms / 1000.0);
  const auto active = std::chrono::duration_cast<Clock::duration>(
      period * (config.duty_cycle_pct / 100.0));
  auto period_start = phase_start;
  int mixed_accumulator = 0;

  while (Clock::now() < phase_end) {
    const auto period_end = std::min(period_start + period, phase_end);
    const auto active_end = std::min(period_start + active, period_end);
    const auto active_start = Clock::now();
    const std::int64_t operations_before_period = stats->operations;
    if (config.workload != Workload::kIdle && config.duty_cycle_pct > 0.0) {
      while (Clock::now() < active_end) {
        BENCHMARKS_RETURN_IF_ERROR(
            LaunchOperation(config, resources, &mixed_accumulator));
        ++stats->operations;
      }
    }
    const auto work_end = Clock::now();
    if (stats->operations > operations_before_period) {
      stats->active_seconds +=
          std::chrono::duration<double>(work_end - active_start).count();
    }
    if (work_end > period_end) {
      ++stats->period_overruns;
      period_start = work_end;
    } else {
      std::this_thread::sleep_until(period_end);
      period_start = period_end;
    }
  }
  stats->elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - phase_start).count();
  return benchmarks::Status();
}

/**
 * @brief Prints the counters of a phase as <prefix>_KEY=VALUE lines
 */
void PrintPhaseStats(const char* prefix, const PhaseStats& stats) {
  std::cout << std::fixed << std::setprecision(6) << prefix
            << "_ELAPSED_SECONDS=" << stats.elapsed_seconds << '\n'
            << prefix << "_ACTIVE_SECONDS=" << stats.active_seconds << '\n'
            << prefix << "_OPERATIONS=" << stats.operations << '\n'
            << prefix << "_PERIOD_OVERRUNS=" << stats.period_overruns << '\n';
}

/**
 * @brief Parses the options, runs the warm-up, measurement and cooldown
 * phases and prints the results
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
  Config config;
  BENCHMARKS_RETURN_IF_ERROR(ParseConfig(arguments, &config));
  PrintConfig(config);

  cudaDeviceProp properties;
  BENCHMARKS_RETURN_IF_ERROR(
      benchmarks::InitializeCudaDevice(config.device, &properties));
  if (config.block_size > properties.maxThreadsPerBlock) {
    return benchmarks::Status::Error("--block-size exceeds the device limit");
  }
  DeviceResources resources;
  BENCHMARKS_RETURN_IF_ERROR(AllocateResources(config, properties, &resources));

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("warmup", PhaseMarker::kWarmup));
  PhaseStats warmup;
  BENCHMARKS_RETURN_IF_ERROR(
      RunPacedPhase(config, resources, config.warmup_seconds, &warmup));

  BENCHMARKS_RETURN_IF_ERROR(
      EmitPhase("measurement", PhaseMarker::kMeasurement));
  PhaseStats measurement;
  BENCHMARKS_RETURN_IF_ERROR(RunPacedPhase(
      config, resources, config.measurement_seconds, &measurement));

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("cooldown", PhaseMarker::kCooldown));
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();

  BENCHMARKS_RETURN_IF_ERROR(EmitPhase("complete", PhaseMarker::kComplete));
  resources = DeviceResources();

  PrintPhaseStats("WARMUP", warmup);
  PrintPhaseStats("MEASUREMENT", measurement);
  std::cout << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed << '\n';
  benchmarks::PrintExecutionSummary(measurement.elapsed_seconds,
                                    measurement.operations);
  return benchmarks::Status();
}

}  // namespace

int main(int argc, char** argv) {
  const benchmarks::Status status = Run(argc, argv);
  return status.ok() ? 0 : benchmarks::ReportError(status);
}
