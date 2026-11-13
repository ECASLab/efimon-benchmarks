/**
 * @file benchmark_config.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Default values and shared benchmark configuration
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_BENCHMARK_CONFIG_HPP_
#define INCLUDE_BENCHMARKS_BENCHMARK_CONFIG_HPP_

#include <cstdint>

namespace benchmarks {

/** Default CUDA device index */
inline constexpr int kDefaultDevice = 0;
/** Default duration of the measured phase in seconds */
inline constexpr double kDefaultDurationSeconds = 10.0;
/** Default number of warm-up operations */
inline constexpr std::int64_t kDefaultWarmupIterations = 5;
/** Default number of threads per CUDA block */
inline constexpr int kDefaultBlockSize = 256;
/** Default buffer size in bytes (256 MiB) */
inline constexpr std::int64_t kDefaultBufferBytes = 268435456;
/** Operations launched between two device synchronizations */
inline constexpr int kLaunchesBetweenSynchronizations = 16;

/**
 * @brief Options shared by every benchmark
 */
struct CommonConfig {
  /** CUDA device index */
  int device = kDefaultDevice;
  /** Duration of the measured phase in seconds */
  double duration_seconds = kDefaultDurationSeconds;
  /** Warm-up operations before the measured phase */
  std::int64_t warmup_iterations = kDefaultWarmupIterations;
  /** Threads per CUDA block */
  int block_size = kDefaultBlockSize;
};

}  // namespace benchmarks

#endif  // INCLUDE_BENCHMARKS_BENCHMARK_CONFIG_HPP_
