/**
 * @file benchmark_config.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Default values and shared benchmark configuration
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_BENCHMARK_CONFIG_HPP_
#define INCLUDE_BENCHMARKS_BENCHMARK_CONFIG_HPP_

#include <cstddef>

namespace benchmarks {

inline constexpr int kDefaultDevice = 0;
inline constexpr double kDefaultDurationSeconds = 10.0;
inline constexpr std::size_t kDefaultWarmupIterations = 5;
inline constexpr int kDefaultBlockSize = 256;
inline constexpr std::size_t kDefaultBufferBytes = 256ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kLaunchesBetweenSynchronizations = 16;

struct CommonConfig {
  int device = kDefaultDevice;
  double duration_seconds = kDefaultDurationSeconds;
  std::size_t warmup_iterations = kDefaultWarmupIterations;
  int block_size = kDefaultBlockSize;
};

}  // namespace benchmarks

#endif  // INCLUDE_BENCHMARKS_BENCHMARK_CONFIG_HPP_
