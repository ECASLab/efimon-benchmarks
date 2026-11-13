/**
 * @file timing.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Steady-clock timer for the measured phase
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_TIMING_HPP_
#define INCLUDE_BENCHMARKS_TIMING_HPP_

#include <chrono>

namespace benchmarks {

class SteadyTimer {
 public:
  SteadyTimer() : start_(std::chrono::steady_clock::now()) {}

  double ElapsedSeconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace benchmarks

#endif  // INCLUDE_BENCHMARKS_TIMING_HPP_
