/**
 * @file benchmark_utils.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Command-line parsing and result reporting helpers
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_
#define INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_

#include "benchmarks/benchmark_config.hpp"

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>

namespace benchmarks {

class ArgumentParser {
 public:
  ArgumentParser(int argc, char** argv);

  bool help_requested() const;
  bool has(const std::string& option) const;
  std::string string_value(const std::string& option,
                           const std::string& default_value) const;
  std::uint64_t unsigned_value(const std::string& option,
                               std::uint64_t default_value) const;
  int integer_value(const std::string& option, int default_value) const;
  double double_value(const std::string& option, double default_value) const;
  void validate_options(std::initializer_list<const char*> allowed) const;

 private:
  std::unordered_map<std::string, std::string> options_;
  bool help_requested_ = false;
};

CommonConfig parse_common_config(const ArgumentParser& arguments);
void print_common_config(const CommonConfig& config);
void print_execution_summary(double elapsed_seconds,
                             std::uint64_t operation_count);
double elapsed_seconds_since(const std::chrono::steady_clock::time_point& start);

}  // namespace benchmarks

#endif  // INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_
