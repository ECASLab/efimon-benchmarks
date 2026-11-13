/**
 * @file benchmark_utils.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Command-line parsing and result reporting helpers
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_
#define INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>

#include "benchmarks/benchmark_config.hpp"

namespace benchmarks {

class ArgumentParser {
 public:
  ArgumentParser(int argc, char** argv);
  ArgumentParser(const ArgumentParser&) = default;
  ArgumentParser& operator=(const ArgumentParser&) = default;
  ArgumentParser(ArgumentParser&&) = default;
  ArgumentParser& operator=(ArgumentParser&&) = default;

  bool HelpRequested() const;
  bool Has(const std::string& option) const;
  std::string StringValue(const std::string& option,
                          const std::string& default_value) const;
  std::uint64_t UnsignedValue(const std::string& option,
                              std::uint64_t default_value) const;
  int IntegerValue(const std::string& option, int default_value) const;
  double DoubleValue(const std::string& option, double default_value) const;
  void ValidateOptions(std::initializer_list<const char*> allowed) const;

 private:
  std::unordered_map<std::string, std::string> options_;
  bool help_requested_ = false;
};

CommonConfig ParseCommonConfig(const ArgumentParser& arguments);
void PrintCommonConfig(const CommonConfig& config);
void PrintExecutionSummary(double elapsed_seconds,
                           std::uint64_t operation_count);
double ElapsedSecondsSince(const std::chrono::steady_clock::time_point& start);

}  // namespace benchmarks

#endif  // INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_
