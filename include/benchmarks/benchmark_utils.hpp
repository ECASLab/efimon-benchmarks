/**
 * @file benchmark_utils.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Command-line parsing and result reporting helpers
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_
#define INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>

#include "benchmarks/benchmark_config.hpp"
#include "benchmarks/status.hpp"

namespace benchmarks {

/**
 * @brief Command-line parser for --option value and --option=value pairs
 *
 * Parse() reads the arguments once. Typed getters convert the stored text on
 * demand and use the default value when the option is absent.
 */
class ArgumentParser {
 public:
  ArgumentParser() = default;
  ArgumentParser(const ArgumentParser&) = default;
  ArgumentParser& operator=(const ArgumentParser&) = default;
  ArgumentParser(ArgumentParser&&) = default;
  ArgumentParser& operator=(ArgumentParser&&) = default;

  /**
   * @brief Parses the command-line arguments
   *
   * @param argc Number of arguments, as received by main()
   * @param argv Arguments, as received by main(). Not retained
   * @return Error on a positional argument, a missing value or a duplicated
   * option
   */
  Status Parse(int argc, char** argv);

  /**
   * @brief Returns whether --help or -h was given
   */
  bool HelpRequested() const;

  /**
   * @brief Returns the option value as text
   *
   * @param option Option name, including the leading "--"
   * @param default_value Value returned when the option is absent
   */
  std::string StringValue(const std::string& option,
                          const std::string& default_value) const;

  /**
   * @brief Reads the option value as a 64-bit integer
   *
   * @param option Option name, including the leading "--"
   * @param default_value Value used when the option is absent
   * @param value Output. Must not be null
   * @return Error if the value is not an integer
   */
  Status Int64Value(const std::string& option, std::int64_t default_value,
                    std::int64_t* value) const;

  /**
   * @brief Reads the option value as an integer
   *
   * @param option Option name, including the leading "--"
   * @param default_value Value used when the option is absent
   * @param value Output. Must not be null
   * @return Error if the value is not an integer
   */
  Status IntegerValue(const std::string& option, int default_value,
                      int* value) const;

  /**
   * @brief Reads the option value as a finite floating-point number
   *
   * @param option Option name, including the leading "--"
   * @param default_value Value used when the option is absent
   * @param value Output. Must not be null
   * @return Error if the value is not a finite number
   */
  Status DoubleValue(const std::string& option, double default_value,
                     double* value) const;

  /**
   * @brief Checks that every given option is supported
   *
   * @param allowed Option names accepted by the benchmark
   * @return Error naming the first unknown option
   */
  Status ValidateOptions(std::initializer_list<const char*> allowed) const;

 private:
  /** Option name to its text value */
  std::unordered_map<std::string, std::string> options_;
  /** Whether --help or -h was given */
  bool help_requested_ = false;
};

/**
 * @brief Reads and validates the options shared by every benchmark
 *
 * @param arguments Parsed command line
 * @param config Output with the given values or the defaults. Must not be
 * null
 * @return Error if a value is invalid or out of range
 */
Status ParseCommonConfig(const ArgumentParser& arguments, CommonConfig* config);

/**
 * @brief Prints the configuration as KEY=VALUE lines on stdout
 *
 * @param config Configuration to print
 */
void PrintCommonConfig(const CommonConfig& config);

/**
 * @brief Prints STATUS=OK, the measured time and the operation count
 *
 * @param elapsed_seconds Duration of the measured phase in seconds
 * @param operation_count Operations or kernel launches in the measured phase
 */
void PrintExecutionSummary(double elapsed_seconds,
                           std::int64_t operation_count);

/**
 * @brief Prints STATUS=ERROR and the error message on stderr
 *
 * @param status Failed status to report
 * @return Exit code for main(), always 1
 */
int ReportError(const Status& status);

}  // namespace benchmarks

#endif  // INCLUDE_BENCHMARKS_BENCHMARK_UTILS_HPP_
