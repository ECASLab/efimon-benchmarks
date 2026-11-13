/**
 * @file benchmark_utils.cpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Command-line parsing and result reporting helpers
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include "benchmarks/benchmark_utils.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>

namespace benchmarks {
namespace {

// Converts the whole text to an integer; trailing characters are an error.
template <typename Integer>
bool ParseInteger(const std::string& text, Integer* value) {
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, *value);
  return !text.empty() && result.ec == std::errc() && result.ptr == end;
}

}  // namespace

Status ArgumentParser::Parse(int argc, char** argv) {
  options_.clear();
  help_requested_ = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      help_requested_ = true;
      continue;
    }
    if (argument.rfind("--", 0) != 0 || argument.size() == 2) {
      return Status::Error("unexpected positional argument: " + argument);
    }

    std::string option;
    std::string value;
    const auto equals = argument.find('=');
    if (equals != std::string::npos) {
      option = argument.substr(0, equals);
      value = argument.substr(equals + 1);
      if (value.empty()) {
        return Status::Error("missing value for " + option);
      }
    } else {
      option = argument;
      if (index + 1 >= argc ||
          std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
        return Status::Error("missing value for " + option);
      }
      value = argv[++index];
    }

    if (!options_.emplace(option, value).second) {
      return Status::Error("duplicate option: " + option);
    }
  }
  return Status();
}

bool ArgumentParser::HelpRequested() const { return help_requested_; }

std::string ArgumentParser::StringValue(
    const std::string& option, const std::string& default_value) const {
  const auto found = options_.find(option);
  return found == options_.end() ? default_value : found->second;
}

Status ArgumentParser::Int64Value(const std::string& option,
                                  std::int64_t default_value,
                                  std::int64_t* value) const {
  const auto found = options_.find(option);
  if (found == options_.end()) {
    *value = default_value;
    return Status();
  }
  if (!ParseInteger(found->second, value)) {
    return Status::Error("invalid integer for " + option + ": " +
                         found->second);
  }
  return Status();
}

Status ArgumentParser::IntegerValue(const std::string& option,
                                    int default_value, int* value) const {
  const auto found = options_.find(option);
  if (found == options_.end()) {
    *value = default_value;
    return Status();
  }
  if (!ParseInteger(found->second, value)) {
    return Status::Error("invalid integer for " + option + ": " +
                         found->second);
  }
  return Status();
}

Status ArgumentParser::DoubleValue(const std::string& option,
                                   double default_value, double* value) const {
  const auto found = options_.find(option);
  if (found == options_.end()) {
    *value = default_value;
    return Status();
  }
  const char* begin = found->second.c_str();
  char* end = nullptr;
  const double parsed = std::strtod(begin, &end);
  if (end == begin || end != begin + found->second.size() ||
      !std::isfinite(parsed)) {
    return Status::Error("invalid number for " + option + ": " + found->second);
  }
  *value = parsed;
  return Status();
}

Status ArgumentParser::ValidateOptions(
    std::initializer_list<const char*> allowed) const {
  std::unordered_set<std::string> allowed_set;
  for (const char* option : allowed) {
    allowed_set.emplace(option);
  }
  for (const auto& [option, unused_value] : options_) {
    (void)unused_value;
    if (allowed_set.find(option) == allowed_set.end()) {
      return Status::Error("unknown option: " + option);
    }
  }
  return Status();
}

Status ParseCommonConfig(const ArgumentParser& arguments,
                         CommonConfig* config) {
  BENCHMARKS_RETURN_IF_ERROR(
      arguments.IntegerValue("--device", kDefaultDevice, &config->device));
  BENCHMARKS_RETURN_IF_ERROR(arguments.DoubleValue(
      "--duration", kDefaultDurationSeconds, &config->duration_seconds));
  BENCHMARKS_RETURN_IF_ERROR(arguments.Int64Value(
      "--warmup", kDefaultWarmupIterations, &config->warmup_iterations));
  BENCHMARKS_RETURN_IF_ERROR(arguments.IntegerValue(
      "--block-size", kDefaultBlockSize, &config->block_size));

  if (config->device < 0) {
    return Status::Error("--device must be non-negative");
  }
  if (!(config->duration_seconds > 0.0)) {
    return Status::Error("--duration must be greater than zero");
  }
  if (config->warmup_iterations < 0) {
    return Status::Error("--warmup must be non-negative");
  }
  if (config->block_size <= 0 || config->block_size > 1024) {
    return Status::Error("--block-size must be in the range [1, 1024]");
  }
  return Status();
}

void PrintCommonConfig(const CommonConfig& config) {
  std::cout << "DEVICE=" << config.device << '\n'
            << "DURATION_SECONDS=" << std::setprecision(12)
            << config.duration_seconds << '\n'
            << "BLOCK_SIZE=" << config.block_size << '\n'
            << "WARMUP=" << config.warmup_iterations << '\n';
}

void PrintExecutionSummary(double elapsed_seconds,
                           std::int64_t operation_count) {
  std::cout << "STATUS=OK\n"
            << "EXECUTION_TIME_SECONDS=" << std::fixed << std::setprecision(6)
            << elapsed_seconds << '\n'
            << "OPERATIONS_OR_KERNEL_LAUNCHES=" << operation_count << '\n';
}

int ReportError(const Status& status) {
  std::cerr << "STATUS=ERROR\nERROR=" << status.message() << '\n';
  return 1;
}

}  // namespace benchmarks
