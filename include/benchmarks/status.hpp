/**
 * @file status.hpp
 * @author Kevin Lobo (2020087823@estudiantec.cr)
 * @brief Result type used to report errors without exceptions
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_BENCHMARKS_STATUS_HPP_
#define INCLUDE_BENCHMARKS_STATUS_HPP_

#include <string>
#include <utility>

namespace benchmarks {

/**
 * @brief Outcome of an operation: success, or an error with a message
 *
 * Functions that can fail return a Status, and callers must check it.
 */
class [[nodiscard]] Status {
 public:
  /**
   * @brief Creates a successful status
   */
  Status() = default;
  Status(const Status&) = default;
  Status& operator=(const Status&) = default;
  Status(Status&&) = default;
  Status& operator=(Status&&) = default;

  /**
   * @brief Returns an error status
   *
   * @param message Description of the error shown to the user
   */
  static Status Error(std::string message) {
    return Status(std::move(message));
  }

  /**
   * @brief Returns whether the operation succeeded
   */
  bool ok() const { return ok_; }

  /**
   * @brief Returns the error description, empty on success
   */
  const std::string& message() const { return message_; }

 private:
  explicit Status(std::string message)
      : ok_(false), message_(std::move(message)) {}

  /** Whether the operation succeeded */
  bool ok_ = true;
  /** Error description, empty on success */
  std::string message_;
};

}  // namespace benchmarks

/**
 * @brief Returns the status from the current function if it is an error
 */
#define BENCHMARKS_RETURN_IF_ERROR(expression)                    \
  do {                                                            \
    const ::benchmarks::Status benchmarks_status_ = (expression); \
    if (!benchmarks_status_.ok()) {                               \
      return benchmarks_status_;                                  \
    }                                                             \
  } while (false)

#endif  // INCLUDE_BENCHMARKS_STATUS_HPP_
