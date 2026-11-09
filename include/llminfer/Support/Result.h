#ifndef LLMINFER_SUPPORT_RESULT_H
#define LLMINFER_SUPPORT_RESULT_H

#include <cassert>
#include <string>
#include <utility>
#include <variant>

namespace llminfer {

/// An empty error represents success.
class [[nodiscard]] Error {
public:
  Error() = default;
  explicit Error(std::string message) : message(std::move(message)) {
    assert(!this->message.empty());
  }

  explicit operator bool() const { return !message.empty(); }
  const std::string &getMessage() const { return message; }

private:
  std::string message;
};

/// A value or its diagnostic. Check the result before accessing either.
template <typename T> class [[nodiscard]] Result {
public:
  Result(T value) : storage(std::move(value)) {}
  Result(Error error) : storage(std::move(error)) {}

  explicit operator bool() const { return std::holds_alternative<T>(storage); }

  T &operator*() { return *getValue(); }
  const T &operator*() const { return *getValue(); }
  T *operator->() { return getValue(); }
  const T *operator->() const { return getValue(); }

  const Error &getError() const {
    const auto *error = std::get_if<Error>(&storage);
    assert(error);
    return *error;
  }

private:
  T *getValue() {
    auto *value = std::get_if<T>(&storage);
    assert(value);
    return value;
  }

  const T *getValue() const {
    const auto *value = std::get_if<T>(&storage);
    assert(value);
    return value;
  }

  std::variant<T, Error> storage;
};

} // namespace llminfer

#endif
