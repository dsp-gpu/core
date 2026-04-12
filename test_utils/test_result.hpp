#pragma once

/// @file test_result.hpp
/// @brief ValidationResult + TestResult + SkipTest — Value Objects для тестов.
/// Зеркало Python: common/result.py
/// Включает Composite-функционал (add_all, first_failed) — review R4.

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <initializer_list>
#include <exception>

namespace gpu_test_utils {

// ══════════════════════════════════════════════════════════════════
// ValidationResult — результат одной проверки
// ══════════════════════════════════════════════════════════════════

struct ValidationResult {
  bool        passed;
  std::string metric_name;
  double      actual_value;   // double: вмещает float без потерь (Python float = 64 бит)
  double      threshold;      // double: точность для мелких ошибок (1e-7 и т.п.)
  std::string message;

  std::string to_string() const {
    std::ostringstream ss;
    ss << (passed ? "[PASS]" : "[FAIL]") << " " << metric_name
       << ": " << std::scientific << std::setprecision(4) << actual_value
       << " (tol=" << threshold << ")";
    if (!message.empty()) ss << " " << message;
    return ss.str();
  }
};

// ══════════════════════════════════════════════════════════════════
// TestResult — сводный результат одного теста
// ══════════════════════════════════════════════════════════════════

struct TestResult {
  std::string                   test_name;
  std::vector<ValidationResult> validations;
  std::string                   error;         // exception message (пусто если OK)
  bool                          skipped = false;
  std::string                   skip_reason;

  bool passed() const {
    if (!error.empty()) return false;
    if (skipped) return false;
    if (validations.empty()) return false;
    for (const auto& v : validations)
      if (!v.passed) return false;
    return true;
  }

  TestResult& add(ValidationResult vr) {
    validations.push_back(std::move(vr));
    return *this;
  }

  /// Добавить все проверки из initializer_list (Composite — review R4)
  TestResult& add_all(std::initializer_list<ValidationResult> checks) {
    for (const auto& vr : checks)
      validations.push_back(vr);
    return *this;
  }

  int count_passed() const {
    int n = 0;
    for (const auto& v : validations)
      if (v.passed) ++n;
    return n;
  }

  /// Первый FAIL (для отчёта), nullptr если все PASS
  const ValidationResult* first_failed() const {
    for (const auto& v : validations)
      if (!v.passed) return &v;
    return nullptr;
  }

  std::string summary() const {
    std::ostringstream ss;
    ss << (passed() ? "[PASS]" : "[FAIL]") << " " << test_name
       << " (" << count_passed() << "/" << validations.size() << " checks)";
    if (!error.empty()) ss << " ERROR: " << error;
    if (skipped) ss << " SKIP: " << skip_reason;
    return ss.str();
  }
};

// ══════════════════════════════════════════════════════════════════
// SkipTest — исключение для пропуска теста
// ══════════════════════════════════════════════════════════════════

class SkipTest : public std::exception {
  std::string reason_;
public:
  explicit SkipTest(const std::string& reason) : reason_(reason) {}
  const char* what() const noexcept override { return reason_.c_str(); }
};

// ── Удобные фабричные функции ────────────────────────────────────

inline ValidationResult PassResult(const std::string& name,
                                   double value = 0.0,
                                   double threshold = 0.0,
                                   const std::string& msg = "") {
  return {true, name, value, threshold, msg};
}

inline ValidationResult FailResult(const std::string& name,
                                   double value = 0.0,
                                   double threshold = 0.0,
                                   const std::string& msg = "") {
  return {false, name, value, threshold, msg};
}

} // namespace gpu_test_utils
