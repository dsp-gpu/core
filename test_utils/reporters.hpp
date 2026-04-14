#pragma once

/// @file reporters.hpp
/// @brief ConsoleTestReporter — форматированный вывод через ConsoleOutput.
/// Review R7: ANSI цвета. Review R8: timing support.
/// Зеркало Python: common/reporters.py (ConsoleReporter)

#include <string>
#include <vector>
#include "test_result.hpp"
#include <core/services/console_output.hpp>

namespace gpu_test_utils {

class ConsoleTestReporter {
  int gpu_id_;
  std::string module_;
  bool use_colors_;

  static constexpr const char* kGreen  = "\033[92m";
  static constexpr const char* kRed    = "\033[91m";
  static constexpr const char* kYellow = "\033[93m";
  static constexpr const char* kBold   = "\033[1m";
  static constexpr const char* kReset  = "\033[0m";

  std::string color(const std::string& text, const char* c) const {
    return use_colors_ ? (std::string(c) + text + kReset) : text;
  }

public:
  ConsoleTestReporter(int gpu_id, const std::string& module,
                      bool use_colors = true)
      : gpu_id_(gpu_id), module_(module), use_colors_(use_colors) {}

  const std::string& module() const { return module_; }

  void print_header(const std::string& test_name) const {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(gpu_id_, module_,
              color("──── " + test_name + " ────", kBold).c_str());
  }

  void print_validation(const ValidationResult& vr) const {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    std::string line = "  " + vr.to_string();
    con.Print(gpu_id_, module_,
              color(line, vr.passed ? kGreen : kRed).c_str());
  }

  void print_test_result(const TestResult& tr, double elapsed_ms = 0.0) const {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    if (tr.skipped) {
      con.Print(gpu_id_, module_,
                color("[SKIP] " + tr.test_name + " (" + tr.skip_reason + ")",
                      kYellow).c_str());
      return;
    }
    for (const auto& vr : tr.validations)
      print_validation(vr);

    std::string status = tr.passed() ? "[PASS]" : "[FAIL]";
    std::string timing = (elapsed_ms > 0.0)
        ? " (" + std::to_string(static_cast<int>(elapsed_ms)) + " ms)"
        : "";
    con.Print(gpu_id_, module_,
              color(status + " " + tr.test_name + timing,
                    tr.passed() ? kGreen : kRed).c_str());

    if (!tr.error.empty())
      con.PrintError(gpu_id_, module_,
                     color("[ERROR] " + tr.test_name + ": " + tr.error,
                           kRed).c_str());
  }

  void print_summary(const std::vector<TestResult>& results) const {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    int n_pass = 0, n_fail = 0, n_skip = 0;
    for (const auto& r : results) {
      if (r.skipped) ++n_skip;
      else if (r.passed()) ++n_pass;
      else ++n_fail;
    }
    con.Print(gpu_id_, module_,
              "════════════════════════════════════════");
    std::string msg = "  SUMMARY: " + std::to_string(n_pass) + " passed, "
                    + std::to_string(n_fail) + " failed";
    if (n_skip > 0) msg += ", " + std::to_string(n_skip) + " skipped";
    con.Print(gpu_id_, module_,
              color(msg, (n_fail == 0) ? kGreen : kRed).c_str());
    con.Print(gpu_id_, module_,
              "════════════════════════════════════════");
  }
};

} // namespace gpu_test_utils
