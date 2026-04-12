#pragma once

/// @file test_runner.hpp
/// @brief TestRunner — координатор тестов (функциональный + классовый API).
/// Review R6: timing (chrono). Review R8: JSON export.
/// Зеркало Python: common/runner.py (TestRunner)

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include "test_result.hpp"
#include "reporters.hpp"
#include "interface/i_backend.hpp"

namespace gpu_test_utils {

// forward
class GpuTestBase;

class TestRunner {
  drv_gpu_lib::IBackend* backend_;
  ConsoleTestReporter    reporter_;
  std::vector<TestResult> results_;

public:
  TestRunner(drv_gpu_lib::IBackend* backend,
             const std::string& module, int gpu_id = 0)
      : backend_(backend)
      , reporter_(gpu_id, module)
  {}

  // ── Функциональный стиль (90% тестов) ──────────────────────────

  /// Тест с одной проверкой → ValidationResult
  void test(const std::string& name,
            std::function<ValidationResult()> test_fn)
  {
    reporter_.print_header(name);
    TestResult tr{name};
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
      auto vr = test_fn();
      tr.add(vr);
    } catch (const SkipTest& e) {
      tr.skipped = true;
      tr.skip_reason = e.what();
    } catch (const std::exception& e) {
      tr.error = e.what();
    }
    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    reporter_.print_test_result(tr, elapsed);
    results_.push_back(std::move(tr));
  }

  /// Тест с множественными проверками → TestResult
  void test(const std::string& name,
            std::function<TestResult()> test_fn)
  {
    reporter_.print_header(name);
    TestResult tr{name};
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
      tr = test_fn();
      tr.test_name = name;
    } catch (const SkipTest& e) {
      tr.skipped = true;
      tr.skip_reason = e.what();
    } catch (const std::exception& e) {
      tr.error = e.what();
    }
    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    reporter_.print_test_result(tr, elapsed);
    results_.push_back(std::move(tr));
  }

  // ── Классовый стиль ───────────────────────────────────────────

  void run(GpuTestBase& test_obj);  // определяется в gpu_test_base.hpp

  // ── Итоги ─────────────────────────────────────────────────────

  void print_summary() const { reporter_.print_summary(results_); }

  const std::vector<TestResult>& results() const { return results_; }

  int count_passed() const {
    int n = 0;
    for (const auto& r : results_) if (r.passed()) ++n;
    return n;
  }

  int count_failed() const {
    int n = 0;
    for (const auto& r : results_)
      if (!r.passed() && !r.skipped) ++n;
    return n;
  }

  bool all_passed() const { return count_failed() == 0; }

  drv_gpu_lib::IBackend* backend() const { return backend_; }

  // ── JSON export (review R8) ───────────────────────────────────

  bool export_json(const std::string& file_path) const {
    std::ofstream f(file_path);
    if (!f.is_open()) return false;
    f << "{\n  \"module\": \"" << reporter_.module() << "\",\n";
    f << "  \"total\": " << results_.size() << ",\n";
    f << "  \"passed\": " << count_passed() << ",\n";
    f << "  \"failed\": " << count_failed() << ",\n";
    f << "  \"tests\": [\n";
    for (size_t i = 0; i < results_.size(); ++i) {
      const auto& r = results_[i];
      f << "    {\"name\": \"" << r.test_name << "\", "
        << "\"passed\": " << (r.passed() ? "true" : "false") << ", "
        << "\"skipped\": " << (r.skipped ? "true" : "false") << ", "
        << "\"checks\": " << r.validations.size() << "}";
      if (i + 1 < results_.size()) f << ",";
      f << "\n";
    }
    f << "  ]\n}\n";
    return true;
  }
};

} // namespace gpu_test_utils
