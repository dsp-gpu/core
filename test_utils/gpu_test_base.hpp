#pragma once

/// @file gpu_test_base.hpp
/// @brief GpuTestBase — Template Method для сложных GPU-тестов (10%).
/// Зеркало Python: common/test_base.py (TestBase)

#include <string>
#include <vector>
#include <stdexcept>
#include "test_result.hpp"
#include "test_runner.hpp"
#include "interface/i_backend.hpp"

namespace gpu_test_utils {

class GpuTestBase {
protected:
  drv_gpu_lib::IBackend* backend_;

public:
  explicit GpuTestBase(drv_gpu_lib::IBackend* backend)
      : backend_(backend) {}

  virtual ~GpuTestBase() = default;

  /// Template Method: неизменный скелет теста.
  TestResult Run() {
    TestResult result{GetName()};
    try {
      Setup();
      GenerateInput();
      RunGpu();
      ComputeReference();
      auto validations = Validate();
      for (auto& vr : validations)
        result.add(std::move(vr));
    } catch (const SkipTest& e) {
      result.skipped = true;
      result.skip_reason = e.what();
    } catch (const std::exception& e) {
      result.error = e.what();
    }
    try {
      Teardown();
    } catch (...) {
      // Teardown не должен маскировать ошибки теста
    }
    return result;
  }

protected:
  virtual std::string GetName() = 0;
  virtual void Setup() {}
  virtual void GenerateInput() = 0;
  virtual void RunGpu() = 0;
  virtual void ComputeReference() = 0;
  virtual std::vector<ValidationResult> Validate() = 0;
  virtual void Teardown() {}
};

// ── TestRunner::run() реализация ────────────────────────────────

inline void TestRunner::run(GpuTestBase& test_obj) {
  auto t0 = std::chrono::high_resolution_clock::now();
  auto tr = test_obj.Run();
  auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();
  reporter_.print_header(tr.test_name);
  reporter_.print_test_result(tr, elapsed);
  results_.push_back(std::move(tr));
}

} // namespace gpu_test_utils
