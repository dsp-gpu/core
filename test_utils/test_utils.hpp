#pragma once

/// @file test_utils.hpp
/// @brief Master include — единая C++ тестовая инфраструктура GPUWorkLib.
///
/// Зеркало Python: Python_test/common/
///
/// Использование (функциональный стиль, 90% тестов):
///   #include "modules/test_utils/test_utils.hpp"
///   using namespace gpu_test_utils;
///
///   void run(IBackend* backend) {
///       TestRunner runner(backend, "MyModule");
///       runner.test("test_name", [&]() {
///           auto data = refs::GenerateCw(12e6f, 4096, 2e6f);
///           // ... GPU compute ...
///           return MaxRelError(gpu, cpu, n, tolerance::kComplex32, "signal");
///       });
///       runner.print_summary();
///   }
///
/// Использование (классовый стиль, 10% сложных):
///   class MyTest : public GpuTestBase { ... };
///   runner.run(MyTest(backend));

// Value Objects + Config
#include "test_result.hpp"
#include "test_configs.hpp"

// References (CPU-эталоны)
#include "references/signal_refs.hpp"
#include "references/statistics_refs.hpp"
#include "references/fft_refs.hpp"

// Validators (review R4: composite объединён с test_result.hpp)
#include "validators/numeric.hpp"
#include "validators/signal.hpp"

// GPU Transfer
#include "gpu_transfer.hpp"

// Runner + Base + Reporters
#include "reporters.hpp"
#include "test_runner.hpp"
#include "gpu_test_base.hpp"
