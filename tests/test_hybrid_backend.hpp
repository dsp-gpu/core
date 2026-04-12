#pragma once

/**
 * @file test_hybrid_backend.hpp
 * @brief Тесты HybridBackend (OpenCL + ROCm)
 *
 * Тесты:
 * 1. init — инициализация обоих sub-backend
 * 2. device_info — получение информации об устройстве
 * 3. opencl_allocate — выделение/освобождение памяти через OpenCL
 * 4. rocm_allocate — выделение/освобождение памяти через ROCm
 * 5. zero_copy_bridge — ZeroCopy через HybridBackend
 * 6. sync — синхронизация обоих backend
 *
 * @note Запускать ТОЛЬКО на Linux + AMD GPU с ROCm!
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../backends/hybrid/hybrid_backend.hpp"
#include "../backends/rocm/zero_copy_bridge.hpp"
#include "../logger/logger.hpp"

#include <CL/cl.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace test_hybrid_backend {

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

static void print_test(const std::string& name, bool passed) {
  std::cout << "  [Hybrid] " << name << ": "
            << (passed ? "PASSED" : "FAILED") << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Initialize both sub-backends
// ════════════════════════════════════════════════════════════════════════════

static void test_init() {
  using namespace drv_gpu_lib;

  HybridBackend hybrid;
  hybrid.Initialize(0);

  bool passed = hybrid.IsInitialized();
  passed &= (hybrid.GetType() == BackendType::OPENCLandROCm);
  passed &= (hybrid.GetOpenCL() != nullptr);
  passed &= (hybrid.GetROCm() != nullptr);
  passed &= hybrid.GetOpenCL()->IsInitialized();
  passed &= hybrid.GetROCm()->IsInitialized();
  passed &= (hybrid.GetDeviceIndex() == 0);

  hybrid.Cleanup();
  print_test("init", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Device info
// ════════════════════════════════════════════════════════════════════════════

static void test_device_info() {
  using namespace drv_gpu_lib;

  HybridBackend hybrid;
  hybrid.Initialize(0);

  auto info = hybrid.GetDeviceInfo();
  std::string name = hybrid.GetDeviceName();

  bool passed = !info.name.empty();
  passed &= !name.empty();
  passed &= (name.find("Hybrid") != std::string::npos);
  passed &= (hybrid.GetNativeContext() != nullptr);  // OpenCL context
  passed &= (hybrid.GetNativeDevice() != nullptr);   // OpenCL device
  passed &= (hybrid.GetNativeQueue() != nullptr);    // OpenCL queue

  std::cout << "  [Hybrid]   Device: " << name << "\n";
  std::cout << "  [Hybrid]   Global mem: "
            << (hybrid.GetGlobalMemorySize() / (1024 * 1024)) << " MB\n";

  hybrid.Cleanup();
  print_test("device_info", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: OpenCL allocate/memcpy/free
// ════════════════════════════════════════════════════════════════════════════

static void test_opencl_allocate() {
  using namespace drv_gpu_lib;

  HybridBackend hybrid;
  hybrid.Initialize(0);

  const size_t N = 256;
  const size_t buf_size = N * sizeof(float);

  // Allocate через HybridBackend (→ OpenCL)
  void* gpu_buf = hybrid.Allocate(buf_size);

  // Write
  std::vector<float> input(N);
  for (size_t i = 0; i < N; ++i) input[i] = static_cast<float>(i) * 2.0f;
  hybrid.MemcpyHostToDevice(gpu_buf, input.data(), buf_size);

  // Read back
  std::vector<float> output(N, 0.0f);
  hybrid.MemcpyDeviceToHost(output.data(), gpu_buf, buf_size);

  // Compare
  float max_err = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float diff = std::fabs(input[i] - output[i]);
    if (diff > max_err) max_err = diff;
  }

  bool passed = (max_err < 1e-6f);

  hybrid.Free(gpu_buf);
  hybrid.Cleanup();
  print_test("opencl_allocate", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: ROCm allocate/memcpy/free (через GetROCm())
// ════════════════════════════════════════════════════════════════════════════

static void test_rocm_allocate() {
  using namespace drv_gpu_lib;

  HybridBackend hybrid;
  hybrid.Initialize(0);

  auto* rocm = hybrid.GetROCm();
  assert(rocm != nullptr);

  const size_t N = 256;
  const size_t buf_size = N * sizeof(float);

  // Allocate через ROCm sub-backend
  void* hip_buf = rocm->Allocate(buf_size);

  // Write
  std::vector<float> input(N);
  for (size_t i = 0; i < N; ++i) input[i] = static_cast<float>(i) * 3.0f;
  rocm->MemcpyHostToDevice(hip_buf, input.data(), buf_size);

  // Read back
  std::vector<float> output(N, 0.0f);
  rocm->MemcpyDeviceToHost(output.data(), hip_buf, buf_size);

  // Compare
  float max_err = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float diff = std::fabs(input[i] - output[i]);
    if (diff > max_err) max_err = diff;
  }

  bool passed = (max_err < 1e-6f);

  rocm->Free(hip_buf);
  hybrid.Cleanup();
  print_test("rocm_allocate", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: ZeroCopy bridge через HybridBackend
// ════════════════════════════════════════════════════════════════════════════

static void test_zero_copy_bridge() {
  using namespace drv_gpu_lib;

  HybridBackend hybrid;
  hybrid.Initialize(0);

  const size_t N = 512;
  const size_t buf_size = N * sizeof(float);

  // 1. Записать данные через OpenCL
  void* cl_buf = hybrid.Allocate(buf_size);
  std::vector<float> input(N);
  for (size_t i = 0; i < N; ++i) input[i] = static_cast<float>(i) * 0.1f;
  hybrid.MemcpyHostToDevice(cl_buf, input.data(), buf_size);

  // 2. Синхронизировать OpenCL
  hybrid.SyncBeforeZeroCopy();

  // 3. Создать ZeroCopy bridge — FORCE_GPU_COPY (HSA Probe ненадёжен на RDNA4 gfx1201)
  bool passed = false;
  try {
    cl_device_id cl_device = static_cast<cl_device_id>(hybrid.GetOpenCL()->GetNativeDevice());

    ZeroCopyBridge bridge;
    bridge.ImportFromOpenCl(static_cast<cl_mem>(cl_buf), buf_size, cl_device,
                            ZeroCopyStrategy::FORCE_GPU_COPY);

    std::cout << "  [Hybrid]   ZeroCopy method: " << ZeroCopyMethodToString(bridge.GetMethod()) << "\n";

    if (bridge.IsActive()) {
      // 4. Прочитать через HIP
      std::vector<float> output(N, 0.0f);
      hipError_t err = hipMemcpy(output.data(), bridge.GetHipPtr(),
                                  buf_size, hipMemcpyDeviceToHost);

      if (err == hipSuccess) {
        float max_error = 0.0f;
        for (size_t i = 0; i < N; ++i) {
          float diff = std::fabs(input[i] - output[i]);
          if (diff > max_error) max_error = diff;
        }
        passed = (max_error < 1e-6f);
        std::cout << "  [Hybrid]   Max error: " << max_error << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cout << "  [Hybrid]   Exception: " << e.what() << "\n";
  }

  hybrid.Free(cl_buf);
  hybrid.Cleanup();
  print_test("zero_copy_bridge", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Synchronize both backends
// ════════════════════════════════════════════════════════════════════════════

static void test_sync() {
  using namespace drv_gpu_lib;

  HybridBackend hybrid;
  hybrid.Initialize(0);

  // Выделяем буферы на обоих backend
  const size_t buf_size = 1024 * sizeof(float);
  void* cl_buf = hybrid.GetOpenCL()->Allocate(buf_size);
  void* hip_buf = hybrid.GetROCm()->Allocate(buf_size);

  // Записываем данные
  std::vector<float> data(1024, 42.0f);
  hybrid.GetOpenCL()->MemcpyHostToDevice(cl_buf, data.data(), buf_size);
  hybrid.GetROCm()->MemcpyHostToDevice(hip_buf, data.data(), buf_size);

  // Synchronize оба
  hybrid.Synchronize();

  // Flush оба
  hybrid.Flush();

  bool passed = true;  // Если не крэшится — passed

  // Cleanup
  hybrid.GetOpenCL()->Free(cl_buf);
  hybrid.GetROCm()->Free(hip_buf);
  hybrid.Cleanup();
  print_test("sync", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Run all
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  std::cout << "\n========== HybridBackend Tests ==========\n";

  test_init();
  test_device_info();
  test_opencl_allocate();
  test_rocm_allocate();
  test_zero_copy_bridge();
  test_sync();

  std::cout << "========== HybridBackend Tests Done ==========\n\n";
}

}  // namespace test_hybrid_backend

#endif  // ENABLE_ROCM
