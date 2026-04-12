#pragma once

/**
 * @file test_hybrid_external_context.hpp
 * @brief Тесты HybridBackend::InitializeFromExternalContexts
 *
 * Проверяет External Context Integration для HybridBackend (OpenCL + ROCm).
 *
 * Сценарии:
 * - Test 1: Базовая инициализация (owns_resources=false для обоих sub-backends)
 * - Test 2: Sub-backends доступны через GetOpenCL() / GetROCm()
 * - Test 3: OpenCL операции через external hybrid
 * - Test 4: ROCm операции через external hybrid
 * - Test 5: Нативные хэндлы совпадают с переданными
 * - Test 6: Cleanup НЕ освобождает внешние ресурсы
 *
 * ВАЖНО: Компилируется ТОЛЬКО при ENABLE_ROCM=1.
 *        Запуск только на Linux с AMD GPU (OpenCL + ROCm).
 *
 * Целевые платформы: gfx1201 (Radeon 9070, RDNA4), gfx908 (MI100, CDNA1)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-09
 */

#if ENABLE_ROCM

#include "backends/hybrid/hybrid_backend.hpp"
#include "backends/opencl/opencl_backend.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <CL/cl.h>
#include <vector>
#include <cmath>
#include <string>

namespace test_hybrid_external_context {

using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
  con.Print(gpu_id, "Hybrid ExternalCtx",
            std::string(passed ? "[+]" : "[X]") + " " + test_name + " ... " +
            (passed ? "PASSED" : "FAILED"));
}

// ════════════════════════════════════════════════════════════════════════════
// ExternalResources — вспомогательный RAII-хранитель внешних хэндлов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct ExternalResources
 * @brief Симуляция «внешнего кода» с готовыми OpenCL + HIP ресурсами
 *
 * В реальной интеграции это был бы OpenCV, clBLAS, MIOpen и т.п.
 * Здесь создаём вручную и управляем временем жизни явно.
 */
struct ExternalResources {
  // OpenCL
  cl_platform_id cl_platform = nullptr;
  cl_device_id   cl_device   = nullptr;
  cl_context     cl_ctx      = nullptr;
  cl_command_queue cl_queue  = nullptr;

  // ROCm/HIP
  hipStream_t hip_stream = nullptr;

  bool valid = false;

  bool Init(int device_index = 0) {
    cl_int err;

    // OpenCL init
    err = clGetPlatformIDs(1, &cl_platform, nullptr);
    if (err != CL_SUCCESS) return false;

    err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
    if (err != CL_SUCCESS) return false;

    cl_ctx = clCreateContext(nullptr, 1, &cl_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {0};
    cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, props, &err);
#else
    cl_queue = clCreateCommandQueue(cl_ctx, cl_device, 0, &err);
#endif
    if (err != CL_SUCCESS) { clReleaseContext(cl_ctx); return false; }

    // HIP stream init
    hipError_t hip_err = hipStreamCreate(&hip_stream);
    if (hip_err != hipSuccess) {
      clReleaseCommandQueue(cl_queue);
      clReleaseContext(cl_ctx);
      return false;
    }

    (void)device_index;
    valid = true;
    return true;
  }

  void Release() {
    if (hip_stream) { hipStreamDestroy(hip_stream); hip_stream = nullptr; }
    if (cl_queue)   { clReleaseCommandQueue(cl_queue); cl_queue = nullptr; }
    if (cl_ctx)     { clReleaseContext(cl_ctx); cl_ctx = nullptr; }
    if (cl_device)  { clReleaseDevice(cl_device); cl_device = nullptr; }
    valid = false;
  }

  ~ExternalResources() { Release(); }
};

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Базовая инициализация — флаги owns_resources
// ════════════════════════════════════════════════════════════════════════════

inline bool test_basic_init(ConsoleOutput& con, int gpu_id) {
  try {
    ExternalResources ext;
    if (!ext.Init(gpu_id)) {
      con.Print(gpu_id, "Hybrid ExternalCtx", "[!] ExternalResources init failed — skip");
      return false;
    }

    HybridBackend hybrid;
    hybrid.InitializeFromExternalContexts(
        gpu_id,
        ext.cl_ctx, ext.cl_device, ext.cl_queue,
        ext.hip_stream);

    bool ok = hybrid.IsInitialized()
           && !hybrid.OwnsResources()
           && hybrid.GetType() == BackendType::OPENCLandROCm
           && hybrid.GetDeviceIndex() == gpu_id;

    // sub-backends тоже не владеют ресурсами
    bool sub_ok = hybrid.GetOpenCL() && !hybrid.GetOpenCL()->OwnsResources()
               && hybrid.GetROCm()   && !hybrid.GetROCm()->OwnsResources();

    print_result(con, gpu_id, "Basic Init (owns_resources=false)", ok && sub_ok);

    hybrid.Cleanup();
    // ext.Release() освобождает ресурсы сам через деструктор
    return ok && sub_ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Hybrid ExternalCtx", "[X] Basic Init — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Sub-backends доступны и инициализированы
// ════════════════════════════════════════════════════════════════════════════

inline bool test_sub_backends_accessible(ConsoleOutput& con, int gpu_id) {
  try {
    ExternalResources ext;
    if (!ext.Init(gpu_id)) return false;

    HybridBackend hybrid;
    hybrid.InitializeFromExternalContexts(
        gpu_id, ext.cl_ctx, ext.cl_device, ext.cl_queue, ext.hip_stream);

    auto* cl  = hybrid.GetOpenCL();
    auto* rocm = hybrid.GetROCm();

    bool ok = (cl   != nullptr) && cl->IsInitialized()
           && (rocm != nullptr) && rocm->IsInitialized();

    if (ok) {
      con.Print(gpu_id, "Hybrid ExternalCtx",
                "    OpenCL: " + cl->GetDeviceName() +
                " | ROCm: " + rocm->GetDeviceName());
    }

    print_result(con, gpu_id, "Sub-backends Accessible", ok);

    hybrid.Cleanup();
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Hybrid ExternalCtx",
              "[X] Sub-backends — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: OpenCL операции через external hybrid (Allocate/Memcpy/Free)
// ════════════════════════════════════════════════════════════════════════════

inline bool test_opencl_operations(ConsoleOutput& con, int gpu_id) {
  try {
    ExternalResources ext;
    if (!ext.Init(gpu_id)) return false;

    HybridBackend hybrid;
    hybrid.InitializeFromExternalContexts(
        gpu_id, ext.cl_ctx, ext.cl_device, ext.cl_queue, ext.hip_stream);

    constexpr size_t kCount = 512;
    constexpr size_t kBytes = kCount * sizeof(float);

    // Allocate через hybrid (делегирует OpenCL)
    void* cl_buf = hybrid.Allocate(kBytes);
    bool alloc_ok = (cl_buf != nullptr);

    bool data_ok = false;
    if (alloc_ok) {
      std::vector<float> src(kCount);
      for (size_t i = 0; i < kCount; ++i) src[i] = static_cast<float>(i) * 1.5f;

      hybrid.MemcpyHostToDevice(cl_buf, src.data(), kBytes);

      std::vector<float> dst(kCount, 0.0f);
      hybrid.MemcpyDeviceToHost(dst.data(), cl_buf, kBytes);
      hybrid.Synchronize();

      data_ok = true;
      for (size_t i = 0; i < kCount; ++i) {
        if (std::abs(dst[i] - src[i]) > 1e-5f) { data_ok = false; break; }
      }

      hybrid.Free(cl_buf);
    }

    bool ok = alloc_ok && data_ok;
    print_result(con, gpu_id, "OpenCL Operations via Hybrid", ok);

    hybrid.Cleanup();
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Hybrid ExternalCtx",
              "[X] OpenCL Ops — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: ROCm операции через external hybrid (через GetROCm())
// ════════════════════════════════════════════════════════════════════════════

inline bool test_rocm_operations(ConsoleOutput& con, int gpu_id) {
  try {
    ExternalResources ext;
    if (!ext.Init(gpu_id)) return false;

    HybridBackend hybrid;
    hybrid.InitializeFromExternalContexts(
        gpu_id, ext.cl_ctx, ext.cl_device, ext.cl_queue, ext.hip_stream);

    auto* rocm = hybrid.GetROCm();
    if (!rocm) { print_result(con, gpu_id, "ROCm Operations via Hybrid", false); return false; }

    constexpr size_t kCount = 512;
    constexpr size_t kBytes = kCount * sizeof(float);

    void* hip_buf = rocm->Allocate(kBytes);
    bool alloc_ok = (hip_buf != nullptr);

    bool data_ok = false;
    if (alloc_ok) {
      std::vector<float> src(kCount);
      for (size_t i = 0; i < kCount; ++i) src[i] = static_cast<float>(i) * 2.0f;

      rocm->MemcpyHostToDevice(hip_buf, src.data(), kBytes);

      std::vector<float> dst(kCount, 0.0f);
      rocm->MemcpyDeviceToHost(dst.data(), hip_buf, kBytes);
      rocm->Synchronize();

      data_ok = true;
      for (size_t i = 0; i < kCount; ++i) {
        if (std::abs(dst[i] - src[i]) > 1e-5f) { data_ok = false; break; }
      }

      rocm->Free(hip_buf);
    }

    bool ok = alloc_ok && data_ok;
    print_result(con, gpu_id, "ROCm Operations via Hybrid", ok);

    hybrid.Cleanup();
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Hybrid ExternalCtx",
              "[X] ROCm Ops — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: Нативные хэндлы совпадают с переданными
// ════════════════════════════════════════════════════════════════════════════

inline bool test_native_handles(ConsoleOutput& con, int gpu_id) {
  try {
    ExternalResources ext;
    if (!ext.Init(gpu_id)) return false;

    HybridBackend hybrid;
    hybrid.InitializeFromExternalContexts(
        gpu_id, ext.cl_ctx, ext.cl_device, ext.cl_queue, ext.hip_stream);

    // HybridBackend делегирует нативные хэндлы OpenCL sub-backend
    bool ctx_match   = (hybrid.GetNativeContext() == static_cast<void*>(ext.cl_ctx));
    bool dev_match   = (hybrid.GetNativeDevice()  == static_cast<void*>(ext.cl_device));
    bool queue_match = (hybrid.GetNativeQueue()   == static_cast<void*>(ext.cl_queue));

    // ROCm stream через GetROCm()
    bool hip_match = hybrid.GetROCm() &&
                     (hybrid.GetROCm()->GetNativeQueue() == static_cast<void*>(ext.hip_stream));

    bool ok = ctx_match && dev_match && queue_match && hip_match;
    print_result(con, gpu_id, "Native Handles Match", ok);

    hybrid.Cleanup();
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Hybrid ExternalCtx",
              "[X] Native Handles — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Cleanup НЕ освобождает внешние ресурсы
// ════════════════════════════════════════════════════════════════════════════

inline bool test_resources_survive_cleanup(ConsoleOutput& con, int gpu_id) {
  try {
    ExternalResources ext;
    if (!ext.Init(gpu_id)) return false;

    {
      HybridBackend hybrid;
      hybrid.InitializeFromExternalContexts(
          gpu_id, ext.cl_ctx, ext.cl_device, ext.cl_queue, ext.hip_stream);
      // ~HybridBackend() → Cleanup() — не должен освобождать ext ресурсы
    }

    // OpenCL контекст живой: операция с ним должна работать
    cl_int err;
    cl_mem test_buf = clCreateBuffer(ext.cl_ctx, CL_MEM_READ_WRITE,
                                      64 * sizeof(float), nullptr, &err);
    bool cl_alive = (err == CL_SUCCESS && test_buf != nullptr);
    if (test_buf) clReleaseMemObject(test_buf);

    // HIP stream живой: hipStreamQuery без ошибок GPU_INVALID
    hipError_t hip_err = hipStreamQuery(ext.hip_stream);
    bool hip_alive = (hip_err == hipSuccess || hip_err == hipErrorNotReady);

    bool ok = cl_alive && hip_alive;
    print_result(con, gpu_id, "External Resources Survive Cleanup", ok);

    return ok;
    // ext.Release() в деструкторе — сами освобождаем
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Hybrid ExternalCtx",
              "[X] Resources Survive — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN RUN
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  constexpr int kGpuId = 0;

  con.Print(kGpuId, "Hybrid ExternalCtx",
            "\n╔══════════════════════════════════════════════════════════════╗");
  con.Print(kGpuId, "Hybrid ExternalCtx",
            "║   DrvGPU: HybridBackend External Context Integration Tests   ║");
  con.Print(kGpuId, "Hybrid ExternalCtx",
            "║   Target: gfx1201 (Radeon 9070) + gfx908 (MI100)            ║");
  con.Print(kGpuId, "Hybrid ExternalCtx",
            "╚══════════════════════════════════════════════════════════════╝");

  int passed = 0, total = 0;
  auto run_test = [&](bool (*fn)(ConsoleOutput&, int)) {
    ++total; if (fn(con, kGpuId)) ++passed;
  };

  run_test(test_basic_init);
  run_test(test_sub_backends_accessible);
  run_test(test_opencl_operations);
  run_test(test_rocm_operations);
  run_test(test_native_handles);
  run_test(test_resources_survive_cleanup);

  con.Print(kGpuId, "Hybrid ExternalCtx",
            "\n━━━━━━ Hybrid External Context: " +
            std::to_string(passed) + "/" + std::to_string(total) + " passed ━━━━━━");
}

}  // namespace test_hybrid_external_context

#endif  // ENABLE_ROCM
