#pragma once

/**
 * @file test_drv_gpu_external.hpp
 * @brief Тесты DrvGPU static factory: CreateFromExternalOpenCL / CreateFromExternalROCm /
 *        CreateFromExternalHybrid
 *
 * Проверяет корректность External Context Integration на уровне DrvGPU facade.
 * Тесты аналогичны test_rocm_external_context / test_hybrid_external_context,
 * но работают через публичный API DrvGPU (без прямого обращения к бэкендам).
 *
 * Сценарии:
 * - Test 1 (OpenCL):  CreateFromExternalOpenCL → IsInitialized, GetBackendType, операции памяти
 * - Test 2 (OpenCL):  Внешний контекст жив после ~DrvGPU
 * - Test 3 (ROCm):    CreateFromExternalROCm   → IsInitialized, GetBackendType, операции памяти
 * - Test 4 (ROCm):    Внешний stream жив после ~DrvGPU
 * - Test 5 (Hybrid):  CreateFromExternalHybrid → IsInitialized, GetBackendType, операции памяти
 * - Test 6 (Hybrid):  Внешние ресурсы живы после ~DrvGPU
 *
 * ВАЖНО: ROCm-тесты (3-6) компилируются ТОЛЬКО при ENABLE_ROCM=1.
 *        Запуск только на Linux с AMD GPU.
 *
 * Целевые платформы: gfx1201 (Radeon 9070, RDNA4), gfx908 (MI100, CDNA1)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-09
 */

#include <core/drv_gpu.hpp>
#include <core/services/console_output.hpp>

#include <CL/cl.h>
#include <vector>
#include <cmath>
#include <string>
#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace test_drv_gpu_external {

using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
  con.Print(gpu_id, "DrvGPU ExternalCtx",
            std::string(passed ? "[+]" : "[X]") + " " + test_name + " ... " +
            (passed ? "PASSED" : "FAILED"));
}

// ════════════════════════════════════════════════════════════════════════════
// OpenCL helper: создать внешние ресурсы
// ════════════════════════════════════════════════════════════════════════════

struct ExtOpenCL {
  cl_platform_id platform = nullptr;
  cl_device_id   device   = nullptr;
  cl_context     ctx      = nullptr;
  cl_command_queue queue  = nullptr;
  bool valid = false;

  bool Init() {
    cl_int err;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) return false;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) return false;
    ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {0};
    queue = clCreateCommandQueueWithProperties(ctx, device, props, &err);
#else
    queue = clCreateCommandQueue(ctx, device, 0, &err);
#endif
    if (err != CL_SUCCESS) { clReleaseContext(ctx); return false; }
    valid = true;
    return true;
  }

  void Release() {
    if (queue)  { clReleaseCommandQueue(queue); queue = nullptr; }
    if (ctx)    { clReleaseContext(ctx); ctx = nullptr; }
    if (device) { clReleaseDevice(device); device = nullptr; }
    valid = false;
  }

  ~ExtOpenCL() { Release(); }
};

// ════════════════════════════════════════════════════════════════════════════
// Test 1: CreateFromExternalOpenCL — базовая работа
// ════════════════════════════════════════════════════════════════════════════

inline bool test_opencl_basic(ConsoleOutput& con, int gpu_id) {
  try {
    ExtOpenCL ext;
    if (!ext.Init()) {
      con.Print(gpu_id, "DrvGPU ExternalCtx", "[!] OpenCL init failed — skip");
      return false;
    }

    auto gpu = DrvGPU::CreateFromExternalOpenCL(gpu_id, ext.ctx, ext.device, ext.queue);

    bool flags_ok = gpu.IsInitialized()
                 && gpu.GetBackendType() == BackendType::OPENCL
                 && gpu.GetDeviceIndex() == gpu_id;

    // Проверяем операции памяти
    constexpr size_t kBytes = 256 * sizeof(float);
    auto& mm = gpu.GetMemoryManager();
    void* buf = mm.Allocate(kBytes);
    bool alloc_ok = (buf != nullptr);

    bool data_ok = false;
    if (alloc_ok) {
      std::vector<float> src(256);
      for (int i = 0; i < 256; ++i) src[i] = static_cast<float>(i);
      gpu.GetBackend().MemcpyHostToDevice(buf, src.data(), kBytes);
      std::vector<float> dst(256, 0.f);
      gpu.GetBackend().MemcpyDeviceToHost(dst.data(), buf, kBytes);
      gpu.Synchronize();
      data_ok = true;
      for (int i = 0; i < 256; ++i)
        if (std::abs(dst[i] - src[i]) > 1e-5f) { data_ok = false; break; }
      mm.Free(buf);
    }

    bool ok = flags_ok && alloc_ok && data_ok;
    print_result(con, gpu_id, "OpenCL CreateFromExternal (flags + memory ops)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "DrvGPU ExternalCtx",
              "[X] OpenCL basic — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: OpenCL контекст жив после ~DrvGPU
// ════════════════════════════════════════════════════════════════════════════

inline bool test_opencl_ctx_survives(ConsoleOutput& con, int gpu_id) {
  try {
    ExtOpenCL ext;
    if (!ext.Init()) return false;

    {
      auto gpu = DrvGPU::CreateFromExternalOpenCL(gpu_id, ext.ctx, ext.device, ext.queue);
      // ~DrvGPU() → Cleanup() → backend_->Cleanup() — НЕ должен освободить ext ресурсы
    }

    // Проверяем что контекст жив: создать буфер
    cl_int err;
    cl_mem test_buf = clCreateBuffer(ext.ctx, CL_MEM_READ_WRITE,
                                     64 * sizeof(float), nullptr, &err);
    bool alive = (err == CL_SUCCESS && test_buf != nullptr);
    if (test_buf) clReleaseMemObject(test_buf);

    print_result(con, gpu_id, "OpenCL context survives ~DrvGPU", alive);
    return alive;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "DrvGPU ExternalCtx",
              "[X] OpenCL ctx survives — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ROCm тесты — только при ENABLE_ROCM
// ════════════════════════════════════════════════════════════════════════════

#if ENABLE_ROCM

// Test 3: CreateFromExternalROCm — базовая работа
inline bool test_rocm_basic(ConsoleOutput& con, int gpu_id) {
  try {
    hipStream_t ext_stream = nullptr;
    if (hipStreamCreate(&ext_stream) != hipSuccess) {
      con.Print(gpu_id, "DrvGPU ExternalCtx", "[!] hipStreamCreate failed — skip");
      return false;
    }

    auto gpu = DrvGPU::CreateFromExternalROCm(gpu_id, ext_stream);

    bool flags_ok = gpu.IsInitialized()
                 && gpu.GetBackendType() == BackendType::ROCm
                 && gpu.GetDeviceIndex() == gpu_id;

    constexpr size_t kBytes = 512 * sizeof(float);
    auto& mm = gpu.GetMemoryManager();
    void* buf = mm.Allocate(kBytes);
    bool alloc_ok = (buf != nullptr);

    bool data_ok = false;
    if (alloc_ok) {
      std::vector<float> src(512);
      for (int i = 0; i < 512; ++i) src[i] = static_cast<float>(i) * 3.0f;
      gpu.GetBackend().MemcpyHostToDevice(buf, src.data(), kBytes);
      std::vector<float> dst(512, 0.f);
      gpu.GetBackend().MemcpyDeviceToHost(dst.data(), buf, kBytes);
      gpu.Synchronize();
      data_ok = true;
      for (int i = 0; i < 512; ++i)
        if (std::abs(dst[i] - src[i]) > 1e-5f) { data_ok = false; break; }
      mm.Free(buf);
    }

    bool ok = flags_ok && alloc_ok && data_ok;
    print_result(con, gpu_id, "ROCm CreateFromExternal (flags + memory ops)", ok);

    hipStreamDestroy(ext_stream);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "DrvGPU ExternalCtx",
              "[X] ROCm basic — exception: " + std::string(e.what()));
    return false;
  }
}

// Test 4: Внешний stream жив после ~DrvGPU
inline bool test_rocm_stream_survives(ConsoleOutput& con, int gpu_id) {
  try {
    hipStream_t ext_stream = nullptr;
    if (hipStreamCreate(&ext_stream) != hipSuccess) return false;

    {
      auto gpu = DrvGPU::CreateFromExternalROCm(gpu_id, ext_stream);
      // ~DrvGPU() → Cleanup() → НЕ должен вызвать hipStreamDestroy
    }

    hipError_t q = hipStreamQuery(ext_stream);
    bool alive = (q == hipSuccess || q == hipErrorNotReady);

    print_result(con, gpu_id, "ROCm stream survives ~DrvGPU", alive);

    hipStreamDestroy(ext_stream);
    return alive;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "DrvGPU ExternalCtx",
              "[X] ROCm stream survives — exception: " + std::string(e.what()));
    return false;
  }
}

// Test 5: CreateFromExternalHybrid — базовая работа
inline bool test_hybrid_basic(ConsoleOutput& con, int gpu_id) {
  try {
    ExtOpenCL ext;
    if (!ext.Init()) return false;

    hipStream_t ext_stream = nullptr;
    if (hipStreamCreate(&ext_stream) != hipSuccess) {
      con.Print(gpu_id, "DrvGPU ExternalCtx", "[!] hipStreamCreate failed — skip");
      return false;
    }

    auto gpu = DrvGPU::CreateFromExternalHybrid(
        gpu_id, ext.ctx, ext.device, ext.queue, ext_stream);

    bool flags_ok = gpu.IsInitialized()
                 && gpu.GetBackendType() == BackendType::OPENCLandROCm
                 && gpu.GetDeviceIndex() == gpu_id;

    constexpr size_t kBytes = 512 * sizeof(float);
    auto& mm = gpu.GetMemoryManager();
    void* buf = mm.Allocate(kBytes);
    bool alloc_ok = (buf != nullptr);

    bool data_ok = false;
    if (alloc_ok) {
      std::vector<float> src(512);
      for (int i = 0; i < 512; ++i) src[i] = static_cast<float>(i) * 2.5f;
      gpu.GetBackend().MemcpyHostToDevice(buf, src.data(), kBytes);
      std::vector<float> dst(512, 0.f);
      gpu.GetBackend().MemcpyDeviceToHost(dst.data(), buf, kBytes);
      gpu.Synchronize();
      data_ok = true;
      for (int i = 0; i < 512; ++i)
        if (std::abs(dst[i] - src[i]) > 1e-5f) { data_ok = false; break; }
      mm.Free(buf);
    }

    bool ok = flags_ok && alloc_ok && data_ok;
    print_result(con, gpu_id, "Hybrid CreateFromExternal (flags + memory ops)", ok);

    hipStreamDestroy(ext_stream);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "DrvGPU ExternalCtx",
              "[X] Hybrid basic — exception: " + std::string(e.what()));
    return false;
  }
}

// Test 6: Внешние ресурсы (OpenCL + HIP) живы после ~DrvGPU
inline bool test_hybrid_resources_survive(ConsoleOutput& con, int gpu_id) {
  try {
    ExtOpenCL ext;
    if (!ext.Init()) return false;

    hipStream_t ext_stream = nullptr;
    if (hipStreamCreate(&ext_stream) != hipSuccess) return false;

    {
      auto gpu = DrvGPU::CreateFromExternalHybrid(
          gpu_id, ext.ctx, ext.device, ext.queue, ext_stream);
      // ~DrvGPU() НЕ должен освободить внешние ресурсы
    }

    // OpenCL жив?
    cl_int err;
    cl_mem test_buf = clCreateBuffer(ext.ctx, CL_MEM_READ_WRITE,
                                     32 * sizeof(float), nullptr, &err);
    bool cl_alive = (err == CL_SUCCESS && test_buf != nullptr);
    if (test_buf) clReleaseMemObject(test_buf);

    // HIP stream жив?
    hipError_t q = hipStreamQuery(ext_stream);
    bool hip_alive = (q == hipSuccess || q == hipErrorNotReady);

    bool ok = cl_alive && hip_alive;
    print_result(con, gpu_id, "Hybrid: external resources survive ~DrvGPU", ok);

    hipStreamDestroy(ext_stream);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "DrvGPU ExternalCtx",
              "[X] Hybrid resources survive — exception: " + std::string(e.what()));
    return false;
  }
}

#endif  // ENABLE_ROCM

// ════════════════════════════════════════════════════════════════════════════
// MAIN RUN
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  constexpr int kGpuId = 0;

  con.Print(kGpuId, "DrvGPU ExternalCtx",
            "\n╔══════════════════════════════════════════════════════════════╗");
  con.Print(kGpuId, "DrvGPU ExternalCtx",
            "║   DrvGPU: External Context Static Factory Tests              ║");
  con.Print(kGpuId, "DrvGPU ExternalCtx",
            "║   OpenCL + ROCm + Hybrid (gfx1201 / gfx908)                 ║");
  con.Print(kGpuId, "DrvGPU ExternalCtx",
            "╚══════════════════════════════════════════════════════════════╝");

  int passed = 0, total = 0;
  auto run_test = [&](bool (*fn)(ConsoleOutput&, int)) {
    ++total; if (fn(con, kGpuId)) ++passed;
  };

  run_test(test_opencl_basic);
  run_test(test_opencl_ctx_survives);

#if ENABLE_ROCM
  run_test(test_rocm_basic);
  run_test(test_rocm_stream_survives);
  run_test(test_hybrid_basic);
  run_test(test_hybrid_resources_survive);
#endif

  con.Print(kGpuId, "DrvGPU ExternalCtx",
            "\n━━━━━━ DrvGPU External Context: " +
            std::to_string(passed) + "/" + std::to_string(total) + " passed ━━━━━━");
}

}  // namespace test_drv_gpu_external
