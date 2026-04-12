#pragma once

/**
 * @file test_rocm_external_context.hpp
 * @brief Тесты ROCmBackend::InitializeFromExternalStream
 *
 * Проверяет корректность External Context Integration для ROCm (HIP).
 * Аналог example_external_context_usage.hpp — но для ROCm backend.
 *
 * Сценарии:
 * - Test 1: Базовая инициализация с внешним stream (owns_resources=false)
 * - Test 2: GPU операции через external backend (Allocate/Memcpy/Free)
 * - Test 3: Cleanup НЕ уничтожает внешний stream
 * - Test 4: Нативные хэндлы соответствуют переданным
 * - Test 5: DeviceInfo доступна через external backend
 *
 * ВАЖНО: Компилируется ТОЛЬКО при ENABLE_ROCM=1.
 *        Запуск только на Linux с AMD GPU и ROCm SDK.
 *
 * Целевые платформы: gfx1201 (Radeon 9070, RDNA4), gfx908 (MI100, CDNA1)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-09
 */

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <cassert>
#include <string>

namespace test_rocm_external_context {

using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
  std::string icon = passed ? "[+]" : "[X]";
  con.Print(gpu_id, "ROCm ExternalCtx", icon + " " + test_name + " ... " +
            (passed ? "PASSED" : "FAILED"));
}

inline bool check_hip(hipError_t err, const std::string& op, ConsoleOutput& con, int gpu_id) {
  if (err != hipSuccess) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[!] HIP error in " + op + ": " + std::string(hipGetErrorString(err)));
    return false;
  }
  return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Базовая инициализация с внешним stream
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверяет InitializeFromExternalStream — флаги и состояние backend
 *
 * Создаём внешний hipStream_t «снаружи» (симуляция внешней библиотеки),
 * передаём в backend. Ожидаем:
 * - IsInitialized() == true
 * - OwnsResources() == false
 * - GetNativeQueue() == external_stream (тот же указатель)
 */
inline bool test_basic_init(ConsoleOutput& con, int gpu_id) {
  try {
    // Создаём "внешний" stream (симуляция другой библиотеки)
    hipStream_t ext_stream = nullptr;
    if (!check_hip(hipStreamCreate(&ext_stream), "hipStreamCreate", con, gpu_id)) {
      return false;
    }

    ROCmBackend backend;
    backend.InitializeFromExternalStream(gpu_id, ext_stream);

    bool ok = backend.IsInitialized()
           && !backend.OwnsResources()           // НЕ владеем stream
           && backend.GetType() == BackendType::ROCm
           && backend.GetDeviceIndex() == gpu_id
           && (backend.GetNativeQueue() == static_cast<void*>(ext_stream));

    print_result(con, gpu_id, "Basic Init (external stream)", ok);

    // backend уничтожается — НЕ должен вызывать hipStreamDestroy
    // ext_stream освобождаем сами
    backend.Cleanup();
    hipStreamDestroy(ext_stream);  // Вызывающий код освобождает stream

    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[X] Basic Init — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: GPU операции через external backend
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Allocate/Memcpy/Free работают корректно с external backend
 *
 * Проверяем полный цикл: выделить буфер → записать данные → прочитать → сравнить.
 * Все операции выполняются на external stream.
 */
inline bool test_gpu_operations(ConsoleOutput& con, int gpu_id) {
  try {
    hipStream_t ext_stream = nullptr;
    if (!check_hip(hipStreamCreate(&ext_stream), "hipStreamCreate", con, gpu_id)) {
      return false;
    }

    ROCmBackend backend;
    backend.InitializeFromExternalStream(gpu_id, ext_stream);

    constexpr size_t kCount = 1024;
    constexpr size_t kBytes = kCount * sizeof(float);

    // Выделяем GPU буфер через external backend
    void* gpu_buf = backend.Allocate(kBytes);
    bool alloc_ok = (gpu_buf != nullptr);

    bool data_ok = false;
    if (alloc_ok) {
      // Записываем тестовые данные host → device
      std::vector<float> host_src(kCount);
      for (size_t i = 0; i < kCount; ++i) {
        host_src[i] = static_cast<float>(i) * 0.5f;
      }
      backend.MemcpyHostToDevice(gpu_buf, host_src.data(), kBytes);

      // Читаем обратно device → host
      std::vector<float> host_dst(kCount, 0.0f);
      backend.MemcpyDeviceToHost(host_dst.data(), gpu_buf, kBytes);
      backend.Synchronize();

      // Проверяем данные
      data_ok = true;
      for (size_t i = 0; i < kCount; ++i) {
        if (std::abs(host_dst[i] - host_src[i]) > 1e-6f) {
          data_ok = false;
          break;
        }
      }

      backend.Free(gpu_buf);
    }

    bool ok = alloc_ok && data_ok;
    print_result(con, gpu_id, "GPU Operations (Alloc/Memcpy/Free)", ok);

    backend.Cleanup();
    hipStreamDestroy(ext_stream);

    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[X] GPU Operations — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: Cleanup НЕ уничтожает внешний stream
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверяет что Cleanup() не трогает чужой stream
 *
 * После backend.Cleanup() внешний stream должен оставаться валидным.
 * Проверяем через hipStreamQuery — если stream живой, hipSuccess или hipErrorNotReady.
 */
inline bool test_stream_survives_cleanup(ConsoleOutput& con, int gpu_id) {
  try {
    hipStream_t ext_stream = nullptr;
    if (!check_hip(hipStreamCreate(&ext_stream), "hipStreamCreate", con, gpu_id)) {
      return false;
    }

    {
      // Создаём backend в блоке — деструктор вызовет Cleanup()
      ROCmBackend backend;
      backend.InitializeFromExternalStream(gpu_id, ext_stream);

      // Backend уничтожается в конце блока
    }  // ~ROCmBackend() → Cleanup() → НЕ должен вызвать hipStreamDestroy

    // Проверяем что stream ещё жив
    hipError_t query_err = hipStreamQuery(ext_stream);
    // hipSuccess = все операции завершены, hipErrorNotReady = ещё выполняются —
    // оба означают что stream валиден. Любая другая ошибка = stream уничтожен.
    bool stream_alive = (query_err == hipSuccess || query_err == hipErrorNotReady);

    print_result(con, gpu_id, "Stream Survives Backend Cleanup", stream_alive);

    // Теперь сами уничтожаем stream
    hipStreamDestroy(ext_stream);

    return stream_alive;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[X] Stream Survives Cleanup — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Нативные хэндлы соответствуют переданным
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief GetNativeQueue() возвращает именно тот stream что мы передали
 *
 * Критично для интеграции: модуль-пользователь должен иметь возможность
 * получить external stream обратно через стандартный IBackend API.
 */
inline bool test_native_handles(ConsoleOutput& con, int gpu_id) {
  try {
    hipStream_t ext_stream = nullptr;
    if (!check_hip(hipStreamCreate(&ext_stream), "hipStreamCreate", con, gpu_id)) {
      return false;
    }

    ROCmBackend backend;
    backend.InitializeFromExternalStream(gpu_id, ext_stream);

    void* native_queue = backend.GetNativeQueue();
    void* native_device = backend.GetNativeDevice();

    // Queue должен быть именно нашим stream
    bool queue_match = (native_queue == static_cast<void*>(ext_stream));
    // Context — всегда nullptr для ROCm (HIP не имеет явного контекста)
    bool context_null = (backend.GetNativeContext() == nullptr);
    // Device — GetNativeDevice() кодирует hipDevice_t (int) через reinterpret_cast.
    // Для device 0 результат == nullptr, что корректно (0 — валидный индекс).
    // Проверяем через GetDeviceIndex() вместо указателя.
    bool device_valid = (backend.GetDeviceIndex() == gpu_id);
    (void)native_device;  // не используем для проверки — см. выше

    bool ok = queue_match && context_null && device_valid;
    print_result(con, gpu_id, "Native Handles Match", ok);

    backend.Cleanup();
    hipStreamDestroy(ext_stream);

    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[X] Native Handles — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: DeviceInfo доступна через external backend
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief GetDeviceInfo() возвращает корректные данные при external init
 *
 * ROCmCore запрашивает hipGetDeviceProperties при InitializeFromExternalStream,
 * поэтому вся информация об устройстве должна быть доступна.
 */
inline bool test_device_info(ConsoleOutput& con, int gpu_id) {
  try {
    hipStream_t ext_stream = nullptr;
    if (!check_hip(hipStreamCreate(&ext_stream), "hipStreamCreate", con, gpu_id)) {
      return false;
    }

    ROCmBackend backend;
    backend.InitializeFromExternalStream(gpu_id, ext_stream);

    auto info = backend.GetDeviceInfo();

    // Должны получить непустые данные
    bool name_ok    = !info.name.empty() && info.name != "Unknown";
    bool vendor_ok  = !info.vendor.empty();
    bool memory_ok  = (info.global_memory_size > 0);
    bool units_ok   = (info.max_compute_units > 0);
    bool wgs_ok     = (info.max_work_group_size > 0);

    bool ok = name_ok && vendor_ok && memory_ok && units_ok && wgs_ok;

    if (ok) {
      con.Print(gpu_id, "ROCm ExternalCtx",
                "    Device: " + info.name +
                " | Arch: " + info.driver_version +
                " | VRAM: " + std::to_string(info.global_memory_size / (1024*1024)) + " MB" +
                " | CU: " + std::to_string(info.max_compute_units));
    }

    print_result(con, gpu_id, "DeviceInfo via External Backend", ok);

    backend.Cleanup();
    hipStreamDestroy(ext_stream);

    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[X] DeviceInfo — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: ROCmCore::OwnsStream() корректно отражает режим
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверяет флаг owns_stream_ в ROCmCore для обоих режимов
 *
 * Normal Initialize() → core.OwnsStream() == true
 * InitializeFromExternalStream() → core.OwnsStream() == false
 */
inline bool test_owns_stream_flag(ConsoleOutput& con, int gpu_id) {
  try {
    bool normal_owns = false;
    {
      ROCmBackend backend_normal;
      backend_normal.Initialize(gpu_id);
      normal_owns = backend_normal.GetCore().OwnsStream();
    }

    hipStream_t ext_stream = nullptr;
    if (!check_hip(hipStreamCreate(&ext_stream), "hipStreamCreate", con, gpu_id)) {
      return false;
    }

    bool external_not_owns = false;
    {
      ROCmBackend backend_ext;
      backend_ext.InitializeFromExternalStream(gpu_id, ext_stream);
      external_not_owns = !backend_ext.GetCore().OwnsStream();
    }

    hipStreamDestroy(ext_stream);

    bool ok = normal_owns && external_not_owns;
    print_result(con, gpu_id, "OwnsStream Flag (normal=true, external=false)", ok);

    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm ExternalCtx",
              "[X] OwnsStream Flag — exception: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN RUN
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  constexpr int kGpuId = 0;

  con.Print(kGpuId, "ROCm ExternalCtx",
            "\n╔══════════════════════════════════════════════════════════════╗");
  con.Print(kGpuId, "ROCm ExternalCtx",
            "║   DrvGPU: ROCm External Context Integration Tests            ║");
  con.Print(kGpuId, "ROCm ExternalCtx",
            "║   Target: gfx1201 (Radeon 9070) + gfx908 (MI100)            ║");
  con.Print(kGpuId, "ROCm ExternalCtx",
            "╚══════════════════════════════════════════════════════════════╝");

  int passed = 0;
  int total  = 0;

  auto run_test = [&](bool (*test_fn)(ConsoleOutput&, int)) {
    ++total;
    if (test_fn(con, kGpuId)) ++passed;
  };

  run_test(test_basic_init);
  run_test(test_gpu_operations);
  run_test(test_stream_survives_cleanup);
  run_test(test_native_handles);
  run_test(test_device_info);
  run_test(test_owns_stream_flag);

  con.Print(kGpuId, "ROCm ExternalCtx",
            "\n━━━━━━ ROCm External Context: " +
            std::to_string(passed) + "/" + std::to_string(total) + " passed ━━━━━━");
}

}  // namespace test_rocm_external_context

#endif  // ENABLE_ROCM
