#pragma once

/**
 * @file test_rocm_backend.hpp
 * @brief Тесты ROCm Backend — Initialize, Allocate, Memcpy, Synchronize
 *
 * ВАЖНО: Тесты компилируются ТОЛЬКО при ENABLE_ROCM=1.
 * На Windows (без ROCm) этот файл полностью пропускается.
 * Запуск тестов — только на Linux с AMD GPU и ROCm SDK.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"
#include "memory/memory_manager.hpp"
#include "memory/gpu_buffer.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <complex>
#include <numeric>
#include <cmath>
#include <cassert>
#include <string>

namespace test_rocm_backend {

using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Утилита: вывод результата теста
// ════════════════════════════════════════════════════════════════════════════

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
  std::string status = passed ? "PASSED" : "FAILED";
  std::string icon = passed ? "[+]" : "[X]";
  con.Print(gpu_id, "ROCm Test", icon + " " + test_name + " ... " + status);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: ROCm Init
// ════════════════════════════════════════════════════════════════════════════

inline bool test_init(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    bool ok = backend.IsInitialized()
           && backend.GetType() == BackendType::ROCm
           && backend.GetDeviceIndex() == gpu_id
           && backend.OwnsResources();

    print_result(con, gpu_id, "ROCm Init", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] ROCm Init EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Device Info
// ════════════════════════════════════════════════════════════════════════════

inline bool test_device_info(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    std::string name = backend.GetDeviceName();
    auto info = backend.GetDeviceInfo();
    size_t global_mem = backend.GetGlobalMemorySize();
    size_t max_wg = backend.GetMaxWorkGroupSize();

    bool ok = !name.empty()
           && name != "Unknown"
           && global_mem > 0
           && max_wg > 0
           && !info.name.empty()
           && info.vendor == "AMD";

    con.Print(gpu_id, "ROCm Test", "  Device: " + name);
    con.Print(gpu_id, "ROCm Test", "  Memory: " + std::to_string(global_mem / (1024*1024)) + " MB");
    con.Print(gpu_id, "ROCm Test", "  Max WG: " + std::to_string(max_wg));

    print_result(con, gpu_id, "Device Info", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] Device Info EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: Allocate / Free
// ════════════════════════════════════════════════════════════════════════════

inline bool test_allocate_free(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    // Выделяем 1 KB
    void* ptr1 = backend.Allocate(1024);
    bool ok1 = (ptr1 != nullptr);

    // Освобождаем
    backend.Free(ptr1);

    // Повторное выделение (проверка что Free прошёл корректно)
    void* ptr2 = backend.Allocate(2048);
    bool ok2 = (ptr2 != nullptr);
    backend.Free(ptr2);

    // Выделяем 1 MB
    void* ptr3 = backend.Allocate(1024 * 1024);
    bool ok3 = (ptr3 != nullptr);
    backend.Free(ptr3);

    bool ok = ok1 && ok2 && ok3;
    print_result(con, gpu_id, "Allocate/Free", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] Allocate/Free EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Memcpy Host <-> Device
// ════════════════════════════════════════════════════════════════════════════

inline bool test_memcpy_h2d_d2h(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    const size_t N = 1024;
    const size_t size_bytes = N * sizeof(float);

    // Подготовка данных
    std::vector<float> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 0.0f);  // 0, 1, 2, ...

    // Выделяем GPU буфер
    void* gpu_ptr = backend.Allocate(size_bytes);
    assert(gpu_ptr != nullptr);

    // Host -> Device
    backend.MemcpyHostToDevice(gpu_ptr, host_data.data(), size_bytes);

    // Device -> Host
    std::vector<float> result(N, -1.0f);
    backend.MemcpyDeviceToHost(result.data(), gpu_ptr, size_bytes);

    backend.Free(gpu_ptr);

    // Проверка данных
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
      if (std::fabs(result[i] - host_data[i]) > 1e-6f) {
        ok = false;
        con.Print(gpu_id, "ROCm Test",
                  "  Mismatch at [" + std::to_string(i) + "]: expected " +
                  std::to_string(host_data[i]) + ", got " + std::to_string(result[i]));
        break;
      }
    }

    print_result(con, gpu_id, "Memcpy H2D/D2H", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] Memcpy H2D/D2H EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: Memcpy Device <-> Device
// ════════════════════════════════════════════════════════════════════════════

inline bool test_memcpy_d2d(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    const size_t N = 512;
    const size_t size_bytes = N * sizeof(float);

    // Подготовка данных
    std::vector<float> host_data(N);
    for (size_t i = 0; i < N; ++i) host_data[i] = static_cast<float>(i * 3.14f);

    // Выделяем 2 GPU буфера
    void* gpu_src = backend.Allocate(size_bytes);
    void* gpu_dst = backend.Allocate(size_bytes);

    // Host -> src
    backend.MemcpyHostToDevice(gpu_src, host_data.data(), size_bytes);

    // src -> dst (Device to Device)
    backend.MemcpyDeviceToDevice(gpu_dst, gpu_src, size_bytes);

    // dst -> Host
    std::vector<float> result(N, -1.0f);
    backend.MemcpyDeviceToHost(result.data(), gpu_dst, size_bytes);

    backend.Free(gpu_src);
    backend.Free(gpu_dst);

    // Проверка данных
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
      if (std::fabs(result[i] - host_data[i]) > 1e-6f) {
        ok = false;
        break;
      }
    }

    print_result(con, gpu_id, "Memcpy D2D", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] Memcpy D2D EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: GPUBuffer with ROCm (через MemoryManager)
// ════════════════════════════════════════════════════════════════════════════

inline bool test_gpu_buffer(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    auto* mem_mgr = backend.GetMemoryManager();
    assert(mem_mgr != nullptr);

    const size_t N = 2048;

    // CreateBuffer -> GPUBuffer<float>
    auto buffer = mem_mgr->CreateBuffer<float>(N);

    // Записать данные
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = static_cast<float>(i) * 0.5f;
    buffer->Write(data);

    // Прочитать данные
    auto result = buffer->Read();

    // Проверка
    bool ok = (result.size() == N);
    for (size_t i = 0; i < N && ok; ++i) {
      if (std::fabs(result[i] - data[i]) > 1e-6f) {
        ok = false;
      }
    }

    print_result(con, gpu_id, "GPUBuffer with ROCm", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] GPUBuffer EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 7: Synchronize
// ════════════════════════════════════════════════════════════════════════════

inline bool test_synchronize(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    const size_t N = 4096;
    const size_t size_bytes = N * sizeof(float);

    // Выделяем и копируем данные
    std::vector<float> data(N, 42.0f);
    void* gpu_ptr = backend.Allocate(size_bytes);
    backend.MemcpyHostToDevice(gpu_ptr, data.data(), size_bytes);

    // Synchronize — должен пройти без ошибок
    backend.Synchronize();

    // Flush — non-blocking, должен пройти без ошибок
    backend.Flush();

    // Читаем обратно после sync
    std::vector<float> result(N, 0.0f);
    backend.MemcpyDeviceToHost(result.data(), gpu_ptr, size_bytes);

    backend.Free(gpu_ptr);

    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i) {
      if (std::fabs(result[i] - 42.0f) > 1e-6f) ok = false;
    }

    print_result(con, gpu_id, "Synchronize", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "ROCm Test", "[X] Synchronize EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Главная функция запуска тестов
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  con.Start();
  int gpu_id = 0;

  con.Print(gpu_id, "ROCm Test", "");
  con.Print(gpu_id, "ROCm Test", "============================================");
  con.Print(gpu_id, "ROCm Test", "  ROCm Backend Tests");
  con.Print(gpu_id, "ROCm Test", "============================================");

  // Проверяем наличие ROCm устройств
  int device_count = ROCmCore::GetAvailableDeviceCount();
  con.Print(gpu_id, "ROCm Test", "Available ROCm devices: " + std::to_string(device_count));

  if (device_count == 0) {
    con.Print(gpu_id, "ROCm Test", "[!] No ROCm devices found — skipping tests");
    return;
  }

  int passed = 0;
  int total = 7;

  if (test_init(con, gpu_id)) ++passed;
  if (test_device_info(con, gpu_id)) ++passed;
  if (test_allocate_free(con, gpu_id)) ++passed;
  if (test_memcpy_h2d_d2h(con, gpu_id)) ++passed;
  if (test_memcpy_d2d(con, gpu_id)) ++passed;
  if (test_gpu_buffer(con, gpu_id)) ++passed;
  if (test_synchronize(con, gpu_id)) ++passed;

  con.Print(gpu_id, "ROCm Test", "");
  con.Print(gpu_id, "ROCm Test", "Results: " + std::to_string(passed) + "/" +
                                  std::to_string(total) + " passed");
  con.Print(gpu_id, "ROCm Test", "============================================");
  con.Print(gpu_id, "ROCm Test", "");
}

}  // namespace test_rocm_backend

#endif  // ENABLE_ROCM
