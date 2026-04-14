#pragma once

/**
 * @file zero_copy_bridge.hpp
 * @brief ZeroCopy мост между OpenCL и ROCm/HIP
 *
 * Импорт OpenCL cl_mem буфера в HIP address space.
 * Поддерживаемые методы ImportFromOpenCl (в порядке приоритета):
 *
 * A. HSA Probe    — GPU VA из cl_mem через hsa_amd_pointer_info (TRUE zero-copy, 0 копий)
 * B. DMA-BUF      — cl_khr_external_memory_dma_buf → hipImportExternalMemory
 * C. GPU Copy     — OpenCL kernel: cl_mem → coarse-grain SVM (VRAM→VRAM, ~8мс для 4ГБ)
 * D. SVM fallback — fine-grain SVM + копия через CPU (медленно, секунды для 4ГБ)
 *
 * Отдельные методы (ручной вызов):
 * - ImportFromSVM  — clSVMAlloc pointer напрямую в HIP (unified VA)
 * - ImportFromGpuVA — прямой GPU VA в HIP (unified VA)
 *
 * ВАЖНО: Linux only! На Windows — стаб, бросающий исключение.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-24
 */

#if ENABLE_ROCM

#include "../opencl/opencl_export.hpp"

#include <hip/hip_runtime.h>
#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: ZeroCopyBridge — мост OpenCL ↔ ROCm
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ZeroCopyBridge
 * @brief Импорт OpenCL cl_mem в HIP address space
 *
 * Использование:
 * @code
 * ZeroCopyBridge bridge;
 * bridge.ImportFromOpenCl(cl_buffer, size, cl_device);  // автовыбор метода
 * void* hip_ptr = bridge.GetHipPtr();
 * my_kernel<<<grid, block>>>((float*)hip_ptr, N);
 * @endcode
 */
class ZeroCopyBridge {
public:
  ZeroCopyBridge();
  ~ZeroCopyBridge();

  // Запрет копирования
  ZeroCopyBridge(const ZeroCopyBridge&) = delete;
  ZeroCopyBridge& operator=(const ZeroCopyBridge&) = delete;

  // Перемещение
  ZeroCopyBridge(ZeroCopyBridge&& other) noexcept;
  ZeroCopyBridge& operator=(ZeroCopyBridge&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════════════
  // Методы импорта
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Импорт через HSA Probe — TRUE zero-copy из cl_mem
   *
   * Извлекает GPU VA из cl_mem через hsa_amd_pointer_info probe.
   * Тот же адрес в VRAM доступен в HIP напрямую.
   * 0 копий, 0 доп. памяти, ~микросекунды.
   *
   * @param cl_buffer OpenCL cl_mem буфер
   * @param buffer_size Размер буфера в байтах
   * @throws std::runtime_error если probe не нашёл GPU VA
   */
  void ImportFromHsaProbe(cl_mem cl_buffer, size_t buffer_size);

  /**
   * @brief Импорт через dma-buf file descriptor
   *
   * hipImportExternalMemory для маппинга dma-buf fd в HIP address space.
   *
   * @param dma_buf_fd File descriptor
   * @param buffer_size Размер буфера в байтах
   * @return hipSuccess при успехе
   */
  hipError_t ImportFromDmaBuf(int dma_buf_fd, size_t buffer_size);

  /**
   * @brief Импорт через GPU virtual address (прямой указатель)
   *
   * Zero overhead — тот же адрес в unified VA space.
   *
   * @param gpu_va GPU virtual address
   * @param buffer_size Размер буфера в байтах
   */
  void ImportFromGpuVA(void* gpu_va, size_t buffer_size);

  /**
   * @brief Импорт через SVM pointer (от clSVMAlloc)
   *
   * SVM pointer доступен в HIP через unified VA space (HSA).
   *
   * @param svm_ptr SVM pointer
   * @param buffer_size Размер буфера в байтах
   */
  void ImportFromSVM(void* svm_ptr, size_t buffer_size);

  /**
   * @brief Универсальный импорт — автоопределение лучшего метода
   *
   * Порядок стратегий (AUTO):
   * A. HSA Probe (TRUE zero-copy, 0 копий, 0 памяти)
   * B. DMA-BUF через OpenCL extension
   * C. GPU Copy Kernel (cl_mem → coarse-grain SVM, VRAM→VRAM, ~8мс для 4ГБ)
   * D. SVM fallback (fine-grain SVM + копия через CPU, медленно)
   *
   * @param cl_buffer OpenCL буфер для импорта
   * @param buffer_size Размер буфера в байтах
   * @param cl_device OpenCL device
   * @param strategy Принудительный выбор стратегии (AUTO по умолчанию)
   * @throws std::runtime_error если ни один метод не сработал
   */
  void ImportFromOpenCl(cl_mem cl_buffer, size_t buffer_size, cl_device_id cl_device,
                        ZeroCopyStrategy strategy = ZeroCopyStrategy::AUTO);

  // ═══════════════════════════════════════════════════════════════════════
  // Доступ к данным
  // ═══════════════════════════════════════════════════════════════════════

  void* GetHipPtr() const { return hip_ptr_; }
  size_t GetSize() const { return size_; }
  bool IsActive() const { return hip_ptr_ != nullptr; }
  ZeroCopyMethod GetMethod() const { return method_; }
  void Release();

private:
  hipExternalMemory_t ext_mem_;   ///< HIP external memory handle (для DMA-BUF)
  void* hip_ptr_;                  ///< HIP device pointer
  size_t size_;                    ///< Размер буфера
  ZeroCopyMethod method_;          ///< Используемый метод
  bool owns_memory_;               ///< true если ext_mem_ или SVM нужно освободить
  cl_context svm_cl_context_;      ///< OpenCL context для clSVMFree (SVM fallback)
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM — Windows stub

#include "../opencl/opencl_export.hpp"
#include <CL/cl.h>
#include <cstddef>
#include <stdexcept>

namespace drv_gpu_lib {

class ZeroCopyBridge {
public:
  ZeroCopyBridge() = default;
  ~ZeroCopyBridge() = default;
  ZeroCopyBridge(const ZeroCopyBridge&) = delete;
  ZeroCopyBridge& operator=(const ZeroCopyBridge&) = delete;
  ZeroCopyBridge(ZeroCopyBridge&&) noexcept = default;
  ZeroCopyBridge& operator=(ZeroCopyBridge&&) noexcept = default;

  void ImportFromHsaProbe(cl_mem, size_t) {
    throw std::runtime_error("ZeroCopyBridge: not available (ENABLE_ROCM=OFF)");
  }
  void ImportFromGpuVA(void*, size_t) {
    throw std::runtime_error("ZeroCopyBridge: not available (ENABLE_ROCM=OFF)");
  }
  void ImportFromSVM(void*, size_t) {
    throw std::runtime_error("ZeroCopyBridge: not available (ENABLE_ROCM=OFF)");
  }
  void ImportFromOpenCl(cl_mem, size_t, cl_device_id,
                        ZeroCopyStrategy = ZeroCopyStrategy::AUTO) {
    throw std::runtime_error("ZeroCopyBridge: not available (ENABLE_ROCM=OFF)");
  }

  void* GetHipPtr() const { return nullptr; }
  size_t GetSize() const { return 0; }
  bool IsActive() const { return false; }
  ZeroCopyMethod GetMethod() const { return ZeroCopyMethod::NONE; }
  void Release() {}
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
