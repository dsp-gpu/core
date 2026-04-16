#pragma once

/**
 * @file hsa_interop.hpp
 * @brief HSA interop: извлечение GPU VA из cl_mem через HSA runtime
 *
 * На AMD ROCm и OpenCL и HIP используют HSA (ROCr) runtime.
 * cl_mem внутренне содержит HSA-аллокацию с GPU VA в unified address space.
 * Этот модуль извлекает GPU VA через hsa_amd_pointer_info probe.
 *
 * GPU VA из cl_mem доступен в HIP напрямую — TRUE zero-copy:
 * - 0 копий данных
 * - 0 дополнительной памяти
 * - ~микросекунды (только адресная арифметика)
 *
 * Алгоритм:
 * 1. Первый вызов: сканирование cl_mem объекта (каждые 8 байт, 0..2048)
 * 2. Для каждого void* значения: hsa_amd_pointer_info → ищем HSA type + совпадение размера
 * 3. Кешируем offset → последующие вызовы O(1)
 *
 * Проверено на:
 * - AMD Radeon RX 9070 (RDNA4, gfx1201), ROCm 7.2.0
 * - offset = +664 (может отличаться на других версиях ROCm)
 *
 * @note Linux + ENABLE_ROCM only
 * @author Кодо (AI Assistant)
 * @date 2026-03-24
 */

#if ENABLE_ROCM

#include <CL/cl.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Константы
// ════════════════════════════════════════════════════════════════════════════

/// Максимальный размер cl_mem объекта для сканирования (байт)
static constexpr int kMaxProbeBytes = 2048;

/// Шаг сканирования (sizeof(void*) на 64-bit)
static constexpr int kProbeStep = 8;


// ════════════════════════════════════════════════════════════════════════════
// Struct: HsaProbeResult
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Результат HSA probe — GPU VA из cl_mem
 */
struct HsaProbeResult {
  void* gpu_va = nullptr;      ///< GPU virtual address (валиден в HIP)
  size_t alloc_size = 0;       ///< Размер HSA-аллокации
  int offset = -1;             ///< Offset внутри cl_mem handle
  bool valid = false;          ///< Успешно ли найден GPU VA
};

// ════════════════════════════════════════════════════════════════════════════
// HSA Lifecycle
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализация HSA runtime (lazy, thread-safe)
 *
 * Вызывает hsa_init() один раз. Повторные вызовы — no-op.
 * HSA runtime живёт до завершения процесса (hsa_shut_down не вызываем —
 * это безопасно, ROCm runtime делает cleanup при exit).
 *
 * @return true если HSA доступна
 */
inline bool EnsureHsaInitialized() {
  static std::once_flag flag;
  static bool ok = false;
  std::call_once(flag, []() {
    hsa_status_t st = hsa_init();
    ok = (st == HSA_STATUS_SUCCESS);
  });
  return ok;
}

/**
 * @brief Проверить доступность HSA runtime
 */
inline bool IsHsaAvailable() {
  return EnsureHsaInitialized();
}

// ════════════════════════════════════════════════════════════════════════════
// ProbeGpuVA — главная функция
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Извлечь GPU VA из cl_mem через HSA pointer info probe
 *
 * Сканирует внутреннюю структуру cl_mem (amd::Memory в CLR),
 * ищет поле содержащее HSA-managed GPU pointer с совпадающим размером.
 *
 * Первый вызов: полное сканирование (~100μs).
 * Последующие: чтение по кешированному offset (~наносекунды).
 *
 * @param cl_buffer cl_mem handle (от clCreateBuffer)
 * @param expected_size Размер буфера в байтах (для верификации)
 * @return HsaProbeResult с GPU VA или valid=false
 *
 * @note Thread-safe (мьютекс на первый probe)
 * @note Не модифицирует cl_mem
 *
 * ══════════════════════════════════════════════════════════════════════
 * ВНИМАНИЕ: ЗАВИСИМОСТЬ ОТ РЕАЛИЗАЦИИ ROCm CLR
 * ══════════════════════════════════════════════════════════════════════
 *
 * Этот код сканирует внутреннюю структуру cl_mem объекта (amd::Memory
 * в ROCm CLR), интерпретируя его как массив void* и проверяя каждый
 * через hsa_amd_pointer_info. Это НЕ документированный OpenCL API.
 *
 * Почему это безопасно на практике:
 * - cl_mem в ROCm CLR — C++ объект на heap (~500-2000 байт)
 * - Мы ограничиваем скан размером буфера через clGetMemObjectInfo
 *   (CL_MEM_SIZE) или kMaxProbeBytes
 * - Каждый кандидат проверяется через hsa_amd_pointer_info:
 *   мусорные значения фильтруются (alignment 4K, HSA type, size match)
 * - Offset кешируется → последующие вызовы не сканируют
 *
 * Почему нет альтернативы:
 * - OpenCL spec не предоставляет API для извлечения GPU VA из cl_mem
 * - CL_MEM_AMD_GPU_VA (cl_amd_svm) — DEPRECATED, убран в OpenCL 3.0
 * - DMA-BUF export — RDNA4 (gfx1201) НЕ поддерживает через OpenCL ext
 * - HSA probe — единственный рабочий путь на RDNA4 + ROCm 7.2
 *
 * Риск: offset может измениться при обновлении ROCm CLR.
 * Защита: кеш offset сбрасывается при несовпадении, полный ре-скан.
 * Проверено: AMD Radeon RX 9070 (RDNA4, gfx1201), ROCm 7.2.0.
 * ══════════════════════════════════════════════════════════════════════
 */
HsaProbeResult ProbeGpuVA(cl_mem cl_buffer, size_t expected_size);

/**
 * @brief Экспорт GPU VA как dma-buf fd (для hipImportExternalMemory)
 *
 * @param gpu_va GPU virtual address (из ProbeGpuVA)
 * @param size Размер буфера
 * @param[out] fd DMA-BUF file descriptor
 * @param[out] offset Offset внутри dma-buf
 * @return true если экспорт успешен
 */
bool ExportGpuVAasDmaBuf(void* gpu_va, size_t size,
                          int* fd, uint64_t* offset);

/**
 * @brief Закрыть dma-buf fd
 */
inline void CloseDmaBuf(int fd) {
  if (fd >= 0) {
    hsa_amd_portable_close_dmabuf(fd);
  }
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
