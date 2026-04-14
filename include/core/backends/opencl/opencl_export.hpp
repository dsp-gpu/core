#pragma once

/**
 * @file opencl_export.hpp
 * @brief ZeroCopy detection и export для OpenCL ↔ ROCm/HIP interop
 *
 * Определение лучшего метода ZeroCopy и вспомогательные функции.
 * Поддерживаемые методы (в порядке приоритета):
 *
 * 1. HSA_PROBE     — извлечение GPU VA из cl_mem через HSA probe (TRUE zero-copy)
 * 2. DMA_BUF       — OpenCL cl_khr_external_memory_dma_buf extension
 * 3. SVM           — fine-grain SVM fallback (через CPU)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-24
 */

#include <CL/cl.h>
#include <string>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Enum: ZeroCopyMethod
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Методы ZeroCopy для OpenCL → ROCm передачи данных
 */
enum class ZeroCopyMethod {
  NONE,          ///< ZeroCopy не поддерживается
  HSA_PROBE,     ///< GPU VA из cl_mem через HSA probe (0 копий, 0 памяти)
  DMA_BUF,       ///< DMA-BUF (OpenCL extension или HSA export)
  GPU_COPY,      ///< cl_mem → coarse-grain SVM через OpenCL kernel (1 VRAM→VRAM копия)
  SVM,           ///< SVM fallback (fine-grain SVM, копия через CPU)
};

/**
 * @brief Принудительный выбор стратегии ZeroCopy
 *
 * AUTO — автовыбор лучшей стратегии (по умолчанию)
 * Остальные — принудительное использование конкретной стратегии
 * (для тестирования, бенчмарков, обхода проблем)
 */
enum class ZeroCopyStrategy {
  AUTO,           ///< Автовыбор (A→B→C→D)
  FORCE_HSA_PROBE,///< Только HSA Probe
  FORCE_DMA_BUF,  ///< Только DMA-BUF
  FORCE_GPU_COPY, ///< Только GPU Copy Kernel
  FORCE_SVM,      ///< Только SVM fallback (через CPU)
};

// ════════════════════════════════════════════════════════════════════════════
// Detection functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверить поддержку HSA probe (извлечение GPU VA из cl_mem)
 *
 * HSA runtime доступен на всех AMD GPU с ROCm.
 * HSA probe работает для любого cl_mem буфера.
 *
 * @return true если HSA runtime инициализируется
 */
inline bool SupportsHsaProbe() {
#if defined(__linux__) && ENABLE_ROCM
  // Compile-time check: HSA доступна на всех ROCm-системах.
  // Runtime проверка (hsa_init) выполняется в IsHsaAvailable() из hsa_interop.hpp
  // при первом вызове ProbeGpuVA / ImportFromOpenCl.
  return true;
#else
  return false;
#endif
}

/**
 * @brief Проверить поддержку dma-buf экспорта через OpenCL extension
 *
 * @param device OpenCL device для проверки
 * @return true если cl_khr_external_memory_dma_buf поддерживается
 */
inline bool SupportsDmaBufExport(cl_device_id device) {
#ifdef __linux__
  if (!device) return false;

  size_t ext_size = 0;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
  if (err != CL_SUCCESS || ext_size == 0) return false;

  std::string extensions(ext_size, '\0');
  err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, &extensions[0], nullptr);
  if (err != CL_SUCCESS) return false;

  return extensions.find("cl_khr_external_memory_dma_buf") != std::string::npos;
#else
  (void)device;
  return false;
#endif
}

/**
 * @brief Проверить поддержку SVM fine-grain buffer
 *
 * @param device OpenCL device
 * @return true если fine-grain SVM поддерживается
 */
inline bool SupportsSVMZeroCopy(cl_device_id device) {
#ifdef __linux__
  if (!device) return false;

  cl_device_svm_capabilities svm_caps = 0;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                                sizeof(svm_caps), &svm_caps, nullptr);
  return (err == CL_SUCCESS) && (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
#else
  (void)device;
  return false;
#endif
}

/**
 * @brief Проверить поддержку SVM coarse-grain buffer (VRAM на dGPU)
 *
 * @param device OpenCL device
 * @return true если coarse-grain SVM поддерживается
 */
inline bool SupportsSVMCoarseGrain(cl_device_id device) {
#ifdef __linux__
  if (!device) return false;

  cl_device_svm_capabilities svm_caps = 0;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                                sizeof(svm_caps), &svm_caps, nullptr);
  return (err == CL_SUCCESS) && (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
#else
  (void)device;
  return false;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// [DEPRECATED] Legacy functions — оставлены для backward compatibility
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief [DEPRECATED] Всегда возвращает false на RDNA4+
 *
 * Проверяла расширения cl_amd_svm/cl_khr_svm, которые не связаны с GPU VA.
 * На OpenCL 3.0 (RDNA4) эти расширения убраны (SVM — core feature).
 * Используйте SupportsHsaProbe() вместо этой функции.
 */
inline bool SupportsAmdGpuVA(cl_device_id device) {
  (void)device;
  return false;  // Deprecated: всегда false
}

// ════════════════════════════════════════════════════════════════════════════
// OpenCL extension constants (для legacy DMA-BUF path)
// ════════════════════════════════════════════════════════════════════════════

#ifndef CL_MEM_LINUX_DMA_BUF_FD_KHR
#define CL_MEM_LINUX_DMA_BUF_FD_KHR 0x10011
#endif

/**
 * @brief Экспортировать cl_mem → dma-buf fd через OpenCL extension
 *
 * Работает только если устройство поддерживает cl_khr_external_memory_dma_buf.
 * На RDNA4 (gfx1201) это расширение отсутствует — используйте HSA dma-buf export.
 *
 * @param buffer OpenCL cl_mem буфер
 * @return dma-buf fd >= 0, или -1 при ошибке
 */
inline int ExportClBufferToFd(cl_mem buffer) {
#ifdef __linux__
  if (!buffer) return -1;

  int dma_buf_fd = -1;
  cl_int err = clGetMemObjectInfo(
      buffer, CL_MEM_LINUX_DMA_BUF_FD_KHR,
      sizeof(int), &dma_buf_fd, nullptr);

  return (err == CL_SUCCESS) ? dma_buf_fd : -1;
#else
  (void)buffer;
  return -1;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// Detection: лучший метод ZeroCopy
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Определить лучший метод ZeroCopy для данного устройства
 *
 * Приоритет:
 * 1. HSA_PROBE — TRUE zero-copy через HSA runtime (ROCm only)
 * 2. DMA_BUF   — OpenCL extension (если поддерживается)
 * 3. SVM       — fine-grain SVM (fallback)
 * 4. NONE
 *
 * @param device OpenCL device
 * @return Лучший доступный метод
 */
inline ZeroCopyMethod DetectBestZeroCopyMethod(cl_device_id device) {
#ifdef __linux__
  // 1. HSA Probe — лучший вариант на ROCm (true zero-copy, 0 копий)
  if (SupportsHsaProbe()) return ZeroCopyMethod::HSA_PROBE;

  // 2. DMA-BUF через OpenCL extension
  if (SupportsDmaBufExport(device)) return ZeroCopyMethod::DMA_BUF;

  // 3. GPU Copy Kernel (coarse-grain SVM + OpenCL kernel, VRAM→VRAM)
  if (SupportsSVMCoarseGrain(device)) return ZeroCopyMethod::GPU_COPY;

  // 4. SVM fine-grain (fallback, через CPU)
  if (SupportsSVMZeroCopy(device)) return ZeroCopyMethod::SVM;

  return ZeroCopyMethod::NONE;
#else
  (void)device;
  return ZeroCopyMethod::NONE;
#endif
}

/**
 * @brief Строковое описание метода ZeroCopy
 */
inline const char* ZeroCopyMethodToString(ZeroCopyMethod method) {
  switch (method) {
    case ZeroCopyMethod::HSA_PROBE:  return "HSA Probe (GPU VA from cl_mem, true zero-copy)";
    case ZeroCopyMethod::DMA_BUF:    return "DMA-BUF (OpenCL extension or HSA export)";
    case ZeroCopyMethod::GPU_COPY:   return "GPU Copy Kernel (cl_mem -> coarse-grain SVM, VRAM-to-VRAM)";
    case ZeroCopyMethod::SVM:        return "SVM (fine-grain, CPU fallback)";
    case ZeroCopyMethod::NONE:       return "None (ZeroCopy not supported)";
    default:                         return "Unknown";
  }
}

}  // namespace drv_gpu_lib
