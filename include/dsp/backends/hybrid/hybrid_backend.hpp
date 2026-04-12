#pragma once

/**
 * @file hybrid_backend.hpp
 * @brief HybridBackend — гибридный OpenCL + ROCm бэкенд для одного GPU
 *
 * HybridBackend содержит оба sub-backend (OpenCL и ROCm) и позволяет:
 * - Выполнять OpenCL операции (существующие kernels)
 * - Выполнять HIP/ROCm операции (hipFFT, rocPRIM, hiprtc kernels)
 * - Обмениваться данными через ZeroCopyBridge (без копирования CPU)
 *
 * Архитектура: Вариант A (обёртка) — HybridBackend : IBackend
 * хранит OpenCLBackend + ROCmBackend, делегирует по контексту.
 *
 * Выбор primary backend:
 * - По умолчанию: OpenCL (основной для legacy пайплайна)
 * - ROCm: для модулей, использующих hipFFT / rocPRIM / hiprtc
 * - Модули сами решают, какой backend использовать через GetOpenCL() / GetROCm()
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../../interface/i_backend.hpp"
#include "../../common/backend_type.hpp"
#include "../../common/gpu_device_info.hpp"
#include "../../memory/memory_manager.hpp"
#include "../../logger/logger.hpp"

#include "../opencl/opencl_backend.hpp"
#include "../rocm/rocm_backend.hpp"
#include "../rocm/zero_copy_bridge.hpp"

#include <memory>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: HybridBackend — OpenCL + ROCm на одном GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class HybridBackend
 * @brief Гибридный бэкенд, объединяющий OpenCL и ROCm для одного GPU
 *
 * По умолчанию операции памяти (Allocate/Free/Memcpy) делегируются OpenCL.
 * Доступ к обоим sub-backend через GetOpenCL() / GetROCm().
 *
 * @code
 * DrvGPU gpu(BackendType::OPENCLandROCm, 0);
 * gpu.Initialize();
 *
 * auto& hybrid = static_cast<HybridBackend&>(gpu.GetBackend());
 *
 * // OpenCL операции
 * auto* cl_backend = hybrid.GetOpenCL();
 * void* cl_buf = cl_backend->Allocate(1024);
 *
 * // ROCm операции
 * auto* rocm_backend = hybrid.GetROCm();
 * void* hip_buf = rocm_backend->Allocate(1024);
 *
 * // ZeroCopy: OpenCL → ROCm
 * auto bridge = hybrid.CreateZeroCopyBridge(
 *     static_cast<cl_mem>(cl_buf), 1024);
 * void* hip_ptr = bridge->GetHipPtr();  // тот же буфер!
 * @endcode
 */
class HybridBackend : public IBackend {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструктор и деструктор
  // ═══════════════════════════════════════════════════════════════

  HybridBackend();
  ~HybridBackend() override;

  // Запрет копирования
  HybridBackend(const HybridBackend&) = delete;
  HybridBackend& operator=(const HybridBackend&) = delete;

  // Перемещение
  HybridBackend(HybridBackend&& other) noexcept;
  HybridBackend& operator=(HybridBackend&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Инициализация
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Инициализировать оба sub-backend для одного GPU
   *
   * Порядок:
   * 1. OpenCLBackend::Initialize(device_index)
   * 2. ROCmBackend::Initialize(device_index)
   * 3. Проверка ZeroCopy capabilities
   *
   * @param device_index Индекс GPU устройства
   */
  void Initialize(int device_index) override;

  /**
   * @brief Инициализация с внешними ресурсами OpenCL + ROCm
   *
   * Позволяет интегрировать HybridBackend в среду, где OpenCL context
   * и HIP stream уже созданы сторонним кодом (OpenCV, clBLAS, hipFFT и т.п.).
   *
   * Внутри делегирует:
   * - OpenCLBackend::InitializeFromExternalContext(ctx, device, queue)
   * - ROCmBackend::InitializeFromExternalStream(device_index, hip_stream)
   *
   * owns_resources_ = false → Cleanup() НЕ освобождает OpenCL context и HIP stream.
   *
   * @param device_index   Индекс GPU (одинаков для обоих sub-backends)
   * @param opencl_context Внешний cl_context
   * @param opencl_device  Внешний cl_device_id
   * @param opencl_queue   Внешняя cl_command_queue
   * @param hip_stream     Внешний hipStream_t
   *
   * @throws std::runtime_error если уже инициализирован
   * @throws std::runtime_error если любой из переданных хэндлов null
   *
   * @code
   * // Интеграция в среду с готовыми OpenCL + HIP ресурсами:
   * cl_context   cl_ctx = ...; cl_device_id cl_dev = ...; cl_command_queue cl_q = ...;
   * hipStream_t  hip_s  = ...;
   *
   * HybridBackend hybrid;
   * hybrid.InitializeFromExternalContexts(0, cl_ctx, cl_dev, cl_q, hip_s);
   * // ZeroCopy работает через общий GPU адресный пространство
   * // Ресурсы НЕ освобождаются при Cleanup()
   * @endcode
   */
  void InitializeFromExternalContexts(
      int device_index,
      cl_context opencl_context,
      cl_device_id opencl_device,
      cl_command_queue opencl_queue,
      hipStream_t hip_stream);

  bool IsInitialized() const override { return initialized_; }
  void Cleanup() override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Владение ресурсами
  // ═══════════════════════════════════════════════════════════════

  void SetOwnsResources(bool owns) override { owns_resources_ = owns; }
  bool OwnsResources() const override { return owns_resources_; }

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Информация
  // ═══════════════════════════════════════════════════════════════

  BackendType GetType() const override { return BackendType::OPENCLandROCm; }
  GPUDeviceInfo GetDeviceInfo() const override;
  int GetDeviceIndex() const override { return device_index_; }
  std::string GetDeviceName() const override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Нативные хэндлы (делегируем OpenCL)
  // ═══════════════════════════════════════════════════════════════

  // Возвращают OpenCL-хэндлы, потому что IBackend интерфейс ориентирован на OpenCL.
  // Для HIP-хэндлов используй GetROCm()->GetNativeContext() etc. напрямую.
  void* GetNativeContext() const override;   // → OpenCL cl_context
  void* GetNativeDevice() const override;    // → OpenCL cl_device_id
  void* GetNativeQueue() const override;     // → OpenCL cl_command_queue

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Память (делегируем OpenCL по умолчанию)
  // ═══════════════════════════════════════════════════════════════

  // Allocate/Free делегируют OpenCL (cl_mem). Для HIP-памяти (hipMalloc)
  // используй GetROCm()->Allocate() напрямую — смешивать нельзя!
  // Free определяет тип буфера по тому, через какой backend он был выделен.
  void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
  void Free(void* ptr) override;

  void MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Синхронизация (оба backend)
  // ═══════════════════════════════════════════════════════════════

  // Синхронизирует ОБА backend — нужно когда не знаешь, в каком из них
  // висит незавершённая работа. Для точечной синхронизации перед ZeroCopy
  // используй SyncBeforeZeroCopy() / SyncAfterZeroCopy() — они дешевле.
  void Synchronize() override;

  // Отправляет накопленные команды в очередь GPU без ожидания завершения.
  // Аналог clFlush + hipStreamQuery: GPU начинает исполнять, CPU не блокируется.
  // Используй перед долгой работой на CPU, чтобы GPU и CPU шли параллельно.
  void Flush() override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Capabilities (от OpenCL backend)
  // Стандартные параметры IBackend берём из OpenCL — единый интерфейс для всех бэкендов.
  // ROCm-специфика (warp size 64, L1/L2 cache size, bank width) — только через GetROCm().
  // ═══════════════════════════════════════════════════════════════

  bool SupportsSVM() const override;
  bool SupportsDoublePrecision() const override;
  size_t GetMaxWorkGroupSize() const override;
  size_t GetGlobalMemorySize() const override;
  size_t GetFreeMemorySize() const override;
  size_t GetLocalMemorySize() const override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: MemoryManager (от OpenCL backend)
  // MemoryManager управляет пулом cl_mem объектов и принадлежит OpenCL-бэкенду.
  // HIP-буферы (hipMalloc) MemoryManager не отслеживает — ими ROCmBackend управляет сам.
  // ═══════════════════════════════════════════════════════════════

  MemoryManager* GetMemoryManager() override;
  const MemoryManager* GetMemoryManager() const override;

  // ═══════════════════════════════════════════════════════════════
  // HybridBackend-specific: доступ к sub-backends
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Получить OpenCL sub-backend
   * @return Указатель на OpenCLBackend (nullptr если не инициализирован)
   */
  OpenCLBackend* GetOpenCL() { return opencl_.get(); }
  const OpenCLBackend* GetOpenCL() const { return opencl_.get(); }

  /**
   * @brief Получить ROCm sub-backend
   * @return Указатель на ROCmBackend (nullptr если не инициализирован)
   */
  ROCmBackend* GetROCm() { return rocm_.get(); }
  const ROCmBackend* GetROCm() const { return rocm_.get(); }

  // ═══════════════════════════════════════════════════════════════
  // HybridBackend-specific: ZeroCopy
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Создать ZeroCopy мост для cl_mem буфера
   *
   * Автоматически определяет лучший метод (AMD GPU VA → DMA-BUF → SVM).
   *
   * @param cl_buffer OpenCL буфер для импорта в HIP
   * @param buffer_size Размер буфера в байтах
   * @return unique_ptr на ZeroCopyBridge (владеет ресурсами)
   * @throws std::runtime_error если ZeroCopy не поддерживается
   */
  std::unique_ptr<ZeroCopyBridge> CreateZeroCopyBridge(
      cl_mem cl_buffer, size_t buffer_size);

  /**
   * @brief Определить лучший метод ZeroCopy для данного GPU
   * @return ZeroCopyMethod (HSA_PROBE, DMA_BUF, SVM или NONE)
   */
  ZeroCopyMethod GetBestZeroCopyMethod() const;

  /**
   * @brief Синхронизировать перед ZeroCopy передачей
   *
   * Вызывает clFinish на OpenCL queue, чтобы гарантировать,
   * что все данные записаны в VRAM перед доступом из HIP.
   */
  void SyncBeforeZeroCopy();

  /**
   * @brief Синхронизировать после ZeroCopy передачи
   *
   * Вызывает hipStreamSynchronize, чтобы гарантировать,
   * что HIP завершил работу с данными перед доступом из OpenCL.
   */
  void SyncAfterZeroCopy();

private:
  int device_index_;    // Индекс GPU — одинаков для OpenCL и ROCm (оба на одном устройстве).
  bool initialized_;    // true после успешного Initialize() обоих sub-backends.
  // owns_resources_: если false — sub-backends созданы снаружи и нельзя их уничтожать.
  // Обычно true (HybridBackend создаёт sub-backends сам в Initialize()).
  bool owns_resources_;

  std::unique_ptr<OpenCLBackend> opencl_;  // Primary: cl_mem, legacy kernels
  std::unique_ptr<ROCmBackend>   rocm_;    // Secondary: hipMalloc, hipFFT, rocPRIM, hiprtc

  // Мьютекс для Initialize/Cleanup — потокобезопасность при многопоточном создании.
  // Не защищает вызовы Allocate/Free/Memcpy — они уже потокобезопасны внутри sub-backends.
  mutable std::mutex mutex_;
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM — Windows stub

#include "../../interface/i_backend.hpp"
#include "../../common/backend_type.hpp"
#include "../../common/gpu_device_info.hpp"
#include "../../memory/memory_manager.hpp"
#include "../rocm/zero_copy_bridge.hpp"

#include <memory>
#include <stdexcept>

namespace drv_gpu_lib {

/**
 * @class HybridBackend
 * @brief Windows stub — HybridBackend не доступен без ROCm
 *
 * Компилируется, но не работает — для Windows используйте BackendType::OPENCL.
 *
 * Асимметрия no-op vs throw:
 * - Allocate / Memcpy* бросают: caller ожидает результат, молча вернуть nullptr нельзя.
 * - Free / Cleanup / Sync / Flush — silent no-op: нет ресурсов → нечего освобождать.
 *   Это безопаснее: код, вызывающий Free(nullptr) или Cleanup() дважды, не должен падать.
 */
class HybridBackend : public IBackend {
public:
  HybridBackend() = default;
  ~HybridBackend() override = default;

  HybridBackend(const HybridBackend&) = delete;
  HybridBackend& operator=(const HybridBackend&) = delete;
  HybridBackend(HybridBackend&&) noexcept = default;
  HybridBackend& operator=(HybridBackend&&) noexcept = default;

  void Initialize(int) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }
  bool IsInitialized() const override { return false; }
  void Cleanup() override {}

  void SetOwnsResources(bool) override {}
  bool OwnsResources() const override { return false; }

  BackendType GetType() const override { return BackendType::OPENCLandROCm; }
  GPUDeviceInfo GetDeviceInfo() const override { return {}; }
  int GetDeviceIndex() const override { return -1; }
  std::string GetDeviceName() const override { return "HybridBackend (stub)"; }

  void* GetNativeContext() const override { return nullptr; }
  void* GetNativeDevice() const override { return nullptr; }
  void* GetNativeQueue() const override { return nullptr; }

  void* Allocate(size_t, unsigned int) override {
    throw std::runtime_error("HybridBackend::Allocate: not available (ENABLE_ROCM=OFF)");
  }
  void Free(void*) override {}
  void MemcpyHostToDevice(void*, const void*, size_t) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }
  void MemcpyDeviceToHost(void*, const void*, size_t) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }
  void MemcpyDeviceToDevice(void*, const void*, size_t) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }

  void Synchronize() override {}
  void Flush() override {}

  bool SupportsSVM() const override { return false; }
  bool SupportsDoublePrecision() const override { return false; }
  size_t GetMaxWorkGroupSize() const override { return 0; }
  size_t GetGlobalMemorySize() const override { return 0; }
  size_t GetFreeMemorySize() const override { return 0; }
  size_t GetLocalMemorySize() const override { return 0; }
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
