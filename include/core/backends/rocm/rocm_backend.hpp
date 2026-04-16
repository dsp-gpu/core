#pragma once

/**
 * @file rocm_backend.hpp
 * @brief Реализация IBackend для ROCm/HIP
 *
 * ROCmBackend - полная реализация бэкенда на базе HIP API.
 *
 * MULTI-GPU (v2.0):
 * Каждый экземпляр ROCmBackend владеет СВОИМ ROCmCore,
 * что позволяет работать с разными AMD GPU параллельно.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/common/backend_type.hpp>
#include <core/common/gpu_device_info.hpp>
#include <core/logger/logger.hpp>
#include <core/memory/memory_manager.hpp>

#include "rocm_core.hpp"
#include "stream_pool.hpp"

#include <hip/hip_runtime.h>
#include <memory>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: ROCmBackend - Реализация бэкенда для HIP/ROCm
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ROCmBackend
 * @brief Реализация IBackend на базе HIP API (ROCm)
 *
 * Один экземпляр = один AMD GPU. Владеет ROCmCore (поток + device handle)
 * и MemoryManager (пул hipMalloc буферов).
 *
 * Архитектура Multi-GPU:
 * - Каждый ROCmBackend создаёт собственный ROCmCore с device_index
 * - HybridBackend содержит ROCmBackend как secondary sub-backend
 * - Модули обращаются к ROCm через HybridBackend::GetROCm()
 *
 * Делегирование:
 * - Все вычисления → ROCmCore::GetStream() (один stream на GPU)
 * - Memory → hipMalloc / hipFree напрямую (без CL_MEM_FLAGS аналогов)
 * - Capabilities → hipDeviceProp_t через ROCmCore
 */
class ROCmBackend : public IBackend {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструктор и деструктор
  // ═══════════════════════════════════════════════════════════════

  ROCmBackend();
  ~ROCmBackend() override;

  // ═══════════════════════════════════════════════════════════════
  // Запрет копирования, разрешение перемещения
  // ═══════════════════════════════════════════════════════════════
  ROCmBackend(const ROCmBackend&) = delete;
  ROCmBackend& operator=(const ROCmBackend&) = delete;
  ROCmBackend(ROCmBackend&& other) noexcept;
  ROCmBackend& operator=(ROCmBackend&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Инициализация
  // ═══════════════════════════════════════════════════════════════

  // Создаёт собственный ROCmCore + MemoryManager для заданного GPU.
  // Thread-safe. При повторном вызове — автоматически вызывает Cleanup() сначала.
  // Бросает std::runtime_error если device_index выходит за пределы или HIP недоступен.
  void Initialize(int device_index) override;

  /**
   * @brief Инициализация с внешним hipStream_t (External Context Integration)
   *
   * Позволяет использовать ROCmBackend с уже существующим HIP stream
   * (например, из другой библиотеки — hipBLAS, hipFFT, MIOpen).
   *
   * Отличия от Initialize():
   * - НЕ создаёт stream (hipStreamCreate не вызывается)
   * - owns_resources_ = false → Cleanup() НЕ уничтожает stream
   * - MemoryManager создаётся собственный (hipMalloc буферы — наши)
   *
   * @param device_index     Индекс AMD GPU (0..N-1)
   * @param external_stream  Внешний поток — вызывающий код управляет его временем жизни
   *
   * @throws std::runtime_error если уже инициализирован (вызови Cleanup() сначала)
   * @throws std::runtime_error если external_stream == nullptr
   * @throws std::runtime_error если device_index невалиден
   *
   * @code
   * // Пример: интеграция с hipBLAS
   * hipStream_t my_stream;
   * hipStreamCreate(&my_stream);
   * hipblasSetStream(blas_handle, my_stream);
   *
   * ROCmBackend backend;
   * backend.InitializeFromExternalStream(0, my_stream);
   * // backend НЕ уничтожит my_stream при Cleanup()
   * // my_stream используется и в hipBLAS, и в нашем backend
   * @endcode
   */
  void InitializeFromExternalStream(int device_index, hipStream_t external_stream);

  bool IsInitialized() const override { return initialized_; }
  // Освобождает MemoryManager (hipFree буферов), затем core_ (hipStreamDestroy).
  // Порядок важен: буферы могут ссылаться на stream_ — core_ уничтожается последним.
  // Идемпотентен; вызывается автоматически из деструктора.
  void Cleanup() override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Управление владением ресурсами
  // ═══════════════════════════════════════════════════════════════

  // owns_resources_ управляет кто уничтожает core_ в Cleanup().
  // false — core_ создан снаружи (например, shared между несколькими бэкендами).
  // В нормальном пути Initialize() выставляет owns=true сам — явный вызов обычно не нужен.
  void SetOwnsResources(bool owns) override { owns_resources_ = owns; }
  bool OwnsResources() const override { return owns_resources_; }

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Информация об устройстве
  // ═══════════════════════════════════════════════════════════════

  BackendType GetType() const override { return BackendType::ROCm; }
  GPUDeviceInfo GetDeviceInfo() const override;
  int GetDeviceIndex() const override { return device_index_; }
  std::string GetDeviceName() const override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Нативные хэндлы
  // ═══════════════════════════════════════════════════════════════

  void* GetNativeContext() const override;
  void* GetNativeDevice() const override;
  void* GetNativeQueue() const override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Управление памятью
  // ═══════════════════════════════════════════════════════════════

  // Выделяет device memory через hipMalloc. flags — игнорируется (HIP не имеет
  // аналога CL_MEM_FLAGS). Возвращает nullptr при ошибке (не бросает).
  void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
  // hipMallocManaged — unified memory. CPU может читать без D2H (для отладки).
  // Освобождать через Free() (hipFree совместим с managed memory).
  void* AllocateManaged(size_t size_bytes) override;
  // hipFree(ptr). Безопасен для nullptr. Логирует ошибки через plog.
  void Free(void* ptr) override;

  // Все три Memcpy — СИНХРОННЫЕ: внутри вызывают Async + hipStreamSynchronize.
  // Совместимость с OpenCL backend, где enqueueWriteBuffer с CL_TRUE блокирует.
  // Для асинхронной передачи — используй hipMemcpy*Async напрямую через GetCore().
  void MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Синхронизация
  // ═══════════════════════════════════════════════════════════════

  // Блокирует CPU до завершения ВСЕХ операций в stream_. Вызывай перед чтением результатов.
  void Synchronize() override;
  // Non-blocking: hipStreamQuery — только «подталкивает» очередь, не ждёт завершения.
  // Аналог clFlush. Для реальной синхронизации используй Synchronize().
  void Flush() override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Возможности устройства
  // ═══════════════════════════════════════════════════════════════

  // HIP unified memory (hipMallocManaged) технически есть, но через IBackend
  // SVM не экспонируем: модули работают с явными HtoD/DtoH копиями.
  // Для unified memory в ROCm используй hipMallocManaged() напрямую через GetCore().
  bool SupportsSVM() const override { return false; }
  bool SupportsDoublePrecision() const override;  // device_props_.arch.hasDoubles (gfx900+)
  size_t GetMaxWorkGroupSize() const override;    // maxThreadsPerBlock из hipDeviceProp_t
  size_t GetGlobalMemorySize() const override;    // totalGlobalMem — кешировано при init
  size_t GetFreeMemorySize() const override;      // hipMemGetInfo — runtime запрос, не кеш
  size_t GetLocalMemorySize() const override;     // sharedMemPerBlock (LDS на AMD)

  // ═══════════════════════════════════════════════════════════════
  // Специфичные для ROCm методы
  // ═══════════════════════════════════════════════════════════════

  // Прямой доступ к ROCmCore для модулей, которым нужны HIP-специфичные хэндлы
  // (hipStream_t для hipFFT, hipDevice_t для hipDeviceGetAttribute и т.п.).
  // Бросает std::runtime_error если не инициализирован.
  ROCmCore& GetCore();
  const ROCmCore& GetCore() const;

  MemoryManager* GetMemoryManager() override;
  const MemoryManager* GetMemoryManager() const override;

  /**
   * @brief Доступ к пулу дополнительных HIP streams для параллельных операций
   *
   * StreamPool инициализируется автоматически в Initialize() с 2 streams.
   * Используется модулями для параллельного запуска kernel'ов на одном GPU.
   * Основной stream (GetCore().GetStream()) — для последовательных операций.
   *
   * @return Ссылка на StreamPool; бросает если не инициализирован
   */
  StreamPool& GetStreamPool();
  const StreamPool& GetStreamPool() const;

protected:
  int device_index_;    ///< Индекс HIP устройства (hipSetDevice argument)
  bool initialized_;    ///< true после успешного Initialize()
  // owns_resources_: если false — core_ был создан снаружи и нельзя его уничтожать.
  // Обычно true (ROCmBackend создаёт core_ сам в Initialize()).
  bool owns_resources_;

  std::unique_ptr<ROCmCore> core_;                  ///< HIP stream + device props — сердце бэкенда
  std::unique_ptr<MemoryManager> memory_manager_;   ///< Пул hipMalloc буферов
  StreamPool stream_pool_;                           ///< Пул дополнительных HIP streams (Ref04)

  // Кешированные хэндлы из core_ — для быстрого доступа без разыменования unique_ptr.
  // Обновляются в Initialize() и обнуляются в Cleanup().
  hipDevice_t device_;   ///< hipDeviceGet handle — integer ID устройства
  hipStream_t stream_;   ///< Основной stream для всех операций этого backend

  mutable std::mutex mutex_;  ///< Защита Initialize()/Cleanup() при многопоточном создании

private:
  // Собирает GPUDeviceInfo из hipDeviceProp_t через ROCmCore.
  // Вынесен в приватный метод, чтобы GetDeviceInfo() оставался const.
  GPUDeviceInfo QueryDeviceInfo() const;
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
