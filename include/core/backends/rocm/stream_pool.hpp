#pragma once

/**
 * @file stream_pool.hpp
 * @brief StreamPool — пул hipStream_t для параллельных GPU-операций (ROCm)
 *
 * ROCm аналог CommandQueuePool (OpenCL).
 * Управляет несколькими HIP streams для параллельного выполнения kernel'ов
 * на одном GPU. Round-robin распределение, thread-safe, RAII.
 *
 * Использование:
 * @code
 * StreamPool pool;
 * pool.Initialize(4, 0);  // 4 потока на GPU 0
 *
 * hipStream_t s1 = pool.GetStream(0);
 * hipStream_t s2 = pool.GetStream(1);
 *
 * hipModuleLaunchKernel(kernel, ..., s1, ...);  // Первый поток
 * hipModuleLaunchKernel(kernel, ..., s2, ...);  // Второй поток (параллельно)
 *
 * pool.SynchronizeAll();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-25
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>

#include <vector>
#include <mutex>
#include <cstddef>

namespace drv_gpu_lib {

/**
 * @class StreamPool
 * @brief Пул HIP streams с round-robin доступом
 *
 * Thread-safe. RAII — streams освобождаются в деструкторе.
 * Каждый stream — независимая очередь команд на GPU.
 */
class StreamPool {
public:
  StreamPool();
  ~StreamPool();

  // No copy
  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;

  // Move
  StreamPool(StreamPool&& other) noexcept;
  StreamPool& operator=(StreamPool&& other) noexcept;

  /**
   * @brief Создать пул streams
   * @param count Количество streams (0 = 2 по умолчанию)
   * @param device_index HIP device index (0, 1, 2, ...)
   * @return true если создан хотя бы один stream
   */
  bool Initialize(int count = 0, int device_index = 0);

  /**
   * @brief Получить stream по индексу (round-robin)
   * @param index Индекс (index % count)
   * @return hipStream_t или nullptr если пул пуст
   */
  hipStream_t GetStream(size_t index = 0);

  /// Количество streams в пуле
  size_t GetStreamCount() const;

  /// Синхронизировать все streams (блокирующий)
  void SynchronizeAll();

  /// Пул инициализирован?
  bool IsInitialized() const { return initialized_; }

  /// Индекс GPU
  int GetDeviceIndex() const { return device_index_; }

private:
  /// Освободить все streams (вызывается из деструктора и Initialize)
  void Cleanup();

  std::vector<hipStream_t> streams_;
  int device_index_ = 0;
  bool initialized_ = false;
  mutable std::mutex mutex_;
};

} // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
