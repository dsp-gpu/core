#pragma once

/**
 * @file rocm_core.hpp
 * @brief Per-device HIP/ROCm контекст (аналог opencl_core.hpp)
 *
 * ROCmCore управляет HIP контекстом для КОНКРЕТНОГО устройства.
 * Каждый экземпляр владеет СВОИМ device по device_index.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <stdexcept>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Утилита: Проверка HIP ошибок
// ════════════════════════════════════════════════════════════════════════════

inline void CheckHIPError(hipError_t error, const std::string& operation) {
  if (error != hipSuccess) {
    std::string error_msg = "HIP Error [" + std::to_string(static_cast<int>(error)) +
                            ": " + hipGetErrorString(error) + "] in " + operation;
    throw std::runtime_error(error_msg);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ROCmCore - Per-Device HIP контекст (Multi-GPU поддержка)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ROCmCore
 * @brief Управляет HIP контекстом для КОНКРЕТНОГО устройства
 *
 * MULTI-GPU ARCHITECTURE:
 * Каждый экземпляр ROCmCore владеет СВОИМ устройством по device_index.
 *
 * Ответственность:
 * - Инициализация HIP runtime и выбор девайса
 * - Создание и владение stream
 * - Информация о девайсе (hipDeviceProp_t)
 * - Thread-safe доступ
 *
 * НЕ управляет:
 * - Буферами (это делает GPUBuffer через IBackend)
 * - Программами/кернелами (это модули)
 */
class ROCmCore {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструктор и деструктор
  // ═══════════════════════════════════════════════════════════════

  explicit ROCmCore(int device_index = 0);
  ~ROCmCore();

  // ═══════════════════════════════════════════════════════════════
  // Инициализация из внешнего stream (External Context Integration)
  // ═══════════════════════════════════════════════════════════════

  // Инициализация с внешним hipStream_t (owns_stream_ = false).
  // Получает device handle и device_props_ — всё кроме stream_.
  // ReleaseResources() НЕ вызывает hipStreamDestroy (stream чужой).
  // Бросает std::runtime_error если device_index невалиден.
  void InitializeFromExternalStream(int device_index, hipStream_t external_stream);

  // ═══════════════════════════════════════════════════════════════
  // Запрет копирования, разрешение перемещения
  // ═══════════════════════════════════════════════════════════════
  ROCmCore(const ROCmCore&) = delete;
  ROCmCore& operator=(const ROCmCore&) = delete;
  ROCmCore(ROCmCore&& other) noexcept;
  ROCmCore& operator=(ROCmCore&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════
  // Инициализация
  // ═══════════════════════════════════════════════════════════════

  // 6 шагов HIP init: hipInit → hipGetDeviceCount → hipSetDevice →
  // hipDeviceGet → hipGetDeviceProperties → hipStreamCreate.
  // Thread-safe (mutex_). Идемпотентен — повторный вызов выдаёт WARNING, не ломает состояние.
  void Initialize();
  // Уничтожает stream (hipStreamDestroy). НЕ вызывает hipDeviceReset() —
  // это сбросило бы состояние GPU для всего процесса (все контексты на устройстве).
  void Cleanup();
  bool IsInitialized() const { return initialized_; }

  // ═══════════════════════════════════════════════════════════════
  // Getters для HIP объектов
  // ═══════════════════════════════════════════════════════════════

  hipDevice_t GetDevice() const { return device_; }
  hipStream_t GetStream() const { return stream_; }
  int GetDeviceIndex() const { return device_index_; }
  // true — stream создан нами (hipStreamCreate); false — stream внешний, не уничтожать.
  bool OwnsStream() const { return owns_stream_; }

  // ═══════════════════════════════════════════════════════════════
  // Информация о девайсе
  // ═══════════════════════════════════════════════════════════════

  std::string GetDeviceInfo() const;    // Форматированный многострочный отчёт (диагностика/лог)
  std::string GetDeviceName() const;    // props.name: напр. "AMD Radeon RX 9070 XT"
  std::string GetVendor() const;        // Всегда "AMD" — ROCm поддерживает только AMD GPU
  std::string GetArchName() const;      // props.gcnArchName: напр. "gfx1201" (RDNA4)
  size_t GetGlobalMemorySize() const;   // totalGlobalMem — кешировано при init, не меняется
  size_t GetFreeMemorySize() const;     // hipMemGetInfo — runtime запрос! Значение изменяется
  size_t GetLocalMemorySize() const;    // sharedMemPerBlock (LDS на AMD), кешировано
  int GetComputeUnits() const;          // multiProcessorCount (CU count)
  size_t GetMaxWorkGroupSize() const;   // maxThreadsPerBlock — hard limit для kernel launch
  size_t GetMaxClockFrequency() const;  // clockRate / 1000: kHz → MHz
  int GetWarpSize() const;              // warpSize из hipDeviceProp_t — авторитетный источник (64 для CDNA/Vega, 32 для RDNA)
  // device_props_.arch.hasDoubles: gfx900+ = 1, некоторые APU (gfx902) = 0 (SW emulation).
  bool SupportsDoublePrecision() const;

  // ═══════════════════════════════════════════════════════════════
  // СТАТИЧЕСКИЕ МЕТОДЫ для обнаружения GPU (Multi-GPU support)
  // ═══════════════════════════════════════════════════════════════

  // Не требует инициализации ROCmCore — вызывает hipGetDeviceCount напрямую.
  // Возвращает 0 при ошибке (нет AMD GPU или не установлен ROCm runtime).
  static int GetAvailableDeviceCount();
  // Диагностика: форматированный список всех AMD GPU (name, arch, VRAM, CU, clock).
  // Вызывается при старте для вывода в лог. Не требует инициализации ROCmCore.
  static std::string GetAllDevicesInfo();

private:
  int device_index_;         ///< Аргумент hipSetDevice — порядковый номер GPU
  bool initialized_;         ///< true после успешного Initialize()

  hipDevice_t device_;       ///< Integer handle устройства (от hipDeviceGet)
  hipStream_t stream_;       ///< Основной поток команд — создаётся в Initialize(), уничтожается в ReleaseResources()
  hipDeviceProp_t device_props_;  ///< Кеш свойств устройства (от hipGetDeviceProperties); заполняется однократно при инициализации

  mutable std::mutex mutex_;  ///< Защита Initialize()/Cleanup() от гонок при одновременном вызове

  // owns_stream_: false если stream_ инициализирован через InitializeFromExternalStream.
  // В этом случае ReleaseResources() НЕ вызывает hipStreamDestroy — stream принадлежит вызывающему коду.
  bool owns_stream_;

  // Выполняет 6 шагов HIP init: hipInit → hipGetDeviceCount → hipSetDevice →
  // hipDeviceGet → hipGetDeviceProperties → hipStreamCreate.
  // Бросает std::runtime_error при любой ошибке.
  void InitializeHIP();

  // Уничтожает stream (hipStreamDestroy) и обнуляет device_.
  // НЕ вызывает hipDeviceReset() — это сбросит всё состояние GPU для всего процесса,
  // включая другие бэкенды и контексты на том же устройстве.
  void ReleaseResources();
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
