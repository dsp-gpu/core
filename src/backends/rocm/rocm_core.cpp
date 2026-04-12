#if ENABLE_ROCM

#include "rocm_core.hpp"
#include "../../logger/logger.hpp"
#include <sstream>
#include <iomanip>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создаёт ROCmCore для заданного GPU
 *
 * Только инициализирует поля; реальный HIP init — в Initialize().
 * ЗАЧЕМ разделены: конструктор не бросает исключений, ошибки — через Initialize().
 *
 * @param device_index Индекс HIP устройства (аргумент hipSetDevice)
 */
ROCmCore::ROCmCore(int device_index)
    : device_index_(device_index),
      initialized_(false),
      owns_stream_(true),
      device_(0),
      stream_(nullptr),
      device_props_{} {
  DRVGPU_LOG_DEBUG_GPU(device_index, "ROCmCore",
                       "Created for device index " + std::to_string(device_index));
}

// ════════════════════════════════════════════════════════════════════════════
// Деструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Освобождает stream; вызывает ReleaseResources() если инициализирован
 */
ROCmCore::~ROCmCore() {
  if (initialized_) {
    ReleaseResources();
    initialized_ = false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Move-конструктор — передаёт владение HIP ресурсами без их пересоздания
 *
 * other.initialized_ → false: деструктор other НЕ вызовет ReleaseResources()
 * на уже переданных хэндлах — иначе double-free stream_.
 */
ROCmCore::ROCmCore(ROCmCore&& other) noexcept
    : device_index_(other.device_index_),
      initialized_(other.initialized_),
      owns_stream_(other.owns_stream_),
      device_(other.device_),
      stream_(other.stream_),
      device_props_(other.device_props_) {
  other.initialized_ = false;
  other.owns_stream_ = false;
  other.device_ = 0;
  other.stream_ = nullptr;
}

/**
 * @brief Move-присваивание — сначала освобождает свои ресурсы, затем перенимает чужие
 *
 * Cleanup() вызывается ПЕРВЫМ — иначе свой stream_ останется без владельца (утечка).
 */
ROCmCore& ROCmCore::operator=(ROCmCore&& other) noexcept {
  if (this != &other) {
    Cleanup();
    device_index_ = other.device_index_;
    initialized_ = other.initialized_;
    owns_stream_ = other.owns_stream_;
    device_ = other.device_;
    stream_ = other.stream_;
    device_props_ = other.device_props_;
    other.initialized_ = false;
    other.owns_stream_ = false;
    other.device_ = 0;
    other.stream_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализирует HIP runtime и создаёт stream для device_index_
 *
 * Thread-safe (mutex_). Идемпотентен — повторный вызов логирует WARNING, не ломает состояние.
 * Делегирует основную работу в InitializeHIP() — 6 шагов.
 *
 * @throws std::runtime_error если HIP недоступен или device_index_ невалиден
 */
void ROCmCore::Initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    DRVGPU_LOG_WARNING_GPU(device_index_, "ROCmCore",
                           "Device " + std::to_string(device_index_) + " already initialized");
    return;
  }

  InitializeHIP();
  initialized_ = true;

  DRVGPU_LOG_INFO_GPU(device_index_, "ROCmCore",
                      "Device " + std::to_string(device_index_) + " initialized: " + GetDeviceName());
}

/**
 * @brief Инициализация с внешним hipStream_t (External Context Integration)
 *
 * Используется когда hipStream_t уже создан внешней библиотекой или приложением.
 * Получает device handle и device_props_ — но НЕ создаёт собственный stream.
 * owns_stream_ = false → ReleaseResources() не вызывает hipStreamDestroy.
 *
 * Порядок (без hipStreamCreate):
 * 1. hipSetDevice — выбрать устройство
 * 2. hipDeviceGet — получить device handle
 * 3. hipGetDeviceProperties — свойства для GetDeviceInfo()/GetGlobalMemorySize() и пр.
 * 4. stream_ = external_stream — сохраняем без владения
 *
 * @param device_index   Индекс HIP устройства (аргумент hipSetDevice)
 * @param external_stream Внешний поток команд — НЕ будет уничтожен при Cleanup()
 * @throws std::runtime_error если device_index невалиден или hipSetDevice/hipDeviceGet упали
 */
void ROCmCore::InitializeFromExternalStream(int device_index, hipStream_t external_stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    DRVGPU_LOG_WARNING_GPU(device_index, "ROCmCore",
                           "Device " + std::to_string(device_index) +
                           " already initialized (InitializeFromExternalStream ignored)");
    return;
  }

  if (!external_stream) {
    throw std::runtime_error("ROCmCore::InitializeFromExternalStream: external_stream is null");
  }

  device_index_ = device_index;

  // Шаг 1: Выбрать устройство (нужно для hipDeviceGet и hipGetDeviceProperties)
  CheckHIPError(hipSetDevice(device_index), "hipSetDevice");

  // Шаг 2: Получить device handle
  CheckHIPError(hipDeviceGet(&device_, device_index), "hipDeviceGet");

  // Шаг 3: Получить свойства устройства (для GetDeviceInfo, GetGlobalMemorySize и пр.)
  CheckHIPError(hipGetDeviceProperties(&device_props_, device_index),
                "hipGetDeviceProperties");

  // Шаг 4: Сохраняем внешний stream без владения
  stream_ = external_stream;
  owns_stream_ = false;

  initialized_ = true;

  DRVGPU_LOG_INFO_GPU(device_index_, "ROCmCore",
                      "Initialized from external stream on device " +
                      std::to_string(device_index_) + " (" + GetDeviceName() + ")" +
                      " [owns_stream=false]");
}

/**
 * @brief 6 шагов HIP инициализации для конкретного устройства
 *
 * Вынесен из Initialize() чтобы тот остался читаемым (логика vs. механика).
 * Вызывается только под mutex_ из Initialize().
 *
 * @throws std::runtime_error при любой ошибке или невалидном device_index_
 */
void ROCmCore::InitializeHIP() {
  // Шаг 1: Инициализация HIP runtime
  CheckHIPError(hipInit(0), "hipInit");

  // Шаг 2: Проверить количество устройств
  int device_count = 0;
  CheckHIPError(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

  if (device_count == 0) {
    throw std::runtime_error("No HIP/ROCm devices found");
  }

  if (device_index_ < 0 || device_index_ >= device_count) {
    throw std::runtime_error(
        "Invalid device index: " + std::to_string(device_index_) +
        ". Available devices: " + std::to_string(device_count));
  }

  // Шаг 3: Выбрать устройство
  CheckHIPError(hipSetDevice(device_index_), "hipSetDevice");

  // Шаг 4: Получить device handle
  CheckHIPError(hipDeviceGet(&device_, device_index_), "hipDeviceGet");

  // Шаг 5: Получить свойства устройства
  CheckHIPError(hipGetDeviceProperties(&device_props_, device_index_),
                "hipGetDeviceProperties");

  // Шаг 6: Создать stream
  CheckHIPError(hipStreamCreate(&stream_), "hipStreamCreate");

  DRVGPU_LOG_DEBUG_GPU(device_index_, "ROCmCore",
                       "HIP initialized for device " + std::to_string(device_index_));
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка ресурсов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Освобождает stream; идемпотентен, thread-safe
 *
 * Double-checked locking: первая проверка без mutex_ — быстрый выход без захвата мьютекса.
 * Вторая под mutex_ — защита от гонки двух потоков, прошедших первую проверку одновременно.
 */
void ROCmCore::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    return;
  }

  ReleaseResources();
  initialized_ = false;
  DRVGPU_LOG_DEBUG_GPU(device_index_, "ROCmCore",
                       "Device " + std::to_string(device_index_) + " cleaned up");
}

/**
 * @brief Разрушает HIP stream; намеренно НЕ вызывает hipDeviceReset()
 *
 * hipDeviceReset() сбрасывает ВСЁ состояние GPU для всего процесса —
 * убил бы другие ROCmBackend и OpenCL контексты на том же устройстве.
 * device_ — integer handle, не требует явного освобождения через HIP API.
 */
void ROCmCore::ReleaseResources() {
  if (stream_ && owns_stream_) {
    // Уничтожаем stream только если он наш (created via hipStreamCreate в InitializeHIP).
    // Если owns_stream_=false — stream принадлежит внешнему коду, не трогаем.
    (void)hipStreamDestroy(stream_);
  }
  stream_ = nullptr;
  owns_stream_ = true;  // сброс в default для переиспользования объекта
  // device_ — integer handle, не требует явного освобождения через HIP API.
  // hipDeviceReset() намеренно НЕ вызывается: он сбросил бы всё состояние GPU
  // для всего процесса, включая другие ROCmBackend / OpenCL контексты на том же устройстве.
  device_ = 0;
}

// ════════════════════════════════════════════════════════════════════════════
// СТАТИЧЕСКИЕ МЕТОДЫ - Multi-GPU Discovery
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Возвращает число доступных AMD GPU без инициализации ROCmCore
 *
 * Вызывает hipGetDeviceCount напрямую. Возвращает 0 при ошибке
 * (нет AMD GPU, ROCm не установлен или ошибка драйвера).
 */
int ROCmCore::GetAvailableDeviceCount() {
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    return 0;
  }
  return count;
}

/**
 * @brief Форматированный список всех доступных AMD GPU (диагностика)
 *
 * Не требует инициализации ROCmCore — читает hipDeviceProp_t напрямую для каждого устройства.
 * ЗАЧЕМ: вызывается при старте приложения для вывода в лог/консоль, чтобы видеть
 * конфигурацию GPU окружения (особенно важно в multi-GPU системах с 10+ GPU).
 *
 * @return Многострочная строка-таблица всех GPU; "No devices found!" если GPU недоступны
 */
std::string ROCmCore::GetAllDevicesInfo() {
  std::ostringstream oss;
  int count = GetAvailableDeviceCount();

  oss << "\n" << std::string(70, '=') << "\n";
  oss << "Available ROCm/HIP GPU Devices\n";
  oss << std::string(70, '=') << "\n\n";

  if (count == 0) {
    oss << "  No devices found!\n";
  } else {
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t props;
      hipError_t err = hipGetDeviceProperties(&props, i);
      if (err != hipSuccess) continue;

      oss << "  [" << i << "] " << props.name << "\n";
      oss << "      GCN Arch: " << props.gcnArchName << "\n";
      oss << "      Memory: " << std::fixed << std::setprecision(2)
          << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
      oss << "      Compute Units: " << props.multiProcessorCount << "\n";
      oss << "      Clock: " << props.clockRate / 1000 << " MHz\n\n";
    }
  }

  oss << std::string(70, '=') << "\n";
  return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о девайсе
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Имя устройства из hipDeviceProp_t.name
 * @return Строка вида "AMD Radeon RX 9070 XT"; "Unknown" если не инициализирован
 */
std::string ROCmCore::GetDeviceName() const {
  if (!initialized_) return "Unknown";
  return std::string(device_props_.name);
}

/**
 * @brief Вендор GPU — всегда "AMD" (ROCm/HIP поддерживает только AMD GPU)
 */
std::string ROCmCore::GetVendor() const {
  return "AMD";
}

/**
 * @brief GCN архитектура из hipDeviceProp_t.gcnArchName
 * @return Строка вида "gfx1201" (RDNA4), "gfx1100" (RDNA3); "Unknown" если не инициализирован
 * @note Используется как суррогат версии драйвера в GPUDeviceInfo.driver_version
 */
std::string ROCmCore::GetArchName() const {
  if (!initialized_) return "Unknown";
  return std::string(device_props_.gcnArchName);
}

/**
 * @brief Общий объём VRAM в байтах (кешировано при инициализации)
 * @return props.totalGlobalMem; 0 если не инициализирован
 */
size_t ROCmCore::GetGlobalMemorySize() const {
  if (!initialized_) return 0;
  return device_props_.totalGlobalMem;
}

/**
 * @brief Текущий объём свободной VRAM в байтах (runtime запрос, не кеш)
 *
 * ЗАЧЕМ не кешировать: свободная VRAM меняется динамически по мере выделений.
 * Fallback при ошибке hipMemGetInfo: 90% от totalGlobalMem — консервативная оценка
 * (иногда нужна для драйверов с поломанным hipMemGetInfo).
 *
 * @return Байты свободной VRAM; 0 если не инициализирован
 */
size_t ROCmCore::GetFreeMemorySize() const {
  if (!initialized_) return 0;

  // hipMemGetInfo возвращает данные для ТЕКУЩЕГО устройства (thread-local hipSetDevice).
  // В multi-GPU системе (10+ GPU) другой поток мог переключить device →
  // без hipSetDevice можем получить free memory не того GPU.
  hipSetDevice(device_index_);

  size_t free_mem = 0;
  size_t total_mem = 0;
  hipError_t err = hipMemGetInfo(&free_mem, &total_mem);
  if (err == hipSuccess) {
    return free_mem;
  }

  // Fallback: hipMemGetInfo не сработал (редко, но бывает на некоторых драйверах) →
  // возвращаем 90% от общего объёма как консервативную оценку.
  return static_cast<size_t>(static_cast<double>(device_props_.totalGlobalMem) * 0.9);
}

/**
 * @brief Объём LDS (Local Data Store) на блок в байтах
 * @return sharedMemPerBlock из hipDeviceProp_t; 0 если не инициализирован
 */
size_t ROCmCore::GetLocalMemorySize() const {
  if (!initialized_) return 0;
  return device_props_.sharedMemPerBlock;
}

/**
 * @brief Число Compute Units (CU) — аналог SM на NVIDIA
 * @return multiProcessorCount; 0 если не инициализирован
 */
int ROCmCore::GetComputeUnits() const {
  if (!initialized_) return 0;
  return device_props_.multiProcessorCount;
}

/**
 * @brief Максимальный размер work-group (threads per block)
 *
 * Это hard-limit GPU: kernel с blockDim > maxThreadsPerBlock не запустится.
 * Типичные значения: 1024 (большинство AMD GPU).
 *
 * @return maxThreadsPerBlock из hipDeviceProp_t; 0 если не инициализирован
 */
size_t ROCmCore::GetMaxWorkGroupSize() const {
  if (!initialized_) return 0;
  return static_cast<size_t>(device_props_.maxThreadsPerBlock);
}

/**
 * @brief Тактовая частота GPU в МГц
 * @return clockRate / 1000 (конвертация kHz → MHz); 0 если не инициализирован
 */
size_t ROCmCore::GetMaxClockFrequency() const {
  if (!initialized_) return 0;
  return static_cast<size_t>(device_props_.clockRate / 1000);  // kHz -> MHz
}

/**
 * @brief Warp (wavefront) size из hipDeviceProp_t — авторитетный источник
 *
 * CDNA / Vega (gfx900-gfx942) → 64 wavefront width
 * RDNA (gfx10xx, gfx11xx, gfx12xx) → 32 wavefront width
 * Не используем строковую эвристику — hipDeviceProp_t.warpSize точнее.
 *
 * @return warpSize из device_props_; 32 по умолчанию если не инициализирован
 */
int ROCmCore::GetWarpSize() const {
  if (!initialized_) return 32;
  return device_props_.warpSize;
}

/**
 * @brief Поддерживает ли устройство float64 в hardware (без software emulation)
 *
 * Флаг device_props_.arch.hasDoubles выставляется HIP для gfx900+.
 * На некоторых APU (gfx902) = 0 — fp64 работает через software emulation (очень медленно).
 *
 * @return true если fp64 реализован в железе; false если SW emulation или не инициализирован
 */
bool ROCmCore::SupportsDoublePrecision() const {
  if (!initialized_) return false;
  // device_props_.arch.hasDoubles — официальный флаг HIP, выставляется для gfx900+.
  // На RDNA4 (gfx1201) и большинстве AMD GPU этот флаг = 1.
  // На некоторых мобильных APU (gfx902) может быть 0 — fp64 медленный (software emulation).
  return device_props_.arch.hasDoubles != 0;
}

/**
 * @brief Форматированный отчёт о свойствах КОНКРЕТНОГО устройства (диагностика)
 *
 * Возвращает многострочную строку со всеми параметрами.
 * ЗАЧЕМ: вывод при инициализации в лог/консоль для диагностики конкретного GPU.
 * В отличие от GetAllDevicesInfo() — только для this->device_index_.
 *
 * @return Таблица свойств GPU; поля с "Unknown"/0 если не инициализирован
 */
std::string ROCmCore::GetDeviceInfo() const {
  std::ostringstream oss;

  oss << "\n" << std::string(70, '=') << "\n";
  oss << "ROCm/HIP Device [" << device_index_ << "] Information\n";
  oss << std::string(70, '=') << "\n\n";

  oss << std::left << std::setw(25) << "Device Index:" << device_index_ << "\n";
  oss << std::left << std::setw(25) << "Device Name:" << GetDeviceName() << "\n";
  oss << std::left << std::setw(25) << "Vendor:" << GetVendor() << "\n";
  oss << std::left << std::setw(25) << "GCN Arch:" << GetArchName() << "\n";

  size_t global_mem = GetGlobalMemorySize();
  size_t local_mem = GetLocalMemorySize();
  oss << std::left << std::setw(25) << "Global Memory:" << std::fixed << std::setprecision(2)
      << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
  oss << std::left << std::setw(25) << "Local Memory:" << (local_mem / 1024.0) << " KB\n";
  oss << std::left << std::setw(25) << "Free Memory:" << std::fixed << std::setprecision(2)
      << (GetFreeMemorySize() / (1024.0 * 1024.0 * 1024.0)) << " GB\n";

  oss << std::left << std::setw(25) << "Compute Units:" << GetComputeUnits() << "\n";
  oss << std::left << std::setw(25) << "Max Work Group Size:" << GetMaxWorkGroupSize() << "\n";
  oss << std::left << std::setw(25) << "Clock:" << GetMaxClockFrequency() << " MHz\n";
  oss << std::left << std::setw(25) << "Double Precision:" << (SupportsDoublePrecision() ? "YES" : "NO") << "\n";

  oss << "\n" << std::string(70, '=') << "\n";
  return oss.str();
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
