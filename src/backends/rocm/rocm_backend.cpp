#if ENABLE_ROCM

#include "rocm_backend.hpp"
#include "../../config/gpu_config.hpp"

#include <sstream>
#include <iomanip>
#include <vector>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создаёт неинициализированный ROCmBackend
 *
 * Все хэндлы — nullptr/0. Реальный HIP init — в Initialize(device_index).
 * ЗАЧЕМ: конструктор без параметров нужен для размещения в контейнерах и
 * отложенной инициализации (device_index известен позже, при парсинге конфига).
 */
ROCmBackend::ROCmBackend()
    : device_index_(-1),
      initialized_(false),
      owns_resources_(true),
      core_(nullptr),
      device_(0),
      stream_(nullptr) {
}

/**
 * @brief Деструктор — вызывает Cleanup() для освобождения HIP ресурсов
 */
ROCmBackend::~ROCmBackend() {
  Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Move-конструктор — передаёт владение HIP ресурсами без пересоздания
 *
 * Обнуляет other.initialized_ = false: деструктор other НЕ вызовет Cleanup() на уже
 * переданных core_ и memory_manager_ — иначе double-free.
 * owns_resources_ → false для other: даже если other думал что владеет core_, теперь владеем мы.
 */
ROCmBackend::ROCmBackend(ROCmBackend&& other) noexcept
    : device_index_(other.device_index_),
      initialized_(other.initialized_),
      owns_resources_(other.owns_resources_),
      core_(std::move(other.core_)),
      memory_manager_(std::move(other.memory_manager_)),
      stream_pool_(std::move(other.stream_pool_)),
      device_(other.device_),
      stream_(other.stream_) {
  // Сбрасываем источник: деструктор other не должен вызвать Cleanup() с уже переданными ресурсами.
  // owns_resources_ → false: даже если other думал что владеет core_, теперь владеем мы.
  other.device_index_ = -1;
  other.initialized_ = false;
  other.owns_resources_ = false;
  other.device_ = 0;
  other.stream_ = nullptr;
}

/**
 * @brief Move-присваивание — освобождает свои ресурсы, затем перенимает чужие
 *
 * Cleanup() вызывается ПЕРВЫМ — иначе свои core_ и memory_manager_ останутся без владельца.
 */
ROCmBackend& ROCmBackend::operator=(ROCmBackend&& other) noexcept {
  if (this != &other) {
    Cleanup();  // Освобождаем свои ресурсы ПЕРЕД перемещением — иначе утечка.
    device_index_ = other.device_index_;
    initialized_ = other.initialized_;
    owns_resources_ = other.owns_resources_;
    core_ = std::move(other.core_);
    memory_manager_ = std::move(other.memory_manager_);
    stream_pool_ = std::move(other.stream_pool_);
    device_ = other.device_;
    stream_ = other.stream_;

    other.device_index_ = -1;
    other.initialized_ = false;
    other.owns_resources_ = false;
    other.device_ = 0;
    other.stream_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализирует ROCm бэкенд для заданного AMD GPU
 *
 * Thread-safe (mutex_). При повторном вызове без Cleanup() — сначала вызывает Cleanup().
 *
 * Порядок:
 * 1. ROCmCore(device_index) → Initialize() — 6 шагов HIP init
 * 2. Кешируем device_ и stream_ из core_ — для быстрого доступа без разыменования unique_ptr
 * 3. MemoryManager(this) — пул hipMalloc буферов
 *
 * @param device_index Индекс AMD GPU (0..N-1; проверяется в ROCmCore::InitializeHIP)
 * @throws std::runtime_error если device_index невалиден или HIP недоступен
 */
void ROCmBackend::Initialize(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    Cleanup();
  }

  device_index_ = device_index;
  owns_resources_ = true;

  DRVGPU_LOG_INFO_GPU(device_index, "ROCmBackend",
                      "Creating ROCmCore for device " + std::to_string(device_index));

  // Создаём СОБСТВЕННЫЙ ROCmCore для этого устройства
  core_ = std::make_unique<ROCmCore>(device_index);
  core_->Initialize();

  // Кешируем нативные хэндлы из core_ — избегаем разыменования unique_ptr в hot path
  device_ = core_->GetDevice();
  stream_ = core_->GetStream();

  // Создаём MemoryManager ПОСЛЕ инициализации core_ — MemoryManager хранит указатель на IBackend
  memory_manager_ = std::make_unique<MemoryManager>(this);

  // StreamPool: 2 дополнительных stream'а для параллельных операций (Ref04)
  stream_pool_.Initialize(2, device_index);

  initialized_ = true;

  DRVGPU_LOG_INFO_GPU(device_index_, "ROCmBackend",
                      "Initialized for device " + std::to_string(device_index) +
                      " (" + core_->GetDeviceName() +
                      ", StreamPool: " + std::to_string(stream_pool_.GetStreamCount()) + " streams)");
}

/**
 * @brief Инициализация с внешним hipStream_t (External Context Integration)
 *
 * Используется когда hipStream_t уже создан внешней библиотекой (hipBLAS, hipFFT, MIOpen).
 * Бэкенд подключается к существующему stream без захвата владения.
 *
 * Порядок:
 * 1. ROCmCore::InitializeFromExternalStream — получает device handle + device_props_
 *    без создания нового stream (owns_stream=false)
 * 2. Кешируем device_ и stream_ из core_
 * 3. MemoryManager(this) — создаём СОБСТВЕННЫЙ пул hipMalloc буферов
 *    (буферы наши, даже если stream внешний)
 * 4. owns_resources_ = false — Cleanup() не будет уничтожать stream
 *
 * @param device_index     Индекс AMD GPU (0..N-1)
 * @param external_stream  Внешний stream — НЕ будет уничтожен при Cleanup()
 * @throws std::runtime_error если уже инициализирован или параметры невалидны
 */
void ROCmBackend::InitializeFromExternalStream(int device_index, hipStream_t external_stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    throw std::runtime_error(
        "ROCmBackend::InitializeFromExternalStream: already initialized. Call Cleanup() first.");
  }

  if (!external_stream) {
    throw std::runtime_error(
        "ROCmBackend::InitializeFromExternalStream: external_stream is null");
  }

  device_index_ = device_index;
  // owns_resources_=false: Cleanup() будет вызывать core_.reset(),
  // но ROCmCore::owns_stream_=false → hipStreamDestroy НЕ вызывается.
  owns_resources_ = false;

  DRVGPU_LOG_INFO_GPU(device_index, "ROCmBackend",
                      "Attaching to external stream on device " + std::to_string(device_index));

  // ROCmCore получает device handle + device_props_, но НЕ создаёт stream
  core_ = std::make_unique<ROCmCore>(device_index);
  core_->InitializeFromExternalStream(device_index, external_stream);

  // Кешируем нативные хэндлы — избегаем разыменования unique_ptr в hot path
  device_ = core_->GetDevice();
  stream_ = external_stream;

  // MemoryManager — наш собственный (hipMalloc/hipFree буферы независимы от stream)
  memory_manager_ = std::make_unique<MemoryManager>(this);

  // StreamPool: 2 дополнительных stream'а (Ref04)
  stream_pool_.Initialize(2, device_index);

  initialized_ = true;

  DRVGPU_LOG_INFO_GPU(device_index_, "ROCmBackend",
                      "Attached to external stream on device " + std::to_string(device_index) +
                      " (" + core_->GetDeviceName() + ") [owns_resources=false"
                      ", StreamPool: " + std::to_string(stream_pool_.GetStreamCount()) + " streams]");
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Освобождает все ресурсы бэкенда
 *
 * Порядок освобождения критичен:
 * 1. memory_manager_ ПЕРВЫМ — буферы MemoryManager могут содержать hipFree вызовы
 *    на stream_ из core_; если core_ уничтожить раньше → dangling hipStream_t.
 * 2. core_ — hipStreamDestroy (если owns_resources_=true)
 *
 * Идемпотентен (ранний выход если !initialized_).
 * Вызывается автоматически из деструктора и из Initialize() при повторном вызове.
 */
void ROCmBackend::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    return;
  }

  int gpu_id_for_log = device_index_;
  DRVGPU_LOG_INFO_GPU(gpu_id_for_log, "ROCmBackend",
                      "Cleanup started for device " + std::to_string(device_index_) +
                      " (owns_resources = " + std::string(owns_resources_ ? "true" : "false") + ")");

  // Освобождаем StreamPool ПЕРЕД core_ — streams зависят от device context
  // StreamPool::~StreamPool() вызовет Cleanup() автоматически, но делаем явно для логирования
  stream_pool_ = StreamPool{};  // move-assign пустого → деструктор старого вызывает Cleanup()

  // Освобождаем MemoryManager
  memory_manager_.reset();

  // core_.reset() вызывается в обоих случаях — ROCmCore корректно обрабатывает owns_stream_:
  // - owns_resources_=true  → ROCmCore::owns_stream_=true  → hipStreamDestroy вызывается
  // - owns_resources_=false → ROCmCore::owns_stream_=false → hipStreamDestroy НЕ вызывается
  // Внешний stream уничтожается вызывающим кодом самостоятельно.
  core_.reset();

  device_ = 0;
  stream_ = nullptr;
  device_index_ = -1;
  initialized_ = false;

  DRVGPU_LOG_INFO_GPU(gpu_id_for_log, "ROCmBackend", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Информация об устройстве
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Возвращает структуру с информацией об устройстве
 *
 * Делегирует в QueryDeviceInfo() — выделен в приватный метод чтобы GetDeviceInfo() оставался const.
 * @return GPUDeviceInfo; поля по умолчанию (пустые) если core_ не инициализирован
 */
GPUDeviceInfo ROCmBackend::GetDeviceInfo() const {
  return QueryDeviceInfo();
}

/**
 * @brief Имя GPU устройства из hipDeviceProp_t.name
 * @return Строка вида "AMD Radeon RX 9070 XT"; "Unknown" если не инициализирован
 */
std::string ROCmBackend::GetDeviceName() const {
  if (!core_ || !core_->IsInitialized()) {
    return "Unknown";
  }
  return core_->GetDeviceName();
}

// ════════════════════════════════════════════════════════════════════════════
// Нативные хэндлы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Нативный контекст для совместимости с IBackend интерфейсом
 *
 * HIP не имеет явного контекстного объекта (в отличие от OpenCL cl_context).
 * Контекст управляется неявно через hipSetDevice — хранить нечего.
 * @return nullptr всегда
 */
void* ROCmBackend::GetNativeContext() const {
  return nullptr;
}

/**
 * @brief Нативный device handle как void*
 *
 * hipDevice_t — это int, поэтому reinterpret_cast через intptr_t:
 * сначала int → intptr_t (value-preserving), затем intptr_t → void* (pointer-sized).
 * ЗАЧЕМ: IBackend интерфейс возвращает void* для backend-агностичного кода.
 */
void* ROCmBackend::GetNativeDevice() const {
  // Делегируем в core_ — авторитетный источник (cached device_ = копия, используется внутри)
  if (!core_) return reinterpret_cast<void*>(static_cast<intptr_t>(device_));
  return reinterpret_cast<void*>(static_cast<intptr_t>(core_->GetDevice()));
}

/**
 * @brief Нативный HIP stream (hipStream_t) как void*
 *
 * hipStream_t — это указатель, безопасное приведение к void*.
 * Использование: hipFFT, hipBLAS принимают hipStream_t — получать через GetCore().GetStream().
 */
void* ROCmBackend::GetNativeQueue() const {
  // Делегируем в core_ — авторитетный источник (cached stream_ = копия, используется внутри)
  if (!core_) return static_cast<void*>(stream_);
  return static_cast<void*>(core_->GetStream());
}

// ════════════════════════════════════════════════════════════════════════════
// Управление памятью
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Выделяет device memory через hipMalloc
 *
 * flags игнорируется — HIP не имеет аналога CL_MEM_FLAGS (READ_ONLY, USE_HOST_PTR и т.п.).
 * Для pinned host memory → hipMallocHost. Для unified memory → hipMallocManaged.
 * Оба варианта вызываются напрямую через GetCore() — не через этот интерфейс.
 *
 * @param size_bytes Размер выделяемого буфера в байтах
 * @param flags Игнорируется (совместимость с IBackend интерфейсом)
 * @return Указатель на device memory; nullptr при ошибке или если не инициализирован
 * @note Не бросает — возвращает nullptr, ошибка записывается в лог
 */
void* ROCmBackend::Allocate(size_t size_bytes, unsigned int /*flags*/) {
  // flags игнорируем: hipMalloc не имеет аналога CL_MEM_FLAGS (READ_ONLY, USE_HOST_PTR и т.п.).
  // Все HIP буферы — обычная device memory (VRAM). Для pinned/unified используй hipMallocHost/hipMallocManaged.
  if (!initialized_) {
    return nullptr;
  }

  void* device_ptr = nullptr;
  hipError_t err = hipMalloc(&device_ptr, size_bytes);

  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "hipMalloc failed: " + std::string(hipGetErrorString(err)) +
                         " (requested " + std::to_string(size_bytes) + " bytes)");
    return nullptr;
  }

  return device_ptr;
}

/**
 * @brief Выделяет unified memory через hipMallocManaged
 *
 * CPU может читать содержимое напрямую без явного hipMemcpy — удобно для отладки
 * и checkpoint-операций. Освобождать через Free() (hipFree совместим).
 *
 * @param size_bytes Размер буфера в байтах
 * @return Указатель на managed memory; nullptr при ошибке
 */
void* ROCmBackend::AllocateManaged(size_t size_bytes) {
  if (!initialized_) {
    return nullptr;
  }

  void* ptr = nullptr;
  hipError_t err = hipMallocManaged(&ptr, size_bytes);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "hipMallocManaged failed: " + std::string(hipGetErrorString(err)) +
                         " (requested " + std::to_string(size_bytes) + " bytes)");
    return nullptr;
  }

  return ptr;
}

/**
 * @brief Освобождает device memory через hipFree
 *
 * Безопасен для nullptr (ранний выход). Логирует ошибки через plog (не бросает).
 * ВАЖНО: не вызывать для pinned (hipMallocHost) или managed (hipMallocManaged) памяти —
 * для них нужны hipFreeHost / hipFree соответственно.
 */
void ROCmBackend::Free(void* ptr) {
  if (ptr) {
    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) {
      DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                           "hipFree failed: " + std::string(hipGetErrorString(err)));
    }
  }
}

/**
 * @brief Копирует данные host → device memory (СИНХРОННО)
 *
 * Внутри: hipMemcpyHtoDAsync (DMA transfer) + hipStreamSynchronize (wait).
 * ЗАЧЕМ синхронно: сохраняем совместимость с OpenCL backend, где enqueueWriteBuffer
 * с CL_TRUE блокирует. Модули рассчитывают что данные доступны сразу после вызова.
 *
 * Для максимальной скорости (pipeline HtoD + kernel): использовать hipMemcpyHtoDAsync
 * напрямую через GetCore().GetStream() + синхронизировать позже перед kernel launch.
 *
 * @note Не бросает — ошибки записываются в лог (plog)
 */
void ROCmBackend::MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) {
  if (!dst || !src || !initialized_) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyHostToDevice - Invalid parameters");
    return;
  }

  hipError_t err = hipMemcpyHtoDAsync(dst, const_cast<void*>(src), size_bytes, stream_);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyHostToDevice failed: " + std::string(hipGetErrorString(err)));
    return;
  }

  // Синхронизируем для совместимости с синхронным API OpenCL backend
  (void)hipStreamSynchronize(stream_);
}

/**
 * @brief Копирует данные device → host memory (СИНХРОННО)
 *
 * Внутри: hipMemcpyDtoHAsync + hipStreamSynchronize.
 * Вызывать ПОСЛЕ того как все kernel'ы завершились (иначе читаем неготовые данные).
 * В нашем flow: kernel → MemcpyDeviceToHost → читаем результат на CPU.
 */
void ROCmBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
  if (!dst || !src || !initialized_) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToHost - Invalid parameters");
    return;
  }

  hipError_t err = hipMemcpyDtoHAsync(dst, const_cast<void*>(src), size_bytes, stream_);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToHost failed: " + std::string(hipGetErrorString(err)));
    return;
  }

  // Синхронизируем для совместимости с синхронным API OpenCL backend.
  (void)hipStreamSynchronize(stream_);
}

/**
 * @brief Копирует данные между двумя device буферами (СИНХРОННО)
 *
 * Внутри: hipMemcpyDtoDAsync + hipStreamSynchronize.
 * ЗАЧЕМ: конвейерное копирование результатов между модулями без возврата на CPU.
 * Типичный случай: output одного kernel → input следующего.
 * На AMD GPU DtoD копия идёт через GPU DMA engine — не нагружает PCIE шину.
 */
void ROCmBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
  if (!dst || !src || !initialized_) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToDevice - Invalid parameters");
    return;
  }

  hipError_t err = hipMemcpyDtoDAsync(dst, const_cast<void*>(src), size_bytes, stream_);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToDevice failed: " + std::string(hipGetErrorString(err)));
    return;
  }

  // Синхронизируем для совместимости с синхронным API OpenCL backend.
  (void)hipStreamSynchronize(stream_);
}

// ════════════════════════════════════════════════════════════════════════════
// Синхронизация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Блокирует CPU до завершения всех операций в stream_
 *
 * hipStreamSynchronize — ждёт пока все операции в очереди stream_ выполнятся.
 * КОГДА вызывать: после запуска kernel'ов, перед чтением результатов на CPU.
 * В нормальном flow: Launch kernel → Synchronize → MemcpyDeviceToHost.
 */
void ROCmBackend::Synchronize() {
  if (stream_) {
    hipError_t err = hipStreamSynchronize(stream_);
    if (err != hipSuccess) {
      DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                           "Synchronize failed: " + std::string(hipGetErrorString(err)));
    }
  }
}

/**
 * @brief Non-blocking «подталкивание» очереди (аналог clFlush)
 *
 * hipStreamQuery проверяет статус stream без блокировки:
 * hipSuccess — все операции завершены, hipErrorNotReady — ещё в процессе.
 * (void) намеренно игнорирует возвращаемое значение — нас интересует только эффект flush.
 * ЗАЧЕМ: в некоторых сценариях без явного flush GPU может задерживать старт операций.
 * Для реальной синхронизации → Synchronize().
 */
void ROCmBackend::Flush() {
  if (stream_) {
    // HIP: hipStreamQuery возвращает hipSuccess если все операции завершены,
    // или hipErrorNotReady если ещё в процессе — non-blocking check
    (void)hipStreamQuery(stream_);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Возможности устройства
// ════════════════════════════════════════════════════════════════════════════

bool ROCmBackend::SupportsDoublePrecision() const {
  if (!core_ || !core_->IsInitialized()) return false;
  return core_->SupportsDoublePrecision();
}

size_t ROCmBackend::GetMaxWorkGroupSize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetMaxWorkGroupSize();
}

size_t ROCmBackend::GetGlobalMemorySize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetGlobalMemorySize();
}

size_t ROCmBackend::GetFreeMemorySize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetFreeMemorySize();
}

size_t ROCmBackend::GetLocalMemorySize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetLocalMemorySize();
}

// ════════════════════════════════════════════════════════════════════════════
// Специфичные для ROCm методы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Прямой доступ к ROCmCore для HIP-специфичных операций
 *
 * ЗАЧЕМ: модули (hipFFT, statistics, heterodyne) нуждаются в hipStream_t для запуска
 * kernel'ов и библиотечных вызовов. GetCore().GetStream() — главный use case.
 *
 * @throws std::runtime_error если Initialize() ещё не вызван
 */
ROCmCore& ROCmBackend::GetCore() {
  if (!core_) {
    throw std::runtime_error("ROCmBackend::GetCore - Core not initialized");
  }
  return *core_;
}

const ROCmCore& ROCmBackend::GetCore() const {
  if (!core_) {
    throw std::runtime_error("ROCmBackend::GetCore - Core not initialized");
  }
  return *core_;
}

/**
 * @brief Доступ к пулу hipMalloc буферов
 * @return Указатель на MemoryManager; nullptr если не инициализирован
 */
MemoryManager* ROCmBackend::GetMemoryManager() {
  return memory_manager_.get();
}

const MemoryManager* ROCmBackend::GetMemoryManager() const {
  return memory_manager_.get();
}

// ════════════════════════════════════════════════════════════════════════════
// StreamPool — пул дополнительных streams (Ref04)
// ════════════════════════════════════════════════════════════════════════════

StreamPool& ROCmBackend::GetStreamPool() {
  if (!initialized_) {
    throw std::runtime_error("ROCmBackend::GetStreamPool - Not initialized");
  }
  return stream_pool_;
}

const StreamPool& ROCmBackend::GetStreamPool() const {
  if (!initialized_) {
    throw std::runtime_error("ROCmBackend::GetStreamPool - Not initialized");
  }
  return stream_pool_;
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Заполняет GPUDeviceInfo из свойств ROCmCore
 *
 * Нюансы заполнения:
 * - driver_version: содержит GCN arch string (напр. "gfx1201"), а НЕ версию драйвера —
 *   hipDeviceProp_t не предоставляет строку версии ROCm. Принято как допустимый суррогат.
 * - max_mem_alloc_size = global_memory: HIP не имеет per-allocation лимита
 *   (в OpenCL CL_DEVICE_MAX_MEM_ALLOC_SIZE = 1/4 от VRAM — здесь таких ограничений нет).
 * - supports_half = false: AMD GPU поддерживает fp16, но через IBackend не экспонируем —
 *   модули работают с float/float2 напрямую без обёртки через IBackend.
 *
 * @return Заполненный GPUDeviceInfo; пустой (поля по умолчанию) если core_ не инициализирован
 */
GPUDeviceInfo ROCmBackend::QueryDeviceInfo() const {
  GPUDeviceInfo info{};

  if (!core_ || !core_->IsInitialized()) {
    return info;
  }

  info.name = core_->GetDeviceName();
  info.vendor = core_->GetVendor();
  info.driver_version = core_->GetArchName();  // GCN arch string (e.g. "gfx1201") вместо версии драйвера — ROCm не предоставляет строку версии через hipDeviceProp_t
  info.opencl_version = "N/A (ROCm/HIP)";
  info.device_index = device_index_;
  info.global_memory_size = core_->GetGlobalMemorySize();
  info.local_memory_size = core_->GetLocalMemorySize();
  info.max_mem_alloc_size = core_->GetGlobalMemorySize();  // HIP не имеет отдельного лимита на одно выделение (в отличие от OpenCL CL_DEVICE_MAX_MEM_ALLOC_SIZE)
  info.max_compute_units = static_cast<size_t>(core_->GetComputeUnits());
  info.max_work_group_size = core_->GetMaxWorkGroupSize();
  info.max_clock_frequency = core_->GetMaxClockFrequency();
  info.supports_svm = false;
  info.supports_double = core_->SupportsDoublePrecision();
  info.supports_half = false;             // Технически AMD поддерживает fp16; не экспонируем через IBackend — модули не используют
  info.supports_unified_memory = false;   // hipMallocManaged не используем в текущей архитектуре

  return info;
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
