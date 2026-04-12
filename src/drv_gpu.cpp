/**
 * @file drv_gpu.cpp
 * @brief Реализация DrvGPU - главного класса библиотеки
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 * @updated 2026-02-02 - Убрали дубликат MemoryManager (теперь в memory_manager.cpp)
 */

#include "drv_gpu.hpp"
#include "config/gpu_config.hpp"
#include "memory/memory_manager.hpp"
#include "backends/opencl/opencl_backend.hpp"
#include "backends/opencl/opencl_core.hpp"
#if ENABLE_ROCM
#include "backends/rocm/rocm_backend.hpp"
#include "backends/hybrid/hybrid_backend.hpp"
#endif
#include "logger/logger.hpp"
#include "services/console_output.hpp"
#include <iostream>
#include <sstream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// DrvGPU Implementation - Главный класс библиотеки
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Static Factory Methods — External Context Integration
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Приватный tagged конструктор для static factory methods.
 *
 * Принимает уже инициализированный backend.
 * НЕ вызывает CreateBackend() / backend_->Initialize() —
 * backend уже готов к работе.
 * Вызывает только InitializeSubsystems() (MemoryManager + ModuleRegistry).
 */
DrvGPU::DrvGPU(ExternalInitTag, BackendType type, int device_index,
               std::unique_ptr<IBackend> backend)
    : backend_type_(type),
      device_index_(device_index),
      initialized_(true),
      backend_(std::move(backend)),
      memory_manager_(nullptr),
      module_registry_(nullptr) {
    InitializeSubsystems();
}

/**
 * @brief Создать DrvGPU из внешнего OpenCL контекста.
 *
 * DrvGPU НЕ освобождает переданные хэндлы при уничтожении.
 * gpu.Initialize() вызывать НЕ нужно — уже инициализирован.
 */
DrvGPU DrvGPU::CreateFromExternalOpenCL(
    int device_index,
    cl_context context,
    cl_device_id device,
    cl_command_queue queue) {

    auto backend = std::make_unique<OpenCLBackend>();
    backend->InitializeFromExternalContext(context, device, queue);

    DRVGPU_LOG_INFO_GPU(device_index, "DrvGPU",
        "CreateFromExternalOpenCL: attached to external context [owns=false]");

    return DrvGPU(ExternalInitTag{}, BackendType::OPENCL, device_index,
                  std::move(backend));
}

#if ENABLE_ROCM

/**
 * @brief Создать DrvGPU из внешнего HIP stream.
 *
 * DrvGPU НЕ вызывает hipStreamDestroy при уничтожении.
 * gpu.Initialize() вызывать НЕ нужно — уже инициализирован.
 */
DrvGPU DrvGPU::CreateFromExternalROCm(
    int device_index,
    hipStream_t stream) {

    auto backend = std::make_unique<ROCmBackend>();
    backend->InitializeFromExternalStream(device_index, stream);

    DRVGPU_LOG_INFO_GPU(device_index, "DrvGPU",
        "CreateFromExternalROCm: attached to external stream [owns=false]");

    return DrvGPU(ExternalInitTag{}, BackendType::ROCm, device_index,
                  std::move(backend));
}

/**
 * @brief Создать DrvGPU из внешних OpenCL + HIP ресурсов (HybridBackend).
 *
 * DrvGPU НЕ освобождает ни один из переданных хэндлов.
 * gpu.Initialize() вызывать НЕ нужно — уже инициализирован.
 */
DrvGPU DrvGPU::CreateFromExternalHybrid(
    int device_index,
    cl_context context,
    cl_device_id device,
    cl_command_queue queue,
    hipStream_t stream) {

    auto backend = std::make_unique<HybridBackend>();
    backend->InitializeFromExternalContexts(device_index, context, device, queue, stream);

    DRVGPU_LOG_INFO_GPU(device_index, "DrvGPU",
        "CreateFromExternalHybrid: attached to external OpenCL+HIP contexts [owns=false]");

    return DrvGPU(ExternalInitTag{}, BackendType::OPENCLandROCm, device_index,
                  std::move(backend));
}

#endif  // ENABLE_ROCM

// ════════════════════════════════════════════════════════════════════════════
// Constructors / Destructor
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор DrvGPU
 * @param backend_type Тип бэкенда (OPENCL, ROCm, etc.)
 * @param device_index Индекс GPU устройства (0-based)
 *
 * Создаёт бэкенд и инициализирует подсистемы:
 * - MemoryManager
 * - ModuleRegistry
 *
 * @throws std::runtime_error если не удалось создать бэкенд
 */
DrvGPU::DrvGPU(BackendType backend_type, int device_index)
    : backend_type_(backend_type),
      device_index_(device_index),
      initialized_(false),
      backend_(nullptr),
      memory_manager_(nullptr),
      module_registry_(nullptr) {
    CreateBackend();
    InitializeSubsystems();
}

/**
 * @brief Деструктор DrvGPU
 * 
 * Вызывает Cleanup() для освобождения всех ресурсов.
 * RAII гарантирует корректную очистку.
 */
DrvGPU::~DrvGPU() {
    Cleanup();
}

/**
 * @brief Move конструктор
 * @param other Перемещаемый объект
 */
DrvGPU::DrvGPU(DrvGPU&& other) noexcept
    : backend_type_(other.backend_type_),
      device_index_(other.device_index_),
      initialized_(other.initialized_),
      backend_(std::move(other.backend_)),
      memory_manager_(std::move(other.memory_manager_)),
      module_registry_(std::move(other.module_registry_)) {
    other.initialized_ = false;
}

/**
 * @brief Move оператор присваивания
 * @param other Перемещаемый объект
 * @return Ссылка на this
 */
DrvGPU& DrvGPU::operator=(DrvGPU&& other) noexcept {
    if (this != &other) {
        Cleanup();
        backend_type_ = other.backend_type_;
        device_index_ = other.device_index_;
        initialized_ = other.initialized_;
        backend_ = std::move(other.backend_);
        memory_manager_ = std::move(other.memory_manager_);
        module_registry_ = std::move(other.module_registry_);
        other.initialized_ = false;
    }
    return *this;
}

/**
 * @brief Создать бэкенд на основе типа (внутренний метод)
 * 
 * Создаёт соответствующий бэкенд:
 * - OPENCL -> OpenCLBackend
 * - ROCm -> (не реализовано, throw)
 * - OPENCLandROCm -> (не реализовано, throw)
 * 
 * @throws std::runtime_error если тип бэкенда не поддерживается
 */
void DrvGPU::CreateBackend() {
    switch (backend_type_) {
        case BackendType::OPENCL:
            backend_ = std::make_unique<OpenCLBackend>();
            break;
        case BackendType::ROCm:
#if ENABLE_ROCM
            backend_ = std::make_unique<ROCmBackend>();
#else
            throw std::runtime_error("ROCm backend not available (ENABLE_ROCM=OFF)");
#endif
            break;
        case BackendType::OPENCLandROCm:
#if ENABLE_ROCM
            backend_ = std::make_unique<HybridBackend>();
#else
            throw std::runtime_error("OPENCLandROCm backend not available (ENABLE_ROCM=OFF)");
#endif
            break;
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

/**
 * @brief Инициализировать подсистемы (внутренний метод)
 * 
 * Создаёт:
 * - MemoryManager для управления памятью
 * - ModuleRegistry для регистрации модулей
 */
void DrvGPU::InitializeSubsystems() {
    memory_manager_ = std::make_unique<MemoryManager>(backend_.get());
    module_registry_ = std::make_unique<ModuleRegistry>(device_index_);
}

/**
 * @brief Инициализировать GPU
 * 
 * Инициализирует бэкенд для указанного устройства.
 * После инициализации DrvGPU готов к работе.
 * 
 * @throws std::runtime_error если backend_ == nullptr или инициализация не удалась
 * 
 * Пример:
 * @code
 * DrvGPU gpu(BackendType::OPENCL, 0);
 * gpu.Initialize(); // Инициализировать GPU
 * @endcode
 */
void DrvGPU::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        DRVGPU_LOG_WARNING_GPU(device_index_, "DrvGPU", "Already initialized");
        return;
    }

    if (!backend_) {
        throw std::runtime_error("DrvGPU: backend is null");
    }

    // Ref03: Загрузка configGPU.json перед инициализацией backend (is_prof и др.)
    if (!GPUConfig::GetInstance().IsLoaded()) {
        GPUConfig::GetInstance().LoadOrCreate("configGPU.json");
    }

    backend_->Initialize(device_index_);
    initialized_ = true;
    DRVGPU_LOG_INFO_GPU(device_index_, "DrvGPU", "Initialized successfully");
}

/**
 * @brief Очистить все ресурсы
 * 
 * Освобождает в порядке:
 * 1. MemoryManager
 * 2. ModuleRegistry
 * 3. Backend
 * 
 * Вызывается автоматически в деструкторе.
 */
void DrvGPU::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (memory_manager_) {
        memory_manager_->Cleanup();
    }

    if (module_registry_) {
        module_registry_->Clear();
    }

    if (backend_) {
        backend_->Cleanup();
    }

    memory_manager_.reset();
    module_registry_.reset();
    backend_.reset();
    initialized_ = false;
    DRVGPU_LOG_INFO_GPU(device_index_, "DrvGPU", "Cleaned up");
}

/**
 * @brief Получить информацию об устройстве
 * @return GPUDeviceInfo с информацией о GPU
 * 
 * @throws std::runtime_error если DrvGPU не инициализирован
 */
GPUDeviceInfo DrvGPU::GetDeviceInfo() const {
    if (!initialized_ || !backend_) {
        throw std::runtime_error("DrvGPU not initialized");
    }
    return backend_->GetDeviceInfo();
}

/**
 * @brief Получить название устройства
 * @return Строка с названием или "Unknown"
 */
std::string DrvGPU::GetDeviceName() const {
    if (!initialized_ || !backend_) {
        return "Unknown";
    }
    auto info = backend_->GetDeviceInfo();
    return info.name;
}

/**
 * @brief Вывести информацию об устройстве в лог
 */
void DrvGPU::PrintDeviceInfo() const {
    if (!initialized_ || !backend_) {
        DRVGPU_LOG_WARNING_GPU(device_index_, "DrvGPU", "Device not initialized");
        return;
    }
    auto info = backend_->GetDeviceInfo();
    DRVGPU_LOG_INFO_GPU(device_index_, "DrvGPU", "Device Info - Name: " + info.name + ", Vendor: " + info.vendor);
}

/**
 * @brief Получить менеджер памяти (не-const версия)
 * @return Ссылка на MemoryManager
 * 
 * @throws std::runtime_error если MemoryManager не инициализирован
 */
MemoryManager& DrvGPU::GetMemoryManager() {
    if (!memory_manager_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    return *memory_manager_;
}

/**
 * @brief Получить менеджер памяти (const версия)
 * @return Константная ссылка на MemoryManager
 */
const MemoryManager& DrvGPU::GetMemoryManager() const {
    if (!memory_manager_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    return *memory_manager_;
}

/**
 * @brief Получить регистр модулей (не-const версия)
 * @return Ссылка на ModuleRegistry
 */
ModuleRegistry& DrvGPU::GetModuleRegistry() {
    if (!module_registry_) {
        throw std::runtime_error("ModuleRegistry not initialized");
    }
    return *module_registry_;
}

/**
 * @brief Получить регистр модулей (const версия)
 * @return Константная ссылка на ModuleRegistry
 */
const ModuleRegistry& DrvGPU::GetModuleRegistry() const {
    if (!module_registry_) {
        throw std::runtime_error("ModuleRegistry not initialized");
    }
    return *module_registry_;
}

/**
 * @brief Получить бэкенд (не-const версия)
 * @return Ссылка на IBackend
 * 
 * @warning Используйте только если абстракции недостаточно!
 */
IBackend& DrvGPU::GetBackend() {
    if (!backend_) {
        throw std::runtime_error("Backend not initialized");
    }
    return *backend_;
}

/**
 * @brief Получить бэкенд (const версия)
 * @return Константная ссылка на IBackend
 */
const IBackend& DrvGPU::GetBackend() const {
    if (!backend_) {
        throw std::runtime_error("Backend not initialized");
    }
    return *backend_;
}

/**
 * @brief Синхронизировать (дождаться завершения всех операций)
 * 
 * Блокирует CPU до завершения всех GPU операций.
 * 
 * @throws std::runtime_error если DrvGPU не инициализирован
 */
void DrvGPU::Synchronize() {
    if (!initialized_ || !backend_) {
        throw std::runtime_error("DrvGPU not initialized");
    }
    backend_->Synchronize();
}

/**
 * @brief Flush всех команд (без ожидания)
 * 
 * Отправляет все команды на выполнение без ожидания.
 */
void DrvGPU::Flush() {
    if (!initialized_ || !backend_) {
        return;
    }
    backend_->Flush();
}

/**
 * @brief Вывести статистику в консоль
 */
void DrvGPU::PrintStatistics() const {
    auto& con = ConsoleOutput::GetInstance();
    std::string stats = "Backend=" + std::to_string(static_cast<int>(backend_type_))
                      + ", Initialized=" + (initialized_ ? "Yes" : "No");
    if (memory_manager_) {
        stats += "\n" + memory_manager_->GetStatistics();
    }
    con.Print(device_index_, "DrvGPU", stats);
}

/**
 * @brief Получить статистику в виде строки
 * @return Строка с статистикой
 */
std::string DrvGPU::GetStatistics() const {
    std::ostringstream oss;
    oss << "DrvGPU Statistics:\n";
    oss << "  Device Index:  " << device_index_ << "\n";
    oss << "  Initialized:   " << (initialized_ ? "Yes" : "No") << "\n";
    if (memory_manager_) {
        oss << memory_manager_->GetStatistics();
    }
    return oss.str();
}

/**
 * @brief Сбросить статистику
 */
void DrvGPU::ResetStatistics() {
    if (memory_manager_) {
        memory_manager_->ResetStatistics();
    }
}

} // namespace drv_gpu_lib