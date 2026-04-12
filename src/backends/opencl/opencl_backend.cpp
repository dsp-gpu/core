#include "opencl_backend.hpp"
#include "../../config/gpu_config.hpp"
#include "../../logger/logger.hpp"

#include <sstream>
#include <iomanip>
#include <vector>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать OpenCLBackend (без инициализации)
 */
OpenCLBackend::OpenCLBackend()
    : device_index_(-1)
    , initialized_(false)
    , owns_resources_(true)
    , core_(nullptr)  // ✅ MULTI-GPU: Per-device core
    , context_(nullptr)
    , device_(nullptr)
    , queue_(nullptr) {
}

/**
 * @brief Деструктор - автоматическая очистка ресурсов
 */
OpenCLBackend::~OpenCLBackend() {
    Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move конструктор и оператор
// ════════════════════════════════════════════════════════════════════════════

OpenCLBackend::OpenCLBackend(OpenCLBackend&& other) noexcept
    : device_index_(other.device_index_)
    , initialized_(other.initialized_)
    , owns_resources_(other.owns_resources_)
    , core_(std::move(other.core_))  // ✅ MULTI-GPU: Move core
    , memory_manager_(std::move(other.memory_manager_))
    , svm_capabilities_(std::move(other.svm_capabilities_))
    , context_(other.context_)
    , device_(other.device_)
    , queue_(other.queue_) {

    // Обнуляем источник
    other.device_index_ = -1;
    other.initialized_ = false;
    other.owns_resources_ = false;
    other.context_ = nullptr;
    other.device_ = nullptr;
    other.queue_ = nullptr;
}

OpenCLBackend& OpenCLBackend::operator=(OpenCLBackend&& other) noexcept {
    if (this != &other) {
        Cleanup();

        device_index_ = other.device_index_;
        initialized_ = other.initialized_;
        owns_resources_ = other.owns_resources_;
        core_ = std::move(other.core_);  // ✅ MULTI-GPU: Move core
        memory_manager_ = std::move(other.memory_manager_);
        svm_capabilities_ = std::move(other.svm_capabilities_);
        context_ = other.context_;
        device_ = other.device_;
        queue_ = other.queue_;

        other.device_index_ = -1;
        other.initialized_ = false;
        other.owns_resources_ = false;
        other.context_ = nullptr;
        other.device_ = nullptr;
        other.queue_ = nullptr;
    }

    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Инициализация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать бэкенд для конкретного устройства
 *
 * ✅ MULTI-GPU: Теперь каждый backend имеет СВОЙ OpenCLCore!
 *
 * Процесс:
 * 1. Создаём OpenCLCore для device_index
 * 2. Инициализируем OpenCLCore (выбор устройства по индексу)
 * 3. Получаем context/device из OpenCLCore
 * 4. Создаём command queue для ЭТОГО устройства
 * 5. Инициализируем SVM capabilities и MemoryManager
 */
void OpenCLBackend::Initialize(int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        Cleanup();
    }

    device_index_ = device_index;
    owns_resources_ = true;

    // ═══════════════════════════════════════════════════════════════════════
    // ✅ MULTI-GPU: Создаём СОБСТВЕННЫЙ OpenCLCore для этого устройства
    // ═══════════════════════════════════════════════════════════════════════

    DRVGPU_LOG_INFO_GPU(device_index, "OpenCLBackend", "Creating OpenCLCore for device " + std::to_string(device_index));

    core_ = std::make_unique<OpenCLCore>(device_index, DeviceType::GPU);
    core_->Initialize();

    // ═══════════════════════════════════════════════════════════════════════
    // Получаем нативные хэндлы из НАШЕГО OpenCLCore
    // ═══════════════════════════════════════════════════════════════════════

    context_ = core_->GetContext();
    device_ = core_->GetDevice();

    DRVGPU_LOG_INFO_GPU(device_index_, "OpenCLBackend", "Got context and device from OpenCLCore");

    // ═══════════════════════════════════════════════════════════════════════
    // Создаём COMMAND QUEUE для этого устройства
    // ═══════════════════════════════════════════════════════════════════════

    cl_int err;
    bool want_prof = !GPUConfig::GetInstance().IsLoaded()
        || GPUConfig::GetInstance().IsProfilingEnabled(device_index);

    #ifdef CL_VERSION_2_0
        static const cl_queue_properties PROPS_PROF[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        static const cl_queue_properties PROPS_NONE[] = {0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_,
            want_prof ? PROPS_PROF : PROPS_NONE, &err);
    #else
        cl_command_queue_properties flags = want_prof
            ? static_cast<cl_command_queue_properties>(CL_QUEUE_PROFILING_ENABLE) : 0;
        queue_ = clCreateCommandQueue(context_, device_, flags, &err);
    #endif

    if (err != CL_SUCCESS || !queue_) {
        core_.reset();  // Очищаем OpenCLCore при ошибке
        throw std::runtime_error(
            "OpenCLBackend::Initialize - Failed to create command queue for device " +
            std::to_string(device_index) + ". Error code: " + std::to_string(err)
        );
    }

    DRVGPU_LOG_INFO_GPU(device_index_, "OpenCLBackend", "Command queue created for device " + std::to_string(device_index));

    // ═══════════════════════════════════════════════════════════════════════
    // SVM capabilities и MemoryManager
    // ═══════════════════════════════════════════════════════════════════════

    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );

    memory_manager_ = std::make_unique<MemoryManager>(this);

    initialized_ = true;

    DRVGPU_LOG_INFO_GPU(device_index_, "OpenCLBackend",
        "Initialized for device " + std::to_string(device_index) +
        " (" + core_->GetDeviceName() + ")");
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация из внешнего OpenCL контекста
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализация из внешнего OpenCL контекста
 *
 * Позволяет использовать OpenCLBackend с уже созданными OpenCL ресурсами
 * (например, из другой библиотеки или приложения).
 *
 * @note НЕ создаёт OpenCLCore, так как ресурсы пришли извне
 * @note Автоматически устанавливает owns_resources_ = false
 * @note Backend НЕ будет освобождать эти ресурсы при Cleanup()
 */
void OpenCLBackend::InitializeFromExternalContext(
    cl_context external_context,
    cl_device_id external_device,
    cl_command_queue external_queue
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        throw std::runtime_error(
            "OpenCLBackend::InitializeFromExternalContext - "
            "Backend is already initialized! Call Cleanup() first."
        );
    }

    if (!external_context || !external_device || !external_queue) {
        throw std::invalid_argument(
            "OpenCLBackend::InitializeFromExternalContext - "
            "All parameters (context, device, queue) must be non-null!"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXTERNAL MODE: Используем внешние ресурсы
    // ═══════════════════════════════════════════════════════════════════════

    context_ = external_context;
    device_ = external_device;
    queue_ = external_queue;
    device_index_ = -1;  // Неизвестен индекс при внешнем контексте
    owns_resources_ = false;  // ❌ НЕ освобождаем внешние ресурсы!

    DRVGPU_LOG_INFO_GPU(device_index_, "OpenCLBackend",
        "Initializing from external OpenCL context (owns_resources = false)");

    // ═══════════════════════════════════════════════════════════════════════
    // НЕ создаём OpenCLCore — ресурсы пришли извне!
    // ═══════════════════════════════════════════════════════════════════════
    core_.reset();  // Убеждаемся что core_ пустой

    // ═══════════════════════════════════════════════════════════════════════
    // SVM capabilities и MemoryManager инициализируем как обычно
    // ═══════════════════════════════════════════════════════════════════════

    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );

    memory_manager_ = std::make_unique<MemoryManager>(this);

    initialized_ = true;

    // Получаем имя устройства для лога (без OpenCLCore)
    char device_name[256] = {};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NAME,
                                 sizeof(device_name), device_name, nullptr);
    std::string device_name_str = (err == CL_SUCCESS) ? device_name : "Unknown Device";

    DRVGPU_LOG_INFO_GPU(device_index_, "OpenCLBackend",
        "Initialized from external context (" + device_name_str + ")");
}

/**
 * @brief Очистить все ресурсы бэкенда
 *
 * ✅ MULTI-GPU: Очищает СОБСТВЕННЫЙ OpenCLCore
 */
void OpenCLBackend::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    int gpu_id_for_log = device_index_;
    DRVGPU_LOG_INFO_GPU(gpu_id_for_log, "OpenCLBackend",
        "Cleanup started for device " + std::to_string(device_index_) +
        " (owns_resources = " + std::string(owns_resources_ ? "true" : "false") + ")");

    // Освобождаем MemoryManager и SVM capabilities
    svm_capabilities_.reset();
    memory_manager_.reset();

    if (owns_resources_) {
        // ═══════════════════════════════════════════════════════════════════
        // OWNING MODE: Освобождаем ресурсы
        // ═══════════════════════════════════════════════════════════════════

        if (queue_) {
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
            DRVGPU_LOG_DEBUG_GPU(device_index_, "OpenCLBackend", "Command queue released");
        }

        // ✅ MULTI-GPU: Очищаем СВОЙ OpenCLCore
        core_.reset();

        context_ = nullptr;
        device_ = nullptr;

    } else {
        // NON-OWNING MODE: Просто обнуляем указатели
        DRVGPU_LOG_DEBUG_GPU(device_index_, "OpenCLBackend", "Non-owning mode: NOT releasing resources");

        queue_ = nullptr;
        context_ = nullptr;
        device_ = nullptr;
        core_.reset();
    }

    device_index_ = -1;
    initialized_ = false;

    DRVGPU_LOG_INFO_GPU(gpu_id_for_log, "OpenCLBackend", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Информация об устройстве
// ════════════════════════════════════════════════════════════════════════════

GPUDeviceInfo OpenCLBackend::GetDeviceInfo() const {
    return QueryDeviceInfo();
}

std::string OpenCLBackend::GetDeviceName() const {
    if (!core_ || !core_->IsInitialized()) {
        return "Unknown";
    }
    return core_->GetDeviceName();
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Нативные хэндлы
// ════════════════════════════════════════════════════════════════════════════

void* OpenCLBackend::GetNativeContext() const {
    return static_cast<void*>(context_);
}

void* OpenCLBackend::GetNativeDevice() const {
    return static_cast<void*>(device_);
}

void* OpenCLBackend::GetNativeQueue() const {
    return static_cast<void*>(queue_);
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Управление памятью
// ════════════════════════════════════════════════════════════════════════════

void* OpenCLBackend::Allocate(size_t size_bytes, unsigned int flags) {
    if (!context_) {
        return nullptr;
    }

    cl_mem_flags mem_flags = CL_MEM_READ_WRITE;

    if (flags & 1) mem_flags |= CL_MEM_HOST_READ_ONLY;
    if (flags & 2) mem_flags |= CL_MEM_HOST_WRITE_ONLY;
    if (flags & 4) mem_flags |= CL_MEM_HOST_NO_ACCESS;

    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(context_, mem_flags, size_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !mem) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLBackend",
            "clCreateBuffer failed: " + std::to_string(err) +
            " (requested " + std::to_string(size_bytes) + " bytes)");
        return nullptr;
    }

    return static_cast<void*>(mem);
}

void OpenCLBackend::Free(void* ptr) {
    if (ptr) {
        clReleaseMemObject(static_cast<cl_mem>(ptr));
    }
}

void OpenCLBackend::MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLBackend", "MemcpyHostToDevice - Invalid parameters");
        return;
    }

    cl_int err = clEnqueueWriteBuffer(
        queue_,
        static_cast<cl_mem>(dst),
        CL_TRUE,
        0,
        size_bytes,
        src,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLBackend", "MemcpyHostToDevice error: " + std::to_string(err));
    }
}

void OpenCLBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }

    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));

    cl_int err = clEnqueueReadBuffer(
        queue_,
        src_mem,
        CL_TRUE,
        0,
        size_bytes,
        dst,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLBackend", "MemcpyDeviceToHost error: " + std::to_string(err));
    }
}

void OpenCLBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }

    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));
    cl_mem dst_mem = static_cast<cl_mem>(dst);

    cl_int err = clEnqueueCopyBuffer(
        queue_,
        src_mem,
        dst_mem,
        0,
        0,
        size_bytes,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR_GPU(device_index_, "OpenCLBackend", "MemcpyDeviceToDevice error: " + std::to_string(err));
        return;
    }

    // Синхронизация для консистентности с HtoD (CL_TRUE) и ROCm backend (hipStreamSynchronize)
    clFinish(queue_);
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Синхронизация
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackend::Synchronize() {
    if (queue_) {
        clFinish(queue_);
    }
}

void OpenCLBackend::Flush() {
    if (queue_) {
        clFlush(queue_);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Возможности устройства
// ════════════════════════════════════════════════════════════════════════════

bool OpenCLBackend::SupportsSVM() const {
    return svm_capabilities_ && svm_capabilities_->HasAnySVM();
}

bool OpenCLBackend::SupportsDoublePrecision() const {
    if (!device_) return false;

    // Проверяем cl_khr_fp64 через расширения устройства
    size_t ext_size = 0;
    if (clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size) != CL_SUCCESS || ext_size == 0) {
        return false;
    }
    std::vector<char> ext_buf(ext_size);
    if (clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_size, ext_buf.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    std::string extensions(ext_buf.data());
    return extensions.find("cl_khr_fp64") != std::string::npos;
}

size_t OpenCLBackend::GetMaxWorkGroupSize() const {
    if (!core_ || !core_->IsInitialized()) {
        return 0;
    }
    return core_->GetMaxWorkGroupSize();
}

size_t OpenCLBackend::GetGlobalMemorySize() const {
    // Если есть core — используем его
    if (core_ && core_->IsInitialized()) {
        return core_->GetGlobalMemorySize();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXTERNAL CONTEXT: Запрашиваем напрямую через device_
    // ═══════════════════════════════════════════════════════════════════════════
    if (device_) {
        cl_ulong mem_size = 0;
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
                                     sizeof(mem_size), &mem_size, nullptr);
        if (err == CL_SUCCESS) {
            return static_cast<size_t>(mem_size);
        }
    }
    return 0;
}

size_t OpenCLBackend::GetFreeMemorySize() const {
    // Если есть core — используем его
    if (core_ && core_->IsInitialized()) {
        return core_->GetFreeMemorySize();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXTERNAL CONTEXT: Запрашиваем напрямую через device_
    // ═══════════════════════════════════════════════════════════════════════════
    if (!device_) {
        return 0;
    }

    // NVIDIA: CL_DEVICE_GLOBAL_FREE_MEMORY_NV (расширение cl_nv_device_attribute_query)
    constexpr cl_device_info CL_DEVICE_GLOBAL_FREE_MEMORY_NV = 0x4006;

    cl_ulong free_mem = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_FREE_MEMORY_NV,
                                 sizeof(free_mem), &free_mem, nullptr);

    if (err == CL_SUCCESS && free_mem > 0) {
        // NVIDIA возвращает значение в KB, конвертируем в bytes
        return static_cast<size_t>(free_mem) * 1024;
    }

    // AMD: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD (расширение cl_amd_device_attribute_query)
    constexpr cl_device_info CL_DEVICE_GLOBAL_FREE_MEMORY_AMD = 0x4039;

    err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                          sizeof(free_mem), &free_mem, nullptr);

    if (err == CL_SUCCESS && free_mem > 0) {
        // AMD возвращает в KB
        return static_cast<size_t>(free_mem) * 1024;
    }

    // Fallback: эвристика — 90% от общей памяти
    size_t total = GetGlobalMemorySize();
    return static_cast<size_t>(static_cast<double>(total) * 0.9);
}

size_t OpenCLBackend::GetLocalMemorySize() const {
    // Если есть core — используем его
    if (core_ && core_->IsInitialized()) {
        return core_->GetLocalMemorySize();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXTERNAL CONTEXT: Запрашиваем напрямую через device_
    // ═══════════════════════════════════════════════════════════════════════════
    if (device_) {
        cl_ulong local_size = 0;
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,
                                     sizeof(local_size), &local_size, nullptr);
        if (err == CL_SUCCESS) {
            return static_cast<size_t>(local_size);
        }
    }
    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Специфичные для OpenCL методы
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore& OpenCLBackend::GetCore() {
    if (!core_) {
        throw std::runtime_error("OpenCLBackend::GetCore - Core not initialized");
    }
    return *core_;
}

const OpenCLCore& OpenCLBackend::GetCore() const {
    if (!core_) {
        throw std::runtime_error("OpenCLBackend::GetCore - Core not initialized");
    }
    return *core_;
}

MemoryManager* OpenCLBackend::GetMemoryManager() {
    return memory_manager_.get();
}

const MemoryManager* OpenCLBackend::GetMemoryManager() const {
    return memory_manager_.get();
}

MemoryManager& OpenCLBackend::GetMemoryManagerRef() {
    return *memory_manager_;
}

const MemoryManager& OpenCLBackend::GetMemoryManagerRef() const {
    return *memory_manager_;
}

const SVMCapabilities& OpenCLBackend::GetSVMCapabilities() const {
    static SVMCapabilities empty_caps;
    return svm_capabilities_ ? *svm_capabilities_ : empty_caps;
}

void OpenCLBackend::InitializeCommandQueuePool(size_t num_queues) {
    (void)num_queues;
    // TODO: интегрировать CommandQueuePool (command_queue_pool.hpp)
}

GPUDeviceInfo OpenCLBackend::QueryDeviceInfo() const {
    GPUDeviceInfo info;

    if (core_ && core_->IsInitialized()) {
        info.name = core_->GetDeviceName();
        info.vendor = core_->GetVendor();
        info.driver_version = core_->GetDriverVersion();
        info.opencl_version = std::to_string(core_->GetOpenCLVersionMajor()) + "." +
                              std::to_string(core_->GetOpenCLVersionMinor());
        info.device_index = device_index_;
        info.global_memory_size = core_->GetGlobalMemorySize();
        info.local_memory_size = core_->GetLocalMemorySize();
        info.max_mem_alloc_size = core_->GetGlobalMemorySize();
        info.max_compute_units = core_->GetComputeUnits();
        info.max_work_group_size = core_->GetMaxWorkGroupSize();
        info.supports_svm = core_->IsSVMSupported();
        info.supports_double = SupportsDoublePrecision();
        info.supports_half = false;
        info.supports_unified_memory = SupportsSVM();
        return info;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXTERNAL CONTEXT: core_ = null, но device_ задан — запрашиваем напрямую
    // ═══════════════════════════════════════════════════════════════════════
    if (!device_) {
        return info;
    }

    auto getStr = [this](cl_device_info param) -> std::string {
        size_t size = 0;
        if (clGetDeviceInfo(device_, param, 0, nullptr, &size) != CL_SUCCESS || size == 0)
            return {};
        std::vector<char> buf(size);
        if (clGetDeviceInfo(device_, param, size, buf.data(), nullptr) != CL_SUCCESS)
            return {};
        return std::string(buf.data());
    };

    auto getUlong = [this](cl_device_info param) -> cl_ulong {
        cl_ulong val = 0;
        clGetDeviceInfo(device_, param, sizeof(val), &val, nullptr);
        return val;
    };

    info.name = getStr(CL_DEVICE_NAME);
    info.vendor = getStr(CL_DEVICE_VENDOR);
    info.driver_version = getStr(CL_DRIVER_VERSION);
    info.opencl_version = getStr(CL_DEVICE_VERSION);  // "OpenCL 3.0 ..." — можно парсить
    info.device_index = device_index_ >= 0 ? device_index_ : 0;
    info.global_memory_size = getUlong(CL_DEVICE_GLOBAL_MEM_SIZE);
    info.local_memory_size = getUlong(CL_DEVICE_LOCAL_MEM_SIZE);
    info.max_mem_alloc_size = getUlong(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    info.max_compute_units = getUlong(CL_DEVICE_MAX_COMPUTE_UNITS);
    info.max_work_group_size = 0;
    size_t wgs = 0;
    if (clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr) == CL_SUCCESS)
        info.max_work_group_size = wgs;
    info.supports_svm = SupportsSVM();
    info.supports_double = SupportsDoublePrecision();
    info.supports_half = false;
    info.supports_unified_memory = SupportsSVM();

    return info;
}

} // namespace drv_gpu_lib
